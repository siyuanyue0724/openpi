from __future__ import annotations

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from enum import Enum
from types import MethodType
from typing import Any

import torch
from torch import nn

from picf_next.lingbot_native.wsa_da3_loss import (
    WSADA3TeacherTargets,
    compute_official_wsa_da3_loss,
)
from picf_next.lingbot_native.wsa_future_expert_runtime import (
    WSA_FSDP_POST_METHOD,
    WSA_FSDP_QKV_METHOD,
)
from picf_next.lingbot_native.wsa_lingbot_training_runtime import (
    WSALingBotAttentionIntervention,
    WSALingBotJointOutput,
    WSALingBotTrainingRuntime,
)

WSA_RELEASED_LAMBDA_3D = 0.1
WSA_FSDP_BLOCK_CLASS = "Future3DBlock"
WSA_FSDP_EXPERT_CLASS = "Future3DExpert"
WSA_FSDP_PARAMETER_PREFIX = (
    "model.qwenvl_with_expert.adr218_wsa_training_runtime.future.expert"
)
WSA_FSDP_EXPERT_METHODS = ("pre_dit", "project_query_layers")
WSA_ADAMW_NAME_PATTERN = "adr218_wsa_training_runtime"
WSA_LARGE_OPTIMIZER = {
    "learning_rate": 1.0e-4,
    "adamw_betas": (0.9, 0.95),
    "adamw_eps": 1.0e-8,
    "weight_decay": 1.0e-2,
    "grad_clip_norm": 1.0,
}
WSA_LARGE_SCHEDULER = {
    "warmup_ratio": 0.05,
    "min_lr_ratio": 0.01,
    "scheduler_type": "cosine",
}
WSA_STEP_LEDGER_ATTRIBUTE = "_adr218_wsa_step_ledger"


class WSALingBotForwardRole(str, Enum):
    """Semantic role of one complete host/future/action forward."""

    PRIMARY_FACTUAL = "primary_factual"
    MEASUREMENT_ONLY = "measurement_only"


@dataclass(frozen=True, slots=True)
class WSALingBotForwardContract:
    """Typed supervision boundary for a complete WSA action forward."""

    role: WSALingBotForwardRole
    teacher_targets: WSADA3TeacherTargets | None = None
    measurement_callback: Callable[[WSALingBotJointOutput], None] | None = None
    attention_intervention: WSALingBotAttentionIntervention | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, WSALingBotForwardRole):
            raise TypeError("WSA LingBot forward role must be typed")
        if self.role is WSALingBotForwardRole.PRIMARY_FACTUAL:
            if not isinstance(self.teacher_targets, WSADA3TeacherTargets):
                raise TypeError("primary factual WSA forward requires typed DA3 targets")
            self.teacher_targets.validate()
            if self.measurement_callback is not None:
                raise ValueError("primary factual WSA forward cannot retain measurement output")
            if self.attention_intervention is not None:
                raise ValueError("primary factual WSA forward cannot carry an intervention")
        else:
            if self.teacher_targets is not None:
                raise ValueError("measurement-only WSA forward cannot carry teacher targets")
            if self.measurement_callback is not None and not callable(
                self.measurement_callback
            ):
                raise TypeError("WSA measurement callback must be callable")
            if self.attention_intervention is not None and not isinstance(
                self.attention_intervention,
                WSALingBotAttentionIntervention,
            ):
                raise TypeError("WSA attention intervention must be typed")


@dataclass(slots=True)
class WSALingBotStepLedger:
    """Non-learned optimizer-transaction audit for WSA loss multiplicity."""

    primary_factual_calls: int = 0
    measurement_only_calls: int = 0
    closed: bool = False

    def record(self, role: WSALingBotForwardRole) -> None:
        if self.closed:
            raise RuntimeError("WSA step ledger is already closed")
        if role is WSALingBotForwardRole.PRIMARY_FACTUAL:
            self.primary_factual_calls += 1
        elif role is WSALingBotForwardRole.MEASUREMENT_ONLY:
            self.measurement_only_calls += 1
        else:  # pragma: no cover - the enum is closed, retain fail-closed behavior.
            raise ValueError(f"unknown WSA forward role: {role!r}")

    def close(self) -> None:
        if self.closed:
            raise RuntimeError("WSA step ledger was closed more than once")
        self.closed = True
        if self.primary_factual_calls != 1:
            raise RuntimeError(
                "WSA optimizer transaction requires exactly one primary factual forward, "
                f"got {self.primary_factual_calls}"
            )

    def receipt(self) -> dict[str, Any]:
        return {
            "schema": "picf-next.adr218-wsa-step-ledger.v1",
            "primary_factual_calls": self.primary_factual_calls,
            "measurement_only_calls": self.measurement_only_calls,
            "closed": self.closed,
        }


def _installed_wsa_runtime(policy: nn.Module) -> WSALingBotTrainingRuntime | None:
    model = getattr(policy, "model", None)
    joint = getattr(model, "qwenvl_with_expert", None)
    runtime = getattr(joint, "adr218_wsa_training_runtime", None)
    if runtime is None:
        return None
    if not isinstance(runtime, WSALingBotTrainingRuntime):
        raise TypeError("installed ADR218 WSA runtime has the wrong type")
    return runtime


def wsa_lingbot_forward_kwargs(
    policy: nn.Module,
    *,
    role: WSALingBotForwardRole,
    teacher_targets: WSADA3TeacherTargets | None = None,
    measurement_callback: Callable[[WSALingBotJointOutput], None] | None = None,
    attention_intervention: WSALingBotAttentionIntervention | None = None,
) -> dict[str, WSALingBotForwardContract]:
    """Return controlled kwargs only when the complete WSA runtime is installed."""

    if not isinstance(role, WSALingBotForwardRole):
        raise TypeError("WSA LingBot forward role must be typed")
    if _installed_wsa_runtime(policy) is None:
        if teacher_targets is not None:
            raise RuntimeError("DA3 targets were supplied without an installed WSA runtime")
        if measurement_callback is not None:
            raise RuntimeError("WSA measurement callback was supplied without its runtime")
        if attention_intervention is not None:
            raise RuntimeError("WSA attention intervention was supplied without its runtime")
        return {}
    return {
        "wsa_lingbot_forward_contract": WSALingBotForwardContract(
            role=role,
            teacher_targets=teacher_targets,
            measurement_callback=measurement_callback,
            attention_intervention=attention_intervention,
        )
    }


@contextmanager
def wsa_lingbot_optimizer_transaction(
    policy: nn.Module,
) -> Iterator[WSALingBotStepLedger | None]:
    """Audit exactly one factual WSA loss across a complete optimizer objective."""

    if _installed_wsa_runtime(policy) is None:
        yield None
        return
    if hasattr(policy, WSA_STEP_LEDGER_ATTRIBUTE):
        raise RuntimeError("a WSA optimizer transaction is already active")
    ledger = WSALingBotStepLedger()
    setattr(policy, WSA_STEP_LEDGER_ATTRIBUTE, ledger)
    clean_exit = False
    try:
        yield ledger
        clean_exit = True
    finally:
        delattr(policy, WSA_STEP_LEDGER_ATTRIBUTE)
        if clean_exit:
            ledger.close()


def configure_wsa_lingbot_optimizer_contract(contract: Any) -> Any:
    """Route WSA out of Muon before installing its donor AdamW owner.

    The released LingBot Muon builder already supports a named AdamW fallback.
    This exclusion is only the lossless first half of the adaptation; the
    resulting parameters are moved to WSA-Large's exact AdamW contract by
    :func:`install_wsa_lingbot_optimizer` before any optimizer state exists.
    """

    patterns = getattr(contract, "muon_exclude_name_patterns", None)
    if not isinstance(patterns, tuple) or any(not isinstance(item, str) for item in patterns):
        raise TypeError("LingBot Muon exclusion contract must be an immutable string tuple")
    return replace(
        contract,
        muon_exclude_name_patterns=tuple(dict.fromkeys((*patterns, WSA_ADAMW_NAME_PATTERN))),
    )


def install_wsa_lingbot_optimizer(policy: nn.Module, optimizer: Any) -> Any:
    """Give every Future3D parameter one exact WSA-Large AdamW owner.

    ``WSA_LargeConfig.get_optimizer_preset`` delegates directly to
    ``torch.optim.AdamW`` with the constants below.  Moving parameters before
    the first step is mathematically identical to building that donor
    optimizer separately and avoids applying LingBot's zero weight decay.
    """

    joint = policy.model.qwenvl_with_expert
    runtime = getattr(joint, "adr218_wsa_training_runtime", None)
    if not isinstance(runtime, WSALingBotTrainingRuntime):
        raise RuntimeError("ADR218 WSA runtime is not installed")
    future_parameters = tuple(
        parameter for parameter in runtime.future.parameters() if parameter.requires_grad
    )
    future_ids = {id(parameter) for parameter in future_parameters}
    if not future_parameters or len(future_ids) != len(future_parameters):
        raise RuntimeError("ADR218 Future3D trainable parameter identity is invalid")
    inner_optimizers = getattr(optimizer, "optimizers", None)
    if not isinstance(inner_optimizers, list) or not inner_optimizers:
        raise TypeError("LingBot combined optimizer exposes no inner optimizers")
    if getattr(optimizer, "state", None):
        raise RuntimeError("Future3D optimizer ownership must be installed before the first step")

    owner_counts = {parameter_id: 0 for parameter_id in future_ids}
    for inner in inner_optimizers:
        retained_groups = []
        for group in inner.param_groups:
            parameters = list(group["params"])
            matched = [parameter for parameter in parameters if id(parameter) in future_ids]
            if matched and type(inner).__name__ != "AdamW":
                raise RuntimeError("Future3D reached a non-AdamW LingBot optimizer")
            for parameter in matched:
                owner_counts[id(parameter)] += 1
            retained = [parameter for parameter in parameters if id(parameter) not in future_ids]
            if retained:
                retained_group = dict(group)
                retained_group["params"] = retained
                retained_groups.append(retained_group)
        inner.param_groups = retained_groups
    if any(count != 1 for count in owner_counts.values()):
        raise RuntimeError("Future3D was not present exactly once in the LingBot optimizer")

    donor_optimizer = torch.optim.AdamW(
        future_parameters,
        lr=WSA_LARGE_OPTIMIZER["learning_rate"],
        betas=WSA_LARGE_OPTIMIZER["adamw_betas"],
        eps=WSA_LARGE_OPTIMIZER["adamw_eps"],
        weight_decay=WSA_LARGE_OPTIMIZER["weight_decay"],
        foreach=False,
        fused=False,
    )
    inner_optimizers.append(donor_optimizer)
    return donor_optimizer


def build_wsa_lingbot_scheduler(
    donor_optimizer: torch.optim.Optimizer,
    *,
    total_train_steps: int,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Build the released WSA-Large native scheduler on its donor optimizer."""

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    if type(donor_optimizer).__name__ != "AdamW":
        raise TypeError("WSA-Large scheduler requires its dedicated AdamW owner")
    if isinstance(total_train_steps, bool) or not isinstance(total_train_steps, int):
        raise TypeError("WSA-Large total training steps must be an integer")
    total_train_steps = max(total_train_steps, 1)
    warmup_steps = int(total_train_steps * WSA_LARGE_SCHEDULER["warmup_ratio"])
    warmup_steps = min(max(warmup_steps, 0), total_train_steps - 1)
    remaining_steps = max(total_train_steps - warmup_steps, 1)
    main_scheduler = CosineAnnealingLR(
        donor_optimizer,
        T_max=remaining_steps,
        eta_min=(
            WSA_LARGE_OPTIMIZER["learning_rate"]
            * WSA_LARGE_SCHEDULER["min_lr_ratio"]
        ),
    )
    if warmup_steps <= 0:
        return main_scheduler
    warmup_scheduler = LinearLR(
        donor_optimizer,
        start_factor=1.0 / warmup_steps,
        end_factor=1.0,
        total_iters=warmup_steps,
    )
    return SequentialLR(
        donor_optimizer,
        schedulers=[warmup_scheduler, main_scheduler],
        milestones=[warmup_steps],
    )


def audit_wsa_lingbot_scheduler(
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    donor_optimizer: torch.optim.Optimizer,
    *,
    total_train_steps: int,
) -> dict[str, Any]:
    """Bind the exact donor schedule and current clock to a durable receipt."""

    expected_warmup_steps = min(
        max(int(total_train_steps * WSA_LARGE_SCHEDULER["warmup_ratio"]), 0),
        max(total_train_steps - 1, 0),
    )
    expected_type = "CosineAnnealingLR" if expected_warmup_steps == 0 else "SequentialLR"
    if type(scheduler).__name__ != expected_type:
        raise RuntimeError("WSA-Large scheduler implementation differs from the donor")
    if scheduler.optimizer is not donor_optimizer:
        raise RuntimeError("WSA-Large scheduler is attached to the wrong optimizer")
    learning_rates = tuple(float(group["lr"]) for group in donor_optimizer.param_groups)
    if not learning_rates or any(not math.isfinite(value) for value in learning_rates):
        raise RuntimeError("WSA-Large scheduler produced invalid learning rates")
    return {
        "schema": "picf-next.adr218-wsa-lingbot-scheduler.v1",
        "source": "WSALargeNativeSchedulerConfig(FastWAMNativeSchedulerConfig)",
        "scheduler": expected_type,
        "scheduler_type": WSA_LARGE_SCHEDULER["scheduler_type"],
        "total_train_steps": total_train_steps,
        "warmup_steps": expected_warmup_steps,
        "warmup_ratio": WSA_LARGE_SCHEDULER["warmup_ratio"],
        "peak_learning_rate": WSA_LARGE_OPTIMIZER["learning_rate"],
        "minimum_learning_rate": (
            WSA_LARGE_OPTIMIZER["learning_rate"]
            * WSA_LARGE_SCHEDULER["min_lr_ratio"]
        ),
        "min_lr_ratio": WSA_LARGE_SCHEDULER["min_lr_ratio"],
        "last_epoch": int(scheduler.last_epoch),
        "learning_rates": learning_rates,
    }


def audit_wsa_lingbot_optimizer(policy: nn.Module, optimizer: Any) -> dict[str, Any]:
    """Prove every complete Future3D parameter has exactly one AdamW owner."""

    joint = policy.model.qwenvl_with_expert
    runtime = getattr(joint, "adr218_wsa_training_runtime", None)
    if not isinstance(runtime, WSALingBotTrainingRuntime):
        raise RuntimeError("ADR218 WSA runtime is not installed")
    future_parameters = {
        id(parameter): parameter
        for parameter in runtime.future.parameters()
        if parameter.requires_grad
    }
    if not future_parameters:
        raise RuntimeError("ADR218 Future3D exposes no trainable parameters")
    inner_optimizers = getattr(optimizer, "optimizers", None)
    if not isinstance(inner_optimizers, list) or not inner_optimizers:
        raise TypeError("LingBot combined optimizer exposes no inner optimizers")
    owners: dict[int, list[str]] = {parameter_id: [] for parameter_id in future_parameters}
    adamw_hyperparameters: list[dict[str, Any]] = []
    for inner in inner_optimizers:
        owner = type(inner).__name__
        for group in inner.param_groups:
            matched = [
                parameter for parameter in group["params"] if id(parameter) in future_parameters
            ]
            if not matched:
                continue
            for parameter in matched:
                owners[id(parameter)].append(owner)
            adamw_hyperparameters.append(
                {
                    "owner": owner,
                    "learning_rate": float(group["lr"]),
                    "betas": tuple(float(value) for value in group["betas"]),
                    "eps": float(group["eps"]),
                    "weight_decay": float(group["weight_decay"]),
                    "foreach": group["foreach"],
                    "fused": group["fused"],
                }
            )
    invalid_owners = {
        parameter_id: value for parameter_id, value in owners.items() if value != ["AdamW"]
    }
    if invalid_owners:
        raise RuntimeError(
            f"Future3D parameters do not have one AdamW owner: {len(invalid_owners)}"
        )
    expected_group = {
        "owner": "AdamW",
        "learning_rate": WSA_LARGE_OPTIMIZER["learning_rate"],
        "betas": WSA_LARGE_OPTIMIZER["adamw_betas"],
        "eps": WSA_LARGE_OPTIMIZER["adamw_eps"],
        "weight_decay": WSA_LARGE_OPTIMIZER["weight_decay"],
        "foreach": False,
        "fused": False,
    }
    if any(group != expected_group for group in adamw_hyperparameters):
        raise RuntimeError("Future3D AdamW hyperparameters differ from WSA-Large")
    return {
        "schema": "picf-next.adr218-wsa-lingbot-optimizer.v1",
        "future_trainable_parameter_tensors": len(future_parameters),
        "future_trainable_parameter_elements": sum(
            parameter.numel() for parameter in future_parameters.values()
        ),
        "optimizer_owner": "AdamW",
        "hyperparameters": expected_group,
        "gradient_clip_norm": WSA_LARGE_OPTIMIZER["grad_clip_norm"],
        "source": "WSA_LargeConfig.get_optimizer_preset",
        "backend_adaptation": (
            "mathematically-identical-single-tensor-adamw-for-dual-a100-peak-memory"
        ),
        "scheduler_scope": "not-authorized-by-this-mechanics-contract",
    }


def register_wsa_lingbot_fsdp_units(
    policy: nn.Module,
    runtime: WSALingBotTrainingRuntime,
) -> tuple[str, ...]:
    """Expose complete Future3D blocks to LingBot's existing FSDP2 builder."""
    block_classes = {type(block).__name__ for block in runtime.future.expert.blocks}
    if block_classes != {WSA_FSDP_BLOCK_CLASS}:
        raise RuntimeError(f"ADR218 Future3D block topology changed: {block_classes}")
    if type(runtime.future.expert).__name__ != WSA_FSDP_EXPERT_CLASS:
        raise RuntimeError("ADR218 Future3D expert topology changed")
    native_units = tuple(getattr(policy, "_no_split_modules", ()))
    if not native_units:
        raise RuntimeError("LingBot policy does not expose native FSDP units")
    units = tuple(
        dict.fromkeys(
            (*native_units, WSA_FSDP_BLOCK_CLASS, WSA_FSDP_EXPERT_CLASS)
        )
    )
    policy._no_split_modules = list(units)
    return units


def register_wsa_lingbot_fsdp_forward_methods(policy: nn.Module) -> dict[str, Any]:
    """Register both staged WSA block calls with composable FSDP2 hooks."""

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    joint = policy.model.qwenvl_with_expert
    runtime = getattr(joint, "adr218_wsa_training_runtime", None)
    if not isinstance(runtime, WSALingBotTrainingRuntime):
        raise RuntimeError("ADR218 WSA runtime is not installed")
    blocks = tuple(runtime.future.expert.blocks)
    if not blocks:
        raise RuntimeError("ADR218 WSA runtime exposes no Future3D blocks")
    for layer_index, block in enumerate(blocks):
        if not isinstance(block, FSDPModule):
            raise RuntimeError(f"Future3D block {layer_index} is not an FSDP2 unit")
        for method_name in (WSA_FSDP_QKV_METHOD, WSA_FSDP_POST_METHOD):
            if not callable(getattr(block, method_name, None)):
                raise RuntimeError(
                    f"Future3D block {layer_index} lacks staged method {method_name}"
                )
            register_fsdp_forward_method(block, method_name)
    expert = runtime.future.expert
    if not isinstance(expert, FSDPModule):
        raise RuntimeError("Future3D expert parent is not an FSDP2 unit")
    for method_name in WSA_FSDP_EXPERT_METHODS:
        if not callable(getattr(expert, method_name, None)):
            raise RuntimeError(f"Future3D expert lacks native method {method_name}")
        register_fsdp_forward_method(expert, method_name)
    receipt = {
        "schema": "picf-next.adr218-wsa-fsdp-forward-methods.v1",
        "block_count": len(blocks),
        "methods_per_block": [WSA_FSDP_QKV_METHOD, WSA_FSDP_POST_METHOD],
        "expert_methods": list(WSA_FSDP_EXPERT_METHODS),
        "registration_count": len(blocks) * 2 + len(WSA_FSDP_EXPERT_METHODS),
    }
    policy._adr218_wsa_fsdp_forward_methods_receipt = receipt
    return receipt


def _wsa_synchronous_picf_training_forward(
    flow: nn.Module,
    *,
    prefix_embs: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    prefix_position_ids: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    deepstack_visual_embeds: list[torch.Tensor] | None,
    suffix_embs: torch.Tensor,
    suffix_pad_masks: torch.Tensor,
    suffix_att_masks: torch.Tensor,
    time_embs: torch.Tensor,
    picf_native_context: Any,
    picf_action_attention_callback: Any = None,
) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Execute the released WSA experts synchronously on a PICF factual step.

    WSA's released mask contains both Future3D-to-action and
    action-to-Future3D edges.  A one-way prefix cache cannot preserve that
    graph, so factual WSA training must use the donor's simultaneous MoT
    execution instead of LingBot's PICF-only action-cache optimization.
    """

    from lingbotvla.models.vla.lingbot_vla.utils import (
        block_suffix_to_fv_,
        make_att_2d_masks,
    )

    if picf_native_context is None:
        raise ValueError("synchronous WSA/PICF training requires its typed context")
    prefix_len = int(prefix_pad_masks.shape[1])
    pad_masks = torch.cat((prefix_pad_masks, suffix_pad_masks), dim=1)
    att_masks = torch.cat((prefix_att_masks, suffix_att_masks), dim=1)
    attention_mask = make_att_2d_masks(pad_masks, att_masks)
    if flow.block_future_depth_to_action:
        attention_mask = block_suffix_to_fv_(
            attention_mask,
            suffix_row_start=prefix_len,
            prefix_len=prefix_len,
            num_task_tokens=flow.num_task_tokens,
        )
    attention_mask = flow._block_suffix_to_future_video_if_enabled_(
        attention_mask,
        suffix_row_start=prefix_len,
        prefix_len=prefix_len,
    )
    position_ids = flow._build_full_position_ids(
        prefix_position_ids,
        prefix_pad_masks,
        suffix_pad_masks,
    )
    outputs, _past_key_values, router_logits = flow.qwenvl_with_expert.forward(
        attention_mask=attention_mask,
        position_ids=position_ids,
        vlm_position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, suffix_embs],
        use_cache=False,
        fill_kv_cache=False,
        ada_cond=time_embs if getattr(flow.config, "adanorm_time", False) else None,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        picf_native_context=picf_native_context,
        picf_action_attention_callback=picf_action_attention_callback,
    )
    if (
        not isinstance(outputs, list)
        or len(outputs) != 2
        or outputs[0] is None
        or outputs[1] is None
    ):
        raise RuntimeError("synchronous WSA/PICF joint forward lost a required stream")
    return outputs[0][:, :prefix_len], outputs[1], router_logits


def install_wsa_lingbot_training_runtime(
    policy: nn.Module,
    runtime: WSALingBotTrainingRuntime,
) -> None:
    """Install the complete WSA training graph without editing donor source."""
    joint = policy.model.qwenvl_with_expert
    if hasattr(joint, "adr218_wsa_training_runtime"):
        raise RuntimeError("ADR218 WSA training runtime is already installed")
    flow = getattr(policy, "model", None)
    cached_training_forward = getattr(flow, "_picf_cached_training_forward", None)
    if not callable(cached_training_forward):
        raise TypeError("LingBot policy lacks the audited PICF training execution hook")
    if hasattr(flow, "_adr218_picf_cached_training_forward"):
        raise RuntimeError("ADR218 WSA/PICF synchronous execution is already installed")
    original_joint_forward = joint.forward
    original_policy_forward = policy.forward
    register_wsa_lingbot_fsdp_units(policy, runtime)
    joint.add_module("adr218_wsa_training_runtime", runtime)
    flow._adr218_picf_cached_training_forward = cached_training_forward
    flow._picf_cached_training_forward = MethodType(
        _wsa_synchronous_picf_training_forward,
        flow,
    )

    def joint_forward(installed_joint: nn.Module, *args: Any, **kwargs: Any):
        inputs_embeds = kwargs.get("inputs_embeds")
        use_cache = kwargs.get("use_cache")
        fill_kv_cache = kwargs.get("fill_kv_cache")
        training_surface = (
            isinstance(inputs_embeds, list)
            and len(inputs_embeds) == 2
            and all(tensor is not None for tensor in inputs_embeds)
            and not use_cache
        )
        if not training_surface:
            return original_joint_forward(*args, **kwargs)
        result = installed_joint.adr218_wsa_training_runtime(
            installed_joint,
            attention_mask=kwargs["attention_mask"],
            position_ids=kwargs["position_ids"],
            past_key_values=kwargs.get("past_key_values"),
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
            ada_cond=kwargs.get("ada_cond"),
            visual_pos_masks=kwargs.get("visual_pos_masks"),
            deepstack_visual_embeds=kwargs.get("deepstack_visual_embeds"),
            picf_native_context=kwargs.get("picf_native_context"),
            picf_action_attention_callback=kwargs.get("picf_action_attention_callback"),
        )
        return result.native_outputs, result.past_key_values, result.router_logits

    def policy_forward(installed_policy: nn.Module, *args: Any, **kwargs: Any):
        forbidden_legacy = tuple(
            name
            for name in ("wsa_da3_teacher_layers", "wsa_da3_view_valid")
            if name in kwargs
        )
        if forbidden_legacy:
            raise RuntimeError(
                "legacy untyped WSA supervision kwargs are forbidden: "
                f"{forbidden_legacy}"
            )
        contract = kwargs.pop("wsa_lingbot_forward_contract", None)
        if not isinstance(contract, WSALingBotForwardContract):
            raise TypeError("complete WSA action forward requires a typed role contract")
        ledger = getattr(installed_policy, WSA_STEP_LEDGER_ATTRIBUTE, None)
        if contract.role is WSALingBotForwardRole.PRIMARY_FACTUAL and not isinstance(
            ledger,
            WSALingBotStepLedger,
        ):
            raise RuntimeError("primary factual WSA forward is outside an optimizer transaction")
        runtime.assert_output_consumed()
        try:
            with runtime.attention_intervention_scope(contract.attention_intervention):
                official_outputs = original_policy_forward(*args, **kwargs)
        except BaseException:
            runtime.discard_latest_output()
            raise
        joint_output = runtime.take_latest_output()
        if contract.role is WSALingBotForwardRole.MEASUREMENT_ONLY:
            if contract.measurement_callback is not None:
                contract.measurement_callback(joint_output)
            if isinstance(ledger, WSALingBotStepLedger):
                ledger.record(contract.role)
            return official_outputs
        teacher_targets = contract.teacher_targets
        if teacher_targets is None:  # The typed contract validates this before execution.
            raise RuntimeError("primary factual WSA forward lost its DA3 targets")
        future_3d_loss, future_logs = compute_official_wsa_da3_loss(
            joint_output.future_projections,
            teacher_targets,
        )
        outputs = list(official_outputs)
        if len(outputs) < 7 or not isinstance(outputs[6], dict):
            raise RuntimeError("released LingBot output cannot accept the WSA loss receipt")
        outputs[0] = outputs[0] + WSA_RELEASED_LAMBDA_3D * future_3d_loss
        loss_dict = dict(outputs[6])
        loss_dict["loss_future_3d_objective"] = future_3d_loss
        loss_dict["loss_future_3d"] = future_3d_loss.detach()
        loss_dict["loss_future_3d_weighted"] = WSA_RELEASED_LAMBDA_3D * future_3d_loss.detach()
        loss_dict.update(future_logs)
        outputs[6] = loss_dict
        if not isinstance(ledger, WSALingBotStepLedger):  # Defensive after target computation.
            raise RuntimeError("primary factual WSA forward lost its optimizer ledger")
        ledger.record(contract.role)
        return tuple(outputs)

    joint.forward = MethodType(joint_forward, joint)
    policy.forward = MethodType(policy_forward, policy)


def wsa_lingbot_installation_receipt(policy: nn.Module) -> dict[str, Any]:
    joint = policy.model.qwenvl_with_expert
    runtime = getattr(joint, "adr218_wsa_training_runtime", None)
    if not isinstance(runtime, WSALingBotTrainingRuntime):
        raise RuntimeError("ADR218 WSA runtime is not installed")
    return {
        "schema": "picf-next.adr218-wsa-lingbot-installation.v1",
        "future_parameter_count": sum(
            parameter.numel() for parameter in runtime.future.parameters()
        ),
        "future_layers": len(runtime.future.expert.blocks),
        "future_slots": runtime.future.expert.num_query_tokens,
        "lambda_3d": WSA_RELEASED_LAMBDA_3D,
        "cached_inference_authorized": False,
        "factual_training_execution": "released-wsa-synchronous-mot",
        "action_coupling": runtime.action_coupling.value,
        "picf_action_cache_used_for_factual_wsa": False,
        "synchronous_action_attention_receipts_authorized": True,
        "training_requires_da3_targets": True,
        "forward_role_contract": [
            role.value for role in WSALingBotForwardRole
        ],
        "primary_factual_per_optimizer_transaction": 1,
        "fsdp_unit_classes": [WSA_FSDP_BLOCK_CLASS, WSA_FSDP_EXPERT_CLASS],
        "fsdp_registered": all(
            name in policy._no_split_modules
            for name in (WSA_FSDP_BLOCK_CLASS, WSA_FSDP_EXPERT_CLASS)
        ),
        "fsdp_forward_methods": getattr(
            policy,
            "_adr218_wsa_fsdp_forward_methods_receipt",
            None,
        ),
    }

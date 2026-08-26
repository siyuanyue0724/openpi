"""PICF-neutral helpers for pinned LingBot-VLA2 deployment commands."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import subprocess
from collections.abc import Callable, Iterable, Mapping
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

from torch import nn

TARGET_ONLY_FIELDS = frozenset(
    {
        "action",
        "actions",
        "action_is_pad",
        "joint_mask",
        "mask",
        "masks",
        "object_id",
        "segmentation",
        "teacher",
        "target",
        "targets",
    }
)

_ALIGNMENT_TEACHER_HEADS = (
    "depth_align_head",
    "future_depth_align_head",
    "current_video_align_head",
    "future_video_align_head",
    "future_video_cls_head",
)

LINGBOT_DETERMINISTIC_MOE_BACKEND = "fused_moe_forward"
LINGBOT_RELEASED_ACTION_SAMPLING_STEPS = 10


def build_lingbot_base_family_identity(
    *,
    source_commit: str,
    native_patch_sha256: str,
    checkpoint_revision: str,
    checkpoint_report: Mapping[str, Any],
    processor_revision: str,
    processor_report: Mapping[str, Any],
    attention_implementation: str,
    trainable_scope: str,
    optimizer_contract: Mapping[str, Any],
    maximum_control_tokens: int,
) -> dict[str, Any]:
    """Bind the shared LingBot initialization without conflating treatments.

    PICF deliberately adds a graph, dense observations, and a VidEoMT source,
    so its complete model-family digest cannot equal the control arm's digest.
    This identity covers only the released host state and matched execution
    semantics that must be common before either treatment is installed.
    Runtime-only collective alignment patches are excluded; their underlying
    native model patch is included explicitly.
    """

    sha_fields = {
        "source_commit": source_commit,
        "native_patch_sha256": native_patch_sha256,
        "checkpoint_revision": checkpoint_revision,
        "processor_revision": processor_revision,
    }
    for name, value in sha_fields.items():
        if (
            not isinstance(value, str)
            or len(value) not in {40, 64}
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ValueError(f"LingBot base-family {name} is not a lowercase digest")
    if attention_implementation not in {"eager", "flex_cached"}:
        raise ValueError("LingBot base-family attention implementation is unsupported")
    if not isinstance(trainable_scope, str) or not trainable_scope:
        raise ValueError("LingBot base-family trainable scope is absent")
    if (
        isinstance(maximum_control_tokens, bool)
        or not isinstance(maximum_control_tokens, int)
        or maximum_control_tokens <= 0
    ):
        raise ValueError("LingBot base-family control-token budget must be positive")

    checkpoint_assets = checkpoint_report.get("checkpoint_assets")
    processor_assets = processor_report.get("processor_assets")
    if not isinstance(checkpoint_assets, list) or not checkpoint_assets:
        raise ValueError("LingBot base-family checkpoint assets are absent")
    if not isinstance(processor_assets, list) or not processor_assets:
        raise ValueError("LingBot base-family processor assets are absent")
    payload = {
        "schema": "picf-next.lingbot-base-family.v1",
        "architecture": "released_lingbot_vla2_action_policy",
        "source_commit": source_commit,
        "native_patch_sha256": native_patch_sha256,
        "checkpoint_revision": checkpoint_revision,
        "checkpoint_assets": deepcopy(checkpoint_assets),
        "processor_revision": processor_revision,
        "processor_assets": deepcopy(processor_assets),
        "attention_implementation": attention_implementation,
        "trainable_scope": trainable_scope,
        "optimizer_contract": deepcopy(dict(optimizer_contract)),
        "maximum_control_tokens": maximum_control_tokens,
    }
    canonical = json.dumps(
        payload,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return {**payload, "artifact_sha256": hashlib.sha256(canonical).hexdigest()}


def require_lingbot_released_action_sampling_steps(config: Any) -> None:
    """Keep training checkpoints deployable with LingBot's released flow solver."""

    value = getattr(config, "num_steps", None)
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value != LINGBOT_RELEASED_ACTION_SAMPLING_STEPS
    ):
        raise ValueError("LingBot training requires the released 10-step action sampling contract")


def select_lingbot_deterministic_moe_backend(
    *,
    action_expert_module: Any,
    fused_moe_module: Any,
) -> dict[str, Any]:
    """Select LingBot's released deterministic fallback for inference.

    The optional Robby kernel accumulates top-k expert outputs with relaxed
    atomics. LingBot already uses ``fused_moe_forward`` for training and as its
    official inference fallback; this selector only disables the optional
    non-deterministic accelerator in the current process.
    """

    fallback = getattr(fused_moe_module, LINGBOT_DETERMINISTIC_MOE_BACKEND, None)
    if not callable(fallback):
        raise RuntimeError("LingBot deterministic fused MoE fallback is unavailable")
    robby_before = getattr(action_expert_module, "robby_moe_forward", None)
    action_expert_module.robby_moe_forward = None
    if getattr(action_expert_module, "robby_moe_forward", object()) is not None:
        raise RuntimeError("LingBot Robby MoE inference accelerator was not disabled")
    return {
        "schema": "picf-next.lingbot-moe-inference-backend.v1",
        "selected": LINGBOT_DETERMINISTIC_MOE_BACKEND,
        "fused_fallback_available": True,
        "robby_available_before_selection": callable(robby_before),
        "robby_disabled": True,
    }


def clip_lingbot_distributed_l2_grad_norm_(
    parameters: Iterable[Any],
    max_norm: float,
    *,
    device: Any,
    dist_module: Any,
    torch_module: Any,
    error_if_nonfinite: bool = True,
) -> float:
    """Clip FSDP2 gradients with one rank-invariant global reduction.

    Sparse MoE routing can leave a DTensor shard with ``grad=None`` on one
    rank while another rank owns a finite shard gradient.  Generic
    ``clip_grad_norm_`` follows only locally present gradients, so its DTensor
    collectives can diverge across ranks.  Here every rank instead contributes
    one scalar to the same reduction; absent gradients contribute zero.

    For a parameter replicated over ``rho`` mesh ranks, each local squared norm
    is weighted by ``1 / rho`` before the world reduction.  Sharded parameters
    need no correction because their local tensors are disjoint.  Partial
    placements are rejected: the squared norm of partial values is not the norm
    of their sum.
    """

    if isinstance(max_norm, bool) or not isinstance(max_norm, (int, float)):
        raise TypeError("LingBot distributed max_norm must be one finite number")
    maximum = float(max_norm)
    if not math.isfinite(maximum) or maximum < 0.0:
        raise ValueError("LingBot distributed max_norm must be finite and non-negative")

    values = tuple(parameters)
    initialized = bool(
        getattr(dist_module, "is_available", lambda: True)()
        and getattr(dist_module, "is_initialized", lambda: False)()
    )
    world_size = int(dist_module.get_world_size()) if initialized else 1
    replication_factors: dict[int, int] = {}
    for parameter in values:
        if not bool(getattr(parameter, "requires_grad", False)):
            continue
        to_local = getattr(parameter, "to_local", None)
        if not callable(to_local):
            if world_size != 1:
                raise RuntimeError(
                    "distributed LingBot clipping requires DTensor placement metadata"
                )
            replication_factors[id(parameter)] = 1
            continue
        placements = getattr(parameter, "placements", None)
        mesh = getattr(parameter, "device_mesh", None)
        if placements is None or mesh is None:
            raise RuntimeError("LingBot DTensor parameter omitted placement metadata")
        replication = 1
        for mesh_dimension, placement in enumerate(placements):
            placement_name = type(placement).__name__
            if placement_name == "Shard":
                continue
            if placement_name == "Replicate":
                replication *= int(mesh.size(mesh_dimension))
                continue
            if placement_name == "Partial":
                raise RuntimeError(
                    "distributed LingBot clipping does not accept Partial DTensor placements"
                )
            raise RuntimeError(
                f"distributed LingBot clipping received unsupported placement {placement_name!r}"
            )
        replication_factors[id(parameter)] = replication

    local_squared = torch_module.zeros((), dtype=torch_module.float64, device=device)
    local_sparse = torch_module.zeros((), dtype=torch_module.float64, device=device)
    local_gradients: list[Any] = []
    for parameter in values:
        gradient = getattr(parameter, "grad", None)
        if gradient is None:
            continue
        local = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
        if bool(getattr(local, "is_sparse", False)):
            local_sparse.add_(1.0)
            continue
        replication = replication_factors.get(id(parameter))
        if replication is None:
            raise RuntimeError("LingBot gradient belongs to a non-trainable parameter")
        square = (
            local.detach()
            .float()
            .square()
            .sum()
            .to(
                device=device,
                dtype=torch_module.float64,
            )
        )
        local_squared.add_(square / replication)
        local_gradients.append(local)

    packed = torch_module.stack((local_squared, local_sparse))
    if initialized and world_size > 1:
        dist_module.all_reduce(packed, op=dist_module.ReduceOp.SUM)
    sparse_count = int(packed[1].item())
    if sparse_count:
        raise RuntimeError("distributed LingBot clipping does not support sparse gradients")
    norm = math.sqrt(float(packed[0].item()))
    if not math.isfinite(norm):
        if error_if_nonfinite:
            raise FloatingPointError("LingBot distributed gradient norm is non-finite")
        return norm

    coefficient = min(1.0, maximum / (norm + 1e-6))
    if coefficient < 1.0:
        with torch_module.no_grad():
            for local in local_gradients:
                local.mul_(coefficient)
    return norm


@dataclass(frozen=True, slots=True)
class LingBotOptimizerContract:
    """Exact released RoboTwin optimizer semantics used by native PICF runs."""

    algorithm: str
    learning_rate: float
    weight_decay: float
    adamw_betas: tuple[float, float]
    adamw_eps: float
    muon_momentum: float
    muon_nesterov: bool
    muon_ns_steps: int
    muon_adjust_lr_fn: str | None
    muon_exclude_name_patterns: tuple[str, ...]
    use_moe: bool
    use_moe_expert_lr: bool
    token_moe_layers: tuple[int, ...]
    token_num_experts: int
    token_top_k: int
    bias_update_speed: float
    bias_centering: bool
    bias_update_interval: int
    sequence_wise_loss_coeff: float
    sequence_wise_mode: str
    router_z_loss_coeff: float
    router_activation: str
    routed_scaling_factor: float
    use_shared_expert_gate: bool
    enable_fp32: bool
    enable_mixed_precision: bool
    scheduler: str
    scheduler_warmup_ratio: float
    scheduler_start_lr: float

    @property
    def metadata(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["adamw_betas"] = list(self.adamw_betas)
        payload["muon_exclude_name_patterns"] = list(self.muon_exclude_name_patterns)
        payload["token_moe_layers"] = list(self.token_moe_layers)
        payload["builder"] = "lingbotvla.optim.build_muon_optimizer"
        payload["moe_hook"] = (
            "lingbotvla.models.vla.lingbot_vla.moe_load_balance.build_moe_load_balance_hook"
        )
        payload["scheduler_implementation"] = "constant_identity_no_state"
        return payload

    def official_arguments(self) -> SimpleNamespace:
        return SimpleNamespace(
            bias_centering=self.bias_centering,
            bias_update_interval=self.bias_update_interval,
            bias_update_speed=self.bias_update_speed,
            muon_adjust_lr_fn=self.muon_adjust_lr_fn,
            muon_exclude_name_patterns=list(self.muon_exclude_name_patterns),
            muon_momentum=self.muon_momentum,
            muon_nesterov=self.muon_nesterov,
            muon_ns_steps=self.muon_ns_steps,
            token_moe_layers=list(self.token_moe_layers),
            token_num_experts=self.token_num_experts,
            token_top_k=self.token_top_k,
            use_moe=self.use_moe,
            use_moe_expert_lr=self.use_moe_expert_lr,
        )


def _finite_config_float(value: Any, name: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"LingBot {name} must be one finite number")
    result = float(value)
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise ValueError(f"LingBot {name} is outside its valid range")
    return result


def _positive_config_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"LingBot {name} must be one integer")
    if value <= 0:
        raise ValueError(f"LingBot {name} must be positive")
    return value


def resolve_lingbot_optimizer_contract(
    training: dict[str, Any],
    *,
    requested_learning_rate: float,
) -> LingBotOptimizerContract:
    """Bind PICF to the pinned release's Muon/AdamW RoboTwin recipe.

    The native runner deliberately supports only the identity constant schedule.
    A non-identity schedule needs scheduler state in the atomic checkpoint and is
    therefore rejected instead of being silently approximated.
    """

    train = training.get("train")
    if not isinstance(train, dict):
        raise ValueError("LingBot training YAML must contain a train mapping")
    if train.get("optimizer", "adamw") != "muon":
        raise ValueError("native PICF requires the released LingBot Muon recipe")
    learning_rate = _finite_config_float(train.get("lr", 5e-5), "train.lr", minimum=0.0)
    if learning_rate <= 0:
        raise ValueError("LingBot train.lr must be positive")
    requested = _finite_config_float(
        requested_learning_rate,
        "requested learning rate",
        minimum=0.0,
    )
    if requested != learning_rate:
        raise ValueError(
            "native PICF learning rate differs from the immutable LingBot training YAML"
        )
    weight_decay = _finite_config_float(
        train.get("weight_decay", 0.0),
        "train.weight_decay",
        minimum=0.0,
    )
    use_moe = train.get("use_moe", False)
    use_moe_expert_lr = train.get("use_moe_expert_lr", False)
    if use_moe is not True or use_moe_expert_lr is not True:
        raise ValueError("released native PICF requires LingBot MoE expert LR scaling")
    layers = train.get("token_moe_layers")
    if (
        not isinstance(layers, list)
        or not layers
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in layers
        )
        or len(set(layers)) != len(layers)
    ):
        raise ValueError("LingBot token_moe_layers must be unique non-negative integers")
    exclude = train.get("muon_exclude_name_patterns", [])
    if not isinstance(exclude, list) or any(not isinstance(value, str) for value in exclude):
        raise TypeError("LingBot muon_exclude_name_patterns must be a string list")
    adjust = train.get("muon_adjust_lr_fn", "match_rms_adamw")
    if adjust not in (None, "original", "match_rms_adamw"):
        raise ValueError("LingBot muon_adjust_lr_fn is unsupported")
    nesterov = train.get("muon_nesterov", True)
    bias_centering = train.get("bias_centering", False)
    use_shared_expert_gate = train.get("use_shared_expert_gate", True)
    enable_fp32 = train.get("enable_fp32", False)
    enable_mixed_precision = train.get("enable_mixed_precision", True)
    if any(
        not isinstance(value, bool)
        for value in (
            nesterov,
            bias_centering,
            use_shared_expert_gate,
            enable_fp32,
            enable_mixed_precision,
        )
    ):
        raise TypeError("LingBot Muon and MoE boolean controls must be booleans")
    if enable_fp32 is not True or enable_mixed_precision is not True:
        raise ValueError("native PICF requires LingBot's FP32 action-expert precision recipe")
    scheduler = train.get("lr_decay_style", "constant")
    warmup_ratio = _finite_config_float(
        train.get("lr_warmup_ratio", 0.0),
        "train.lr_warmup_ratio",
        minimum=0.0,
    )
    start_lr = _finite_config_float(
        train.get("lr_start", 0.0),
        "train.lr_start",
        minimum=0.0,
    )
    if scheduler != "constant" or warmup_ratio != 0.0 or start_lr != 0.0:
        raise ValueError("native PICF currently supports only LingBot's identity constant schedule")
    num_experts = _positive_config_int(train.get("token_num_experts", 32), "token_num_experts")
    top_k = _positive_config_int(train.get("token_top_k", 1), "token_top_k")
    if top_k > num_experts:
        raise ValueError("LingBot token_top_k exceeds token_num_experts")
    sequence_wise_mode = train.get("sequence_wise_mode", "per_sequence")
    if sequence_wise_mode not in ("per_sequence", "global"):
        raise ValueError("LingBot sequence_wise_mode is unsupported")
    router_activation = train.get("router_activation", "softmax")
    if router_activation not in ("softmax", "sigmoid"):
        raise ValueError("LingBot router_activation is unsupported")
    routed_scaling_factor = _finite_config_float(
        train.get("routed_scaling_factor", 1.0),
        "train.routed_scaling_factor",
        minimum=0.0,
    )
    if routed_scaling_factor <= 0:
        raise ValueError("LingBot train.routed_scaling_factor must be positive")
    return LingBotOptimizerContract(
        algorithm="lingbot_distributed_muon_with_adamw_fallback",
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        adamw_betas=(0.9, 0.95),
        adamw_eps=1e-8,
        muon_momentum=_finite_config_float(
            train.get("muon_momentum", 0.95),
            "train.muon_momentum",
            minimum=0.0,
        ),
        muon_nesterov=nesterov,
        muon_ns_steps=_positive_config_int(train.get("muon_ns_steps", 5), "muon_ns_steps"),
        muon_adjust_lr_fn=adjust,
        muon_exclude_name_patterns=tuple(exclude),
        use_moe=use_moe,
        use_moe_expert_lr=use_moe_expert_lr,
        token_moe_layers=tuple(layers),
        token_num_experts=num_experts,
        token_top_k=top_k,
        bias_update_speed=_finite_config_float(
            train.get("bias_update_speed", 0.001),
            "train.bias_update_speed",
            minimum=0.0,
        ),
        bias_centering=bias_centering,
        bias_update_interval=_positive_config_int(
            train.get("bias_update_interval", 1),
            "bias_update_interval",
        ),
        sequence_wise_loss_coeff=_finite_config_float(
            train.get("sequence_wise_loss_coeff", 0.0),
            "train.sequence_wise_loss_coeff",
            minimum=0.0,
        ),
        sequence_wise_mode=sequence_wise_mode,
        router_z_loss_coeff=_finite_config_float(
            train.get("router_z_loss_coeff", 0.0),
            "train.router_z_loss_coeff",
            minimum=0.0,
        ),
        router_activation=router_activation,
        routed_scaling_factor=routed_scaling_factor,
        use_shared_expert_gate=use_shared_expert_gate,
        enable_fp32=enable_fp32,
        enable_mixed_precision=enable_mixed_precision,
        scheduler=scheduler,
        scheduler_warmup_ratio=warmup_ratio,
        scheduler_start_lr=start_lr,
    )


def require_lingbot_exact_resume_contract(
    contract: LingBotOptimizerContract,
) -> None:
    """Reject released optimizer modes whose runtime state is not serialized.

    LingBot's load-balancing hook keeps both its modulo phase and accumulated
    ``tokens_per_expert`` counts outside ``state_dict``.  With an update
    interval of one, every optimizer step consumes and clears those counts, so
    an atomic post-step checkpoint is a complete trajectory boundary.  Longer
    intervals require upstream checkpoint support for the hook state.
    """

    if not isinstance(contract, LingBotOptimizerContract):
        raise TypeError("native LingBot exact resume requires its frozen optimizer contract")
    if contract.bias_update_interval != 1:
        raise ValueError(
            "native LingBot exact resume requires bias_update_interval=1 because "
            "the released load-balance hook does not serialize its phase or "
            "tokens_per_expert accumulator"
        )


def build_lingbot_official_optimizer(
    model: Any,
    contract: LingBotOptimizerContract,
    *,
    build_muon_optimizer: Callable[..., Any],
    build_moe_load_balance_hook: Callable[..., Any],
) -> Any:
    """Invoke the pinned LingBot optimizer and loss-free MoE hook unchanged."""

    if not isinstance(contract, LingBotOptimizerContract):
        raise TypeError("native LingBot optimizer requires its frozen contract")
    arguments = contract.official_arguments()
    optimizer = build_muon_optimizer(
        model,
        arguments,
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
        adamw_betas=contract.adamw_betas,
        adamw_eps=contract.adamw_eps,
    )
    hook = build_moe_load_balance_hook(
        model,
        coeff=contract.bias_update_speed,
        bias_centering=contract.bias_centering,
        update_interval=contract.bias_update_interval,
    )
    optimizer.register_step_pre_hook(hook)
    return optimizer


def configure_picf_optimizer_learning_rates(
    optimizer: Any,
    graph: nn.Module,
    *,
    picf_multiplier: float,
    modality_bridge_multiplier: float,
) -> dict[str, Any]:
    """Stratify the released optimizer without changing its update algorithms.

    LingBot's builder has already separated Muon and AdamW parameters at this
    point.  Splitting each existing group by ownership therefore preserves all
    released optimizer metadata while allowing newly initialized PICF
    parameters and the copied pretrained modality bridge to use different
    learning-rate scales.  This must run before the first optimizer step and
    before loading a checkpoint so fresh and resumed group layouts agree.
    """

    multipliers = {
        "lingbot_host": 1.0,
        "picf_graph": float(picf_multiplier),
        "pretrained_modality_bridge": float(modality_bridge_multiplier),
    }
    if any(not math.isfinite(value) or value <= 0.0 for value in multipliers.values()):
        raise ValueError("PICF optimizer learning-rate multipliers must be finite and positive")
    state = getattr(optimizer, "state", None)
    if state is None or len(state):
        raise RuntimeError("PICF learning-rate stratification requires a fresh optimizer")
    param_groups = getattr(optimizer, "param_groups", None)
    if not isinstance(param_groups, list) or not param_groups:
        raise TypeError("released LingBot optimizer does not expose mutable parameter groups")

    all_graph_parameters = {
        id(parameter): (name, bool(parameter.requires_grad))
        for name, parameter in graph.named_parameters()
    }
    graph_parameters = {
        parameter_id: name
        for parameter_id, (name, requires_grad) in all_graph_parameters.items()
        if requires_grad
    }
    frozen_graph_parameters = {
        parameter_id: name
        for parameter_id, (name, requires_grad) in all_graph_parameters.items()
        if not requires_grad
    }
    if not graph_parameters:
        raise RuntimeError("installed PICF graph has no trainable parameters")
    bridge_parameters = {
        parameter_id
        for parameter_id, name in graph_parameters.items()
        if name.startswith("modality_bridge.")
    }
    optimizer_parameter_ids: set[int] = set()
    replacement: list[dict[str, Any]] = []
    role_numel = {name: 0 for name in multipliers}
    role_parameter_count = {name: 0 for name in multipliers}

    for source_group in param_groups:
        if not isinstance(source_group, dict) or "params" not in source_group:
            raise TypeError("released LingBot optimizer group is malformed")
        try:
            source_lr = float(source_group["lr"])
        except (KeyError, TypeError, ValueError) as error:
            raise TypeError(
                "released LingBot optimizer group has no numeric learning rate"
            ) from error
        if not math.isfinite(source_lr) or source_lr <= 0.0:
            raise ValueError("released LingBot optimizer group learning rate is invalid")
        buckets: dict[str, list[Any]] = {name: [] for name in multipliers}
        for parameter in source_group["params"]:
            parameter_id = id(parameter)
            if parameter_id in optimizer_parameter_ids:
                raise RuntimeError("released LingBot optimizer contains a duplicate parameter")
            optimizer_parameter_ids.add(parameter_id)
            if parameter_id in frozen_graph_parameters:
                raise RuntimeError(
                    "released LingBot optimizer contains frozen PICF parameter: "
                    f"{frozen_graph_parameters[parameter_id]}"
                )
            if parameter_id in bridge_parameters:
                role = "pretrained_modality_bridge"
            elif parameter_id in graph_parameters:
                role = "picf_graph"
            else:
                role = "lingbot_host"
            buckets[role].append(parameter)
            role_parameter_count[role] += 1
            role_numel[role] += int(parameter.numel())

        for role, parameters in buckets.items():
            if not parameters:
                continue
            group = dict(source_group)
            group["params"] = parameters
            group["lr"] = source_lr * multipliers[role]
            if "initial_lr" in group:
                group["initial_lr"] = float(group["initial_lr"]) * multipliers[role]
            group["picf_learning_rate_role"] = role
            replacement.append(group)

    missing_graph_parameters = set(graph_parameters).difference(optimizer_parameter_ids)
    if missing_graph_parameters:
        names = sorted(graph_parameters[value] for value in missing_graph_parameters)
        raise RuntimeError(f"PICF graph parameters are absent from the optimizer: {names}")
    if bridge_parameters and not role_parameter_count["pretrained_modality_bridge"]:
        raise RuntimeError("pretrained modality bridge was not assigned its learning-rate group")
    optimizer.param_groups = replacement
    return {
        "schema": "picf-next.optimizer-learning-rate-stratification/v1",
        "multipliers": multipliers,
        "parameter_count": role_parameter_count,
        "parameter_numel": role_numel,
        "group_count": len(replacement),
    }


def build_lingbot_representation_optimizer(
    model: Any,
    contract: LingBotOptimizerContract,
    *,
    build_muon_optimizer: Callable[..., Any],
) -> Any:
    """Use LingBot's released update rule without an unexecuted action-MoE hook.

    Representation training never constructs the action suffix, so its expert
    router has no token counts to balance. Registering the stateful action hook
    would create a phase-private mutation unrelated to the representation
    objective. Muon/AdamW grouping, learning rate and expert scaling remain the
    exact released implementation.
    """

    if not isinstance(contract, LingBotOptimizerContract):
        raise TypeError("native LingBot representation optimizer requires its frozen contract")
    return build_muon_optimizer(
        model,
        contract.official_arguments(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
        adamw_betas=contract.adamw_betas,
        adamw_eps=contract.adamw_eps,
    )


def build_lingbot_query_only_optimizer(
    model: Any,
    contract: LingBotOptimizerContract,
    *,
    build_optimizer: Callable[..., Any],
) -> Any:
    """Use LingBot's released AdamW fallback for an embedding-only scope.

    LingBot's released Muon builder deliberately routes embedding parameters
    to AdamW, but it rejects a model with no Muon-eligible matrix weights.  A
    query-only causal gate has exactly that scope, so invoke the released
    ``build_optimizer(optimizer_type="adamw")`` path with the same immutable
    optimizer contract instead of inventing a probe-specific update rule.
    """

    if not isinstance(contract, LingBotOptimizerContract):
        raise TypeError("native LingBot query-only optimizer requires its frozen contract")
    return build_optimizer(
        model,
        lr=contract.learning_rate,
        betas=contract.adamw_betas,
        eps=contract.adamw_eps,
        weight_decay=contract.weight_decay,
        fused=False,
        optimizer_type="adamw",
        post_training=True,
    )


def build_lingbot_fixed_batch_probe_optimizer(
    model: Any,
    contract: LingBotOptimizerContract,
    *,
    build_muon_optimizer: Callable[..., Any],
) -> Any:
    """Use the released update rule without changing MoE routing state.

    The fixed-batch capacity experiment compares trainable parameter scopes.
    Registering LingBot's load-balance hook would mutate routing bias even when
    the shared host is frozen, invalidating the readout-only control.  This
    evidence-only builder keeps Muon/AdamW, LR and expert scaling unchanged and
    omits only that stateful pre-step hook.
    """

    if not isinstance(contract, LingBotOptimizerContract):
        raise TypeError("native LingBot probe optimizer requires its frozen contract")
    return build_muon_optimizer(
        model,
        contract.official_arguments(),
        lr=contract.learning_rate,
        weight_decay=contract.weight_decay,
        adamw_betas=contract.adamw_betas,
        adamw_eps=contract.adamw_eps,
    )


def _owned_parameters(value: Any) -> tuple[Any, ...]:
    parameters = getattr(value, "parameters", None)
    if callable(parameters):
        result = parameters()
        if not isinstance(result, Iterable):
            raise TypeError("LingBot alignment component returned non-iterable parameters")
        return tuple(result)
    if callable(getattr(value, "numel", None)):
        return (value,)
    raise TypeError("LingBot alignment component does not expose parameters")


def strip_targetless_alignment_teacher_heads(policy: Any) -> dict[str, Any]:
    """Remove only terminal teacher decoders from the targetless action path."""

    flow = getattr(policy, "model", None)
    if flow is None or not bool(getattr(flow, "use_depth_align", False)):
        raise RuntimeError("LingBot targetless deployment requires released alignment queries")
    if "compute_alignment_losses" not in inspect.signature(flow.forward).parameters:
        raise RuntimeError("LingBot forward cannot disable alignment teacher losses")

    expected_heads = {"depth_align_head"}
    preserved = {"depth_align_embs"}
    if bool(getattr(flow, "use_future_depth", False)):
        expected_heads.add("future_depth_align_head")
        preserved.add("future_depth_align_embs")
    if bool(getattr(flow, "use_current_video_patch", False)):
        expected_heads.add("current_video_align_head")
        preserved.add("current_video_align_embs")
    if bool(getattr(flow, "use_future_video_patch", False)):
        expected_heads.add("future_video_align_head")
        if not bool(getattr(flow, "future_video_share_future_depth_query", False)) or bool(
            getattr(flow, "use_shared_future_task_proj", False)
        ):
            preserved.add("future_video_align_embs")
    if bool(getattr(flow, "use_future_video_cls", False)):
        expected_heads.add("future_video_cls_head")
        preserved.add("future_video_cls_align_emb")
    if bool(getattr(flow, "use_current_shared_task_proj", False)):
        preserved.add("current_shared_task_proj")
    if bool(getattr(flow, "use_shared_future_task_proj", False)):
        preserved.add("future_shared_task_proj")

    present_heads = {name for name in _ALIGNMENT_TEACHER_HEADS if hasattr(flow, name)}
    if present_heads != expected_heads:
        raise RuntimeError(
            "LingBot alignment teacher-head topology differs from the released configuration"
        )
    missing_preserved = sorted(name for name in preserved if not hasattr(flow, name))
    if missing_preserved:
        raise RuntimeError(
            f"LingBot targetless deployment would lose query producers: {missing_preserved}"
        )

    preserved_parameter_ids = {
        id(parameter) for name in preserved for parameter in _owned_parameters(getattr(flow, name))
    }
    removed = []
    removed_parameter_ids: set[int] = set()
    for name in sorted(present_heads):
        head = getattr(flow, name)
        parameters = _owned_parameters(head)
        parameter_ids = {id(parameter) for parameter in parameters}
        if removed_parameter_ids.intersection(parameter_ids):
            raise RuntimeError("LingBot alignment teacher heads unexpectedly share parameters")
        if preserved_parameter_ids.intersection(parameter_ids):
            raise RuntimeError("LingBot teacher decoder shares parameters with retained queries")
        removed.append(
            {
                "name": name,
                "parameter_count": len(parameters),
                "numel": sum(int(parameter.numel()) for parameter in parameters),
                "storage_bytes": sum(
                    int(parameter.numel()) * int(parameter.element_size())
                    for parameter in parameters
                ),
            }
        )
        removed_parameter_ids.update(parameter_ids)
        delattr(flow, name)

    if any(hasattr(flow, name) for name in present_heads):
        raise RuntimeError("LingBot alignment teacher-head removal was incomplete")
    report = {
        "schema": "picf-next.targetless-alignment-teacher-prune.v1",
        "removed": removed,
        "removed_numel": sum(value["numel"] for value in removed),
        "removed_storage_bytes": sum(value["storage_bytes"] for value in removed),
        "retained_query_components": sorted(preserved),
    }
    flow._picf_targetless_alignment_teacher_prune = report
    return report


def register_native_fsdp_forward_methods(policy: Any) -> None:
    """Register every non-``forward`` native root path with FSDP2 hooks."""

    from torch.distributed.fsdp import FSDPModule, register_fsdp_forward_method

    if not isinstance(policy, FSDPModule) or not isinstance(policy, nn.Module):
        raise RuntimeError("LingBot policy must be the root FSDP2 unit")
    method_names = (
        "sample_actions",
        "picf_native_prior_forward",
        "picf_native_observation_forward",
        "picf_native_frozen_posterior_action_forward",
    )
    missing = tuple(name for name in method_names if not callable(getattr(policy, name, None)))
    if missing:
        raise TypeError(f"LingBot policy lacks audited native root methods: {missing}")
    for name in method_names:
        register_fsdp_forward_method(policy, name)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    array = tensor.detach().to(dtype=tensor.new_zeros(()).float().dtype, device="cpu")
    return hashlib.sha256(array.contiguous().numpy().tobytes()).hexdigest()


def load_lingbot_training_config(path: Path) -> dict[str, Any]:
    """Load released YAML with LingBot-compatible typed scalar semantics."""

    if not isinstance(path, Path) or not path.is_file():
        raise FileNotFoundError(path)
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as error:
        raise RuntimeError("released LingBot YAML requires the pinned structured parser") from error
    resolved = OmegaConf.to_container(
        OmegaConf.load(path),
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(resolved, dict) or any(
        not isinstance(name, str) or not name for name in resolved
    ):
        raise TypeError("released LingBot YAML must resolve to a string-keyed mapping")
    return cast(dict[str, Any], resolved)


def _merge_training_sections(training: dict[str, Any]) -> dict[str, Any]:
    model = training.get("model")
    train = training.get("train")
    if not isinstance(model, dict) or not isinstance(train, dict):
        raise ValueError("LingBot training YAML must contain model and train mappings")
    merged = deepcopy(model)
    merged.update(deepcopy(train))
    return merged


def _resolve_training_config(
    training: dict[str, Any],
    *,
    checkpoint_dir: Path,
    processor_dir: Path,
    num_steps: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve released assets while preserving the official model choices."""

    if num_steps <= 0:
        raise ValueError("num-steps must be positive")
    resolved = deepcopy(training)
    merged = _merge_training_sections(resolved)
    merged["tokenizer_path"] = str(processor_dir.resolve())
    merged["model_path"] = str(checkpoint_dir.resolve())
    merged["use_cache"] = True
    merged["use_compile"] = False
    merged["attention_implementation"] = "eager"
    merged["vit_attn_implementation"] = "eager"
    merged["num_steps"] = num_steps

    align = deepcopy(merged.get("align_params") or {})
    depth = deepcopy(align.get("depth") or {})
    video = deepcopy(align.get("video") or {})
    depth["morgbd_path"] = str((checkpoint_dir / "depth/model.pt").resolve())
    video["ckpt_path"] = str((checkpoint_dir / "dino_video/teacher_step_10000.pth").resolve())
    video["config_path"] = str((checkpoint_dir / "dino_video/config.yaml").resolve())
    if depth:
        align["depth"] = depth
    if video:
        align["video"] = video
    if align:
        merged["align_params"] = align

    data = resolved.get("data")
    if not isinstance(data, dict):
        raise ValueError("LingBot training YAML must contain a data mapping")
    return merged, deepcopy(data)


def _asset_manifest(root: Path, required: tuple[str, ...]) -> list[dict[str, Any]]:
    manifest = []
    for relative in sorted(required):
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
        )
    return manifest


def _cuda_memory(torch: Any, device: Any) -> dict[str, int] | None:
    if device.type != "cuda":
        return None
    return {
        "allocated": int(torch.cuda.memory_allocated(device)),
        "reserved": int(torch.cuda.memory_reserved(device)),
        "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
    }


def _merge_qwen_config(config: Any, qwen_config: Any) -> None:
    """Copy the exact Qwen fields used by LingBot's released deploy wrapper."""

    config_dict = qwen_config.to_dict()
    text_keys = {
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "max_position_embeddings",
        "hidden_act",
        "tie_word_embeddings",
        "tokenizer_path",
    }
    text_config = config_dict.get("text_config", {})
    for key in text_keys:
        if key in text_config:
            setattr(config, key, text_config[key])
        elif key in config_dict:
            setattr(config, key, config_dict[key])
    if "vision_config" not in config_dict:
        raise ValueError("pinned Qwen config contains no vision_config")
    config.vision_config = qwen_config.vision_config


class _RouteTrace:
    """Hash LingBot action-MoE top-k choices without changing its forward."""

    def __init__(self, torch: Any, blocks: list[Any]) -> None:
        self._torch = torch
        self._blocks = blocks
        self._digest = hashlib.sha256()
        self.calls = 0
        self.tokens = 0
        self._handles = [
            block.register_forward_pre_hook(self._hook(index, block))
            for index, block in enumerate(blocks)
        ]

    def _hook(self, layer_index: int, block: Any):
        torch = self._torch

        def capture(_module: Any, args: tuple[Any, ...]) -> None:
            hidden = args[0].detach()
            flat = hidden.reshape(-1, hidden.shape[-1])
            with torch.no_grad(), torch.amp.autocast(flat.device.type, enabled=False):
                logits = torch.nn.functional.linear(flat.float(), block.gate.weight.float())
                if block._router_activation == "sigmoid":
                    scores = logits.sigmoid()
                else:
                    scores = torch.nn.functional.softmax(logits, dim=1, dtype=torch.float32)
                scores = scores + block.e_score_correction_bias.unsqueeze(0)
                selected = torch.topk(scores, block.top_k, dim=-1).indices
            self._digest.update(f"{layer_index}:{tuple(hidden.shape)}:".encode())
            self._digest.update(selected.to(dtype=torch.int64, device="cpu").numpy().tobytes())
            self.calls += 1
            self.tokens += int(selected.shape[0])

        return capture

    def finish(self) -> dict[str, Any]:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        return {
            "sha256": self._digest.hexdigest(),
            "calls": self.calls,
            "tokens": self.tokens,
            "layers": len(self._blocks),
        }


def _git_output(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

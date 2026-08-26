"""Fail-closed official-forward and optimizer ownership for native training."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from numbers import Real
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer

from picf_next.lingbot_native.addresses import (
    EpisodeAddressState,
    deterministic_episode_permutation,
)
from picf_next.lingbot_native.calvin import (
    NativeCALVINRouting,
    audit_native_calvin_model_inputs,
)
from picf_next.lingbot_native.controls import (
    ExecutedControlBatch,
    executed_control_chain_reset,
)
from picf_next.lingbot_native.future_latent_alignment import (
    FutureLatentAlignmentResult,
    FutureLatentTargetBatch,
    installed_future_latent_alignment,
)
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativePriorStepper,
    native_context_from_persistent_state,
    native_context_from_prior_trace,
)
from picf_next.lingbot_native.modalities import (
    NativeModalityBatch,
    NativeModalityOmissionPlan,
)
from picf_next.lingbot_native.physical_relations import (
    NativeObjectQueryPosteriorOutput,
    PhysicalRelationOutput,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.row_binding import RowBindings
from picf_next.lingbot_native.source_mask import (
    QwenPackedPatchMask,
    QwenWholeViewOmission,
    qwen_source_masked_model_inputs,
    qwen_whole_view_omitted_model_inputs,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    AddressedLayerwisePriorTrace,
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePersistentState,
    NativePosteriorState,
    NativeVidEoMTPairedPosteriorState,
    clone_persistent_state,
    persistent_state_tensor,
    stack_persistent_states,
    unbind_persistent_state,
)
from picf_next.lingbot_native.temporal import (
    NativeLaneConfig,
    NativeLaneError,
    NativeLaneStamp,
    NativeLaneTransaction,
    NativeTrainingLaneBank,
)
from picf_next.lingbot_native.wsa_da3_loss import WSADA3TeacherTargets
from picf_next.lingbot_native.wsa_lingbot_install import (
    WSA_RELEASED_LAMBDA_3D,
    WSALingBotAttentionIntervention,
    WSALingBotForwardRole,
    wsa_lingbot_forward_kwargs,
)
from picf_next.lingbot_wla_calvin import WLACalvinTargetBatch

_LEGACY_RELATION_FIELDS = (
    "support_logits",
    "visible_support",
    "ownership",
    "task_relevance",
    "task_relevance_logits",
    "task_embedding",
    "match_embeddings",
    "row_embeddings",
    "relation_temperature",
    "dense_task_grounding",
    "dense_task_grounding_logits",
    "existence",
    "existence_logits",
    "sensor_valid",
)
_PHYSICAL_RELATION_FIELDS = (
    "support_logits",
    "visible_support",
    "ownership",
    "ownership_log_probability",
    "existence",
    "existence_logits",
    "row_embeddings",
    "relation_temperature",
    "sensor_valid",
)
_NATIVE_OBJECT_QUERY_RELATION_FIELDS = (
    "posterior_rows",
    "support_logits",
    "object_logits",
    "object_probability",
)
_KNOWN_RELATION_FIELDS = frozenset((*_LEGACY_RELATION_FIELDS, *_PHYSICAL_RELATION_FIELDS))


def native_persistent_output(context: LingBotNativeContext) -> NativePersistentState:
    """Return the architecture-owned recurrent state, never an observational readout."""

    if context.posterior_memory is not None:
        return context.posterior_memory
    if context.posterior_state is not None:
        return context.posterior_state
    raise RuntimeError("native context omitted its persistent output")


_OBSERVATION_MODEL_INPUTS = (
    "images",
    "img_masks",
    "lang_tokens",
    "lang_masks",
    "image_grid_thw",
)


def _policy_root(policy: nn.Module) -> nn.Module:
    root = policy
    seen: set[int] = set()
    while True:
        candidate = getattr(root, "module", None)
        if not isinstance(candidate, nn.Module):
            return root
        if id(root) in seen:
            raise ValueError("native policy wrappers contain a module cycle")
        seen.add(id(root))
        root = candidate


def _finite_scalar(name: str, value: Any, *, require_grad: bool) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() != 1 or not value.is_floating_point():
        raise RuntimeError(f"official LingBot {name} must be one floating scalar")
    if require_grad and not value.requires_grad:
        raise RuntimeError(f"official LingBot {name} is detached from the training graph")
    return value


def _detached_numeric_scalar(name: str, value: Any, *, reference: torch.Tensor) -> torch.Tensor:
    """Normalize released scalar metrics in wide precision without attaching them."""

    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or not value.is_floating_point():
            raise RuntimeError(f"official LingBot {name} must be one floating scalar")
        return value.detach().to(device=reference.device, dtype=torch.float64).reshape(())
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise RuntimeError(f"official LingBot {name} must be one finite real scalar")
    return torch.tensor(float(value), device=reference.device, dtype=torch.float64)


def _one_output_ulp(value: torch.Tensor) -> torch.Tensor:
    """Return one representable output interval in float64."""

    detached = value.detach()
    upper = torch.nextafter(detached, torch.full_like(detached, math.inf))
    lower = torch.nextafter(detached, torch.full_like(detached, -math.inf))
    upper_distance = (upper - detached).abs()
    lower_distance = (detached - lower).abs()
    upper_distance = torch.where(
        torch.isfinite(upper_distance),
        upper_distance,
        lower_distance,
    )
    lower_distance = torch.where(
        torch.isfinite(lower_distance),
        lower_distance,
        upper_distance,
    )
    return torch.maximum(upper_distance, lower_distance).to(torch.float64)


def _raise_first_failed_tensor_check(
    checks: tuple[tuple[str, torch.Tensor], ...],
) -> None:
    """Synchronize once on the valid path while preserving precise failures."""

    if not checks:
        return
    device = checks[0][1].device
    for _message, predicate in checks:
        if predicate.dtype != torch.bool or predicate.numel() != 1:
            raise TypeError("native tensor contract predicates must be boolean scalars")
        if predicate.device != device:
            raise RuntimeError("native tensor contract predicates changed device")
    packed = torch.stack(tuple(predicate.reshape(()) for _message, predicate in checks))
    if bool(packed.all().item()):
        return
    for message, predicate in checks:
        if not bool(predicate.item()):
            raise RuntimeError(message)
    raise RuntimeError("native tensor contract failed without an identifiable predicate")


def _targetless_official_loss_contract(
    outputs: tuple[Any, ...],
    *,
    total_loss: torch.Tensor,
    action_loss: torch.Tensor,
    auxiliary_objectives: tuple[tuple[str, torch.Tensor], ...] = (),
) -> tuple[torch.Tensor, tuple[tuple[str, torch.Tensor], ...]]:
    """Build one batched contract for the targetless released objective."""

    checks: list[tuple[str, torch.Tensor]] = [
        (
            "official LingBot total loss must be one finite non-negative floating scalar",
            torch.isfinite(total_loss).all() & total_loss.ge(0),
        ),
        (
            "official LingBot action loss must be one finite non-negative floating scalar",
            torch.isfinite(action_loss).all() & action_loss.ge(0),
        ),
    ]
    for index, name in ((2, "depth loss"), (3, "future depth loss"), (4, "video loss")):
        value = _detached_numeric_scalar(name, outputs[index], reference=total_loss)
        checks.append(
            (
                f"official LingBot {name} must be zero when teacher losses are off",
                torch.isfinite(value).all() & value.eq(0),
            )
        )
    metrics = outputs[6]
    if not isinstance(metrics, dict) or "router_z_loss" not in metrics:
        raise RuntimeError("released LingBot metrics omit router_z_loss")
    sequence_loss = _detached_numeric_scalar(
        "sequence-wise MoE loss",
        outputs[5],
        reference=total_loss,
    )
    router_loss = _detached_numeric_scalar(
        "router z loss",
        metrics["router_z_loss"],
        reference=total_loss,
    )
    checks.append(
        (
            "released LingBot sequence-wise MoE regularizer must be finite and non-negative",
            torch.isfinite(sequence_loss).all() & sequence_loss.ge(0),
        )
    )
    checks.append(
        (
            "released LingBot router z regularizer must be finite and non-negative",
            torch.isfinite(router_loss).all() & router_loss.ge(0),
        )
    )
    # The released graph may reassociate the three non-negative summands under
    # compilation. Two rounded additions can differ by one output ULP from the
    # exact wide-precision sum even though no objective term changed.
    auxiliary_sum = torch.zeros((), device=total_loss.device, dtype=torch.float64)
    for name, objective in auxiliary_objectives:
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(objective, torch.Tensor)
            or objective.numel() != 1
            or not objective.is_floating_point()
        ):
            raise TypeError("official auxiliary objectives must be named floating scalars")
        checks.append(
            (
                f"official {name} objective must be finite and non-negative",
                torch.isfinite(objective).all() & objective.ge(0),
            )
        )
        auxiliary_sum = auxiliary_sum + objective.detach().to(torch.float64).reshape(())
    expected = (
        action_loss.detach().to(torch.float64)
        + sequence_loss
        + router_loss
        + auxiliary_sum
    )
    reconstruction_error = (total_loss.detach().to(torch.float64) - expected).abs()
    checks.append(
        (
            "official LingBot targetless total differs from its explicit objective ledger "
            "by more than one output ULP",
            reconstruction_error <= _one_output_ulp(total_loss),
        )
    )
    regularizer = total_loss - action_loss
    for _name, objective in auxiliary_objectives:
        regularizer = regularizer - objective
    return regularizer, tuple(checks)


@dataclass(frozen=True, slots=True)
class OfficialPolicyForwardResult:
    """The unmodified 11-field LingBot action-policy output."""

    official_outputs: tuple[Any, ...]
    official_total_loss: torch.Tensor
    official_action_loss: torch.Tensor
    official_moe_regularizer: torch.Tensor


@dataclass(frozen=True, slots=True)
class NativePolicyForwardResult:
    """The released 11-field LingBot output plus finalized native state."""

    official_outputs: tuple[Any, ...]
    official_total_loss: torch.Tensor
    official_action_loss: torch.Tensor
    official_moe_regularizer: torch.Tensor
    context: LingBotNativeContext
    future_latent_alignment: FutureLatentAlignmentResult | None = None
    action_backend: str = "lingbot_released"
    backend_metrics: Mapping[str, torch.Tensor] | None = None


def run_official_policy_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
) -> OfficialPolicyForwardResult:
    """Train the released LingBot policy without installing or invoking PICF."""

    return _run_official_policy_forward(
        policy,
        model_inputs=model_inputs,
        require_grad=True,
    )


@torch.no_grad()
def run_official_policy_diagnostic_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
) -> OfficialPolicyForwardResult:
    """Evaluate the released targetless action root without a PICF context."""

    return _run_official_policy_forward(
        policy,
        model_inputs=model_inputs,
        require_grad=False,
    )


def _run_official_policy_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    require_grad: bool,
) -> OfficialPolicyForwardResult:
    if not isinstance(policy, nn.Module) or not policy.training:
        raise ValueError("official policy forward requires a policy in train mode")
    if not isinstance(require_grad, bool):
        raise TypeError("official policy gradient requirement must be boolean")
    audit_native_calvin_model_inputs(model_inputs, require_randomness=True)
    with torch.set_grad_enabled(require_grad):
        outputs = policy(
            **dict(model_inputs),
            compute_alignment_losses=False,
        )
    if not isinstance(outputs, tuple) or len(outputs) != 11:
        raise RuntimeError("released LingBot policy forward must return exactly 11 fields")
    total_loss = _finite_scalar("total loss", outputs[0], require_grad=require_grad)
    action_loss = _finite_scalar("action loss", outputs[1], require_grad=require_grad)
    moe_regularizer, checks = _targetless_official_loss_contract(
        outputs,
        total_loss=total_loss,
        action_loss=action_loss,
    )
    _raise_first_failed_tensor_check(checks)
    return OfficialPolicyForwardResult(
        official_outputs=outputs,
        official_total_loss=total_loss,
        official_action_loss=action_loss,
        official_moe_regularizer=moe_regularizer,
    )


def run_native_policy_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
    action_attention_callback: Callable[..., Any] | None = None,
    future_latent_target: FutureLatentTargetBatch | None = None,
    wsa_da3_teacher_targets: WSADA3TeacherTargets | None = None,
    wsa_forward_role: WSALingBotForwardRole = WSALingBotForwardRole.PRIMARY_FACTUAL,
    wla_world_target: WLACalvinTargetBatch | None = None,
) -> NativePolicyForwardResult:
    """Call the released policy and verify the exact ADR-74 write boundary."""

    return _run_native_policy_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
        require_official_grad=True,
        require_prediction_grad=True,
        required_relation_grad_fields=(),
        action_attention_callback=action_attention_callback,
        future_latent_target=future_latent_target,
        wsa_da3_teacher_targets=wsa_da3_teacher_targets,
        wsa_forward_role=wsa_forward_role,
        wla_world_target=wla_world_target,
    )


def run_native_policy_relation_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
) -> NativePolicyForwardResult:
    """Train relation ownership while observing a frozen official policy loss."""

    if context.prediction_request is not None:
        raise ValueError("relation training cannot construct predictive queries")

    return _run_native_policy_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
        require_official_grad=False,
        require_prediction_grad=False,
        required_relation_grad_fields=("ownership",),
        wsa_forward_role=WSALingBotForwardRole.MEASUREMENT_ONLY,
    )


def run_native_policy_representation_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
) -> LingBotNativeContext:
    """Train the shared observation host and PICF graph without an action suffix."""

    graph = _native_graph_for_policy(policy)
    required_relation_grad_fields = (
        ("ownership", "existence")
        if graph.task_independent
        else (
            "ownership",
            "task_relevance",
            "dense_task_grounding",
            "existence",
        )
    )
    return _run_native_observation_training_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
        require_prediction_grad=context.prediction_request is not None,
        required_relation_grad_fields=required_relation_grad_fields,
    )


@torch.no_grad()
def run_native_policy_diagnostic_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
    action_attention_callback: Callable[..., Any] | None = None,
    wsa_measurement_callback: Callable[[Any], None] | None = None,
    wsa_attention_intervention: WSALingBotAttentionIntervention | None = None,
) -> NativePolicyForwardResult:
    """Run the exact official host without retaining a training graph.

    This entrypoint exists only for sparse counterfactual audits.  It executes
    the same released policy and validates the same write boundary as the
    training forward; it does not expose a cheaper surrogate model.
    """

    return _run_native_policy_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
        require_official_grad=False,
        require_prediction_grad=False,
        required_relation_grad_fields=(),
        action_attention_callback=action_attention_callback,
        wsa_measurement_callback=wsa_measurement_callback,
        wsa_attention_intervention=wsa_attention_intervention,
        wsa_forward_role=WSALingBotForwardRole.MEASUREMENT_ONLY,
    )


@torch.no_grad()
def run_native_policy_observation_diagnostic_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
) -> LingBotNativeContext:
    """Run the exact observation-only host root without retaining a graph."""

    return _run_native_observation_training_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
        require_prediction_grad=False,
        required_relation_grad_fields=(),
    )


def _run_native_policy_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
    require_official_grad: bool,
    require_prediction_grad: bool,
    required_relation_grad_fields: tuple[str, ...],
    action_attention_callback: Callable[..., Any] | None = None,
    wsa_measurement_callback: Callable[[Any], None] | None = None,
    wsa_attention_intervention: WSALingBotAttentionIntervention | None = None,
    future_latent_target: FutureLatentTargetBatch | None = None,
    wsa_da3_teacher_targets: WSADA3TeacherTargets | None = None,
    wsa_forward_role: WSALingBotForwardRole | None = None,
    wla_world_target: WLACalvinTargetBatch | None = None,
) -> NativePolicyForwardResult:
    """Verify one official forward under the requested gradient contract."""

    if not isinstance(policy, nn.Module) or not policy.training:
        raise ValueError("native policy training forward requires a policy in train mode")
    if not isinstance(require_official_grad, bool) or not isinstance(
        require_prediction_grad,
        bool,
    ):
        raise TypeError("native policy forward gradient requirements must be boolean")
    if (
        not isinstance(required_relation_grad_fields, tuple)
        or any(
            not isinstance(name, str) or name not in _KNOWN_RELATION_FIELDS
            for name in required_relation_grad_fields
        )
        or len(set(required_relation_grad_fields)) != len(required_relation_grad_fields)
    ):
        raise ValueError("native relation gradient fields must be unique known field names")
    if not isinstance(context, LingBotNativeContext):
        raise TypeError("native policy training forward requires a LingBotNativeContext")
    if wla_world_target is not None and not isinstance(
        wla_world_target,
        WLACalvinTargetBatch,
    ):
        raise TypeError("native WLA training requires its typed CALVIN target")
    if any(
        value is not None
        for value in (context.native_roles, context.native_valid, context.instruction_last_index)
    ):
        raise ValueError("native policy training context was already bound by a host forward")
    resolved_wsa_role = (
        WSALingBotForwardRole.PRIMARY_FACTUAL
        if wsa_forward_role is None and require_official_grad
        else (
            WSALingBotForwardRole.MEASUREMENT_ONLY
            if wsa_forward_role is None
            else wsa_forward_role
        )
    )
    if not isinstance(resolved_wsa_role, WSALingBotForwardRole):
        raise TypeError("native policy forward requires a typed WSA role")
    if (
        wsa_measurement_callback is not None
        and resolved_wsa_role is not WSALingBotForwardRole.MEASUREMENT_ONLY
    ):
        raise ValueError("WSA measurement callback requires a measurement-only forward")
    if (
        wsa_attention_intervention is not None
        and resolved_wsa_role is not WSALingBotForwardRole.MEASUREMENT_ONLY
    ):
        raise ValueError("WSA attention intervention requires a measurement-only forward")
    if (
        resolved_wsa_role is WSALingBotForwardRole.MEASUREMENT_ONLY
        and wsa_da3_teacher_targets is not None
    ):
        raise ValueError("measurement-only native policy forward cannot consume DA3 targets")
    graph = _native_graph_for_policy(policy)
    wla_forward = getattr(policy, "picf_wla_calvin_forward", None)
    if wla_forward is not None:
        if not callable(wla_forward):
            raise RuntimeError("installed WLA root forward is not callable")
        if any(
            value is not None
            for value in (
                action_attention_callback,
                wsa_measurement_callback,
                wsa_attention_intervention,
                future_latent_target,
                wsa_da3_teacher_targets,
            )
        ):
            raise ValueError(
                "complete WLA action/world training is exclusive with legacy action, "
                "FLARE, and WSA callback objectives"
            )
        audit_native_calvin_model_inputs(model_inputs, require_randomness=True)
        _audit_root_forward_input_contract(model_inputs, context)
        with torch.set_grad_enabled(
            require_official_grad
            or require_prediction_grad
            or bool(required_relation_grad_fields)
        ):
            wla_output = wla_forward(
                model_inputs=dict(model_inputs),
                picf_native_context=context,
                target_images=(
                    None if wla_world_target is None else wla_world_target.images
                ),
                require_world=wla_world_target is not None,
            )
        total_loss = _finite_scalar(
            "WLA total loss",
            wla_output.total_loss,
            require_grad=require_official_grad,
        )
        action_loss = _finite_scalar(
            "WLA action loss",
            wla_output.action_loss,
            require_grad=require_official_grad,
        )
        world_loss = wla_output.world_loss
        if world_loss is not None:
            world_loss = _finite_scalar(
                "WLA world loss",
                world_loss,
                require_grad=require_official_grad,
            )
        checks = _validate_finalized_native_context(
            context=context,
            graph=graph,
            advertised_native_outputs=wla_output.native_root_outputs,
            root_output_dtype=_fsdp_root_output_dtype(policy),
            require_prediction_grad=require_prediction_grad,
            required_relation_grad_fields=required_relation_grad_fields,
        )
        _raise_first_failed_tensor_check(checks)
        zero = total_loss - total_loss
        return NativePolicyForwardResult(
            official_outputs=(),
            official_total_loss=total_loss,
            official_action_loss=action_loss,
            official_moe_regularizer=zero,
            context=context,
            action_backend="wla_complete",
            backend_metrics={
                "loss_action": action_loss,
                **({} if world_loss is None else {"loss_world": world_loss}),
            },
        )
    future_alignment = installed_future_latent_alignment(policy)
    future_context = None
    if future_latent_target is not None:
        if future_alignment is None:
            raise ValueError("FLARE target was supplied without an installed alignment module")
        future_context = future_alignment.new_forward_context(future_latent_target)
    elif future_alignment is not None and require_official_grad:
        raise ValueError("active FLARE action training requires a future target batch")
    audit_native_calvin_model_inputs(model_inputs, require_randomness=True)
    _audit_root_forward_input_contract(model_inputs, context)
    with torch.set_grad_enabled(
        require_official_grad or require_prediction_grad or bool(required_relation_grad_fields)
    ):
        policy_kwargs = {
            **dict(model_inputs),
            "picf_native_context": context,
            "compute_alignment_losses": False,
        }
        if action_attention_callback is not None:
            policy_kwargs["picf_action_attention_callback"] = action_attention_callback
        if future_context is not None:
            policy_kwargs["picf_future_latent_context"] = future_context
        policy_kwargs.update(
            wsa_lingbot_forward_kwargs(
                policy,
                role=resolved_wsa_role,
                teacher_targets=wsa_da3_teacher_targets,
                measurement_callback=wsa_measurement_callback,
                attention_intervention=wsa_attention_intervention,
            )
        )
        outputs = policy(**policy_kwargs)
    if not isinstance(outputs, tuple) or len(outputs) != 12:
        raise RuntimeError(
            "PICF LingBot training forward must return 11 official fields "
            "and one native root-output tuple"
        )
    official_outputs = outputs[:11]
    advertised_native_outputs = outputs[11]
    total_loss = _finite_scalar(
        "total loss",
        official_outputs[0],
        require_grad=require_official_grad,
    )
    action_loss = _finite_scalar(
        "action loss",
        official_outputs[1],
        require_grad=require_official_grad,
    )
    metrics = official_outputs[6]
    if not isinstance(metrics, dict):
        raise RuntimeError("released LingBot metrics must be a dictionary")
    wsa_checks: list[tuple[str, torch.Tensor]] = []
    auxiliary_objectives: tuple[tuple[str, torch.Tensor], ...] = ()
    wsa_objective = metrics.get("loss_future_3d_objective")
    if wsa_da3_teacher_targets is None:
        if wsa_objective is not None:
            raise RuntimeError("measurement-only WSA forward leaked a Future3D objective")
    else:
        wsa_objective = _finite_scalar(
            "WSA Future3D objective",
            wsa_objective,
            require_grad=require_official_grad,
        )
        logged_future = _detached_numeric_scalar(
            "WSA Future3D metric",
            metrics.get("loss_future_3d"),
            reference=total_loss,
        )
        logged_weighted = _detached_numeric_scalar(
            "weighted WSA Future3D metric",
            metrics.get("loss_future_3d_weighted"),
            reference=total_loss,
        )
        expected_future = wsa_objective.detach().to(torch.float64)
        expected_weighted = WSA_RELEASED_LAMBDA_3D * expected_future
        wsa_checks.extend(
            (
                (
                    "logged WSA Future3D loss differs from its objective tensor",
                    (logged_future - expected_future).abs()
                    <= _one_output_ulp(wsa_objective),
                ),
                (
                    "logged weighted WSA Future3D loss differs from released lambda_3d",
                    (logged_weighted - expected_weighted).abs()
                    <= _one_output_ulp(total_loss),
                ),
            )
        )
        auxiliary_objectives = (
            ("WSA Future3D", WSA_RELEASED_LAMBDA_3D * wsa_objective),
        )
    moe_regularizer, tensor_checks = _targetless_official_loss_contract(
        official_outputs,
        total_loss=total_loss,
        action_loss=action_loss,
        auxiliary_objectives=auxiliary_objectives,
    )
    checks = [
        *wsa_checks,
        *tensor_checks,
        *_validate_finalized_native_context(
            context=context,
            graph=graph,
            advertised_native_outputs=advertised_native_outputs,
            root_output_dtype=_fsdp_root_output_dtype(policy),
            require_prediction_grad=require_prediction_grad,
            required_relation_grad_fields=required_relation_grad_fields,
        ),
    ]
    _raise_first_failed_tensor_check(tuple(checks))
    future_result = None
    if future_context is not None:
        if future_alignment is None:
            raise AssertionError("validated FLARE context lost its alignment module")
        future_result = future_context.finalized_result(
            require_grad=require_official_grad,
        )
    return NativePolicyForwardResult(
        official_outputs=official_outputs,
        official_total_loss=total_loss,
        official_action_loss=action_loss,
        official_moe_regularizer=moe_regularizer,
        context=context,
        future_latent_alignment=future_result,
    )


def _native_graph_for_policy(policy: nn.Module) -> LingBotNativeGraph:
    root = _policy_root(policy)
    host = getattr(getattr(root, "model", None), "qwenvl_with_expert", None)
    graph = getattr(host, "picf_native_graph", None)
    if not isinstance(graph, LingBotNativeGraph):
        raise ValueError("the exact strict native graph is not installed in LingBot")
    return graph


def _fsdp_root_output_dtype(policy: nn.Module) -> torch.dtype | None:
    """Read the output cast declared by the actual FSDP2 root, if present."""

    get_state = getattr(policy, "_get_fsdp_state", None)
    if get_state is None:
        return None
    if not callable(get_state):
        raise RuntimeError("native policy exposes a malformed FSDP2 state accessor")
    state = get_state()
    mixed_precision = getattr(state, "_mp_policy", None)
    if mixed_precision is None:
        raise RuntimeError("native FSDP2 root omitted its mixed-precision policy")
    output_dtype = getattr(mixed_precision, "output_dtype", None)
    if output_dtype is not None and not isinstance(output_dtype, torch.dtype):
        raise RuntimeError("native FSDP2 root declares a malformed output dtype")
    return output_dtype


def _matches_declared_root_output_cast(
    advertised: Any,
    expected: torch.Tensor,
    *,
    output_dtype: torch.dtype | None,
) -> bool:
    """Accept only the exact cast performed by FSDP2 after hook registration."""

    if advertised is expected:
        return True
    if (
        output_dtype is None
        or not isinstance(advertised, torch.Tensor)
        or not advertised.is_floating_point()
        or expected.dtype == output_dtype
        or advertised.dtype != output_dtype
        or advertised.shape != expected.shape
        or advertised.stride() != expected.stride()
        or advertised.device != expected.device
        or advertised.layout != expected.layout
        or advertised.requires_grad != expected.requires_grad
        or (expected.requires_grad and advertised.is_leaf)
    ):
        return False
    return bool(torch.equal(advertised, expected.to(dtype=output_dtype)))


def _validate_finalized_native_context(
    *,
    context: LingBotNativeContext,
    graph: LingBotNativeGraph,
    advertised_native_outputs: Any,
    root_output_dtype: torch.dtype | None,
    require_prediction_grad: bool,
    required_relation_grad_fields: tuple[str, ...],
) -> tuple[tuple[str, torch.Tensor], ...]:
    if not isinstance(advertised_native_outputs, tuple):
        raise RuntimeError("PICF LingBot native root outputs must be a tuple")
    expected_native_outputs = context.root_output_tensors()
    identity_mismatches = tuple(
        index
        for index, (advertised, expected) in enumerate(
            zip(advertised_native_outputs, expected_native_outputs, strict=False)
        )
        if not _matches_declared_root_output_cast(
            advertised,
            expected,
            output_dtype=root_output_dtype,
        )
    )
    if len(advertised_native_outputs) != len(expected_native_outputs) or identity_mismatches:
        diagnostics: list[str] = []
        for index in identity_mismatches:
            advertised = advertised_native_outputs[index]
            expected = expected_native_outputs[index]
            if not isinstance(advertised, torch.Tensor) or not isinstance(expected, torch.Tensor):
                diagnostics.append(
                    f"{index}:types={type(advertised).__name__}/{type(expected).__name__}"
                )
                continue
            try:
                value_equal = bool(torch.equal(advertised, expected))
            except (RuntimeError, TypeError):
                value_equal = False
            try:
                storage_alias = (
                    advertised.untyped_storage().data_ptr() == expected.untyped_storage().data_ptr()
                    and advertised.storage_offset() == expected.storage_offset()
                )
            except (RuntimeError, TypeError):
                storage_alias = False
            diagnostics.append(
                f"{index}:equal={value_equal},alias={storage_alias},"
                f"shape={tuple(advertised.shape)}/{tuple(expected.shape)},"
                f"dtype={advertised.dtype}/{expected.dtype},"
                f"type={type(advertised).__name__}/{type(expected).__name__}"
            )
        raise RuntimeError(
            "PICF LingBot native root outputs differ from the finalized context; "
            f"lengths={len(advertised_native_outputs)}/{len(expected_native_outputs)}; "
            f"mismatches={';'.join(diagnostics)}"
        )
    checks: list[tuple[str, torch.Tensor]] = []
    if (
        context.prior_state is None
        or context.posterior_state is None
        or context.relation_output is None
        or not context._finalized
    ):
        raise RuntimeError("LingBot did not finalize every native training output")
    if graph.layerwise_recurrence != isinstance(
        context.posterior_memory,
        NativeLayerwisePosteriorState,
    ):
        raise RuntimeError("native persistent output differs from the graph state ABI")
    if (context.prediction_request is None) != (context.prediction_hidden is None):
        raise RuntimeError("native predictive hidden state differs from its request contract")
    expected_prediction_outputs = {
        name: width for name, width in graph.config.predictive_target_widths
    }
    if context.prediction_request is None:
        if context.prediction_outputs:
            raise RuntimeError("native forward emitted predictions without a request")
    else:
        hidden = context.prediction_hidden
        if hidden is None:
            raise RuntimeError("native prediction request did not produce hidden state")
        checks.append(
            (
                "native predictive hidden state contains NaN or infinity",
                torch.isfinite(hidden).all(),
            )
        )
        if set(context.prediction_outputs) != set(expected_prediction_outputs):
            raise RuntimeError("native predictive outputs differ from the graph target contract")
        for name, width in expected_prediction_outputs.items():
            prediction = context.prediction_outputs[name]
            expected_shape = (*hidden.shape[:-1], width)
            if prediction.shape != expected_shape:
                raise RuntimeError(f"native predictive output {name!r} has the wrong shape")
            if prediction.device != hidden.device or prediction.dtype != hidden.dtype:
                raise RuntimeError(f"native predictive output {name!r} changed device or dtype")
            if require_prediction_grad and not prediction.requires_grad:
                raise RuntimeError(f"native predictive output {name!r} is detached")
            checks.append(
                (
                    f"native predictive output {name!r} contains NaN or infinity",
                    torch.isfinite(prediction).all(),
                )
            )
    relation_output = context.relation_output
    native_object_query_relation = isinstance(
        relation_output,
        NativeObjectQueryPosteriorOutput,
    )
    if native_object_query_relation:
        relation_fields = _NATIVE_OBJECT_QUERY_RELATION_FIELDS
        if required_relation_grad_fields:
            raise RuntimeError(
                "native object-query posterior forbids legacy relation-gradient fields"
            )
    elif isinstance(relation_output, PhysicalRelationOutput):
        relation_fields = _PHYSICAL_RELATION_FIELDS
    elif isinstance(relation_output, RelationOutput):
        relation_fields = _LEGACY_RELATION_FIELDS
    else:
        raise TypeError("native relation output has an unknown architecture interface")
    undeclared_required = set(required_relation_grad_fields) - set(relation_fields)
    if undeclared_required:
        raise RuntimeError(
            "required native relation fields cross architecture interfaces: "
            f"{sorted(undeclared_required)}"
        )
    for name in relation_fields:
        value = getattr(context.relation_output, name)
        if value is None:
            if name in required_relation_grad_fields:
                raise RuntimeError(f"required native relation output {name!r} is absent")
            continue
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"native relation output {name!r} is not a tensor")
        if name in required_relation_grad_fields and not value.requires_grad:
            raise RuntimeError(f"native relation output {name!r} is detached")
        if value.is_floating_point():
            checks.append(
                (
                    f"native relation output {name} contains NaN or infinity",
                    torch.isfinite(value).all(),
                )
            )
    expected_intermediate_layers = (
        ()
        if native_object_query_relation
        else (
            graph.config.relation_supervision_layers
            if context.supervise_intermediate_relations
            else ()
        )
    )
    if tuple(context.intermediate_relation_outputs) != expected_intermediate_layers:
        raise RuntimeError("native intermediate relation outputs differ from the graph contract")
    require_intermediate_grad = require_prediction_grad or bool(required_relation_grad_fields)
    for layer, relation in context.intermediate_relation_outputs.items():
        if native_object_query_relation:
            raise RuntimeError(
                "native object-query posterior forbids a second intermediate relation readout"
            )
        if not isinstance(relation, type(relation_output)):
            raise TypeError(f"native intermediate relation at layer {layer} has the wrong type")
        if relation.ownership.shape != context.relation_output.ownership.shape:
            raise RuntimeError(
                f"native intermediate ownership at layer {layer} has the wrong shape"
            )
        for name in relation_fields:
            value = getattr(relation, name)
            if value is None:
                continue
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"native intermediate relation output {name!r} is not a tensor")
            if require_intermediate_grad and name == "ownership" and not value.requires_grad:
                raise RuntimeError(f"native intermediate ownership at layer {layer} is detached")
            if value.is_floating_point():
                checks.append(
                    (
                        f"native intermediate relation {name} at layer {layer} "
                        "contains NaN or infinity",
                        torch.isfinite(value).all(),
                    )
                )
    return tuple(checks)


def _audit_root_forward_input_contract(
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
) -> None:
    """Require explicit tensor typing before the FSDP2 root sees a mutable context."""

    reference = context.controls.values
    if not reference.is_floating_point():
        raise TypeError("native root control values must be floating point")
    device = reference.device
    dtype = reference.dtype
    if context.controls.delta_time.dtype != dtype:
        raise TypeError("native root control timestamps differ from the compute dtype")
    for name, value in model_inputs.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.device != device:
            raise ValueError(f"native root input {name!r} differs from the context device")
        if value.is_floating_point() and value.dtype != dtype:
            raise TypeError(f"native root floating input {name!r} differs from the compute dtype")
    if context.previous_state is not None and (
        context.previous_state.rows.device != device or context.previous_state.rows.dtype != dtype
    ):
        raise TypeError("native root previous posterior differs from the compute tensor contract")
    if context.previous_memory is not None and (
        context.previous_memory.layer_rows.device != device
        or context.previous_memory.layer_rows.dtype != dtype
    ):
        raise TypeError("native root previous layerwise memory differs from the compute contract")
    if context.prior_trace is not None and (
        context.prior_trace.layer_rows.device != device
        or context.prior_trace.layer_rows.dtype != dtype
    ):
        raise TypeError("native root prior trace differs from the compute tensor contract")
    if context.modalities is not None and (
        context.modalities.device != device or context.modalities.dtype != dtype
    ):
        raise TypeError("native root modality batch differs from the compute tensor contract")
    if context.prediction_request is not None and (
        context.prediction_request.addresses.device != device
        or context.prediction_request.addresses.dtype != dtype
    ):
        raise TypeError("native root prediction addresses differ from the compute tensor contract")


def _run_native_observation_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    context: LingBotNativeContext,
    require_prediction_grad: bool = True,
    required_relation_grad_fields: tuple[str, ...] = (),
) -> LingBotNativeContext:
    """Run the released observation prefix without constructing an action graph."""

    if not isinstance(policy, nn.Module) or not policy.training:
        raise ValueError("native observation forward requires a policy in train mode")
    if not isinstance(context, LingBotNativeContext):
        raise TypeError("native observation forward requires a LingBotNativeContext")
    if any(
        value is not None
        for value in (context.native_roles, context.native_valid, context.instruction_last_index)
    ):
        raise ValueError("native observation context was already bound by a host forward")
    graph = _native_graph_for_policy(policy)
    audit_native_calvin_model_inputs(model_inputs, require_randomness=True)
    _audit_root_forward_input_contract(model_inputs, context)
    missing = tuple(name for name in _OBSERVATION_MODEL_INPUTS if name not in model_inputs)
    if missing:
        raise KeyError(f"official LingBot observation inputs are incomplete: {missing}")
    root_forward = getattr(policy, "picf_native_observation_forward", None)
    if not callable(root_forward):
        raise TypeError("LingBot policy lacks the audited observation-only root method")
    advertised_native_outputs = root_forward(
        **{name: model_inputs[name] for name in _OBSERVATION_MODEL_INPUTS},
        picf_native_context=context,
    )
    checks = _validate_finalized_native_context(
        context=context,
        graph=graph,
        advertised_native_outputs=advertised_native_outputs,
        root_output_dtype=_fsdp_root_output_dtype(policy),
        require_prediction_grad=require_prediction_grad,
        required_relation_grad_fields=required_relation_grad_fields,
    )
    _raise_first_failed_tensor_check(checks)
    return context


@dataclass(frozen=True, slots=True)
class NativeSourceMaskedPrediction:
    """Only the legal training query result; no committable state is exposed."""

    prediction_hidden: torch.Tensor
    prediction_outputs: Mapping[str, torch.Tensor]
    source_mask_digest: str


@dataclass(frozen=True, slots=True)
class NativeOmittedModalityPrediction:
    """Training-only output whose complete target modality was absent upstream."""

    omitted_name: str
    prediction_hidden: torch.Tensor
    prediction_outputs: Mapping[str, torch.Tensor]
    omission_digest: str


def run_native_source_masked_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    prediction_request: NativePredictionRequest,
    source_mask: QwenPackedPatchMask,
    modalities: NativeModalityBatch | None = None,
) -> NativeSourceMaskedPrediction:
    """Run the weight-shared current-grid branch without exposing its rows."""

    if not isinstance(prediction_request, NativePredictionRequest):
        raise TypeError("source-masked prediction requires a NativePredictionRequest")
    if (
        prediction_request.source is not PredictionSource.POSTERIOR
        or prediction_request.evidence is not PredictionEvidence.CURRENT_RANDOM_GRID
    ):
        raise ValueError("Qwen source masking is only the current-grid posterior estimator")
    if not isinstance(source_mask, QwenPackedPatchMask):
        raise TypeError("source-masked prediction requires a QwenPackedPatchMask")
    images = model_inputs.get("images")
    image_valid = model_inputs.get("img_masks")
    image_grid = model_inputs.get("image_grid_thw")
    if not isinstance(images, torch.Tensor):
        raise TypeError("official LingBot packed images must be a tensor")
    if not isinstance(image_valid, torch.Tensor):
        raise TypeError("official LingBot image validity must be a tensor")
    if not isinstance(image_grid, torch.Tensor):
        raise TypeError("official LingBot packed image fields must be tensors")
    if not torch.equal(image_valid, source_mask.image_valid):
        raise ValueError("source-mask image availability differs from official LingBot inputs")
    if not torch.equal(image_grid, source_mask.image_grid_thw):
        raise ValueError("source-mask Qwen grid differs from official LingBot inputs")
    masked_inputs = qwen_source_masked_model_inputs(model_inputs, source_mask)
    context = native_context_from_persistent_state(
        controls=controls,
        persistent_state=previous_state,
        persistent_state_valid=previous_state_valid,
        prediction_request=prediction_request,
        modalities=modalities,
    )
    context = _run_native_observation_training_forward(
        policy,
        model_inputs=masked_inputs,
        context=context,
    )
    hidden = context.prediction_hidden
    if hidden is None or not hidden.requires_grad:
        raise RuntimeError("source-masked prediction hidden is invalid or detached")
    return NativeSourceMaskedPrediction(
        prediction_hidden=hidden,
        prediction_outputs=context.prediction_outputs,
        source_mask_digest=source_mask.digest,
    )


def run_native_omitted_modality_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    prediction_request: NativePredictionRequest,
    modalities: NativeModalityBatch,
    omission: NativeModalityOmissionPlan,
) -> NativeOmittedModalityPrediction:
    """Run one whole-modality JEPA branch without exposing committable rows."""

    if not isinstance(prediction_request, NativePredictionRequest):
        raise TypeError("omitted-modality prediction requires a NativePredictionRequest")
    if (
        prediction_request.source is not PredictionSource.POSTERIOR
        or prediction_request.evidence is not PredictionEvidence.OMITTED_MODALITY
    ):
        raise ValueError("whole-modality omission requires posterior omitted-modality evidence")
    if not isinstance(modalities, NativeModalityBatch) or not isinstance(
        omission, NativeModalityOmissionPlan
    ):
        raise TypeError("whole-modality prediction requires typed source and omission contracts")
    expected_valid = omission.source_valid[:, None].expand_as(prediction_request.valid)
    if not torch.equal(prediction_request.valid, expected_valid):
        raise ValueError("omitted-modality query validity differs from source availability")
    source_modalities = omission.apply(modalities)
    context = native_context_from_persistent_state(
        controls=controls,
        persistent_state=previous_state,
        persistent_state_valid=previous_state_valid,
        prediction_request=prediction_request,
        modalities=source_modalities,
    )
    context = _run_native_observation_training_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
    )
    hidden = context.prediction_hidden
    if hidden is None or not hidden.requires_grad:
        raise RuntimeError("omitted-modality prediction hidden is invalid or detached")
    return NativeOmittedModalityPrediction(
        omitted_name=omission.omitted_name,
        prediction_hidden=hidden,
        prediction_outputs=context.prediction_outputs,
        omission_digest=omission.digest,
    )


def run_native_omitted_image_view_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    prediction_request: NativePredictionRequest,
    omission: QwenWholeViewOmission,
    modalities: NativeModalityBatch | None = None,
) -> NativeOmittedModalityPrediction:
    """Run one official missing-image branch through the weight-shared host."""

    if not isinstance(prediction_request, NativePredictionRequest):
        raise TypeError("omitted-image prediction requires a NativePredictionRequest")
    if (
        prediction_request.source is not PredictionSource.POSTERIOR
        or prediction_request.evidence is not PredictionEvidence.OMITTED_MODALITY
        or (prediction_request.addresses != 0).any()
    ):
        raise ValueError("omitted-image prediction requires a zero-address posterior query")
    if not isinstance(omission, QwenWholeViewOmission):
        raise TypeError("omitted-image prediction requires a QwenWholeViewOmission")
    expected_valid = omission.source_valid[:, None].expand_as(prediction_request.valid)
    if not torch.equal(prediction_request.valid, expected_valid):
        raise ValueError("omitted-image query validity differs from source availability")
    source_inputs = qwen_whole_view_omitted_model_inputs(model_inputs, omission)
    context = native_context_from_persistent_state(
        controls=controls,
        persistent_state=previous_state,
        persistent_state_valid=previous_state_valid,
        prediction_request=prediction_request,
        modalities=modalities,
    )
    context = _run_native_observation_training_forward(
        policy,
        model_inputs=source_inputs,
        context=context,
    )
    hidden = context.prediction_hidden
    if hidden is None or not hidden.requires_grad:
        raise RuntimeError("omitted-image prediction hidden is invalid or detached")
    return NativeOmittedModalityPrediction(
        omitted_name=omission.omitted_name,
        prediction_hidden=hidden,
        prediction_outputs=context.prediction_outputs,
        omission_digest=omission.digest,
    )


@dataclass(frozen=True, slots=True)
class NativeLocalBPTTStep:
    """One source-clean official host step in a sampled local window."""

    model_inputs: Mapping[str, Any]
    controls: ExecutedControlBatch
    prediction_request: NativePredictionRequest | None = None
    modalities: NativeModalityBatch | None = None
    future_latent_target: FutureLatentTargetBatch | None = None
    wsa_da3_teacher_targets: WSADA3TeacherTargets | None = None
    wla_world_target: WLACalvinTargetBatch | None = None


@dataclass(frozen=True, slots=True)
class NativeLocalBPTTAuxiliary:
    """Observation-derived loss surfaces without a state that could reach a lane."""

    relation_output: RelationOutput | PhysicalRelationOutput
    intermediate_relation_outputs: Mapping[int, RelationOutput | PhysicalRelationOutput]
    prediction_hidden: torch.Tensor | None
    prediction_outputs: Mapping[str, torch.Tensor]


@dataclass(frozen=True, slots=True)
class NativeLocalBPTTResult:
    """The current frame is committable; later local frames are loss-only."""

    primary: NativePolicyForwardResult
    auxiliary: tuple[NativeLocalBPTTAuxiliary, ...]


@dataclass(frozen=True, slots=True)
class NativeV3FilterPredictionSpec:
    """Paired current-time queries for one shared v3 predictive readout."""

    prior_request: NativePredictionRequest
    posterior_request: NativePredictionRequest
    target_name: str

    def __post_init__(self) -> None:
        if not isinstance(self.prior_request, NativePredictionRequest) or not isinstance(
            self.posterior_request,
            NativePredictionRequest,
        ):
            raise TypeError("v3 filter prediction requires two typed requests")
        if (
            self.prior_request.source is not PredictionSource.PRIOR
            or self.prior_request.evidence is not PredictionEvidence.CURRENT_PRIOR
        ):
            raise ValueError("v3 Pass A requires PRIOR/CURRENT_PRIOR evidence")
        if (
            self.posterior_request.source is not PredictionSource.POSTERIOR
            or self.posterior_request.evidence is not PredictionEvidence.CURRENT_POSTERIOR
        ):
            raise ValueError("v3 Pass B requires POSTERIOR/CURRENT_POSTERIOR evidence")
        if not isinstance(self.target_name, str) or not self.target_name:
            raise ValueError("v3 filter predictive target name must be non-empty")
        for name in ("route_ids", "horizons", "addresses"):
            prior_value = getattr(self.prior_request, name)
            posterior_value = getattr(self.posterior_request, name)
            if (
                prior_value.shape != posterior_value.shape
                or prior_value.dtype != posterior_value.dtype
                or prior_value.device != posterior_value.device
                or not torch.equal(prior_value, posterior_value)
            ):
                raise ValueError(f"v3 prior/posterior filter requests differ in {name}")


@dataclass(frozen=True, slots=True)
class NativeV3FilterPredictions:
    """Attached prior/posterior coordinates emitted by one two-pass frame."""

    spec: NativeV3FilterPredictionSpec
    prior: torch.Tensor
    posterior: torch.Tensor

    def __post_init__(self) -> None:
        expected = (
            self.spec.prior_request.batch_size,
            self.prior.shape[1],
            self.spec.prior_request.query_count,
        )
        if (
            self.prior.ndim != 4
            or self.posterior.ndim != 4
            or self.prior.shape != self.posterior.shape
            or self.prior.shape[:3] != expected
        ):
            raise ValueError("v3 filter predictions differ from their paired request axes")
        if not self.prior.requires_grad or not self.posterior.requires_grad:
            raise RuntimeError("v3 filter predictions must remain attached")
        if not torch.isfinite(self.prior).all() or not torch.isfinite(self.posterior).all():
            raise RuntimeError("v3 filter predictions contain NaN or infinity")


@dataclass(frozen=True, slots=True)
class NativeV3TwoPassStep:
    """One explicit prior-then-correction frame in a v3 probe window."""

    model_inputs: Mapping[str, Any]
    controls: ExecutedControlBatch
    filter_prediction: NativeV3FilterPredictionSpec | None = None
    modalities: NativeModalityBatch | None = None
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = ()
    prior_host_steps: int | None = None
    prior_gradient_suffix_steps: int | None = None
    future_latent_target: FutureLatentTargetBatch | None = None
    wsa_da3_teacher_targets: WSADA3TeacherTargets | None = None
    wla_world_target: WLACalvinTargetBatch | None = None

    def __post_init__(self) -> None:
        if self.prior_control_chunks:
            if self.prior_control_chunks[-1] is not self.controls:
                raise ValueError("final v3 prior-control chunk must be correction controls")
            if any(
                chunk.batch_size != self.controls.batch_size
                or chunk.action_dim != self.controls.action_dim
                or chunk.values.device != self.controls.values.device
                or chunk.values.dtype != self.controls.values.dtype
                for chunk in self.prior_control_chunks
            ):
                raise ValueError("v3 prior-control chunks differ from correction controls")
        if self.prior_host_steps is not None and (
            isinstance(self.prior_host_steps, bool)
            or not isinstance(self.prior_host_steps, int)
            or self.prior_host_steps < len(self.effective_prior_control_chunks)
        ):
            raise ValueError("v3 prior host steps cannot be shorter than the control chain")
        resolved_host_steps = (
            len(self.effective_prior_control_chunks)
            if self.prior_host_steps is None
            else self.prior_host_steps
        )
        if self.prior_gradient_suffix_steps is not None and (
            isinstance(self.prior_gradient_suffix_steps, bool)
            or not isinstance(self.prior_gradient_suffix_steps, int)
            or not 1 <= self.prior_gradient_suffix_steps <= resolved_host_steps
        ):
            raise ValueError(
                "v3 prior gradient suffix must be positive and no longer than the host schedule"
            )

    @property
    def effective_prior_control_chunks(self) -> tuple[ExecutedControlBatch, ...]:
        return self.prior_control_chunks or (self.controls,)


@dataclass(frozen=True, slots=True)
class NativeV3TwoPassForwardResult:
    """One factual v3 prior pass followed by the complete official policy pass."""

    prior_trace: NativeLayerwisePriorTrace
    policy_forward: NativePolicyForwardResult
    filter_predictions: NativeV3FilterPredictions | None = None

    @property
    def committable_context(self) -> LingBotNativeContext:
        return self.policy_forward.context


@dataclass(frozen=True, slots=True)
class NativeV3TwoPassSequenceResult:
    """A 1..4 frame probe whose factual first posterior is the only commit surface."""

    primary: NativePolicyForwardResult
    auxiliary: tuple[NativeLocalBPTTAuxiliary, ...]
    prior_traces: tuple[NativeLayerwisePriorTrace, ...]
    filter_predictions: tuple[NativeV3FilterPredictions | None, ...]

    @property
    def committable_context(self) -> LingBotNativeContext:
        return self.primary.context


@dataclass(frozen=True, slots=True)
class NativeV3AttachedEgressResult:
    """One low-token next-prior result with no posterior or lane commit surface."""

    prior_trace: NativeLayerwisePriorTrace
    request: NativePredictionRequest
    target_name: str
    prediction: torch.Tensor


@dataclass(frozen=True, slots=True)
class NativeV3OmittedStaticPolicyResult:
    """Complete omitted-view policy output stripped of every committable state."""

    official_outputs: tuple[Any, ...]
    official_total_loss: torch.Tensor
    official_action_loss: torch.Tensor
    official_moe_regularizer: torch.Tensor
    request: NativePredictionRequest
    omitted_name: str
    prediction_hidden: torch.Tensor
    prediction_outputs: Mapping[str, torch.Tensor]
    omission_digest: str


def native_policy_uses_v3_two_pass(policy: nn.Module) -> bool:
    """Return whether the exact graph installed on this policy is the v3 ABI."""

    return _native_graph_for_policy(policy).unified_predict_correct


def _v3_graph_for_policy(policy: nn.Module) -> LingBotNativeGraph:
    graph = _native_graph_for_policy(policy)
    if not graph.unified_predict_correct:
        raise ValueError("v3 training orchestration requires the unified predict-correct graph")
    return graph


def _validate_v3_prior_source(
    *,
    previous_memory: NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None,
    previous_memory_valid: torch.Tensor,
    controls: ExecutedControlBatch,
    require_attached: bool,
) -> None:
    if previous_memory is not None and not isinstance(
        previous_memory,
        NativeLayerwisePosteriorState | NativeLayerwisePriorTrace,
    ):
        raise TypeError("v3 prior input must use a typed layerwise source or None")
    if not isinstance(previous_memory_valid, torch.Tensor):
        raise TypeError("v3 prior input requires explicit tensor validity")
    if (
        previous_memory_valid.shape != (controls.batch_size,)
        or previous_memory_valid.dtype != torch.bool
        or previous_memory_valid.device != controls.values.device
    ):
        raise ValueError("v3 prior validity must be boolean [batch] on the controls device")
    if previous_memory is None and bool(previous_memory_valid.any().item()):
        raise ValueError("v3 prior validity cannot select absent layerwise memory")
    reset = (controls.reset & controls.token_valid).any(dim=1)
    if bool((reset & previous_memory_valid).any().item()):
        raise ValueError("v3 episode reset requires explicitly invalid previous memory")
    if (
        require_attached
        and previous_memory is not None
        and not previous_memory.layer_rows.requires_grad
    ):
        raise RuntimeError("v3 attached egress received detached posterior memory")


def _run_native_v3_prior_pass(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    previous_memory: NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None,
    previous_memory_valid: torch.Tensor,
    controls: ExecutedControlBatch,
    filter_prediction: NativeV3FilterPredictionSpec | None,
    require_attached_memory: bool,
    require_grad: bool,
) -> tuple[NativeLayerwisePriorTrace, torch.Tensor | None]:
    _validate_v3_prior_source(
        previous_memory=previous_memory,
        previous_memory_valid=previous_memory_valid,
        controls=controls,
        require_attached=require_attached_memory,
    )
    stepper = LingBotNativePriorStepper(policy, graph)
    episode_address_state = (
        previous_memory.episode_address_state
        if isinstance(
            previous_memory,
            AddressedLayerwisePosteriorState | AddressedLayerwisePriorTrace,
        )
        else None
    )
    prediction = None
    if filter_prediction is None:
        value = stepper(
            previous_memory,
            controls,
            previous_memory_valid=previous_memory_valid,
            episode_address_state=episode_address_state,
        )
    else:
        if filter_prediction.target_name not in dict(graph.config.predictive_target_widths):
            raise ValueError("v3 filter target is absent from the installed graph")
        value, prediction = stepper.step_with_prediction(
            previous_memory,
            controls,
            filter_prediction.prior_request,
            target_name=filter_prediction.target_name,
            previous_memory_valid=previous_memory_valid,
            episode_address_state=episode_address_state,
        )
    if not isinstance(value, NativeLayerwisePriorTrace):
        raise RuntimeError("v3 prior pass did not return a typed layerwise trace")
    if require_grad and not value.layer_rows.requires_grad:
        raise RuntimeError("v3 prior trace detached before correction")
    if require_grad and prediction is not None and not prediction.requires_grad:
        raise RuntimeError("v3 prior prediction detached from the shared readout")
    return value, prediction


def _invalid_v3_control_padding(reference: ExecutedControlBatch) -> ExecutedControlBatch:
    shape = (reference.batch_size, 1, reference.action_dim)
    return ExecutedControlBatch(
        values=torch.zeros(shape, dtype=reference.values.dtype, device=reference.values.device),
        field_valid=torch.zeros(shape, dtype=torch.bool, device=reference.values.device),
        token_valid=torch.zeros(
            reference.batch_size,
            1,
            dtype=torch.bool,
            device=reference.values.device,
        ),
        delta_time=torch.zeros(
            reference.batch_size,
            1,
            dtype=reference.delta_time.dtype,
            device=reference.values.device,
        ),
        reset=torch.zeros(
            reference.batch_size,
            1,
            dtype=torch.bool,
            device=reference.values.device,
        ),
        acknowledged=torch.zeros(
            reference.batch_size,
            1,
            dtype=torch.bool,
            device=reference.values.device,
        ),
    )


def _identity_carry_v3_prior_source(
    source: NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None,
    candidate: NativeLayerwisePriorTrace,
) -> NativeLayerwisePriorTrace:
    """Carry source values exactly while retaining a zero-gradient host dependency."""

    if source is None:
        source_rows = torch.zeros_like(candidate.layer_rows)
    else:
        source_rows = source.layer_rows
        if source_rows.shape != candidate.layer_rows.shape:
            raise ValueError("v3 padding candidate differs from its recurrent source")
    condition = torch.zeros((), dtype=torch.bool, device=candidate.layer_rows.device)
    layer_rows = torch.where(condition, candidate.layer_rows, source_rows)
    if isinstance(candidate, AddressedLayerwisePriorTrace):
        if isinstance(
            source,
            AddressedLayerwisePosteriorState | AddressedLayerwisePriorTrace,
        ) and not source.episode_address_state.same_assignment(candidate.episode_address_state):
            raise ValueError("v3 padding changed the episode address assignment")
        return AddressedLayerwisePriorTrace(
            layer_rows=layer_rows,
            episode_address_state=candidate.episode_address_state,
            architecture_identity=candidate.architecture_identity,
        )
    if isinstance(
        source,
        AddressedLayerwisePosteriorState | AddressedLayerwisePriorTrace,
    ):
        raise RuntimeError("v3 padding dropped the addressed prior receipt")
    return NativeLayerwisePriorTrace(layer_rows)


def run_native_v3_prior_chain(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    previous_memory: NativeLayerwisePosteriorState | None,
    previous_memory_valid: torch.Tensor,
    control_chunks: tuple[ExecutedControlBatch, ...],
    filter_prediction: NativeV3FilterPredictionSpec | None,
    require_attached_memory: bool,
    host_step_count: int | None = None,
    gradient_suffix_steps: int | None = None,
    require_grad: bool = True,
) -> tuple[NativeLayerwisePriorTrace, torch.Tensor | None]:
    """Advance an exact control span under one static distributed host schedule.

    An optional trailing gradient suffix implements standard recurrent burn-in:
    every earlier control still advances the shared host state, but only the
    final host calls retain autograd state. This bounds memory without dropping,
    pooling, or reordering any executed control.
    """

    if not control_chunks or any(
        not isinstance(chunk, ExecutedControlBatch) for chunk in control_chunks
    ):
        raise ValueError("v3 prior chain requires one or more typed control chunks")
    reference = control_chunks[0]
    if any(
        chunk.batch_size != reference.batch_size
        or chunk.action_dim != reference.action_dim
        or chunk.values.device != reference.values.device
        or chunk.values.dtype != reference.values.dtype
        for chunk in control_chunks
    ):
        raise ValueError("v3 prior chain controls must share one batch/device/chart")

    resolved_host_steps = len(control_chunks) if host_step_count is None else host_step_count
    if (
        isinstance(resolved_host_steps, bool)
        or not isinstance(resolved_host_steps, int)
        or resolved_host_steps < len(control_chunks)
    ):
        raise ValueError("v3 prior host schedule cannot be shorter than the control chain")
    if not isinstance(require_grad, bool):
        raise TypeError("v3 prior gradient requirement must be boolean")
    resolved_gradient_suffix = (
        resolved_host_steps if gradient_suffix_steps is None else gradient_suffix_steps
    )
    if (
        isinstance(resolved_gradient_suffix, bool)
        or not isinstance(resolved_gradient_suffix, int)
        or not 1 <= resolved_gradient_suffix <= resolved_host_steps
    ):
        raise ValueError(
            "v3 prior gradient suffix must be positive and no longer than the host schedule"
        )
    if require_attached_memory and resolved_gradient_suffix != resolved_host_steps:
        raise ValueError("attached v3 egress cannot truncate its recurrent gradient path")
    if require_attached_memory and (
        previous_memory is None or not previous_memory.layer_rows.requires_grad
    ):
        raise RuntimeError("v3 attached egress received detached posterior memory")

    source: NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None = previous_memory
    source_valid = previous_memory_valid
    prediction = None
    padding = _invalid_v3_control_padding(reference)
    padding_steps = resolved_host_steps - len(control_chunks)
    rng_devices = (
        []
        if reference.values.device.type != "cuda"
        else [reference.values.device.index or torch.cuda.current_device()]
    )
    first_gradient_host_step = resolved_host_steps - resolved_gradient_suffix
    for _index in range(padding_steps):
        attached = require_grad and _index >= first_gradient_host_step
        with torch.set_grad_enabled(attached), torch.random.fork_rng(devices=rng_devices):
            candidate, _ = _run_native_v3_prior_pass(
                policy,
                graph=graph,
                previous_memory=source,
                previous_memory_valid=source_valid,
                controls=padding,
                filter_prediction=None,
                require_attached_memory=False,
                require_grad=attached,
            )
        source = _identity_carry_v3_prior_source(source, candidate)
    for index, controls in enumerate(control_chunks):
        final = index == len(control_chunks) - 1
        host_index = padding_steps + index
        attached = require_grad and host_index >= first_gradient_host_step
        with torch.set_grad_enabled(attached):
            source, prediction = _run_native_v3_prior_pass(
                policy,
                graph=graph,
                previous_memory=source,
                previous_memory_valid=source_valid,
                controls=controls,
                filter_prediction=filter_prediction if final else None,
                require_attached_memory=attached
                and (
                    require_attached_memory
                    or host_index > first_gradient_host_step
                ),
                require_grad=attached,
            )
        source_valid = torch.ones(
            controls.batch_size,
            dtype=torch.bool,
            device=controls.values.device,
        )
    if not isinstance(source, NativeLayerwisePriorTrace):
        raise RuntimeError("v3 prior chain did not finish on a transient prior trace")
    return source, prediction


def run_native_v3_two_pass_policy_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_memory: NativeLayerwisePosteriorState | None,
    previous_memory_valid: torch.Tensor,
    filter_prediction: NativeV3FilterPredictionSpec | None = None,
    modalities: NativeModalityBatch | None = None,
    supervise_intermediate_relations: bool = False,
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = (),
    prior_host_steps: int | None = None,
    prior_gradient_suffix_steps: int | None = None,
    posterior_adoption_route: torch.Tensor | None = None,
    action_attention_callback: Callable[..., Any] | None = None,
    future_latent_target: FutureLatentTargetBatch | None = None,
    wsa_da3_teacher_targets: WSADA3TeacherTargets | None = None,
    wla_world_target: WLACalvinTargetBatch | None = None,
) -> NativeV3TwoPassForwardResult:
    """Run v3 Pass A then complete factual correction/action through one policy."""

    graph = _v3_graph_for_policy(policy)
    control_chunks = prior_control_chunks or (controls,)
    if control_chunks[-1] is not controls:
        raise ValueError("final v3 prior-control chunk must be correction controls")
    prior_trace, prior_prediction = run_native_v3_prior_chain(
        policy,
        graph=graph,
        previous_memory=previous_memory,
        previous_memory_valid=previous_memory_valid,
        control_chunks=control_chunks,
        filter_prediction=filter_prediction,
        require_attached_memory=False,
        host_step_count=prior_host_steps,
        gradient_suffix_steps=prior_gradient_suffix_steps,
    )
    context = native_context_from_prior_trace(
        controls=controls,
        prior_trace=prior_trace,
        prediction_request=(
            None if filter_prediction is None else filter_prediction.posterior_request
        ),
        modalities=modalities,
        supervise_intermediate_relations=supervise_intermediate_relations,
        posterior_adoption_route=posterior_adoption_route,
    )
    policy_forward = run_native_policy_training_forward(
        policy,
        model_inputs=model_inputs,
        context=context,
        action_attention_callback=action_attention_callback,
        future_latent_target=future_latent_target,
        wsa_da3_teacher_targets=wsa_da3_teacher_targets,
        wla_world_target=wla_world_target,
    )
    predictions = None
    if filter_prediction is not None:
        if prior_prediction is None:
            raise RuntimeError("v3 Pass A omitted its requested current-prior prediction")
        try:
            posterior_prediction = policy_forward.context.prediction_outputs[
                filter_prediction.target_name
            ]
        except KeyError as error:
            raise RuntimeError("v3 Pass B omitted its current-posterior prediction") from error
        predictions = NativeV3FilterPredictions(
            spec=filter_prediction,
            prior=prior_prediction,
            posterior=posterior_prediction,
        )
    return NativeV3TwoPassForwardResult(
        prior_trace=prior_trace,
        policy_forward=policy_forward,
        filter_predictions=predictions,
    )


def run_native_v3_two_pass_sequence(
    policy: nn.Module,
    *,
    steps: tuple[NativeV3TwoPassStep, ...],
    previous_memory: NativeLayerwisePosteriorState | None,
    previous_memory_valid: torch.Tensor,
    posterior_adoption_route: torch.Tensor | None = None,
    action_attention_callback: Callable[..., Any] | None = None,
) -> NativeV3TwoPassSequenceResult:
    """Run an explicit 1..4-frame v3 probe with prior before every correction."""

    if not 1 <= len(steps) <= 4:
        raise ValueError("v3 two-pass sequence requires exactly 1..4 frames")
    if any(not isinstance(step, NativeV3TwoPassStep) for step in steps):
        raise TypeError("v3 sequence steps must use NativeV3TwoPassStep")
    batch_size = steps[0].controls.batch_size
    if any(step.controls.batch_size != batch_size for step in steps):
        raise ValueError("v3 two-pass sequence frames must share one batch size")
    if any(step.future_latent_target is not None for step in steps[1:]):
        raise ValueError("FLARE target belongs only to the factual action frame")
    if any(step.wsa_da3_teacher_targets is not None for step in steps[1:]):
        raise ValueError("WSA DA3 targets belong only to the factual action frame")
    if any(step.wla_world_target is not None for step in steps[1:]):
        raise ValueError("WLA world target belongs only to the factual action frame")
    graph = _v3_graph_for_policy(policy)
    if any(
        bool(executed_control_chain_reset(step.effective_prior_control_chunks).any().item())
        for step in steps[1:]
    ):
        raise ValueError("a v3 two-pass sequence cannot cross an episode reset")

    first = run_native_v3_two_pass_policy_training_forward(
        policy,
        model_inputs=steps[0].model_inputs,
        controls=steps[0].controls,
        previous_memory=previous_memory,
        previous_memory_valid=previous_memory_valid,
        filter_prediction=steps[0].filter_prediction,
        modalities=steps[0].modalities,
        supervise_intermediate_relations=bool(graph.config.relation_supervision_layers),
        prior_control_chunks=steps[0].prior_control_chunks,
        prior_host_steps=steps[0].prior_host_steps,
        prior_gradient_suffix_steps=steps[0].prior_gradient_suffix_steps,
        posterior_adoption_route=posterior_adoption_route,
        action_attention_callback=action_attention_callback,
        future_latent_target=steps[0].future_latent_target,
        wsa_da3_teacher_targets=steps[0].wsa_da3_teacher_targets,
        wla_world_target=steps[0].wla_world_target,
    )
    current_memory = native_persistent_output(first.policy_forward.context)
    if not isinstance(current_memory, NativeLayerwisePosteriorState):
        raise RuntimeError("v3 factual correction omitted its layerwise posterior memory")
    current_valid = torch.ones(
        batch_size,
        dtype=torch.bool,
        device=current_memory.layer_rows.device,
    )
    prior_traces = [first.prior_trace]
    filter_predictions = [first.filter_predictions]
    auxiliary: list[NativeLocalBPTTAuxiliary] = []

    for step in steps[1:]:
        prior_trace, prior_prediction = run_native_v3_prior_chain(
            policy,
            graph=graph,
            previous_memory=current_memory,
            previous_memory_valid=current_valid,
            control_chunks=step.effective_prior_control_chunks,
            filter_prediction=step.filter_prediction,
            require_attached_memory=True,
            host_step_count=step.prior_host_steps,
            gradient_suffix_steps=step.prior_gradient_suffix_steps,
        )
        context = native_context_from_prior_trace(
            controls=step.controls,
            prior_trace=prior_trace,
            prediction_request=(
                None if step.filter_prediction is None else step.filter_prediction.posterior_request
            ),
            modalities=step.modalities,
            supervise_intermediate_relations=bool(graph.config.relation_supervision_layers),
        )
        context = _run_native_observation_training_forward(
            policy,
            model_inputs=step.model_inputs,
            context=context,
        )
        relation_output = context.relation_output
        if relation_output is None:
            raise RuntimeError("v3 auxiliary correction omitted its relation output")
        auxiliary.append(
            NativeLocalBPTTAuxiliary(
                relation_output=relation_output,
                intermediate_relation_outputs=context.intermediate_relation_outputs,
                prediction_hidden=context.prediction_hidden,
                prediction_outputs=context.prediction_outputs,
            )
        )
        predictions = None
        if step.filter_prediction is not None:
            if prior_prediction is None:
                raise RuntimeError("v3 auxiliary Pass A omitted its current-prior prediction")
            try:
                posterior_prediction = context.prediction_outputs[
                    step.filter_prediction.target_name
                ]
            except KeyError as error:
                raise RuntimeError(
                    "v3 auxiliary Pass B omitted its current-posterior prediction"
                ) from error
            predictions = NativeV3FilterPredictions(
                spec=step.filter_prediction,
                prior=prior_prediction,
                posterior=posterior_prediction,
            )
        prior_traces.append(prior_trace)
        filter_predictions.append(predictions)
        current_memory = native_persistent_output(context)
        if not isinstance(current_memory, NativeLayerwisePosteriorState):
            raise RuntimeError("v3 auxiliary correction omitted layerwise posterior memory")
        if not current_memory.layer_rows.requires_grad:
            raise RuntimeError("v3 posterior memory detached inside the probe sequence")
    return NativeV3TwoPassSequenceResult(
        primary=first.policy_forward,
        auxiliary=tuple(auxiliary),
        prior_traces=tuple(prior_traces),
        filter_predictions=tuple(filter_predictions),
    )


def run_native_v3_attached_egress(
    policy: nn.Module,
    *,
    posterior_memory: NativeLayerwisePosteriorState,
    posterior_memory_valid: torch.Tensor,
    controls: ExecutedControlBatch,
    prediction_request: NativePredictionRequest,
    target_name: str,
    prior_control_chunks: tuple[ExecutedControlBatch, ...] = (),
    prior_host_steps: int | None = None,
) -> NativeV3AttachedEgressResult:
    """Predict one next-frame target from attached M_t without a full-image pass."""

    if (
        not isinstance(prediction_request, NativePredictionRequest)
        or prediction_request.source is not PredictionSource.PRIOR
        or prediction_request.evidence is not PredictionEvidence.CURRENT_PRIOR
    ):
        raise ValueError("v3 egress requires a PRIOR/CURRENT_PRIOR request")
    if not isinstance(target_name, str) or not target_name:
        raise ValueError("v3 egress target name must be non-empty")
    if not bool(posterior_memory_valid.all().item()):
        raise ValueError("v3 attached egress requires valid factual posterior memory")
    graph = _v3_graph_for_policy(policy)
    spec = NativeV3FilterPredictionSpec(
        prior_request=prediction_request,
        posterior_request=NativePredictionRequest(
            source=PredictionSource.POSTERIOR,
            evidence=PredictionEvidence.CURRENT_POSTERIOR,
            route_ids=prediction_request.route_ids,
            horizons=prediction_request.horizons,
            addresses=prediction_request.addresses,
            valid=prediction_request.valid,
        ),
        target_name=target_name,
    )
    control_chunks = prior_control_chunks or (controls,)
    if control_chunks[-1] is not controls:
        raise ValueError("final egress prior-control chunk must be egress controls")
    if bool(executed_control_chain_reset(control_chunks).any().item()):
        raise ValueError("v3 attached egress cannot cross an episode reset")
    prior_trace, prediction = run_native_v3_prior_chain(
        policy,
        graph=graph,
        previous_memory=posterior_memory,
        previous_memory_valid=posterior_memory_valid,
        control_chunks=control_chunks,
        filter_prediction=spec,
        require_attached_memory=True,
        host_step_count=prior_host_steps,
    )
    if prediction is None:
        raise RuntimeError("v3 attached egress omitted its current-prior prediction")
    return NativeV3AttachedEgressResult(
        prior_trace=prior_trace,
        request=prediction_request,
        target_name=target_name,
        prediction=prediction,
    )


def run_native_v3_omitted_static_view_policy_training_forward(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    prior_trace: NativeLayerwisePriorTrace,
    prediction_request: NativePredictionRequest,
    omission: QwenWholeViewOmission,
    modalities: NativeModalityBatch | None = None,
    posterior_adoption_route: torch.Tensor | None = None,
    supervise_intermediate_relations: bool = False,
) -> NativeV3OmittedStaticPolicyResult:
    """Reuse factual Pass A in a complete, uncommittable omitted-view action pass."""

    _v3_graph_for_policy(policy)
    if not isinstance(prior_trace, NativeLayerwisePriorTrace):
        raise TypeError("v3 omitted-view correction requires the factual prior trace")
    if (
        not isinstance(prediction_request, NativePredictionRequest)
        or prediction_request.source is not PredictionSource.POSTERIOR
        or prediction_request.evidence is not PredictionEvidence.OMITTED_MODALITY
    ):
        raise ValueError("v3 omitted-view prediction requires posterior omitted evidence")
    if not isinstance(omission, QwenWholeViewOmission):
        raise TypeError("v3 omitted-view action requires a QwenWholeViewOmission")
    expected_valid = omission.source_valid[:, None].expand_as(prediction_request.valid)
    if not torch.equal(prediction_request.valid, expected_valid):
        raise ValueError("v3 omitted-view query validity differs from source availability")
    source_inputs = qwen_whole_view_omitted_model_inputs(model_inputs, omission)
    for name in ("actions", "noise", "time"):
        if source_inputs.get(name) is not model_inputs.get(name):
            raise RuntimeError(f"v3 omitted-view branch changed official {name}")
    policy_forward = run_native_policy_training_forward(
        policy,
        model_inputs=source_inputs,
        context=native_context_from_prior_trace(
            controls=controls,
            prior_trace=prior_trace,
            prediction_request=prediction_request,
            modalities=modalities,
            posterior_adoption_route=posterior_adoption_route,
            supervise_intermediate_relations=supervise_intermediate_relations,
        ),
        wsa_forward_role=WSALingBotForwardRole.MEASUREMENT_ONLY,
    )
    hidden = policy_forward.context.prediction_hidden
    if hidden is None or not hidden.requires_grad:
        raise RuntimeError("v3 omitted-view prediction hidden is invalid or detached")
    return NativeV3OmittedStaticPolicyResult(
        official_outputs=policy_forward.official_outputs,
        official_total_loss=policy_forward.official_total_loss,
        official_action_loss=policy_forward.official_action_loss,
        official_moe_regularizer=policy_forward.official_moe_regularizer,
        request=prediction_request,
        omitted_name=omission.omitted_name,
        prediction_hidden=hidden,
        prediction_outputs=policy_forward.context.prediction_outputs,
        omission_digest=omission.digest,
    )


@dataclass(frozen=True, slots=True)
class NativeRepresentationWindowResult:
    """One to four action-free shared-host contexts on a contiguous lane window."""

    contexts: tuple[LingBotNativeContext, ...]


def run_native_representation_window(
    policy: nn.Module,
    *,
    steps: tuple[NativeLocalBPTTStep, ...],
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
) -> NativeRepresentationWindowResult:
    """Differentiate the same observation root through one contiguous 1..4-step window."""

    if not 1 <= len(steps) <= 4:
        raise ValueError("native representation window requires exactly 1..4 host steps")
    if any(not isinstance(step, NativeLocalBPTTStep) for step in steps):
        raise TypeError("native representation window steps must use NativeLocalBPTTStep")
    batch_size = steps[0].controls.batch_size
    if any(step.controls.batch_size != batch_size for step in steps):
        raise ValueError("native representation window steps must share one batch size")
    graph = _native_graph_for_policy(policy)
    continuation_resets = [
        step.controls.reset[step.controls.token_valid].any() for step in steps[1:]
    ]
    reset_crossing = (
        torch.stack(continuation_resets).any()
        if continuation_resets
        else torch.zeros((), dtype=torch.bool, device=steps[0].controls.values.device)
    )
    if bool(reset_crossing.item()):
        raise ValueError("a representation window cannot cross an episode reset")

    current_state = previous_state
    current_valid = previous_state_valid
    contexts: list[LingBotNativeContext] = []
    for step_index, step in enumerate(steps):
        context = run_native_policy_representation_training_forward(
            policy,
            model_inputs=step.model_inputs,
            context=native_context_from_persistent_state(
                controls=step.controls,
                persistent_state=current_state,
                persistent_state_valid=current_valid,
                prediction_request=step.prediction_request,
                modalities=step.modalities,
                supervise_intermediate_relations=bool(graph.config.relation_supervision_layers),
            ),
        )
        contexts.append(context)
        current_state = native_persistent_output(context)
        state_tensor = persistent_state_tensor(current_state)
        current_valid = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=state_tensor.device,
        )
        if step_index and not state_tensor.requires_grad:
            raise RuntimeError("native representation state detached inside its local window")
    return NativeRepresentationWindowResult(contexts=tuple(contexts))


def run_native_local_bptt(
    policy: nn.Module,
    *,
    steps: tuple[NativeLocalBPTTStep, ...],
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
) -> NativeLocalBPTTResult:
    """Differentiate through one policy step and 1..3 observation-only steps."""

    return _run_native_local_bptt(
        policy,
        steps=steps,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        relation_probe=False,
    )


def run_native_relation_local_bptt(
    policy: nn.Module,
    *,
    steps: tuple[NativeLocalBPTTStep, ...],
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
) -> NativeLocalBPTTResult:
    """Train ownership over a local window without requiring a trainable state writer."""

    if any(not isinstance(step, NativeLocalBPTTStep) for step in steps):
        raise TypeError("native relation local BPTT steps must use NativeLocalBPTTStep")
    if any(step.prediction_request is not None for step in steps):
        raise ValueError("relation local BPTT cannot construct predictive queries")
    return _run_native_local_bptt(
        policy,
        steps=steps,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        relation_probe=True,
    )


def _run_native_local_bptt(
    policy: nn.Module,
    *,
    steps: tuple[NativeLocalBPTTStep, ...],
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
    relation_probe: bool,
) -> NativeLocalBPTTResult:
    if not 2 <= len(steps) <= 4:
        raise ValueError("native local BPTT requires exactly 2..4 shared-host steps")
    if any(not isinstance(step, NativeLocalBPTTStep) for step in steps):
        raise TypeError("native local BPTT steps must use NativeLocalBPTTStep")
    batch_size = steps[0].controls.batch_size
    if any(step.controls.batch_size != batch_size for step in steps):
        raise ValueError("native local BPTT steps must share one batch size")
    if any(step.future_latent_target is not None for step in steps[1:]):
        raise ValueError("FLARE target belongs only to the factual action frame")
    if any(step.wsa_da3_teacher_targets is not None for step in steps[1:]):
        raise ValueError("WSA DA3 targets belong only to the factual action frame")
    if any(step.wla_world_target is not None for step in steps[1:]):
        raise ValueError("WLA world target belongs only to the factual action frame")
    if relation_probe and steps[0].future_latent_target is not None:
        raise ValueError("relation-only local BPTT cannot consume a FLARE action target")
    graph = _native_graph_for_policy(policy)
    supervise_intermediate_relations = bool(graph.config.relation_supervision_layers)
    reset_crossing = torch.stack(
        tuple(step.controls.reset[step.controls.token_valid].any() for step in steps[1:])
    ).any()
    if bool(reset_crossing.item()):
        raise ValueError("a local BPTT window cannot cross an episode reset")

    current_state = previous_state
    current_valid = previous_state_valid
    primary: NativePolicyForwardResult | None = None
    auxiliary: list[NativeLocalBPTTAuxiliary] = []
    for step_index, step in enumerate(steps):
        context = native_context_from_persistent_state(
            controls=step.controls,
            persistent_state=current_state,
            persistent_state_valid=current_valid,
            prediction_request=step.prediction_request,
            modalities=step.modalities,
            supervise_intermediate_relations=supervise_intermediate_relations,
        )
        if step_index == 0:
            primary_forward = (
                run_native_policy_relation_training_forward
                if relation_probe
                else run_native_policy_training_forward
            )
            primary_kwargs = {
                "model_inputs": step.model_inputs,
                "context": context,
            }
            if not relation_probe:
                primary_kwargs["future_latent_target"] = step.future_latent_target
                primary_kwargs["wsa_da3_teacher_targets"] = step.wsa_da3_teacher_targets
                primary_kwargs["wla_world_target"] = step.wla_world_target
            primary = primary_forward(policy, **primary_kwargs)
            context = primary.context
        else:
            context = _run_native_observation_training_forward(
                policy,
                model_inputs=step.model_inputs,
                context=context,
                require_prediction_grad=not relation_probe,
                required_relation_grad_fields=("ownership",) if relation_probe else (),
            )
            relation_output = context.relation_output
            if relation_output is None:
                raise RuntimeError("native local BPTT auxiliary omitted its relation output")
            auxiliary.append(
                NativeLocalBPTTAuxiliary(
                    relation_output=relation_output,
                    intermediate_relation_outputs=context.intermediate_relation_outputs,
                    prediction_hidden=context.prediction_hidden,
                    prediction_outputs=context.prediction_outputs,
                )
            )
        current_state = native_persistent_output(context)
        state_tensor = persistent_state_tensor(current_state)
        current_valid = torch.ones(
            batch_size,
            dtype=torch.bool,
            device=state_tensor.device,
        )
        if step_index and not relation_probe and not state_tensor.requires_grad:
            raise RuntimeError("native local BPTT state was detached inside the sampled window")
    if primary is None:  # The validated 2..4-step contract makes this defensive only.
        raise RuntimeError("native local BPTT omitted its primary policy forward")
    return NativeLocalBPTTResult(primary=primary, auxiliary=tuple(auxiliary))


def reconstruct_native_state_no_grad(
    policy: nn.Module,
    *,
    steps: tuple[NativeLocalBPTTStep, ...],
) -> NativePersistentState:
    """Replay one fixed-weight contiguous prefix for a read-only diagnostic."""

    if not steps:
        raise ValueError("native state reconstruction requires a non-empty episode prefix")
    if any(step.prediction_request is not None for step in steps):
        raise ValueError("native state reconstruction cannot contain prediction queries")
    state: NativePersistentState | None = None
    for index, step in enumerate(steps):
        valid = torch.full(
            (step.controls.batch_size,),
            index > 0,
            dtype=torch.bool,
            device=step.controls.values.device,
        )
        state = run_native_state_reconstruction_step(
            policy,
            model_inputs=step.model_inputs,
            controls=step.controls,
            previous_state=state,
            previous_state_valid=valid,
            modalities=step.modalities,
        )
    if state is None:  # The non-empty loop above must publish one state.
        raise RuntimeError("native state reconstruction did not publish a posterior")
    return clone_persistent_state(state)


def run_native_state_reconstruction_step(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor,
    modalities: NativeModalityBatch | None = None,
) -> NativePersistentState:
    """Execute one weight-shared no-grad update for fixed-weight replay audit."""

    if previous_state_valid.shape != (controls.batch_size,) or (
        previous_state_valid.dtype != torch.bool
    ):
        raise ValueError("native reconstruction validity must be boolean [batch]")
    if previous_state_valid.device != controls.values.device:
        raise ValueError("native reconstruction validity and controls must share one device")
    if previous_state is None and previous_state_valid.any():
        raise ValueError("native reconstruction cannot validate an absent previous state")
    with torch.no_grad():
        result = _run_native_policy_forward(
            policy,
            model_inputs=model_inputs,
            context=native_context_from_persistent_state(
                controls=controls,
                persistent_state=previous_state,
                persistent_state_valid=previous_state_valid,
                modalities=modalities,
            ),
            require_official_grad=False,
            require_prediction_grad=False,
            required_relation_grad_fields=(),
        )
    state = native_persistent_output(result.context)
    return clone_persistent_state(state)


@dataclass(frozen=True, slots=True)
class NativePreparedLaneBatch:
    """Read-only recurrent inputs and stamps for one stream microbatch."""

    routing: NativeCALVINRouting
    previous_state: NativePersistentState
    previous_state_valid: torch.Tensor
    wrong_time_state: NativePersistentState
    wrong_time_state_valid: torch.Tensor
    previous_row_bindings: tuple[RowBindings, ...]
    next_state_ages: tuple[int, ...]
    optimizer_lags: tuple[int, ...]
    source_weight_version: int
    attempt_token: int
    preparation_token: int


def _lane_episode_address_state(
    config: NativeLaneConfig,
    episode_key: str,
) -> EpisodeAddressState:
    if not config.addressed or config.episode_address_codebook_sha256 is None:
        raise ValueError("episode address construction requires an addressed lane contract")
    payload = json.dumps(
        {
            "episode_key": episode_key,
            "lane_contract_digest": config.contract_digest,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    episode_id = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") >> 1
    episode_ids = torch.tensor([episode_id], dtype=torch.long, device=config.device)
    return EpisodeAddressState(
        permutation=deterministic_episode_permutation(episode_ids, config.capacity),
        codebook_sha256=config.episode_address_codebook_sha256,
    )


def _zero_lane_state(
    config: NativeLaneConfig,
    *,
    episode_key: str,
) -> NativePersistentState:
    shape = (
        (1, config.capacity, config.host_width)
        if config.num_layers is None
        else (1, config.num_layers, config.capacity, config.host_width)
    )
    zero_tensor = torch.zeros(shape, device=config.device, dtype=config.dtype)
    if config.paired:
        if config.paired_architecture_identity is None or config.paired_source_width is None:
            raise RuntimeError("paired lane contract lost its source identity")
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=zero_tensor,
            source_queries=torch.zeros(
                1,
                config.capacity,
                config.paired_source_width,
                device=config.device,
                dtype=config.paired_source_dtype,
            ),
            architecture_identity=config.paired_architecture_identity,
        )
    if config.addressed:
        if config.addressed_architecture_identity is None:
            raise RuntimeError("addressed lane contract lost its architecture identity")
        return AddressedLayerwisePosteriorState(
            layer_rows=zero_tensor,
            episode_address_state=_lane_episode_address_state(config, episode_key),
            architecture_identity=config.addressed_architecture_identity,
        )
    if config.num_layers is None:
        return NativePosteriorState(zero_tensor)
    return NativeLayerwisePosteriorState(zero_tensor)


def native_cold_state_for_episode_keys(
    config: NativeLaneConfig,
    *,
    episode_keys: tuple[str, ...],
) -> NativePersistentState:
    """Build the canonical invalid-memory state used at episode cold start."""

    if not episode_keys or any(not isinstance(key, str) or not key for key in episode_keys):
        raise ValueError("cold native state requires one or more non-empty episode keys")
    return stack_persistent_states(
        tuple(_zero_lane_state(config, episode_key=key) for key in episode_keys)
    )


def _zero_like_lane_state(state: NativePersistentState) -> NativePersistentState:
    zero_tensor = torch.zeros_like(persistent_state_tensor(state))
    if isinstance(state, NativeVidEoMTPairedPosteriorState):
        return NativeVidEoMTPairedPosteriorState(
            layer_rows=zero_tensor,
            source_queries=torch.zeros_like(state.source_queries),
            architecture_identity=state.architecture_identity,
        )
    if isinstance(state, AddressedLayerwisePosteriorState):
        return AddressedLayerwisePosteriorState(
            layer_rows=zero_tensor,
            episode_address_state=state.episode_address_state,
            architecture_identity=state.architecture_identity,
        )
    if isinstance(state, NativePosteriorState):
        return NativePosteriorState(zero_tensor)
    return NativeLayerwisePosteriorState(zero_tensor)


class NativeOptimizerLaneAttempt:
    """Collect disjoint accumulation microbatches for one optimizer step."""

    def __init__(
        self,
        coordinator: NativeTrainingLaneCoordinator,
        *,
        token: int,
        optimizer_step: int,
        source_weight_version: int,
    ) -> None:
        self._coordinator = coordinator
        self.token = token
        self.optimizer_step = optimizer_step
        self.source_weight_version = source_weight_version
        self._transactions: list[NativeLaneTransaction] = []
        self._seen_lanes: set[int] = set()
        self._prepared: NativePreparedLaneBatch | None = None
        self._next_preparation_token = 0
        self._closed = False

    def _require_open(self) -> None:
        if self._closed or self._coordinator._active is not self:
            raise NativeLaneError("native optimizer lane attempt is closed")
        if self._coordinator.poisoned:
            raise NativeLaneError("native lane coordinator is poisoned; restart from a checkpoint")

    def prepare(self, routing: NativeCALVINRouting) -> NativePreparedLaneBatch:
        """Read detached previous rows before one official host forward."""

        self._require_open()
        if self._prepared is not None:
            raise NativeLaneError("finish or discard the current prepared microbatch first")
        if not isinstance(routing, NativeCALVINRouting):
            raise TypeError("native lane preparation requires NativeCALVINRouting")
        if routing.optimizer_step != self.optimizer_step:
            raise NativeLaneError("CALVIN routing belongs to another optimizer step")
        lane_ids = routing.lane_ids
        if len(set(lane_ids)) != len(lane_ids) or self._seen_lanes.intersection(lane_ids):
            raise NativeLaneError("an optimizer step may consume each recurrent lane only once")

        config = self._coordinator.bank.config
        states: list[NativePersistentState] = []
        valid: list[bool] = []
        wrong_time_states: list[NativePersistentState] = []
        wrong_time_valid: list[bool] = []
        previous_row_bindings: list[RowBindings] = []
        ages: list[int] = []
        optimizer_lags: list[int] = []
        for lane_id, episode_key, frame_index, reset in zip(
            routing.lane_ids,
            routing.episode_keys,
            routing.frame_indices,
            routing.reset,
            strict=True,
        ):
            if reset:
                zero = _zero_lane_state(config, episode_key=episode_key)
                states.append(zero)
                wrong_time_states.append(zero)
                valid.append(False)
                wrong_time_valid.append(False)
                previous_row_bindings.append(())
                ages.append(0)
                optimizer_lags.append(0)
                continue
            read = self._coordinator.bank.read(
                lane_id,
                episode_key=episode_key,
                next_frame_index=frame_index,
                optimizer_step=self.optimizer_step,
                source_weight_version=self.source_weight_version,
            )
            if read is None:
                raise NativeLaneError("a non-reset stream lane has no previous posterior")
            states.append(read.state)
            previous_row_bindings.append(read.row_bindings)
            predecessor = self._coordinator.bank.read_predecessor(
                lane_id,
                episode_key=episode_key,
                next_frame_index=frame_index,
                source_weight_version=self.source_weight_version,
            )
            if predecessor is None:
                wrong_time_states.append(_zero_like_lane_state(read.state))
                wrong_time_valid.append(False)
            else:
                wrong_time_states.append(predecessor)
                wrong_time_valid.append(True)
            valid.append(True)
            ages.append(read.stamp.state_age + 1)
            optimizer_lags.append(read.optimizer_lag)
        prepared = NativePreparedLaneBatch(
            routing=routing,
            previous_state=stack_persistent_states(tuple(states)),
            previous_state_valid=torch.tensor(valid, dtype=torch.bool, device=config.device),
            wrong_time_state=stack_persistent_states(tuple(wrong_time_states)),
            wrong_time_state_valid=torch.tensor(
                wrong_time_valid,
                dtype=torch.bool,
                device=config.device,
            ),
            previous_row_bindings=tuple(previous_row_bindings),
            next_state_ages=tuple(ages),
            optimizer_lags=tuple(optimizer_lags),
            source_weight_version=self.source_weight_version,
            attempt_token=self.token,
            preparation_token=self._next_preparation_token,
        )
        self._next_preparation_token += 1
        self._prepared = prepared
        return prepared

    def discard(self, prepared: NativePreparedLaneBatch) -> None:
        """Release a read-only preparation after a failed forward."""

        self._require_open()
        if self._prepared is not prepared:
            raise NativeLaneError("prepared microbatch is unknown or already finalized")
        self._prepared = None

    def stage(
        self,
        prepared: NativePreparedLaneBatch,
        posterior_state: NativePersistentState,
        *,
        row_bindings_by_batch: tuple[RowBindings, ...],
    ) -> None:
        """Stage detached final rows; publication still waits for optimizer success."""

        self._require_open()
        if self._prepared is not prepared:
            raise NativeLaneError("prepared microbatch is unknown or already finalized")
        if (
            prepared.attempt_token != self.token
            or prepared.source_weight_version != self.source_weight_version
        ):
            raise NativeLaneError("prepared microbatch belongs to another lane attempt")
        if not isinstance(
            posterior_state,
            (NativePosteriorState, NativeLayerwisePosteriorState),
        ):
            raise TypeError("native lane staging requires a typed persistent state")
        if posterior_state.batch_size != prepared.routing.batch_size:
            raise ValueError("posterior batch and prepared CALVIN routing differ")
        if len(row_bindings_by_batch) != prepared.routing.batch_size:
            raise ValueError("row bindings and prepared CALVIN routing differ")
        states = unbind_persistent_state(posterior_state)
        staged: list[NativeLaneTransaction] = []
        try:
            for index, state in enumerate(states):
                routing = prepared.routing
                transaction = self._coordinator.bank.stage(
                    routing.lane_ids[index],
                    state,
                    NativeLaneStamp(
                        episode_key=routing.episode_keys[index],
                        frame_index=routing.frame_indices[index],
                        state_age=prepared.next_state_ages[index],
                        producer_optimizer_step=self.optimizer_step,
                        source_weight_version=self.source_weight_version,
                    ),
                    reset=routing.reset[index],
                    row_bindings=row_bindings_by_batch[index],
                )
                staged.append(transaction)
        except BaseException:
            if staged:
                self._coordinator.bank.abort_batch(tuple(staged))
            raise
        self._transactions.extend(staged)
        self._seen_lanes.update(prepared.routing.lane_ids)
        self._prepared = None

    def _abort_transactions(self) -> None:
        if self._transactions:
            self._coordinator.bank.abort_batch(tuple(self._transactions))
            self._transactions.clear()

    def abort(self) -> None:
        """Abort before an optimizer is attempted; this is safely retryable."""

        self._require_open()
        self._prepared = None
        self._abort_transactions()
        self._closed = True
        self._coordinator._close(self)

    def finish(self, optimizer_attempt: Callable[[], int | None]) -> bool:
        """Run one optimizer attempt, then atomically publish all staged lanes."""

        self._require_open()
        if self._prepared is not None:
            raise NativeLaneError("cannot optimize with an unfinished prepared microbatch")
        if not self._transactions:
            raise NativeLaneError("cannot optimize without staged recurrent lanes")
        if not callable(optimizer_attempt):
            raise TypeError("optimizer_attempt must be callable")
        try:
            successful_step = optimizer_attempt()
        except BaseException:
            self._coordinator._poisoned = True
            self._abort_transactions()
            self._closed = True
            self._coordinator._close(self)
            raise
        if successful_step is None:
            self._abort_transactions()
            self._closed = True
            self._coordinator._close(self)
            return False
        if (
            isinstance(successful_step, bool)
            or not isinstance(successful_step, int)
            or successful_step != self.optimizer_step + 1
        ):
            self._coordinator._poisoned = True
            self._abort_transactions()
            self._closed = True
            self._coordinator._close(self)
            raise NativeLaneError("optimizer returned an ambiguous or non-contiguous step")
        try:
            self._coordinator.bank.commit_batch_after_optimizer(
                tuple(self._transactions),
                successful_optimizer_step=successful_step,
            )
        except BaseException:
            self._coordinator._poisoned = True
            self._abort_transactions()
            self._closed = True
            self._coordinator._close(self)
            raise
        self._transactions.clear()
        self._closed = True
        self._coordinator._close(self)
        return True

    def finish_stateless(self, optimizer_attempt: Callable[[], int | None]) -> bool:
        """Run one optimizer update while proving that no lane can be published."""

        self._require_open()
        if self._prepared is not None:
            raise NativeLaneError("cannot optimize with an unfinished prepared microbatch")
        if self._transactions:
            raise NativeLaneError("stateless optimizer attempt cannot publish recurrent lanes")
        if not callable(optimizer_attempt):
            raise TypeError("optimizer_attempt must be callable")
        try:
            successful_step = optimizer_attempt()
        except BaseException:
            self._coordinator._poisoned = True
            self._closed = True
            self._coordinator._close(self)
            raise
        if successful_step is None:
            self._closed = True
            self._coordinator._close(self)
            return False
        if (
            isinstance(successful_step, bool)
            or not isinstance(successful_step, int)
            or successful_step != self.optimizer_step + 1
        ):
            self._coordinator._poisoned = True
            self._closed = True
            self._coordinator._close(self)
            raise NativeLaneError("optimizer returned an ambiguous or non-contiguous step")
        self._closed = True
        self._coordinator._close(self)
        return True


class NativeTrainingLaneCoordinator:
    """Fail-closed transaction owner above the detached real-age lane bank."""

    def __init__(self, bank: NativeTrainingLaneBank) -> None:
        if not isinstance(bank, NativeTrainingLaneBank):
            raise TypeError("native lane coordinator requires a NativeTrainingLaneBank")
        self.bank = bank
        self._active: NativeOptimizerLaneAttempt | None = None
        self._next_token = 0
        self._poisoned = False

    @property
    def poisoned(self) -> bool:
        return self._poisoned

    def begin(
        self,
        *,
        optimizer_step: int,
        source_weight_version: int,
    ) -> NativeOptimizerLaneAttempt:
        if self._poisoned:
            raise NativeLaneError("native lane coordinator is poisoned; restart from a checkpoint")
        if self._active is not None:
            raise NativeLaneError("another native optimizer lane attempt is active")
        counters = (optimizer_step, source_weight_version)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in counters
        ):
            raise ValueError("optimizer and source-weight versions must be non-negative integers")
        attempt = NativeOptimizerLaneAttempt(
            self,
            token=self._next_token,
            optimizer_step=optimizer_step,
            source_weight_version=source_weight_version,
        )
        self._next_token += 1
        self._active = attempt
        return attempt

    def _close(self, attempt: NativeOptimizerLaneAttempt) -> None:
        if self._active is not attempt:
            raise NativeLaneError("native lane coordinator closed an unknown attempt")
        self._active = None


@dataclass(frozen=True, slots=True)
class NativeParameterManifest:
    """Content-independent schema of every trainable parameter in one run."""

    canonical_names: tuple[str, ...]
    parameter_count: int
    trainable_numel: int
    schema_sha256: str


def _trainable_inventory(
    modules: Mapping[str, nn.Module],
) -> tuple[dict[int, nn.Parameter], dict[int, tuple[str, ...]], NativeParameterManifest]:
    if not modules:
        raise ValueError("native parameter audit requires at least one module root")
    parameters: dict[int, nn.Parameter] = {}
    names: dict[int, list[str]] = {}
    roots: dict[int, str] = {}
    for root_name, module in modules.items():
        if not isinstance(root_name, str) or not root_name:
            raise ValueError("native parameter roots require non-empty names")
        if not isinstance(module, nn.Module):
            raise TypeError(f"native parameter root {root_name} is not a module")
        for local_name, parameter in module.named_parameters(remove_duplicate=False):
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            previous_root = roots.get(parameter_id)
            if previous_root is not None and previous_root != root_name:
                raise ValueError(
                    "a trainable parameter is exposed by multiple native module roots: "
                    f"{previous_root}, {root_name}"
                )
            roots[parameter_id] = root_name
            parameters[parameter_id] = parameter
            names.setdefault(parameter_id, []).append(f"{root_name}.{local_name}")
    if not parameters:
        raise ValueError("native parameter roots contain no trainable parameters")

    frozen_names = {key: tuple(sorted(set(value))) for key, value in names.items()}
    records = []
    for parameter_id, parameter in parameters.items():
        aliases = frozen_names[parameter_id]
        records.append(
            {
                "aliases": aliases,
                "dtype": str(parameter.dtype),
                "numel": parameter.numel(),
                "shape": tuple(parameter.shape),
            }
        )
    records.sort(key=lambda value: value["aliases"])
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    manifest = NativeParameterManifest(
        canonical_names=tuple(record["aliases"][0] for record in records),
        parameter_count=len(parameters),
        trainable_numel=sum(parameter.numel() for parameter in parameters.values()),
        schema_sha256=hashlib.sha256(encoded).hexdigest(),
    )
    return parameters, frozen_names, manifest


def audit_native_optimizer_coverage(
    *,
    modules: Mapping[str, nn.Module],
    optimizer: Optimizer,
) -> NativeParameterManifest:
    """Require exact optimizer ownership of all and only declared trainables."""

    if not isinstance(optimizer, Optimizer):
        raise TypeError("native optimizer audit requires a torch Optimizer")
    expected, names, manifest = _trainable_inventory(modules)
    owned: dict[int, nn.Parameter] = {}
    duplicates: list[str] = []
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if not isinstance(parameter, nn.Parameter):
                raise TypeError("optimizer groups may contain only Parameters")
            parameter_id = id(parameter)
            if parameter_id in owned:
                duplicates.append(names.get(parameter_id, (f"id={parameter_id}",))[0])
            owned[parameter_id] = parameter
    if duplicates:
        raise ValueError(f"optimizer owns duplicate parameters: {sorted(duplicates)}")
    missing = sorted(names[key][0] for key in expected.keys() - owned.keys())
    unexpected = sorted(f"id={key}" for key in owned.keys() - expected.keys())
    if missing or unexpected:
        raise ValueError(
            f"native optimizer coverage mismatch: missing={missing}, unexpected={unexpected}"
        )
    return manifest

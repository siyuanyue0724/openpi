"""Counterfactual instruments for the LingBot-native correction objective.

Nothing in this module is a learned model component.  The helpers rerun the
exact shared host under controlled input interventions and score every output
against one frozen, loss-side target and assignment.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativePriorStepper,
)
from picf_next.lingbot_native.modalities import NativeModalityBatch, NativeModalityStream
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_cache import LINGBOT_PREDICTIVE_TARGET_SPACE
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveTarget,
    PredictiveRowAssignment,
    native_predictive_term,
)
from picf_next.lingbot_native.state import NativePosteriorState
from picf_next.lingbot_native.supervision import SequenceAssignment
from picf_next.lingbot_native.temporal import rollout_native_prior_prediction
from picf_next.lingbot_native.training import (
    run_native_policy_diagnostic_forward,
    run_native_policy_observation_diagnostic_forward,
)

PREDICTIVE_CORRECTION_COUNTERFACTUAL_SCHEMA = (
    "picf-next.lingbot-predictive-correction-counterfactual/v1"
)
PREDICTIVE_FUTURE_COUNTERFACTUAL_SCHEMA = (
    "picf-next.lingbot-predictive-future-counterfactual/v1"
)
PREDICTIVE_FIXED_BATCH_FIT_SCHEMA = "picf-next.lingbot-predictive-fixed-batch-fit/v3"
BEHAVIOR_CAUSAL_PROBE_SCHEMA = "picf-next.lingbot-behavior-causal-probe/v2"
BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_SCHEMA = (
    "picf-next.lingbot-behavior-posterior-control-factorial/v1"
)

BEHAVIOR_POSTERIOR_FACTORIAL_LEVELS = ("factual", "zero", "batch_shift")
BEHAVIOR_CONTROL_FACTORIAL_LEVELS = ("factual", "zero", "batch_shift")
BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS = tuple(
    (posterior_level, control_level)
    for posterior_level in BEHAVIOR_POSTERIOR_FACTORIAL_LEVELS
    for control_level in BEHAVIOR_CONTROL_FACTORIAL_LEVELS
)

ZERO_SOURCE = "zero_source"
ABSENT_SOURCE = "absent_source"
ROW_SHIFT_SOURCE = "row_shift_source"
BATCH_SHIFT_SOURCE = "batch_shift_source"
WRONG_TIME_SOURCE = "wrong_time_source"
ZERO_CONTROL = "zero_control"
BATCH_SHIFT_CONTROL = "batch_shift_control"
ZERO_CURRENT_OBSERVATION = "zero_current_observation"
MATCHED_NOISE_SOURCE = "matched_noise_source"

_INTERVENTION_NAMES = frozenset(
    {
        ABSENT_SOURCE,
        ZERO_SOURCE,
        ROW_SHIFT_SOURCE,
        BATCH_SHIFT_SOURCE,
        WRONG_TIME_SOURCE,
        ZERO_CONTROL,
        BATCH_SHIFT_CONTROL,
        ZERO_CURRENT_OBSERVATION,
        MATCHED_NOISE_SOURCE,
    }
)
_REQUIRED_INTERVENTIONS = frozenset(
    {
        ZERO_SOURCE,
        ZERO_CONTROL,
        ZERO_CURRENT_OBSERVATION,
    }
)
_FUTURE_REQUIRED_INTERVENTIONS = _REQUIRED_INTERVENTIONS | {ABSENT_SOURCE}
PREDICTIVE_FIXED_BATCH_ARMS = (
    "full_host",
    "native_graph_only",
    "readout_only",
    "shuffled_target",
)


def _finite(value: object, *, name: str, non_negative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be finite")
    measured = float(value)
    if not math.isfinite(measured) or (non_negative and measured < 0):
        raise ValueError(f"{name} must be finite" + (" and non-negative" if non_negative else ""))
    return measured


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _cyclic_indices(size: int, *, device: torch.device) -> torch.Tensor:
    if size < 2:
        raise ValueError("a cyclic counterfactual requires at least two elements")
    return torch.arange(size, device=device).roll(1)


def _validate_state_pair(
    state: NativePosteriorState,
    valid: torch.Tensor,
    *,
    reference: NativePosteriorState,
    name: str,
) -> None:
    if not isinstance(state, NativePosteriorState):
        raise TypeError(f"{name} must use NativePosteriorState")
    if state.rows.shape != reference.rows.shape:
        raise ValueError(f"{name} state shape differs from the factual state")
    if state.rows.device != reference.rows.device or state.rows.dtype != reference.rows.dtype:
        raise ValueError(f"{name} state placement differs from the factual state")
    if valid.shape != (state.batch_size,) or valid.dtype != torch.bool:
        raise ValueError(f"{name} validity must be boolean [batch]")
    if valid.device != state.rows.device:
        raise ValueError(f"{name} validity and state must share one device")


def _validate_control_pair(
    controls: ExecutedControlBatch,
    *,
    reference: ExecutedControlBatch,
    name: str,
) -> None:
    if not isinstance(controls, ExecutedControlBatch):
        raise TypeError(f"{name} must use ExecutedControlBatch")
    if controls.values.shape != reference.values.shape:
        raise ValueError(f"{name} shape differs from factual controls")
    if controls.values.device != reference.values.device:
        raise ValueError(f"{name} placement differs from factual controls")
    if controls.values.dtype != reference.values.dtype:
        raise ValueError(f"{name} dtype differs from factual controls")


def zero_executed_control(controls: ExecutedControlBatch) -> ExecutedControlBatch:
    """Replace executed action values with a no-op over the same elapsed time."""

    if not isinstance(controls, ExecutedControlBatch):
        raise TypeError("zero-control intervention requires ExecutedControlBatch")
    return ExecutedControlBatch(
        values=torch.zeros_like(controls.values),
        field_valid=controls.field_valid,
        token_valid=controls.token_valid,
        delta_time=controls.delta_time,
        reset=controls.reset,
        acknowledged=controls.acknowledged,
    )


def batch_shift_executed_control(controls: ExecutedControlBatch) -> ExecutedControlBatch:
    """Pair each sample with another sample's complete acknowledged control."""

    if not isinstance(controls, ExecutedControlBatch):
        raise TypeError("control shuffle requires ExecutedControlBatch")
    indices = _cyclic_indices(controls.batch_size, device=controls.values.device)
    return ExecutedControlBatch(
        values=controls.values.index_select(0, indices),
        field_valid=controls.field_valid.index_select(0, indices),
        token_valid=controls.token_valid.index_select(0, indices),
        delta_time=controls.delta_time.index_select(0, indices),
        reset=controls.reset.index_select(0, indices),
        acknowledged=controls.acknowledged.index_select(0, indices),
    )


def reverse_executed_control(controls: ExecutedControlBatch) -> ExecutedControlBatch:
    """Reverse a complete future action interval without changing its fields."""

    if not isinstance(controls, ExecutedControlBatch):
        raise TypeError("control reversal requires ExecutedControlBatch")
    return ExecutedControlBatch(
        values=controls.values.flip(1),
        field_valid=controls.field_valid.flip(1),
        token_valid=controls.token_valid.flip(1),
        delta_time=controls.delta_time.flip(1),
        reset=controls.reset.flip(1),
        acknowledged=controls.acknowledged.flip(1),
    )


def _collect_tensor_leaves(value: object, *, prefix: str) -> dict[str, torch.Tensor]:
    leaves: dict[str, torch.Tensor] = {}
    if isinstance(value, torch.Tensor):
        leaves[prefix] = value.detach().clone()
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            leaves.update(_collect_tensor_leaves(value[key], prefix=f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            leaves.update(_collect_tensor_leaves(item, prefix=f"{prefix}.{index}"))
    elif is_dataclass(value) and not isinstance(value, type):
        for field in fields(value):
            leaves.update(
                _collect_tensor_leaves(
                    getattr(value, field.name),
                    prefix=f"{prefix}.{field.name}",
                )
            )
    return leaves


def _deploy_tensor_snapshot(result: object) -> dict[str, torch.Tensor]:
    official_outputs = getattr(result, "official_outputs", None)
    context = getattr(result, "context", None)
    if not isinstance(official_outputs, tuple) or not isinstance(context, LingBotNativeContext):
        raise TypeError("behavior causal probe received an invalid policy result")
    snapshot = _collect_tensor_leaves(official_outputs, prefix="official")
    for state_name in ("prior_state", "posterior_state"):
        state = getattr(context, state_name)
        if not isinstance(state, NativePosteriorState):
            raise RuntimeError(f"behavior causal probe omitted {state_name}")
        snapshot[f"context.{state_name}.rows"] = state.rows.detach().clone()
    relation = context.relation_output
    if relation is None:
        raise RuntimeError("behavior causal probe omitted final relation output")
    snapshot.update(_collect_tensor_leaves(relation, prefix="context.relation"))
    for layer, intermediate in sorted(context.intermediate_relation_outputs.items()):
        snapshot.update(
            _collect_tensor_leaves(intermediate, prefix=f"context.intermediate.{layer}")
        )
    return snapshot


@dataclass(frozen=True, slots=True)
class NativeBehaviorCausalProbe:
    """Fixed-weight proof for the isolated shared-weight behavior view."""

    horizon: int
    deploy_tensor_count: int
    intervention_prediction_changed: tuple[tuple[str, bool], ...]

    def __post_init__(self) -> None:
        _positive_integer(self.horizon, name="behavior causal horizon")
        _positive_integer(self.deploy_tensor_count, name="deploy tensor count")
        expected = ("peer_replace_shuffle", "reverse", "zero")
        names = tuple(name for name, _changed in self.intervention_prediction_changed)
        if names != expected:
            raise ValueError("behavior causal interventions differ from the frozen set")
        if not any(changed for _name, changed in self.intervention_prediction_changed):
            raise ValueError("future-control interventions changed no prediction output")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": BEHAVIOR_CAUSAL_PROBE_SCHEMA,
            "status": "PASS",
            "horizon": self.horizon,
            "deploy_tensor_count": self.deploy_tensor_count,
            "deploy_bit_identical": True,
            "fresh_primary_rerun_bit_identical": True,
            "cold_deploy_omits_future_controls": True,
            "deploy_isolation": "separate_same_weight_auxiliary_forward",
            "intervention_prediction_changed": dict(self.intervention_prediction_changed),
        }


def _validate_behavior_factorial_prediction(
    prediction: torch.Tensor,
    *,
    name: str,
) -> None:
    if (
        not isinstance(prediction, torch.Tensor)
        or prediction.ndim != 4
        or min(prediction.shape) <= 0
        or not prediction.is_floating_point()
    ):
        raise ValueError(f"{name} must be floating [batch,rows,queries,width]")
    if prediction.requires_grad or prediction.grad_fn is not None:
        raise ValueError(f"{name} must be detached")
    if not torch.isfinite(prediction).all():
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class NativeBehaviorPosteriorControlCell:
    """One detached prediction from the row-only posterior/control factorial."""

    posterior_level: str
    control_level: str
    prediction: torch.Tensor

    def __post_init__(self) -> None:
        if self.key not in BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS:
            raise ValueError("behavior factorial cell names an unsupported level pair")
        _validate_behavior_factorial_prediction(
            self.prediction,
            name=f"behavior factorial prediction {self.key!r}",
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.posterior_level, self.control_level)


@dataclass(frozen=True, slots=True)
class NativeBehaviorPosteriorControlPredictions:
    """Exact row-only 3x3 outputs plus a deterministic factual repeat."""

    request: NativePredictionRequest
    cells: tuple[NativeBehaviorPosteriorControlCell, ...]
    factual_repeat: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.request, NativePredictionRequest):
            raise TypeError("behavior factorial requires a typed prediction request")
        if (
            self.request.source is not PredictionSource.PRIOR
            or self.request.evidence is not PredictionEvidence.FUTURE
        ):
            raise ValueError("behavior factorial requires a prior-row future request")
        if not isinstance(self.cells, tuple) or any(
            not isinstance(cell, NativeBehaviorPosteriorControlCell) for cell in self.cells
        ):
            raise TypeError("behavior factorial cells must be one typed tuple")
        if tuple(cell.key for cell in self.cells) != BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS:
            raise ValueError("behavior factorial cells are missing, duplicated or out of order")

        reference = self.cells[0].prediction
        if (
            reference.shape[0] != self.request.batch_size
            or reference.shape[2] != self.request.query_count
            or reference.device != self.request.addresses.device
        ):
            raise ValueError("behavior factorial prediction axes differ from its request")
        for cell in self.cells[1:]:
            prediction = cell.prediction
            if (
                prediction.shape != reference.shape
                or prediction.device != reference.device
                or prediction.dtype != reference.dtype
            ):
                raise ValueError("behavior factorial predictions differ in shape or placement")

        _validate_behavior_factorial_prediction(
            self.factual_repeat,
            name="behavior factorial factual repeat",
        )
        if (
            self.factual_repeat.shape != reference.shape
            or self.factual_repeat.device != reference.device
            or self.factual_repeat.dtype != reference.dtype
        ):
            raise ValueError("behavior factorial factual repeat differs in shape or placement")
        if not torch.equal(self.factual_repeat, reference):
            raise ValueError("behavior factorial factual repeat is not bit-identical")

    @property
    def schema(self) -> str:
        return BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_SCHEMA

    def as_mapping(self) -> dict[tuple[str, str], torch.Tensor]:
        return {cell.key: cell.prediction for cell in self.cells}

    def prediction_for(self, posterior_level: str, control_level: str) -> torch.Tensor:
        key = (posterior_level, control_level)
        if key not in BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS:
            raise KeyError(key)
        return self.as_mapping()[key]


@dataclass(frozen=True, slots=True)
class NativeBehaviorPosteriorControlCellDiagnostics:
    """Loss and prediction displacement for one frozen row-only cell."""

    posterior_level: str
    control_level: str
    loss: float
    loss_margin_over_factual: float
    normalized_prediction_l1: float

    def __post_init__(self) -> None:
        if self.key not in BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS:
            raise ValueError("behavior factorial diagnostic names an unsupported cell")
        _finite(self.loss, name="behavior factorial loss", non_negative=True)
        _finite(self.loss_margin_over_factual, name="behavior factorial loss margin")
        _finite(
            self.normalized_prediction_l1,
            name="behavior factorial prediction distance",
            non_negative=True,
        )

    @property
    def key(self) -> tuple[str, str]:
        return (self.posterior_level, self.control_level)

    def as_dict(self) -> dict[str, str | float]:
        return {
            "posterior_level": self.posterior_level,
            "control_level": self.control_level,
            "loss": self.loss,
            "loss_margin_over_factual": self.loss_margin_over_factual,
            "normalized_prediction_l1": self.normalized_prediction_l1,
        }


@dataclass(frozen=True, slots=True)
class NativeBehaviorPosteriorControlDiagnostics:
    """Frozen-target posterior/control main effects and interactions."""

    valid_target_count: int
    factual_loss: float
    cells: tuple[NativeBehaviorPosteriorControlCellDiagnostics, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.valid_target_count, name="behavior factorial valid target count")
        _finite(self.factual_loss, name="behavior factorial factual loss", non_negative=True)
        if tuple(cell.key for cell in self.cells) != BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS:
            raise ValueError("behavior factorial diagnostics are incomplete or out of order")
        for cell in self.cells:
            if not math.isclose(
                cell.loss_margin_over_factual,
                cell.loss - self.factual_loss,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError("behavior factorial loss margin is inconsistent")

    def _loss(self, posterior_level: str, control_level: str) -> float:
        key = (posterior_level, control_level)
        return next(cell.loss for cell in self.cells if cell.key == key)

    @property
    def posterior_margins_at_factual_control(self) -> dict[str, float]:
        return {
            level: self._loss(level, "factual") - self.factual_loss
            for level in BEHAVIOR_POSTERIOR_FACTORIAL_LEVELS[1:]
        }

    @property
    def control_margins_at_factual_posterior(self) -> dict[str, float]:
        return {
            level: self._loss("factual", level) - self.factual_loss
            for level in BEHAVIOR_CONTROL_FACTORIAL_LEVELS[1:]
        }

    @property
    def interactions(self) -> dict[str, float]:
        return {
            f"{posterior_level}__{control_level}": (
                self._loss(posterior_level, control_level)
                - self._loss(posterior_level, "factual")
                - self._loss("factual", control_level)
                + self.factual_loss
            )
            for posterior_level in BEHAVIOR_POSTERIOR_FACTORIAL_LEVELS[1:]
            for control_level in BEHAVIOR_CONTROL_FACTORIAL_LEVELS[1:]
        }

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_SCHEMA,
            "valid_target_count": self.valid_target_count,
            "factual_loss": self.factual_loss,
            "cells": [cell.as_dict() for cell in self.cells],
            "posterior_margins_at_factual_control": self.posterior_margins_at_factual_control,
            "control_margins_at_factual_posterior": self.control_margins_at_factual_posterior,
            "interactions": self.interactions,
        }


def run_native_behavior_posterior_control_forwards(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    factual_state: NativePosteriorState,
    peer_state: NativePosteriorState,
    request: NativePredictionRequest,
    prediction_controls: ExecutedControlBatch,
    peer_prediction_controls: ExecutedControlBatch,
) -> NativeBehaviorPosteriorControlPredictions:
    """Run the exact row-only 3x3 posterior/control intervention."""

    if not isinstance(policy, nn.Module) or not policy.training:
        raise ValueError("behavior factorial requires the training-mode shared host")
    if not isinstance(graph, LingBotNativeGraph):
        raise TypeError("behavior factorial requires the installed native graph")
    if not isinstance(request, NativePredictionRequest):
        raise TypeError("behavior factorial requires a typed prediction request")
    if (
        request.source is not PredictionSource.PRIOR
        or request.evidence is not PredictionEvidence.FUTURE
    ):
        raise ValueError("behavior factorial requires a prior-row future request")
    if not isinstance(factual_state, NativePosteriorState):
        raise TypeError("factual posterior must use NativePosteriorState")
    if not isinstance(peer_state, NativePosteriorState):
        raise TypeError("peer posterior must use NativePosteriorState")

    _validate_control_pair(
        prediction_controls,
        reference=prediction_controls,
        name="factual future controls",
    )
    _validate_control_pair(
        peer_prediction_controls,
        reference=prediction_controls,
        name="peer future controls",
    )
    state_valid = torch.ones(
        factual_state.batch_size,
        dtype=torch.bool,
        device=factual_state.rows.device,
    )
    _validate_state_pair(
        factual_state,
        state_valid,
        reference=factual_state,
        name="factual posterior",
    )
    _validate_state_pair(
        peer_state,
        state_valid,
        reference=factual_state,
        name="peer posterior",
    )
    if (
        prediction_controls.batch_size != factual_state.batch_size
        or peer_prediction_controls.batch_size != factual_state.batch_size
        or request.batch_size != factual_state.batch_size
    ):
        raise ValueError("behavior factorial inputs have different batch sizes")
    if (
        prediction_controls.values.device != factual_state.rows.device
        or prediction_controls.values.dtype != factual_state.rows.dtype
        or request.addresses.device != factual_state.rows.device
        or request.addresses.dtype != factual_state.rows.dtype
    ):
        raise ValueError("behavior factorial inputs differ in device or compute dtype")

    states = {
        "factual": NativePosteriorState(factual_state.rows.detach().clone()),
        "zero": NativePosteriorState(torch.zeros_like(factual_state.rows)),
        "batch_shift": NativePosteriorState(peer_state.rows.detach().clone()),
    }
    controls = {
        "factual": prediction_controls,
        "zero": zero_executed_control(prediction_controls),
        "batch_shift": peer_prediction_controls,
    }
    rng_devices = [factual_state.rows.device] if factual_state.rows.device.type == "cuda" else []
    stepper = LingBotNativePriorStepper(policy, graph)

    def run_cell(posterior_level: str, control_level: str) -> torch.Tensor:
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            _next_state, prediction = stepper.step_with_prediction(
                states[posterior_level],
                controls[control_level],
                request,
                target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
            )
        return prediction.detach().clone()

    cells = tuple(
        NativeBehaviorPosteriorControlCell(
            posterior_level=posterior_level,
            control_level=control_level,
            prediction=run_cell(posterior_level, control_level),
        )
        for posterior_level, control_level in BEHAVIOR_POSTERIOR_CONTROL_FACTORIAL_CELLS
    )
    factual_repeat = run_cell("factual", "factual")
    return NativeBehaviorPosteriorControlPredictions(
        request=request,
        cells=cells,
        factual_repeat=factual_repeat,
    )


def behavior_posterior_control_diagnostics(
    predictions: NativeBehaviorPosteriorControlPredictions,
    *,
    target: NativePredictiveTarget,
    assignment: SequenceAssignment,
    row_binding_valid: torch.Tensor,
    loss_power: float = 1.0,
) -> NativeBehaviorPosteriorControlDiagnostics:
    """Score all nine cells against one unchanged target and assignment."""

    if not isinstance(predictions, NativeBehaviorPosteriorControlPredictions):
        raise TypeError("behavior factorial diagnostics require typed predictions")
    if (
        row_binding_valid.shape != assignment.row_to_track.shape
        or row_binding_valid.dtype != torch.bool
        or row_binding_valid.device != assignment.row_to_track.device
    ):
        raise ValueError("behavior factorial row-binding validity must be boolean [batch,row]")
    factual_prediction = predictions.prediction_for("factual", "factual")
    factual_term = native_predictive_term(
        prediction=factual_prediction,
        request=predictions.request,
        target=target,
        assignment=assignment,
        row_binding_valid=row_binding_valid,
        weight=1.0,
        loss_power=loss_power,
    )
    valid_count = int(factual_term.valid.sum().item())
    if valid_count <= 0:
        raise ValueError("behavior factorial diagnostics require a valid target")
    factual_loss = float(factual_term.normalized().item())
    normalized_factual = F.layer_norm(
        factual_prediction.float(),
        (factual_prediction.shape[-1],),
    )
    prediction_valid = factual_term.valid.unsqueeze(-1).expand_as(normalized_factual)
    denominator = prediction_valid.sum().clamp_min(1)

    diagnostics: list[NativeBehaviorPosteriorControlCellDiagnostics] = []
    for cell in predictions.cells:
        term = native_predictive_term(
            prediction=cell.prediction,
            request=predictions.request,
            target=target,
            assignment=assignment,
            row_binding_valid=row_binding_valid,
            weight=1.0,
            loss_power=loss_power,
        )
        if not torch.equal(term.valid, factual_term.valid):
            raise RuntimeError("a behavior factorial intervention changed target validity")
        loss = float(term.normalized().item())
        normalized = F.layer_norm(cell.prediction.float(), (cell.prediction.shape[-1],))
        distance = (normalized - normalized_factual).abs().masked_fill(
            ~prediction_valid,
            0,
        ).sum() / denominator.to(normalized.dtype)
        diagnostics.append(
            NativeBehaviorPosteriorControlCellDiagnostics(
                posterior_level=cell.posterior_level,
                control_level=cell.control_level,
                loss=loss,
                loss_margin_over_factual=loss - factual_loss,
                normalized_prediction_l1=float(distance.item()),
            )
        )
    return NativeBehaviorPosteriorControlDiagnostics(
        valid_target_count=valid_count,
        factual_loss=factual_loss,
        cells=tuple(diagnostics),
    )


def run_native_behavior_causal_probe(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_state: NativePosteriorState | None,
    previous_state_valid: torch.Tensor | None,
    request: NativePredictionRequest,
    prediction_controls: ExecutedControlBatch,
    peer_prediction_controls: ExecutedControlBatch,
    modalities: NativeModalityBatch | None = None,
) -> NativeBehaviorCausalProbe:
    """Run one released-policy intervention suite before any optimizer update."""

    if (
        not isinstance(graph, LingBotNativeGraph)
        or request.source is not PredictionSource.PRIOR
        or request.evidence is not PredictionEvidence.FUTURE
        or request.query_count != 1
    ):
        raise ValueError("behavior causal probe requires one prior-row future query")
    _validate_control_pair(
        peer_prediction_controls,
        reference=prediction_controls,
        name="peer future controls",
    )
    variants = {
        "factual": prediction_controls,
        "peer_replace_shuffle": peer_prediction_controls,
        "reverse": reverse_executed_control(prediction_controls),
        "zero": zero_executed_control(prediction_controls),
    }
    rng_devices = [controls.values.device] if controls.values.device.type == "cuda" else []
    predictions: dict[str, torch.Tensor] = {}
    with torch.random.fork_rng(devices=rng_devices):
        primary = run_native_policy_diagnostic_forward(
            policy,
            model_inputs=model_inputs,
            context=LingBotNativeContext(
                controls=controls,
                previous_state=previous_state,
                previous_state_valid=previous_state_valid,
                modalities=modalities,
            ),
        )
    if primary.context.prediction_request is not None:
        raise RuntimeError("behavior causal primary forward was not a cold deploy call")
    current_state = primary.context.posterior_state
    if current_state is None:
        raise RuntimeError("behavior causal primary forward omitted its posterior state")
    factual_deploy = _deploy_tensor_snapshot(primary)
    stepper = LingBotNativePriorStepper(policy, graph)
    for name, future_controls in variants.items():
        with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
            _next_state, prediction = stepper.step_with_prediction(
                current_state,
                future_controls,
                request,
                target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
            )
        predictions[name] = prediction.detach().clone()
    after_auxiliary = _deploy_tensor_snapshot(primary)
    if set(after_auxiliary) != set(factual_deploy):
        raise RuntimeError("behavior auxiliary forward mutated the deploy tensor schema")
    for key, factual_value in factual_deploy.items():
        if not torch.equal(after_auxiliary[key], factual_value):
            raise RuntimeError(f"behavior auxiliary forward mutated deploy tensor {key!r}")
    with torch.random.fork_rng(devices=rng_devices):
        rerun_primary = run_native_policy_diagnostic_forward(
            policy,
            model_inputs=model_inputs,
            context=LingBotNativeContext(
                controls=controls,
                previous_state=previous_state,
                previous_state_valid=previous_state_valid,
                modalities=modalities,
            ),
        )
    rerun_deploy = _deploy_tensor_snapshot(rerun_primary)
    if set(rerun_deploy) != set(factual_deploy):
        raise RuntimeError("behavior auxiliary pass changed fresh deploy tensor schema")
    for key, factual_value in factual_deploy.items():
        if not torch.equal(rerun_deploy[key], factual_value):
            raise RuntimeError(f"behavior auxiliary pass changed fresh deploy tensor {key!r}")
    factual_prediction = predictions.pop("factual")
    changed = tuple(
        (name, not torch.equal(value, factual_prediction))
        for name, value in sorted(predictions.items())
    )
    return NativeBehaviorCausalProbe(
        horizon=int(request.horizons[request.valid].unique().item()),
        deploy_tensor_count=len(factual_deploy),
        intervention_prediction_changed=changed,
    )


def zero_current_observation(
    model_inputs: Mapping[str, Any],
    modalities: NativeModalityBatch | None,
) -> tuple[dict[str, Any], NativeModalityBatch | None]:
    """Zero current physical observations while retaining their public geometry."""

    images = model_inputs.get("images")
    if not isinstance(images, torch.Tensor) or not images.is_floating_point():
        raise ValueError("current-observation intervention requires floating model input images")
    state = model_inputs.get("state")
    if not isinstance(state, torch.Tensor) or not state.is_floating_point():
        raise ValueError("current-observation intervention requires floating robot state")
    if state.shape[0] != images.shape[0] or state.device != images.device:
        raise ValueError("current images and robot state must share batch and device")
    changed = dict(model_inputs)
    changed["images"] = torch.zeros_like(images)
    changed["state"] = torch.zeros_like(state)
    if modalities is None:
        return changed, None
    return changed, NativeModalityBatch(
        tuple(
            NativeModalityStream(
                name=stream.name,
                tokens=torch.zeros_like(stream.tokens),
                valid=stream.valid,
            )
            for stream in modalities.streams
        )
    )


@dataclass(frozen=True, slots=True)
class NativeCorrectionCounterfactualPredictions:
    """Detached outputs from the exact host under frozen interventions."""

    request: NativePredictionRequest
    factual: torch.Tensor
    interventions: tuple[tuple[str, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if (
            self.request.source is not PredictionSource.PRIOR
            or self.request.evidence is not PredictionEvidence.CURRENT_CORRECTION
        ):
            raise ValueError("correction probes require a prior-current request")
        if self.factual.ndim != 4 or self.factual.shape[0] != self.request.batch_size:
            raise ValueError("factual correction output must be [batch,rows,queries,width]")
        if self.factual.shape[2] != self.request.query_count or self.factual.shape[-1] < 2:
            raise ValueError("factual correction output differs from its request")
        names = tuple(name for name, _value in self.interventions)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("counterfactual interventions must be sorted and unique")
        if not set(names) >= _REQUIRED_INTERVENTIONS or not set(names) <= _INTERVENTION_NAMES:
            raise ValueError("counterfactual interventions differ from the frozen probe set")
        tensors = (self.factual, *(value for _name, value in self.interventions))
        if any(
            value.shape != self.factual.shape
            or value.device != self.factual.device
            or value.dtype != self.factual.dtype
            for value in tensors
        ):
            raise ValueError("counterfactual prediction tensors must share shape and placement")
        if any(
            value.requires_grad or value.grad_fn is not None or not torch.isfinite(value).all()
            for value in tensors
        ):
            raise ValueError("counterfactual predictions must be finite and detached")

    def as_mapping(self) -> dict[str, torch.Tensor]:
        return {"factual": self.factual, **dict(self.interventions)}


def run_native_correction_counterfactual_forwards(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    previous_state: NativePosteriorState,
    previous_state_valid: torch.Tensor,
    request: NativePredictionRequest,
    modalities: NativeModalityBatch | None = None,
    wrong_batch_state: NativePosteriorState | None = None,
    wrong_batch_state_valid: torch.Tensor | None = None,
    wrong_time_state: NativePosteriorState | None = None,
    wrong_time_state_valid: torch.Tensor | None = None,
    wrong_controls: ExecutedControlBatch | None = None,
) -> NativeCorrectionCounterfactualPredictions:
    """Execute sparse interventions on the exact observation-prefix host."""

    if not isinstance(policy, nn.Module) or not policy.training:
        raise ValueError("counterfactual probes require the training-mode shared host")
    if not isinstance(request, NativePredictionRequest):
        raise TypeError("counterfactual probes require a typed prediction request")
    if (
        request.source is not PredictionSource.PRIOR
        or request.evidence is not PredictionEvidence.CURRENT_CORRECTION
    ):
        raise ValueError("counterfactual probes require prior-current correction")
    _validate_state_pair(
        previous_state,
        previous_state_valid,
        reference=previous_state,
        name="factual",
    )
    _validate_control_pair(controls, reference=controls, name="factual controls")
    if request.batch_size != previous_state.batch_size or controls.batch_size != request.batch_size:
        raise ValueError("correction request, state and controls have different batches")
    if request.route_ids.device != previous_state.rows.device:
        raise ValueError("correction request and state must share one device")

    supplied_wrong_batch = (wrong_batch_state is not None, wrong_batch_state_valid is not None)
    if supplied_wrong_batch[0] != supplied_wrong_batch[1]:
        raise ValueError("wrong-batch state and validity must be supplied together")
    supplied_wrong_time = (wrong_time_state is not None, wrong_time_state_valid is not None)
    if supplied_wrong_time[0] != supplied_wrong_time[1]:
        raise ValueError("wrong-time state and validity must be supplied together")
    if wrong_batch_state is not None and wrong_batch_state_valid is not None:
        _validate_state_pair(
            wrong_batch_state,
            wrong_batch_state_valid,
            reference=previous_state,
            name="wrong-batch",
        )
    if wrong_time_state is not None and wrong_time_state_valid is not None:
        _validate_state_pair(
            wrong_time_state,
            wrong_time_state_valid,
            reference=previous_state,
            name="wrong-time",
        )
    if wrong_controls is not None:
        _validate_control_pair(wrong_controls, reference=controls, name="wrong controls")

    observation_inputs, observation_modalities = zero_current_observation(
        model_inputs,
        modalities,
    )
    variants: dict[
        str,
        tuple[
            Mapping[str, Any],
            ExecutedControlBatch,
            NativePosteriorState,
            torch.Tensor,
            NativeModalityBatch | None,
        ],
    ] = {
        "factual": (
            model_inputs,
            controls,
            previous_state,
            previous_state_valid,
            modalities,
        ),
        ZERO_SOURCE: (
            model_inputs,
            controls,
            NativePosteriorState(torch.zeros_like(previous_state.rows)),
            previous_state_valid,
            modalities,
        ),
        ZERO_CONTROL: (
            model_inputs,
            zero_executed_control(controls),
            previous_state,
            previous_state_valid,
            modalities,
        ),
        ZERO_CURRENT_OBSERVATION: (
            observation_inputs,
            controls,
            previous_state,
            previous_state_valid,
            observation_modalities,
        ),
    }
    if previous_state.capacity > 1:
        permutation = _cyclic_indices(previous_state.capacity, device=previous_state.rows.device)
        variants[ROW_SHIFT_SOURCE] = (
            model_inputs,
            controls,
            previous_state.permute_rows(permutation),
            previous_state_valid,
            modalities,
        )
    if wrong_batch_state is None and previous_state.batch_size > 1:
        batch_indices = _cyclic_indices(
            previous_state.batch_size,
            device=previous_state.rows.device,
        )
        wrong_batch_state = previous_state.index_select(batch_indices)
        wrong_batch_state_valid = previous_state_valid.index_select(0, batch_indices)
    if wrong_batch_state is not None and wrong_batch_state_valid is not None:
        variants[BATCH_SHIFT_SOURCE] = (
            model_inputs,
            controls,
            wrong_batch_state,
            wrong_batch_state_valid,
            modalities,
        )
    if wrong_time_state is not None and wrong_time_state_valid is not None:
        variants[WRONG_TIME_SOURCE] = (
            model_inputs,
            controls,
            wrong_time_state,
            wrong_time_state_valid,
            modalities,
        )
    if wrong_controls is None and controls.batch_size > 1:
        wrong_controls = batch_shift_executed_control(controls)
    if wrong_controls is not None:
        variants[BATCH_SHIFT_CONTROL] = (
            model_inputs,
            wrong_controls,
            previous_state,
            previous_state_valid,
            modalities,
        )

    outputs: dict[str, torch.Tensor] = {}
    rng_devices: list[torch.device] = (
        [previous_state.rows.device] if previous_state.rows.device.type == "cuda" else []
    )
    for name in ("factual", *sorted(set(variants) - {"factual"})):
        inputs, variant_controls, state, state_valid, variant_modalities = variants[name]
        # Every fork starts from and restores the same outer RNG state, so
        # stochastic host operations cannot masquerade as intervention effects.
        with torch.random.fork_rng(devices=rng_devices):
            result = run_native_policy_observation_diagnostic_forward(
                policy,
                model_inputs=inputs,
                context=LingBotNativeContext(
                    controls=variant_controls,
                    previous_state=state,
                    previous_state_valid=state_valid,
                    prediction_request=request,
                    modalities=variant_modalities,
                ),
            )
        try:
            prediction = result.prediction_outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
        except KeyError as error:
            raise RuntimeError(
                "correction probe omitted the registered predictive output"
            ) from error
        outputs[name] = prediction.detach().clone()

    return NativeCorrectionCounterfactualPredictions(
        request=request,
        factual=outputs.pop("factual"),
        interventions=tuple(sorted(outputs.items())),
    )


@dataclass(frozen=True, slots=True)
class NativeFutureCounterfactualPredictions:
    """Detached outputs from the exact correction-then-prior P2 route."""

    request: NativePredictionRequest
    factual: torch.Tensor
    interventions: tuple[tuple[str, torch.Tensor], ...]
    factual_context: LingBotNativeContext

    def __post_init__(self) -> None:
        if (
            self.request.source is not PredictionSource.PRIOR
            or self.request.evidence is not PredictionEvidence.FUTURE
        ):
            raise ValueError("future probes require a prior-to-future request")
        if (
            not isinstance(self.factual_context, LingBotNativeContext)
            or self.factual_context.posterior_state is None
            or self.factual_context.relation_output is None
            or not self.factual_context._finalized
        ):
            raise ValueError("future probes require their finalized factual correction context")
        if self.factual.ndim != 4 or self.factual.shape[0] != self.request.batch_size:
            raise ValueError("factual future output must be [batch,rows,queries,width]")
        if self.factual.shape[2] != self.request.query_count or self.factual.shape[-1] < 2:
            raise ValueError("factual future output differs from its request")
        names = tuple(name for name, _value in self.interventions)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("future interventions must be sorted and unique")
        if not set(names) >= _FUTURE_REQUIRED_INTERVENTIONS or not set(names) <= (
            _INTERVENTION_NAMES
        ):
            raise ValueError("future interventions differ from the frozen probe set")
        tensors = (self.factual, *(value for _name, value in self.interventions))
        if any(
            value.shape != self.factual.shape
            or value.device != self.factual.device
            or value.dtype != self.factual.dtype
            for value in tensors
        ):
            raise ValueError("future prediction tensors must share shape and placement")
        if any(
            value.requires_grad or value.grad_fn is not None or not torch.isfinite(value).all()
            for value in tensors
        ):
            raise ValueError("future counterfactual predictions must be finite and detached")

    def as_mapping(self) -> dict[str, torch.Tensor]:
        return {"factual": self.factual, **dict(self.interventions)}


def run_native_future_counterfactual_forwards(
    policy: nn.Module,
    *,
    stepper: LingBotNativePriorStepper,
    model_inputs: Mapping[str, Any],
    controls: ExecutedControlBatch,
    future_controls: tuple[ExecutedControlBatch, ...],
    previous_state: NativePosteriorState,
    previous_state_valid: torch.Tensor,
    request: NativePredictionRequest,
    modalities: NativeModalityBatch | None = None,
    wrong_batch_state: NativePosteriorState | None = None,
    wrong_batch_state_valid: torch.Tensor | None = None,
    wrong_time_state: NativePosteriorState | None = None,
    wrong_time_state_valid: torch.Tensor | None = None,
    neutral_state: NativePosteriorState | None = None,
    neutral_state_valid: torch.Tensor | None = None,
    wrong_future_controls: tuple[ExecutedControlBatch, ...] | None = None,
) -> NativeFutureCounterfactualPredictions:
    """Rerun the exact P2 correction and controlled prior under sparse interventions."""

    if not isinstance(policy, nn.Module) or not policy.training:
        raise ValueError("future counterfactual probes require the training-mode shared host")
    if not isinstance(stepper, LingBotNativePriorStepper):
        raise TypeError("future counterfactual probes require the native prior stepper")
    if (
        request.source is not PredictionSource.PRIOR
        or request.evidence is not PredictionEvidence.FUTURE
    ):
        raise ValueError("future counterfactual probes require a prior-to-future request")
    _validate_state_pair(
        previous_state,
        previous_state_valid,
        reference=previous_state,
        name="factual",
    )
    _validate_control_pair(controls, reference=controls, name="factual controls")
    if request.batch_size != previous_state.batch_size or controls.batch_size != request.batch_size:
        raise ValueError("future request, state and current controls have different batches")
    if request.route_ids.device != previous_state.rows.device:
        raise ValueError("future request and state must share one device")
    horizons = request.horizons[request.valid].detach().cpu().tolist()
    if not horizons or len(set(int(value) for value in horizons)) != 1:
        raise ValueError("future counterfactual request must use one valid shared horizon")
    horizon = int(horizons[0])
    if len(future_controls) != horizon:
        raise ValueError("future counterfactual controls must exactly cover the request horizon")
    for index, control in enumerate(future_controls):
        _validate_control_pair(
            control,
            reference=future_controls[0],
            name=f"future control {index}",
        )
        if control.batch_size != request.batch_size:
            raise ValueError("future control and request batches differ")

    supplied_wrong_batch = (wrong_batch_state is not None, wrong_batch_state_valid is not None)
    if supplied_wrong_batch[0] != supplied_wrong_batch[1]:
        raise ValueError("wrong-batch state and validity must be supplied together")
    supplied_wrong_time = (wrong_time_state is not None, wrong_time_state_valid is not None)
    if supplied_wrong_time[0] != supplied_wrong_time[1]:
        raise ValueError("wrong-time state and validity must be supplied together")
    if wrong_batch_state is not None and wrong_batch_state_valid is not None:
        _validate_state_pair(
            wrong_batch_state,
            wrong_batch_state_valid,
            reference=previous_state,
            name="wrong-batch",
        )
    if wrong_time_state is not None and wrong_time_state_valid is not None:
        _validate_state_pair(
            wrong_time_state,
            wrong_time_state_valid,
            reference=previous_state,
            name="wrong-time",
        )
    supplied_neutral = (neutral_state is not None, neutral_state_valid is not None)
    if supplied_neutral[0] != supplied_neutral[1]:
        raise ValueError("neutral control state and validity must be supplied together")
    if neutral_state is not None and neutral_state_valid is not None:
        _validate_state_pair(
            neutral_state,
            neutral_state_valid,
            reference=previous_state,
            name="matched-noise",
        )
    if wrong_future_controls is not None:
        if len(wrong_future_controls) != len(future_controls):
            raise ValueError("wrong future controls differ from the factual horizon")
        for index, (wrong, factual) in enumerate(
            zip(wrong_future_controls, future_controls, strict=True)
        ):
            _validate_control_pair(wrong, reference=factual, name=f"wrong future control {index}")

    observation_inputs, observation_modalities = zero_current_observation(
        model_inputs,
        modalities,
    )
    zero_future_controls = tuple(zero_executed_control(value) for value in future_controls)
    variants: dict[
        str,
        tuple[
            Mapping[str, Any],
            ExecutedControlBatch,
            NativePosteriorState,
            torch.Tensor,
            NativeModalityBatch | None,
            tuple[ExecutedControlBatch, ...],
        ],
    ] = {
        "factual": (
            model_inputs,
            controls,
            previous_state,
            previous_state_valid,
            modalities,
            future_controls,
        ),
        ABSENT_SOURCE: (
            model_inputs,
            controls,
            NativePosteriorState(torch.zeros_like(previous_state.rows)),
            torch.zeros_like(previous_state_valid),
            modalities,
            future_controls,
        ),
        ZERO_SOURCE: (
            model_inputs,
            controls,
            NativePosteriorState(torch.zeros_like(previous_state.rows)),
            previous_state_valid,
            modalities,
            future_controls,
        ),
        ZERO_CONTROL: (
            model_inputs,
            controls,
            previous_state,
            previous_state_valid,
            modalities,
            zero_future_controls,
        ),
        ZERO_CURRENT_OBSERVATION: (
            observation_inputs,
            controls,
            previous_state,
            previous_state_valid,
            observation_modalities,
            future_controls,
        ),
    }
    if previous_state.capacity > 1:
        permutation = _cyclic_indices(previous_state.capacity, device=previous_state.rows.device)
        variants[ROW_SHIFT_SOURCE] = (
            model_inputs,
            controls,
            previous_state.permute_rows(permutation),
            previous_state_valid,
            modalities,
            future_controls,
        )
    if wrong_batch_state is None and previous_state.batch_size > 1:
        batch_indices = _cyclic_indices(
            previous_state.batch_size,
            device=previous_state.rows.device,
        )
        wrong_batch_state = previous_state.index_select(batch_indices)
        wrong_batch_state_valid = previous_state_valid.index_select(0, batch_indices)
    if wrong_batch_state is not None and wrong_batch_state_valid is not None:
        variants[BATCH_SHIFT_SOURCE] = (
            model_inputs,
            controls,
            wrong_batch_state,
            wrong_batch_state_valid,
            modalities,
            future_controls,
        )
    if wrong_time_state is not None and wrong_time_state_valid is not None:
        variants[WRONG_TIME_SOURCE] = (
            model_inputs,
            controls,
            wrong_time_state,
            wrong_time_state_valid,
            modalities,
            future_controls,
        )
    if neutral_state is not None and neutral_state_valid is not None:
        variants[MATCHED_NOISE_SOURCE] = (
            model_inputs,
            controls,
            neutral_state,
            neutral_state_valid,
            modalities,
            future_controls,
        )
    if wrong_future_controls is None and request.batch_size > 1:
        wrong_future_controls = tuple(
            batch_shift_executed_control(value) for value in future_controls
        )
    if wrong_future_controls is not None:
        variants[BATCH_SHIFT_CONTROL] = (
            model_inputs,
            controls,
            previous_state,
            previous_state_valid,
            modalities,
            wrong_future_controls,
        )

    outputs: dict[str, torch.Tensor] = {}
    factual_context: LingBotNativeContext | None = None
    rng_devices: list[torch.device] = (
        [previous_state.rows.device] if previous_state.rows.device.type == "cuda" else []
    )
    for name in ("factual", *sorted(set(variants) - {"factual"})):
        inputs, current_control, state, state_valid, variant_modalities, rollout_controls = (
            variants[name]
        )
        with torch.random.fork_rng(devices=rng_devices), torch.no_grad():
            context = run_native_policy_observation_diagnostic_forward(
                policy,
                model_inputs=inputs,
                context=LingBotNativeContext(
                    controls=current_control,
                    previous_state=state,
                    previous_state_valid=state_valid,
                    modalities=variant_modalities,
                    supervise_intermediate_relations=bool(
                        stepper.graph.config.relation_supervision_layers
                    ),
                ),
            )
            posterior = context.posterior_state
            if posterior is None:
                raise RuntimeError("future counterfactual correction produced no posterior state")
            rollout = rollout_native_prior_prediction(
                stepper,
                posterior,
                rollout_controls,
                request=request,
                target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
            )
        outputs[name] = rollout.prediction.detach().clone()
        if name == "factual":
            factual_context = context

    if factual_context is None:
        raise RuntimeError("future counterfactual probe lost its factual context")

    return NativeFutureCounterfactualPredictions(
        request=request,
        factual=outputs.pop("factual"),
        interventions=tuple(sorted(outputs.items())),
        factual_context=factual_context,
    )


@dataclass(frozen=True, slots=True)
class PredictiveInterventionDiagnostics:
    name: str
    loss: float
    loss_margin_over_factual: float
    normalized_prediction_l1: float

    def __post_init__(self) -> None:
        if self.name not in _INTERVENTION_NAMES:
            raise ValueError("unknown predictive intervention")
        _finite(self.loss, name="counterfactual loss", non_negative=True)
        _finite(self.loss_margin_over_factual, name="counterfactual loss margin")
        _finite(
            self.normalized_prediction_l1,
            name="counterfactual prediction distance",
            non_negative=True,
        )

    def as_dict(self) -> dict[str, str | float]:
        return {
            "name": self.name,
            "loss": self.loss,
            "loss_margin_over_factual": self.loss_margin_over_factual,
            "normalized_prediction_l1": self.normalized_prediction_l1,
        }


@dataclass(frozen=True, slots=True)
class PredictiveCorrectionCounterfactualDiagnostics:
    valid_target_count: int
    factual_loss: float
    interventions: tuple[PredictiveInterventionDiagnostics, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.valid_target_count, name="valid target count")
        _finite(self.factual_loss, name="factual correction loss", non_negative=True)
        names = tuple(value.name for value in self.interventions)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("predictive intervention diagnostics must be sorted and unique")
        if not set(names) >= _REQUIRED_INTERVENTIONS:
            raise ValueError("predictive diagnostics omit a required counterfactual")
        for value in self.interventions:
            if not math.isclose(
                value.loss_margin_over_factual,
                value.loss - self.factual_loss,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError("predictive intervention loss margin is inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PREDICTIVE_CORRECTION_COUNTERFACTUAL_SCHEMA,
            "valid_target_count": self.valid_target_count,
            "factual_loss": self.factual_loss,
            "interventions": [value.as_dict() for value in self.interventions],
        }


@dataclass(frozen=True, slots=True)
class PredictiveFutureCounterfactualDiagnostics:
    """Loss-side evidence that a future prediction uses its causal inputs."""

    valid_target_count: int
    factual_loss: float
    interventions: tuple[PredictiveInterventionDiagnostics, ...]

    def __post_init__(self) -> None:
        _positive_integer(self.valid_target_count, name="valid future target count")
        _finite(self.factual_loss, name="factual future loss", non_negative=True)
        names = tuple(value.name for value in self.interventions)
        if names != tuple(sorted(names)) or len(set(names)) != len(names):
            raise ValueError("future intervention diagnostics must be sorted and unique")
        if not set(names) >= _FUTURE_REQUIRED_INTERVENTIONS:
            raise ValueError("future diagnostics omit a required counterfactual")
        for value in self.interventions:
            if not math.isclose(
                value.loss_margin_over_factual,
                value.loss - self.factual_loss,
                rel_tol=1e-6,
                abs_tol=1e-8,
            ):
                raise ValueError("future intervention loss margin is inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PREDICTIVE_FUTURE_COUNTERFACTUAL_SCHEMA,
            "valid_target_count": self.valid_target_count,
            "factual_loss": self.factual_loss,
            "interventions": [value.as_dict() for value in self.interventions],
        }


def _score_predictive_counterfactuals(
    *,
    request: NativePredictionRequest,
    factual: torch.Tensor,
    interventions: tuple[tuple[str, torch.Tensor], ...],
    target: NativePredictiveTarget,
    assignment: PredictiveRowAssignment,
    row_binding_valid: torch.Tensor,
    loss_power: float,
) -> tuple[int, float, tuple[PredictiveInterventionDiagnostics, ...]]:
    if (
        row_binding_valid.shape != assignment.row_to_track.shape
        or row_binding_valid.dtype != torch.bool
        or row_binding_valid.device != assignment.row_to_track.device
    ):
        raise ValueError("counterfactual row-binding validity must be boolean [batch,row]")
    factual_term = native_predictive_term(
        prediction=factual,
        request=request,
        target=target,
        assignment=assignment,
        row_binding_valid=row_binding_valid,
        weight=1.0,
        loss_power=loss_power,
    )
    valid_count = int(factual_term.valid.sum().item())
    if valid_count <= 0:
        raise ValueError("counterfactual diagnostics require a valid target")
    factual_loss = float(factual_term.normalized().item())
    factual_normalized = F.layer_norm(factual.float(), (factual.shape[-1],))
    prediction_valid = request.valid[:, None, :, None].expand_as(factual_normalized)
    denominator = prediction_valid.sum().clamp_min(1)

    diagnostics: list[PredictiveInterventionDiagnostics] = []
    for name, prediction in interventions:
        term = native_predictive_term(
            prediction=prediction,
            request=request,
            target=target,
            assignment=assignment,
            row_binding_valid=row_binding_valid,
            weight=1.0,
            loss_power=loss_power,
        )
        if not torch.equal(term.valid, factual_term.valid):
            raise RuntimeError("a counterfactual changed loss-side target validity")
        loss = float(term.normalized().item())
        normalized = F.layer_norm(prediction.float(), (prediction.shape[-1],))
        distance = (normalized - factual_normalized).abs().masked_fill(
            ~prediction_valid, 0
        ).sum() / denominator.to(normalized.dtype)
        diagnostics.append(
            PredictiveInterventionDiagnostics(
                name=name,
                loss=loss,
                loss_margin_over_factual=loss - factual_loss,
                normalized_prediction_l1=float(distance.item()),
            )
        )
    return valid_count, factual_loss, tuple(diagnostics)


def predictive_correction_counterfactual_diagnostics(
    predictions: NativeCorrectionCounterfactualPredictions,
    *,
    target: NativePredictiveTarget,
    assignment: SequenceAssignment,
    row_binding_valid: torch.Tensor,
    loss_power: float = 1.0,
) -> PredictiveCorrectionCounterfactualDiagnostics:
    """Score all interventions against one unchanged target and assignment."""

    valid_count, factual_loss, diagnostics = _score_predictive_counterfactuals(
        request=predictions.request,
        factual=predictions.factual,
        interventions=predictions.interventions,
        target=target,
        assignment=assignment,
        row_binding_valid=row_binding_valid,
        loss_power=loss_power,
    )
    return PredictiveCorrectionCounterfactualDiagnostics(
        valid_target_count=valid_count,
        factual_loss=factual_loss,
        interventions=diagnostics,
    )


def predictive_future_counterfactual_diagnostics(
    predictions: NativeFutureCounterfactualPredictions,
    *,
    target: NativePredictiveTarget,
    assignment: PredictiveRowAssignment,
    row_binding_valid: torch.Tensor,
    loss_power: float = 1.0,
) -> PredictiveFutureCounterfactualDiagnostics:
    """Score future-route interventions against one frozen target and row gauge."""

    valid_count, factual_loss, diagnostics = _score_predictive_counterfactuals(
        request=predictions.request,
        factual=predictions.factual,
        interventions=predictions.interventions,
        target=target,
        assignment=assignment,
        row_binding_valid=row_binding_valid,
        loss_power=loss_power,
    )
    return PredictiveFutureCounterfactualDiagnostics(
        valid_target_count=valid_count,
        factual_loss=factual_loss,
        interventions=diagnostics,
    )


def predictive_correction_counterfactual_from_mapping(
    value: object,
) -> PredictiveCorrectionCounterfactualDiagnostics:
    """Parse and recompute all scalar invariants in a persisted probe report."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "valid_target_count",
        "factual_loss",
        "interventions",
    }:
        raise ValueError("predictive counterfactual fields differ from schema")
    if value["schema"] != PREDICTIVE_CORRECTION_COUNTERFACTUAL_SCHEMA:
        raise ValueError("predictive counterfactual schema is incompatible")
    raw_interventions = value["interventions"]
    if not isinstance(raw_interventions, list):
        raise ValueError("predictive counterfactual interventions must be a list")
    interventions: list[PredictiveInterventionDiagnostics] = []
    for raw in raw_interventions:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "loss",
            "loss_margin_over_factual",
            "normalized_prediction_l1",
        }:
            raise ValueError("predictive intervention fields differ from schema")
        name = raw["name"]
        if not isinstance(name, str):
            raise ValueError("predictive intervention name must be text")
        interventions.append(
            PredictiveInterventionDiagnostics(
                name=name,
                loss=_finite(raw["loss"], name="counterfactual loss", non_negative=True),
                loss_margin_over_factual=_finite(
                    raw["loss_margin_over_factual"],
                    name="counterfactual loss margin",
                ),
                normalized_prediction_l1=_finite(
                    raw["normalized_prediction_l1"],
                    name="counterfactual prediction distance",
                    non_negative=True,
                ),
            )
        )
    return PredictiveCorrectionCounterfactualDiagnostics(
        valid_target_count=_positive_integer(
            value["valid_target_count"],
            name="valid target count",
        ),
        factual_loss=_finite(
            value["factual_loss"],
            name="factual correction loss",
            non_negative=True,
        ),
        interventions=tuple(interventions),
    )


@dataclass(frozen=True, slots=True)
class PredictiveFixedBatchArmDiagnostics:
    name: str
    curve_point_count: int
    loss_curve: tuple[float, ...]
    initial_loss: float
    final_loss: float
    best_loss: float
    mean_loss: float
    normalized_auc: float
    absolute_reduction: float

    def __post_init__(self) -> None:
        if self.name not in PREDICTIVE_FIXED_BATCH_ARMS:
            raise ValueError("unknown predictive fixed-batch arm")
        if (
            _positive_integer(
                self.curve_point_count,
                name="fixed-batch curve-point count",
            )
            < 2
        ):
            raise ValueError("fixed-batch fit requires at least two curve points")
        if not isinstance(self.loss_curve, tuple) or len(self.loss_curve) != self.curve_point_count:
            raise ValueError("fixed-batch loss curve differs from its curve-point count")
        curve = tuple(
            _finite(value, name="fixed-batch curve loss", non_negative=True)
            for value in self.loss_curve
        )
        for name, value in (
            ("initial", self.initial_loss),
            ("final", self.final_loss),
            ("best", self.best_loss),
            ("mean", self.mean_loss),
            ("AUC", self.normalized_auc),
        ):
            _finite(value, name=f"fixed-batch {name} loss", non_negative=True)
        _finite(self.absolute_reduction, name="fixed-batch loss reduction")
        derived = {
            "initial": curve[0],
            "final": curve[-1],
            "best": min(curve),
            "mean": sum(curve) / len(curve),
            "AUC": sum(
                (left + right) / 2 for left, right in zip(curve[:-1], curve[1:], strict=True)
            )
            / (len(curve) - 1),
            "reduction": curve[0] - curve[-1],
        }
        reported = {
            "initial": self.initial_loss,
            "final": self.final_loss,
            "best": self.best_loss,
            "mean": self.mean_loss,
            "AUC": self.normalized_auc,
            "reduction": self.absolute_reduction,
        }
        if any(
            not math.isclose(
                reported[name],
                expected,
                rel_tol=1e-6,
                abs_tol=1e-8,
            )
            for name, expected in derived.items()
        ):
            raise ValueError("fixed-batch curve summaries are inconsistent")

    def as_dict(self) -> dict[str, str | int | float | list[float]]:
        return {
            "name": self.name,
            "curve_point_count": self.curve_point_count,
            "loss_curve": list(self.loss_curve),
            "initial_loss": self.initial_loss,
            "final_loss": self.final_loss,
            "best_loss": self.best_loss,
            "mean_loss": self.mean_loss,
            "normalized_auc": self.normalized_auc,
            "absolute_reduction": self.absolute_reduction,
        }


@dataclass(frozen=True, slots=True)
class PredictiveFixedBatchFitDiagnostics:
    arms: tuple[PredictiveFixedBatchArmDiagnostics, ...]
    full_host_final_advantage_over_native_graph: float
    full_host_final_advantage_over_readout: float
    full_host_final_advantage_over_shuffled_target: float

    def __post_init__(self) -> None:
        names = tuple(value.name for value in self.arms)
        if names != PREDICTIVE_FIXED_BATCH_ARMS:
            raise ValueError("fixed-batch arms must use the frozen order")
        if len({value.curve_point_count for value in self.arms}) != 1:
            raise ValueError("fixed-batch arms must use the same curve-point budget")
        by_name = {value.name: value for value in self.arms}
        expected_native_graph = (
            by_name["native_graph_only"].final_loss - by_name["full_host"].final_loss
        )
        expected_readout = by_name["readout_only"].final_loss - by_name["full_host"].final_loss
        expected_shuffle = by_name["shuffled_target"].final_loss - by_name["full_host"].final_loss
        if any(
            not math.isclose(reported, expected, rel_tol=1e-6, abs_tol=1e-8)
            for reported, expected in (
                (
                    self.full_host_final_advantage_over_native_graph,
                    expected_native_graph,
                ),
                (self.full_host_final_advantage_over_readout, expected_readout),
                (
                    self.full_host_final_advantage_over_shuffled_target,
                    expected_shuffle,
                ),
            )
        ):
            raise ValueError("fixed-batch fit advantages are inconsistent")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": PREDICTIVE_FIXED_BATCH_FIT_SCHEMA,
            "arms": [value.as_dict() for value in self.arms],
            "full_host_final_advantage_over_native_graph": (
                self.full_host_final_advantage_over_native_graph
            ),
            "full_host_final_advantage_over_readout": (self.full_host_final_advantage_over_readout),
            "full_host_final_advantage_over_shuffled_target": (
                self.full_host_final_advantage_over_shuffled_target
            ),
        }


def predictive_fixed_batch_fit_diagnostics(
    loss_curves: Mapping[str, Sequence[float] | torch.Tensor],
) -> PredictiveFixedBatchFitDiagnostics:
    """Summarize equal-budget fit traces without inventing a pass threshold."""

    if not isinstance(loss_curves, Mapping) or tuple(loss_curves) != PREDICTIVE_FIXED_BATCH_ARMS:
        raise ValueError("fixed-batch loss curves must use the frozen ordered arms")
    arms: list[PredictiveFixedBatchArmDiagnostics] = []
    expected_steps: int | None = None
    for name in PREDICTIVE_FIXED_BATCH_ARMS:
        raw = loss_curves[name]
        if isinstance(raw, torch.Tensor):
            if raw.ndim != 1 or raw.requires_grad or raw.grad_fn is not None:
                raise ValueError("fixed-batch loss curves must be detached rank-one tensors")
            values = tuple(float(item) for item in raw.detach().cpu().tolist())
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            values = tuple(_finite(item, name=f"{name} loss", non_negative=True) for item in raw)
        else:
            raise TypeError("fixed-batch loss curves must be sequences or tensors")
        if len(values) < 2:
            raise ValueError("fixed-batch loss curves require at least two curve points")
        if expected_steps is None:
            expected_steps = len(values)
        elif len(values) != expected_steps:
            raise ValueError("fixed-batch loss curves must use equal curve-point counts")
        normalized_auc = sum(
            (left + right) / 2 for left, right in zip(values[:-1], values[1:], strict=True)
        ) / (len(values) - 1)
        arms.append(
            PredictiveFixedBatchArmDiagnostics(
                name=name,
                curve_point_count=len(values),
                loss_curve=values,
                initial_loss=values[0],
                final_loss=values[-1],
                best_loss=min(values),
                mean_loss=sum(values) / len(values),
                normalized_auc=normalized_auc,
                absolute_reduction=values[0] - values[-1],
            )
        )
    by_name = {value.name: value for value in arms}
    return PredictiveFixedBatchFitDiagnostics(
        arms=tuple(arms),
        full_host_final_advantage_over_native_graph=(
            by_name["native_graph_only"].final_loss - by_name["full_host"].final_loss
        ),
        full_host_final_advantage_over_readout=(
            by_name["readout_only"].final_loss - by_name["full_host"].final_loss
        ),
        full_host_final_advantage_over_shuffled_target=(
            by_name["shuffled_target"].final_loss - by_name["full_host"].final_loss
        ),
    )


def predictive_fixed_batch_fit_from_mapping(
    value: object,
) -> PredictiveFixedBatchFitDiagnostics:
    """Parse a persisted fit report and recompute every derived invariant."""

    if not isinstance(value, Mapping) or set(value) != {
        "schema",
        "arms",
        "full_host_final_advantage_over_native_graph",
        "full_host_final_advantage_over_readout",
        "full_host_final_advantage_over_shuffled_target",
    }:
        raise ValueError("predictive fixed-batch fields differ from schema")
    if value["schema"] != PREDICTIVE_FIXED_BATCH_FIT_SCHEMA:
        raise ValueError("predictive fixed-batch schema is incompatible")
    raw_arms = value["arms"]
    if not isinstance(raw_arms, list):
        raise ValueError("predictive fixed-batch arms must be a list")
    arms: list[PredictiveFixedBatchArmDiagnostics] = []
    for raw in raw_arms:
        if not isinstance(raw, Mapping) or set(raw) != {
            "name",
            "curve_point_count",
            "loss_curve",
            "initial_loss",
            "final_loss",
            "best_loss",
            "mean_loss",
            "normalized_auc",
            "absolute_reduction",
        }:
            raise ValueError("predictive fixed-batch arm fields differ from schema")
        name = raw["name"]
        if not isinstance(name, str):
            raise ValueError("predictive fixed-batch arm name must be text")
        raw_curve = raw["loss_curve"]
        if not isinstance(raw_curve, list):
            raise ValueError("predictive fixed-batch loss curve must be a list")
        arms.append(
            PredictiveFixedBatchArmDiagnostics(
                name=name,
                curve_point_count=_positive_integer(
                    raw["curve_point_count"],
                    name="fixed-batch curve-point count",
                ),
                loss_curve=tuple(
                    _finite(value, name="fixed-batch curve loss", non_negative=True)
                    for value in raw_curve
                ),
                initial_loss=_finite(
                    raw["initial_loss"],
                    name="fixed-batch initial loss",
                    non_negative=True,
                ),
                final_loss=_finite(
                    raw["final_loss"],
                    name="fixed-batch final loss",
                    non_negative=True,
                ),
                best_loss=_finite(
                    raw["best_loss"],
                    name="fixed-batch best loss",
                    non_negative=True,
                ),
                mean_loss=_finite(
                    raw["mean_loss"],
                    name="fixed-batch mean loss",
                    non_negative=True,
                ),
                normalized_auc=_finite(
                    raw["normalized_auc"],
                    name="fixed-batch AUC loss",
                    non_negative=True,
                ),
                absolute_reduction=_finite(
                    raw["absolute_reduction"],
                    name="fixed-batch loss reduction",
                ),
            )
        )
    return PredictiveFixedBatchFitDiagnostics(
        arms=tuple(arms),
        full_host_final_advantage_over_native_graph=_finite(
            value["full_host_final_advantage_over_native_graph"],
            name="full-host final advantage over native graph",
        ),
        full_host_final_advantage_over_readout=_finite(
            value["full_host_final_advantage_over_readout"],
            name="full-host final advantage over readout",
        ),
        full_host_final_advantage_over_shuffled_target=_finite(
            value["full_host_final_advantage_over_shuffled_target"],
            name="full-host final advantage over shuffled target",
        ),
    )

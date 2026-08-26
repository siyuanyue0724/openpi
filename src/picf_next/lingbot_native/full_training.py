"""One leak-closed full-objective transaction for LingBot-native PICF."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import nn

from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.lingbot_native.calvin import CollatedNativeCALVINBatch
from picf_next.lingbot_native.calvin_objective import (
    NativeCALVINObjectiveResult,
    NativeStructuralLossConfig,
    compose_native_calvin_objective,
)
from picf_next.lingbot_native.calvin_supervision import TaskIdentityResolver
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.current_grid_cache import LingBotCurrentGridTargetCache
from picf_next.lingbot_native.host import (
    LingBotNativeContext,
    LingBotNativeGraph,
    LingBotNativePriorStepper,
)
from picf_next.lingbot_native.objective import NativeObjectiveConfig
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
    make_native_future_request,
)
from picf_next.lingbot_native.predictive_cache import (
    LINGBOT_PREDICTIVE_TARGET_SPACE,
    LingBotPredictiveTargetCache,
)
from picf_next.lingbot_native.predictive_objective import (
    NativePredictiveLossInput,
    NativePredictiveTarget,
)
from picf_next.lingbot_native.relations import RelationOutput
from picf_next.lingbot_native.row_binding import RowBindings
from picf_next.lingbot_native.source_mask import (
    QwenPackedPatchMask,
    QwenWholeViewOmission,
    qwen_mask_query_addresses,
)
from picf_next.lingbot_native.state import NativePersistentState, NativePosteriorState
from picf_next.lingbot_native.temporal import NativePriorPredictiveRollout
from picf_next.lingbot_native.training import (
    NativeLocalBPTTStep,
    NativePolicyForwardResult,
    run_native_local_bptt,
    run_native_omitted_image_view_training_forward,
    run_native_policy_relation_training_forward,
    run_native_policy_training_forward,
    run_native_relation_local_bptt,
    run_native_representation_window,
    run_native_source_masked_training_forward,
)


def _validate_identity_source_phase(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("native predictive identity source phase must be non-negative")


@dataclass(frozen=True, slots=True)
class NativeCorrectionBranch:
    """A prior-row prediction of the current independently cached evidence."""

    batch: CollatedNativeCALVINBatch
    request: NativePredictionRequest
    prediction: torch.Tensor
    identity_source_phase: int

    def __post_init__(self) -> None:
        if (
            self.request.source is not PredictionSource.PRIOR
            or self.request.evidence is not PredictionEvidence.CURRENT_CORRECTION
        ):
            raise ValueError("native correction branch requires prior current evidence")
        expected = (
            self.request.batch_size,
            self.prediction.shape[1],
            self.request.query_count,
        )
        if self.prediction.ndim != 4 or self.prediction.shape[:3] != expected:
            raise ValueError("native correction prediction differs from its request axes")
        if not self.prediction.requires_grad or not torch.isfinite(self.prediction).all():
            raise ValueError("native correction prediction must be finite and attached")
        _validate_identity_source_phase(self.identity_source_phase)


@dataclass(frozen=True, slots=True)
class NativeFilterPhaseBranch:
    """One ADR149 prior/posterior readout in the shared object coordinate system."""

    batch: CollatedNativeCALVINBatch
    request: NativePredictionRequest
    prediction: torch.Tensor
    identity_source_phase: int

    def __post_init__(self) -> None:
        valid_phase = (self.request.source, self.request.evidence) in {
            (PredictionSource.PRIOR, PredictionEvidence.CURRENT_PRIOR),
            (PredictionSource.POSTERIOR, PredictionEvidence.CURRENT_POSTERIOR),
        }
        if not valid_phase:
            raise ValueError("native filter branch requires an explicit current latent phase")
        expected = (
            self.request.batch_size,
            self.prediction.shape[1],
            self.request.query_count,
        )
        if self.prediction.ndim != 4 or self.prediction.shape[:3] != expected:
            raise ValueError("native filter prediction differs from its request axes")
        if not self.prediction.requires_grad or not torch.isfinite(self.prediction).all():
            raise ValueError("native filter prediction must be finite and attached")
        _validate_identity_source_phase(self.identity_source_phase)


@dataclass(frozen=True, slots=True)
class NativeFutureBranch:
    """A completed row-paired future query and its source-frame identity."""

    batch: CollatedNativeCALVINBatch
    request: NativePredictionRequest
    target_name: str
    prediction: torch.Tensor
    identity_source_phase: int

    def __post_init__(self) -> None:
        if self.request.evidence is not PredictionEvidence.FUTURE:
            raise ValueError("native future branch requires future evidence")
        if self.request.source not in (PredictionSource.POSTERIOR, PredictionSource.PRIOR):
            raise ValueError("native future branch requires posterior or prior rows")
        if not isinstance(self.target_name, str) or not self.target_name:
            raise ValueError("native future target name must be non-empty")
        if self.prediction.ndim != 4:
            raise ValueError("native future prediction must be [batch,rows,queries,width]")
        expected = (
            self.request.batch_size,
            self.prediction.shape[1],
            self.request.query_count,
        )
        if self.prediction.shape[:3] != expected:
            raise ValueError("native future prediction differs from its request axes")
        if not self.prediction.requires_grad or not torch.isfinite(self.prediction).all():
            raise ValueError("native future prediction must be finite and attached")
        _validate_identity_source_phase(self.identity_source_phase)


NativeOvershootFactory = Callable[[NativePosteriorState], NativePriorPredictiveRollout]


@dataclass(frozen=True, slots=True)
class NativeCurrentGridBranch:
    """An uncommittable current-grid query from the weight-shared masked host."""

    batch: CollatedNativeCALVINBatch
    request: NativePredictionRequest
    source_mask: QwenPackedPatchMask
    prediction: torch.Tensor
    identity_source_phase: int

    def __post_init__(self) -> None:
        if (
            self.request.source is not PredictionSource.POSTERIOR
            or self.request.evidence is not PredictionEvidence.CURRENT_RANDOM_GRID
        ):
            raise ValueError("current-grid branch requires masked posterior evidence")
        expected = (
            self.request.batch_size,
            self.prediction.shape[1],
            self.request.query_count,
        )
        if self.prediction.ndim != 4 or self.prediction.shape[:3] != expected:
            raise ValueError("current-grid prediction differs from its request axes")
        if not self.prediction.requires_grad or not torch.isfinite(self.prediction).all():
            raise ValueError("current-grid prediction must be finite and attached")
        if self.source_mask.query_count != self.request.query_count:
            raise ValueError("current-grid source mask and request query counts differ")
        _validate_identity_source_phase(self.identity_source_phase)


@dataclass(frozen=True, slots=True)
class NativeOmittedStaticBranch:
    """An uncommittable cross-view object query with the static view absent."""

    batch: CollatedNativeCALVINBatch
    request: NativePredictionRequest
    omission: QwenWholeViewOmission
    prediction: torch.Tensor
    identity_source_phase: int

    def __post_init__(self) -> None:
        if (
            self.request.source is not PredictionSource.POSTERIOR
            or self.request.evidence is not PredictionEvidence.OMITTED_MODALITY
        ):
            raise ValueError("omitted-static branch requires posterior omission evidence")
        expected = (
            self.request.batch_size,
            self.prediction.shape[1],
            self.request.query_count,
        )
        if self.prediction.ndim != 4 or self.prediction.shape[:3] != expected:
            raise ValueError("omitted-static prediction differs from its request axes")
        if not self.prediction.requires_grad or not torch.isfinite(self.prediction).all():
            raise ValueError("omitted-static prediction must be finite and attached")
        if self.omission.omitted_view_index != 0 or self.request.query_count != 1:
            raise ValueError("omitted-static branch requires one static-view query")
        _validate_identity_source_phase(self.identity_source_phase)


@dataclass(frozen=True, slots=True)
class NativeFullObjectiveStepResult:
    """All outputs from one action/predictive/structural training transaction."""

    primary: NativePolicyForwardResult
    final_relation: RelationOutput
    current_grid_branch: NativeCurrentGridBranch | None
    omitted_static_branch: NativeOmittedStaticBranch | None
    correction_branches: tuple[NativeCorrectionBranch, ...]
    future_branches: tuple[NativeFutureBranch, ...]
    overshoot: NativePriorPredictiveRollout | None
    objective: NativeCALVINObjectiveResult


@dataclass(frozen=True, slots=True)
class NativeRepresentationObjectiveStepResult:
    """All outputs from one predictive/structural shared-host transaction."""

    primary: LingBotNativeContext
    final_relation: RelationOutput
    current_grid_branch: NativeCurrentGridBranch | None
    omitted_static_branch: NativeOmittedStaticBranch | None
    correction_branches: tuple[NativeCorrectionBranch, ...]
    future_branches: tuple[NativeFutureBranch, ...]
    overshoot: NativePriorPredictiveRollout | None
    objective: NativeCALVINObjectiveResult


@dataclass(frozen=True, slots=True)
class _NativeObjectiveBranches:
    current_grid: NativeCurrentGridBranch | None
    omitted_static: NativeOmittedStaticBranch | None
    corrections: tuple[NativeCorrectionBranch, ...]
    futures: tuple[NativeFutureBranch, ...]
    overshoot: NativePriorPredictiveRollout | None
    objective: NativeCALVINObjectiveResult


def _intermediate_relation_time_axis(
    mappings_by_time: Sequence[Mapping[int, RelationOutput]],
) -> dict[int, tuple[RelationOutput, ...]]:
    """Transpose graph-validated depth mappings onto the objective time axis."""

    if not mappings_by_time:
        raise ValueError("intermediate relation time axis cannot be empty")
    layers = tuple(mappings_by_time[0])
    if any(tuple(mapping) != layers for mapping in mappings_by_time):
        raise RuntimeError("intermediate relation depths changed across the local window")
    return {layer: tuple(mapping[layer] for mapping in mappings_by_time) for layer in layers}


@dataclass(frozen=True, slots=True)
class NativeRelationProbeStepResult:
    """Loss-only ownership transaction with no predictive query or checkpoint state."""

    primary: NativePolicyForwardResult
    objective: NativeCALVINObjectiveResult


def make_native_current_correction_request(
    *,
    batch_size: int,
    valid: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    route_id: int = 0,
    address_width: int = 0,
) -> NativePredictionRequest:
    """Create one prior-row prediction of independently cached current evidence."""

    integers = (batch_size, route_id, address_width)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("native correction request dimensions must be integers")
    if batch_size <= 0 or route_id < 0 or address_width < 0:
        raise ValueError("native correction request dimensions are outside their valid range")
    target_device = torch.device(device)
    if valid.shape != (batch_size,) or valid.dtype != torch.bool:
        raise ValueError("native correction validity must be boolean [batch]")
    if valid.device != target_device:
        raise ValueError("native correction validity and request must share one device")
    return NativePredictionRequest(
        source=PredictionSource.PRIOR,
        evidence=PredictionEvidence.CURRENT_CORRECTION,
        route_ids=torch.full((batch_size, 1), route_id, dtype=torch.long, device=target_device),
        horizons=torch.zeros(batch_size, 1, dtype=torch.long, device=target_device),
        addresses=torch.zeros(
            batch_size,
            1,
            address_width,
            dtype=dtype,
            device=target_device,
        ),
        valid=valid[:, None],
    )


def make_native_current_filter_request(
    *,
    source: PredictionSource,
    batch_size: int,
    valid: torch.Tensor,
    device: torch.device | str,
    dtype: torch.dtype,
    route_id: int = 0,
    address_width: int = 0,
) -> NativePredictionRequest:
    """Create one v3 current-summary request with an explicit filter phase."""

    evidence = {
        PredictionSource.PRIOR: PredictionEvidence.CURRENT_PRIOR,
        PredictionSource.POSTERIOR: PredictionEvidence.CURRENT_POSTERIOR,
    }.get(source)
    if evidence is None:
        raise ValueError("native filter request source must be prior or posterior")
    integers = (batch_size, route_id, address_width)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("native filter request dimensions must be integers")
    if batch_size <= 0 or route_id < 0 or address_width < 0:
        raise ValueError("native filter request dimensions are outside their valid range")
    target_device = torch.device(device)
    if valid.shape != (batch_size,) or valid.dtype != torch.bool:
        raise ValueError("native filter validity must be boolean [batch]")
    if valid.device != target_device:
        raise ValueError("native filter validity and request must share one device")
    return NativePredictionRequest(
        source=source,
        evidence=evidence,
        route_ids=torch.full((batch_size, 1), route_id, dtype=torch.long, device=target_device),
        horizons=torch.zeros(batch_size, 1, dtype=torch.long, device=target_device),
        addresses=torch.zeros(
            batch_size,
            1,
            address_width,
            dtype=dtype,
            device=target_device,
        ),
        valid=valid[:, None],
    )


def make_native_current_grid_request(
    *,
    source_mask: QwenPackedPatchMask,
    route_id: int,
    dtype: torch.dtype,
) -> NativePredictionRequest:
    """Create exact spatial queries from a label-independent Qwen mask plan."""

    if not isinstance(source_mask, QwenPackedPatchMask) or source_mask.query_count <= 0:
        raise ValueError("current-grid request requires a nonempty Qwen source mask")
    if isinstance(route_id, bool) or not isinstance(route_id, int) or route_id < 0:
        raise ValueError("current-grid route ID must be a non-negative integer")
    addresses = qwen_mask_query_addresses(source_mask, dtype=dtype)
    batch, queries = source_mask.query_valid.shape
    return NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.CURRENT_RANDOM_GRID,
        route_ids=torch.full(
            (batch, queries),
            route_id,
            dtype=torch.long,
            device=addresses.device,
        ),
        horizons=torch.zeros(
            batch,
            queries,
            dtype=torch.long,
            device=addresses.device,
        ),
        addresses=addresses,
        valid=source_mask.query_valid,
    )


def make_native_omitted_static_request(
    *,
    omission: QwenWholeViewOmission,
    route_id: int,
    address_width: int,
    dtype: torch.dtype,
) -> NativePredictionRequest:
    """Create one zero-address query for an independently available static view."""

    if not isinstance(omission, QwenWholeViewOmission) or omission.omitted_view_index != 0:
        raise ValueError("omitted-static request requires static Qwen view index 0")
    if (
        isinstance(route_id, bool)
        or not isinstance(route_id, int)
        or route_id < 0
        or isinstance(address_width, bool)
        or not isinstance(address_width, int)
        or address_width < 0
    ):
        raise ValueError("omitted-static route and address width must be non-negative integers")
    batch = omission.image_valid.shape[0]
    device = omission.image_valid.device
    return NativePredictionRequest(
        source=PredictionSource.POSTERIOR,
        evidence=PredictionEvidence.OMITTED_MODALITY,
        route_ids=torch.full((batch, 1), route_id, dtype=torch.long, device=device),
        horizons=torch.zeros(batch, 1, dtype=torch.long, device=device),
        addresses=torch.zeros(batch, 1, address_width, dtype=dtype, device=device),
        valid=omission.source_valid[:, None],
    )


def _static_rgb_sha256(batch: CollatedNativeCALVINBatch) -> tuple[str, ...]:
    hashes: list[str] = []
    for request in batch.structural_target_requests:
        source_hashes = dict(request.source_sensor_sha256)
        try:
            hashes.append(source_hashes["rgb_static"])
        except KeyError as error:
            raise ValueError("native predictive source lacks the static RGB digest") from error
    return tuple(hashes)


def _gripper_rgb_sha256(batch: CollatedNativeCALVINBatch) -> tuple[str, ...]:
    hashes: list[str] = []
    for request in batch.structural_target_requests:
        source_hashes = dict(request.source_sensor_sha256)
        try:
            hashes.append(source_hashes["rgb_gripper"])
        except KeyError as error:
            raise ValueError("native predictive source lacks the gripper RGB digest") from error
    return tuple(hashes)


def _future_loss_inputs(
    *,
    branches: tuple[NativeFutureBranch, ...],
    cache: LingBotPredictiveTargetCache,
    track_identity_keys: tuple[tuple[str, ...], ...],
    weight: float,
    loss_power: float,
) -> tuple[NativePredictiveLossInput, ...]:
    values: list[NativePredictiveLossInput] = []
    for branch in branches:
        if branch.target_name != LINGBOT_PREDICTIVE_TARGET_SPACE:
            raise ValueError("native future branch uses the wrong predictive target space")
        requests = branch.batch.structural_target_requests
        target = cache.target_for(
            source_global_indices=tuple(value.source_global_index for value in requests),
            source_rgb_sha256=_static_rgb_sha256(branch.batch),
            track_identity_keys=track_identity_keys,
            request=branch.request,
            device=branch.prediction.device,
        )
        values.append(
            NativePredictiveLossInput(
                prediction=branch.prediction,
                request=branch.request,
                target=target,
                weight=weight,
                identity_source_phase=branch.identity_source_phase,
                loss_power=loss_power,
            )
        )
    return tuple(values)


def _correction_loss_inputs(
    *,
    branches: tuple[NativeCorrectionBranch, ...],
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    minimum_visible_fraction: float,
    weight: float,
    loss_power: float,
) -> tuple[NativePredictiveLossInput, ...]:
    values: list[NativePredictiveLossInput] = []
    for branch in branches:
        target = native_current_correction_target(
            branch=branch,
            cache=cache,
            physical_sidecar=physical_sidecar,
            track_identity_keys=track_identity_keys,
            minimum_visible_fraction=minimum_visible_fraction,
            device=branch.prediction.device,
        )
        values.append(
            NativePredictiveLossInput(
                prediction=branch.prediction,
                request=branch.request,
                target=target,
                weight=weight,
                identity_source_phase=branch.identity_source_phase,
                loss_power=loss_power,
            )
        )
    return tuple(values)


def _filter_phase_loss_inputs(
    *,
    branches: tuple[NativeFilterPhaseBranch, ...],
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    minimum_visible_fraction: float,
    weight: float,
    loss_power: float,
) -> tuple[NativePredictiveLossInput, ...]:
    """Resolve v3 prior/posterior terms only after causal row assignment exists."""

    values: list[NativePredictiveLossInput] = []
    for branch in branches:
        target = native_current_filter_target(
            branch=branch,
            cache=cache,
            physical_sidecar=physical_sidecar,
            track_identity_keys=track_identity_keys,
            minimum_visible_fraction=minimum_visible_fraction,
            device=branch.prediction.device,
        )
        values.append(
            NativePredictiveLossInput(
                prediction=branch.prediction,
                request=branch.request,
                target=target,
                weight=weight,
                identity_source_phase=branch.identity_source_phase,
                loss_power=loss_power,
            )
        )
    return tuple(values)


def native_current_filter_target(
    *,
    branch: NativeFilterPhaseBranch,
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    minimum_visible_fraction: float,
    device: torch.device | str,
) -> NativePredictiveTarget:
    """Resolve one detached target shared by both v3 filter phases."""

    if not isinstance(branch, NativeFilterPhaseBranch):
        raise TypeError("current-filter target resolution requires a filter branch")
    requests = branch.batch.structural_target_requests
    return cache.current_correction_summary_target_for(
        source_global_indices=tuple(value.source_global_index for value in requests),
        source_static_rgb_sha256=_static_rgb_sha256(branch.batch),
        track_identity_keys=track_identity_keys,
        request=branch.request,
        physical_sidecar=physical_sidecar,
        minimum_visible_fraction=minimum_visible_fraction,
        device=device,
    )


def native_current_correction_target(
    *,
    branch: NativeCorrectionBranch,
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    minimum_visible_fraction: float,
    device: torch.device | str,
) -> NativePredictiveTarget:
    """Resolve the exact detached target shared by training and diagnostics."""

    if not isinstance(branch, NativeCorrectionBranch):
        raise TypeError("current-correction target resolution requires a correction branch")
    requests = branch.batch.structural_target_requests
    return cache.current_correction_summary_target_for(
        source_global_indices=tuple(value.source_global_index for value in requests),
        source_static_rgb_sha256=_static_rgb_sha256(branch.batch),
        track_identity_keys=track_identity_keys,
        request=branch.request,
        physical_sidecar=physical_sidecar,
        minimum_visible_fraction=minimum_visible_fraction,
        device=device,
    )


def _correction_valid_by_time(
    *,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    previous_state: NativePersistentState | None,
    previous_state_valid: torch.Tensor | None,
) -> tuple[torch.Tensor, ...]:
    """Mark only priors backed by a previous posterior and no reset event."""

    first_batch = batches[0]
    device = first_batch.controls.values.device
    batch_size = first_batch.routing.batch_size
    if previous_state_valid is None:
        first_valid = torch.full(
            (batch_size,),
            previous_state is not None,
            dtype=torch.bool,
            device=device,
        )
    else:
        if previous_state_valid.shape != (batch_size,) or previous_state_valid.dtype != torch.bool:
            raise ValueError("previous-state validity must be boolean [batch]")
        if previous_state_valid.device != device:
            raise ValueError("previous-state validity and batches must share one device")
        first_valid = previous_state_valid
    if previous_state is None and first_valid.any():
        raise ValueError("an absent previous state cannot support current correction")
    first_reset = first_batch.prior_control_reset
    if (first_valid & first_reset).any():
        raise ValueError("a reset sample cannot read a valid previous posterior")
    if ((~first_reset) & ~first_valid).any():
        raise ValueError("a continuation sample requires a valid previous posterior")

    values: list[torch.Tensor] = []
    for time_index, batch in enumerate(batches):
        if batch.routing.batch_size != batch_size or batch.controls.values.device != device:
            raise ValueError("current-correction batches must share batch size and device")
        reset = batch.prior_control_reset
        inherited = first_valid if time_index == 0 else torch.ones_like(first_valid)
        values.append(inherited & ~reset)
    return tuple(values)


def _validate_prior_bindings_have_valid_source(
    *,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
    first_prior_valid: torch.Tensor,
) -> None:
    """Reject identities whose recurrent source is absent or reset."""

    if len(prior_row_bindings_by_batch) != first_prior_valid.numel():
        raise ValueError("prior row bindings and source validity have different batches")
    invalid_with_bindings = [
        index
        for index, bindings in enumerate(prior_row_bindings_by_batch)
        if bindings and not bool(first_prior_valid[index].item())
    ]
    if invalid_with_bindings:
        raise ValueError("prior row bindings require a valid non-reset recurrent source")


def _current_grid_loss_input(
    *,
    branch: NativeCurrentGridBranch,
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    merge_size: int,
    weight: float,
    loss_power: float,
) -> NativePredictiveLossInput:
    if (branch.source_mask.query_view_indices != 0).any():
        raise ValueError("the frozen current-grid target cache supports only the static view")
    static_grid = branch.source_mask.image_grid_thw[:, 0, 1:] // merge_size
    requests = branch.batch.structural_target_requests
    target = cache.target_for(
        source_global_indices=tuple(value.source_global_index for value in requests),
        source_rgb_sha256=_static_rgb_sha256(branch.batch),
        track_identity_keys=track_identity_keys,
        selected_token_indices=branch.source_mask.query_token_indices,
        merged_grid_hw=static_grid,
        request=branch.request,
        physical_sidecar=physical_sidecar,
        device=branch.prediction.device,
    )
    return NativePredictiveLossInput(
        prediction=branch.prediction,
        request=branch.request,
        target=target,
        weight=weight,
        identity_source_phase=branch.identity_source_phase,
        loss_power=loss_power,
    )


def _omitted_static_loss_input(
    *,
    branch: NativeOmittedStaticBranch,
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    weight: float,
    loss_power: float,
) -> NativePredictiveLossInput:
    target = omitted_static_target(
        batch=branch.batch,
        request=branch.request,
        omission=branch.omission,
        cache=cache,
        physical_sidecar=physical_sidecar,
        track_identity_keys=track_identity_keys,
        device=branch.prediction.device,
    )
    return NativePredictiveLossInput(
        prediction=branch.prediction,
        request=branch.request,
        target=target,
        weight=weight,
        identity_source_phase=branch.identity_source_phase,
        loss_power=loss_power,
    )


def omitted_static_target(
    *,
    batch: CollatedNativeCALVINBatch,
    request: NativePredictionRequest,
    omission: QwenWholeViewOmission,
    cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    track_identity_keys: tuple[tuple[str, ...], ...],
    device: torch.device,
) -> NativePredictiveTarget:
    """Materialize the frozen omitted-view target without running its policy branch."""

    requests = batch.structural_target_requests
    return cache.omitted_static_summary_target_for(
        source_global_indices=tuple(value.source_global_index for value in requests),
        source_static_rgb_sha256=_static_rgb_sha256(batch),
        source_gripper_rgb_sha256=_gripper_rgb_sha256(batch),
        track_identity_keys=track_identity_keys,
        request=request,
        omission=omission,
        physical_sidecar=physical_sidecar,
        device=device,
    )


def run_native_calvin_relation_probe_objective(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    previous_state: NativePosteriorState | None,
    previous_state_valid: torch.Tensor | None,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    objective_config: NativeObjectiveConfig,
    structural_config: NativeStructuralLossConfig,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
) -> NativeRelationProbeStepResult:
    """Run an ownership-only diagnostic over the exact shared host.

    The transaction deliberately creates no predictive query, source-masked
    branch, overshoot, or committable lane state.  The official action loss is
    measured but frozen; ownership is the only surface the caller may optimize.
    """

    if not batches or len(batches) not in (1, 2, 3, 4):
        raise ValueError("native relation probe requires one primary or a 2..4 step window")
    if not isinstance(graph, LingBotNativeGraph):
        raise TypeError("native relation probe requires the installed LingBot graph")
    if (
        not isinstance(objective_config, NativeObjectiveConfig)
        or objective_config.predictive_weight != 0
    ):
        raise ValueError("native relation probe requires zero predictive family weight")
    if (
        objective_config.structural_weight <= 0
        or not isinstance(structural_config, NativeStructuralLossConfig)
        or structural_config.ownership_weight <= 0
    ):
        raise ValueError("native relation probe requires positive structural ownership weight")
    relation_prior_valid = _correction_valid_by_time(
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
    )
    _validate_prior_bindings_have_valid_source(
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        first_prior_valid=relation_prior_valid[0],
    )

    if len(batches) == 1:
        primary = run_native_policy_relation_training_forward(
            policy,
            model_inputs=batches[0].model_inputs,
            context=LingBotNativeContext(
                controls=batches[0].controls,
                previous_state=previous_state,
                previous_state_valid=previous_state_valid,
                modalities=batches[0].modalities,
                supervise_intermediate_relations=bool(graph.config.relation_supervision_layers),
            ),
        )
        primary_relation = primary.context.relation_output
        if primary_relation is None:
            raise RuntimeError("native relation probe omitted its primary relation output")
        relations = (primary_relation,)
        intermediate_relations_by_layer = _intermediate_relation_time_axis(
            (primary.context.intermediate_relation_outputs,)
        )
    else:
        local = run_native_relation_local_bptt(
            policy,
            steps=tuple(
                NativeLocalBPTTStep(
                    model_inputs=batch.model_inputs,
                    controls=batch.controls,
                    modalities=batch.modalities,
                )
                for batch in batches
            ),
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
        )
        primary = local.primary
        primary_relation = primary.context.relation_output
        if primary_relation is None:
            raise RuntimeError("native relation probe omitted its primary relation output")
        relations = (primary_relation, *(value.relation_output for value in local.auxiliary))
        intermediate_relations_by_layer = _intermediate_relation_time_axis(
            (
                primary.context.intermediate_relation_outputs,
                *(value.intermediate_relation_outputs for value in local.auxiliary),
            )
        )

    composed = compose_native_calvin_objective(
        official_policy_loss=primary.official_total_loss,
        requests_by_time=tuple(batch.structural_target_requests for batch in batches),
        model_inputs_by_time=tuple(batch.model_inputs for batch in batches),
        relations=relations,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=task_identity_resolver,
        patch_size=patch_size,
        merge_size=merge_size,
        objective_config=objective_config,
        structural_config=structural_config,
        require_policy_loss_grad=False,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        intermediate_relations_by_layer=intermediate_relations_by_layer,
    )
    ownership = composed.objective.normalized_terms.get("set/ownership")
    if ownership is None or not ownership.requires_grad:
        raise RuntimeError("native relation probe produced detached ownership loss")
    return NativeRelationProbeStepResult(primary=primary, objective=composed)


def _compose_native_objective_branches(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    previous_state: NativePosteriorState | None,
    previous_state_valid: torch.Tensor | None,
    primary_context: LingBotNativeContext,
    requests: tuple[NativePredictionRequest | None, ...],
    relations: tuple[RelationOutput, ...],
    intermediate_relations_by_layer: Mapping[int, Sequence[RelationOutput]],
    prediction_outputs: tuple[Mapping[str, torch.Tensor], ...],
    behavior_future_branch: NativeFutureBranch | None,
    official_policy_loss: torch.Tensor | None,
    require_policy_loss_grad: bool,
    predictive_cache: LingBotPredictiveTargetCache,
    current_grid_cache: LingBotCurrentGridTargetCache,
    source_mask: QwenPackedPatchMask | None,
    omitted_static_view: QwenWholeViewOmission | None,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    objective_config: NativeObjectiveConfig,
    structural_config: NativeStructuralLossConfig,
    overshoot_factory: NativeOvershootFactory | None,
    predictive_term_weight: float,
    current_grid_term_weight: float,
    omitted_static_term_weight: float,
    predictive_loss_power: float,
    minimum_supervised_fraction: float,
    capacity_seeds: Sequence[int | None] | None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
) -> _NativeObjectiveBranches:
    """Resolve shared predictive branches and post-forward loss-only targets."""

    if any(
        request is not None and LINGBOT_PREDICTIVE_TARGET_SPACE not in value
        for request, value in zip(requests, prediction_outputs, strict=True)
    ):
        raise RuntimeError("native full-graph correction query omitted its projected output")
    posterior_state = primary_context.posterior_state
    if posterior_state is None:
        raise RuntimeError("native objective omitted its primary posterior state")
    overshoot = None if overshoot_factory is None else overshoot_factory(posterior_state)
    if overshoot is not None and overshoot.state.batch_size != posterior_state.batch_size:
        raise ValueError("native overshoot and primary posterior batch sizes differ")

    current_grid_branch = None
    omitted_static_branch = None
    if source_mask is not None:
        if graph.config.prediction_address_width != 2:
            raise ValueError("current-grid training requires a 2D prediction-address contract")
        current_request = make_native_current_grid_request(
            source_mask=source_mask,
            route_id=current_grid_cache.contract.route_id,
            dtype=batches[0].controls.values.dtype,
        )
        masked = run_native_source_masked_training_forward(
            policy,
            model_inputs=batches[0].model_inputs,
            controls=batches[0].controls,
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
            prediction_request=current_request,
            source_mask=source_mask,
            modalities=batches[0].modalities,
        )
        try:
            current_prediction = masked.prediction_outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
        except KeyError as error:
            raise RuntimeError("current-grid branch omitted its DINO prediction") from error
        current_grid_branch = NativeCurrentGridBranch(
            batch=batches[0],
            request=current_request,
            source_mask=source_mask,
            prediction=current_prediction,
            # The counterfactual source omits selected current evidence.  Until
            # a branch-local row assignment exists, only identities carried by
            # the incoming prior are provably aligned with the main branch.
            identity_source_phase=0,
        )
    elif omitted_static_view is not None:
        omitted_request = make_native_omitted_static_request(
            omission=omitted_static_view,
            route_id=current_grid_cache.contract.route_id,
            address_width=graph.config.prediction_address_width,
            dtype=batches[0].controls.values.dtype,
        )
        omitted = run_native_omitted_image_view_training_forward(
            policy,
            model_inputs=batches[0].model_inputs,
            controls=batches[0].controls,
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
            prediction_request=omitted_request,
            omission=omitted_static_view,
            modalities=batches[0].modalities,
        )
        try:
            omitted_prediction = omitted.prediction_outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
        except KeyError as error:
            raise RuntimeError("omitted-static branch omitted its DINO prediction") from error
        omitted_static_branch = NativeOmittedStaticBranch(
            batch=batches[0],
            request=omitted_request,
            omission=omitted_static_view,
            prediction=omitted_prediction,
            # A retained camera may contain the object, but no branch-local
            # matcher yet proves that a fresh symmetric row has the same gauge.
            identity_source_phase=0,
        )

    correction_values: list[NativeCorrectionBranch] = []
    future_values: list[NativeFutureBranch] = (
        [] if behavior_future_branch is None else [behavior_future_branch]
    )
    for time_index, (batch, request, outputs) in enumerate(
        zip(batches, requests, prediction_outputs, strict=True)
    ):
        if request is None:
            continue
        prediction = outputs[LINGBOT_PREDICTIVE_TARGET_SPACE]
        if request.evidence is PredictionEvidence.CURRENT_CORRECTION:
            correction_values.append(
                NativeCorrectionBranch(
                    batch=batch,
                    request=request,
                    prediction=prediction,
                    identity_source_phase=2 * time_index,
                )
            )
        elif request.evidence is PredictionEvidence.FUTURE:
            future_values.append(
                NativeFutureBranch(
                    batch=batch,
                    request=request,
                    target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
                    prediction=prediction,
                    identity_source_phase=(
                        2 * time_index + int(request.source is PredictionSource.POSTERIOR)
                    ),
                )
            )
        else:
            raise ValueError("native primary objective received an unsupported prediction request")
    correction_branches = tuple(correction_values)
    if overshoot is not None:
        future_values.append(
            NativeFutureBranch(
                batch=batches[0],
                request=overshoot.request,
                target_name=overshoot.target_name,
                prediction=overshoot.prediction,
                identity_source_phase=1,
            )
        )
    future_branches = tuple(future_values)

    composed = compose_native_calvin_objective(
        official_policy_loss=official_policy_loss,
        requests_by_time=tuple(batch.structural_target_requests for batch in batches),
        model_inputs_by_time=tuple(batch.model_inputs for batch in batches),
        relations=relations,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=task_identity_resolver,
        patch_size=patch_size,
        merge_size=merge_size,
        objective_config=objective_config,
        structural_config=structural_config,
        require_policy_loss_grad=require_policy_loss_grad,
        predictive_input_factory=lambda target_bundle: (
            _correction_loss_inputs(
                branches=correction_branches,
                cache=current_grid_cache,
                physical_sidecar=physical_sidecar,
                track_identity_keys=target_bundle.identity_keys_by_batch,
                minimum_visible_fraction=predictive_cache.contract.minimum_visible_fraction,
                weight=predictive_term_weight,
                loss_power=predictive_loss_power,
            )
            + _future_loss_inputs(
                branches=future_branches,
                cache=predictive_cache,
                track_identity_keys=target_bundle.identity_keys_by_batch,
                weight=predictive_term_weight,
                loss_power=predictive_loss_power,
            )
            + (
                ()
                if current_grid_branch is None
                else (
                    _current_grid_loss_input(
                        branch=current_grid_branch,
                        cache=current_grid_cache,
                        physical_sidecar=physical_sidecar,
                        track_identity_keys=target_bundle.identity_keys_by_batch,
                        merge_size=merge_size,
                        weight=current_grid_term_weight,
                        loss_power=predictive_loss_power,
                    ),
                )
            )
            + (
                ()
                if omitted_static_branch is None
                else (
                    _omitted_static_loss_input(
                        branch=omitted_static_branch,
                        cache=current_grid_cache,
                        physical_sidecar=physical_sidecar,
                        track_identity_keys=target_bundle.identity_keys_by_batch,
                        weight=omitted_static_term_weight,
                        loss_power=predictive_loss_power,
                    ),
                )
            )
        ),
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        intermediate_relations_by_layer=intermediate_relations_by_layer,
    )
    return _NativeObjectiveBranches(
        current_grid=current_grid_branch,
        omitted_static=omitted_static_branch,
        corrections=correction_branches,
        futures=future_branches,
        overshoot=overshoot,
        objective=composed,
    )


def _native_primary_prediction_requests(
    *,
    graph: LingBotNativeGraph,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    correction_valid: tuple[torch.Tensor, ...],
    route_id: int,
    behavior_conditioned: bool,
) -> tuple[NativePredictionRequest | None, ...]:
    """Build one deploy-safe correction query per observation time."""

    if not isinstance(behavior_conditioned, bool):
        raise TypeError("behavior-conditioned request selection must be boolean")
    if behavior_conditioned:
        return tuple(None for _batch in batches)
    return tuple(
        make_native_current_correction_request(
            batch_size=batch.routing.batch_size,
            valid=valid,
            device=batch.controls.values.device,
            dtype=batch.controls.values.dtype,
            route_id=route_id,
            address_width=graph.config.prediction_address_width,
        )
        for batch, valid in zip(batches, correction_valid, strict=True)
    )


def _native_behavior_future_branch(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    batch: CollatedNativeCALVINBatch,
    current_state: NativePosteriorState,
    route_id: int,
    behavior_prediction_controls: ExecutedControlBatch | None,
    behavior_prediction_horizon: int | None,
) -> NativeFutureBranch | None:
    """Run a loss-only controlled-future view after the deploy forward is complete."""

    if (behavior_prediction_controls is None) != (behavior_prediction_horizon is None):
        raise ValueError("behavior prediction controls and horizon must be provided together")
    if behavior_prediction_controls is None:
        return None
    horizon = behavior_prediction_horizon
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
        raise ValueError("behavior prediction horizon must be a positive integer")
    if behavior_prediction_controls.batch_size != batch.routing.batch_size:
        raise ValueError("behavior prediction controls and primary batch sizes differ")
    available = behavior_prediction_controls.token_valid.sum(dim=1)
    if ((available != 0) & (available != horizon)).any():
        raise ValueError(
            "behavior prediction controls must be empty or exactly equal the requested horizon"
        )
    valid = available == horizon
    if not bool(valid.any().item()):
        raise ValueError("behavior prediction batch has no horizon-valid sample")
    request = make_native_future_request(
        source=PredictionSource.PRIOR,
        batch_size=batch.routing.batch_size,
        horizon=horizon,
        valid=valid,
        device=batch.controls.values.device,
        dtype=batch.controls.values.dtype,
        route_id=route_id,
        address_width=graph.config.prediction_address_width,
    )
    if not current_state.rows.requires_grad:
        raise RuntimeError("behavior prediction received a detached deploy posterior")
    current_state.rows.retain_grad()
    _predicted_state, prediction = LingBotNativePriorStepper(
        policy,
        graph,
    ).step_with_prediction(
        current_state,
        behavior_prediction_controls,
        request,
        target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
    )
    return NativeFutureBranch(
        batch=batch,
        request=request,
        target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
        prediction=prediction,
        identity_source_phase=1,
    )


def run_native_calvin_full_objective(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    previous_state: NativePosteriorState | None,
    previous_state_valid: torch.Tensor | None,
    predictive_cache: LingBotPredictiveTargetCache,
    current_grid_cache: LingBotCurrentGridTargetCache,
    source_mask: QwenPackedPatchMask | None = None,
    omitted_static_view: QwenWholeViewOmission | None = None,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    objective_config: NativeObjectiveConfig,
    structural_config: NativeStructuralLossConfig,
    overshoot_factory: NativeOvershootFactory | None = None,
    behavior_prediction_controls: ExecutedControlBatch | None = None,
    behavior_prediction_horizon: int | None = None,
    predictive_term_weight: float = 1.0,
    current_grid_term_weight: float = 1.0,
    omitted_static_term_weight: float = 1.0,
    predictive_loss_power: float = 1.0,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
) -> NativeFullObjectiveStepResult:
    """Run shared host forwards, then resolve every independent loss target."""

    if not batches or len(batches) not in (1, 2, 3, 4):
        raise ValueError("native full objective requires one primary or a 2..4 step local window")
    if not isinstance(graph, LingBotNativeGraph):
        raise TypeError("native full objective requires the installed LingBot graph")
    if overshoot_factory is not None and not callable(overshoot_factory):
        raise TypeError("native overshoot factory must be callable")
    if overshoot_factory is not None and behavior_prediction_controls is not None:
        raise ValueError("recursive overshoot and behavior-conditioned prediction are exclusive")
    source_branch_count = int(source_mask is not None) + int(omitted_static_view is not None)
    if source_branch_count > 1:
        raise ValueError("current-grid and omitted-static branches are mutually exclusive")
    correction_valid = _correction_valid_by_time(
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
    )
    _validate_prior_bindings_have_valid_source(
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        first_prior_valid=correction_valid[0],
    )
    requests = _native_primary_prediction_requests(
        graph=graph,
        batches=batches,
        correction_valid=correction_valid,
        route_id=current_grid_cache.contract.route_id,
        behavior_conditioned=behavior_prediction_controls is not None,
    )

    if len(batches) == 1:
        context = LingBotNativeContext(
            controls=batches[0].controls,
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
            prediction_request=requests[0],
            modalities=batches[0].modalities,
            supervise_intermediate_relations=bool(graph.config.relation_supervision_layers),
        )
        primary = run_native_policy_training_forward(
            policy,
            model_inputs=batches[0].model_inputs,
            context=context,
        )
        primary_relation = primary.context.relation_output
        if primary_relation is None:
            raise RuntimeError("native primary forward omitted its relation output")
        relations = (primary_relation,)
        intermediate_relations_by_layer = _intermediate_relation_time_axis(
            (primary.context.intermediate_relation_outputs,)
        )
        prediction_outputs = (primary.context.prediction_outputs,)
    else:
        local = run_native_local_bptt(
            policy,
            steps=tuple(
                NativeLocalBPTTStep(
                    model_inputs=batch.model_inputs,
                    controls=batch.controls,
                    prediction_request=request,
                    modalities=batch.modalities,
                )
                for batch, request in zip(batches, requests, strict=True)
            ),
            previous_state=previous_state,
            previous_state_valid=previous_state_valid,
        )
        primary = local.primary
        primary_relation = primary.context.relation_output
        if primary_relation is None:
            raise RuntimeError("native local BPTT primary omitted its relation output")
        relations = (primary_relation, *(value.relation_output for value in local.auxiliary))
        intermediate_relations_by_layer = _intermediate_relation_time_axis(
            (
                primary.context.intermediate_relation_outputs,
                *(value.intermediate_relation_outputs for value in local.auxiliary),
            )
        )
        prediction_outputs = (
            primary.context.prediction_outputs,
            *(value.prediction_outputs for value in local.auxiliary),
        )
        # Retain only referenced relation/prediction graphs before an optional
        # source branch; auxiliary frames never construct an action suffix.
        del local

    current_state = primary.context.posterior_state
    if current_state is None:
        raise RuntimeError("native primary forward omitted its deploy posterior state")
    behavior_future_branch = _native_behavior_future_branch(
        policy,
        graph=graph,
        batch=batches[0],
        current_state=current_state,
        route_id=current_grid_cache.contract.route_id,
        behavior_prediction_controls=behavior_prediction_controls,
        behavior_prediction_horizon=behavior_prediction_horizon,
    )

    if objective_config.action_weight <= 0:
        raise ValueError("native full objective requires an active action family")
    branches = _compose_native_objective_branches(
        policy,
        graph=graph,
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        primary_context=primary.context,
        requests=requests,
        relations=relations,
        intermediate_relations_by_layer=intermediate_relations_by_layer,
        prediction_outputs=prediction_outputs,
        behavior_future_branch=behavior_future_branch,
        official_policy_loss=primary.official_total_loss,
        require_policy_loss_grad=True,
        predictive_cache=predictive_cache,
        current_grid_cache=current_grid_cache,
        source_mask=source_mask,
        omitted_static_view=omitted_static_view,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=task_identity_resolver,
        patch_size=patch_size,
        merge_size=merge_size,
        objective_config=objective_config,
        structural_config=structural_config,
        overshoot_factory=overshoot_factory,
        predictive_term_weight=predictive_term_weight,
        current_grid_term_weight=current_grid_term_weight,
        omitted_static_term_weight=omitted_static_term_weight,
        predictive_loss_power=predictive_loss_power,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
    )
    return NativeFullObjectiveStepResult(
        primary=primary,
        final_relation=relations[-1],
        current_grid_branch=branches.current_grid,
        omitted_static_branch=branches.omitted_static,
        correction_branches=branches.corrections,
        future_branches=branches.futures,
        overshoot=branches.overshoot,
        objective=branches.objective,
    )


def run_native_calvin_representation_objective(
    policy: nn.Module,
    *,
    graph: LingBotNativeGraph,
    batches: tuple[CollatedNativeCALVINBatch, ...],
    previous_state: NativePosteriorState | None,
    previous_state_valid: torch.Tensor | None,
    predictive_cache: LingBotPredictiveTargetCache,
    current_grid_cache: LingBotCurrentGridTargetCache,
    source_mask: QwenPackedPatchMask | None = None,
    omitted_static_view: QwenWholeViewOmission | None = None,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    capacity: int,
    task_identity_resolver: TaskIdentityResolver,
    patch_size: int,
    merge_size: int,
    objective_config: NativeObjectiveConfig,
    structural_config: NativeStructuralLossConfig,
    overshoot_factory: NativeOvershootFactory | None = None,
    behavior_prediction_controls: ExecutedControlBatch | None = None,
    behavior_prediction_horizon: int | None = None,
    predictive_term_weight: float = 1.0,
    current_grid_term_weight: float = 1.0,
    omitted_static_term_weight: float = 1.0,
    predictive_loss_power: float = 1.0,
    minimum_supervised_fraction: float = 0.0,
    capacity_seeds: Sequence[int | None] | None = None,
    prior_row_bindings_by_batch: tuple[RowBindings, ...],
) -> NativeRepresentationObjectiveStepResult:
    """Train every representation family through the shared host without action."""

    if not batches or len(batches) not in (1, 2, 3, 4):
        raise ValueError(
            "native representation objective requires one primary or a 2..4 step window"
        )
    if not isinstance(graph, LingBotNativeGraph):
        raise TypeError("native representation objective requires the installed LingBot graph")
    if objective_config.action_weight != 0:
        raise ValueError("native representation objective requires zero action-family weight")
    if objective_config.predictive_weight <= 0 or objective_config.structural_weight <= 0:
        raise ValueError(
            "native representation objective requires predictive and structural families"
        )
    if overshoot_factory is not None and not callable(overshoot_factory):
        raise TypeError("native representation overshoot factory must be callable")
    if overshoot_factory is not None and behavior_prediction_controls is not None:
        raise ValueError("recursive overshoot and behavior-conditioned prediction are exclusive")
    source_branch_count = int(source_mask is not None) + int(omitted_static_view is not None)
    if source_branch_count > 1:
        raise ValueError("current-grid and omitted-static branches are mutually exclusive")

    correction_valid = _correction_valid_by_time(
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
    )
    _validate_prior_bindings_have_valid_source(
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
        first_prior_valid=correction_valid[0],
    )
    requests = _native_primary_prediction_requests(
        graph=graph,
        batches=batches,
        correction_valid=correction_valid,
        route_id=current_grid_cache.contract.route_id,
        behavior_conditioned=behavior_prediction_controls is not None,
    )

    window = run_native_representation_window(
        policy,
        steps=tuple(
            NativeLocalBPTTStep(
                model_inputs=batch.model_inputs,
                controls=batch.controls,
                prediction_request=request,
                modalities=batch.modalities,
            )
            for batch, request in zip(batches, requests, strict=True)
        ),
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
    )
    contexts = window.contexts
    primary = contexts[0]
    relations = tuple(context.relation_output for context in contexts)
    if any(relation is None for relation in relations):
        raise RuntimeError("native representation forward omitted its relation output")
    typed_relations = tuple(
        relation for relation in relations if isinstance(relation, RelationOutput)
    )
    if len(typed_relations) != len(relations):
        raise RuntimeError("native representation relation output has the wrong type")
    intermediate_relations_by_layer = _intermediate_relation_time_axis(
        tuple(context.intermediate_relation_outputs for context in contexts)
    )
    prediction_outputs = tuple(context.prediction_outputs for context in contexts)
    del window
    current_state = primary.posterior_state
    if current_state is None:
        raise RuntimeError("native representation forward omitted its deploy posterior state")
    behavior_future_branch = _native_behavior_future_branch(
        policy,
        graph=graph,
        batch=batches[0],
        current_state=current_state,
        route_id=current_grid_cache.contract.route_id,
        behavior_prediction_controls=behavior_prediction_controls,
        behavior_prediction_horizon=behavior_prediction_horizon,
    )
    branches = _compose_native_objective_branches(
        policy,
        graph=graph,
        batches=batches,
        previous_state=previous_state,
        previous_state_valid=previous_state_valid,
        primary_context=primary,
        requests=requests,
        relations=typed_relations,
        intermediate_relations_by_layer=intermediate_relations_by_layer,
        prediction_outputs=prediction_outputs,
        behavior_future_branch=behavior_future_branch,
        official_policy_loss=None,
        require_policy_loss_grad=False,
        predictive_cache=predictive_cache,
        current_grid_cache=current_grid_cache,
        source_mask=source_mask,
        omitted_static_view=omitted_static_view,
        physical_sidecar=physical_sidecar,
        capacity=capacity,
        task_identity_resolver=task_identity_resolver,
        patch_size=patch_size,
        merge_size=merge_size,
        objective_config=objective_config,
        structural_config=structural_config,
        overshoot_factory=overshoot_factory,
        predictive_term_weight=predictive_term_weight,
        current_grid_term_weight=current_grid_term_weight,
        omitted_static_term_weight=omitted_static_term_weight,
        predictive_loss_power=predictive_loss_power,
        minimum_supervised_fraction=minimum_supervised_fraction,
        capacity_seeds=capacity_seeds,
        prior_row_bindings_by_batch=prior_row_bindings_by_batch,
    )
    return NativeRepresentationObjectiveStepResult(
        primary=primary,
        final_relation=typed_relations[-1],
        current_grid_branch=branches.current_grid,
        omitted_static_branch=branches.omitted_static,
        correction_branches=branches.corrections,
        future_branches=branches.futures,
        overshoot=branches.overshoot,
        objective=branches.objective,
    )

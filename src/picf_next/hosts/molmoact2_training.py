"""Explicit truncated-sequence training orchestration for MolmoAct2 and PICF.

This module does not define another representation learner or objective.  It
connects the host-neutral PICF posterior to the pinned official MolmoAct2
LeRobot policy while keeping supervision outside the deploy-time evidence
path.  Context frames truncate gradients; gradient transitions retain the
posterior graph and use the official action loss unchanged.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from torch import nn

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinPICFEvidenceFrame,
    CalvinStatefulTransitionDataset,
    CalvinStatefulTransitionSample,
    CalvinTrainingWindow,
)
from picf_next.data.calvin_loss_targets import (
    CALVIN_PHYSICAL_SOURCE_FIELDS,
    CalvinSourceFrameLossTargetRequest,
    CalvinStatefulLossTargetRequest,
    calvin_physical_source_hashes,
)
from picf_next.data.calvin_physical_supervision_schema import CALVIN_CAMERA_SPECS
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_rollout_targets import (
    CalvinPhysicalGeometryProvider,
    build_calvin_geometry_rollout_sample,
)
from picf_next.data.molmoact2_raster import project_molmoact2_resize_segmentation
from picf_next.data.robot_record import RobotTransitionRecord
from picf_next.data.rollout_targets import build_object_geometry_rollout_target
from picf_next.geometry import PhysicalGeometryContract
from picf_next.hosts.context import PICFActionEvidence
from picf_next.hosts.molmoact2_layout import (
    MOLMO_VISION_PATCH_MODALITY,
    MolmoAct2VisionPatchLayout,
)
from picf_next.models.core import PICFCore, PICFCoreOutput
from picf_next.models.dynamics_loss import (
    ObjectGeometryRolloutTarget,
    ObjectLifecycleInventoryTarget,
)
from picf_next.models.evidence import ModalityTokenSpan, NativeTokenBank
from picf_next.models.objective import PICFObjective, PICFObjectiveOutput
from picf_next.models.set_loss import ObjectSetTarget
from picf_next.models.temporal import ObjectBeliefBatch
from picf_next.training.control import PlannedSample, PlannedStreamMicrobatch, derive_subseed
from picf_next.training.stateful_runner import StatefulForwardOutput

if TYPE_CHECKING:
    from picf_next.hosts.molmoact2 import MolmoAct2PICFActionExpert

_CONTINUOUS_ACTION_TARGET_KEYS = frozenset({"action", "action_dim_is_pad", "action_horizon_is_pad"})


@dataclass(frozen=True, slots=True)
class MolmoAct2PICFTrainingConfig:
    """One explicit truncated-BPTT contract, independent of action horizon."""

    detached_context_frames: int
    gradient_transitions: int
    picf_core_lr: float
    require_explicit_flow_randomness: bool = False
    include_posterior_action_context: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.detached_context_frames, int)
            or isinstance(self.detached_context_frames, bool)
            or self.detached_context_frames < 0
        ):
            raise ValueError("detached_context_frames cannot be negative")
        if (
            not isinstance(self.gradient_transitions, int)
            or isinstance(self.gradient_transitions, bool)
            or self.gradient_transitions <= 0
        ):
            raise ValueError("gradient_transitions must be positive")
        if (
            isinstance(self.picf_core_lr, bool)
            or not math.isfinite(self.picf_core_lr)
            or self.picf_core_lr <= 0.0
        ):
            raise ValueError("picf_core_lr must be positive")
        if not isinstance(self.require_explicit_flow_randomness, bool):
            raise ValueError("require_explicit_flow_randomness must be a boolean")
        if not isinstance(self.include_posterior_action_context, bool):
            raise ValueError("include_posterior_action_context must be a boolean")

    @property
    def sequence_length(self) -> int:
        return self.detached_context_frames + self.gradient_transitions


@dataclass(frozen=True, slots=True)
class MolmoAct2PICFTransition:
    """Deploy-visible evidence at one transition plus an optional action target.

    `native_banks`, `host_observation_inputs`, `previous_executed_action` and
    `delta_t_s` are the only PICF inputs. The action is the
    command that caused the current observation, never the current action
    target. ``host_observation_inputs`` contains only the official deploy-time
    Molmo fields and is converted into both native VLM embeddings and one
    lossless dense patch bank with a single ViT call. `host_batch` is passed
    exclusively to the official policy loss and is required only on
    gradient-bearing transitions. ``flow_timesteps`` and ``flow_noise`` are
    explicit Monte Carlo inputs to that official loss; they are not observations
    or targets and never enter PICF. Mask, box and instance supervision therefore
    cannot enter object discovery through this type.
    """

    # May be empty only when ``host_observation_inputs`` produces the native
    # same-forward Molmo vision bank. Additional encoders append external banks.
    native_banks: tuple[NativeTokenBank, ...]
    previous_executed_action: torch.Tensor
    delta_t_s: torch.Tensor
    host_observation_inputs: Mapping[str, torch.Tensor] | None = None
    host_batch: Mapping[str, torch.Tensor] | None = None
    flow_timesteps: torch.Tensor | None = None
    flow_noise: torch.Tensor | None = None


@dataclass(frozen=True, slots=True)
class MolmoAct2PICFTrainingOutput:
    loss: torch.Tensor
    metrics: Mapping[str, float]
    action_losses: tuple[torch.Tensor, ...]
    core_outputs: tuple[PICFCoreOutput, ...]
    evidences: tuple[PICFActionEvidence, ...]
    vision_patch_layouts: tuple[MolmoAct2VisionPatchLayout | None, ...]
    final_belief: ObjectBeliefBatch


@dataclass(frozen=True, slots=True)
class MolmoAct2PICFJointTrainingOutput:
    """Official action output plus the one explicit PICF objective."""

    sequence: MolmoAct2PICFTrainingOutput
    objective: PICFObjectiveOutput

    @property
    def loss(self) -> torch.Tensor:
        return self.objective.loss


@dataclass(frozen=True, slots=True)
class MolmoAct2HostObservationView:
    """Deploy-visible non-image fields for one official host observation.

    The view deliberately omits the demonstrator action, source indices and
    every structural target. Sensor payloads arrive through the separate causal
    evidence prefix, so a host processor can build language/state/image inputs
    without receiving a training target.
    """

    task: str
    embodiment: str
    control_mode: str
    control_frame: str
    state_axes: tuple[str, ...]
    state_units: tuple[str, ...]
    state: tuple[float, ...]
    state_valid: tuple[bool, ...]
    timestamp_s: float
    delta_t_s: float


@dataclass(frozen=True, slots=True)
class CalvinStatefulEvidenceRequest:
    """Host-side metadata plus a strictly target-free causal sensor prefix."""

    sample_key: str
    augmentation_seed: int
    evidence_prefix: tuple[CalvinPICFEvidenceFrame, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.sample_key, str) or not self.sample_key:
            raise ValueError("stateful evidence sample key cannot be empty")
        if (
            not isinstance(self.augmentation_seed, int)
            or isinstance(self.augmentation_seed, bool)
            or self.augmentation_seed < 0
        ):
            raise ValueError("stateful evidence augmentation seed must be non-negative")
        if (
            not isinstance(self.evidence_prefix, tuple)
            or not self.evidence_prefix
            or any(not isinstance(frame, CalvinPICFEvidenceFrame) for frame in self.evidence_prefix)
        ):
            raise TypeError("stateful evidence must contain a nonempty CALVIN evidence prefix")
        timestamps = tuple(frame.timestamp_s for frame in self.evidence_prefix)
        if any(right <= left for left, right in zip(timestamps, timestamps[1:], strict=False)):
            raise ValueError("stateful evidence prefix must be strictly chronological")

    @property
    def evidence_frame(self) -> CalvinPICFEvidenceFrame:
        """Return the current frame for single-frame encoder compatibility."""

        return self.evidence_prefix[-1]


@dataclass(frozen=True, slots=True)
class CalvinStatefulLossTargets:
    """Loss-only labels constructed after the deploy-visible forward completes."""

    set_targets: tuple[ObjectSetTarget, ...] | None = None
    lifecycle_targets: tuple[ObjectLifecycleInventoryTarget | None, ...] | None = None
    geometry_rollout_target: ObjectGeometryRolloutTarget | None = None


@dataclass(frozen=True, slots=True)
class CalvinStatefulLossTargetLayout:
    """Prediction-free token layout exposed to the post-forward label builder.

    The layout is a cloned bool validity tensor plus immutable modality spans and
    two explicit numeric planes. ``target_dtype`` is the canonical float32
    supervision plane used by set, binding, lifecycle and physical-state
    losses. ``rollout_input_dtype`` is the posterior dtype used only for values
    that are replayed through the transition, such as executed actions and
    delta time. Keeping these planes separate prevents CUDA autocast details
    from changing label precision or silently promoting transition inputs. The
    layout deliberately contains no projected features, ownership, object
    queries, posterior state or action-layer evidence.
    """

    token_valid: torch.Tensor
    spans: tuple[ModalityTokenSpan, ...]
    target_dtype: torch.dtype
    rollout_input_dtype: torch.dtype
    vision_patch_layout: MolmoAct2VisionPatchLayout | None = None

    def __post_init__(self) -> None:
        if self.token_valid.ndim != 2 or self.token_valid.dtype != torch.bool:
            raise ValueError("loss-target token validity must be bool batch-by-token")
        if self.token_valid.requires_grad:
            raise ValueError("loss-target token validity cannot require gradients")
        if not self.spans:
            raise ValueError("loss-target layout requires at least one modality span")
        cursor = 0
        modalities: set[str] = set()
        for span in self.spans:
            if not isinstance(span, ModalityTokenSpan):
                raise TypeError("loss-target spans must use ModalityTokenSpan")
            if not span.modality or span.modality in modalities:
                raise ValueError("loss-target modality spans must be nonempty and unique")
            if span.start != cursor or span.stop <= span.start:
                raise ValueError("loss-target modality spans must form a contiguous partition")
            modalities.add(span.modality)
            cursor = span.stop
        if cursor != self.token_valid.shape[1]:
            raise ValueError("loss-target spans must cover the complete token layout")
        if self.target_dtype != torch.float32:
            raise ValueError("loss-target supervision dtype must be canonical float32")
        if not self.rollout_input_dtype.is_floating_point:
            raise ValueError("loss-target rollout input dtype must be floating point")
        if self.vision_patch_layout is not None:
            if len(self.vision_patch_layout.rows) != self.token_valid.shape[0]:
                raise ValueError("vision patch layout batch differs from loss-target validity")
            vision_spans = tuple(
                span for span in self.spans if span.modality == MOLMO_VISION_PATCH_MODALITY
            )
            if len(vision_spans) != 1:
                raise ValueError("vision patch layout requires exactly one Molmo dense-bank span")
            if (
                vision_spans[0].stop - vision_spans[0].start
                != self.vision_patch_layout.tokens_per_row
            ):
                raise ValueError("vision patch layout token count differs from projected evidence")


CalvinNativeBankBuilder = Callable[
    [tuple[tuple[CalvinPICFEvidenceFrame, ...], ...]],
    Sequence[NativeTokenBank],
]
CalvinStatefulNativeBankBuilder = Callable[
    [tuple[CalvinStatefulEvidenceRequest, ...]],
    Sequence[NativeTokenBank],
]
CalvinStatefulLossTargetBuilder = Callable[
    [tuple[CalvinStatefulLossTargetRequest, ...], CalvinStatefulLossTargetLayout],
    CalvinStatefulLossTargets,
]
CalvinHostBatchBuilder = Callable[
    [tuple[RobotTransitionRecord, ...]],
    Mapping[str, torch.Tensor],
]
CalvinStatefulHostBatchBuilder = Callable[
    [tuple[CalvinStatefulTransitionSample, ...]],
    Mapping[str, torch.Tensor],
]
CalvinHostObservationBuilder = Callable[
    [
        tuple[tuple[CalvinPICFEvidenceFrame, ...], ...],
        tuple[MolmoAct2HostObservationView, ...],
    ],
    Mapping[str, torch.Tensor],
]


class CalvinGeometryOvershootingTargetBuilder:
    """Build bounded physical rollouts exclusively on the post-forward loss path.

    The builder reuses the production transition's seven-dimensional CALVIN
    controls and independently generated physical geometry. It cannot inspect
    images, language, model predictions, posterior rows or action targets from
    the current transition. Identity alignment remains the responsibility of
    the current-frame set target, so this class introduces no second tracker.
    """

    def __init__(
        self,
        index: CalvinDatasetIndex,
        *,
        geometry_contract: PhysicalGeometryContract,
        geometry_provider: CalvinPhysicalGeometryProvider,
        maximum_horizon: int,
        supervised_horizons: Sequence[int],
    ) -> None:
        if not isinstance(index, CalvinDatasetIndex):
            raise TypeError("CALVIN geometry builder requires a CalvinDatasetIndex")
        if not isinstance(geometry_contract, PhysicalGeometryContract):
            raise TypeError("CALVIN geometry builder requires a physical geometry contract")
        if not callable(geometry_provider):
            raise TypeError("CALVIN geometry provider must be callable")
        if (
            not isinstance(maximum_horizon, int)
            or isinstance(maximum_horizon, bool)
            or maximum_horizon <= 0
        ):
            raise ValueError("CALVIN geometry rollout horizon must be positive")
        horizons = tuple(supervised_horizons)
        if (
            not horizons
            or tuple(sorted(horizons)) != horizons
            or len(set(horizons)) != len(horizons)
            or horizons[0] != 1
            or any(
                not isinstance(horizon, int)
                or isinstance(horizon, bool)
                or not 1 <= horizon <= maximum_horizon
                for horizon in horizons
            )
        ):
            raise ValueError(
                "CALVIN supervised horizons must be sorted unique, include one, "
                "and lie within the maximum horizon"
            )
        self.index = index
        self.geometry_contract = geometry_contract
        self.geometry_provider = geometry_provider
        self.maximum_horizon = maximum_horizon
        self.supervised_horizons = horizons

    def __call__(
        self,
        requests: tuple[CalvinStatefulLossTargetRequest, ...],
        layout: CalvinStatefulLossTargetLayout,
    ) -> CalvinStatefulLossTargets:
        if not requests:
            raise ValueError("CALVIN geometry builder requires at least one request")
        samples = tuple(
            build_calvin_geometry_rollout_sample(
                self.index,
                segment_index=request.segment_index,
                global_index=request.source_global_index,
                maximum_horizon=self.maximum_horizon,
                supervised_horizons=self.supervised_horizons,
                geometry_contract=self.geometry_contract,
                geometry_provider=self.geometry_provider,
            )
            for request in requests
        )
        return CalvinStatefulLossTargets(
            geometry_rollout_target=build_object_geometry_rollout_target(
                samples,
                action_dim=7,
                geometry_contract=self.geometry_contract,
                device=layout.token_valid.device,
                input_dtype=layout.rollout_input_dtype,
                target_dtype=layout.target_dtype,
            )
        )


def calvin_visible_object_target_request(
    sample: CalvinStatefulTransitionSample,
    *,
    augmentation_seed: int = 0,
) -> CalvinStatefulLossTargetRequest:
    """Build the canonical loss-only locator for one current CALVIN frame."""

    if not isinstance(sample, CalvinStatefulTransitionSample):
        raise TypeError("visible-object target request requires a stateful CALVIN sample")
    return CalvinStatefulLossTargetRequest(
        sample_key=sample.sample_key,
        segment_index=sample.record.task_index,
        source_global_index=sample.record.global_index,
        augmentation_seed=augmentation_seed,
        source_sensor_sha256=calvin_physical_source_hashes(sample.record),
    )


class CalvinVisibleObjectTargetBuilder:
    """Build current measurable sets and alive inventories from one physical sidecar.

    Simulator labels are resolved only after the deploy-visible forward. The
    current set contains every physical instance with positive supervised mass
    on the native vision grid across the official cameras. Fully occluded and
    zero-support objects remain absent from the measurement set but present in
    the independent lifecycle inventory. Small-support observations remain soft
    ownership targets; observation quality is calibrated by the discovery
    model rather than converted into a resolution-dependent hard label.

    Lifecycle ``visibility`` is the conditional event that an alive identity
    produced a supervised current measurement, not raw simulator visibility.
    The complete physical inventory therefore supervises this event as one for
    members of the current set and zero for every other alive identity. This
    leaves ambiguous ownership pixels unknown without hiding a known missed
    detection from the temporal sensor model.
    """

    def __init__(self, sidecar: CalvinPhysicalSupervisionSidecar) -> None:
        if not isinstance(sidecar, CalvinPhysicalSupervisionSidecar):
            raise TypeError("CALVIN visible targets require a physical supervision sidecar")
        self.sidecar = sidecar

    def __call__(
        self,
        requests: tuple[CalvinStatefulLossTargetRequest, ...],
        layout: CalvinStatefulLossTargetLayout,
    ) -> CalvinStatefulLossTargets:
        if any(not isinstance(request, CalvinStatefulLossTargetRequest) for request in requests):
            raise TypeError("stateful visible targets require stateful loss-target requests")
        return self._build(requests, layout, source_frames=False)

    def source_frames(
        self,
        requests: tuple[CalvinSourceFrameLossTargetRequest, ...],
        layout: CalvinStatefulLossTargetLayout,
    ) -> CalvinStatefulLossTargets:
        """Build object targets without inventing a language-segment identity."""

        if any(not isinstance(request, CalvinSourceFrameLossTargetRequest) for request in requests):
            raise TypeError("source-frame visible targets require source-frame requests")
        return self._build(requests, layout, source_frames=True)

    def measurement_frames(
        self,
        physical_frames: tuple[CalvinPhysicalSupervisionFrame, ...],
        source_sensor_sha256: tuple[tuple[tuple[str, str], ...], ...],
        layout: CalvinStatefulLossTargetLayout,
    ) -> tuple[ObjectSetTarget, ...]:
        """Project verified current-frame supervision without lifecycle labels.

        This path is for observation interventions whose current physical set is
        known but whose visibility event is not predictable from the prior.  It
        deliberately returns only the measurement-set target: synthetic object
        removal must not become supervision for transition detectability.
        """

        if any(not isinstance(frame, CalvinPhysicalSupervisionFrame) for frame in physical_frames):
            raise TypeError("measurement targets require physical supervision frames")
        hashes = tuple(dict(row) for row in source_sensor_sha256)
        result = self._build_physical_frames(
            physical_frames,
            hashes,
            layout,
            include_lifecycle=False,
        )
        if result.set_targets is None or result.lifecycle_targets is not None:
            raise RuntimeError("measurement-only target construction changed its contract")
        return result.set_targets

    def _build(
        self,
        requests: tuple[
            CalvinStatefulLossTargetRequest | CalvinSourceFrameLossTargetRequest,
            ...,
        ],
        layout: CalvinStatefulLossTargetLayout,
        *,
        source_frames: bool,
    ) -> CalvinStatefulLossTargets:
        if not requests or len(requests) != layout.token_valid.shape[0]:
            raise ValueError("CALVIN visible target requests must match the token-layout batch")
        physical_frames: list[CalvinPhysicalSupervisionFrame] = []
        source_hashes: list[dict[str, str]] = []
        for request in requests:
            if source_frames:
                if not isinstance(request, CalvinSourceFrameLossTargetRequest):
                    raise TypeError("source-frame request type changed inside target construction")
                physical = self.sidecar.source_frame(request.source_global_index)
            else:
                if not isinstance(request, CalvinStatefulLossTargetRequest):
                    raise TypeError("stateful request type changed inside target construction")
                physical = self.sidecar(
                    request.segment_index,
                    request.source_global_index,
                )
            physical_frames.append(physical)
            source_hashes.append(request.source_sensor_hash_by_field)
        return self._build_physical_frames(
            tuple(physical_frames),
            tuple(source_hashes),
            layout,
            include_lifecycle=True,
        )

    def _build_physical_frames(
        self,
        physical_frames: tuple[CalvinPhysicalSupervisionFrame, ...],
        source_hashes: tuple[dict[str, str], ...],
        layout: CalvinStatefulLossTargetLayout,
        *,
        include_lifecycle: bool,
    ) -> CalvinStatefulLossTargets:
        if (
            not physical_frames
            or len(physical_frames) != layout.token_valid.shape[0]
            or len(source_hashes) != len(physical_frames)
        ):
            raise ValueError("CALVIN physical frames must match the token-layout batch")
        vision_layout = layout.vision_patch_layout
        if vision_layout is None:
            raise ValueError("CALVIN visible targets require a MolmoAct2 vision patch layout")
        if not vision_layout.semantic_image_keys:
            raise ValueError("CALVIN visible targets require explicit semantic image keys")
        vision_spans = tuple(
            span for span in layout.spans if span.modality == MOLMO_VISION_PATCH_MODALITY
        )
        if len(vision_spans) != 1:
            raise ValueError("CALVIN visible targets require one Molmo dense vision span")
        vision_span = vision_spans[0]
        set_targets: list[ObjectSetTarget] = []
        lifecycle_targets: list[ObjectLifecycleInventoryTarget] = []

        for batch_index, (physical, request_hashes) in enumerate(
            zip(physical_frames, source_hashes, strict=True)
        ):
            if set(request_hashes) != CALVIN_PHYSICAL_SOURCE_FIELDS:
                raise ValueError("CALVIN visible targets require complete source sensor hashes")
            spec_by_camera = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
            camera_names = tuple(camera.camera_name for camera in physical.cameras)
            if len(set(camera_names)) != len(camera_names):
                raise ValueError("CALVIN sidecar contains duplicate cameras")
            if set(camera_names) != set(spec_by_camera):
                raise ValueError("CALVIN sidecar camera set differs from its contract")
            for camera in physical.cameras:
                spec = spec_by_camera.get(camera.camera_name)
                if spec is None:
                    raise ValueError("CALVIN sidecar contains an unknown camera")
                if (
                    request_hashes[str(spec["source_rgb_field"])] != camera.source_rgb_sha256
                    or request_hashes[str(spec["source_depth_field"])] != camera.source_depth_sha256
                ):
                    raise ValueError("CALVIN sidecar source sensor hash differs from the batch")
            camera_by_key = {camera.host_image_key: camera for camera in physical.cameras}
            image_spans = vision_layout.rows[batch_index]
            if tuple(span.image_key for span in image_spans) != tuple(camera_by_key):
                raise ValueError("CALVIN sidecar cameras differ from MolmoAct2 processor order")
            projected_cameras = []
            for image_span in image_spans:
                camera = camera_by_key[image_span.image_key]
                projected = project_molmoact2_resize_segmentation(
                    camera.owner_index.astype(np.int64, copy=False),
                    instance_ids=tuple(range(1, len(physical.identity_keys) + 1)),
                    image_token_pooling=np.asarray(image_span.image_token_pooling, dtype=np.int64),
                    image_grid=np.asarray(image_span.image_grid, dtype=np.int64),
                    image_num_crops=image_span.image_num_crops,
                    pixel_supervised=camera.owner_supervised,
                ).patch
                start = vision_span.start + image_span.start
                stop = vision_span.start + image_span.stop
                expected_valid = layout.token_valid[batch_index, start:stop]
                target_valid = torch.from_numpy(projected.token_valid).to(
                    device=expected_valid.device
                )
                if not torch.equal(target_valid, expected_valid):
                    raise ValueError("CALVIN owner target validity differs from dense ViT patches")
                target_supervised = torch.from_numpy(projected.supervised).to(
                    device=expected_valid.device
                )
                projected_cameras.append((projected, start, stop, target_supervised))

            visible_owner_indices = tuple(
                sorted(
                    {
                        int(owner)
                        for projected, _start, _stop, _supervised in projected_cameras
                        for owner in projected.instance_ids
                    }
                )
            )
            if any(
                owner <= 0 or owner > len(physical.identity_keys) for owner in visible_owner_indices
            ):
                raise ValueError("CALVIN visible owner references an unknown physical identity")
            visible_keys = tuple(
                physical.identity_keys[owner - 1] for owner in visible_owner_indices
            )
            owner_to_object = {owner: index for index, owner in enumerate(visible_owner_indices)}
            ownership = torch.zeros(
                layout.token_valid.shape[1],
                len(visible_keys) + 1,
                device=layout.token_valid.device,
                dtype=layout.target_dtype,
            )
            supervised = torch.zeros_like(layout.token_valid[batch_index])

            for projected, start, stop, target_supervised in projected_cameras:
                local = ownership[start:stop]
                for source_column, owner in enumerate(projected.instance_ids):
                    destination = owner_to_object.get(owner)
                    if destination is None:
                        raise ValueError("CALVIN projected owner is absent from visible inventory")
                    local[:, destination] = torch.from_numpy(
                        projected.object_probability[:, source_column]
                    ).to(device=local.device, dtype=local.dtype)
                local[:, -1] = torch.from_numpy(projected.context_probability).to(
                    device=local.device,
                    dtype=local.dtype,
                )
                local[~target_supervised] = 0.0
                supervised[start:stop] = target_supervised

            if visible_owner_indices:
                object_mass = ownership[supervised, :-1].sum(dim=0)
                if not bool(torch.all(object_mass > 0.0)):
                    raise RuntimeError(
                        "CALVIN token-visible inventory contains an owner without "
                        "supervised ownership mass"
                    )
            geometry_rows = torch.as_tensor(
                [owner - 1 for owner in visible_owner_indices],
                device=physical.geometry.device,
                dtype=torch.long,
            )
            visible_geometry = physical.geometry[geometry_rows].to(
                device=layout.token_valid.device,
                dtype=layout.target_dtype,
            )
            visible_geometry_variance = physical.geometry_variance[geometry_rows].to(
                device=layout.token_valid.device,
                dtype=layout.target_dtype,
            )
            visible_geometry_supervised = physical.geometry_supervised[geometry_rows].to(
                device=layout.token_valid.device
            )
            set_targets.append(
                ObjectSetTarget(
                    ownership=ownership,
                    token_valid=layout.token_valid[batch_index].clone(),
                    token_supervised=supervised,
                    object_inventory_complete=True,
                    geometry=visible_geometry,
                    geometry_variance=visible_geometry_variance,
                    geometry_supervised=visible_geometry_supervised,
                    geometry_contract=physical.geometry_contract,
                    temporal_identity_keys=visible_keys,
                )
            )
            if include_lifecycle:
                visible_owner_set = set(visible_owner_indices)
                lifecycle_targets.append(
                    ObjectLifecycleInventoryTarget(
                        alive_identity_keys=physical.identity_keys,
                        inventory_complete=True,
                        visibility=torch.as_tensor(
                            [
                                1.0 if owner in visible_owner_set else 0.0
                                for owner in range(1, len(physical.identity_keys) + 1)
                            ],
                            device=layout.token_valid.device,
                            dtype=layout.target_dtype,
                        ),
                        visibility_supervised=torch.ones(
                            len(physical.identity_keys),
                            device=layout.token_valid.device,
                            dtype=torch.bool,
                        ),
                    )
                )
        return CalvinStatefulLossTargets(
            set_targets=tuple(set_targets),
            lifecycle_targets=tuple(lifecycle_targets) if include_lifecycle else None,
        )


def compose_calvin_loss_target_builders(
    *builders: CalvinStatefulLossTargetBuilder,
) -> CalvinStatefulLossTargetBuilder:
    """Compose disjoint loss-only sources without implicit precedence."""

    frozen_builders = tuple(builders)
    if not frozen_builders or any(not callable(builder) for builder in frozen_builders):
        raise TypeError("CALVIN loss-target composition requires callable builders")

    def build(
        requests: tuple[CalvinStatefulLossTargetRequest, ...],
        layout: CalvinStatefulLossTargetLayout,
    ) -> CalvinStatefulLossTargets:
        merged: dict[str, object | None] = {
            "set_targets": None,
            "lifecycle_targets": None,
            "geometry_rollout_target": None,
        }
        for builder in frozen_builders:
            targets = builder(requests, layout)
            if not isinstance(targets, CalvinStatefulLossTargets):
                raise TypeError("loss-target builder must return CalvinStatefulLossTargets")
            for field in merged:
                value = getattr(targets, field)
                if value is None:
                    continue
                if merged[field] is not None:
                    raise ValueError(f"multiple CALVIN loss-target builders produced {field}")
                merged[field] = value
        return CalvinStatefulLossTargets(
            set_targets=cast(tuple[ObjectSetTarget, ...] | None, merged["set_targets"]),
            lifecycle_targets=cast(
                tuple[ObjectLifecycleInventoryTarget | None, ...] | None,
                merged["lifecycle_targets"],
            ),
            geometry_rollout_target=cast(
                ObjectGeometryRolloutTarget | None,
                merged["geometry_rollout_target"],
            ),
        )

    return build


def materialize_molmoact2_flow_randomness(
    policy: nn.Module,
    planned_samples: Sequence[PlannedSample],
    actions: torch.Tensor,
    *,
    transition_index: int | Sequence[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize official flow-matching randomness without mutating global RNG.

    The frozen global sample plan owns the parent streams. A transition-specific
    subseed makes truncated sequences random-access and insensitive to callback
    order. Timestep sampling delegates to the pinned official MolmoAct2 helper,
    while Gaussian noise uses an explicit device generator.
    """

    if (
        actions.ndim != 3
        or any(dimension <= 0 for dimension in actions.shape)
        or not actions.is_floating_point()
    ):
        raise ValueError("actions must be one floating [batch, horizon, dimension] tensor")
    if actions.device.type not in {"cpu", "cuda"}:
        raise ValueError("explicit MolmoAct2 flow randomness supports CPU or CUDA actions")
    samples = tuple(planned_samples)
    if len(samples) != actions.shape[0]:
        raise ValueError("planned sample count must equal the action batch size")
    if isinstance(transition_index, int) and not isinstance(transition_index, bool):
        transition_indices = (transition_index,) * len(samples)
    else:
        transition_indices = (
            tuple(transition_index) if isinstance(transition_index, Sequence) else ()
        )
    if len(transition_indices) != len(samples) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in transition_indices
    ):
        raise ValueError(
            "transition_index must be one non-negative integer or one per planned sample"
        )

    config = getattr(policy, "config", None)
    if config is None:
        raise TypeError("policy must expose the official MolmoAct2 config")
    policy_module = importlib.import_module(type(policy).__module__)
    sample_beta_timesteps = getattr(policy_module, "_sample_beta_timesteps", None)
    if not callable(sample_beta_timesteps):
        raise RuntimeError("pinned MolmoAct2 policy has no official Beta timestep sampler")

    num_flow_timesteps = config.num_flow_timesteps
    if (
        not isinstance(num_flow_timesteps, int)
        or isinstance(num_flow_timesteps, bool)
        or num_flow_timesteps <= 0
    ):
        raise ValueError("official MolmoAct2 num_flow_timesteps must be a positive integer")
    timestep_rows: list[torch.Tensor] = []
    noise_rows: list[torch.Tensor] = []
    for sample, sample_transition_index in zip(samples, transition_indices, strict=True):
        coordinates = ("transition", str(sample_transition_index))
        timestep_seed = derive_subseed(sample.flow_timestep_seed, *coordinates)
        if actions.device.type == "cpu":
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(timestep_seed)
                timestep_row = sample_beta_timesteps(
                    batch_size=num_flow_timesteps,
                    device=actions.device,
                    cutoff=config.flow_matching_cutoff,
                    time_offset=config.flow_matching_time_offset,
                    time_scale=config.flow_matching_time_scale,
                    alpha=config.flow_matching_beta_alpha,
                    beta=config.flow_matching_beta_beta,
                )
        else:
            device_index = actions.device.index
            if device_index is None:
                device_index = torch.cuda.current_device()
            with (
                torch.random.fork_rng(devices=[device_index]),
                torch.cuda.device(device_index),
            ):
                torch.cuda.manual_seed(timestep_seed)
                timestep_row = sample_beta_timesteps(
                    batch_size=num_flow_timesteps,
                    device=actions.device,
                    cutoff=config.flow_matching_cutoff,
                    time_offset=config.flow_matching_time_offset,
                    time_scale=config.flow_matching_time_scale,
                    alpha=config.flow_matching_beta_alpha,
                    beta=config.flow_matching_beta_beta,
                )
        timestep_rows.append(timestep_row)

        noise_generator = torch.Generator(device=actions.device)
        noise_generator.manual_seed(derive_subseed(sample.flow_noise_seed, *coordinates))
        noise_rows.append(
            torch.randn(
                (num_flow_timesteps, actions.shape[1], actions.shape[2]),
                device=actions.device,
                dtype=actions.dtype,
                generator=noise_generator,
            )
        )

    return torch.stack(timestep_rows), torch.stack(noise_rows)


def molmoact2_host_observation_view(
    record: RobotTransitionRecord,
) -> MolmoAct2HostObservationView:
    """Return the canonical deploy-visible host view for one CALVIN record."""

    return MolmoAct2HostObservationView(
        task=record.task,
        embodiment=record.embodiment,
        control_mode=record.control_mode,
        control_frame=record.control_frame,
        state_axes=record.state_axes,
        state_units=record.state_units,
        state=tuple(float(value) for value in record.state),
        state_valid=tuple(bool(value) for value in record.state_valid),
        timestamp_s=record.timestamp_s,
        delta_t_s=record.delta_t_s,
    )


def _validate_native_bank_causal_cutoff(
    native_banks: Sequence[NativeTokenBank],
    current_records: Sequence[RobotTransitionRecord],
) -> None:
    """Reject valid external evidence newer than its current CALVIN record."""

    if not current_records:
        raise ValueError("causal cutoff validation requires current CALVIN records")
    batch_size = len(current_records)
    for bank in native_banks:
        timestamps = bank.timestamps
        if timestamps is None:
            continue
        if (
            timestamps.shape != bank.valid.shape
            or timestamps.shape[0] != batch_size
            or timestamps.device != bank.tokens.device
            or timestamps.dtype not in {torch.float32, torch.float64}
        ):
            raise ValueError(
                f"{bank.modality} timestamps cannot be audited against the CALVIN cutoff"
            )
        cutoff = torch.as_tensor(
            [record.timestamp_s for record in current_records],
            device=timestamps.device,
            dtype=timestamps.dtype,
        )
        tolerance = 4.0 * torch.finfo(timestamps.dtype).eps * cutoff.abs().clamp_min(1.0)
        future = bank.valid & (timestamps > (cutoff + tolerance).unsqueeze(1))
        if future.any():
            raise ValueError(f"{bank.modality} evidence crosses the current causal cutoff")


def assemble_calvin_molmoact2_transitions(
    windows: Sequence[CalvinTrainingWindow],
    config: MolmoAct2PICFTrainingConfig,
    *,
    build_native_banks: CalvinNativeBankBuilder,
    build_host_batch: CalvinHostBatchBuilder,
    build_host_observation_inputs: CalvinHostObservationBuilder | None = None,
) -> tuple[MolmoAct2PICFTransition, ...]:
    """Assemble one causal batched sequence without exposing targets to encoders.

    `build_native_banks` receives only target-free causal sensor prefixes ending
    at the current transition; future observations are absent from its type-level
    boundary. The host batch
    callback receives only the current records on gradient-bearing transitions
    and is never forwarded into PICF. It may construct the official action
    target chunk by resolving each record's immutable source identity, but it
    cannot inspect future records through this API. This is the production
    boundary that prevents the current demonstrator target from being confused
    with the command that caused the current observation.
    """

    frozen_windows = tuple(windows)
    if not frozen_windows:
        raise ValueError("at least one CALVIN posterior window is required")
    if any(len(window.records) != config.sequence_length for window in frozen_windows):
        raise ValueError("every CALVIN window must match the configured sequence length")
    evidence_windows = tuple(window.picf_evidence_frames for window in frozen_windows)
    action_histories = tuple(window.previous_executed_actions for window in frozen_windows)
    batch_size = len(frozen_windows)
    transitions = []
    for time_index in range(config.sequence_length):
        causal_prefixes = tuple(
            evidence_window[: time_index + 1] for evidence_window in evidence_windows
        )
        native_banks = tuple(build_native_banks(causal_prefixes))
        if not native_banks:
            raise ValueError("CALVIN PICF transition requires at least one native token bank")
        reference = native_banks[0].tokens
        if any(
            bank.tokens.shape[0] != batch_size
            or bank.tokens.device != reference.device
            or bank.tokens.dtype != reference.dtype
            for bank in native_banks
        ):
            raise ValueError("native banks must share CALVIN batch size, dtype and device")
        previous_action = torch.stack(
            [
                torch.as_tensor(
                    history[time_index].copy(),
                    device=reference.device,
                    dtype=reference.dtype,
                )
                for history in action_histories
            ]
        )
        current_records = tuple(window.records[time_index] for window in frozen_windows)
        _validate_native_bank_causal_cutoff(native_banks, current_records)
        delta_t_s = torch.as_tensor(
            [record.delta_t_s for record in current_records],
            device=reference.device,
            dtype=reference.dtype,
        )
        host_batch = None
        host_observation_inputs = None
        if build_host_observation_inputs is not None:
            host_observations = tuple(
                molmoact2_host_observation_view(record) for record in current_records
            )
            host_observation_inputs = dict(
                build_host_observation_inputs(causal_prefixes, host_observations)
            )
            if not host_observation_inputs:
                raise ValueError("MolmoAct2 host observation inputs cannot be empty")
        if time_index >= config.detached_context_frames:
            host_batch = dict(build_host_batch(current_records))
            if not host_batch:
                raise ValueError("gradient transition host batch cannot be empty")
        transitions.append(
            MolmoAct2PICFTransition(
                native_banks=native_banks,
                previous_executed_action=previous_action,
                delta_t_s=delta_t_s,
                host_observation_inputs=host_observation_inputs,
                host_batch=host_batch,
            )
        )
    return tuple(transitions)


def assemble_calvin_stateful_molmoact2_transition(
    samples: Sequence[CalvinStatefulTransitionSample],
    *,
    native_banks: Sequence[NativeTokenBank],
    build_host_batch: CalvinStatefulHostBatchBuilder,
    build_host_observation_inputs: CalvinHostObservationBuilder | None = None,
    flow_timesteps: torch.Tensor | None = None,
    flow_noise: torch.Tensor | None = None,
    tensor_device: torch.device | str | None = None,
    tensor_dtype: torch.dtype | None = None,
) -> MolmoAct2PICFTransition:
    """Assemble one production stream step without replaying an observation prefix.

    Frozen extra-modality features may be resolved by immutable sample key before
    this function, but only the resulting :class:`NativeTokenBank` values cross
    the model boundary. The observation callback receives target-free current
    evidence, while the target callback receives typed stateful samples so it can
    use the already materialized action horizon without decoding sensor frames a
    second time.
    Crucially, the causal previous action comes from the stateful sample contract,
    not from an isolated one-record window that would reset it to zero every update.
    """

    frozen_samples = tuple(samples)
    frozen_banks = tuple(native_banks)
    if not frozen_samples:
        raise ValueError("at least one CALVIN stateful transition sample is required")
    batch_size = len(frozen_samples)
    if frozen_banks:
        reference = frozen_banks[0].tokens
        if any(
            bank.tokens.shape[0] != batch_size
            or bank.tokens.device != reference.device
            or bank.tokens.dtype != reference.dtype
            for bank in frozen_banks
        ):
            raise ValueError("native banks must share CALVIN batch size, dtype and device")
        if tensor_device is not None and reference.device != torch.device(tensor_device):
            raise ValueError("native banks differ from the requested posterior device")
        if tensor_dtype is not None and reference.dtype != tensor_dtype:
            raise ValueError("native banks differ from the requested posterior dtype")
        target_device = reference.device
        target_dtype = reference.dtype
    else:
        if build_host_observation_inputs is None:
            raise ValueError(
                "an empty external bank set requires same-forward Molmo observation inputs"
            )
        if tensor_device is None or tensor_dtype is None or not tensor_dtype.is_floating_point:
            raise ValueError(
                "same-forward-only assembly requires an explicit floating posterior "
                "device and dtype"
            )
        target_device = torch.device(tensor_device)
        target_dtype = tensor_dtype

    previous_action = torch.stack(
        [
            torch.as_tensor(
                sample.previous_executed_action.copy(),
                device=target_device,
                dtype=target_dtype,
            )
            for sample in frozen_samples
        ]
    )
    current_records = tuple(sample.record for sample in frozen_samples)
    _validate_native_bank_causal_cutoff(frozen_banks, current_records)
    current_evidence = tuple((sample.picf_evidence_frame,) for sample in frozen_samples)
    delta_t_s = torch.as_tensor(
        [record.delta_t_s for record in current_records],
        device=target_device,
        dtype=target_dtype,
    )
    host_observation_inputs = None
    if build_host_observation_inputs is not None:
        host_views = tuple(molmoact2_host_observation_view(record) for record in current_records)
        host_observation_inputs = dict(build_host_observation_inputs(current_evidence, host_views))
        if not host_observation_inputs:
            raise ValueError("MolmoAct2 host observation inputs cannot be empty")
    host_batch = dict(build_host_batch(frozen_samples))
    if not host_batch:
        raise ValueError("CALVIN stateful gradient transition host batch cannot be empty")
    return MolmoAct2PICFTransition(
        native_banks=frozen_banks,
        previous_executed_action=previous_action,
        delta_t_s=delta_t_s,
        host_observation_inputs=host_observation_inputs,
        host_batch=host_batch,
        flow_timesteps=flow_timesteps,
        flow_noise=flow_noise,
    )


def detach_object_belief(belief: ObjectBeliefBatch) -> ObjectBeliefBatch:
    """Detach every floating posterior field at a truncated-BPTT boundary."""

    return ObjectBeliefBatch(
        address_mean=belief.address_mean.detach(),
        content_mean=belief.content_mean.detach(),
        geometry_mean=belief.geometry_mean.detach(),
        geometry_covariance_diag=belief.geometry_covariance_diag.detach(),
        existence_logits=belief.existence_logits.detach(),
        visibility_given_existence_logits=(belief.visibility_given_existence_logits.detach()),
        measurement_age_s=belief.measurement_age_s.detach(),
        valid=belief.valid.detach(),
        age=belief.age.detach(),
    )


def action_evidence_from_core(
    output: PICFCoreOutput,
    *,
    direct_context_banks: tuple[NativeTokenBank, ...] = (),
    include_posterior: bool = True,
) -> PICFActionEvidence:
    """Expose posterior measurements plus non-correcting native context.

    A clip-conditioned feature can contain observations already summarized by
    the prior belief. It remains useful to the action expert, but treating it as
    another independent measurement would count that history twice. Such banks
    therefore bypass discovery and receive exact context/dustbin ownership.
    """

    if not isinstance(include_posterior, bool):
        raise TypeError("posterior action-context selection must be boolean")
    measurement_banks = output.projection.native_banks
    measurement_modalities = {bank.modality for bank in measurement_banks}
    if len(measurement_modalities) != len(measurement_banks):
        raise ValueError("core output contains duplicate measurement modalities")
    if any(bank.modality in measurement_modalities for bank in direct_context_banks):
        raise ValueError("direct context duplicates a posterior measurement modality")

    object_count = output.action_bank.address.shape[1]
    context_ownership = []
    for bank in direct_context_banks:
        current = bank.current_measurement_valid
        if current is None:
            raise ValueError("direct context requires an explicit non-measurement role")
        if current.dtype != torch.bool or current.shape != bank.valid.shape:
            raise ValueError("direct context role must be bool batch-by-token")
        if current.device != bank.tokens.device or bool(current.any()):
            raise ValueError("direct context cannot contain current posterior measurements")
        ownership = torch.zeros(
            (*bank.valid.shape, object_count + 1),
            device=bank.tokens.device,
            dtype=bank.tokens.dtype,
        )
        ownership[..., -1] = 1.0
        context_ownership.append(ownership)

    if not include_posterior:
        return PICFActionEvidence(
            dense_banks=(*measurement_banks, *direct_context_banks),
            object_address=None,
            object_value=None,
            object_valid=None,
            object_log_prior=None,
            dense_ownership=None,
        )

    return PICFActionEvidence(
        dense_banks=(*measurement_banks, *direct_context_banks),
        object_address=output.action_bank.address,
        object_value=output.action_bank.value,
        object_valid=output.action_bank.valid,
        object_log_prior=output.action_bank.log_prior,
        dense_ownership=(*output.dense_ownership, *context_ownership),
    )


class MolmoAct2PICFTrainingBridge(nn.Module):
    """Run one explicit posterior sequence through the official action loss.

    The policy must already contain a `MolmoAct2PICFActionExpert` installed by
    `install_molmoact2_lerobot_picf_adapter`. Structural target losses are
    intentionally computed by the caller from `core_outputs`; this bridge never
    receives them and cannot feed them into action context.
    """

    def __init__(
        self,
        policy: nn.Module,
        core: PICFCore,
        config: MolmoAct2PICFTrainingConfig,
    ) -> None:
        super().__init__()
        from picf_next.hosts.molmoact2 import MolmoAct2PICFActionExpert

        adapter = getattr(policy, "action_layer_adapter", None)
        if not isinstance(adapter, MolmoAct2PICFActionExpert):
            raise TypeError(
                "policy must contain an installed MolmoAct2PICFActionExpert before "
                "training-bridge construction"
            )
        if not callable(getattr(policy, "get_optim_params", None)):
            raise TypeError("policy does not expose the pinned LeRobot optimizer contract")
        self.policy = policy
        self.core = core
        self.config = config

    @property
    def action_adapter(self) -> MolmoAct2PICFActionExpert:
        from picf_next.hosts.molmoact2 import MolmoAct2PICFActionExpert

        adapter = getattr(self.policy, "action_layer_adapter", None)
        if not isinstance(adapter, MolmoAct2PICFActionExpert):
            raise RuntimeError("the installed PICF action adapter changed after construction")
        return adapter

    def _prepare_transition(
        self,
        transition: MolmoAct2PICFTransition,
        *,
        require_host_batch: bool,
    ) -> tuple[
        tuple[NativeTokenBank, ...],
        tuple[NativeTokenBank, ...],
        dict[str, torch.Tensor] | None,
        torch.Tensor | None,
        torch.Tensor | None,
        MolmoAct2VisionPatchLayout | None,
        torch.LongTensor | None,
    ]:
        """Resolve one observation without allowing target fields into PICF."""

        native_banks = transition.native_banks
        prepared_inputs: Mapping[str, torch.Tensor] | None = None
        action_condition_input_ids = None
        vision_patch_layout = None
        if transition.host_observation_inputs is not None:
            from picf_next.hosts.molmoact2 import prepare_molmoact2_lerobot_observation

            prepared = prepare_molmoact2_lerobot_observation(
                self.policy,
                transition.host_observation_inputs,
            )
            prepared_inputs = prepared.model_inputs
            action_condition_input_ids = prepared.action_condition_input_ids
            vision_patch_layout = prepared.vision_patch_layout
            if prepared.vision_patch_bank is not None:
                if any(
                    bank.modality == prepared.vision_patch_bank.modality for bank in native_banks
                ):
                    raise ValueError(
                        "Molmo native vision patches appear in both external and same-forward banks"
                    )
                native_banks = (*native_banks, prepared.vision_patch_bank)
        if not native_banks:
            raise ValueError(
                "PICF transition requires an external bank or a same-forward Molmo vision bank"
            )

        flow_timesteps = transition.flow_timesteps
        flow_noise = transition.flow_noise
        if (flow_timesteps is None) != (flow_noise is None):
            raise ValueError("flow_timesteps and flow_noise must be supplied together")
        if not require_host_batch:
            if flow_timesteps is not None:
                raise ValueError("detached context transitions cannot carry flow randomness")
            if transition.host_batch is not None:
                raise ValueError("detached context transitions cannot carry an ignored host_batch")
            measurement_banks, direct_context_banks = self._partition_native_banks(native_banks)
            return (
                measurement_banks,
                direct_context_banks,
                None,
                None,
                None,
                vision_patch_layout,
                None,
            )
        if transition.host_batch is None:
            raise ValueError("gradient transition requires an official MolmoAct2 host_batch")
        if self.config.require_explicit_flow_randomness and flow_timesteps is None:
            raise ValueError("gradient transition requires explicit flow randomness")
        host_batch = dict(transition.host_batch)
        if prepared_inputs is not None:
            collisions = sorted(set(host_batch) & set(prepared_inputs))
            if collisions:
                raise ValueError(
                    f"host targets duplicate prepared deploy-time fields: {collisions}"
                )
            raw_visual = sorted(
                set(host_batch)
                & {
                    "input_ids",
                    "pixel_values",
                    "image_token_pooling",
                    "image_grids",
                    "image_num_crops",
                    "pixel_values_videos",
                    "video_token_pooling",
                    "video_grids",
                    "attention_mask",
                    "position_ids",
                    "token_type_ids",
                    "inputs_embeds",
                }
            )
            if raw_visual:
                raise ValueError(
                    "host_batch must contain targets only when separate observation inputs are "
                    f"provided, got {raw_visual}"
                )
            host_batch.update(prepared_inputs)
        measurement_banks, direct_context_banks = self._partition_native_banks(native_banks)
        return (
            measurement_banks,
            direct_context_banks,
            host_batch,
            flow_timesteps,
            flow_noise,
            vision_patch_layout,
            action_condition_input_ids,
        )

    def _partition_native_banks(
        self,
        native_banks: tuple[NativeTokenBank, ...],
    ) -> tuple[tuple[NativeTokenBank, ...], tuple[NativeTokenBank, ...]]:
        """Separate Bayesian measurements from read-only action context."""

        configured = set(self.core.projector.specs)
        modalities = tuple(bank.modality for bank in native_banks)
        if len(set(modalities)) != len(modalities):
            raise ValueError("PICF transition contains duplicate native modalities")
        measurement = tuple(bank for bank in native_banks if bank.modality in configured)
        direct_context = tuple(bank for bank in native_banks if bank.modality not in configured)
        if not measurement:
            raise ValueError("PICF transition contains no configured posterior measurement bank")
        configured_action_modalities = set(self.action_adapter.dense_k_proj)
        unknown = sorted(
            bank.modality
            for bank in direct_context
            if bank.modality not in configured_action_modalities
        )
        if unknown:
            raise ValueError(f"direct action context modalities are not configured: {unknown}")
        for bank in direct_context:
            current = bank.current_measurement_valid
            if current is None:
                raise ValueError(
                    f"direct action context {bank.modality} requires an explicit role mask"
                )
            if (
                current.dtype != torch.bool
                or current.shape != bank.valid.shape
                or current.device != bank.tokens.device
                or bool(current.any())
            ):
                raise ValueError(
                    f"direct action context {bank.modality} cannot update the posterior"
                )
            timestamps = bank.timestamps
            if (
                timestamps is None
                or timestamps.shape != bank.valid.shape
                or timestamps.device != bank.tokens.device
                or timestamps.dtype not in {torch.float32, torch.float64}
            ):
                raise ValueError(
                    f"direct action context {bank.modality} requires auditable timestamps"
                )
        return measurement, direct_context

    def forward(
        self,
        transitions: Sequence[MolmoAct2PICFTransition],
        initial_belief: ObjectBeliefBatch,
    ) -> MolmoAct2PICFTrainingOutput:
        if len(transitions) != self.config.sequence_length:
            raise ValueError(
                "transition count must equal detached_context_frames + gradient_transitions"
            )

        belief = detach_object_belief(initial_belief)
        split = self.config.detached_context_frames
        for transition in transitions[:split]:
            with torch.no_grad():
                native_banks, _, _, _, _, _, _ = self._prepare_transition(
                    transition,
                    require_host_batch=False,
                )
                warmup = self.core(
                    native_banks,
                    belief,
                    transition.previous_executed_action,
                    transition.delta_t_s,
                )
            belief = detach_object_belief(warmup.posterior.belief)

        action_losses: list[torch.Tensor] = []
        core_outputs: list[PICFCoreOutput] = []
        evidences: list[PICFActionEvidence] = []
        vision_patch_layouts: list[MolmoAct2VisionPatchLayout | None] = []
        metric_sums: dict[str, float] = {}
        gradient_transitions = transitions[split:]
        for index, transition in enumerate(gradient_transitions):
            (
                native_banks,
                direct_context_banks,
                host_batch,
                flow_timesteps,
                flow_noise,
                vision_patch_layout,
                action_condition_input_ids,
            ) = self._prepare_transition(
                transition,
                require_host_batch=True,
            )
            if host_batch is None:
                raise RuntimeError(f"gradient transition {index} lost its host batch")
            output = self.core(
                native_banks,
                belief,
                transition.previous_executed_action,
                transition.delta_t_s,
            )
            evidence = action_evidence_from_core(
                output,
                direct_context_banks=direct_context_banks,
                include_posterior=self.config.include_posterior_action_context,
            )
            context = self.action_adapter.prepare_picf_context(evidence)
            action_loss, metrics = self.policy(
                host_batch,
                reduction="mean",
                action_layer_context=context,
                flow_timesteps=flow_timesteps,
                flow_noise=flow_noise,
                action_condition_input_ids=action_condition_input_ids,
            )
            if action_loss.ndim != 0 or not torch.isfinite(action_loss):
                raise ValueError("official MolmoAct2 mean action loss must be one finite scalar")
            action_losses.append(action_loss)
            core_outputs.append(output)
            evidences.append(evidence)
            vision_patch_layouts.append(vision_patch_layout)
            belief = output.posterior.belief
            for name, value in metrics.items():
                if isinstance(value, bool):
                    raise ValueError(f"MolmoAct2 metric {name!r} cannot be boolean")
                if isinstance(value, int | float):
                    numeric = float(value)
                    if not math.isfinite(numeric):
                        raise ValueError(f"MolmoAct2 metric {name!r} must be finite")
                    metric_sums[name] = metric_sums.get(name, 0.0) + numeric

        loss = torch.stack(action_losses).mean()
        denominator = float(len(action_losses))
        averaged_metrics = {
            name: value / denominator for name, value in sorted(metric_sums.items())
        }
        averaged_metrics["picf_sequence_action_loss"] = float(loss.detach().float().item())
        return MolmoAct2PICFTrainingOutput(
            loss=loss,
            metrics=averaged_metrics,
            action_losses=tuple(action_losses),
            core_outputs=tuple(core_outputs),
            evidences=tuple(evidences),
            vision_patch_layouts=tuple(vision_patch_layouts),
            final_belief=belief,
        )

    def get_optim_params(self) -> list[dict[str, Any]]:
        """Return complete, disjoint host/adapter/core optimizer groups."""

        raw_groups = self.policy.get_optim_params()
        groups: list[dict[str, Any]] = []
        grouped_ids: set[int] = set()
        for raw_group in raw_groups:
            group = dict(raw_group)
            parameters = list(group.get("params", ()))
            if not parameters:
                continue
            parameter_ids = {id(parameter) for parameter in parameters}
            if len(parameter_ids) != len(parameters) or grouped_ids & parameter_ids:
                raise RuntimeError("MolmoAct2 optimizer groups contain duplicate parameters")
            grouped_ids.update(parameter_ids)
            group["params"] = parameters
            groups.append(group)

        core_parameters = [
            parameter for parameter in self.core.parameters() if parameter.requires_grad
        ]
        core_ids = {id(parameter) for parameter in core_parameters}
        if len(core_ids) != len(core_parameters) or grouped_ids & core_ids:
            raise RuntimeError("PICF core parameters overlap host optimizer groups")
        if core_parameters:
            groups.append({"params": core_parameters, "lr": self.config.picf_core_lr})
            grouped_ids.update(core_ids)

        expected_ids = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if grouped_ids != expected_ids:
            missing = len(expected_ids - grouped_ids)
            unexpected = len(grouped_ids - expected_ids)
            raise RuntimeError(
                f"optimizer coverage mismatch: missing={missing}, unexpected={unexpected}"
            )
        return groups


class MolmoAct2PICFJointTrainingBridge(nn.Module):
    """Consume official action and structural targets without a second forward.

    Object targets are accepted only after the deploy-visible sequence bridge
    has produced its outputs. They cannot affect discovery, posterior
    association, native host embeddings or the official action trajectory.
    """

    def __init__(
        self,
        sequence_bridge: MolmoAct2PICFTrainingBridge,
        objective: PICFObjective,
    ) -> None:
        super().__init__()
        self.sequence_bridge = sequence_bridge
        core_parameter = next(sequence_bridge.core.parameters())
        # Pair-density calibration follows official SigLIP and remains float32;
        # only its device follows the bf16 host/core.
        self.objective = objective.to(device=core_parameter.device)

    def forward(
        self,
        transitions: Sequence[MolmoAct2PICFTransition],
        initial_belief: ObjectBeliefBatch,
        *,
        set_targets: Sequence[Sequence[ObjectSetTarget]] | None,
        lifecycle_targets: Sequence[Sequence[ObjectLifecycleInventoryTarget | None]] | None = None,
        initial_loss_track_keys_by_row: Sequence[Sequence[str | None]] | None = None,
        geometry_rollout_target: ObjectGeometryRolloutTarget | None = None,
    ) -> MolmoAct2PICFJointTrainingOutput:
        sequence = self.sequence_bridge(transitions, initial_belief)
        objective = self.objective(
            sequence.core_outputs,
            action_loss=sequence.loss,
            set_targets=set_targets,
            lifecycle_targets=lifecycle_targets,
            initial_loss_track_keys_by_row=initial_loss_track_keys_by_row,
            geometry_rollout_target=geometry_rollout_target,
            transition=self.sequence_bridge.core.posterior_filter.transition,
        )
        return MolmoAct2PICFJointTrainingOutput(
            sequence=sequence,
            objective=objective,
        )

    def get_optim_params(self) -> list[dict[str, Any]]:
        groups = self.sequence_bridge.get_optim_params()
        grouped_ids = {id(parameter) for group in groups for parameter in group["params"]}
        objective_parameters = [
            parameter for parameter in self.objective.parameters() if parameter.requires_grad
        ]
        objective_ids = {id(parameter) for parameter in objective_parameters}
        if len(objective_ids) != len(objective_parameters) or grouped_ids & objective_ids:
            raise RuntimeError("PICF objective parameters overlap existing optimizer groups")
        if objective_parameters:
            groups.append(
                {
                    "params": objective_parameters,
                    "lr": self.sequence_bridge.config.picf_core_lr,
                }
            )
            grouped_ids.update(objective_ids)
        expected_ids = {id(parameter) for parameter in self.parameters() if parameter.requires_grad}
        if grouped_ids != expected_ids:
            missing = len(expected_ids - grouped_ids)
            unexpected = len(grouped_ids - expected_ids)
            raise RuntimeError(
                f"joint optimizer coverage mismatch: missing={missing}, unexpected={unexpected}"
            )
        return groups


class CalvinStatefulMolmoAct2TrainingModule(nn.Module):
    """Close the production one-transition CALVIN/MolmoAct2/PICF loop.

    The frozen plan resolves only immutable sample identity and randomness.  The
    native-bank callback receives a target-free evidence request. Structural
    labels are constructed by a separate callback *after* the action/discovery
    forward, then enter only the explicit objective. The typed request excludes
    actions, task text, sensor payloads and predictions. This closes the direct
    interface path; builders remain auditable code and must resolve only the
    preregistered sidecar by immutable sample identity.
    """

    def __init__(
        self,
        dataset: CalvinStatefulTransitionDataset,
        joint_bridge: MolmoAct2PICFJointTrainingBridge,
        *,
        build_host_batch: CalvinStatefulHostBatchBuilder,
        build_host_observation_inputs: CalvinHostObservationBuilder | None = None,
        build_native_banks: CalvinStatefulNativeBankBuilder | None = None,
        build_loss_targets: CalvinStatefulLossTargetBuilder | None = None,
        native_evidence_history_frames: int = 1,
    ) -> None:
        super().__init__()
        if not isinstance(dataset, CalvinStatefulTransitionDataset):
            raise TypeError("dataset must be a CalvinStatefulTransitionDataset")
        if not isinstance(joint_bridge, MolmoAct2PICFJointTrainingBridge):
            raise TypeError("joint_bridge must be a MolmoAct2PICFJointTrainingBridge")
        if build_native_banks is not None and not callable(build_native_banks):
            raise TypeError("stateful native-bank builder must be callable when supplied")
        if (
            not isinstance(native_evidence_history_frames, int)
            or isinstance(native_evidence_history_frames, bool)
            or native_evidence_history_frames <= 0
        ):
            raise ValueError("native_evidence_history_frames must be a positive integer")
        if build_native_banks is None and native_evidence_history_frames != 1:
            raise ValueError("native evidence history requires a native-bank builder")
        if not callable(build_host_batch):
            raise TypeError("stateful host-batch builder must be callable")
        if not callable(build_host_observation_inputs):
            raise TypeError(
                "stateful production training requires a separate target-free "
                "host observation builder"
            )
        if build_loss_targets is not None and not callable(build_loss_targets):
            raise TypeError("loss-target builder must be callable")
        config = joint_bridge.sequence_bridge.config
        if config.detached_context_frames != 0 or config.gradient_transitions != 1:
            raise ValueError(
                "stateful production training requires zero replay frames and one transition"
            )
        policy_config = getattr(joint_bridge.sequence_bridge.policy, "config", None)
        if getattr(policy_config, "action_mode", None) != "continuous":
            raise ValueError(
                "stateful production training currently supports only continuous MolmoAct2 "
                "actions so demonstrator action tokens cannot enter the observation prefix"
            )
        objective = joint_bridge.objective
        dynamics_config = objective.dynamics_criterion.config
        lifecycle_supervision_active = objective.config.dynamics_weight > 0.0 and (
            dynamics_config.survival_weight > 0.0 or dynamics_config.visibility_weight > 0.0
        )
        geometry_overshooting_active = objective.geometry_overshooting_criterion.config.weight > 0.0
        requires_loss_targets = (
            objective.config.set_weight > 0.0
            or objective.config.binding_weight > 0.0
            or lifecycle_supervision_active
            or geometry_overshooting_active
        )
        if requires_loss_targets and build_loss_targets is None:
            raise ValueError("active structural objectives require a post-forward target builder")

        self.dataset = dataset
        self.joint_bridge = joint_bridge
        self.build_native_banks = build_native_banks
        self.native_evidence_history_frames = native_evidence_history_frames
        self.build_host_batch = build_host_batch
        self.build_host_observation_inputs = build_host_observation_inputs
        self.build_loss_targets = build_loss_targets

    def forward(
        self,
        microbatch: PlannedStreamMicrobatch,
        initial_belief: ObjectBeliefBatch,
        initial_loss_track_keys_by_row: tuple[tuple[str | None, ...], ...],
    ) -> StatefulForwardOutput:
        if not isinstance(microbatch, PlannedStreamMicrobatch):
            raise TypeError("stateful CALVIN forward requires a PlannedStreamMicrobatch")
        planned_transitions = tuple(microbatch.transitions)
        if not planned_transitions:
            raise ValueError("stateful CALVIN microbatch cannot be empty")
        if initial_belief.valid.shape[0] != len(planned_transitions):
            raise ValueError("initial posterior batch differs from the planned CALVIN batch")
        if len(initial_loss_track_keys_by_row) != len(planned_transitions):
            raise ValueError("initial loss tracks differ from the planned CALVIN batch")

        samples: list[CalvinStatefulTransitionSample] = []
        evidence_requests: list[CalvinStatefulEvidenceRequest] = []
        loss_target_requests: list[CalvinStatefulLossTargetRequest] = []
        for planned in planned_transitions:
            sample = self.dataset.by_key(planned.sample.sample_key)
            if sample.episode_key != planned.episode_key:
                raise ValueError("planned CALVIN episode key differs from the dataset manifest")
            if sample.transition_index != planned.transition_index:
                raise ValueError("planned CALVIN transition index differs from the dataset sample")
            samples.append(sample)
            evidence_requests.append(
                CalvinStatefulEvidenceRequest(
                    sample_key=sample.sample_key,
                    augmentation_seed=planned.sample.augmentation_seed,
                    evidence_prefix=self.dataset.evidence_prefix_by_key(
                        sample.sample_key,
                        maximum_source_frames=self.native_evidence_history_frames,
                    ),
                )
            )
            loss_target_requests.append(
                calvin_visible_object_target_request(
                    sample,
                    augmentation_seed=planned.sample.augmentation_seed,
                )
            )

        frozen_samples = tuple(samples)
        native_banks = (
            tuple(self.build_native_banks(tuple(evidence_requests)))
            if self.build_native_banks is not None
            else ()
        )
        transition = assemble_calvin_stateful_molmoact2_transition(
            frozen_samples,
            native_banks=native_banks,
            build_host_batch=self.build_host_batch,
            build_host_observation_inputs=self.build_host_observation_inputs,
            tensor_device=initial_belief.address_mean.device,
            tensor_dtype=initial_belief.address_mean.dtype,
        )
        if transition.host_batch is None:
            raise RuntimeError("stateful CALVIN transition lost its action target batch")
        unexpected_target_keys = sorted(set(transition.host_batch) - _CONTINUOUS_ACTION_TARGET_KEYS)
        if unexpected_target_keys:
            raise ValueError(
                "stateful MolmoAct2 target batch contains non-target fields: "
                f"{unexpected_target_keys}"
            )
        if "action" not in transition.host_batch:
            raise ValueError("stateful MolmoAct2 target batch requires action")
        if self.joint_bridge.sequence_bridge.config.require_explicit_flow_randomness:
            actions = transition.host_batch.get("action")
            if not isinstance(actions, torch.Tensor):
                raise ValueError(
                    "explicit MolmoAct2 flow randomness requires tensor action targets"
                )
            flow_timesteps, flow_noise = materialize_molmoact2_flow_randomness(
                self.joint_bridge.sequence_bridge.policy,
                tuple(planned.sample for planned in planned_transitions),
                actions,
                transition_index=tuple(planned.transition_index for planned in planned_transitions),
            )
            transition = replace(
                transition,
                flow_timesteps=flow_timesteps,
                flow_noise=flow_noise,
            )

        # Deliberately complete the deploy-visible path before resolving masks,
        # simulator identities or other structural supervision.
        sequence = self.joint_bridge.sequence_bridge((transition,), initial_belief)
        projection = sequence.core_outputs[0].projection
        target_layout = CalvinStatefulLossTargetLayout(
            token_valid=projection.current_measurement_valid.detach().clone(),
            spans=projection.spans,
            target_dtype=torch.float32,
            rollout_input_dtype=(sequence.core_outputs[-1].posterior.belief.address_mean.dtype),
            vision_patch_layout=sequence.vision_patch_layouts[0],
        )
        loss_targets = (
            self.build_loss_targets(tuple(loss_target_requests), target_layout)
            if self.build_loss_targets is not None
            else CalvinStatefulLossTargets()
        )
        if not isinstance(loss_targets, CalvinStatefulLossTargets):
            raise TypeError("loss-target builder must return CalvinStatefulLossTargets")
        set_targets = (loss_targets.set_targets,) if loss_targets.set_targets is not None else None
        lifecycle_targets = (
            (loss_targets.lifecycle_targets,)
            if loss_targets.lifecycle_targets is not None
            else None
        )
        objective = self.joint_bridge.objective(
            sequence.core_outputs,
            action_loss=sequence.loss,
            set_targets=set_targets,
            lifecycle_targets=lifecycle_targets,
            initial_loss_track_keys_by_row=initial_loss_track_keys_by_row,
            geometry_rollout_target=loss_targets.geometry_rollout_target,
            transition=(self.joint_bridge.sequence_bridge.core.posterior_filter.transition),
        )
        metrics = dict(sequence.metrics)
        for name, value in objective.losses.items():
            if value.ndim != 0 or not torch.isfinite(value):
                raise ValueError(f"PICF objective metric {name!r} must be one finite scalar")
            metrics[f"picf_{name}"] = float(value.detach().float().item())
        action_adapter = self.joint_bridge.sequence_bridge.action_adapter
        for domain, gates in (
            ("dense", action_adapter.dense_gates),
            ("object", action_adapter.object_gates),
        ):
            detached_gates = gates.detach().float()
            if not torch.isfinite(detached_gates).all():
                raise ValueError(f"PICF {domain} residual gates must be finite")
            metrics[f"picf_{domain}_gate_abs_mean"] = float(detached_gates.abs().mean().item())
            metrics[f"picf_{domain}_gate_abs_max"] = float(detached_gates.abs().max().item())
        posterior_outputs = tuple(output.posterior for output in sequence.core_outputs)
        transition_predictions = tuple(output.prior_prediction for output in posterior_outputs)
        transition_valid = torch.cat(
            [prediction.belief.valid.reshape(-1) for prediction in transition_predictions]
        ).float()
        transition_row_count = transition_valid.sum()
        transition_normalizer = transition_row_count.clamp_min(1.0)
        conditional_detection = torch.cat(
            [
                prediction.conditional_detection_probability.detach().float().reshape(-1)
                for prediction in transition_predictions
            ]
        )
        stored_rows = torch.stack(
            [output.belief.valid.float().sum(dim=-1).mean() for output in posterior_outputs]
        ).mean()
        map_rows = torch.stack(
            [output.map_present.float().sum(dim=-1).mean() for output in posterior_outputs]
        ).mean()
        support_births = torch.stack(
            [output.born.float().sum(dim=-1).mean() for output in posterior_outputs]
        ).mean()
        map_births = torch.stack(
            [
                (output.born & output.map_present).float().sum(dim=-1).mean()
                for output in posterior_outputs
            ]
        ).mean()
        tentative_ownership_leak = torch.stack(
            [
                output.ownership[..., :-1]
                .masked_fill(output.map_present.unsqueeze(1), 0.0)
                .abs()
                .amax()
                for output in posterior_outputs
            ]
        ).amax()
        relation = posterior_outputs[-1]
        discoveries = tuple(output.discovery for output in sequence.core_outputs)
        query_existence = torch.cat(
            [discovery.existence.detach().float().reshape(-1) for discovery in discoveries]
        )
        query_localization_confidence = torch.cat(
            [
                discovery.localization_confidence.detach().float().reshape(-1)
                for discovery in discoveries
            ]
        )
        query_measurement_probability = torch.cat(
            [
                discovery.measurement_probability.detach().float().reshape(-1)
                for discovery in discoveries
            ]
        )
        query_mask_quality = torch.cat(
            [discovery.mask_quality.detach().float().reshape(-1) for discovery in discoveries]
        )
        query_mask_coherence = torch.cat(
            [
                discovery.mask_coherence_score.detach().float().reshape(-1)
                for discovery in discoveries
            ]
        )
        metrics.update(
            {
                "picf_posterior_stored_rows": float(stored_rows.item()),
                "picf_posterior_map_rows": float(map_rows.item()),
                "picf_posterior_tentative_rows": float((stored_rows - map_rows).item()),
                "picf_posterior_support_births": float(support_births.item()),
                "picf_posterior_map_births": float(map_births.item()),
                "picf_posterior_tentative_ownership_leak_max": float(
                    tentative_ownership_leak.item()
                ),
                "picf_detection_probability_rows": float(transition_row_count.item()),
                "picf_conditional_detection_probability_mean": float(
                    (
                        (conditional_detection * transition_valid).sum() / transition_normalizer
                    ).item()
                ),
                "picf_address_relation_logit_scale": float(
                    relation.address_relation_logit_scale.detach().float().item()
                ),
                "picf_address_relation_logit_bias": float(
                    relation.address_relation_logit_bias.detach().float().item()
                ),
                "picf_query_existence_mean": float(query_existence.mean().item()),
                "picf_query_localization_confidence_mean": float(
                    query_localization_confidence.mean().item()
                ),
                "picf_query_localization_confidence_min": float(
                    query_localization_confidence.min().item()
                ),
                "picf_query_localization_confidence_max": float(
                    query_localization_confidence.max().item()
                ),
                "picf_query_measurement_probability_mean": float(
                    query_measurement_probability.mean().item()
                ),
                "picf_query_mask_quality_mean": float(query_mask_quality.mean().item()),
                "picf_query_mask_coherence_score_mean": float(query_mask_coherence.mean().item()),
            }
        )
        metrics.update(
            {f"picf_{name}": float(value) for name, value in objective.diagnostics.items()}
        )
        return StatefulForwardOutput(
            loss=objective.loss,
            final_belief=sequence.final_belief,
            metrics=metrics,
            final_loss_track_keys_by_row=objective.loss_track_keys_by_row,
        )

    def get_optim_params(self) -> list[dict[str, Any]]:
        return self.joint_bridge.get_optim_params()

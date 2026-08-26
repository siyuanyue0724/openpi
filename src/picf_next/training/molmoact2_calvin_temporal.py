"""Task-independent CALVIN batches for stationary MolmoAct2 PICF training."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_loss_targets import CalvinSourceFrameLossTargetRequest
from picf_next.data.calvin_rollout_targets import (
    CalvinSourcePhysicalGeometryProvider,
    build_calvin_source_geometry_rollout_sample,
)
from picf_next.data.molmoact2_source_cache import MolmoAct2SourceFeatureCache
from picf_next.data.rollout_targets import build_object_geometry_rollout_target
from picf_next.geometry import PhysicalGeometryContract
from picf_next.hosts.molmoact2_training import (
    CalvinStatefulLossTargetLayout,
    CalvinStatefulLossTargets,
)
from picf_next.models.dynamics_loss import ObjectGeometryRolloutTarget
from picf_next.models.evidence import ModalityTokenSpan
from picf_next.training.stationary_temporal import (
    StationaryTemporalObservation,
    StationaryTemporalSupervision,
)
from picf_next.training.temporal_clips import StationaryTemporalClip

CalvinSourceVisibleTargetBuilder = Callable[
    [tuple[CalvinSourceFrameLossTargetRequest, ...], CalvinStatefulLossTargetLayout],
    CalvinStatefulLossTargets,
]


@dataclass(frozen=True, slots=True)
class CalvinStationaryTemporalBatch:
    """Observation clip plus callbacks that materialize labels only post-forward."""

    observations: tuple[StationaryTemporalObservation, ...]
    source_indices_by_frame: tuple[tuple[int, ...], ...]
    requests_by_frame: tuple[tuple[CalvinSourceFrameLossTargetRequest, ...], ...]
    layouts: tuple[CalvinStatefulLossTargetLayout, ...]
    prefix_length: int
    train_length: int
    visible_target_builder: CalvinSourceVisibleTargetBuilder
    index: CalvinDatasetIndex
    geometry_contract: PhysicalGeometryContract
    geometry_provider: CalvinSourcePhysicalGeometryProvider
    maximum_horizon: int
    supervised_horizons: tuple[int, ...]

    def __post_init__(self) -> None:
        frame_count = self.prefix_length + self.train_length
        if (
            frame_count <= 0
            or len(self.observations) != frame_count
            or len(self.source_indices_by_frame) != frame_count
            or len(self.requests_by_frame) != frame_count
            or len(self.layouts) != frame_count
        ):
            raise ValueError("CALVIN stationary batch frame planes must align exactly")

    def build_supervision(self, frame_index: int) -> StationaryTemporalSupervision:
        """Resolve current physical labels after the corresponding core call."""

        if (
            not isinstance(frame_index, int)
            or isinstance(frame_index, bool)
            or not 0 <= frame_index < len(self.observations)
        ):
            raise ValueError("CALVIN stationary supervision frame is out of range")
        targets = self.visible_target_builder(
            self.requests_by_frame[frame_index],
            self.layouts[frame_index],
        )
        if not isinstance(targets, CalvinStatefulLossTargets):
            raise TypeError("CALVIN source visible builder returned an invalid target bundle")
        if targets.set_targets is None or targets.lifecycle_targets is None:
            raise ValueError("CALVIN source visible builder omitted structural supervision")
        if targets.geometry_rollout_target is not None:
            raise ValueError("current-frame visible builder cannot inject a future rollout")
        return StationaryTemporalSupervision(
            set_targets=targets.set_targets,
            lifecycle_targets=targets.lifecycle_targets,
        )

    def build_geometry_rollout(self) -> ObjectGeometryRolloutTarget:
        """Build future physical labels from the final train frame only."""

        final_indices = self.source_indices_by_frame[-1]
        samples = tuple(
            build_calvin_source_geometry_rollout_sample(
                self.index,
                global_index=global_index,
                maximum_horizon=self.maximum_horizon,
                supervised_horizons=self.supervised_horizons,
                geometry_contract=self.geometry_contract,
                geometry_provider=self.geometry_provider,
            )
            for global_index in final_indices
        )
        reference = self.observations[-1].native_banks[0].tokens
        return build_object_geometry_rollout_target(
            samples,
            action_dim=7,
            geometry_contract=self.geometry_contract,
            device=reference.device,
            input_dtype=reference.dtype,
            target_dtype=torch.float32,
        )


class CalvinStationaryTemporalBatchBuilder:
    """Build causal reset/replay batches without decoding task annotations."""

    def __init__(
        self,
        index: CalvinDatasetIndex,
        cache: MolmoAct2SourceFeatureCache,
        *,
        visible_target_builder: CalvinSourceVisibleTargetBuilder,
        geometry_contract: PhysicalGeometryContract,
        geometry_provider: CalvinSourcePhysicalGeometryProvider,
        maximum_horizon: int,
        supervised_horizons: Sequence[int],
    ) -> None:
        if not isinstance(index, CalvinDatasetIndex):
            raise TypeError("CALVIN stationary batches require a CalvinDatasetIndex")
        if not isinstance(cache, MolmoAct2SourceFeatureCache):
            raise TypeError("CALVIN stationary batches require a Molmo source feature cache")
        if not callable(visible_target_builder):
            raise TypeError("CALVIN stationary visible-target builder must be callable")
        if not isinstance(geometry_contract, PhysicalGeometryContract):
            raise TypeError("CALVIN stationary batches require a geometry contract")
        if not callable(geometry_provider):
            raise TypeError("CALVIN stationary geometry provider must be callable")
        if (
            not isinstance(maximum_horizon, int)
            or isinstance(maximum_horizon, bool)
            or maximum_horizon <= 0
        ):
            raise ValueError("CALVIN stationary rollout horizon must be positive")
        horizons = tuple(supervised_horizons)
        if (
            not horizons
            or any(
                not isinstance(horizon, int) or isinstance(horizon, bool) or horizon <= 0
                for horizon in horizons
            )
            or horizons != tuple(sorted(set(horizons)))
            or horizons[0] != 1
            or horizons[-1] > maximum_horizon
        ):
            raise ValueError("CALVIN stationary horizons must be sorted unique and include one")
        self.index = index
        self.cache = cache
        self.visible_target_builder = visible_target_builder
        self.geometry_contract = geometry_contract
        self.geometry_provider = geometry_provider
        self.maximum_horizon = maximum_horizon
        self.supervised_horizons = horizons

    def build(
        self,
        clips: Sequence[StationaryTemporalClip],
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.bfloat16,
    ) -> CalvinStationaryTemporalBatch:
        frozen_clips = tuple(clips)
        if not frozen_clips or any(
            not isinstance(clip, StationaryTemporalClip) for clip in frozen_clips
        ):
            raise TypeError("CALVIN stationary batch requires nonempty temporal clips")
        prefix_length = frozen_clips[0].prefix_length
        train_length = frozen_clips[0].train_length
        if any(
            clip.prefix_length != prefix_length or clip.train_length != train_length
            for clip in frozen_clips
        ):
            raise ValueError("batched CALVIN clips must share prefix and train lengths")
        for clip in frozen_clips:
            first_episode = self.index.source_episode(clip.start_global_index)
            final_episode = self.index.source_episode(clip.stop_global_index - 1)
            if first_episode.index != final_episode.index:
                raise ValueError("CALVIN stationary clip crosses a source-episode reset boundary")
            if clip.stop_global_index - 1 >= final_episode.end:
                raise ValueError("CALVIN stationary train suffix leaves no future rollout frame")
            for global_index in range(clip.start_global_index, clip.stop_global_index):
                self.cache.record(global_index)

        observations = []
        indices_by_frame = []
        requests_by_frame = []
        layouts = []
        for frame_offset in range(prefix_length + train_length):
            indices = tuple(clip.start_global_index + frame_offset for clip in frozen_clips)
            bank = self.cache.native_bank(indices, device=device, dtype=dtype)
            if frame_offset == 0:
                previous_action = torch.zeros(
                    len(frozen_clips),
                    7,
                    device=device,
                    dtype=dtype,
                )
            else:
                previous_action = torch.as_tensor(
                    np.stack([self.index.action(global_index - 1) for global_index in indices]),
                    device=device,
                    dtype=dtype,
                )
            delta_t = torch.full(
                (len(frozen_clips),),
                1.0 / float(self.index.control_hz),
                device=device,
                dtype=dtype,
            )
            observations.append(
                StationaryTemporalObservation(
                    native_banks=(bank,),
                    previous_executed_action=previous_action,
                    delta_t_s=delta_t,
                )
            )
            indices_by_frame.append(indices)
            requests_by_frame.append(tuple(self.cache.target_request(value) for value in indices))
            layouts.append(
                CalvinStatefulLossTargetLayout(
                    token_valid=bank.valid.detach().clone(),
                    spans=(ModalityTokenSpan(bank.modality, 0, self.cache.token_count),),
                    target_dtype=torch.float32,
                    rollout_input_dtype=dtype,
                    vision_patch_layout=self.cache.vision_layout(len(frozen_clips)),
                )
            )
        return CalvinStationaryTemporalBatch(
            observations=tuple(observations),
            source_indices_by_frame=tuple(indices_by_frame),
            requests_by_frame=tuple(requests_by_frame),
            layouts=tuple(layouts),
            prefix_length=prefix_length,
            train_length=train_length,
            visible_target_builder=self.visible_target_builder,
            index=self.index,
            geometry_contract=self.geometry_contract,
            geometry_provider=self.geometry_provider,
            maximum_horizon=self.maximum_horizon,
            supervised_horizons=self.supervised_horizons,
        )

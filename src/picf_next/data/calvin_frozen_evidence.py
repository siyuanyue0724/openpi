"""Content-addressed frozen full-modal evidence on CALVIN's physical time axis.

This module performs deterministic sensor preparation only. It contains no
task scorer, object identity, anchor ownership or posterior lifecycle logic.
Those relations remain learned inside the shared LingBot host.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass

import numpy as np

from picf_next.content_addressing import canonical_payload_sha256, ndarray_sha256
from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data.calvin import (
    CalvinPhysicalTransitionDataset,
    CalvinPICFEvidenceFrame,
)
from picf_next.data.calvin_pointcloud import CalvinCalibratedPointCloudBuilder
from picf_next.data.calvin_tactile import (
    CALVIN_TACTILE_FRAME_COUNT,
    CALVIN_TACTILE_HARDWARE_TYPE,
    CALVIN_TACTILE_POSE_RECONSTRUCTION,
    CALVIN_TACTILE_POSE_SOURCE_FILES_SHA256,
    CALVIN_TACTILE_SOURCE_COMMIT,
    CALVIN_TACTILE_SOURCE_FILES_SHA256,
    CALVIN_TACTILE_STREAM_NAMES,
    build_calvin_tactile_encoder_clips,
    calvin_digit_sensor_poses_world,
    calvin_tactile_source_frames,
)
from picf_next.data.calvin_tactile_calibration import LoadedCalvinTactileBackgrounds
from picf_next.data.causal_video import build_calvin_causal_video_clip
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheContract,
    DenseEvidenceCacheRecord,
)
from picf_next.encoders.anytouch2 import (
    ANYTOUCH2_GEOMETRY_WIDTH,
    ANYTOUCH2_TOKEN_WIDTH,
    ANYTOUCH2_TOKENS_PER_SENSOR,
    AnyTouch2DenseEncoder,
)
from picf_next.encoders.spatiallm_sonata import (
    SPATIALLM_SONATA_TOKEN_WIDTH,
    SpatialLMSonataDenseEncoder,
)
from picf_next.encoders.vjepa21 import (
    VJEPA21_CALVIN_GEOMETRY_WIDTH,
    VJEPA21_CALVIN_VIEW_NAMES,
    Vjepa21DenseEncoder,
    combine_vjepa21_calvin_views,
    vjepa21_calvin_encoder_contract,
)

_VJEPA_SENSOR_BY_VIEW = {
    "static": "observation.images.rgb_static",
    "gripper": "observation.images.rgb_gripper",
}
_SOURCE_NAME_BY_SENSOR = {
    "observation.images.rgb_static": "rgb_static",
    "observation.images.rgb_gripper": "rgb_gripper",
    "observation.depth.static": "depth_static",
    "observation.depth.gripper": "depth_gripper",
}


def _frame_values(frame: CalvinPICFEvidenceFrame) -> Mapping[str, np.ndarray]:
    if not isinstance(frame, CalvinPICFEvidenceFrame):
        raise TypeError("CALVIN frozen evidence requires one typed sensor frame")
    values = {item.key: item.value for item in frame.sensor_observations}
    if len(values) != len(frame.sensor_observations):
        raise ContractError("CALVIN frozen evidence frame repeats a sensor key")
    return values


def _rebind_contract(evidence: DenseEvidence, encoder_contract: str) -> DenseEvidence:
    if evidence.encoder_contract == encoder_contract:
        return evidence
    return DenseEvidence(
        modality=evidence.modality,
        encoder_contract=encoder_contract,
        tokens=evidence.tokens,
        available=evidence.available,
        timestamps=evidence.timestamps,
        confidence=evidence.confidence,
        geometry=evidence.geometry,
        group_ids=evidence.group_ids,
        current_measurement_valid=evidence.current_measurement_valid,
    )


def _cache_contract(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    coverage_plan_sha256: str,
    modality: str,
    encoder_contract: str,
    token_width: int,
    geometry_width: int,
    maximum_tokens: int,
    has_group_ids: bool,
    token_dtype: str,
) -> DenseEvidenceCacheContract:
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise ContractError("CALVIN frozen evidence requires a dataset file manifest")
    return DenseEvidenceCacheContract(
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_tree_sha256=manifest.tree_sha256,
        coverage_plan_sha256=coverage_plan_sha256,
        modality=modality,
        encoder_contract=encoder_contract,
        token_width=token_width,
        geometry_width=geometry_width,
        maximum_tokens=maximum_tokens,
        has_group_ids=has_group_ids,
        token_dtype=token_dtype,
    )


def _record_bounds(
    dataset: CalvinPhysicalTransitionDataset,
    *,
    start_record: int,
    maximum_records: int | None,
) -> range:
    if isinstance(start_record, bool) or not isinstance(start_record, int) or start_record < 0:
        raise ContractError("CALVIN frozen evidence start record must be nonnegative")
    if maximum_records is not None and (
        isinstance(maximum_records, bool)
        or not isinstance(maximum_records, int)
        or maximum_records <= 0
    ):
        raise ContractError("CALVIN frozen evidence maximum records must be positive")
    stop = len(dataset) if maximum_records is None else min(len(dataset), maximum_records)
    if start_record > stop:
        raise ContractError("CALVIN frozen evidence resume offset exceeds requested coverage")
    return range(start_record, stop)


@dataclass(slots=True)
class CalvinVjepa21EvidenceBuilder:
    dataset: CalvinPhysicalTransitionDataset
    encoder: Vjepa21DenseEncoder
    coverage_plan_sha256: str
    token_dtype: str = "float16"

    @property
    def encoder_contract(self) -> str:
        return vjepa21_calvin_encoder_contract(self.encoder.encoder_contract)

    @property
    def cache_contract(self) -> DenseEvidenceCacheContract:
        return _cache_contract(
            self.dataset,
            coverage_plan_sha256=self.coverage_plan_sha256,
            modality="vjepa",
            encoder_contract=self.encoder_contract,
            token_width=self.encoder.config.token_width,
            geometry_width=VJEPA21_CALVIN_GEOMETRY_WIDTH,
            maximum_tokens=len(VJEPA21_CALVIN_VIEW_NAMES) * self.encoder.config.token_count,
            has_group_ids=False,
            token_dtype=self.token_dtype,
        )

    def record(self, sample_key: str) -> DenseEvidenceCacheRecord:
        return self.records_for_sample_keys((sample_key,))[0]

    def records_for_sample_keys(
        self,
        sample_keys: tuple[str, ...],
    ) -> tuple[DenseEvidenceCacheRecord, ...]:
        """Batch only the frozen encoder call; preserve exact per-event records."""

        if not isinstance(sample_keys, tuple) or not sample_keys:
            raise ContractError("V-JEPA2.1 evidence batch must be a nonempty tuple")
        clips = []
        sources = []
        source_indices = []
        for sample_key in sample_keys:
            source_indices.append(self.dataset.source_global_index_by_key(sample_key))
            prefix = self.dataset.evidence_prefix_by_key(
                sample_key,
                maximum_source_frames=self.encoder.config.frame_count,
            )
            source_by_view: dict[str, list[str]] = {}
            for view_name in VJEPA21_CALVIN_VIEW_NAMES:
                clip = build_calvin_causal_video_clip(
                    prefix,
                    sensor_key=_VJEPA_SENSOR_BY_VIEW[view_name],
                    maximum_frames=self.encoder.config.frame_count,
                    # Source selection accepts the first physical frame. The
                    # adapter performs explicit 64-frame causal left padding.
                    tubelet_size=1,
                )
                if clip is None:
                    raise RuntimeError("a physical CALVIN event produced no V-JEPA2.1 source clip")
                clips.append(clip)
                source_by_view[view_name] = list(clip.source_frame_sha256)
            sources.append(source_by_view)
        encoded = self.encoder.encode_causal_clips(tuple(clips))
        views_per_sample = len(VJEPA21_CALVIN_VIEW_NAMES)
        records = []
        for index, (sample_key, source_index, source_by_view) in enumerate(
            zip(sample_keys, source_indices, sources, strict=True)
        ):
            start = index * views_per_sample
            evidence = combine_vjepa21_calvin_views(
                {
                    name: encoded[start + view_index]
                    for view_index, name in enumerate(VJEPA21_CALVIN_VIEW_NAMES)
                }
            )
            if evidence.encoder_contract != self.encoder_contract:
                raise RuntimeError("combined CALVIN V-JEPA2.1 contract changed")
            source_input_sha256 = canonical_payload_sha256(
                "picf-next.calvin-vjepa21-input/v1",
                {
                    "encoder_contract": self.encoder_contract,
                    "ordered_views": [
                        {"name": name, "source_frame_sha256": source_by_view[name]}
                        for name in VJEPA21_CALVIN_VIEW_NAMES
                    ],
                },
            )
            records.append(
                DenseEvidenceCacheRecord(
                    source_global_index=source_index,
                    sample_key=sample_key,
                    source_input_sha256=source_input_sha256,
                    evidence=evidence,
                )
            )
        return tuple(records)

    def records(
        self,
        start_record: int = 0,
        *,
        maximum_records: int | None = None,
    ) -> Iterator[DenseEvidenceCacheRecord]:
        for position in _record_bounds(
            self.dataset,
            start_record=start_record,
            maximum_records=maximum_records,
        ):
            yield self.record(self.dataset.sample_keys[position])


@dataclass(slots=True)
class CalvinSonataEvidenceBuilder:
    dataset: CalvinPhysicalTransitionDataset
    point_builder: CalvinCalibratedPointCloudBuilder
    encoder: SpatialLMSonataDenseEncoder
    coverage_plan_sha256: str
    token_dtype: str = "float16"

    def __post_init__(self) -> None:
        if self.point_builder.maximum_points != self.encoder.config.maximum_points:
            raise ContractError("CALVIN point builder and Sonata token budgets differ")

    @property
    def encoder_contract(self) -> str:
        digest = canonical_payload_sha256(
            "picf-next.calvin-sonata-producer/v1",
            {
                "point_builder": self.point_builder.encoder_input_contract,
                "sonata": self.encoder.encoder_contract,
            },
        )
        return f"{self.encoder.encoder_contract}/calvin-dual-rgbd@{digest}/v1"

    @property
    def cache_contract(self) -> DenseEvidenceCacheContract:
        return _cache_contract(
            self.dataset,
            coverage_plan_sha256=self.coverage_plan_sha256,
            modality="sonata",
            encoder_contract=self.encoder_contract,
            token_width=SPATIALLM_SONATA_TOKEN_WIDTH,
            geometry_width=self.encoder.config.geometry_width,
            maximum_tokens=self.encoder.config.maximum_points,
            has_group_ids=self.encoder.config.return_full_resolution,
            token_dtype=self.token_dtype,
        )

    def record(self, sample_key: str) -> DenseEvidenceCacheRecord:
        source_global_index = self.dataset.source_global_index_by_key(sample_key)
        frame = self.dataset.index.source_picf_evidence_frame(source_global_index)
        values = _frame_values(frame)
        state = self.dataset.index.source_robot_state(source_global_index)
        source_frame = {
            source_name: values[sensor_name]
            for sensor_name, source_name in _SOURCE_NAME_BY_SENSOR.items()
        }
        source_frame["robot_obs"] = state
        cloud = self.point_builder.build(source_frame)
        evidence = self.encoder.encode_points(
            xyz_world=cloud.xyz_world,
            colors=cloud.colors,
            view_ids=cloud.view_ids,
            timestamp_s=frame.timestamp_s,
        )
        evidence = _rebind_contract(evidence, self.encoder_contract)
        source_input_sha256 = canonical_payload_sha256(
            "picf-next.calvin-sonata-input/v1",
            {
                "arrays": {
                    name: ndarray_sha256(name, value)
                    for name, value in sorted(source_frame.items())
                },
                "encoder_contract": self.encoder_contract,
                "timestamp_s": frame.timestamp_s.hex(),
            },
        )
        return DenseEvidenceCacheRecord(
            source_global_index=source_global_index,
            sample_key=sample_key,
            source_input_sha256=source_input_sha256,
            evidence=evidence,
        )

    def records(
        self,
        start_record: int = 0,
        *,
        maximum_records: int | None = None,
    ) -> Iterator[DenseEvidenceCacheRecord]:
        for position in _record_bounds(
            self.dataset,
            start_record=start_record,
            maximum_records=maximum_records,
        ):
            yield self.record(self.dataset.sample_keys[position])


@dataclass(slots=True)
class CalvinAnyTouch2EvidenceBuilder:
    dataset: CalvinPhysicalTransitionDataset
    calibration: LoadedCalvinTactileBackgrounds
    encoder: AnyTouch2DenseEncoder
    coverage_plan_sha256: str
    token_dtype: str = "float16"

    def __post_init__(self) -> None:
        manifest = self.dataset.index.dataset_manifest
        if manifest is None or self.calibration.dataset_tree_sha256 != manifest.tree_sha256:
            raise ContractError("CALVIN tactile calibration belongs to another dataset tree")

    @property
    def encoder_contract(self) -> str:
        digest = canonical_payload_sha256(
            "picf-next.calvin-anytouch2-producer/v1",
            {
                "anytouch2": self.encoder.encoder_contract,
                "calibration_receipt_payload_sha256": (self.calibration.receipt_payload_sha256),
                "hardware_type": CALVIN_TACTILE_HARDWARE_TYPE,
                "official_source_commit": CALVIN_TACTILE_SOURCE_COMMIT,
                "official_pose_source_files_sha256": CALVIN_TACTILE_POSE_SOURCE_FILES_SHA256,
                "official_tactile_source_files_sha256": CALVIN_TACTILE_SOURCE_FILES_SHA256,
                "pose_reconstruction": CALVIN_TACTILE_POSE_RECONSTRUCTION,
                "streams": CALVIN_TACTILE_STREAM_NAMES,
            },
        )
        return f"{self.encoder.encoder_contract}/calvin-dual-digit@{digest}/v1"

    @property
    def cache_contract(self) -> DenseEvidenceCacheContract:
        return _cache_contract(
            self.dataset,
            coverage_plan_sha256=self.coverage_plan_sha256,
            modality="anytouch",
            encoder_contract=self.encoder_contract,
            token_width=ANYTOUCH2_TOKEN_WIDTH,
            geometry_width=ANYTOUCH2_GEOMETRY_WIDTH,
            maximum_tokens=len(CALVIN_TACTILE_STREAM_NAMES) * ANYTOUCH2_TOKENS_PER_SENSOR,
            has_group_ids=True,
            token_dtype=self.token_dtype,
        )

    def record(self, sample_key: str) -> DenseEvidenceCacheRecord:
        source_global_index = self.dataset.source_global_index_by_key(sample_key)
        prefix = self.dataset.evidence_prefix_by_key(
            sample_key,
            maximum_source_frames=CALVIN_TACTILE_FRAME_COUNT,
        )
        clips = build_calvin_tactile_encoder_clips(
            prefix,
            validity_thresholds_m=self.calibration.validity_thresholds_m,
        )
        first_global_index = source_global_index - len(prefix) + 1
        states = tuple(
            self.dataset.index.source_robot_state(global_index)
            for global_index in range(first_global_index, source_global_index + 1)
        )
        poses_by_frame = tuple(calvin_digit_sensor_poses_world(state) for state in states)
        clips_by_sensor = {clip.stream_name: clip.as_array() for clip in clips}
        sensor_types = {clip.stream_name: clip.hardware_type for clip in clips}
        backgrounds = {
            clip.stream_name: self.calibration.backgrounds_by_stream[clip.stream_name]
            for clip in clips
        }
        poses = {}
        timestamps = {}
        for clip in clips:
            selected = tuple(frame[clip.stream_name] for frame in poses_by_frame)[
                -CALVIN_TACTILE_FRAME_COUNT:
            ]
            padded = (selected[0],) * clip.padding_count + selected
            if len(padded) != CALVIN_TACTILE_FRAME_COUNT:
                raise RuntimeError("CALVIN tactile pose padding differs from its image clip")
            poses[clip.stream_name] = padded
            timestamps[clip.stream_name] = clip.frame_timestamps_s
        evidence = self.encoder.encode_active_sensors(
            clips_by_sensor=clips_by_sensor,
            sensor_types_by_stream=sensor_types,
            backgrounds_by_sensor=backgrounds,
            poses_world_by_sensor=poses,
            timestamps_by_sensor=timestamps,
        )
        evidence = _rebind_contract(evidence, self.encoder_contract)
        tactile_sources = [
            tactile for frame in prefix for tactile in calvin_tactile_source_frames(frame)
        ]
        source_input_sha256 = canonical_payload_sha256(
            "picf-next.calvin-anytouch2-input/v1",
            {
                "calibration_archive_sha256": self.calibration.archive_sha256,
                "calibration_receipt_sha256": self.calibration.receipt_sha256,
                "encoder_contract": self.encoder_contract,
                "robot_states": [
                    ndarray_sha256(f"robot_obs[{index}]", state)
                    for index, state in enumerate(states)
                ],
                "tactile_sources": [
                    {"stream": item.stream_name, "sha256": item.source_sha256}
                    for item in tactile_sources
                ],
            },
        )
        return DenseEvidenceCacheRecord(
            source_global_index=source_global_index,
            sample_key=sample_key,
            source_input_sha256=source_input_sha256,
            evidence=evidence,
        )

    def records(
        self,
        start_record: int = 0,
        *,
        maximum_records: int | None = None,
    ) -> Iterator[DenseEvidenceCacheRecord]:
        for position in _record_bounds(
            self.dataset,
            start_record=start_record,
            maximum_records=maximum_records,
        ):
            yield self.record(self.dataset.sample_keys[position])

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data import calvin_frozen_evidence as frozen_module
from picf_next.data.calvin import CalvinPICFEvidenceFrame, CalvinPICFSensorObservation
from picf_next.data.calvin_dense_evidence_source_audit import (
    audit_calvin_dense_evidence_source_inputs,
    calvin_anytouch2_source_input_sha256,
    calvin_sonata_source_input_sha256,
    calvin_vjepa21_source_input_sha256,
    validate_calvin_dense_evidence_source_audit,
)
from picf_next.data.calvin_frozen_evidence import (
    CalvinAnyTouch2EvidenceBuilder,
    CalvinSonataEvidenceBuilder,
    CalvinVjepa21EvidenceBuilder,
)
from picf_next.data.calvin_pointcloud import CalibratedPointCloud
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheRecord,
    FrozenDenseEvidenceCacheBank,
    publish_dense_evidence_cache,
)
from picf_next.encoders.vjepa21 import vjepa21_calvin_encoder_contract


def _readonly(value: np.ndarray) -> np.ndarray:
    value.setflags(write=False)
    return value


def _frame(timestamp_s: float, *, value: int = 0) -> CalvinPICFEvidenceFrame:
    tactile_rgb = np.full((160, 120, 6), value, dtype=np.uint8)
    tactile_depth = np.full((160, 120, 2), 0.01 + value * 0.001, dtype=np.float32)
    values = {
        "observation.images.rgb_static": np.full((8, 8, 3), value, dtype=np.uint8),
        "observation.images.rgb_gripper": np.full((6, 6, 3), value + 1, dtype=np.uint8),
        "observation.depth.static": np.full((8, 8), 1.0 + value, dtype=np.float32),
        "observation.depth.gripper": np.full((6, 6), 2.0 + value, dtype=np.float32),
        "observation.tactile.rgb": tactile_rgb,
        "observation.tactile.depth": tactile_depth,
    }
    observations = tuple(
        CalvinPICFSensorObservation(
            key=name,
            value=_readonly(array),
            timestamp_s=timestamp_s,
            units="fixture",
        )
        for name, array in values.items()
    )
    return CalvinPICFEvidenceFrame(
        sensor_observations=observations,
        timestamp_s=timestamp_s,
        delta_t_s=1.0,
    )


class _Index:
    dataset_manifest = SimpleNamespace(
        dataset_id="calvin-fixture",
        dataset_revision="revision",
        tree_sha256="d" * 64,
    )

    def __init__(self, frames: tuple[CalvinPICFEvidenceFrame, ...]) -> None:
        self.frames = frames
        self.states = []
        for index in range(len(frames)):
            state = np.zeros(15, dtype=np.float32)
            state[0] = float(index)
            self.states.append(_readonly(state))

    def source_picf_evidence_frame(self, global_index: int) -> CalvinPICFEvidenceFrame:
        return self.frames[global_index - 10]

    def source_robot_state(self, global_index: int) -> np.ndarray:
        return self.states[global_index - 10]


class _PhysicalDataset:
    sample_keys = ("physical-10", "physical-11")

    def __init__(self, frames: tuple[CalvinPICFEvidenceFrame, ...]) -> None:
        self.index = _Index(frames)

    def __len__(self) -> int:
        return len(self.sample_keys)

    def source_global_index_by_key(self, sample_key: str) -> int:
        return 10 + self.sample_keys.index(sample_key)

    def evidence_prefix_by_key(
        self, sample_key: str, *, maximum_source_frames: int
    ) -> tuple[CalvinPICFEvidenceFrame, ...]:
        position = self.sample_keys.index(sample_key)
        return self.index.frames[max(0, position - maximum_source_frames + 1) : position + 1]


class _VjepaEncoder:
    encoder_contract = "vjepa-fixture/v1"
    config = SimpleNamespace(frame_count=4, token_width=3, token_count=2)

    def __init__(self) -> None:
        self.clips = []

    def encode_causal_clip(self, clip):
        self.clips.append(clip)
        return DenseEvidence(
            modality="vjepa",
            encoder_contract=self.encoder_contract,
            tokens=np.ones((2, 3), dtype=np.float32),
            available=True,
            timestamps=np.full(2, clip.current_timestamp_s, dtype=np.float32),
            confidence=np.ones(2, dtype=np.float32),
            geometry=np.zeros((2, 2), dtype=np.float32),
            current_measurement_valid=np.ones(2, dtype=np.bool_),
        )

    def encode_causal_clips(self, clips):
        return tuple(self.encode_causal_clip(clip) for clip in clips)


def _combine_fixture(evidence_by_view):
    ordered = [evidence_by_view["static"], evidence_by_view["gripper"]]
    return DenseEvidence(
        modality="vjepa",
        encoder_contract=vjepa21_calvin_encoder_contract(ordered[0].encoder_contract),
        tokens=np.concatenate([item.tokens for item in ordered]),
        available=True,
        timestamps=np.concatenate([item.timestamps for item in ordered]),
        confidence=np.ones(4, dtype=np.float32),
        geometry=np.zeros((4, 4), dtype=np.float32),
        current_measurement_valid=np.ones(4, dtype=np.bool_),
    )


def test_vjepa_builder_uses_both_physical_views_and_accepts_first_frame(monkeypatch) -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    encoder = _VjepaEncoder()
    monkeypatch.setattr(frozen_module, "combine_vjepa21_calvin_views", _combine_fixture)
    builder = CalvinVjepa21EvidenceBuilder(
        dataset,
        encoder,
        "c" * 64,  # type: ignore[arg-type]
    )

    first = builder.record("physical-10")
    second = builder.record("physical-11")

    assert [clip.sensor_key for clip in encoder.clips] == [
        "observation.images.rgb_static",
        "observation.images.rgb_gripper",
        "observation.images.rgb_static",
        "observation.images.rgb_gripper",
    ]
    assert [len(clip.images) for clip in encoder.clips] == [1, 1, 2, 2]
    assert first.evidence.token_count == 4
    assert first.source_input_sha256 != second.source_input_sha256
    assert builder.cache_contract.maximum_tokens == 4
    assert first.source_input_sha256 == calvin_vjepa21_source_input_sha256(
        dataset,
        "physical-10",
        encoder_contract=builder.encoder_contract,
        frame_count=encoder.config.frame_count,
    )


def test_vjepa_builder_batches_without_changing_record_order(monkeypatch) -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    encoder = _VjepaEncoder()
    monkeypatch.setattr(frozen_module, "combine_vjepa21_calvin_views", _combine_fixture)
    builder = CalvinVjepa21EvidenceBuilder(dataset, encoder, "c" * 64)  # type: ignore[arg-type]

    records = builder.records_for_sample_keys(("physical-10", "physical-11"))

    assert tuple(record.sample_key for record in records) == ("physical-10", "physical-11")
    assert tuple(record.source_global_index for record in records) == (10, 11)
    assert [clip.sensor_key for clip in encoder.clips] == [
        "observation.images.rgb_static",
        "observation.images.rgb_gripper",
        "observation.images.rgb_static",
        "observation.images.rgb_gripper",
    ]


class _PointBuilder:
    maximum_points = 4
    encoder_input_contract = "calvin-point-fixture/v1"

    def __init__(self) -> None:
        self.frames = []

    def build(self, frame):
        self.frames.append(frame)
        return CalibratedPointCloud(
            xyz_world=_readonly(np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32)),
            colors=_readonly(np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)),
            view_ids=_readonly(np.asarray([0], dtype=np.int64)),
        )


class _SonataEncoder:
    encoder_contract = "sonata-fixture/v1"
    config = SimpleNamespace(
        geometry_width=4,
        maximum_points=4,
        return_full_resolution=True,
    )

    def encode_points(self, *, xyz_world, colors, view_ids, timestamp_s):
        del colors
        count = xyz_world.shape[0]
        return DenseEvidence(
            modality="sonata",
            encoder_contract=self.encoder_contract,
            tokens=np.ones((count, 512), dtype=np.float32),
            available=True,
            timestamps=np.full(count, timestamp_s, dtype=np.float32),
            confidence=np.ones(count, dtype=np.float32),
            geometry=np.concatenate((xyz_world, view_ids.astype(np.float32)[:, None]), axis=1),
            group_ids=view_ids,
            current_measurement_valid=np.ones(count, dtype=np.bool_),
        )


def test_sonata_builder_binds_rgbd_state_and_calibration_without_task_fields() -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    point_builder = _PointBuilder()
    builder = CalvinSonataEvidenceBuilder(
        dataset,
        point_builder,
        _SonataEncoder(),  # type: ignore[arg-type]
        "c" * 64,
    )

    first = builder.record("physical-10")
    second = builder.record("physical-11")

    assert set(point_builder.frames[0]) == {
        "depth_gripper",
        "depth_static",
        "rgb_gripper",
        "rgb_static",
        "robot_obs",
    }
    assert first.source_input_sha256 != second.source_input_sha256
    assert first.evidence.encoder_contract == builder.encoder_contract
    assert builder.cache_contract.maximum_tokens == 4
    assert first.source_input_sha256 == calvin_sonata_source_input_sha256(
        dataset,
        "physical-10",
        encoder_contract=builder.encoder_contract,
    )


class _AnyTouchEncoder:
    encoder_contract = "anytouch-fixture/v1"

    def __init__(self) -> None:
        self.calls = []

    def encode_active_sensors(self, **kwargs):
        self.calls.append(kwargs)
        names = tuple(sorted(kwargs["clips_by_sensor"]))
        count = len(names)
        return DenseEvidence(
            modality="anytouch",
            encoder_contract=self.encoder_contract,
            tokens=np.ones((count, 768), dtype=np.float32),
            available=bool(count),
            timestamps=np.full(count, 0.0, dtype=np.float32),
            confidence=np.ones(count, dtype=np.float32),
            geometry=np.zeros((count, 18), dtype=np.float32),
            group_ids=np.arange(count, dtype=np.int64),
            current_measurement_valid=np.ones(count, dtype=np.bool_),
        )


def _calibration():
    backgrounds = {
        name: _readonly(np.zeros((160, 120, 3), dtype=np.float32))
        for name in ("left_digit", "right_digit")
    }
    return SimpleNamespace(
        backgrounds_by_stream=backgrounds,
        validity_thresholds_m={"left_digit": 0.001, "right_digit": 0.001},
        archive_sha256="a" * 64,
        receipt_sha256="b" * 64,
        receipt_payload_sha256="c" * 64,
        dataset_tree_sha256="d" * 64,
    )


def test_anytouch_builder_left_pads_images_poses_and_binds_calibration() -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    encoder = _AnyTouchEncoder()
    builder = CalvinAnyTouch2EvidenceBuilder(
        dataset,
        _calibration(),
        encoder,  # type: ignore[arg-type]
        "c" * 64,
    )

    record = builder.record("physical-10")

    call = encoder.calls[0]
    assert set(call["clips_by_sensor"]) == {"left_digit", "right_digit"}
    assert all(value.shape == (4, 160, 120, 3) for value in call["clips_by_sensor"].values())
    assert all(len(value) == 4 for value in call["poses_world_by_sensor"].values())
    assert record.evidence.encoder_contract == builder.encoder_contract
    assert builder.cache_contract.maximum_tokens == 2 * 398
    assert record.source_input_sha256 == calvin_anytouch2_source_input_sha256(
        dataset,
        "physical-10",
        encoder_contract=builder.encoder_contract,
        calibration=builder.calibration,
    )


def test_record_iteration_resumes_at_authenticated_record_offset(monkeypatch) -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    monkeypatch.setattr(frozen_module, "combine_vjepa21_calvin_views", _combine_fixture)
    builder = CalvinVjepa21EvidenceBuilder(
        dataset,
        _VjepaEncoder(),
        "c" * 64,  # type: ignore[arg-type]
    )

    records = tuple(builder.records(1, maximum_records=2))

    assert [record.sample_key for record in records] == ["physical-11"]
    assert [record.source_global_index for record in records] == [11]


def test_source_audit_recomputes_all_three_cache_identities(tmp_path, monkeypatch) -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    monkeypatch.setattr(frozen_module, "combine_vjepa21_calvin_views", _combine_fixture)
    vjepa_builder = CalvinVjepa21EvidenceBuilder(
        dataset,
        _VjepaEncoder(),
        "c" * 64,  # type: ignore[arg-type]
    )
    sonata_builder = CalvinSonataEvidenceBuilder(
        dataset,
        _PointBuilder(),
        _SonataEncoder(),  # type: ignore[arg-type]
        "c" * 64,
    )
    anytouch_builder = CalvinAnyTouch2EvidenceBuilder(
        dataset,
        _calibration(),
        _AnyTouchEncoder(),  # type: ignore[arg-type]
        "c" * 64,
    )
    builders = (anytouch_builder, sonata_builder, vjepa_builder)
    roots = []
    manifest_sha256s = []
    for builder in builders:
        root = tmp_path / builder.cache_contract.modality
        manifest_sha256s.append(
            publish_dense_evidence_cache(
                root,
                contract=builder.cache_contract,
                records=tuple(builder.records()),
            )
        )
        roots.append(root)
    bank = FrozenDenseEvidenceCacheBank.load(
        roots,
        manifest_sha256s=manifest_sha256s,
        dataset_tree_sha256="d" * 64,
    )

    report = audit_calvin_dense_evidence_source_inputs(
        dataset,  # type: ignore[arg-type]
        bank,
        cache_manifest_sha256_by_modality=dict(zip(bank.modalities, manifest_sha256s, strict=True)),
        calibration=anytouch_builder.calibration,
        vjepa_frame_count=4,
    )
    parallel_report = audit_calvin_dense_evidence_source_inputs(
        dataset,  # type: ignore[arg-type]
        bank,
        cache_manifest_sha256_by_modality=dict(zip(bank.modalities, manifest_sha256s, strict=True)),
        calibration=anytouch_builder.calibration,
        vjepa_frame_count=4,
        workers=2,
    )

    assert report["status"] == "PASS"
    assert parallel_report == report
    with pytest.raises(ContractError, match="worker count"):
        audit_calvin_dense_evidence_source_inputs(
            dataset,  # type: ignore[arg-type]
            bank,
            cache_manifest_sha256_by_modality=dict(
                zip(bank.modalities, manifest_sha256s, strict=True)
            ),
            calibration=anytouch_builder.calibration,
            vjepa_frame_count=4,
            workers=0,
        )
    assert report["record_count"] == 2
    assert report["record_start"] == 0
    assert report["record_stop"] == 2
    validated = validate_calvin_dense_evidence_source_audit(
        report,
        dataset_id="calvin-fixture",
        dataset_revision="revision",
        dataset_tree_sha256="d" * 64,
        coverage_plan_sha256="c" * 64,
        cache_manifest_sha256_by_modality=dict(zip(bank.modalities, manifest_sha256s, strict=True)),
        record_count=2,
    )
    assert validated == report

    tampered = {**report, "records_sha256": "0" * 64}
    with pytest.raises(ContractError, match="artifact hash changed"):
        validate_calvin_dense_evidence_source_audit(
            tampered,
            dataset_id="calvin-fixture",
            dataset_revision="revision",
            dataset_tree_sha256="d" * 64,
            coverage_plan_sha256="c" * 64,
            cache_manifest_sha256_by_modality=dict(
                zip(bank.modalities, manifest_sha256s, strict=True)
            ),
            record_count=2,
        )


def test_source_audit_rejects_one_cache_hash_that_does_not_match_raw_input(
    tmp_path, monkeypatch
) -> None:
    dataset = _PhysicalDataset((_frame(0.0), _frame(1.0, value=2)))
    monkeypatch.setattr(frozen_module, "combine_vjepa21_calvin_views", _combine_fixture)
    vjepa_builder = CalvinVjepa21EvidenceBuilder(
        dataset,
        _VjepaEncoder(),
        "c" * 64,  # type: ignore[arg-type]
    )
    sonata_builder = CalvinSonataEvidenceBuilder(
        dataset,
        _PointBuilder(),
        _SonataEncoder(),  # type: ignore[arg-type]
        "c" * 64,
    )
    anytouch_builder = CalvinAnyTouch2EvidenceBuilder(
        dataset,
        _calibration(),
        _AnyTouchEncoder(),  # type: ignore[arg-type]
        "c" * 64,
    )
    roots = []
    manifest_sha256s = []
    for builder in (anytouch_builder, sonata_builder, vjepa_builder):
        records = list(builder.records())
        if builder.cache_contract.modality == "sonata":
            first = records[0]
            records[0] = DenseEvidenceCacheRecord(
                source_global_index=first.source_global_index,
                sample_key=first.sample_key,
                source_input_sha256="0" * 64,
                evidence=first.evidence,
            )
        root = tmp_path / builder.cache_contract.modality
        manifest_sha256s.append(
            publish_dense_evidence_cache(root, contract=builder.cache_contract, records=records)
        )
        roots.append(root)
    bank = FrozenDenseEvidenceCacheBank.load(
        roots,
        manifest_sha256s=manifest_sha256s,
        dataset_tree_sha256="d" * 64,
    )

    with pytest.raises(ContractError, match="source identity differs"):
        audit_calvin_dense_evidence_source_inputs(
            dataset,  # type: ignore[arg-type]
            bank,
            cache_manifest_sha256_by_modality=dict(
                zip(bank.modalities, manifest_sha256s, strict=True)
            ),
            calibration=anytouch_builder.calibration,
            vjepa_frame_count=4,
        )

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from picf_next.contracts import ContractError, DenseEvidence
from picf_next.data.calvin import (
    CALVIN_DEBUG_REVISION,
    CALVIN_HOST_IMAGE_KEYS,
    CALVIN_OBSERVATION_SPECS,
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
    CalvinMolmoAct2Dataset,
    CalvinPhysicalTransitionDataset,
    CalvinPosteriorWindowDataset,
    CalvinRawActionControlSpan,
    CalvinStatefulTransitionDataset,
    collate_calvin_molmoact2,
    decode_calvin_frame,
    validate_calvin_source_frame,
)
from picf_next.data.calvin_multimodal import (
    calvin_encoder_inputs,
    validate_calvin_encoded_evidence,
    validate_calvin_evidence_timestamps,
)
from picf_next.data.dataset_manifest import (
    build_dataset_file_manifest,
    load_dataset_file_manifest,
)
from tools.audit_calvin_causal_video import audit_stateful_dataset, render_contact_sheet


def _frame() -> dict[str, np.ndarray]:
    robot_obs = np.zeros(15, dtype=np.float64)
    absolute = np.zeros(7, dtype=np.float64)
    absolute[:3] = (0.01, -0.02, 0.03)
    absolute[3:6] = (0.025, -0.05, 0.075)
    absolute[-1] = -1.0
    relative = np.array((0.5, -1.0, 1.0, 0.5, -1.0, 1.0, -1.0), dtype=np.float64)
    frame = {
        "robot_obs": robot_obs,
        "actions": absolute,
        "rel_actions": relative,
        "scene_obs": np.arange(24, dtype=np.float64),
    }
    for source_key, _, shape, dtype, _ in CALVIN_OBSERVATION_SPECS:
        frame[source_key] = np.zeros(shape, dtype=dtype)
    return frame


def _decode(frame: dict[str, np.ndarray] | None = None):
    return decode_calvin_frame(
        _frame() if frame is None else frame,
        source_path=Path("episode_0000010.npz"),
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        episode=CalvinEpisode(0, 10, 20),
        segment=CalvinLanguageSegment(0, 10, 15, "move_block", "move the block", 0),
        global_index=10,
    )


def test_calvin_decoder_is_complete_immutable_and_has_no_privileged_state() -> None:
    record = _decode()

    assert record.state.shape == (15,)
    assert record.action.shape == (7,)
    assert record.action.tolist() == pytest.approx([0.5, -1.0, 1.0, 0.5, -1.0, 1.0, -1.0])
    assert record.delta_t_s == pytest.approx(1.0 / 30.0)
    assert record.transition_valid
    assert len(record.array_observations) == 6
    assert all(not item.value.flags.writeable for item in record.array_observations)
    assert "scene_obs" not in {item.key for item in record.array_observations}
    assert "actions" not in {item.key for item in record.array_observations}

    encoder_inputs = calvin_encoder_inputs(record)
    assert encoder_inputs.static_rgb.shape == (200, 200, 3)
    assert encoder_inputs.wrist_rgb.shape == (84, 84, 3)
    assert tuple(item.shape for item in encoder_inputs.tactile_rgb) == (
        (160, 120, 3),
        (160, 120, 3),
    )


def test_calvin_decoder_rejects_action_shape_sensor_and_conversion_drift() -> None:
    frame = _frame()
    frame["rel_actions"] = frame["rel_actions"].copy()
    frame["rel_actions"][0] = -0.5
    with pytest.raises(ContractError, match="official conversion"):
        _decode(frame)

    frame = _frame()
    frame["rgb_static"] = np.zeros((199, 200, 3), dtype=np.uint8)
    with pytest.raises(ContractError, match="rgb_static shape"):
        _decode(frame)

    frame = _frame()
    frame["depth_static"] = frame["depth_static"].astype(np.float64)
    with pytest.raises(ContractError, match="depth_static dtype"):
        validate_calvin_source_frame(frame)

    with pytest.raises(ContractError, match="verify_relative_action"):
        validate_calvin_source_frame(_frame(), verify_relative_action=1)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="control rate"):
        decode_calvin_frame(
            _frame(),
            source_path=Path("episode_0000010.npz"),
            dataset_id="calvin-test",
            dataset_revision="sha256:test",
            episode=CalvinEpisode(0, 10, 20),
            segment=CalvinLanguageSegment(0, 10, 15, "move_block", "move the block", 0),
            global_index=10,
            control_hz=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ContractError, match="global index"):
        decode_calvin_frame(
            _frame(),
            source_path=Path("episode_0000010.npz"),
            dataset_id="calvin-test",
            dataset_revision="sha256:test",
            episode=CalvinEpisode(0, 10, 20),
            segment=CalvinLanguageSegment(0, 10, 15, "move_block", "move the block", 0),
            global_index=10.5,  # type: ignore[arg-type]
        )


def test_calvin_indices_and_touch_activity_are_strictly_typed() -> None:
    with pytest.raises(ContractError, match="episode bounds must be integers"):
        CalvinEpisode(True, 0, 1)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="language indices must be integers"):
        CalvinLanguageSegment(0, 0, 1, "task", "instruction", 0.0)  # type: ignore[arg-type]
    with pytest.raises(ContractError, match="language metadata"):
        CalvinLanguageSegment(0, 0, 1, 7, "instruction", 0)  # type: ignore[arg-type]

    record = _decode()
    evidence = (
        _encoded_evidence("vjepa", token_count=1, timestamp_s=record.timestamp_s),
        _encoded_evidence("sonata", token_count=1, timestamp_s=record.timestamp_s),
        _encoded_evidence(
            "anytouch",
            token_count=1,
            timestamp_s=record.timestamp_s,
            group_ids=np.zeros(1, dtype=np.int64),
        ),
    )
    with pytest.raises(ContractError, match="active_touch_streams"):
        validate_calvin_encoded_evidence(  # type: ignore[arg-type]
            record, evidence, active_touch_streams=True
        )


def _encoded_evidence(
    modality: str,
    *,
    token_count: int,
    timestamp_s: float,
    available: bool = True,
    group_ids: np.ndarray | None = None,
) -> DenseEvidence:
    return DenseEvidence(
        modality=modality,
        encoder_contract=f"{modality}.test.v1",
        tokens=np.zeros((token_count, 4), dtype=np.float32),
        available=available,
        timestamps=np.full(token_count, timestamp_s, dtype=np.float32),
        confidence=np.ones(token_count, dtype=np.float32),
        group_ids=group_ids,
    )


def test_calvin_multimodal_boundary_enforces_complete_causal_touch_semantics() -> None:
    record = _decode()
    active = (
        _encoded_evidence("vjepa", token_count=8, timestamp_s=record.timestamp_s),
        _encoded_evidence("sonata", token_count=5, timestamp_s=record.timestamp_s),
        _encoded_evidence(
            "anytouch",
            token_count=6,
            timestamp_s=record.timestamp_s,
            group_ids=np.zeros(6, dtype=np.int64),
        ),
    )
    validate_calvin_encoded_evidence(record, active, active_touch_streams=("left_digit",))

    inactive = (
        active[0],
        active[1],
        _encoded_evidence(
            "anytouch",
            token_count=0,
            timestamp_s=record.timestamp_s,
            available=False,
        ),
    )
    validate_calvin_encoded_evidence(record, inactive, active_touch_streams=())

    split_touch = _encoded_evidence(
        "anytouch",
        token_count=6,
        timestamp_s=record.timestamp_s,
        group_ids=np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64),
    )
    validate_calvin_encoded_evidence(
        record,
        (active[0], active[1], split_touch),
        active_touch_streams=("left_digit", "right_digit"),
    )
    with pytest.raises(ContractError, match="one-to-one"):
        validate_calvin_encoded_evidence(
            record,
            (active[0], active[1], split_touch),
            active_touch_streams=("left_digit",),
        )
    with pytest.raises(ContractError, match="duplicate modalities"):
        validate_calvin_encoded_evidence(
            record,
            (active[0], active[0], active[2]),
            active_touch_streams=("left_digit",),
        )

    future = _encoded_evidence(
        "vjepa",
        token_count=8,
        timestamp_s=record.timestamp_s + 1.0,
    )
    with pytest.raises(ContractError, match="future timestamp"):
        validate_calvin_encoded_evidence(
            record,
            (future, active[1], active[2]),
            active_touch_streams=("left_digit",),
        )


def test_calvin_cached_evidence_boundary_rejects_future_timestamps() -> None:
    present = _encoded_evidence("vjepa", token_count=2, timestamp_s=0.5)
    validate_calvin_evidence_timestamps((present,), observation_timestamp_s=0.5)

    non_exact_observation = 4.3
    quantized_present = _encoded_evidence(
        "sonata",
        token_count=2,
        timestamp_s=float(np.float32(non_exact_observation)),
    )
    validate_calvin_evidence_timestamps(
        (quantized_present,),
        observation_timestamp_s=non_exact_observation,
    )

    next_float32 = float(np.nextafter(np.float32(non_exact_observation), np.float32(np.inf)))
    quantized_future = _encoded_evidence(
        "sonata",
        token_count=2,
        timestamp_s=next_float32,
    )
    with pytest.raises(ContractError, match="future timestamp"):
        validate_calvin_evidence_timestamps(
            (quantized_future,),
            observation_timestamp_s=non_exact_observation,
        )

    future = _encoded_evidence("vjepa", token_count=2, timestamp_s=0.5001)
    with pytest.raises(ContractError, match="future timestamp"):
        validate_calvin_evidence_timestamps((future,), observation_timestamp_s=0.5)


def _write_split(root: Path) -> None:
    (root / ".hydra").mkdir(parents=True)
    (root / "lang_annotations").mkdir()
    (root / ".hydra" / "merged_config.yaml").write_text("env:\n  control_freq: 30\n")
    np.save(root / "ep_start_end_ids.npy", np.array([[10, 17]], dtype=np.int64))
    np.save(root / "ep_lens.npy", np.array(8, dtype=np.int64))
    annotations = {
        "language": {
            "ann": ["move the block", "turn on the light"],
            "task": ["move_block", "turn_on_light"],
        },
        "info": {"indx": [(10, 14), (13, 17)]},
    }
    np.save(root / "lang_annotations" / "auto_lang_ann.npy", annotations)
    for step in range(10, 18):
        frame = _frame()
        frame["actions"][0] = (step - 10) * 0.001
        frame["rel_actions"][0] = (step - 10) * 0.05
        np.savez(root / f"episode_{step:07d}.npz", **frame)


def _split_manifest(split: Path):
    relative_paths = (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
        *(f"episode_{step:07d}.npz" for step in range(10, 18)),
    )
    return build_dataset_file_manifest(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        split_name="training",
        relative_paths=relative_paths,
    )


def _rewrite_language_intervals(split: Path, intervals: list[tuple[int, int]]) -> None:
    annotation_path = split / "lang_annotations" / "auto_lang_ann.npy"
    annotations = np.load(annotation_path, allow_pickle=True).item()
    annotations["info"]["indx"] = intervals
    np.save(annotation_path, annotations)


def test_manifest_bound_index_rejects_frame_mutation_after_load(tmp_path: Path) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    with (split / "episode_0000010.npz").open("ab") as stream:
        stream.write(b"post-validation mutation")

    with pytest.raises(ContractError, match="differs from frozen manifest|byte limit"):
        index.record(0, 10)


def test_validated_source_frame_arrays_are_manifest_bound_and_immutable(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )

    arrays = index.validated_source_frame_arrays(
        10,
        fields=("scene_obs", "rgb_static"),
    )
    assert tuple(arrays) == ("scene_obs", "rgb_static")
    assert not arrays["scene_obs"].flags.writeable
    assert not arrays["rgb_static"].flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        arrays["scene_obs"][0] = 7.0
    with pytest.raises(TypeError):
        arrays["extra"] = np.zeros(1)  # type: ignore[index]
    with pytest.raises(ContractError, match="requested fields"):
        index.validated_source_frame_arrays(10, fields=("not_present",))

    with (split / "episode_0000010.npz").open("ab") as stream:
        stream.write(b"post-validation mutation")
    with pytest.raises(ContractError, match="differs from frozen manifest|byte limit"):
        index.validated_source_frame_arrays(10, fields=("scene_obs",))


def test_calvin_index_rejects_unmanifested_object_array_metadata(tmp_path: Path) -> None:
    split = tmp_path / "training"
    _write_split(split)

    with pytest.raises(ContractError, match="content-addressed dataset manifest"):
        CalvinDatasetIndex.load(split)


@pytest.mark.parametrize(
    ("filename", "value", "message"),
    [
        ("ep_start_end_ids.npy", np.array([[10.0, 17.0]]), "episode bounds"),
        ("ep_lens.npy", np.array(8.0), "episode lengths"),
    ],
)
def test_calvin_index_rejects_numeric_metadata_that_would_be_truncated(
    tmp_path: Path,
    filename: str,
    value: np.ndarray,
    message: str,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    np.save(split / filename, value)
    with pytest.raises(ContractError, match=message):
        CalvinDatasetIndex.load(
            split,
            dataset_id="calvin-test",
            dataset_revision="sha256:test",
            verify_files=False,
            dataset_manifest=_split_manifest(split),
        )


def test_calvin_index_rejects_noninteger_language_interval_and_nonstring_text(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    annotation_path = split / "lang_annotations" / "auto_lang_ann.npy"
    annotations = np.load(annotation_path, allow_pickle=True).item()
    annotations["info"]["indx"][0] = (10.5, 14.0)
    np.save(annotation_path, annotations)
    with pytest.raises(ContractError, match="language intervals"):
        CalvinDatasetIndex.load(
            split,
            dataset_id="calvin-test",
            dataset_revision="sha256:test",
            verify_files=False,
            dataset_manifest=_split_manifest(split),
        )

    annotations["info"]["indx"][0] = (10, 14)
    annotations["language"]["task"][0] = 5
    np.save(annotation_path, annotations)
    with pytest.raises(ContractError, match="task keys and instructions"):
        CalvinDatasetIndex.load(
            split,
            dataset_id="calvin-test",
            dataset_revision="sha256:test",
            verify_files=False,
            dataset_manifest=_split_manifest(split),
        )


def test_calvin_index_windows_and_action_chunks_do_not_cross_language_segments(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )

    windows = list(index.iter_windows(2, segment_indices=[0]))
    assert [tuple(record.global_index for record in window.records) for window in windows] == [
        (10, 11),
        (11, 12),
        (12, 13),
    ]
    assert all(record.task == "move the block" for window in windows for record in window.records)
    first_actions = windows[0].previous_executed_actions
    assert len(first_actions) == len(windows[0].records)
    np.testing.assert_array_equal(first_actions[0], np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(first_actions[1], windows[0].records[0].action)
    assert not first_actions[0].flags.writeable
    assert not first_actions[1].flags.writeable
    evidence_frame = windows[0].picf_evidence_frames[0]
    assert not hasattr(evidence_frame, "action")
    assert not hasattr(evidence_frame, "task")
    assert not hasattr(evidence_frame, "state")
    assert not hasattr(evidence_frame, "global_index")
    assert not hasattr(evidence_frame.sensor_observations[0], "source_path")
    assert tuple(item.key for item in evidence_frame.sensor_observations) == tuple(
        item.key for item in windows[0].records[0].array_observations
    )

    sample = index.molmoact2_sample(0, 13, action_horizon=4)
    assert sample.action.shape == (4, 7)
    assert sample.action_is_pad.tolist() == [False, True, True, True]
    assert sample.observation["task"] == "move the block"
    assert "observation.depth.static" not in sample.observation
    assert "scene_obs" not in sample.observation
    source_observation = index.molmoact2_source_observation(17)
    assert set(source_observation.images) == set(CALVIN_HOST_IMAGE_KEYS)
    assert source_observation.state.shape == (15,)
    assert source_observation.state_valid.all()
    assert source_observation.timestamp_s == pytest.approx(7 / 30)
    assert source_observation.delta_t_s == pytest.approx(1 / 30)
    assert not hasattr(source_observation, "task")
    assert not hasattr(source_observation, "action")
    assert not hasattr(source_observation, "global_index")
    assert not source_observation.images[CALVIN_HOST_IMAGE_KEYS[0]].flags.writeable
    assert not source_observation.images[CALVIN_HOST_IMAGE_KEYS[1]].flags.writeable
    assert not source_observation.state.flags.writeable

    host_dataset = CalvinMolmoAct2Dataset(index, action_horizon=4)
    posterior_dataset = CalvinPosteriorWindowDataset(index, sequence_length=2)
    assert host_dataset.index is index
    assert posterior_dataset.index is index
    assert len(host_dataset) == 8
    assert len(posterior_dataset) == 6
    assert host_dataset[0].source_global_index == 10
    assert [record.global_index for record in posterior_dataset[-1].records] == [15, 16]
    np.testing.assert_array_equal(
        posterior_dataset[-1].previous_executed_actions[1],
        posterior_dataset[-1].records[0].action,
    )

    batch = collate_calvin_molmoact2([host_dataset[0], host_dataset[1]])
    assert batch["observation"]["observation.images.image"].shape == (2, 200, 200, 3)
    assert batch["observation"]["observation.images.wrist_image"].shape == (2, 84, 84, 3)
    assert batch["observation"]["observation.state"].shape == (2, 15)
    assert batch["action"].shape == (2, 4, 7)
    assert batch["complementary_data"]["action_is_pad"].shape == (2, 4)
    assert batch["complementary_data"]["task"] == ["move the block", "move the block"]
    with pytest.raises(ContractError, match="unknown CALVIN language segment"):
        index.record(-1, 10)


def test_calvin_stateful_dataset_preserves_causal_actions_and_segment_identity(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=4)

    assert len(dataset) == 8
    assert len(dataset.episode_manifest) == 2
    assert [len(episode.sample_keys) for episode in dataset.episode_manifest] == [4, 4]
    assert len(set(dataset.sample_keys)) == len(dataset)
    assert "segment-00000000" in dataset.episode_manifest[0].sample_keys[-1]
    assert "frame-00000013" in dataset.episode_manifest[0].sample_keys[-1]
    assert "segment-00000001" in dataset.episode_manifest[1].sample_keys[0]
    assert "frame-00000013" in dataset.episode_manifest[1].sample_keys[0]

    first = dataset.by_key(dataset.episode_manifest[0].sample_keys[0])
    second = dataset.by_key(dataset.episode_manifest[0].sample_keys[1])
    overlapping_reset = dataset.by_key(dataset.episode_manifest[1].sample_keys[0])
    assert first.transition_index == 0
    assert second.transition_index == 1
    np.testing.assert_array_equal(first.previous_executed_action, np.zeros(7, dtype=np.float32))
    np.testing.assert_array_equal(second.previous_executed_action, first.record.action)
    assert second.previous_executed_action[0] != second.record.action[0]
    np.testing.assert_array_equal(
        overlapping_reset.previous_executed_action,
        np.zeros(7, dtype=np.float32),
    )
    assert overlapping_reset.record.global_index == first.record.global_index + 3
    assert overlapping_reset.sample_key != dataset.episode_manifest[0].sample_keys[-1]
    assert overlapping_reset.episode_key != first.episode_key
    assert not hasattr(second.picf_evidence_frame, "action")
    assert not hasattr(second.picf_evidence_frame, "task")
    assert not second.previous_executed_action.flags.writeable
    with pytest.raises(KeyError, match="unknown CALVIN stateful sample key"):
        dataset.by_key("missing")


def test_stateful_evidence_prefix_is_target_free_causal_and_segment_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=4)

    def reject_record(*_args, **_kwargs):
        raise AssertionError("target-free history must not decode a full transition record")

    def reject_action(*_args, **_kwargs):
        raise AssertionError("target-free history must not read an action target")

    monkeypatch.setattr(index, "record", reject_record)
    monkeypatch.setattr(index, "action", reject_action)

    segment_zero_key = dataset.episode_manifest[0].sample_keys[-1]
    prefix = dataset.evidence_prefix_by_key(segment_zero_key, maximum_source_frames=3)
    assert len(prefix) == 3
    assert [frame.timestamp_s for frame in prefix] == pytest.approx([1 / 30, 2 / 30, 3 / 30])
    assert all(not hasattr(frame, "action") for frame in prefix)
    assert all(not hasattr(frame, "task") for frame in prefix)
    assert all(not hasattr(frame, "global_index") for frame in prefix)
    assert all(
        tuple(item.key for item in frame.sensor_observations)
        == tuple(spec[1] for spec in CALVIN_OBSERVATION_SPECS)
        for frame in prefix
    )

    overlapping_segment_start = dataset.episode_manifest[1].sample_keys[0]
    reset_prefix = dataset.evidence_prefix_by_key(
        overlapping_segment_start,
        maximum_source_frames=4,
    )
    assert len(reset_prefix) == 1
    assert reset_prefix[0].timestamp_s == pytest.approx(3 / 30)

    with pytest.raises(ContractError, match="positive integer"):
        dataset.evidence_prefix_by_key(segment_zero_key, maximum_source_frames=0)
    with pytest.raises(KeyError, match="unknown CALVIN stateful sample key"):
        dataset.evidence_prefix_by_key("missing", maximum_source_frames=4)
    with pytest.raises(ContractError, match="global index must be an integer"):
        index.picf_evidence_frame(0, True)


def test_physical_evidence_prefix_uses_raw_episode_time_without_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=4)

    def reject_record(*_args, **_kwargs):
        raise AssertionError("physical evidence history must not decode a target record")

    def reject_action(*_args, **_kwargs):
        raise AssertionError("physical evidence history must not read an action target")

    monkeypatch.setattr(index, "record", reject_record)
    monkeypatch.setattr(index, "action", reject_action)
    event_key = next(
        key for key in dataset.sample_keys if dataset.source_global_index_by_key(key) == 13
    )
    assert dataset.timestamp_s_by_key(event_key) == pytest.approx(3 / 30)
    prefix = dataset.evidence_prefix_by_key(event_key, maximum_source_frames=4)

    assert len(prefix) == 4
    assert [frame.timestamp_s for frame in prefix] == pytest.approx([0.0, 1 / 30, 2 / 30, 3 / 30])
    assert all(not hasattr(frame, "action") for frame in prefix)
    assert all(not hasattr(frame, "task") for frame in prefix)
    assert all(not hasattr(frame, "global_index") for frame in prefix)
    assert tuple(item.key for item in prefix[-1].sensor_observations) == tuple(
        spec[1] for spec in CALVIN_OBSERVATION_SPECS
    )
    with pytest.raises(ContractError, match="positive integer"):
        dataset.evidence_prefix_by_key(event_key, maximum_source_frames=0)
    with pytest.raises(ContractError, match="global index must be an integer"):
        index.source_picf_evidence_frame(True)


def test_causal_video_audit_covers_every_transition_and_renders_tasks(tmp_path: Path) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=4)
    report, samples = audit_stateful_dataset(dataset, maximum_frames=4, tubelet_size=2)

    assert report["transitions"] == 8
    assert report["language_segments"] == 2
    for sensor_key in (
        "observation.images.rgb_static",
        "observation.images.rgb_gripper",
    ):
        sensor = report["sensors"][sensor_key]
        assert sensor["frame_count_histogram"] == {"0": 2, "2": 4, "4": 2}
        assert sensor["complete_tubelet_clips"] == 6
        assert sensor["full_window_clips"] == 2
    assert len(samples) == 2
    sheet = tmp_path / "causal-video.png"
    render_contact_sheet(samples, sheet)
    with Image.open(sheet) as image:
        assert image.width > 1000
        assert image.height > 200


def test_stateful_action_horizons_decode_one_full_frame_and_cache_overlap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    index.clear_action_cache()
    original_record = index.record
    original_load_action = index._load_action
    full_record_calls: list[tuple[int, int]] = []
    action_load_calls: list[int] = []

    def counted_record(segment_index: int, global_index: int):
        full_record_calls.append((segment_index, global_index))
        return original_record(segment_index, global_index)

    def counted_load_action(global_index: int):
        action_load_calls.append(global_index)
        return original_load_action(global_index)

    monkeypatch.setattr(index, "record", counted_record)
    monkeypatch.setattr(index, "_load_action", counted_load_action)
    first = index.stateful_transition_sample(0, 10, action_horizon=4)
    second = index.stateful_transition_sample(0, 11, action_horizon=4)

    assert full_record_calls == [(0, 10), (0, 11)]
    assert action_load_calls == [11, 12, 13, 10]
    np.testing.assert_array_equal(first.host_sample.action[1], second.record.action)
    np.testing.assert_array_equal(second.previous_executed_action, first.record.action)
    assert not first.host_sample.action.flags.writeable
    assert not second.previous_executed_action.flags.writeable


def test_physical_event_sweep_matches_brute_force_and_fails_closed_on_manifest_drift(
    tmp_path: Path,
) -> None:
    episode = CalvinEpisode(0, 100, 111)
    segments = (
        CalvinLanguageSegment(0, 104, 109, "task_0", "instruction 0", 0),
        CalvinLanguageSegment(1, 100, 103, "task_1", "instruction 1", 0),
        CalvinLanguageSegment(2, 102, 106, "task_2", "instruction 2", 0),
        CalvinLanguageSegment(3, 108, 111, "task_3", "instruction 3", 0),
    )
    index = CalvinDatasetIndex(
        split_root=tmp_path,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        control_hz=30,
        episodes=(episode,),
        segments=segments,
    )

    manifest = index.physical_episode_manifest(0)
    brute_force = tuple(
        (
            global_index,
            tuple(
                segment.index
                for segment in sorted(segments, key=lambda candidate: candidate.index)
                if segment.start <= global_index < segment.end
            ),
        )
        for global_index in range(episode.start, episode.end)
        if any(segment.start <= global_index < segment.end for segment in segments)
    )
    swept = tuple(
        (
            event.global_index,
            tuple(segment.index for segment in event.candidate_segments),
        )
        for event in manifest.events
    )
    assert swept == brute_force
    assert tuple(index.iter_physical_events()) == manifest.events
    assert tuple(index.physical_event(event.global_index) for event in manifest.events) == (
        manifest.events
    )
    assert len({event.event_key for event in manifest.events}) == len(manifest.events)
    assert tuple(segment.index for segment in manifest.event_at(104).candidate_segments) == (0, 2)

    overlap = manifest.event_at(104)
    with pytest.raises(ContractError, match="sorted by annotation index"):
        replace(overlap, candidate_segments=tuple(reversed(overlap.candidate_segments)))
    omitted_candidate = replace(overlap, candidate_segments=(segments[0],))
    replaced_events = tuple(
        omitted_candidate if event.global_index == overlap.global_index else event
        for event in manifest.events
    )
    with pytest.raises(ContractError, match="omitted or reordered"):
        replace(manifest, events=replaced_events)
    with pytest.raises(ContractError, match="duplicates, gaps, or source-order"):
        replace(manifest, events=manifest.events[:-1])
    with pytest.raises(ContractError, match="duplicates, gaps, or source-order"):
        replace(manifest, events=(manifest.events[0], *manifest.events))
    with pytest.raises(ContractError, match="duplicates, gaps, or source-order"):
        replace(manifest, events=tuple(reversed(manifest.events)))


def test_physical_overlap_requires_explicit_prompt_and_clips_selected_host_horizon(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )

    events = tuple(index.iter_physical_events())
    assert [event.global_index for event in events] == list(range(10, 17))
    event = index.physical_event(13)
    assert tuple(segment.index for segment in event.candidate_segments) == (0, 1)
    assert not hasattr(event, "task")
    assert not hasattr(event, "instruction")
    assert not hasattr(event, "canonical_segment")
    with pytest.raises(TypeError, match="selected_segment_index"):
        index.physical_sample(13, action_horizon=4)  # type: ignore[call-arg]
    with pytest.raises(ContractError, match="not an exact physical-event candidate"):
        index.physical_sample(13, selected_segment_index=7, action_horizon=4)

    move = index.physical_sample(13, selected_segment_index=0, action_horizon=4)
    light = index.physical_sample(13, selected_segment_index=1, action_horizon=4)
    assert move.event == light.event == event
    assert move.selected_segment.index == 0
    assert light.selected_segment.index == 1
    assert move.host_sample.observation["task"] == "move the block"
    assert light.host_sample.observation["task"] == "turn on the light"
    assert move.host_sample.action_is_pad.tolist() == [False, True, True, True]
    assert light.host_sample.action_is_pad.tolist() == [False, False, False, False]
    assert not move.reset
    assert not light.reset
    forged_left_censored_span = CalvinRawActionControlSpan.from_raw_actions(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        episode=event.episode,
        start_global_index=event.episode.start,
        end_global_index=event.global_index,
        action_global_indices=(10, 11, 12),
        raw_actions=np.stack((index.action(10), index.action(11), index.action(12))),
        left_censored_start=True,
    )
    with pytest.raises(ContractError, match="control span"):
        replace(light, incoming_control_span=forged_left_censored_span)
    batch = collate_calvin_molmoact2([move.host_sample, light.host_sample])
    assert batch["complementary_data"]["task"] == ["move the block", "turn on the light"]
    assert batch["action"].shape == (2, 4, 7)


def test_physical_control_spans_cover_label_gaps_and_reject_order_or_digest_drift(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    _rewrite_language_intervals(split, [(12, 14), (16, 17)])
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )

    assert [event.global_index for event in index.iter_physical_events()] == [12, 13, 16]
    first = index.physical_sample(12, selected_segment_index=0, action_horizon=4)
    assert first.reset
    assert not first.event.at_raw_episode_start
    assert first.event.reset_global_index == 10
    assert first.incoming_control_span.left_censored_start
    assert first.incoming_control_span.start_global_index == 10
    assert first.incoming_control_span.end_global_index == 12
    assert first.incoming_control_span.action_global_indices == (10, 11)
    np.testing.assert_array_equal(
        first.incoming_control_span.raw_actions,
        np.stack((index.action(10), index.action(11))),
    )

    after_gap = index.physical_sample(16, selected_segment_index=1, action_horizon=4)
    span = after_gap.incoming_control_span
    assert not after_gap.reset
    assert after_gap.event.reset_global_index is None
    assert not span.left_censored_start
    assert span.start_global_index == 13
    assert span.end_global_index == 16
    assert span.action_global_indices == (13, 14, 15)
    np.testing.assert_array_equal(
        span.raw_actions,
        np.stack((index.action(13), index.action(14), index.action(15))),
    )
    assert span.sha256 == span.recomputed_sha256

    with pytest.raises(ContractError, match="duplicates, gaps, or source-order"):
        replace(span, action_global_indices=(13, 15, 14))
    with pytest.raises(ContractError, match="duplicates, gaps, or source-order"):
        replace(span, action_global_indices=(13, 15))
    reordered_actions = span.raw_actions[::-1].copy()
    with pytest.raises(ContractError, match="digest disagrees"):
        replace(span, raw_actions=reordered_actions)
    with pytest.raises(ContractError, match="digest disagrees"):
        replace(span, sha256="0" * 64)


def test_physical_reset_and_nested_values_are_raw_boundary_exact_and_immutable(
    tmp_path: Path,
) -> None:
    episodes = (CalvinEpisode(0, 10, 13), CalvinEpisode(1, 20, 22))
    segments = (
        CalvinLanguageSegment(0, 11, 13, "late", "late label", 0),
        CalvinLanguageSegment(1, 20, 22, "start", "boundary label", 1),
    )
    metadata_only = CalvinDatasetIndex(
        split_root=tmp_path,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        control_hz=30,
        episodes=episodes,
        segments=segments,
    )
    assert metadata_only.physical_event(11).reset
    assert not metadata_only.physical_event(11).at_raw_episode_start
    assert not metadata_only.physical_event(12).reset
    assert metadata_only.physical_event(20).reset
    assert metadata_only.physical_event(20).at_raw_episode_start

    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    boundary = index.physical_sample(10, selected_segment_index=0, action_horizon=4)
    assert boundary.reset
    assert boundary.event.at_raw_episode_start
    assert boundary.event.reset_global_index == 10
    assert boundary.incoming_control_span.left_censored_start
    assert boundary.incoming_control_span.action_global_indices == ()
    assert boundary.incoming_control_span.raw_actions.shape == (0, 7)
    assert boundary.incoming_control_span.sha256 == boundary.incoming_control_span.recomputed_sha256

    with pytest.raises(FrozenInstanceError):
        boundary.event.global_index = 11  # type: ignore[misc]
    with pytest.raises(ValueError, match="WRITEABLE"):
        boundary.incoming_control_span.raw_actions.setflags(write=True)
    with pytest.raises(TypeError):
        boundary.host_sample.observation["task"] = "changed"  # type: ignore[index]
    with pytest.raises(ValueError, match="read-only"):
        boundary.host_sample.action[0, 0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        boundary.record.action[0] = 1.0

    source_actions = np.ones((1, 7), dtype=np.float32)
    copied_span = CalvinRawActionControlSpan.from_raw_actions(
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        episode=episodes[0],
        start_global_index=10,
        end_global_index=11,
        action_global_indices=(10,),
        raw_actions=source_actions,
        left_censored_start=False,
    )
    source_actions[0, 0] = 9.0
    assert copied_span.raw_actions[0, 0] == 1.0
    assert not copied_span.raw_actions.flags.writeable


def test_physical_contract_is_additive_and_preserves_legacy_segment_resets(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    legacy = CalvinStatefulTransitionDataset(index, action_horizon=4)

    assert len(legacy) == 8
    assert len(tuple(index.iter_physical_events())) == 7
    legacy_overlap = legacy.by_key(legacy.episode_manifest[1].sample_keys[0])
    physical_overlap = index.physical_sample(
        13,
        selected_segment_index=1,
        action_horizon=4,
    )
    assert legacy_overlap.transition_index == 0
    np.testing.assert_array_equal(
        legacy_overlap.previous_executed_action,
        np.zeros(7, dtype=np.float32),
    )
    assert not physical_overlap.reset
    assert physical_overlap.incoming_control_span.action_global_indices == (12,)
    np.testing.assert_array_equal(
        physical_overlap.incoming_control_span.raw_actions[0],
        index.action(12),
    )

    legacy_host = index.molmoact2_sample(0, 13, action_horizon=4)
    selected_host = index.physical_sample(
        13,
        selected_segment_index=0,
        action_horizon=4,
    ).host_sample
    assert legacy_host.observation["task"] == selected_host.observation["task"]
    np.testing.assert_array_equal(legacy_host.action, selected_host.action)
    np.testing.assert_array_equal(legacy_host.action_is_pad, selected_host.action_is_pad)


def test_physical_transition_dataset_deduplicates_events_and_requires_prompt_overlay(
    tmp_path: Path,
) -> None:
    split = tmp_path / "training"
    _write_split(split)
    index = CalvinDatasetIndex.load(
        split,
        dataset_id="calvin-test",
        dataset_revision="sha256:test",
        dataset_manifest=_split_manifest(split),
    )
    legacy = CalvinStatefulTransitionDataset(index, action_horizon=4)
    physical = CalvinPhysicalTransitionDataset(index, action_horizon=4)

    assert len(legacy) == 8
    assert len(physical) == 7
    assert len(physical.sample_keys) == len(set(physical.sample_keys))
    (episode,) = physical.episode_manifest
    assert episode.episode_key == "calvin-source-episode-00000000"
    assert episode.sample_keys == physical.sample_keys
    first_key = physical.sample_keys[0]
    assert physical.source_global_index_by_key(first_key) == 10
    assert physical.future_source_global_indices_by_key(first_key, count=4) == (
        11,
        12,
        13,
        14,
    )
    last_key = physical.sample_keys[-1]
    with pytest.raises(ContractError, match="crosses a raw episode reset"):
        physical.future_source_global_indices_by_key(last_key, count=2)

    overlap_key = index.physical_event(13).event_key
    assert physical.candidate_segment_indices_by_key(overlap_key) == (0, 1)
    with pytest.raises(TypeError, match="selected_segment_index"):
        physical.by_key(overlap_key)  # type: ignore[call-arg]
    move = physical.by_key(overlap_key, selected_segment_index=0)
    light = physical.by_key(overlap_key, selected_segment_index=1)
    assert move.sample_key == light.sample_key == overlap_key
    assert move.episode_key == light.episode_key == episode.episode_key
    assert move.transition_index == light.transition_index == 3
    assert move.record.task == "move the block"
    assert light.record.task == "turn on the light"


def test_real_calvin_debug_split_decodes_when_present() -> None:
    split = (
        Path(__file__).parents[1] / "data" / "calvin_download" / "calvin_debug_dataset" / "training"
    )
    if not split.is_dir():
        pytest.skip("local official CALVIN debug split is absent")
    manifest = load_dataset_file_manifest(
        Path(__file__).parents[1] / "evidence/calvin_dataset_audit/training_source_manifest.json"
    )
    index = CalvinDatasetIndex.load(
        split,
        dataset_revision=CALVIN_DEBUG_REVISION,
        dataset_manifest=manifest,
    )
    assert len(index.segments) == 9
    record = index.record(0, index.segments[0].start)
    assert record.task == "move the light switch to turn on the yellow light"
    assert record.action.shape == (7,)
    assert record.array_observations[4].value.shape == (160, 120, 6)

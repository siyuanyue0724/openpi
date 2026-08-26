from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.dataset_manifest import build_dataset_file_manifest
from picf_next.lingbot_native.calvin import (
    build_native_calvin_episode_domain,
    build_native_calvin_physical_stream_plan,
    build_native_calvin_stream_plan,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    REPRESENTATION_TRIAL_SPLIT_SCHEMA,
    RepresentationTrialSplit,
    build_representation_trial_split,
    build_representation_trial_split_with_reference_evaluation,
    verify_representation_trial_split_training_evidence,
)
from picf_next.training.control import load_frozen_episode_stream_plan

ROOT = Path(__file__).resolve().parents[2]


def _dataset(tmp_path: Path) -> CalvinStatefulTransitionDataset:
    split_root = tmp_path / "training"
    split_root.mkdir()
    (split_root / "manifest-stub").write_bytes(b"representation-split-test")
    manifest = build_dataset_file_manifest(
        split_root,
        dataset_id="calvin-representation-test",
        dataset_revision="sha256:representation-test",
        split_name=split_root.name,
        relative_paths=("manifest-stub",),
    )
    episodes: list[CalvinEpisode] = []
    segments: list[CalvinLanguageSegment] = []
    for episode_index in range(40):
        start = episode_index * 20
        episodes.append(CalvinEpisode(episode_index, start, start + 19))
        segments.extend(
            (
                CalvinLanguageSegment(
                    len(segments),
                    start,
                    start + 9,
                    "task-a",
                    "move object a",
                    episode_index,
                ),
                CalvinLanguageSegment(
                    len(segments) + 1,
                    start + 10,
                    start + 19,
                    "task-b",
                    "move object b",
                    episode_index,
                ),
            )
        )
    index = CalvinDatasetIndex(
        split_root=split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        control_hz=30,
        episodes=tuple(episodes),
        segments=tuple(segments),
        dataset_manifest=manifest,
    )
    return CalvinStatefulTransitionDataset(index, action_horizon=4)


def _split(tmp_path: Path) -> tuple[CalvinStatefulTransitionDataset, RepresentationTrialSplit]:
    dataset = _dataset(tmp_path)
    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-split-test",
        seed=71,
        global_batch_size=2,
        total_steps=8,
    )
    split = build_representation_trial_split(
        plan,
        dataset,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )
    return dataset, split


def test_representation_split_is_source_disjoint_balanced_and_roundtrips(
    tmp_path: Path,
) -> None:
    dataset, split = _split(tmp_path)
    training_sources = set(split.training_source_episode_indices)
    validation_sources = {item.source_episode_index for item in split.validation_segments}
    heldout_sources = {item.source_episode_index for item in split.heldout_segments}

    assert split.training_sample_count == 8
    assert split.stream_domain_excluded_source_episode_indices == ()
    assert not training_sources & validation_sources
    assert not training_sources & heldout_sources
    assert not validation_sources & heldout_sources
    assert {item.task_key for item in split.validation_segments} == {
        "task-a",
        "task-b",
    }
    assert {item.task_key for item in split.heldout_segments} == {
        "task-a",
        "task-b",
    }
    assert len(split.validation_segments) == 4
    assert len(split.heldout_segments) == 4
    for partition in (split.validation_segments, split.heldout_segments):
        for task_key in ("task-a", "task-b"):
            task_sources = {
                item.source_episode_index for item in partition if item.task_key == task_key
            }
            assert len(task_sources) == 2
    expected_training_segments = {
        dataset.locator_by_key(transition.sample.sample_key).segment_index
        for step in range(4)
        for transition in build_native_calvin_stream_plan(
            dataset,
            comparison_id="representation-split-test",
            seed=71,
            global_batch_size=2,
            total_steps=8,
        )
        .global_batch(step)
        .transitions
    }
    assert set(split.training_segment_indices) == expected_training_segments

    artifact = tmp_path / "representation-split.json"
    split.write(artifact)
    assert RepresentationTrialSplit.load(artifact) == split
    assert RepresentationTrialSplit.from_dict(split.as_dict()) == split


def test_representation_split_selection_never_decodes_a_training_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-no-decode",
        seed=31,
        global_batch_size=2,
        total_steps=3,
    )

    def reject_decode(_sample_key: str) -> None:
        raise AssertionError("source-only split selection decoded a model sample")

    monkeypatch.setattr(dataset, "by_key", reject_decode)
    split = build_representation_trial_split(
        plan,
        dataset,
        training_steps=3,
        partition_seed=23,
    )
    assert split.training_sample_count == 6


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("training_sample_count", 7),
        ("training_sample_keys_sha256", "0" * 64),
        ("training_source_global_indices_sha256", "1" * 64),
        ("training_segment_indices", (999,)),
        ("training_source_episode_indices", (998,)),
    ),
)
def test_representation_split_runtime_verification_rebuilds_all_training_evidence(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    dataset, split = _split(tmp_path)
    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-split-test",
        seed=71,
        global_batch_size=2,
        total_steps=8,
    )

    assert verify_representation_trial_split_training_evidence(split, plan, dataset) is split
    changed = replace(split, **{field: value})
    with pytest.raises(ValueError, match="training evidence differs"):
        verify_representation_trial_split_training_evidence(changed, plan, dataset)


def test_representation_split_rejects_tamper_and_source_overlap(tmp_path: Path) -> None:
    _, split = _split(tmp_path)
    payload = split.as_dict()
    payload["training_steps"] = 5
    with pytest.raises(ValueError, match="SHA-256 changed"):
        RepresentationTrialSplit.from_dict(payload)

    validation = list(split.validation_segments)
    validation[0] = replace(
        validation[0],
        source_episode_index=split.training_source_episode_indices[0],
    )
    with pytest.raises(ValueError, match="reused a training source episode"):
        replace(split, validation_segments=tuple(validation))

    artifact = tmp_path / "tampered.json"
    artifact.write_text(json.dumps({"schema": "wrong"}), encoding="ascii")
    with pytest.raises(ValueError, match="fields differ"):
        RepresentationTrialSplit.load(artifact)


def test_representation_split_rejects_plan_dataset_identity_drift(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-identity",
        seed=13,
        global_batch_size=2,
        total_steps=2,
    )
    changed = replace(plan, dataset_revision="sha256:changed")
    with pytest.raises(ValueError, match="identities differ"):
        build_representation_trial_split(
            changed,
            dataset,
            training_steps=2,
            partition_seed=11,
        )


def test_stream_plan_excludes_complete_source_episodes(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    excluded = (0, 3, 7)
    plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-excluded-source-test",
        seed=13,
        global_batch_size=2,
        total_steps=4,
        lane_interleave_factor=2,
        excluded_source_episode_indices=excluded,
    )
    observed_sources = {
        int(dataset.index.segments[dataset.locator_by_key(sample_key).segment_index].episode_index)
        for episode in plan.episodes
        for sample_key in episode.sample_keys
    }
    assert observed_sources.isdisjoint(excluded)
    assert len(plan.episodes) == len(dataset.episode_manifest) - 2 * len(excluded)
    assert plan.episodes == build_native_calvin_episode_domain(
        dataset,
        excluded_source_episode_indices=excluded,
    )
    metadata = tmp_path / "excluded-stream.json"
    plan.write_metadata(metadata)
    assert (
        load_frozen_episode_stream_plan(
            metadata,
            episodes=build_native_calvin_episode_domain(
                dataset,
                excluded_source_episode_indices=excluded,
            ),
        )
        == plan
    )
    with pytest.raises(ValueError, match="episode manifest differs"):
        load_frozen_episode_stream_plan(
            metadata,
            episodes=build_native_calvin_episode_domain(dataset),
        )

    with pytest.raises(ValueError, match="unique and sorted"):
        build_native_calvin_stream_plan(
            dataset,
            comparison_id="representation-excluded-source-test",
            seed=13,
            global_batch_size=2,
            total_steps=4,
            excluded_source_episode_indices=(3, 0),
        )
    with pytest.raises(ValueError, match="outside the dataset"):
        build_native_calvin_stream_plan(
            dataset,
            comparison_id="representation-excluded-source-test",
            seed=13,
            global_batch_size=2,
            total_steps=4,
            excluded_source_episode_indices=(len(dataset.index.episodes),),
        )


def _reference_compatible_interleaved_trial(
    dataset: CalvinStatefulTransitionDataset,
) -> tuple[RepresentationTrialSplit, RepresentationTrialSplit]:
    baseline_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-reference-test",
        seed=0,
        global_batch_size=2,
        total_steps=4,
    )
    reference = build_representation_trial_split(
        baseline_plan,
        dataset,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )
    candidate_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-reference-test",
        seed=0,
        global_batch_size=2,
        total_steps=4,
        lane_interleave_factor=4,
        excluded_source_episode_indices=reference.evaluation_source_episode_indices,
    )
    candidate = build_representation_trial_split_with_reference_evaluation(
        candidate_plan,
        dataset,
        training_steps=4,
        evaluation_reference=reference,
    )
    return reference, candidate


def test_reference_evaluation_split_preserves_exact_bank_and_roundtrips(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    reference, candidate = _reference_compatible_interleaved_trial(dataset)

    assert candidate.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
    assert candidate.evaluation_reference_split_artifact_sha256 == reference.artifact_sha256
    assert candidate.validation_segments == reference.validation_segments
    assert candidate.heldout_segments == reference.heldout_segments
    assert candidate.training_sample_count == reference.training_sample_count
    assert candidate.evaluation_source_episode_indices == (
        reference.evaluation_source_episode_indices
    )
    assert candidate.stream_domain_excluded_source_episode_indices == (
        candidate.evaluation_source_episode_indices
    )
    assert not set(candidate.training_source_episode_indices) & {
        item.source_episode_index
        for item in (*candidate.validation_segments, *candidate.heldout_segments)
    }
    assert RepresentationTrialSplit.from_dict(candidate.as_dict()) == candidate


def test_reference_evaluation_split_rejects_source_metadata_tamper(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    reference, candidate = _reference_compatible_interleaved_trial(dataset)
    changed_validation = list(reference.validation_segments)
    changed_validation[0] = replace(
        changed_validation[0],
        source_start=changed_validation[0].source_start + 1,
    )
    tampered = replace(reference, validation_segments=tuple(changed_validation))
    candidate_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id=candidate.comparison_id,
        seed=0,
        global_batch_size=2,
        total_steps=4,
        lane_interleave_factor=4,
        excluded_source_episode_indices=reference.evaluation_source_episode_indices,
    )

    with pytest.raises(ValueError, match="differs from source metadata"):
        build_representation_trial_split_with_reference_evaluation(
            candidate_plan,
            dataset,
            training_steps=4,
            evaluation_reference=tampered,
        )


def test_reference_evaluation_split_can_freeze_one_bank_across_training_budgets(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    reference_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-cross-budget-test",
        seed=0,
        global_batch_size=2,
        total_steps=4,
    )
    reference = build_representation_trial_split(
        reference_plan,
        dataset,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )
    candidate_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-cross-budget-test",
        seed=0,
        global_batch_size=2,
        total_steps=6,
        excluded_source_episode_indices=reference.evaluation_source_episode_indices,
    )

    with pytest.raises(ValueError, match="same training budget"):
        build_representation_trial_split_with_reference_evaluation(
            candidate_plan,
            dataset,
            training_steps=6,
            evaluation_reference=reference,
        )

    candidate = build_representation_trial_split_with_reference_evaluation(
        candidate_plan,
        dataset,
        training_steps=6,
        evaluation_reference=reference,
        require_equal_training_budget=False,
    )
    assert candidate.training_steps == 6
    assert candidate.training_sample_count == 12
    assert candidate.validation_segments == reference.validation_segments
    assert candidate.heldout_segments == reference.heldout_segments
    assert not set(candidate.training_source_episode_indices) & set(
        reference.evaluation_source_episode_indices
    )


def test_reference_evaluation_split_rejects_full_unexcluded_domain(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    reference_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-full-domain-test",
        seed=0,
        global_batch_size=2,
        total_steps=4,
    )
    reference = build_representation_trial_split(
        reference_plan,
        dataset,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )

    with pytest.raises(ValueError, match="must exactly exclude"):
        build_representation_trial_split_with_reference_evaluation(
            reference_plan,
            dataset,
            training_steps=4,
            evaluation_reference=reference,
        )


def test_reference_evaluation_split_rejects_extra_episode_filter(
    tmp_path: Path,
) -> None:
    dataset = _dataset(tmp_path)
    reference_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-extra-filter-test",
        seed=0,
        global_batch_size=2,
        total_steps=4,
    )
    reference = build_representation_trial_split(
        reference_plan,
        dataset,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )
    extra_source = next(
        index
        for index in range(len(dataset.index.episodes))
        if index not in reference.evaluation_source_episode_indices
    )
    excluded = tuple(sorted((*reference.evaluation_source_episode_indices, extra_source)))
    candidate_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id=reference.comparison_id,
        seed=0,
        global_batch_size=2,
        total_steps=4,
        excluded_source_episode_indices=excluded,
    )

    with pytest.raises(ValueError, match="must exactly exclude"):
        build_representation_trial_split_with_reference_evaluation(
            candidate_plan,
            dataset,
            training_steps=4,
            evaluation_reference=reference,
        )


def test_reference_evaluation_split_accepts_exact_physical_domain(
    tmp_path: Path,
) -> None:
    stateful = _dataset(tmp_path)
    reference_plan = build_native_calvin_stream_plan(
        stateful,
        comparison_id="representation-physical-domain-test",
        seed=0,
        global_batch_size=2,
        total_steps=4,
    )
    reference = build_representation_trial_split(
        reference_plan,
        stateful,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )
    physical = CalvinPhysicalTransitionDataset(stateful.index, action_horizon=4)
    candidate_plan = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id=reference.comparison_id,
        seed=0,
        global_batch_size=2,
        total_steps=4,
        excluded_source_episode_indices=reference.evaluation_source_episode_indices,
    )

    candidate = build_representation_trial_split_with_reference_evaluation(
        candidate_plan,
        physical,
        training_steps=4,
        evaluation_reference=reference,
    )
    assert candidate.stream_domain_excluded_source_episode_indices == (
        reference.evaluation_source_episode_indices
    )


def test_packaged_k1_reference_preserves_historical_v1_bytes() -> None:
    path = ROOT / "references/experiments/lingbot-representation-k1-200-reference-split.json"
    split = RepresentationTrialSplit.load(path)

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "392fd6b9ba6b15e015d39a14e5036bbd7eeaad407b44d1a9ab3bfda2835a31b7"
    )
    assert split.schema == REPRESENTATION_TRIAL_SPLIT_SCHEMA
    assert split.artifact_sha256 == (
        "b325631b03801d1d915edde400602f9d9734884de74f181dc1638fa96b1e8a00"
    )
    assert split.stream_plan_sha256 == (
        "1cdb8c619638b667cdeb45cca0e642b843fd3f96810fdf88c4d960ea46bdd995"
    )
    assert len(split.validation_segments) == 68
    assert len(split.heldout_segments) == 68

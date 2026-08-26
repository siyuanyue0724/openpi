from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.dataset_manifest import build_dataset_file_manifest
from picf_next.data.dense_evidence_coverage import (
    CALVIN_FULL_DENSE_MODALITIES,
    DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1,
    DenseEvidenceCoveragePlan,
    build_calvin_dense_evidence_coverage_plan,
)
from picf_next.lingbot_native.calvin import build_native_calvin_physical_stream_plan
from picf_next.lingbot_native.entity_evaluation_plan import build_entity_evaluation_plan
from picf_next.lingbot_native.representation_split import (
    build_representation_trial_split,
)


def _datasets(
    tmp_path: Path,
) -> tuple[CalvinPhysicalTransitionDataset, CalvinStatefulTransitionDataset]:
    split_root = tmp_path / "training"
    split_root.mkdir()
    (split_root / "manifest-stub").write_bytes(b"dense-evidence-coverage-test")
    manifest = build_dataset_file_manifest(
        split_root,
        dataset_id="calvin-coverage-test",
        dataset_revision="sha256:coverage-test",
        split_name="training",
        relative_paths=("manifest-stub",),
    )
    episodes = tuple(CalvinEpisode(index, index * 20, index * 20 + 19) for index in range(40))
    segments: list[CalvinLanguageSegment] = []
    for episode in episodes:
        segments.extend(
            (
                CalvinLanguageSegment(
                    len(segments),
                    episode.start,
                    episode.start + 9,
                    "task-a",
                    "move object a",
                    episode.index,
                ),
                CalvinLanguageSegment(
                    len(segments) + 1,
                    episode.start + 10,
                    episode.end,
                    "task-b",
                    "move object b",
                    episode.index,
                ),
            )
        )
    index = CalvinDatasetIndex(
        split_root=split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        control_hz=30,
        episodes=episodes,
        segments=tuple(segments),
        dataset_manifest=manifest,
    )
    return (
        CalvinPhysicalTransitionDataset(index, action_horizon=1),
        CalvinStatefulTransitionDataset(index, action_horizon=1),
    )


def _coverage(tmp_path: Path) -> DenseEvidenceCoveragePlan:
    physical, evaluation = _datasets(tmp_path)
    stream = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id="coverage-test",
        seed=17,
        global_batch_size=2,
        total_steps=4,
    )
    split = build_representation_trial_split(
        stream,
        physical,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    evaluation_plan = build_entity_evaluation_plan(split, evaluation, world_size=2)
    return build_calvin_dense_evidence_coverage_plan(
        stream_plan=stream,
        representation_split=split,
        evaluation_plan=evaluation_plan,
        physical_dataset=physical,
        evaluation_dataset=evaluation,
    )


def test_dense_evidence_coverage_is_exact_source_disjoint_and_roundtrips(
    tmp_path: Path,
) -> None:
    coverage = _coverage(tmp_path)

    assert coverage.modalities == CALVIN_FULL_DENSE_MODALITIES
    assert coverage.training_visit_count == 8
    training = tuple(record for record in coverage.records if record.partition == "training")
    evaluation = tuple(record for record in coverage.records if record.partition == "evaluation")
    assert len(training) == 8
    assert len(evaluation) == 6
    assert {record.source_global_index for record in training}.isdisjoint(
        record.source_global_index for record in evaluation
    )
    assert coverage.record_identities == tuple(sorted(coverage.record_identities))

    path = tmp_path / "coverage.json"
    coverage.write(path)
    assert DenseEvidenceCoveragePlan.load(path) == coverage
    with pytest.raises(FileExistsError):
        coverage.write(path)


def test_dense_evidence_coverage_can_bind_a_prefix_of_the_complete_stream(
    tmp_path: Path,
) -> None:
    physical, evaluation = _datasets(tmp_path)
    stream = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id="coverage-test",
        seed=17,
        global_batch_size=2,
        total_steps=4,
    )
    split = build_representation_trial_split(
        stream,
        physical,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    evaluation_plan = build_entity_evaluation_plan(split, evaluation, world_size=2)

    prefix = build_calvin_dense_evidence_coverage_plan(
        stream_plan=stream,
        representation_split=split,
        evaluation_plan=evaluation_plan,
        physical_dataset=physical,
        evaluation_dataset=evaluation,
        training_step_prefix=2,
    )

    assert prefix.stream_plan_sha256 == stream.plan_sha256
    assert prefix.representation_split_sha256 == split.artifact_sha256
    assert prefix.training_visit_count == 4
    assert sum(record.partition == "training" for record in prefix.records) == 4


def test_dense_evidence_coverage_can_reproduce_a_frozen_v1_contract(
    tmp_path: Path,
) -> None:
    physical, evaluation = _datasets(tmp_path)
    stream = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id="coverage-v1-test",
        seed=17,
        global_batch_size=2,
        total_steps=4,
    )
    split = build_representation_trial_split(
        stream,
        physical,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    evaluation_plan = build_entity_evaluation_plan(split, evaluation, world_size=2)

    coverage = build_calvin_dense_evidence_coverage_plan(
        stream_plan=stream,
        representation_split=split,
        evaluation_plan=evaluation_plan,
        physical_dataset=physical,
        evaluation_dataset=evaluation,
        schema=DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1,
    )

    assert coverage.schema == DENSE_EVIDENCE_COVERAGE_PLAN_SCHEMA_V1
    path = tmp_path / "coverage-v1.json"
    coverage.write(path)
    assert DenseEvidenceCoveragePlan.load(path) == coverage


def test_dense_evidence_coverage_includes_exact_causal_evaluation_history(
    tmp_path: Path,
) -> None:
    physical, evaluation = _datasets(tmp_path)
    stream = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id="coverage-history-test",
        seed=17,
        global_batch_size=2,
        total_steps=4,
    )
    split = build_representation_trial_split(
        stream,
        physical,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    evaluation_plan = build_entity_evaluation_plan(split, evaluation, world_size=2)

    coverage = build_calvin_dense_evidence_coverage_plan(
        stream_plan=stream,
        representation_split=split,
        evaluation_plan=evaluation_plan,
        physical_dataset=physical,
        evaluation_dataset=evaluation,
        evaluation_history_transitions=4,
    )

    eligible = tuple(item for item in evaluation_plan.items if item.transition_index >= 4)
    expected_history = {
        evaluation.source_global_index_by_key(history_key)
        for item in eligible
        for history_key in evaluation.history_sample_keys(item.sample_key)[-4:]
    }
    current = {item.source_global_index for item in evaluation_plan.items}
    evaluation_records = {
        record.source_global_index
        for record in coverage.records
        if record.partition == "evaluation"
    }
    assert coverage.evaluation_item_count == len(evaluation_plan.items)
    assert coverage.evaluation_record_count == len(current | expected_history)
    assert coverage.evaluation_history_transition_count == 4
    assert coverage.evaluation_history_visit_count == len(eligible) * 4
    assert evaluation_records == current | expected_history

    path = tmp_path / "history-coverage.json"
    coverage.write(path)
    assert DenseEvidenceCoveragePlan.load(path) == coverage


@pytest.mark.parametrize("training_step_prefix", (0, 5, True, 1.5))
def test_dense_evidence_coverage_rejects_invalid_stream_prefix(
    tmp_path: Path,
    training_step_prefix: object,
) -> None:
    physical, evaluation = _datasets(tmp_path)
    stream = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id="coverage-test",
        seed=17,
        global_batch_size=2,
        total_steps=4,
    )
    split = build_representation_trial_split(
        stream,
        physical,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    evaluation_plan = build_entity_evaluation_plan(split, evaluation, world_size=2)

    with pytest.raises(ContractError, match="training-step prefix"):
        build_calvin_dense_evidence_coverage_plan(
            stream_plan=stream,
            representation_split=split,
            evaluation_plan=evaluation_plan,
            physical_dataset=physical,
            evaluation_dataset=evaluation,
            training_step_prefix=training_step_prefix,  # type: ignore[arg-type]
        )


def test_dense_evidence_coverage_rejects_record_and_artifact_tamper(tmp_path: Path) -> None:
    coverage = _coverage(tmp_path)
    payload = coverage.as_dict()
    payload["records"][0]["sample_key"] = "changed"  # type: ignore[index]
    with pytest.raises(ContractError, match="record digest"):
        DenseEvidenceCoveragePlan.from_dict(payload)

    payload = coverage.as_dict()
    payload["training_visit_count"] = coverage.training_visit_count + 1
    with pytest.raises(ContractError, match="artifact SHA-256"):
        DenseEvidenceCoveragePlan.from_dict(payload)


def test_dense_evidence_coverage_rejects_nonreproducible_evaluation(tmp_path: Path) -> None:
    physical, evaluation = _datasets(tmp_path)
    stream = build_native_calvin_physical_stream_plan(
        physical,
        comparison_id="coverage-test",
        seed=17,
        global_batch_size=2,
        total_steps=4,
    )
    split = build_representation_trial_split(
        stream,
        physical,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    plan = build_entity_evaluation_plan(split, evaluation, world_size=2)
    changed_items = list(plan.items)
    changed_items[0] = replace(
        changed_items[0],
        source_global_index=changed_items[0].source_global_index + 1,
    )
    changed_plan = replace(plan, items=tuple(changed_items))

    with pytest.raises(ContractError, match="not source reproducible"):
        build_calvin_dense_evidence_coverage_plan(
            stream_plan=stream,
            representation_split=split,
            evaluation_plan=changed_plan,
            physical_dataset=physical,
            evaluation_dataset=evaluation,
        )


def test_dense_evidence_coverage_loader_rejects_wrong_schema(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": "wrong"}), encoding="ascii")
    with pytest.raises(ContractError, match="fields differ"):
        DenseEvidenceCoveragePlan.load(path)

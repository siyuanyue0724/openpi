from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.dataset_manifest import (
    build_dataset_file_manifest,
    load_dataset_file_manifest,
)
from picf_next.lingbot_native.calvin import build_native_calvin_stream_plan
from picf_next.lingbot_native.representation_evaluation import (
    RepresentationEvaluationPlan,
    build_representation_evaluation_plan,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    RepresentationTrialSplit,
    build_representation_trial_split,
)
from picf_next.training.control import (
    EpisodeSampleSequence,
    FrozenEpisodeStreamPlan,
    FrozenResetMixtureStreamPlan,
    load_frozen_episode_stream_plan,
)
from tools import build_lingbot_representation_evaluation_plan as evaluation_tool
from tools import build_lingbot_representation_split as split_tool


def _source_tree(tmp_path: Path) -> tuple[Path, Path]:
    split = tmp_path / "training"
    (split / ".hydra").mkdir(parents=True)
    (split / "lang_annotations").mkdir()
    (split / ".hydra/merged_config.yaml").write_text(
        "env:\n  control_freq: 30\n",
        encoding="ascii",
    )
    episode_count = 90
    starts = np.arange(episode_count, dtype=np.int64) * 4
    bounds = np.stack((starts, starts + 3), axis=1)
    np.save(split / "ep_start_end_ids.npy", bounds)
    np.save(split / "ep_lens.npy", np.full(episode_count, 4, dtype=np.int64))
    task_keys = tuple(f"task-{index % 3}" for index in range(episode_count))
    annotations = {
        "language": {
            "ann": [f"perform {task_key}" for task_key in task_keys],
            "task": list(task_keys),
        },
        "info": {"indx": [tuple(row) for row in bounds.tolist()]},
    }
    np.save(split / "lang_annotations/auto_lang_ann.npy", annotations)
    relative_paths = (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
    )
    manifest = build_dataset_file_manifest(
        split,
        dataset_id="representation-split-tool-test",
        dataset_revision="sha256:representation-split-tool-test",
        split_name=split.name,
        relative_paths=relative_paths,
    )
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="ascii",
    )
    return split, manifest_path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_representation_split_tool_publishes_one_interleaved_identity(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    split_root, manifest_path = _source_tree(tmp_path)
    stream_output = tmp_path / "stream-plan.json"
    split_output = tmp_path / "representation-split.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(split_tool.__file__),
            "--dataset-split",
            str(split_root),
            "--dataset-manifest",
            str(manifest_path),
            "--comparison-id",
            "representation-interleave-test",
            "--plan-seed",
            "71",
            "--global-batch-size",
            "2",
            "--total-steps",
            "8",
            "--lane-interleave-factor",
            "4",
            "--partition-seed",
            "19",
            "--segments-per-task",
            "2",
            "--stream-plan-output",
            str(stream_output),
            "--representation-split-output",
            str(split_output),
        ],
    )

    split_tool.main()

    report = json.loads(capsys.readouterr().out)
    assert report["lane_count"] == 8
    assert report["lane_interleave_factor"] == 4
    assert report["training_sample_count"] == 16
    representation_split = RepresentationTrialSplit.load(split_output)
    manifest = load_dataset_file_manifest(manifest_path)
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    plan = FrozenEpisodeStreamPlan.from_metadata(
        stream_output,
        episodes=tuple(
            EpisodeSampleSequence(
                episode_key=episode.episode_key,
                sample_keys=episode.sample_keys,
            )
            for episode in dataset.episode_manifest
        ),
    )
    assert plan.plan_sha256 == representation_split.stream_plan_sha256
    assert plan.lane_interleave_factor == 4


def test_representation_split_tool_preserves_reference_evaluation_bank(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    split_root, manifest_path = _source_tree(tmp_path)
    manifest = load_dataset_file_manifest(manifest_path)
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    plan_seed = 0
    baseline = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-reference-tool-test",
        seed=plan_seed,
        global_batch_size=2,
        total_steps=8,
    )
    reference = build_representation_trial_split(
        baseline,
        dataset,
        training_steps=8,
        partition_seed=19,
        segments_per_task=2,
    )
    reference_path = tmp_path / "reference-split.json"
    reference.write(reference_path)
    stream_output = tmp_path / "candidate-stream-plan.json"
    split_output = tmp_path / "candidate-representation-split.json"
    report_output = tmp_path / "candidate-build-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(split_tool.__file__),
            "--dataset-split",
            str(split_root),
            "--dataset-manifest",
            str(manifest_path),
            "--comparison-id",
            "representation-reference-tool-test",
            "--plan-seed",
            str(plan_seed),
            "--global-batch-size",
            "2",
            "--total-steps",
            "8",
            "--lane-interleave-factor",
            "4",
            "--evaluation-reference-split",
            str(reference_path),
            "--evaluation-reference-split-sha256",
            _sha256(reference_path),
            "--segments-per-task",
            "2",
            "--stream-plan-output",
            str(stream_output),
            "--representation-split-output",
            str(split_output),
            "--build-report-output",
            str(report_output),
        ],
    )

    split_tool.main()

    report = json.loads(capsys.readouterr().out)
    candidate_split = RepresentationTrialSplit.load(split_output)
    assert candidate_split.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
    assert candidate_split.validation_segments == reference.validation_segments
    assert candidate_split.heldout_segments == reference.heldout_segments
    assert report["evaluation_reference_preserved"] is True
    assert report["baseline_noninterleaved_stream"]["lane_interleave_factor"] == 1
    assert report["candidate_stream"]["lane_interleave_factor"] == 4
    assert report["candidate_stream"]["lane_revisit_lag_minimum"] == 4
    assert report["candidate_stream"]["lane_revisit_lag_maximum"] == 4
    assert report["excluded_evaluation_source_episode_count"] == len(
        reference.evaluation_source_episode_indices
    )
    assert report["baseline_overlap"]["sample_key_sequence_equal"] is False
    assert report["baseline_overlap"]["positionally_equal_sample_count"] < 16
    assert report["baseline_overlap"]["sample_key_multiset_intersection_count"] <= 16
    assert (
        report["evaluation_source_overlap"][
            "candidate_training_validation_source_episode_intersection_count"
        ]
        == 0
    )
    assert (
        report["evaluation_source_overlap"][
            "candidate_training_heldout_source_episode_intersection_count"
        ]
        == 0
    )
    assert (
        report["evaluation_source_overlap"]["validation_heldout_source_episode_intersection_count"]
        == 0
    )
    assert report["build_report_file_sha256"] == _sha256(report_output)

    evaluation_output = tmp_path / "candidate-evaluation-plan.json"

    def task_identity_resolver(_task_key: str) -> None:
        return None

    monkeypatch.setattr(
        evaluation_tool,
        "calvin_exact_task_loss_identities",
        task_identity_resolver,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(evaluation_tool.__file__),
            "--dataset-split",
            str(split_root),
            "--dataset-manifest",
            str(manifest_path),
            "--representation-split",
            str(split_output),
            "--representation-split-sha256",
            _sha256(split_output),
            "--evaluation-reference-split",
            str(reference_path),
            "--evaluation-reference-split-sha256",
            _sha256(reference_path),
            "--output",
            str(evaluation_output),
        ],
    )

    evaluation_tool.main()

    evaluation_report = json.loads(capsys.readouterr().out)
    assert evaluation_report["evaluation_reference_preserved"] is True
    assert (
        evaluation_report["evaluation_reference_split_artifact_sha256"] == reference.artifact_sha256
    )
    candidate_evaluation = RepresentationEvaluationPlan.load(evaluation_output)
    reference_evaluation = build_representation_evaluation_plan(
        reference,
        dataset,
        task_identity_resolver=task_identity_resolver,
    )
    assert candidate_evaluation.items == reference_evaluation.items
    assert candidate_evaluation.artifact_sha256 != reference_evaluation.artifact_sha256
    assert (
        candidate_evaluation.evaluation_reference_plan_sha256
        == reference_evaluation.artifact_sha256
    )
    assert candidate_evaluation.replay_seed_sha256 == reference_evaluation.artifact_sha256
    assert (
        evaluation_report["evaluation_reference_plan_artifact_sha256"]
        == reference_evaluation.artifact_sha256
    )


def test_representation_split_tool_builds_adr149_physical_stream(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    split_root, manifest_path = _source_tree(tmp_path)
    stream_output = tmp_path / "physical-stream-plan.json"
    split_output = tmp_path / "physical-representation-split.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(split_tool.__file__),
            "--dataset-split",
            str(split_root),
            "--dataset-manifest",
            str(manifest_path),
            "--comparison-id",
            "lingbot-vla2-native-picf-full",
            "--plan-seed",
            "71",
            "--global-batch-size",
            "2",
            "--total-steps",
            "8",
            "--physical-event-stream",
            "--partition-seed",
            "19",
            "--segments-per-task",
            "2",
            "--stream-plan-output",
            str(stream_output),
            "--representation-split-output",
            str(split_output),
        ],
    )

    split_tool.main()

    report = json.loads(capsys.readouterr().out)
    split = RepresentationTrialSplit.load(split_output)
    assert report["physical_event_stream"] is True
    assert report["candidate_stream"]["sample_count"] == 16
    assert split.stream_plan_sha256 == report["stream_plan_sha256"]
    assert split.training_sample_count == 16


def test_representation_split_tool_freezes_reference_bank_across_budgets(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    split_root, manifest_path = _source_tree(tmp_path)
    manifest = load_dataset_file_manifest(manifest_path)
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    reference_plan = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-cross-budget-tool-test",
        seed=0,
        global_batch_size=2,
        total_steps=8,
    )
    reference = build_representation_trial_split(
        reference_plan,
        dataset,
        training_steps=8,
        partition_seed=19,
        segments_per_task=2,
    )
    reference_path = tmp_path / "reference-split.json"
    reference.write(reference_path)
    stream_output = tmp_path / "candidate-stream-plan.json"
    split_output = tmp_path / "candidate-split.json"
    report_output = tmp_path / "candidate-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(split_tool.__file__),
            "--dataset-split",
            str(split_root),
            "--dataset-manifest",
            str(manifest_path),
            "--comparison-id",
            "representation-cross-budget-tool-test",
            "--plan-seed",
            "0",
            "--global-batch-size",
            "2",
            "--total-steps",
            "12",
            "--lane-interleave-factor",
            "1",
            "--evaluation-reference-split",
            str(reference_path),
            "--evaluation-reference-split-sha256",
            _sha256(reference_path),
            "--allow-reference-budget-change",
            "--segments-per-task",
            "2",
            "--stream-plan-output",
            str(stream_output),
            "--representation-split-output",
            str(split_output),
            "--build-report-output",
            str(report_output),
        ],
    )

    split_tool.main()

    report = json.loads(capsys.readouterr().out)
    candidate = RepresentationTrialSplit.load(split_output)
    assert report["evaluation_reference_budget_change"] is True
    assert report["baseline_noninterleaved_stream"] is None
    assert report["candidate_stream"]["lane_interleave_factor"] == 1
    assert candidate.training_steps == 12
    assert candidate.training_sample_count == 24
    assert candidate.validation_segments == reference.validation_segments
    assert candidate.heldout_segments == reference.heldout_segments
    assert not set(candidate.training_source_episode_indices) & set(
        reference.evaluation_source_episode_indices
    )


def test_representation_split_tool_builds_and_restores_exact_reset_mixture(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    split_root, manifest_path = _source_tree(tmp_path)
    manifest = load_dataset_file_manifest(manifest_path)
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    baseline = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-reset-mixture-tool-test",
        seed=0,
        global_batch_size=2,
        total_steps=16,
    )
    reference = build_representation_trial_split(
        baseline,
        dataset,
        training_steps=16,
        partition_seed=19,
        segments_per_task=2,
    )
    reference_path = tmp_path / "reset-mixture-reference-split.json"
    reference.write(reference_path)
    stream_output = tmp_path / "reset-mixture-stream-plan.json"
    split_output = tmp_path / "reset-mixture-representation-split.json"
    report_output = tmp_path / "reset-mixture-build-report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(split_tool.__file__),
            "--dataset-split",
            str(split_root),
            "--dataset-manifest",
            str(manifest_path),
            "--comparison-id",
            "representation-reset-mixture-tool-test",
            "--plan-seed",
            "0",
            "--global-batch-size",
            "2",
            "--total-steps",
            "16",
            "--lane-interleave-factor",
            "4",
            "--reset-mixture-numerator",
            "1",
            "--reset-mixture-denominator",
            "2",
            "--evaluation-reference-split",
            str(reference_path),
            "--evaluation-reference-split-sha256",
            _sha256(reference_path),
            "--segments-per-task",
            "2",
            "--stream-plan-output",
            str(stream_output),
            "--representation-split-output",
            str(split_output),
            "--build-report-output",
            str(report_output),
        ],
    )

    split_tool.main()

    report = json.loads(capsys.readouterr().out)
    assert report["candidate_stream"]["estimator_component_sample_counts"] == {
        "causal": 16,
        "reset": 16,
    }
    assert report["candidate_stream"]["lane_revisit_lag_minimum"] == 8
    assert report["candidate_stream"]["lane_revisit_lag_maximum"] == 8
    restored = load_frozen_episode_stream_plan(
        stream_output,
        episodes=tuple(
            EpisodeSampleSequence(
                episode_key=episode.episode_key,
                sample_keys=episode.sample_keys,
            )
            for episode in dataset.episode_manifest
            if dataset.index.segments[episode.segment_index].episode_index
            not in reference.evaluation_source_episode_indices
        ),
    )
    assert isinstance(restored, FrozenResetMixtureStreamPlan)
    assert restored.plan_sha256 == RepresentationTrialSplit.load(split_output).stream_plan_sha256

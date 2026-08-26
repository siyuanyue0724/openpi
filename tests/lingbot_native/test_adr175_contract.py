from __future__ import annotations

import json
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    build_dataset_file_manifest,
    file_sha256,
)
from picf_next.eval.calvin_task_relevance import (
    calvin_task_physical_relevance_inventory,
)
from picf_next.lingbot_native.adr175_contract import (
    ADR175_AMBIGUOUS_SET_ONLY,
    ADR175_AMBIGUOUS_STRATUM,
    Adr175BroadSupportContract,
    build_adr175_broad_support_contract,
    canonical_sha256,
)
from picf_next.lingbot_native.calvin import (
    build_native_calvin_physical_sample_plan,
    build_native_calvin_physical_stream_plan,
)
from picf_next.lingbot_native.entity_evaluation_plan import (
    EntityEvaluationPlan,
    build_entity_evaluation_plan,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    RepresentationEvaluationSegment,
    RepresentationTrialSplit,
    build_representation_trial_split_with_reference_evaluation,
)
from picf_next.training.control import FrozenEpisodeStreamPlan
from tools import build_adr175_broad_support_contract as contract_tool


@dataclass(frozen=True, slots=True)
class _BroadFixture:
    split_root: Path
    manifest: DatasetFileManifest
    manifest_path: Path
    dataset: CalvinPhysicalTransitionDataset
    stream_plan: FrozenEpisodeStreamPlan
    stream_plan_path: Path
    representation_split: RepresentationTrialSplit
    representation_split_path: Path
    entity_evaluation_plan: EntityEvaluationPlan
    entity_evaluation_plan_path: Path
    training_prefix_steps: int


def _broad_fixture(tmp_path: Path) -> _BroadFixture:
    split_root = tmp_path / "training"
    (split_root / ".hydra").mkdir(parents=True)
    (split_root / "lang_annotations").mkdir()
    (split_root / ".hydra/merged_config.yaml").write_text(
        "env:\n  control_freq: 30\n",
        encoding="ascii",
    )

    inventory = calvin_task_physical_relevance_inventory()
    roles = ("training", "validation-0", "validation-1", "heldout-0", "heldout-1")
    episodes: list[CalvinEpisode] = []
    segments: list[CalvinLanguageSegment] = []
    training_sources: list[int] = []
    training_segments: list[int] = []
    validation: list[RepresentationEvaluationSegment] = []
    heldout: list[RepresentationEvaluationSegment] = []
    role_by_episode: dict[int, str] = {}
    for relevance in inventory:
        for role in roles:
            episode_index = len(episodes)
            start = episode_index * 4
            episode = CalvinEpisode(episode_index, start, start + 3)
            segment = CalvinLanguageSegment(
                len(segments),
                start,
                start + 3,
                relevance.task_key,
                f"perform {relevance.task_key}",
                episode_index,
            )
            episodes.append(episode)
            segments.append(segment)
            role_by_episode[episode_index] = role
            if role == "training":
                training_sources.append(episode_index)
                training_segments.append(segment.index)
            elif role.startswith("validation"):
                validation.append(
                    RepresentationEvaluationSegment(
                        task_key=segment.task_key,
                        segment_index=segment.index,
                        source_episode_index=episode_index,
                        source_start=segment.start,
                        source_end=segment.end,
                    )
                )
            else:
                heldout.append(
                    RepresentationEvaluationSegment(
                        task_key=segment.task_key,
                        segment_index=segment.index,
                        source_episode_index=episode_index,
                        source_start=segment.start,
                        source_end=segment.end,
                    )
                )

    bounds = np.asarray([[episode.start, episode.end] for episode in episodes], dtype=np.int64)
    np.save(split_root / "ep_start_end_ids.npy", bounds)
    np.save(split_root / "ep_lens.npy", np.full(len(episodes), 4, dtype=np.int64))
    annotations = {
        "language": {
            "ann": [segment.instruction for segment in segments],
            "task": [segment.task_key for segment in segments],
        },
        "info": {"indx": [(segment.start, segment.end) for segment in segments]},
    }
    np.save(split_root / "lang_annotations/auto_lang_ann.npy", annotations)
    relative_paths = (
        ".hydra/merged_config.yaml",
        "ep_lens.npy",
        "ep_start_end_ids.npy",
        "lang_annotations/auto_lang_ann.npy",
    )
    manifest = build_dataset_file_manifest(
        split_root,
        dataset_id="adr175-broad-test",
        dataset_revision="sha256:adr175-broad-test",
        split_name=split_root.name,
        relative_paths=relative_paths,
    )
    manifest_path = tmp_path / "dataset-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest.to_dict(), sort_keys=True) + "\n",
        encoding="ascii",
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
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    evaluation_sources = tuple(
        sorted(
            episode_index for episode_index, role in role_by_episode.items() if role != "training"
        )
    )
    stream_plan = build_native_calvin_physical_stream_plan(
        dataset,
        comparison_id="adr175-matched-three-arm",
        seed=20260721,
        global_batch_size=2,
        total_steps=80,
        excluded_source_episode_indices=evaluation_sources,
    )
    stream_plan_path = tmp_path / "stream-plan.json"
    stream_plan.write_metadata(stream_plan_path)
    full_sample_keys = [
        transition.sample.sample_key
        for optimizer_step in range(stream_plan.total_steps)
        for transition in stream_plan.global_batch(optimizer_step).transitions
    ]
    full_source_indices = [
        dataset.source_global_index_by_key(sample_key) for sample_key in full_sample_keys
    ]
    representation_split = RepresentationTrialSplit(
        schema=REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_manifest_sha256=manifest.tree_sha256,
        comparison_id=stream_plan.comparison_id,
        stream_plan_sha256=stream_plan.plan_sha256,
        partition_seed=20260729,
        training_steps=stream_plan.total_steps,
        training_sample_count=len(full_sample_keys),
        training_sample_keys_sha256=canonical_sha256(full_sample_keys),
        training_source_global_indices_sha256=canonical_sha256(full_source_indices),
        training_segment_indices=tuple(sorted(training_segments)),
        training_source_episode_indices=tuple(sorted(training_sources)),
        segments_per_task=2,
        validation_segments=tuple(
            sorted(validation, key=lambda item: (item.task_key, item.segment_index))
        ),
        heldout_segments=tuple(
            sorted(heldout, key=lambda item: (item.task_key, item.segment_index))
        ),
        evaluation_reference_split_artifact_sha256="a" * 64,
    )
    representation_split_path = tmp_path / "representation-split.json"
    representation_split.write(representation_split_path)
    stateful_dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    entity_evaluation_plan = build_entity_evaluation_plan(
        representation_split,
        stateful_dataset,
        world_size=2,
    )
    entity_evaluation_plan_path = tmp_path / "entity-evaluation-plan.json"
    entity_evaluation_plan.write(entity_evaluation_plan_path)
    return _BroadFixture(
        split_root=split_root,
        manifest=manifest,
        manifest_path=manifest_path,
        dataset=dataset,
        stream_plan=stream_plan,
        stream_plan_path=stream_plan_path,
        representation_split=representation_split,
        representation_split_path=representation_split_path,
        entity_evaluation_plan=entity_evaluation_plan,
        entity_evaluation_plan_path=entity_evaluation_plan_path,
        training_prefix_steps=60,
    )


def _build(fixture: _BroadFixture) -> Adr175BroadSupportContract:
    return build_adr175_broad_support_contract(
        dataset=fixture.dataset,
        stream_plan=fixture.stream_plan,
        representation_split=fixture.representation_split,
        entity_evaluation_plan=fixture.entity_evaluation_plan,
        training_prefix_steps=fixture.training_prefix_steps,
    )


def test_adr175_contract_roundtrips_exclusively_and_rejects_tampering(
    tmp_path: Path,
) -> None:
    fixture = _broad_fixture(tmp_path)
    contract = _build(fixture)
    output = tmp_path / "adr175-contract.json"

    contract.write(output)
    assert Adr175BroadSupportContract.load(output) == contract
    assert Adr175BroadSupportContract.from_dict(contract.as_dict()) == contract
    with pytest.raises(FileExistsError):
        contract.write(output)

    payload = json.loads(output.read_text(encoding="ascii"))
    payload["training_prefix_steps"] += 1
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="ascii")
    with pytest.raises(ValueError, match="artifact SHA-256|sample count"):
        Adr175BroadSupportContract.load(output)


def test_adr175_contract_freezes_29_exact_and_5_ambiguous_tasks(tmp_path: Path) -> None:
    contract = _build(_broad_fixture(tmp_path))
    evaluation_tasks = {item.task_key for item in contract.evaluation_items}
    exact_tasks = {item.task_key for item in contract.evaluation_items if item.exact_action_target}
    ambiguous_tasks = evaluation_tasks - exact_tasks

    assert contract.exact_task_count == 29
    assert contract.ambiguous_task_count == 5
    assert len(evaluation_tasks) == 34
    assert len(exact_tasks) == 29
    assert len(ambiguous_tasks) == 5
    assert len(contract.evaluation_items) == 136
    assert all(
        sum(
            item.partition == partition and item.task_key == task_key
            for item in contract.evaluation_items
        )
        == 2
        for partition in ("validation", "heldout")
        for task_key in evaluation_tasks
    )


def test_adr175_contract_accepts_exact_reset_only_sample_domain(tmp_path: Path) -> None:
    fixture = _broad_fixture(tmp_path)
    sample_plan = build_native_calvin_physical_sample_plan(
        fixture.dataset,
        comparison_id=fixture.stream_plan.comparison_id,
        seed=20260816,
        global_batch_size=2,
        total_steps=80,
        excluded_source_episode_indices=(
            fixture.representation_split.evaluation_source_episode_indices
        ),
    )
    sample_split = build_representation_trial_split_with_reference_evaluation(
        sample_plan,
        fixture.dataset,
        training_steps=sample_plan.total_steps,
        evaluation_reference=fixture.representation_split,
        require_equal_training_budget=True,
    )
    sample_evaluation_plan = build_entity_evaluation_plan(
        sample_split,
        CalvinStatefulTransitionDataset(fixture.dataset.index, action_horizon=1),
        world_size=2,
    )

    contract = build_adr175_broad_support_contract(
        dataset=fixture.dataset,
        stream_plan=sample_plan,
        representation_split=sample_split,
        entity_evaluation_plan=sample_evaluation_plan,
        training_prefix_steps=60,
    )

    assert len(contract.training_coverage) == 34
    assert all(item.visit_count > 0 for item in contract.training_coverage)
    assert contract.training_prefix_unique_source_episode_indices == tuple(
        sorted(fixture.representation_split.training_source_episode_indices)
    )


def test_adr175_ambiguous_tasks_never_emit_singleton_targets(tmp_path: Path) -> None:
    contract = _build(_broad_fixture(tmp_path))
    ambiguous = [item for item in contract.evaluation_items if not item.exact_action_target]

    assert ambiguous
    assert all(item.action_target_identity_keys == () for item in ambiguous)
    assert all(item.stratum.kind == ADR175_AMBIGUOUS_STRATUM for item in ambiguous)
    assert all(item.stratum.key[1] == ADR175_AMBIGUOUS_SET_ONLY for item in ambiguous)
    with pytest.raises(ValueError, match="singleton target"):
        replace(ambiguous[0], action_target_identity_keys=("movable/block_red",))


def test_adr175_same_plan_replays_identical_cross_arm_receipts(tmp_path: Path) -> None:
    fixture = _broad_fixture(tmp_path)
    lbot = _build(fixture)
    physical_set = _build(fixture)
    native_attention = _build(fixture)

    assert lbot == physical_set == native_attention
    assert {contract.stream_plan_sha256 for contract in (lbot, physical_set, native_attention)} == {
        fixture.stream_plan.plan_sha256
    }
    assert (
        len(
            {
                contract.matched_arm_input_sha256
                for contract in (lbot, physical_set, native_attention)
            }
        )
        == 1
    )
    assert (
        len(
            {
                contract.training_prefix_prompt_receipts_sha256
                for contract in (lbot, physical_set, native_attention)
            }
        )
        == 1
    )

    mismatched = replace(fixture.representation_split, stream_plan_sha256="0" * 64)
    with pytest.raises(ValueError, match="not bound"):
        build_adr175_broad_support_contract(
            dataset=fixture.dataset,
            stream_plan=fixture.stream_plan,
            representation_split=mismatched,
            entity_evaluation_plan=fixture.entity_evaluation_plan,
            training_prefix_steps=fixture.training_prefix_steps,
        )


def test_adr175_contract_binds_and_revalidates_the_entity_evaluation_plan(
    tmp_path: Path,
) -> None:
    fixture = _broad_fixture(tmp_path)
    contract = _build(fixture)
    first, *remaining = fixture.entity_evaluation_plan.items
    tampered = replace(
        fixture.entity_evaluation_plan,
        items=(replace(first, sample_key="undeclared-evaluation-sample"), *remaining),
    )

    assert contract.entity_evaluation_plan_artifact_sha256 == (
        fixture.entity_evaluation_plan.artifact_sha256
    )
    with pytest.raises(ValueError, match="sample key is absent"):
        build_adr175_broad_support_contract(
            dataset=fixture.dataset,
            stream_plan=fixture.stream_plan,
            representation_split=fixture.representation_split,
            entity_evaluation_plan=tampered,
            training_prefix_steps=fixture.training_prefix_steps,
        )


def test_adr175_prefix_covers_every_task_without_materializing_sensors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _broad_fixture(tmp_path)

    def reject_materialization(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ADR-175 contract attempted to materialize sensor or sidecar data")

    monkeypatch.setattr(fixture.dataset, "by_key", reject_materialization)
    monkeypatch.setattr(fixture.dataset, "evidence_prefix_by_key", reject_materialization)
    monkeypatch.setattr(fixture.dataset.index, "physical_sample", reject_materialization)
    contract = _build(fixture)

    assert len(contract.training_coverage) == 34
    assert all(item.visit_count > 0 for item in contract.training_coverage)
    assert sum(item.visit_count for item in contract.training_coverage) == (
        fixture.training_prefix_steps * fixture.stream_plan.global_batch_size
    )
    assert contract.training_prefix_sample_count == (
        fixture.training_prefix_steps * fixture.stream_plan.global_batch_size
    )
    assert set(contract.training_prefix_unique_source_episode_indices) == {
        source
        for item in contract.training_coverage
        for source in item.unique_source_episode_indices
    }


def test_adr175_contract_cli_verifies_all_content_pins_and_publishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fixture = _broad_fixture(tmp_path)
    output = tmp_path / "cli-contract.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(contract_tool.__file__),
            "--dataset-split",
            str(fixture.split_root),
            "--dataset-id",
            fixture.manifest.dataset_id,
            "--dataset-revision",
            fixture.manifest.dataset_revision,
            "--dataset-tree-sha256",
            fixture.manifest.tree_sha256,
            "--dataset-manifest",
            str(fixture.manifest_path),
            "--dataset-manifest-file-sha256",
            file_sha256(fixture.manifest_path),
            "--stream-plan",
            str(fixture.stream_plan_path),
            "--stream-plan-file-sha256",
            file_sha256(fixture.stream_plan_path),
            "--stream-plan-sha256",
            fixture.stream_plan.plan_sha256,
            "--representation-split",
            str(fixture.representation_split_path),
            "--representation-split-file-sha256",
            file_sha256(fixture.representation_split_path),
            "--representation-split-artifact-sha256",
            fixture.representation_split.artifact_sha256,
            "--entity-evaluation-plan",
            str(fixture.entity_evaluation_plan_path),
            "--entity-evaluation-plan-file-sha256",
            file_sha256(fixture.entity_evaluation_plan_path),
            "--entity-evaluation-plan-artifact-sha256",
            fixture.entity_evaluation_plan.artifact_sha256,
            "--training-prefix-steps",
            str(fixture.training_prefix_steps),
            "--output",
            str(output),
        ],
    )

    contract_tool.main()

    report = json.loads(capsys.readouterr().out)
    contract = Adr175BroadSupportContract.load(output)
    assert report["artifact_sha256"] == contract.artifact_sha256
    assert report["matched_arm_input_sha256"] == contract.matched_arm_input_sha256
    assert contract.exact_task_count == 29
    assert contract.ambiguous_task_count == 5

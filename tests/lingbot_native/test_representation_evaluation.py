from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinEpisode,
    CalvinLanguageSegment,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.dataset_manifest import build_dataset_file_manifest
from picf_next.lingbot_native.calvin import build_native_calvin_stream_plan
from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.representation_baseline import (
    build_representation_baseline_replay_report,
    build_representation_evaluation_baseline,
    validate_representation_baseline_plan,
    validate_representation_evaluation_baseline,
)
from picf_next.lingbot_native.representation_evaluation import (
    REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA,
    REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA,
    RepresentationEvaluationItem,
    RepresentationEvaluationPlan,
    _target_disjoint_donor_rotations,
    build_representation_evaluation_plan,
    build_representation_evaluation_sample,
    build_representation_evaluation_snapshot,
    build_representation_ownership_row,
    build_representation_token_evidence,
    build_representation_warm_evaluation_plan,
    representation_target_mass_sha256,
    summarize_representation_ownership_rows,
    validate_representation_evaluation_partition,
    validate_representation_evaluation_sample,
    validate_representation_evaluation_snapshot,
    validate_representation_evaluation_visual_files,
    validate_representation_ownership_row,
    validate_representation_ownership_summary,
    validate_representation_token_evidence,
)
from picf_next.lingbot_native.representation_gate import (
    build_representation_numeric_gate,
    validate_representation_numeric_gate,
    write_representation_numeric_gate,
)
from picf_next.lingbot_native.representation_split import (
    RepresentationTrialSplit,
    build_representation_trial_split,
)
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.lingbot_native.task_diagnostics import build_task_row_diagnostics
from picf_next.objective import UnifiedObjective


def _dataset(tmp_path: Path) -> CalvinStatefulTransitionDataset:
    split_root = tmp_path / "training"
    split_root.mkdir()
    (split_root / "manifest-stub").write_bytes(b"representation-evaluation-test")
    manifest = build_dataset_file_manifest(
        split_root,
        dataset_id="calvin-representation-evaluation-test",
        dataset_revision="sha256:representation-evaluation-test",
        split_name=split_root.name,
        relative_paths=("manifest-stub",),
    )
    episodes: list[CalvinEpisode] = []
    segments: list[CalvinLanguageSegment] = []
    tasks = (
        ("task-a", "move object a"),
        ("task-b", "move object b"),
        ("task-c", "move object c"),
    )
    for episode_index in range(60):
        episode_start = episode_index * 30
        episodes.append(CalvinEpisode(episode_index, episode_start, episode_start + 29))
        for task_offset, (task_key, annotation) in enumerate(tasks):
            start = episode_start + task_offset * 10
            segments.append(
                CalvinLanguageSegment(
                    len(segments),
                    start,
                    start + 9,
                    task_key,
                    annotation,
                    episode_index,
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


def _split(
    tmp_path: Path,
) -> tuple[CalvinStatefulTransitionDataset, RepresentationTrialSplit]:
    dataset = _dataset(tmp_path)
    stream = build_native_calvin_stream_plan(
        dataset,
        comparison_id="representation-evaluation-test",
        seed=71,
        global_batch_size=2,
        total_steps=8,
    )
    split = build_representation_trial_split(
        stream,
        dataset,
        training_steps=4,
        partition_seed=19,
        segments_per_task=2,
    )
    return dataset, split


def _task_identity_resolver(task_key: str) -> tuple[str, ...]:
    return (f"physical/{task_key}",)


def _plan(
    split: RepresentationTrialSplit,
    dataset: CalvinStatefulTransitionDataset,
) -> RepresentationEvaluationPlan:
    return build_representation_evaluation_plan(
        split,
        dataset,
        task_identity_resolver=_task_identity_resolver,
    )


def test_warm_evaluation_plan_is_one_age_eight_sample_per_task_and_roundtrips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, split = _split(tmp_path)

    def reject_decode(_sample_key: str) -> None:
        raise AssertionError("warm evaluation planning decoded a model sample")

    monkeypatch.setattr(dataset, "by_key", reject_decode)
    plan = build_representation_warm_evaluation_plan(
        split,
        dataset,
        task_identity_resolver=_task_identity_resolver,
    )

    assert plan.schema == REPRESENTATION_WARM_EVALUATION_PLAN_SCHEMA
    assert plan.history_transitions == 8
    assert len(plan.items) == 6
    for partition in ("validation", "heldout"):
        items = tuple(item for item in plan.items if item.partition == partition)
        assert len(items) == 3
        assert len({item.task_key for item in items}) == 3
        assert all(
            dataset.episode_manifest[item.segment_index].sample_keys[8] == item.sample_key
            for item in items
        )
        assert all(
            dataset.locator_by_key(item.sample_key).global_index == item.source_global_index
            for item in items
        )

    output = tmp_path / "warm-evaluation-plan.json"
    plan.write(output)
    assert RepresentationEvaluationPlan.load(output) == plan


@pytest.mark.parametrize("history_transitions", (1, 2, 4, 8))
def test_warm_evaluation_plan_supports_preregistered_horizon_sweep(
    tmp_path: Path,
    history_transitions: int,
) -> None:
    dataset, split = _split(tmp_path)
    plan = build_representation_warm_evaluation_plan(
        split,
        dataset,
        task_identity_resolver=_task_identity_resolver,
        history_transitions=history_transitions,
    )

    assert plan.history_transitions == history_transitions
    assert all(
        dataset.episode_manifest[item.segment_index].sample_keys[history_transitions]
        == item.sample_key
        for item in plan.items
    )


def _task_row_diagnostic() -> dict[str, object]:
    support = torch.zeros(1, 1, 2, 2)
    predictions = NativeSequencePredictions(
        support_logits=support,
        ownership=torch.softmax(
            torch.cat((support, torch.zeros(1, 1, 2, 1)), dim=-1),
            dim=-1,
        ),
        existence_logits=torch.zeros(1, 1, 2),
        task_relevance_logits=torch.tensor([[-0.5, 0.5]]),
        dense_task_grounding_logits=torch.zeros(1, 1, 2),
    )
    targets = NativeSequenceTargets(
        masks=torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]]),
        mask_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        existence=torch.ones(1, 1, 2),
        existence_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        task_relevance=torch.tensor([[1.0, 0.0]]),
        task_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 2),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    objective = NativeCALVINObjectiveResult(
        objective=UnifiedObjective(
            total=torch.zeros(()),
            normalized_terms={},
            valid_counts={},
        ),
        predictions=predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.tensor([[1, 0]])),
        track_identity_keys_by_batch=(("target", "context"),),
        row_bindings_by_batch=((("context", 0), ("target", 1)),),
        predictive_terms=(),
        structural_terms=(),
    )
    return build_task_row_diagnostics(objective)[0]


def _sample_evidence(
    item: RepresentationEvaluationItem,
    *,
    factual_instruction: str,
    shuffled_instruction: str,
    shuffled_target_instruction: str,
    checkpoint_global_step: int = 0,
    visual_root: Path | None = None,
    factual_logits: tuple[float, ...] = (0.9, 0.2, -0.1),
    shuffled_task_logits: tuple[float, ...] = (0.1, 0.4, 0.2),
    shuffled_target_mass: tuple[float, ...] = (0.0, 1.0, 0.0),
    target_prediction: tuple[float, ...] = (0.8, 0.2, 0.1),
    official_action_loss: float = 0.2,
) -> dict[str, object]:
    target_row = build_representation_ownership_row(
        row_index=0,
        track_index=0,
        identity_key="target",
        is_task_target=True,
        prediction=target_prediction,
        target=(1.0, 0.0, 0.0),
        weight=(1.0, 1.0, 1.0),
    )
    context_row = build_representation_ownership_row(
        row_index=1,
        track_index=1,
        identity_key="context",
        is_task_target=False,
        prediction=(0.1, 0.8, 0.2),
        target=(0.0, 1.0, 0.0),
        weight=(1.0, 1.0, 1.0),
    )
    rows = (target_row, context_row)
    summary = summarize_representation_ownership_rows(rows)
    relative_visual = Path("visuals") / item.partition / f"{item.ordinal}.png"
    visual_bytes = None
    visual_sha256 = "4" * 64
    visual_size = 128
    if visual_root is not None:
        visual_bytes = (
            f"{item.partition}\0{item.ordinal}\0{item.sample_key}\0{factual_instruction}".encode()
        )
        path = visual_root / relative_visual
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(visual_bytes)
        visual_sha256 = hashlib.sha256(visual_bytes).hexdigest()
        visual_size = len(visual_bytes)
    visual = {
        "schema": "picf-next.lingbot-native-relation-visual.v5",
        "path": relative_visual.as_posix(),
        "sha256": visual_sha256,
        "bytes": visual_size,
        "global_step": checkpoint_global_step,
        "input_weight_global_step": checkpoint_global_step,
        "weight_boundary": "checkpoint_evaluation",
        "rank": item.rank,
        "sample_key": item.sample_key,
        "task": factual_instruction,
        "loss_only_labels_visible_to_model": False,
    }
    return build_representation_evaluation_sample(
        checkpoint_global_step=checkpoint_global_step,
        item=item,
        factual_task_instruction_sha256=hashlib.sha256(factual_instruction.encode()).hexdigest(),
        shuffled_task_instruction_sha256=hashlib.sha256(shuffled_instruction.encode()).hexdigest(),
        shuffled_target_instruction_sha256=hashlib.sha256(
            shuffled_target_instruction.encode()
        ).hexdigest(),
        factual_token_evidence=build_representation_token_evidence(
            logits=factual_logits,
            target_mass=(1.0, 0.0, 0.0),
        ),
        shuffled_task_token_evidence=build_representation_token_evidence(
            logits=shuffled_task_logits,
            target_mass=(1.0, 0.0, 0.0),
        ),
        shuffled_target_token_evidence=build_representation_token_evidence(
            logits=factual_logits,
            target_mass=shuffled_target_mass,
        ),
        factual_task_row_diagnostic=_task_row_diagnostic(),
        shuffled_task_row_diagnostic=_task_row_diagnostic(),
        factual_ownership_rows=rows,
        factual_ownership_summary=summary,
        shuffled_task_ownership_rows=rows,
        shuffled_task_ownership_summary=summary,
        official_action_loss=official_action_loss,
        factual_forward_seconds=1.2,
        shuffled_task_forward_seconds=0.8,
        peak_cuda_reserved_bytes=1024,
        factual_relation_sha256="1" * 64,
        factual_target_sha256="2" * 64,
        shuffled_task_relation_sha256="3" * 64,
        shuffled_task_target_sha256="2" * 64,
        shuffled_target_target_sha256=representation_target_mass_sha256(
            item.shuffled_target_target_identity_keys,
            shuffled_target_mass,
        ),
        visual_artifact=visual,
    )


def test_representation_evaluation_plan_is_source_only_balanced_and_roundtrips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, split = _split(tmp_path)

    def reject_decode(_sample_key: str) -> None:
        raise AssertionError("evaluation planning decoded a model sample")

    monkeypatch.setattr(dataset, "by_key", reject_decode)
    plan = _plan(split, dataset)
    assert len(plan.items) == 12
    assert plan.representation_split_sha256 == split.artifact_sha256

    for partition in ("validation", "heldout"):
        items = tuple(item for item in plan.items if item.partition == partition)
        assert len(items) == 6
        assert [len(plan.items_for(partition, rank)) for rank in range(2)] == [3, 3]
        factual = {item.sample_key for item in items}
        assert {item.shuffled_task_sample_key for item in items} == factual
        assert {item.shuffled_target_sample_key for item in items} == factual
        for item in items:
            locator = dataset.locator_by_key(item.sample_key)
            assert locator.global_index == item.source_global_index
            assert locator.global_index == dataset.index.segments[locator.segment_index].start
            assert dataset.task_key_by_key(item.shuffled_task_sample_key) != item.task_key
            assert dataset.task_key_by_key(item.shuffled_target_sample_key) != item.task_key
            assert set(item.factual_target_identity_keys).isdisjoint(
                item.shuffled_task_target_identity_keys
            )
            assert set(item.factual_target_identity_keys).isdisjoint(
                item.shuffled_target_target_identity_keys
            )

    artifact = tmp_path / "representation-evaluation-plan.json"
    plan.write(artifact)
    assert RepresentationEvaluationPlan.load(artifact) == plan
    assert RepresentationEvaluationPlan.from_dict(plan.as_dict()) == plan
    with pytest.raises(FileExistsError, match="plan path exists"):
        plan.write(artifact)

    reference_derived = build_representation_evaluation_plan(
        split,
        dataset,
        task_identity_resolver=_task_identity_resolver,
        evaluation_reference_plan_sha256=plan.artifact_sha256,
    )
    assert reference_derived.schema == REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA
    assert reference_derived.items == plan.items
    assert reference_derived.artifact_sha256 != plan.artifact_sha256
    assert reference_derived.replay_seed_sha256 == plan.artifact_sha256
    assert RepresentationEvaluationPlan.from_dict(reference_derived.as_dict()) == reference_derived


def test_representation_evaluation_plan_rejects_identity_and_artifact_tamper(
    tmp_path: Path,
) -> None:
    dataset, split = _split(tmp_path)
    plan = _plan(split, dataset)

    payload = plan.as_dict()
    payload["artifact_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="artifact SHA-256 changed"):
        RepresentationEvaluationPlan.from_dict(payload)

    changed_split = replace(
        split,
        dataset_manifest_sha256="1" * 64,
    )
    with pytest.raises(ValueError, match="dataset differs"):
        _plan(changed_split, dataset)
    with pytest.raises(ValueError, match="dataset differs"):
        _plan(
            replace(split, dataset_revision="another-revision"),
            dataset,
        )

    items = list(plan.items)
    first = items[0]
    with pytest.raises(ValueError, match="retained its factual sample"):
        replace(first, shuffled_task_sample_key=first.sample_key)
    with pytest.raises(ValueError, match="retained a factual target identity"):
        replace(
            first,
            shuffled_task_target_identity_keys=first.factual_target_identity_keys,
        )

    same_task = next(
        item
        for item in plan.items
        if item.partition == first.partition
        and item.task_key == first.task_key
        and item.sample_key != first.sample_key
    )
    donor_consumer_index = next(
        index
        for index, item in enumerate(items)
        if item.shuffled_task_sample_key == same_task.sample_key
    )
    donor_consumer = items[donor_consumer_index]
    items[0] = replace(first, shuffled_task_sample_key=same_task.sample_key)
    items[donor_consumer_index] = replace(
        donor_consumer,
        shuffled_task_sample_key=first.shuffled_task_sample_key,
    )
    with pytest.raises(ValueError, match="retained its factual task"):
        replace(plan, items=tuple(items))


def test_representation_control_donors_reject_same_physical_target() -> None:
    tasks = (
        "blue-lift-drawer",
        "blue-lift-slider",
        "pink-lift-drawer",
        "pink-lift-slider",
        "red-lift-drawer",
        "red-lift-slider",
    )
    identities = {task: (f"physical/{task.split('-', maxsplit=1)[0]}-block",) for task in tasks}
    task_control, target_control = _target_disjoint_donor_rotations(tasks, identities)

    assert set(task_control) == set(tasks)
    assert set(task_control.values()) == set(tasks)
    assert set(target_control) == set(tasks)
    assert set(target_control.values()) == set(tasks)
    for task in tasks:
        assert task_control[task] != target_control[task]
        assert set(identities[task]).isdisjoint(identities[task_control[task]])
        assert set(identities[task]).isdisjoint(identities[target_control[task]])

    impossible = {
        "blue-a": ("physical/blue-block",),
        "blue-b": ("physical/blue-block",),
        "red-a": ("physical/red-block",),
    }
    with pytest.raises(ValueError, match="cannot form two disjoint target controls"):
        _target_disjoint_donor_rotations(tuple(impossible), impossible)


def test_representation_token_evidence_recomputes_fractional_metrics() -> None:
    evidence = build_representation_token_evidence(
        logits=(0.9, 0.4, -0.1, -0.3),
        target_mass=(0.75, 0.25, 0.0, 0.0),
    )
    assert evidence["metrics"]["eligible"] is True
    assert evidence["metrics"]["fractional_weighted_auc"] > 0.5
    assert validate_representation_token_evidence(copy.deepcopy(evidence)) == evidence

    tampered = copy.deepcopy(evidence)
    tampered["metrics"]["target_background_logit_margin"] = 100.0
    with pytest.raises(ValueError, match="were not recomputed"):
        validate_representation_token_evidence(tampered)


def test_representation_ownership_rows_and_summary_recompute_raw_evidence() -> None:
    target_row = build_representation_ownership_row(
        row_index=0,
        track_index=3,
        identity_key="blue-block",
        is_task_target=True,
        prediction=(0.8, 0.2),
        target=(1.0, 0.0),
        weight=(1.0, 2.0),
    )
    context_row = build_representation_ownership_row(
        row_index=1,
        track_index=4,
        identity_key="drawer",
        is_task_target=False,
        prediction=(0.1, 0.9),
        target=(0.0, 1.0),
        weight=(1.0, 1.0),
    )
    assert target_row["intersection"] == pytest.approx(0.8)
    assert target_row["union"] == pytest.approx(1.4)
    assert target_row["soft_iou"] == pytest.approx(0.8 / 1.4)
    assert target_row["target_mass_concentration"] == pytest.approx(0.8 / 1.2)
    assert validate_representation_ownership_row(copy.deepcopy(target_row)) == target_row

    summary = summarize_representation_ownership_rows((target_row, context_row))
    assert summary["row_count"] == 2
    assert summary["task_target_row_count"] == 1
    assert summary["target_soft_iou"] == target_row["soft_iou"]
    assert summary["target_mass_concentration"] == target_row["target_mass_concentration"]
    assert (
        validate_representation_ownership_summary(
            copy.deepcopy(summary),
            rows=(target_row, context_row),
        )
        == summary
    )

    tampered_row = copy.deepcopy(target_row)
    tampered_row["soft_iou"] = 0.99
    with pytest.raises(ValueError, match="were not recomputed"):
        validate_representation_ownership_row(tampered_row)

    tampered_summary = copy.deepcopy(summary)
    tampered_summary["macro_soft_iou"] = 0.99
    with pytest.raises(ValueError, match="was not recomputed"):
        validate_representation_ownership_summary(
            tampered_summary,
            rows=(target_row, context_row),
        )


def test_representation_ownership_summary_handles_no_task_target() -> None:
    row = build_representation_ownership_row(
        row_index=0,
        track_index=1,
        identity_key="context",
        is_task_target=False,
        prediction=(0.2, 0.8),
        target=(0.0, 1.0),
        weight=(1.0, 1.0),
    )
    summary = summarize_representation_ownership_rows((row,))
    assert summary["task_target_row_count"] == 0
    assert summary["target_soft_iou"] is None
    assert summary["target_mass_concentration"] is None


def test_representation_evaluation_sample_recomputes_all_nested_evidence() -> None:
    item = RepresentationEvaluationItem(
        partition="heldout",
        ordinal=0,
        rank=0,
        task_key="task-a",
        segment_index=1,
        source_episode_index=2,
        source_global_index=30,
        sample_key="sample-a",
        shuffled_task_sample_key="sample-b",
        shuffled_target_sample_key="sample-c",
        factual_target_identity_keys=("physical/task-a",),
        shuffled_task_target_identity_keys=("physical/task-b",),
        shuffled_target_target_identity_keys=("physical/task-c",),
        factual_task_instruction_sha256=hashlib.sha256(b"move object a").hexdigest(),
        shuffled_task_instruction_sha256=hashlib.sha256(b"move object b").hexdigest(),
        shuffled_target_instruction_sha256=hashlib.sha256(b"move object c").hexdigest(),
    )
    sample = _sample_evidence(
        item,
        factual_instruction="move object a",
        shuffled_instruction="move object b",
        shuffled_target_instruction="move object c",
    )
    assert (
        validate_representation_evaluation_sample(
            copy.deepcopy(sample),
            expected_item=item,
        )
        == sample
    )

    tampered = copy.deepcopy(sample)
    tampered["factual_token_evidence"]["metrics"]["fractional_weighted_auc"] = 0.0
    with pytest.raises(ValueError, match="were not recomputed"):
        validate_representation_evaluation_sample(tampered, expected_item=item)

    wrong_visual = copy.deepcopy(sample)
    wrong_visual["visual_artifact"]["input_weight_global_step"] = 1
    with pytest.raises(ValueError, match="visual provenance changed"):
        validate_representation_evaluation_sample(wrong_visual, expected_item=item)


def test_representation_evaluation_snapshot_is_complete_task_macro_and_tamper_proof(
    tmp_path: Path,
) -> None:
    dataset, split = _split(tmp_path)
    plan = _plan(split, dataset)
    task_by_sample = {item.sample_key: item.task_key for item in plan.items}
    instruction_by_task = {
        "task-a": "move object a",
        "task-b": "move object b",
        "task-c": "move object c",
    }
    samples = [
        _sample_evidence(
            item,
            factual_instruction=instruction_by_task[item.task_key],
            shuffled_instruction=instruction_by_task[task_by_sample[item.shuffled_task_sample_key]],
            shuffled_target_instruction=instruction_by_task[
                task_by_sample[item.shuffled_target_sample_key]
            ],
            visual_root=tmp_path,
        )
        for item in plan.items
    ]
    snapshot = build_representation_evaluation_snapshot(
        checkpoint_global_step=0,
        implementation_sha256="5" * 64,
        model_family_sha256="6" * 64,
        representation_split_sha256=split.artifact_sha256,
        representation_evaluation_plan=plan,
        representation_frozen_action_state_sha256="7" * 64,
        samples=samples,
    )
    assert (
        validate_representation_evaluation_snapshot(
            copy.deepcopy(snapshot),
            plan=plan,
        )
        == snapshot
    )
    visual_paths = validate_representation_evaluation_visual_files(
        snapshot,
        plan=plan,
        output_root=tmp_path,
    )
    assert len(visual_paths) == len(plan.items)
    for partition in ("validation", "heldout"):
        summary = snapshot["partition_summaries"][partition]
        assert summary["sample_count"] == 6
        assert summary["task_count"] == 3
        assert summary["token_eligible_task_count"] == 3
        assert summary["control_eligible_task_count"] == 3
        assert summary["rank_one_task_count"] == 3
        assert summary["rank_one_task_fraction"] == 1.0
        assert summary["mean_task_shuffled_task_auc_degradation"] > 0
        assert summary["mean_task_shuffled_target_auc_degradation"] > 0
        assert (
            validate_representation_evaluation_partition(
                copy.deepcopy(summary),
                samples=[sample for sample in samples if sample["partition"] == partition],
                partition=partition,
            )
            == summary
        )

    tampered = copy.deepcopy(snapshot)
    tampered["partition_summaries"]["heldout"]["mean_task_fractional_weighted_auc"] = 0.0
    with pytest.raises(ValueError, match="summaries were not recomputed"):
        validate_representation_evaluation_snapshot(tampered, plan=plan)

    changed_artifact = copy.deepcopy(snapshot)
    changed_artifact["artifact_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="artifact SHA-256 changed"):
        validate_representation_evaluation_snapshot(changed_artifact, plan=plan)

    visual_paths[0].write_bytes(b"tampered")
    with pytest.raises(ValueError, match="bytes differ"):
        validate_representation_evaluation_visual_files(
            snapshot,
            plan=plan,
            output_root=tmp_path,
        )


def test_reference_derived_step_zero_must_exactly_replay_historical_baseline(
    tmp_path: Path,
) -> None:
    dataset, split = _split(tmp_path)
    source_plan = _plan(split, dataset)
    task_by_sample = {item.sample_key: item.task_key for item in source_plan.items}
    instruction_by_task = {
        "task-a": "move object a",
        "task-b": "move object b",
        "task-c": "move object c",
    }
    samples = [
        _sample_evidence(
            item,
            factual_instruction=instruction_by_task[item.task_key],
            shuffled_instruction=instruction_by_task[task_by_sample[item.shuffled_task_sample_key]],
            shuffled_target_instruction=instruction_by_task[
                task_by_sample[item.shuffled_target_sample_key]
            ],
            visual_root=tmp_path,
        )
        for item in source_plan.items
    ]
    source_snapshot = build_representation_evaluation_snapshot(
        checkpoint_global_step=0,
        implementation_sha256="5" * 64,
        model_family_sha256="6" * 64,
        representation_split_sha256=split.artifact_sha256,
        representation_evaluation_plan=source_plan,
        representation_frozen_action_state_sha256="7" * 64,
        samples=samples,
    )
    baseline = build_representation_evaluation_baseline(
        source_snapshot=source_snapshot,
        source_snapshot_file_sha256="8" * 64,
        source_evaluation_plan=source_plan,
        source_evaluation_plan_file_sha256="9" * 64,
        source_visual_root=tmp_path,
    )
    assert validate_representation_evaluation_baseline(copy.deepcopy(baseline)) == baseline
    assert baseline["source_replay_seed_sha256"] == source_plan.artifact_sha256

    candidate_plan = replace(
        source_plan,
        representation_split_sha256="a" * 64,
        schema=REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA,
        evaluation_reference_plan_sha256=source_plan.artifact_sha256,
    )

    def candidate_snapshot(
        candidate_samples: list[dict[str, object]],
        *,
        model_family_sha256: str = "c" * 64,
    ) -> dict[str, object]:
        return build_representation_evaluation_snapshot(
            checkpoint_global_step=0,
            implementation_sha256="b" * 64,
            model_family_sha256=model_family_sha256,
            representation_split_sha256="a" * 64,
            representation_evaluation_plan=candidate_plan,
            representation_frozen_action_state_sha256="7" * 64,
            samples=candidate_samples,
        )

    exact_plan_snapshot = candidate_snapshot(samples)
    exact_plan_baseline = build_representation_evaluation_baseline(
        source_snapshot=exact_plan_snapshot,
        source_snapshot_file_sha256="d" * 64,
        source_evaluation_plan=candidate_plan,
        source_evaluation_plan_file_sha256="e" * 64,
        source_visual_root=tmp_path,
    )
    exact_plan_report = build_representation_baseline_replay_report(
        baseline=exact_plan_baseline,
        candidate_snapshot=exact_plan_snapshot,
        candidate_plan=candidate_plan,
        candidate_visual_root=tmp_path,
    )
    assert exact_plan_report["status"] == "PASS"
    assert exact_plan_report["replay_seed_sha256"] == source_plan.artifact_sha256

    report = build_representation_baseline_replay_report(
        baseline=baseline,
        candidate_snapshot=candidate_snapshot(samples),
        candidate_plan=candidate_plan,
        candidate_visual_root=tmp_path,
    )
    assert report["status"] == "PASS"
    assert report["sample_count"] == len(samples)
    assert report["replay_seed_sha256"] == source_plan.artifact_sha256
    assert report["source_model_family_sha256"] == "6" * 64
    assert report["candidate_model_family_sha256"] == "c" * 64

    timing_changed = copy.deepcopy(samples)
    timing_changed[0]["forward_seconds"]["factual"] = 9.0
    timing_changed[0]["peak_cuda_reserved_bytes"] = 2048
    build_representation_baseline_replay_report(
        baseline=baseline,
        candidate_snapshot=candidate_snapshot(timing_changed),
        candidate_plan=candidate_plan,
        candidate_visual_root=tmp_path,
    )

    evidence_changed = copy.deepcopy(samples)
    evidence_changed[0]["official_action_loss"] = 0.3
    with pytest.raises(ValueError, match="changed deterministic evidence"):
        build_representation_baseline_replay_report(
            baseline=baseline,
            candidate_snapshot=candidate_snapshot(evidence_changed),
            candidate_plan=candidate_plan,
            candidate_visual_root=tmp_path,
        )

    changed_plan = replace(
        candidate_plan,
        representation_split_sha256="f" * 64,
        evaluation_reference_plan_sha256="f" * 64,
    )
    with pytest.raises(ValueError, match="changed the historical replay seed"):
        build_representation_baseline_replay_report(
            baseline=exact_plan_baseline,
            candidate_snapshot=exact_plan_snapshot,
            candidate_plan=changed_plan,
            candidate_visual_root=tmp_path,
        )

    transitive_source_plan = replace(
        source_plan,
        representation_split_sha256="1" * 64,
        schema=REPRESENTATION_REFERENCE_EVALUATION_PLAN_SCHEMA,
        evaluation_reference_plan_sha256=source_plan.artifact_sha256,
    )
    transitive_source_snapshot = build_representation_evaluation_snapshot(
        checkpoint_global_step=0,
        implementation_sha256="2" * 64,
        model_family_sha256="3" * 64,
        representation_split_sha256="1" * 64,
        representation_evaluation_plan=transitive_source_plan,
        representation_frozen_action_state_sha256="7" * 64,
        samples=samples,
    )
    transitive_baseline = build_representation_evaluation_baseline(
        source_snapshot=transitive_source_snapshot,
        source_snapshot_file_sha256="4" * 64,
        source_evaluation_plan=transitive_source_plan,
        source_evaluation_plan_file_sha256="5" * 64,
        source_visual_root=tmp_path,
    )
    assert (
        transitive_baseline["source_evaluation_plan_artifact_sha256"]
        == transitive_source_plan.artifact_sha256
    )
    assert transitive_baseline["source_replay_seed_sha256"] == source_plan.artifact_sha256
    transitive_candidate = replace(
        transitive_source_plan,
        representation_split_sha256="6" * 64,
    )
    validate_representation_baseline_plan(
        transitive_baseline,
        candidate_plan=transitive_candidate,
    )
    with pytest.raises(ValueError, match="changed the historical replay seed"):
        validate_representation_baseline_plan(
            transitive_baseline,
            candidate_plan=replace(
                transitive_candidate,
                evaluation_reference_plan_sha256=transitive_source_plan.artifact_sha256,
            ),
        )


def test_representation_numeric_gate_recomputes_preregistered_step_200_decision(
    tmp_path: Path,
) -> None:
    dataset, split = _split(tmp_path)
    plan = _plan(split, dataset)
    task_by_sample = {item.sample_key: item.task_key for item in plan.items}
    instruction_by_task = {
        "task-a": "move object a",
        "task-b": "move object b",
        "task-c": "move object c",
    }

    def samples(step: int, *, decision: bool) -> list[dict[str, object]]:
        return [
            _sample_evidence(
                item,
                factual_instruction=instruction_by_task[item.task_key],
                shuffled_instruction=instruction_by_task[
                    task_by_sample[item.shuffled_task_sample_key]
                ],
                shuffled_target_instruction=instruction_by_task[
                    task_by_sample[item.shuffled_target_sample_key]
                ],
                checkpoint_global_step=step,
                factual_logits=(1.0, 0.2, -0.5) if decision else (-1.0, 1.0, 0.5),
                shuffled_task_logits=(-1.0, 1.0, 0.5),
                target_prediction=(0.8, 0.1, 0.1) if decision else (0.01, 0.4, 0.4),
                official_action_loss=0.24 if decision else 0.2,
            )
            for item in plan.items
        ]

    baseline = build_representation_evaluation_snapshot(
        checkpoint_global_step=0,
        implementation_sha256="a" * 64,
        model_family_sha256="b" * 64,
        representation_split_sha256=split.artifact_sha256,
        representation_evaluation_plan=plan,
        representation_frozen_action_state_sha256="c" * 64,
        samples=samples(0, decision=False),
    )
    decision = build_representation_evaluation_snapshot(
        checkpoint_global_step=200,
        implementation_sha256="a" * 64,
        model_family_sha256="b" * 64,
        representation_split_sha256=split.artifact_sha256,
        representation_evaluation_plan=plan,
        representation_frozen_action_state_sha256="c" * 64,
        samples=samples(200, decision=True),
    )
    gate = build_representation_numeric_gate(baseline, decision, plan=plan)
    assert gate["status"] == "PASS_PENDING_VISUAL_REVIEW"
    assert gate["authorizes_joint_adoption"] is False
    assert all(check["passed"] for check in gate["checks"].values())
    assert (
        validate_representation_numeric_gate(
            copy.deepcopy(gate),
            baseline_snapshot=baseline,
            decision_snapshot=decision,
            plan=plan,
        )
        == gate
    )

    output = tmp_path / "representation-numeric-gate.json"
    write_representation_numeric_gate(output, gate)
    assert json.loads(output.read_text(encoding="ascii")) == gate
    with pytest.raises(FileExistsError, match="gate path exists"):
        write_representation_numeric_gate(output, gate)

    weak_control_samples = [
        _sample_evidence(
            item,
            factual_instruction=instruction_by_task[item.task_key],
            shuffled_instruction=instruction_by_task[task_by_sample[item.shuffled_task_sample_key]],
            shuffled_target_instruction=instruction_by_task[
                task_by_sample[item.shuffled_target_sample_key]
            ],
            checkpoint_global_step=200,
            factual_logits=(1.0, 0.2, -0.5),
            shuffled_task_logits=(
                (-1.0, 1.0, 0.5) if item.task_key == "task-a" else (1.0, 0.2, -0.5)
            ),
            target_prediction=(0.8, 0.1, 0.1),
            official_action_loss=0.24,
        )
        for item in plan.items
    ]
    weak_control_decision = build_representation_evaluation_snapshot(
        checkpoint_global_step=200,
        implementation_sha256="a" * 64,
        model_family_sha256="b" * 64,
        representation_split_sha256=split.artifact_sha256,
        representation_evaluation_plan=plan,
        representation_frozen_action_state_sha256="c" * 64,
        samples=weak_control_samples,
    )
    weak_control_gate = build_representation_numeric_gate(
        baseline,
        weak_control_decision,
        plan=plan,
    )
    assert weak_control_gate["checks"]["shuffled_task_auc_degradation"]["passed"] is True
    assert (
        weak_control_gate["checks"]["positive_shuffled_task_degradation_fraction"]["passed"]
        is False
    )
    assert weak_control_gate["status"] == "FAIL"

    failed_decision = copy.deepcopy(decision)
    failed_decision["representation_frozen_action_state_sha256"] = "d" * 64
    failed_payload = {
        name: failed_decision[name] for name in failed_decision if name != "artifact_sha256"
    }
    failed_decision["artifact_sha256"] = hashlib.sha256(
        json.dumps(
            failed_payload,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()
    failed_gate = build_representation_numeric_gate(
        baseline,
        failed_decision,
        plan=plan,
    )
    assert failed_gate["status"] == "FAIL"
    assert failed_gate["checks"]["frozen_action_state_unchanged"]["passed"] is False

    tampered = copy.deepcopy(gate)
    tampered["checks"]["mean_task_auc_delta"]["passed"] = False
    with pytest.raises(ValueError, match="was not recomputed"):
        validate_representation_numeric_gate(
            tampered,
            baseline_snapshot=baseline,
            decision_snapshot=decision,
            plan=plan,
        )

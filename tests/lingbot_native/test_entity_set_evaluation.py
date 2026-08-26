from __future__ import annotations

import itertools
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
from picf_next.lingbot_native.entity_evaluation_plan import (
    ENTITY_EVALUATION_PARTITIONS,
    EntityEvaluationPlan,
    build_distributed_causal_warm_evaluation_schedule,
    build_distributed_entity_evaluation_schedule,
    build_entity_evaluation_plan,
)
from picf_next.lingbot_native.entity_set_evaluation import (
    ENTITY_AREA_STRATA,
    evaluate_physical_entity_frame,
    maximum_token_grid_soft_iou,
    summarize_entity_evaluation_partition,
)
from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFrameAssignment,
    PhysicalFramePredictions,
    PhysicalFrameTargets,
)
from picf_next.lingbot_native.representation_split import build_representation_trial_split


def _predictions(logits: torch.Tensor) -> PhysicalFramePredictions:
    context = torch.full((*logits.shape[:2], 1), -6.0, dtype=logits.dtype)
    context[:, -1] = 8.0
    return PhysicalFramePredictions(
        support_logits=logits,
        ownership_log_probability=torch.log_softmax(torch.cat((logits, context), dim=-1), dim=-1),
        existence_logits=torch.tensor([[6.0, 6.0, -6.0]], dtype=logits.dtype),
        sensor_valid=torch.ones(logits.shape[:2], dtype=torch.bool),
    )


def _targets() -> PhysicalFrameTargets:
    masks = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]],
        dtype=torch.float32,
    )
    return PhysicalFrameTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.ones(1, 2),
        existence_valid=torch.ones(1, 2, dtype=torch.bool),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 4),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )


def test_entity_frame_metrics_reward_separated_rows_and_context() -> None:
    logits = torch.tensor(
        [[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0], [-8.0, 8.0, -8.0], [-8.0, -8.0, -8.0]]]
    )
    predictions = _predictions(logits)
    evidence = evaluate_physical_entity_frame(
        predictions,
        _targets(),
        PhysicalFrameAssignment(torch.tensor([[0, 1, -1]])),
        identity_keys=("first", "second"),
    )

    assert evidence["target_visible_count"] == 2
    assert evidence["target_evidence_count"] == 2
    assert evidence["matched_evidence_count"] == 2
    assert evidence["predicted_count_at_0_5"] == 2
    assert evidence["cardinality_absolute_error_at_0_5"] == 0
    assert float(evidence["context_region_probability"]) > 0.99
    assert float(evidence["object_ownership_target_recall"]) > 0.99
    assert float(evidence["mean_pairwise_support_overlap"]) < 0.002
    rows = evidence["rows"]
    assert isinstance(rows, list) and len(rows) == 2
    assert all(float(row["support_soft_iou"]) > 0.99 for row in rows)
    assert all(float(row["support_soft_iou_ceiling"]) == pytest.approx(1.0) for row in rows)
    assert all(float(row["support_soft_iou_efficiency"]) > 0.99 for row in rows)
    assert all(float(row["ownership_soft_iou"]) > 0.99 for row in rows)


def test_entity_frame_metrics_separate_existence_only_from_spatial_evidence() -> None:
    logits = torch.tensor(
        [[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0], [-8.0, -8.0, -8.0], [-8.0, -8.0, -8.0]]]
    )
    predictions = _predictions(logits)
    masks = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    targets = PhysicalFrameTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.ones(1, 2),
        existence_valid=torch.tensor([[False, True]]),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 4),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )

    evidence = evaluate_physical_entity_frame(
        predictions,
        targets,
        PhysicalFrameAssignment(torch.tensor([[0, 1, -1]])),
        identity_keys=("visible", "occluded"),
    )

    assert evidence["target_evidence_count"] == 2
    assert evidence["matched_evidence_count"] == 2
    assert evidence["target_visible_count"] == 1
    assert evidence["matched_count"] == 1
    assert evidence["cardinality_absolute_error_at_0_5"] == 0
    rows = evidence["rows"]
    assert isinstance(rows, list) and len(rows) == 1
    assert rows[0]["identity_key"] == "visible"


def test_entity_frame_metrics_allow_only_proven_carried_rows_without_current_evidence() -> None:
    predictions = _predictions(
        torch.tensor(
            [[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0], [-8.0, -8.0, -8.0], [-8.0, -8.0, -8.0]]]
        )
    )
    masks = torch.tensor(
        [[[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    targets = PhysicalFrameTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.tensor([[1.0, 0.0]]),
        existence_valid=torch.tensor([[True, False]]),
        track_valid=torch.ones(1, 2, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 4),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    assignment = PhysicalFrameAssignment(
        torch.tensor([[0, 1, -1]]),
        carried_rows=torch.tensor([[False, True, False]]),
    )

    evidence = evaluate_physical_entity_frame(
        predictions,
        targets,
        assignment,
        identity_keys=("visible", "temporally-carried"),
    )

    assert evidence["target_evidence_count"] == 1
    assert evidence["matched_evidence_count"] == 1
    assert evidence["matched_assignment_count"] == 2
    assert evidence["carried_unknown_count"] == 1
    assert evidence["matched_count"] == 1
    assert evidence["cardinality_supervision_complete"] is False
    assert evidence["cardinality_absolute_error_at_0_5"] is None
    assert [row["identity_key"] for row in evidence["rows"]] == ["visible"]

    with pytest.raises(RuntimeError, match="unproven current-frame physical track"):
        evaluate_physical_entity_frame(
            predictions,
            targets,
            PhysicalFrameAssignment(torch.tensor([[0, 1, -1]])),
            identity_keys=("visible", "not-proven-carried"),
        )


def test_entity_frame_metrics_exclude_reserved_unknown_rows_from_no_object_metric() -> None:
    logits = torch.tensor(
        [[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0], [-8.0, -8.0, -8.0], [-8.0, -8.0, -8.0]]]
    )
    predictions = PhysicalFramePredictions(
        support_logits=logits,
        ownership_log_probability=_predictions(logits).ownership_log_probability,
        existence_logits=torch.tensor([[6.0, 6.0, -6.0]]),
        sensor_valid=torch.ones(1, 4, dtype=torch.bool),
    )
    masks = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]])
    targets = PhysicalFrameTargets(
        masks=masks,
        mask_valid=torch.ones_like(masks, dtype=torch.bool),
        existence=torch.ones(1, 1),
        existence_valid=torch.ones(1, 1, dtype=torch.bool),
        track_valid=torch.ones(1, 1, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 4),
        inventory_exhaustive=torch.ones(1, dtype=torch.bool),
        exclusive_ownership=True,
    )

    evidence = evaluate_physical_entity_frame(
        predictions,
        targets,
        PhysicalFrameAssignment(
            torch.tensor([[0, -1, -1]]),
            reserved_rows=torch.tensor([[False, True, False]]),
        ),
        identity_keys=("visible",),
    )

    assert evidence["reserved_unknown_count"] == 1
    assert evidence["cardinality_supervision_complete"] is False
    assert evidence["cardinality_absolute_error_at_0_5"] is None
    assert float(evidence["mean_unmatched_existence_probability"]) < 0.01


def test_entity_frame_metrics_reject_wrong_assignment_set_with_equal_count() -> None:
    predictions = _predictions(torch.zeros(1, 4, 3))
    targets = _targets()

    with pytest.raises(RuntimeError, match="omitted an eligible physical track"):
        evaluate_physical_entity_frame(
            predictions,
            targets,
            PhysicalFrameAssignment(torch.tensor([[0, 2, -1]])),
            identity_keys=("first", "second"),
        )


def test_token_grid_soft_iou_ceiling_is_exact_for_fractional_target() -> None:
    target = torch.tensor([0.25, 0.0, 0.0])
    weight = torch.ones(3)

    assert float(maximum_token_grid_soft_iou(target, weight)) == pytest.approx(0.25)


def test_token_grid_soft_iou_ceiling_matches_exhaustive_vertices() -> None:
    target = torch.tensor([0.8, 0.45, 0.2, 0.05])
    weight = torch.tensor([1.0, 0.5, 2.0, 1.5])
    target_mass = (target * weight).sum()
    candidates = []
    for values in itertools.product((0.0, 1.0), repeat=target.numel()):
        prediction = torch.tensor(values)
        intersection = (prediction * target * weight).sum()
        union = target_mass + (prediction * (1 - target) * weight).sum()
        candidates.append(float(intersection / union))

    assert float(maximum_token_grid_soft_iou(target, weight)) == pytest.approx(max(candidates))


def test_token_grid_soft_iou_ceiling_rejects_absent_target() -> None:
    with pytest.raises(ValueError, match="positive target mass"):
        maximum_token_grid_soft_iou(torch.zeros(3), torch.ones(3))


def test_entity_frame_metrics_reject_identity_axis_mismatch() -> None:
    predictions = _predictions(torch.zeros(1, 4, 3))
    with pytest.raises(ValueError, match="identities differ"):
        evaluate_physical_entity_frame(
            predictions,
            _targets(),
            PhysicalFrameAssignment(torch.tensor([[0, 1, -1]])),
            identity_keys=("first",),
        )


def test_entity_partition_summary_preserves_empty_area_strata() -> None:
    predictions = _predictions(
        torch.tensor(
            [[[8.0, -8.0, -8.0], [8.0, -8.0, -8.0], [-8.0, 8.0, -8.0], [-8.0, -8.0, -8.0]]]
        )
    )
    sample = evaluate_physical_entity_frame(
        predictions,
        _targets(),
        PhysicalFrameAssignment(torch.tensor([[0, 1, -1]])),
        identity_keys=("first", "second"),
    )
    sample.update(partition="heldout", sample_key="sample", task_key="task")
    summary = summarize_entity_evaluation_partition((sample,), partition="heldout")

    assert summary["sample_count"] == 1
    assert summary["entity_count"] == 2
    assert set(summary["area_strata"]) == {item[0] for item in ENTITY_AREA_STRATA}
    assert summary["area_strata"]["lt_2_percent"]["entity_count"] == 0
    assert summary["area_strata"]["lt_2_percent"]["mean_support_soft_iou"] is None
    assert summary["mean_support_soft_iou_ceiling"] == pytest.approx(1.0)
    assert float(summary["mean_support_soft_iou_efficiency"]) > 0.99


def _dataset(tmp_path: Path) -> CalvinStatefulTransitionDataset:
    split_root = tmp_path / "training"
    split_root.mkdir()
    (split_root / "manifest-stub").write_bytes(b"entity-evaluation-test")
    manifest = build_dataset_file_manifest(
        split_root,
        dataset_id="calvin-entity-evaluation-test",
        dataset_revision="sha256:calvin-entity-evaluation-test",
        split_name=split_root.name,
        relative_paths=("manifest-stub",),
    )
    episodes: list[CalvinEpisode] = []
    segments: list[CalvinLanguageSegment] = []
    tasks = (("task-a", "do a"), ("task-b", "do b"), ("task-c", "do c"))
    for episode_index in range(60):
        episode_start = episode_index * 30
        episodes.append(CalvinEpisode(episode_index, episode_start, episode_start + 29))
        for task_offset, (task_key, instruction) in enumerate(tasks):
            start = episode_start + task_offset * 10
            segments.append(
                CalvinLanguageSegment(
                    len(segments),
                    start,
                    start + 9,
                    task_key,
                    instruction,
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


def test_entity_evaluation_plan_is_label_free_source_disjoint_and_roundtrips(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = _dataset(tmp_path)
    stream = build_native_calvin_stream_plan(
        dataset,
        comparison_id="entity-evaluation-test",
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

    def reject_decode(_sample_key: str) -> None:
        raise AssertionError("entity evaluation planning decoded a frame")

    monkeypatch.setattr(dataset, "by_key", reject_decode)
    plan = build_entity_evaluation_plan(split, dataset)

    assert len(plan.items) == 9
    for partition in ENTITY_EVALUATION_PARTITIONS:
        items = tuple(item for item in plan.items if item.partition == partition)
        assert len(items) == (3 if partition == "validation" else 6)
        assert len({item.task_key for item in items}) == 3
        for task_key in {item.task_key for item in items}:
            task_items = tuple(item for item in items if item.task_key == task_key)
            expected_count = 1 if partition == "validation" else 2
            assert len(task_items) == expected_count
            assert len({item.source_episode_index for item in task_items}) == expected_count
    validation_sources = {
        item.source_episode_index for item in plan.items if item.partition == "validation"
    }
    heldout_sources = {
        item.source_episode_index for item in plan.items if item.partition == "heldout"
    }
    assert validation_sources.isdisjoint(heldout_sources)
    assert validation_sources.isdisjoint(split.training_source_episode_indices)
    assert heldout_sources.isdisjoint(split.training_source_episode_indices)

    output = tmp_path / "entity-evaluation-plan.json"
    plan.write(output)
    assert EntityEvaluationPlan.load(output) == plan

    four_rank = build_entity_evaluation_plan(split, dataset, world_size=4)
    assert [item.sample_key for item in four_rank.items] == [item.sample_key for item in plan.items]
    assert [item.rank for item in four_rank.items] == [
        ordinal % 4 for ordinal in range(len(four_rank.items))
    ]
    assert four_rank.world_size == 4

    schedules = tuple(
        build_distributed_entity_evaluation_schedule(four_rank, rank=rank)
        for rank in range(4)
    )
    assert {len(schedule) for schedule in schedules} == {3}
    assert sorted(
        work.item.ordinal
        for schedule in schedules
        for work in schedule
        if not work.is_padding
    ) == list(range(len(four_rank.items)))
    assert [
        (rank, work.item.ordinal)
        for rank, schedule in enumerate(schedules)
        for work in schedule
        if work.is_padding
    ] == [(1, 0), (2, 1), (3, 2)]

    warm_schedules = tuple(
        build_distributed_causal_warm_evaluation_schedule(
            four_rank,
            rank=rank,
            history_transitions=4,
        )
        for rank in range(4)
    )
    assert len({len(schedule) for schedule in warm_schedules}) == 1
    warm_scientific = sorted(
        (
            work.item
            for schedule in warm_schedules
            for work in schedule
            if not work.is_padding
        ),
        key=lambda item: item.ordinal,
    )
    assert warm_scientific == [
        item for item in four_rank.items if item.transition_index >= 4
    ]
    assert all(
        work.item.transition_index >= 4
        for schedule in warm_schedules
        for work in schedule
    )


def test_causal_warm_schedule_rejects_fake_or_empty_history(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    stream = build_native_calvin_stream_plan(
        dataset,
        comparison_id="warm-evaluation-test",
        seed=73,
        global_batch_size=2,
        total_steps=8,
    )
    split = build_representation_trial_split(
        stream,
        dataset,
        training_steps=4,
        partition_seed=23,
        segments_per_task=2,
    )
    plan = build_entity_evaluation_plan(split, dataset, world_size=4)

    with pytest.raises(ValueError, match="positive integer"):
        build_distributed_causal_warm_evaluation_schedule(
            plan,
            rank=0,
            history_transitions=0,
        )
    with pytest.raises(ValueError, match="no eligible samples"):
        build_distributed_causal_warm_evaluation_schedule(
            plan,
            rank=0,
            history_transitions=10,
        )

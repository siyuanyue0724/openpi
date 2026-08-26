from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.adr175_validation import canonical_sha256
from tools import compare_adr175_evaluation_snapshots as comparison


def _partition_summary(*, action: float, entity: float) -> dict[str, object]:
    return {
        "action_loss": action,
        "conditional_selectivity": 0.2,
        "entity_set_score": entity,
        "entity_set_summary": {
            "area_strata": {
                key: {"mean_support_soft_iou_efficiency": entity}
                for key in ("lt_2_percent", "2_to_5_percent", "ge_5_percent")
            },
            "mean_cardinality_absolute_error_at_0_5": 1.0,
            "mean_context_region_probability": 0.8,
            "mean_existence_probability": 0.7,
            "mean_mean_pairwise_support_overlap": 0.1,
            "mean_object_ownership_target_recall": 0.5,
            "mean_ownership_soft_iou": 0.3,
            "mean_ownership_target_recall": 0.4,
            "mean_support_soft_iou": 0.2,
        },
        "posterior_adoption": 0.4,
    }


def _snapshot(path: Path, *, step: int) -> None:
    semantic = {
        "arm": "native-attention",
        "checkpoint_global_step": step,
        "entity_evaluation_plan_sha256": "1" * 64,
        "evaluation_input_sha256": "2" * 64,
        "implementation_sha256": "3" * 64,
        "model_family_sha256": "4" * 64,
        "partition_summaries": {
            partition: _partition_summary(action=1.0, entity=0.1)
            for partition in ("validation", "heldout")
        },
        "representation_split_sha256": "5" * 64,
        "samples": [],
        "status": "PASS",
        "stream_plan_sha256": "6" * 64,
    }
    path.write_text(
        json.dumps(
            {**semantic, "artifact_sha256": canonical_sha256(semantic)},
            sort_keys=True,
        ),
        encoding="ascii",
    )


def _sample(
    *,
    task: str,
    partition: str,
    replicate: int,
    action: float,
    entity: float,
    selectivity: float | None,
    target_valid: bool = True,
    target_observable: bool = True,
) -> dict[str, object]:
    identity_key = "part" if target_observable else "different/visible_entity"
    return {
        "task_key": task,
        "partition": partition,
        "sample_key": f"{partition}/{task}/{replicate}",
        "source_episode_index": (0 if partition == "validation" else 100) + replicate,
        "official_action_loss": action,
        "posterior_adoption": 0.5 if target_valid else None,
        "conditional_selectivity": selectivity,
        "target_valid": target_valid,
        "entity_evidence": {
            "rows": [
                {
                    "identity_key": identity_key,
                    "support_soft_iou_efficiency": entity,
                }
            ],
            "target_visible_count": 1,
        },
    }


def test_adr175_snapshot_comparison_rejects_an_unregistered_candidate_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = tmp_path / "step0.json"
    candidate = tmp_path / "step250.json"
    _snapshot(baseline, step=0)
    _snapshot(candidate, step=250)
    monkeypatch.setattr(
        comparison,
        "parse_args",
        lambda: argparse.Namespace(
            baseline=baseline,
            candidate=candidate,
            expected_baseline_step=0,
            expected_candidate_step=500,
            output=None,
        ),
    )

    with pytest.raises(ValueError, match="candidate milestone differs"):
        comparison.main()


def test_adr175_snapshot_joint_gate_requires_both_improve_but_ten_percent_once() -> None:
    baseline = _partition_summary(action=1.0, entity=0.1)
    validation = _partition_summary(action=1.0, entity=0.105)
    heldout = _partition_summary(action=1.0, entity=0.115)

    validation_gate = comparison.operational_gate(
        comparison.summary({"partition_summaries": {"validation": baseline}}, "validation"),
        comparison.summary({"partition_summaries": {"validation": validation}}, "validation"),
    )
    heldout_gate = comparison.operational_gate(
        comparison.summary({"partition_summaries": {"heldout": baseline}}, "heldout"),
        comparison.summary({"partition_summaries": {"heldout": heldout}}, "heldout"),
    )

    assert validation_gate["entity_improved"] is True
    assert validation_gate["entity_relative_10pct"] is False
    assert heldout_gate["entity_improved"] is True
    assert heldout_gate["entity_relative_10pct"] is True


def test_exact_attention_uses_task_partition_macros_not_heldout_replicates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "ADR175_EXACT_TASK_TARGETS", (("task", ("part",)),))
    baseline_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=1.0,
            entity=0.2,
            selectivity=0.5,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=1.0,
                entity=0.2,
                selectivity=0.2,
            )
            for replicate in range(2)
        ],
    ]
    candidate_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=0.9,
            entity=0.3,
            selectivity=0.4,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=0.9,
                entity=0.3,
                selectivity=0.3,
            )
            for replicate in range(2)
        ],
    ]
    baseline = {(index,): sample for index, sample in enumerate(baseline_samples)}
    candidate = {(index,): sample for index, sample in enumerate(candidate_samples)}

    result = comparison.exact_attention(baseline, candidate)

    assert result["partition_positive_count"] == 1
    assert result["tasks_positive_both_partitions"] == 0
    assert result["tasks_positive_one_partition"] == 1


def test_exact_attention_conservatively_censors_an_invisible_target_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "ADR175_EXACT_TASK_TARGETS", (("task", ("part",)),))
    baseline_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=1.0,
            entity=0.2,
            selectivity=None,
            target_valid=False,
            target_observable=False,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=1.0,
                entity=0.2,
                selectivity=0.1,
            )
            for replicate in range(2)
        ],
    ]
    candidate_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=0.9,
            entity=0.3,
            selectivity=None,
            target_valid=False,
            target_observable=False,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=0.9,
                entity=0.3,
                selectivity=0.2,
            )
            for replicate in range(2)
        ],
    ]
    baseline = {(index,): sample for index, sample in enumerate(baseline_samples)}
    candidate = {(index,): sample for index, sample in enumerate(candidate_samples)}

    result = comparison.exact_attention(baseline, candidate)

    assert result["censored_partition_count"] == 1
    assert result["valid_partition_count"] == 1
    assert result["tasks_positive_both_partitions"] == 0
    assert result["tasks_positive_one_partition"] == 1
    assert result["task_details"]["task"]["validation"]["censored"] is True


def test_exact_attention_scores_an_observable_but_unresolved_target_as_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(comparison, "ADR175_EXACT_TASK_TARGETS", (("task", ("part",)),))
    baseline_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=1.0,
            entity=0.2,
            selectivity=0.4,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=1.0,
                entity=0.2,
                selectivity=0.2,
            )
            for replicate in range(2)
        ],
    ]
    candidate_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=0.9,
            entity=0.3,
            selectivity=None,
            target_valid=False,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=0.9,
                entity=0.3,
                selectivity=0.3,
            )
            for replicate in range(2)
        ],
    ]
    baseline = {(index,): sample for index, sample in enumerate(baseline_samples)}
    candidate = {(index,): sample for index, sample in enumerate(candidate_samples)}

    result = comparison.exact_attention(baseline, candidate)

    validation = result["task_details"]["task"]["validation"]
    assert validation["censored"] is False
    assert validation["conditional_selectivity"]["candidate"] == 0.0
    assert validation["candidate_target_resolved_count"] == 0
    assert result["tasks_positive_both_partitions"] == 0
    assert result["tasks_positive_one_partition"] == 1


def test_sample_improvements_requires_both_partition_macros() -> None:
    baseline_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=0.5,
            entity=0.5,
            selectivity=0.2,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=1.0,
                entity=0.2,
                selectivity=0.2,
            )
            for replicate in range(2)
        ],
    ]
    candidate_samples = [
        _sample(
            task="task",
            partition="validation",
            replicate=0,
            action=0.6,
            entity=0.4,
            selectivity=0.2,
        ),
        *[
            _sample(
                task="task",
                partition="heldout",
                replicate=replicate,
                action=0.9,
                entity=0.3,
                selectivity=0.2,
            )
            for replicate in range(2)
        ],
    ]
    baseline = {(index,): sample for index, sample in enumerate(baseline_samples)}
    candidate = {(index,): sample for index, sample in enumerate(candidate_samples)}

    result = comparison.sample_improvements(baseline, candidate)

    assert result["action_sample_positive_count"] == 2
    assert result["entity_sample_positive_count"] == 2
    assert result["action_task_both_partitions_count"] == 0
    assert result["entity_task_both_partitions_count"] == 0

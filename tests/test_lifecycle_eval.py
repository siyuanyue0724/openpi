from __future__ import annotations

import pytest

from picf_next.eval.lifecycle import (
    audit_lifecycle_reports,
    audit_partitioned_visibility_target_sequences,
    audit_visibility_target_sequences,
    partition_contiguous_visibility_targets,
)


def test_visibility_target_timeline_breaks_at_source_gaps() -> None:
    targets = {
        10: {"object/a": 1},
        11: {"object/a": 0},
        14: {"object/a": 0},
    }

    sequences = partition_contiguous_visibility_targets((10, 11, 14), targets)

    assert sequences == ((targets[10], targets[11]), (targets[14],))
    census = audit_visibility_target_sequences(sequences)
    assert census["transition_count"]["1->0"] == 1
    assert census["transition_count"]["0->0"] == 0


def test_visibility_target_census_counts_only_adjacent_supervised_pairs() -> None:
    result = audit_visibility_target_sequences(
        (
            (
                {"object/a": 1, "object/b": 0},
                {"object/a": 1, "object/b": None},
                {"object/a": 0, "object/b": 1},
                {"object/a": 1, "object/b": 1},
            ),
            (
                {"object/a": 0},
                {"object/a": 0},
            ),
        )
    )

    assert result["sequence_count"] == 2
    assert result["frame_occurrence_count"] == 6
    assert result["label_count"] == {"0": 4, "1": 5}
    assert result["transition_count"] == {
        "0->0": 1,
        "0->1": 1,
        "1->0": 1,
        "1->1": 2,
    }
    assert result["adjacent_identity_pair_count"] == 7
    assert result["supervised_transition_pair_count"] == 5
    assert result["transition_count_by_identity"]["object/a"] == {
        "0->0": 1,
        "0->1": 1,
        "1->0": 1,
        "1->1": 1,
    }
    assert result["next_supervised_transition_count"] == {
        "0->0": 1,
        "0->1": 2,
        "1->0": 1,
        "1->1": 2,
    }
    assert result["bridged_transition_count"] == {
        "0->0": 0,
        "0->1": 1,
        "1->0": 0,
        "1->1": 0,
    }
    assert result["bridged_unknown_run_length"]["count"] == 1
    assert result["bridged_unknown_run_length"]["maximum"] == 1.0
    assert result["next_supervised_elapsed_steps_by_transition"]["0->1"]["maximum"] == 2.0
    assert result["hidden_run_length"]["count"] == 3
    assert result["hidden_run_length"]["maximum"] == 2.0
    hazard = result["hidden_reappearance_hazard"]
    assert hazard["exact_by_elapsed_hidden_frames"]["1"] == {
        "at_risk_count": 2,
        "reappearance_count": 1,
        "remained_hidden_count": 1,
        "reappearance_hazard": 0.5,
    }
    assert hazard["reacquired_run_length"]["count"] == 1
    assert hazard["right_censored_run_length"]["maximum"] == 2.0
    assert hazard["unknown_censored_run_length"]["maximum"] == 1.0
    seen = hazard["seen_then_hidden"]
    assert seen["exact_by_elapsed_hidden_frames"]["1"]["reappearance_hazard"] == 1.0
    assert seen["reacquired_run_length"]["maximum"] == 1.0
    assert seen["right_censored_run_length"]["count"] == 0


def test_visibility_target_census_estimates_duration_conditioned_reappearance() -> None:
    result = audit_visibility_target_sequences(
        (
            (
                {"object/a": 1},
                {"object/a": 0},
                {"object/a": 0},
                {"object/a": 0},
                {"object/a": 1},
            ),
            (
                {"object/a": 0},
                {"object/a": 0},
            ),
        )
    )

    hazard = result["hidden_reappearance_hazard"]
    assert hazard["exact_by_elapsed_hidden_frames"] == {
        "1": {
            "at_risk_count": 2,
            "reappearance_count": 0,
            "remained_hidden_count": 2,
            "reappearance_hazard": 0.0,
        },
        "2": {
            "at_risk_count": 1,
            "reappearance_count": 0,
            "remained_hidden_count": 1,
            "reappearance_hazard": 0.0,
        },
        "3": {
            "at_risk_count": 1,
            "reappearance_count": 1,
            "remained_hidden_count": 0,
            "reappearance_hazard": 1.0,
        },
    }
    assert hazard["binned_by_elapsed_hidden_frames"]["1"]["at_risk_count"] == 2
    assert hazard["binned_by_elapsed_hidden_frames"]["3-4"]["reappearance_hazard"] == 1.0
    assert hazard["reacquired_run_length"]["maximum"] == 3.0
    assert hazard["right_censored_run_length"]["maximum"] == 2.0
    assert hazard["death_terminated_run_length"]["count"] == 0
    seen = hazard["seen_then_hidden"]
    assert seen["exact_by_elapsed_hidden_frames"] == {
        "1": {
            "at_risk_count": 1,
            "reappearance_count": 0,
            "remained_hidden_count": 1,
            "reappearance_hazard": 0.0,
        },
        "2": {
            "at_risk_count": 1,
            "reappearance_count": 0,
            "remained_hidden_count": 1,
            "reappearance_hazard": 0.0,
        },
        "3": {
            "at_risk_count": 1,
            "reappearance_count": 1,
            "remained_hidden_count": 0,
            "reappearance_hazard": 1.0,
        },
    }
    assert seen["reacquired_run_length"]["maximum"] == 3.0
    assert seen["right_censored_run_length"]["count"] == 0


def test_visibility_target_census_rejects_boolean_labels() -> None:
    with pytest.raises(ValueError, match="0, 1 or None"):
        audit_visibility_target_sequences((({"object/a": True},),))


def test_partitioned_visibility_census_never_pools_split_statistics() -> None:
    result = audit_partitioned_visibility_target_sequences(
        {
            "heldout": (({"object": 0}, {"object": 1}),),
            "train": (({"object": 1}, {"object": 1}, {"object": 0}),),
        }
    )

    assert tuple(result) == ("heldout", "train")
    assert result["train"]["transition_count"] == {
        "0->0": 0,
        "0->1": 0,
        "1->0": 1,
        "1->1": 1,
    }
    assert result["heldout"]["transition_count"] == {
        "0->0": 0,
        "0->1": 1,
        "1->0": 0,
        "1->1": 0,
    }


def _trace(
    *,
    key: str | None,
    detection: float,
    target: int,
    supervised: bool = True,
    identity_key: str | None = None,
    previous_visibility: float | None = None,
    persistence: float | None = None,
    reappearance: float | None = None,
) -> dict[str, object]:
    existence = 0.8 if key is not None else 0.0
    trace: dict[str, object] = {
        "prior_key": key,
        "identity_key": key if identity_key is None else identity_key,
        "prior_existence_probability": existence,
        "prior_visibility_probability": existence * detection,
        "target_visibility": float(target),
        "target_visibility_supervised": supervised,
    }
    kernel_values = previous_visibility, persistence, reappearance
    if any(value is not None for value in kernel_values):
        if not all(value is not None for value in kernel_values):
            raise ValueError("test transition-kernel fields are atomic")
        trace.update(
            {
                "previous_conditional_visibility_probability": previous_visibility,
                "visibility_persistence_probability": persistence,
                "visibility_reappearance_probability": reappearance,
                "prior_conditional_detection_probability": detection,
            }
        )
    return trace


def _report(
    frames: list[dict[str, object]],
    *,
    schema: str = "picf-next.molmoact2-m3-temporal-audit.v8",
) -> dict[str, object]:
    return {
        "schema": schema,
        "checkpoint_code_revision": "abc123",
        "checkpoint_model_sha256": "a" * 64,
        "rows": frames,
    }


def test_lifecycle_audit_measures_transitions_and_hidden_runs() -> None:
    labels = (1, 0, 0, 1)
    detections = (0.9, 0.8, 0.2, 0.3)
    frames = [
        {
            "rank": 0,
            "step": step,
            "episode_key": "episode-a",
            "episode_reset": step == 1,
            "row_traces": [
                _trace(key="object-a", detection=detection, target=target),
            ],
        }
        for step, (target, detection) in enumerate(
            zip(labels, detections, strict=True),
            start=1,
        )
    ]

    result = audit_lifecycle_reports([_report(frames)])

    assert result["supervised_row_count"] == 4
    assert result["transition_count"] == {
        "0->0": 1,
        "0->1": 1,
        "1->0": 1,
        "1->1": 0,
    }
    assert result["hidden_run_length"]["count"] == 1
    assert result["hidden_run_length"]["mean"] == 2.0
    calibration = result["conditional_detection_calibration"]
    assert calibration is not None
    assert calibration["positive_count"] == 2
    assert calibration["negative_count"] == 2


def test_lifecycle_audit_breaks_transitions_at_unknown_and_episode_reset() -> None:
    frames = [
        {
            "rank": 0,
            "step": 1,
            "episode_key": "episode-a",
            "episode_reset": True,
            "row_traces": [_trace(key="object-a", detection=0.8, target=0)],
        },
        {
            "rank": 0,
            "step": 2,
            "episode_key": "episode-a",
            "episode_reset": False,
            "row_traces": [
                _trace(key="object-a", detection=0.5, target=0, supervised=False),
            ],
        },
        {
            "rank": 0,
            "step": 3,
            "episode_key": "episode-b",
            "episode_reset": True,
            "row_traces": [_trace(key="object-a", detection=0.2, target=0)],
        },
    ]

    result = audit_lifecycle_reports([_report(frames)])

    assert sum(result["transition_count"].values()) == 0
    assert result["unsupervised_row_count"] == 1
    assert result["hidden_run_length"]["count"] == 2


def test_lifecycle_audit_calibrates_the_two_state_visibility_kernel() -> None:
    rows = (
        (1, 0.60, 0.5, 0.9, 0.3),
        (0, 0.22, 0.9, 0.2, 0.4),
        (0, 0.17, 0.1, 0.8, 0.1),
        (1, 0.86, 0.2, 0.7, 0.9),
    )
    frames = [
        {
            "rank": 0,
            "step": step,
            "episode_key": "episode-a",
            "episode_reset": step == 1,
            "row_traces": [
                _trace(
                    key="object-a",
                    detection=detection,
                    target=target,
                    previous_visibility=previous_visibility,
                    persistence=persistence,
                    reappearance=reappearance,
                )
            ],
        }
        for step, (target, detection, previous_visibility, persistence, reappearance) in enumerate(
            rows,
            start=1,
        )
    ]

    result = audit_lifecycle_reports(
        [_report(frames, schema="picf-next.molmoact2-m3-temporal-audit.v9")]
    )

    kernel = result["visibility_transition_kernel"]
    assert kernel is not None
    assert kernel["maximum_mixture_residual"] < 1e-12
    assert kernel["visible_origin_calibration"]["sample_count"] == 1
    assert kernel["visible_origin_calibration"]["negative_count"] == 1
    assert kernel["hidden_origin_calibration"]["sample_count"] == 2
    assert kernel["hidden_origin_calibration"]["positive_count"] == 1


def test_lifecycle_audit_excludes_identity_changes() -> None:
    report = _report(
        [
            {
                "rank": 0,
                "step": 1,
                "episode_key": "episode-a",
                "episode_reset": True,
                "row_traces": [
                    _trace(
                        key="object-a",
                        identity_key="object-b",
                        detection=0.9,
                        target=1,
                    )
                ],
            }
        ]
    )

    result = audit_lifecycle_reports([report])

    assert result["supervised_row_count"] == 0
    assert result["identity_changed_row_count"] == 1
    assert result["conditional_detection_calibration"] is None


def test_lifecycle_audit_rejects_duplicate_rank_step() -> None:
    frame = {
        "rank": 0,
        "step": 1,
        "episode_key": "episode-a",
        "episode_reset": True,
        "row_traces": [],
    }

    with pytest.raises(ValueError, match="duplicate rank/step"):
        audit_lifecycle_reports([_report([frame]), _report([frame])])

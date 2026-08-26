# ruff: noqa: E402  # Optional torch gate must precede torch-backed project imports.
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")
from safetensors.torch import save_file

from picf_next.eval.cardinality import (
    binary_calibration_metrics,
    continuous_calibration_metrics,
    count_metrics,
    poisson_binomial_distribution,
    poisson_binomial_mode,
    query_usage_summary,
    select_count_threshold,
    task_usage_summary,
    threshold_sweep,
)
from tools.audit_molmoact2_m2_cardinality import _load_current_frame_state


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cardinality_audit_loads_bound_full_training_safetensors(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    model_path = checkpoint / "model.safetensors"
    save_file(
        {
            "host.weight": torch.ones(2),
            "joint_bridge.sequence_bridge.core.discovery.weight": torch.arange(3),
            "joint_bridge.sequence_bridge.core.discovery.bias": torch.arange(2),
            "joint_bridge.sequence_bridge.core.posterior_filter.weight": torch.ones(4),
        },
        model_path,
    )
    model_sha256 = _sha256(model_path)
    (checkpoint / "picf_control.json").write_text(
        json.dumps(
            {
                "schema": "picf-next.checkpoint-control-manifest.v2",
                "state_files": {
                    "model.safetensors": {
                        "sha256": model_sha256,
                        "size_bytes": model_path.stat().st_size,
                    }
                },
            }
        ),
        encoding="ascii",
    )

    state, actual_path, actual_sha256 = _load_current_frame_state(
        checkpoint,
        expected_names={"discovery.weight", "discovery.bias"},
    )

    assert set(state) == {"discovery.weight", "discovery.bias"}
    torch.testing.assert_close(state["discovery.weight"], torch.arange(3))
    assert actual_path == model_path
    assert actual_sha256 == model_sha256


def test_cardinality_audit_rejects_mixed_direct_and_prefixed_core_keys(tmp_path) -> None:
    checkpoint = tmp_path / "mixed.safetensors"
    save_file(
        {
            "discovery.weight": torch.ones(1),
            "joint_bridge.sequence_bridge.core.discovery.weight": torch.ones(1),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="mixes direct and full-training"):
        _load_current_frame_state(
            checkpoint,
            expected_names={"discovery.weight"},
        )


def test_cardinality_audit_rejects_unknown_full_core_namespace(tmp_path) -> None:
    checkpoint = tmp_path / "unknown.safetensors"
    save_file(
        {
            "joint_bridge.sequence_bridge.core.discovery.weight": torch.ones(1),
            "joint_bridge.sequence_bridge.core.unexpected.weight": torch.ones(1),
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="unexpected PICF core"):
        _load_current_frame_state(
            checkpoint,
            expected_names={"discovery.weight"},
        )


def test_binary_calibration_metrics_are_proper_scores() -> None:
    report = binary_calibration_metrics(
        [0.8, 0.6, 0.3, 0.1],
        [1, 1, 0, 0],
        bins=2,
    )

    assert report["brier"] == pytest.approx((0.2**2 + 0.4**2 + 0.3**2 + 0.1**2) / 4)
    assert report["positive_probability"]["mean"] == pytest.approx(0.7)
    assert report["negative_probability"]["mean"] == pytest.approx(0.2)
    assert report["expected_calibration_error"] == pytest.approx(0.25)
    assert report["negative_log_likelihood"] > 0.0


def test_continuous_calibration_metrics_score_soft_correctness_targets() -> None:
    report = continuous_calibration_metrics(
        [0.9, 0.6, 0.2, 0.1],
        [1.0, 0.4, 0.3, 0.0],
        bins=2,
    )

    assert report["mean_absolute_error"] == pytest.approx(0.125)
    assert report["mean_squared_error"] == pytest.approx(0.0175)
    assert report["prediction"]["mean"] == pytest.approx(0.45)
    assert report["target"]["mean"] == pytest.approx(0.425)
    assert report["expected_calibration_error"] == pytest.approx(0.025)


@pytest.mark.parametrize(
    ("predictions", "targets"),
    (([], []), ([0.5], []), ([float("nan")], [0.5]), ([0.5], [1.1])),
)
def test_continuous_calibration_metrics_reject_invalid_inputs(
    predictions: list[float],
    targets: list[float],
) -> None:
    with pytest.raises(ValueError):
        continuous_calibration_metrics(predictions, targets)


def test_count_metrics_separate_hard_and_posterior_mean_counts() -> None:
    report = count_metrics(
        [[0.9, 0.8, 0.1], [0.7, 0.4, 0.2]],
        [2, 2],
        threshold=0.5,
    )

    assert report["hard_count_mean"] == pytest.approx(1.5)
    assert report["hard_count_mae"] == pytest.approx(0.5)
    assert report["hard_exact_count_accuracy"] == pytest.approx(0.5)
    assert report["posterior_expected_count_mean"] == pytest.approx(1.55)
    assert report["posterior_expected_count_mae"] == pytest.approx(0.45)


def test_poisson_binomial_count_posterior_is_exact_and_normalized() -> None:
    distribution = poisson_binomial_distribution([0.6, 0.6])

    assert distribution == pytest.approx((0.16, 0.48, 0.36))
    assert sum(distribution) == pytest.approx(1.0)
    assert poisson_binomial_mode([0.6, 0.6]) == 1


def test_count_mode_can_differ_from_independent_set_map_cardinality() -> None:
    report = count_metrics([[0.6, 0.6]], [1], threshold=0.5)

    assert report["hard_count_mean"] == 2
    assert report["hard_exact_count_accuracy"] == 0.0
    assert report["posterior_mode_count_mean"] == 1
    assert report["posterior_mode_count_mae"] == 0.0
    assert report["posterior_mode_exact_count_accuracy"] == 1.0


def test_neutral_existence_is_not_counted_as_positive_object_evidence() -> None:
    report = count_metrics([[0.9, 0.5, 0.1]], [1], threshold=0.5)

    assert report["hard_count_mean"] == 1
    assert report["hard_exact_count_accuracy"] == 1


def test_threshold_selection_is_validation_only_and_tie_breaks_toward_half() -> None:
    probabilities = [[0.8, 0.52, 0.2], [0.9, 0.48, 0.1]]
    targets = [1, 1]
    rows = threshold_sweep(probabilities, targets)

    selected = select_count_threshold(rows)

    assert selected["hard_count_mae"] == 0.0
    assert selected["hard_exact_count_accuracy"] == 1.0
    assert selected["threshold"] == pytest.approx(0.52)


def test_query_usage_summary_reports_slot_priors_without_relabelling_queries() -> None:
    rows = [
        {"target_count": 2, "matched_query_indices": [0, 2], "task_key": "a"},
        {"target_count": 1, "matched_query_indices": [0], "task_key": "a"},
        {"target_count": 1, "matched_query_indices": [1], "task_key": "b"},
    ]

    summary = query_usage_summary(rows, query_count=3)
    per_task = task_usage_summary(rows, query_count=3)

    assert summary["target_count"]["mean"] == pytest.approx(4 / 3)
    assert [
        row["matched_sample_fraction"] for row in summary["query_match_frequency"]
    ] == pytest.approx([2 / 3, 1 / 3, 1 / 3])
    assert per_task["a"]["sample_count"] == 2
    assert per_task["b"]["query_match_frequency"][1]["matched_sample_fraction"] == 1.0

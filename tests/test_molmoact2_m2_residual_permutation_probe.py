from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.run_molmoact2_m2_residual_permutation_probe import (
    _calibration_loss,
    _control_target_alignment,
    _decision,
    _identity_stratified_derangement,
    _load_config,
    _within_group_centered_rank_correlation,
)


def _example(sample: str, identity: str, residual: float) -> dict:
    return {
        "object_key": f"{sample}\x1f{identity}",
        "sample_key": sample,
        "identity_key": identity,
        "squared_residual": np.asarray([residual, residual + 0.1, residual + 0.2]),
        "supervised": np.asarray([True, True, True]),
    }


def _arm(*, global_rank: float, within_rank: float, nll: float) -> dict:
    return {
        split: {
            "global_uncertainty_error_spearman": global_rank,
            "within_identity_axis_centered_rank_correlation": within_rank,
            "gaussian_nll_without_constant": nll,
            "aggregate_error_to_variance_ratio": 1.0,
            "row_count": 10,
        }
        for split in ("train", "validation", "heldout")
    }


def _acceptance() -> dict:
    return {
        "maximum_absolute_control_target_within_identity_rank_correlation": 0.05,
        "minimum_aligned_control_within_identity_axis_rank_margin": 0.1,
        "minimum_global_uncertainty_error_spearman": 0.0,
        "minimum_within_identity_axis_centered_rank_correlation": 0.1,
        "require_aligned_nll_below_control_on_validation_and_heldout": True,
        "require_aligned_nll_below_reset_on_heldout": True,
    }


def test_checked_in_residual_permutation_config_is_valid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = _load_config(
        root / "configs/training/molmoact2_calvin_m2_residual_permutation_probe.json"
    )

    assert config["optimization"]["passes"] == 1
    assert config["protocol"]["checkpoint_selection"] == ("fixed-final-step-no-reselection")


def test_identity_stratified_derangement_is_complete_and_has_no_fixed_points() -> None:
    examples = [
        _example(f"sample-{index}", identity, float(index))
        for identity in ("block", "button")
        for index in range(7)
    ]

    mapping = _identity_stratified_derangement(examples, seed=17)
    by_key = {example["object_key"]: example for example in examples}

    assert set(mapping) == set(by_key)
    assert set(mapping.values()) == set(by_key)
    assert mapping == _identity_stratified_derangement(examples, seed=17)
    for source, target in mapping.items():
        assert source != target
        assert by_key[source]["sample_key"] != by_key[target]["sample_key"]
        assert by_key[source]["identity_key"] == by_key[target]["identity_key"]


def test_centered_rank_removes_between_identity_difficulty() -> None:
    error = [1.0, 2.0, 3.0, 101.0, 102.0, 103.0]
    same_within_order = [11.0, 12.0, 13.0, 1.0, 2.0, 3.0]
    groups = ["a", "a", "a", "b", "b", "b"]

    correlation = _within_group_centered_rank_correlation(
        error,
        same_within_order,
        groups,
    )

    assert correlation == 1.0


def test_control_alignment_reports_the_actual_deranged_residual_relation() -> None:
    examples = [
        _example(f"sample-{index}", identity, float(index + offset))
        for identity, offset in (("block", 0), ("button", 100))
        for index in range(11)
    ]
    mapping = _identity_stratified_derangement(examples, seed=23)

    alignment = _control_target_alignment(examples, mapping)

    assert alignment["observation_count"] == 22
    assert alignment["identity_count"] == 2
    assert alignment["fixed_point_count"] == 0
    assert alignment["within_identity_centered_rank_correlation"] is not None


def test_decision_requires_conditional_rank_and_paired_nll_gain() -> None:
    metrics = {
        "reset": _arm(global_rank=0.0, within_rank=0.0, nll=0.6),
        "aligned": _arm(global_rank=0.5, within_rank=0.3, nll=0.2),
        "control": _arm(global_rank=0.4, within_rank=0.0, nll=0.4),
    }

    decision = _decision(
        metrics=metrics,
        control_target_alignment={
            "within_identity_centered_rank_correlation": 0.0,
        },
        frozen_non_variance_state_exact=True,
        reset_metric_reproduction_exact=True,
        acceptance=_acceptance(),
    )

    assert decision["status"] == "SUPPORTS_CONDITIONAL_RESIDUAL_CALIBRATION"
    assert decision["failed_checks"] == []
    assert decision["later_gates_authorized"] == []
    assert decision["production_training_changes_authorized"] == []


def test_decision_rejects_a_static_identity_prior_and_correlated_control() -> None:
    metrics = {
        "reset": _arm(global_rank=0.0, within_rank=0.0, nll=0.6),
        "aligned": _arm(global_rank=0.5, within_rank=0.04, nll=0.2),
        "control": _arm(global_rank=0.45, within_rank=0.02, nll=0.21),
    }

    decision = _decision(
        metrics=metrics,
        control_target_alignment={
            "within_identity_centered_rank_correlation": 0.2,
        },
        frozen_non_variance_state_exact=True,
        reset_metric_reproduction_exact=True,
        acceptance=_acceptance(),
    )

    assert decision["status"] == "DOES_NOT_SUPPORT_CONDITIONAL_RESIDUAL_CALIBRATION"
    assert "control_targets_decorrelated_within_identity" in decision["failed_checks"]
    assert "aligned_heldout_within_identity_axis_rank" in decision["failed_checks"]
    assert "aligned_beats_control_heldout_within_identity_axis_rank" in decision["failed_checks"]


def test_calibration_loss_preserves_production_bfloat16_variance_path() -> None:
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("production calibration path requires CUDA autocast")

    device = torch.device("cuda:0")
    head = torch.nn.Linear(2, 2).to(device)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[0.5, -0.25], [-0.125, 0.75]], device=device))
        head.bias.copy_(torch.tensor([0.1, -0.2], device=device))
    examples = [
        {
            "query_feature": torch.tensor([0.3, -0.7], dtype=torch.bfloat16),
            "squared_residual": torch.tensor([0.4, 0.9]),
            "measurement_variance": torch.tensor([0.01, 0.02]),
            "supervised": torch.tensor([True, True]),
        }
    ]

    actual = _calibration_loss(
        head,
        examples,
        examples,
        minimum_variance=1e-4,
        device=device,
    )
    features = examples[0]["query_feature"].to(device).unsqueeze(0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        raw = head(features)
    variance = (torch.nn.functional.softplus(raw.float()) + 1e-4).to(torch.bfloat16).float()
    combined = variance + examples[0]["measurement_variance"].to(device)
    squared = examples[0]["squared_residual"].to(device)
    expected = 0.5 * (squared / combined + combined.log())

    torch.testing.assert_close(actual, expected.mean())

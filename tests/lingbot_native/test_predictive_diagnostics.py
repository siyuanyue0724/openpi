from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.predictive_diagnostics import (
    predictive_latent_diagnostics,
    predictive_latent_diagnostics_from_mapping,
    predictive_target_pretraining_readiness,
    predictive_temporal_diagnostics,
    predictive_temporal_diagnostics_from_mapping,
    predictive_temporal_pretraining_readiness,
    predictive_visible_support_diagnostics,
    predictive_visible_support_diagnostics_from_mapping,
)


def test_predictive_diagnostics_separate_identity_across_target_frames() -> None:
    features = torch.tensor(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
            [2.1, -1.9, 0.1, -0.1],
            [0.1, -0.1, 2.1, -1.9],
        ]
    )
    result = predictive_latent_diagnostics(
        features,
        identity_keys=("object/a", "object/b", "object/a", "object/b"),
        target_group_keys=("frame/1", "frame/1", "frame/2", "frame/2"),
        pair_chunk_size=2,
    )

    assert result.retrieval_query_count == 4
    assert result.identity_top1_accuracy == 1.0
    assert result.identity_chance_accuracy == 0.5
    assert result.same_identity_cosine is not None
    assert result.different_identity_cosine is not None
    assert result.same_identity_cosine > result.different_identity_cosine
    assert result.effective_rank > 1
    assert not result.obvious_numerical_collapse
    assert predictive_latent_diagnostics_from_mapping(result.as_dict()) == result
    assert predictive_target_pretraining_readiness(result) == (True, ())


def test_predictive_diagnostics_detect_exact_target_collapse() -> None:
    features = torch.tensor([[2.0, -1.0, 0.0]]).expand(4, -1).clone()
    result = predictive_latent_diagnostics(
        features,
        identity_keys=("a", "b", "a", "b"),
        target_group_keys=("f1", "f1", "f2", "f2"),
    )

    assert result.mean_dimension_variance == 0.0
    assert result.effective_rank == 0.0
    assert result.obvious_numerical_collapse
    ready, failures = predictive_target_pretraining_readiness(result)
    assert not ready
    assert "obvious_numerical_collapse" in failures


def test_predictive_diagnostics_exclude_same_target_group_duplicates() -> None:
    features = torch.eye(4)
    result = predictive_latent_diagnostics(
        features,
        identity_keys=("a", "a", "b", "b"),
        target_group_keys=("same-frame",) * 4,
    )

    assert result.retrieval_query_count == 0
    assert result.identity_top1_accuracy is None
    assert result.same_identity_cosine is None
    assert result.different_identity_cosine is None


@pytest.mark.parametrize(
    ("features", "identities", "groups", "message"),
    (
        (torch.ones(1, 3), ("a",), ("f",), "at least two"),
        (torch.ones(2, 3), ("a",), ("f", "g"), "match"),
        (torch.tensor([[1.0, float("nan")], [0.0, 1.0]]), ("a", "b"), ("f", "g"), "finite"),
    ),
)
def test_predictive_diagnostics_reject_invalid_inputs(
    features: torch.Tensor,
    identities: tuple[str, ...],
    groups: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        predictive_latent_diagnostics(
            features,
            identity_keys=identities,
            target_group_keys=groups,
        )


def test_predictive_diagnostic_parser_rejects_edited_rank_fraction() -> None:
    result = predictive_latent_diagnostics(
        torch.eye(4),
        identity_keys=("a", "b", "a", "b"),
        target_group_keys=("f1", "f1", "f2", "f2"),
    ).as_dict()
    result["effective_rank_fraction"] = 0.0

    with pytest.raises(ValueError, match="inconsistent"):
        predictive_latent_diagnostics_from_mapping(result)


def test_predictive_diagnostic_parser_recomputes_collapse_flag() -> None:
    result = predictive_latent_diagnostics(
        torch.tensor([[2.0, -1.0, 0.0]]).expand(4, -1).clone(),
        identity_keys=("a", "b", "a", "b"),
        target_group_keys=("f1", "f1", "f2", "f2"),
    ).as_dict()
    assert result["obvious_numerical_collapse"] is True
    result["obvious_numerical_collapse"] = False

    with pytest.raises(ValueError, match="inconsistent"):
        predictive_latent_diagnostics_from_mapping(result)


def test_predictive_visible_support_diagnostics_bind_full_scan_and_sample() -> None:
    result = predictive_visible_support_diagnostics(
        torch.tensor([0.01, 0.04, 0.09, 0.16]),
        supported_count=8,
        total_importance=0.8,
        minimum_importance=0.01,
        maximum_importance=0.25,
    )

    assert result.sampled_count == 4
    assert result.supported_count == 8
    assert result.mean_visible_image_fraction == pytest.approx(0.1)
    assert predictive_visible_support_diagnostics_from_mapping(result.as_dict()) == result

    edited = result.as_dict()
    edited["sampled_p05_visible_image_fraction"] = 0.2
    with pytest.raises(ValueError, match="inconsistent"):
        predictive_visible_support_diagnostics_from_mapping(edited)


@pytest.mark.parametrize(
    ("sample", "supported_count", "total", "minimum", "maximum"),
    (
        (torch.tensor([0.0, 0.1]), 2, 0.1, 0.0, 0.1),
        (torch.tensor([0.1, 0.2]), 1, 0.3, 0.1, 0.2),
        (torch.tensor([0.1, 0.2]), 2, 0.8, 0.1, 0.2),
    ),
)
def test_predictive_visible_support_diagnostics_reject_invalid_moments(
    sample: torch.Tensor,
    supported_count: int,
    total: float,
    minimum: float,
    maximum: float,
) -> None:
    with pytest.raises((ValueError, TypeError)):
        predictive_visible_support_diagnostics(
            sample,
            supported_count=supported_count,
            total_importance=total,
            minimum_importance=minimum,
            maximum_importance=maximum,
        )


def test_predictive_temporal_diagnostics_measure_current_copy_baseline() -> None:
    current = torch.tensor(
        [
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
            [2.0, -2.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, -2.0],
        ]
    )
    future = current + torch.tensor(
        [
            [0.0, 0.3, -0.3, 0.0],
            [0.2, 0.0, 0.0, -0.2],
            [0.0, 0.6, -0.6, 0.0],
            [0.4, 0.0, 0.0, -0.4],
        ]
    )
    result = predictive_temporal_diagnostics(
        current,
        future,
        identity_keys=("a", "b", "a", "b"),
        horizons=(1, 1, 2, 2),
    )

    assert result.mean_current_future_l1 > 0
    assert result.sampled_p90_current_future_l1 >= result.sampled_median_current_future_l1
    assert result.horizon_count == 2
    assert not result.obvious_no_temporal_content
    assert predictive_temporal_diagnostics_from_mapping(result.as_dict()) == result
    assert predictive_temporal_pretraining_readiness(result) == (True, ())


def test_predictive_temporal_diagnostics_reject_exact_current_copy_targets() -> None:
    features = torch.tensor([[2.0, -2.0, 0.0], [0.0, 2.0, -2.0], [2.0, 0.0, -2.0]])
    result = predictive_temporal_diagnostics(
        features,
        features.clone(),
        identity_keys=("a", "b", "a"),
        horizons=(1, 1, 2),
    )

    assert result.mean_current_future_l1 == 0.0
    assert result.numerically_unchanged_fraction == 1.0
    assert result.obvious_no_temporal_content
    assert predictive_temporal_pretraining_readiness(result) == (
        False,
        ("no_measurable_current_to_future_target_change",),
    )
    edited = result.as_dict()
    edited["obvious_no_temporal_content"] = False
    with pytest.raises(ValueError, match="inconsistent"):
        predictive_temporal_diagnostics_from_mapping(edited)

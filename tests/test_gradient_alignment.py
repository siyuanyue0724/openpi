from __future__ import annotations

import math

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.gradient_alignment import (
    GradientPairMoments,
    WeightedGradientPairMoments,
    WeightedGradientTripleMoments,
    qwen_gradient_group,
    summarize_qwen_gradient_alignment,
    summarize_weighted_qwen_gradient_alignment,
    summarize_weighted_qwen_gradient_triple,
)


def test_qwen_gradient_group_covers_declared_trainable_surfaces() -> None:
    prefix = "model.qwenvl_with_expert.qwenvl.model."
    assert (
        qwen_gradient_group(f"{prefix}language_model.embed_tokens.weight")
        == "language_embedding_tied_lm_head"
    )
    assert (
        qwen_gradient_group(f"{prefix}language_model.layers.7.self_attn.q_proj.weight")
        == "language_layer_07"
    )
    assert qwen_gradient_group(f"{prefix}language_model.norm.weight") == "language_final_norm"
    assert qwen_gradient_group(f"{prefix}visual.merger.linear_fc1.weight") == "visual_merger"


def test_qwen_gradient_group_rejects_an_unplanned_small_head() -> None:
    with pytest.raises(ContractError, match="undeclared"):
        qwen_gradient_group("model.picf_private_scorer.weight")


def test_gradient_alignment_reports_conflict_and_mean_descent() -> None:
    prefix = "model.qwenvl_with_expert.qwenvl.model."
    report = summarize_qwen_gradient_alignment(
        {
            f"{prefix}language_model.embed_tokens.weight": GradientPairMoments(
                dot=-1.0,
                lattice8_squared_norm=4.0,
                lattice14_squared_norm=1.0,
                elements=2,
            ),
            f"{prefix}language_model.layers.0.self_attn.q_proj.weight": GradientPairMoments(
                dot=2.0,
                lattice8_squared_norm=4.0,
                lattice14_squared_norm=4.0,
                elements=4,
            ),
            f"{prefix}visual.merger.linear_fc1.weight": GradientPairMoments(
                dot=-2.0,
                lattice8_squared_norm=1.0,
                lattice14_squared_norm=4.0,
                elements=3,
            ),
        }
    )

    global_report = report["global"]
    assert isinstance(global_report, dict)
    assert global_report["dot_product"] == -1.0
    assert global_report["element_count"] == 9
    assert global_report["parameter_tensor_negative_dot_count"] == 2
    assert global_report["parameter_tensor_negative_dot_mass_fraction"] == 0.6
    assert math.isclose(global_report["cosine"], -1.0 / math.sqrt(81.0))
    assert global_report["mean_gradient_descends_lattice8"] is True
    assert global_report["mean_gradient_descends_lattice14"] is True

    groups = report["groups"]
    assert isinstance(groups, dict)
    assert groups["visual_merger"]["mean_gradient_descends_lattice8"] is False


@pytest.mark.parametrize(
    "kwargs",
    (
        {"dot": float("nan"), "lattice8_squared_norm": 1.0, "lattice14_squared_norm": 1.0},
        {"dot": 0.0, "lattice8_squared_norm": -1.0, "lattice14_squared_norm": 1.0},
    ),
)
def test_gradient_pair_moments_reject_invalid_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ContractError, match="invalid"):
        GradientPairMoments(**kwargs, elements=1)


def test_weighted_gradient_alignment_reports_the_actual_mixed_direction() -> None:
    prefix = "model.qwenvl_with_expert.qwenvl.model."
    report = summarize_weighted_qwen_gradient_alignment(
        {
            f"{prefix}language_model.embed_tokens.weight": WeightedGradientPairMoments(
                dot=-3.0,
                first_squared_norm=4.0,
                second_squared_norm=100.0,
                elements=2,
            ),
            f"{prefix}visual.merger.linear_fc1.weight": WeightedGradientPairMoments(
                dot=1.0,
                first_squared_norm=1.0,
                second_squared_norm=4.0,
                elements=3,
            ),
        },
        first_objective="calvin",
        second_objective="retention",
        first_weight=1.0,
        second_weight=0.1,
    )

    global_report = report["global"]
    assert isinstance(global_report, dict)
    assert report["first_objective"] == "calvin"
    assert report["second_objective"] == "retention"
    assert global_report["dot_product"] == -2.0
    assert global_report["mixed_gradient_first_directional_inner_product"] == pytest.approx(4.8)
    assert global_report["mixed_gradient_second_directional_inner_product"] == pytest.approx(8.4)
    assert global_report["mixed_gradient_descends_first_objective"] is True
    assert global_report["mixed_gradient_descends_second_objective"] is True
    assert global_report["mixed_gradient_norm"] == pytest.approx(math.sqrt(5.64))

    groups = report["groups"]
    assert isinstance(groups, dict)
    assert (
        groups["language_embedding_tied_lm_head"]["mixed_gradient_descends_second_objective"]
        is True
    )


def test_weighted_gradient_alignment_rejects_invalid_objectives_and_weights() -> None:
    prefix = "model.qwenvl_with_expert.qwenvl.model."
    moments = {
        f"{prefix}language_model.embed_tokens.weight": WeightedGradientPairMoments(
            dot=0.0,
            first_squared_norm=1.0,
            second_squared_norm=1.0,
            elements=1,
        ),
        f"{prefix}visual.merger.linear_fc1.weight": WeightedGradientPairMoments(
            dot=0.0,
            first_squared_norm=1.0,
            second_squared_norm=1.0,
            elements=1,
        ),
    }
    with pytest.raises(ContractError, match="distinct"):
        summarize_weighted_qwen_gradient_alignment(
            moments,
            first_objective="same",
            second_objective="same",
            first_weight=1.0,
            second_weight=0.1,
        )
    with pytest.raises(ContractError, match="positive"):
        summarize_weighted_qwen_gradient_alignment(
            moments,
            first_objective="calvin",
            second_objective="retention",
            first_weight=1.0,
            second_weight=0.0,
        )


def test_weighted_gradient_triple_reports_exact_candidate_direction() -> None:
    prefix = "model.qwenvl_with_expert.qwenvl.model."
    report = summarize_weighted_qwen_gradient_triple(
        {
            f"{prefix}language_model.embed_tokens.weight": WeightedGradientTripleMoments(
                first_squared_norm=4.0,
                second_squared_norm=9.0,
                third_squared_norm=1.0,
                first_second_dot=1.0,
                first_third_dot=-1.0,
                second_third_dot=2.0,
                elements=2,
            ),
            f"{prefix}visual.merger.linear_fc1.weight": WeightedGradientTripleMoments(
                first_squared_norm=1.0,
                second_squared_norm=1.0,
                third_squared_norm=4.0,
                first_second_dot=-1.0,
                first_third_dot=0.0,
                second_third_dot=0.0,
                elements=3,
            ),
        },
        objective_names=("target", "scene", "public"),
        weights=(0.5, 0.5, 0.1),
    )

    global_report = report["global"]
    assert isinstance(global_report, dict)
    assert report["objective_weights"] == {"target": 0.5, "scene": 0.5, "public": 0.1}
    assert global_report["element_count"] == 5
    assert global_report["gradient_norms"] == {
        "target": pytest.approx(math.sqrt(5.0)),
        "scene": pytest.approx(math.sqrt(10.0)),
        "public": pytest.approx(math.sqrt(5.0)),
    }
    assert global_report["gradient_squared_norms"] == {
        "target": pytest.approx(5.0),
        "scene": pytest.approx(10.0),
        "public": pytest.approx(5.0),
    }
    assert global_report["mixed_gradient_directional_inner_products"] == {
        "target": pytest.approx(2.4),
        "scene": pytest.approx(5.2),
        "public": pytest.approx(1.0),
    }
    assert global_report["mixed_gradient_descends"] == {
        "target": True,
        "scene": True,
        "public": True,
    }
    assert global_report["mixed_gradient_norm"] == pytest.approx(math.sqrt(3.9))
    assert global_report["pairwise_dot_products"] == {
        "target__scene": pytest.approx(0.0),
        "target__public": pytest.approx(-1.0),
        "scene__public": pytest.approx(2.0),
    }
    assert report["groups"]["visual_merger"]["mixed_gradient_descends"]["target"] is False


def test_weighted_gradient_triple_rejects_impossible_gram_matrix() -> None:
    with pytest.raises(ContractError, match="Cauchy-Schwarz"):
        WeightedGradientTripleMoments(
            first_squared_norm=1.0,
            second_squared_norm=1.0,
            third_squared_norm=1.0,
            first_second_dot=2.0,
            first_third_dot=0.0,
            second_third_dot=0.0,
            elements=1,
        )

    with pytest.raises(ContractError, match="positive semidefinite"):
        WeightedGradientTripleMoments(
            first_squared_norm=1e-20,
            second_squared_norm=1e-20,
            third_squared_norm=1e-20,
            first_second_dot=0.9e-20,
            first_third_dot=0.9e-20,
            second_third_dot=-0.9e-20,
            elements=1,
        )

    with pytest.raises(ContractError, match="Cauchy-Schwarz"):
        WeightedGradientTripleMoments(
            first_squared_norm=1e-20,
            second_squared_norm=1e-20,
            third_squared_norm=1e-20,
            first_second_dot=1e-15,
            first_third_dot=0.0,
            second_third_dot=0.0,
            elements=1,
        )

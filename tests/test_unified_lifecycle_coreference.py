from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.unified.coreference import (  # noqa: E402
    grouped_relation_evidence,
    responsibility_weighted_message,
    shared_qk_coreference,
)
from picf_next.unified.lifecycle import (  # noqa: E402
    categorical_lifecycle_prior,
    deterministic_logdet_ci_weights,
    footprint_evidence,
    generalized_covariance_intersection,
    logarithmic_lifecycle_pool,
    nonempty_probability,
    posterior_expected_age,
    reliability_simplex,
)


def test_lifecycle_prior_and_derived_nonempty_are_exact() -> None:
    continuation = torch.tensor([[0.7, 0.2]])
    birth_hazard = torch.tensor([[0.5, 0.25]])
    log_probs = categorical_lifecycle_prior(continuation, birth_hazard)
    expected = torch.tensor([[[0.7, 0.15, 0.15], [0.2, 0.2, 0.6]]])
    torch.testing.assert_close(log_probs.exp(), expected)
    torch.testing.assert_close(nonempty_probability(log_probs), expected[..., :2].sum(-1))


def test_uninformative_or_missing_modality_returns_lifecycle_prior() -> None:
    prior = torch.log_softmax(torch.tensor([[[2.0, -0.5, 0.1]]]), dim=-1)
    constant_opinions = torch.full((1, 1, 2, 3), 9.0)
    weights = torch.tensor([[[0.3, 0.2]]])
    fused = logarithmic_lifecycle_pool(prior, constant_opinions, weights)
    torch.testing.assert_close(fused, prior)

    informative = constant_opinions.clone()
    informative[..., 0, 0] += 4
    absent = torch.zeros((1, 1, 2), dtype=torch.bool)
    fused_absent = logarithmic_lifecycle_pool(prior, informative, weights, available=absent)
    torch.testing.assert_close(fused_absent, prior)


def test_reliability_simplex_renormalizes_only_over_present_modalities() -> None:
    available = torch.tensor([[True, False, True], [False, False, False]])
    weights = reliability_simplex(
        torch.tensor(2.0),
        torch.tensor([1.0, 4.0, 1.0]),
        available,
    )
    torch.testing.assert_close(weights.sum(-1), torch.ones(2))
    torch.testing.assert_close(weights[0], torch.tensor([0.5, 0.25, 0.0, 0.25]))
    torch.testing.assert_close(weights[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))


def test_expected_age_marginalizes_continue_and_birth() -> None:
    prior_age = torch.tensor([[10.0]])
    posterior = torch.tensor([[[0.6, 0.3, 0.1]]]).log()
    age = posterior_expected_age(prior_age, posterior, elapsed_time=2.0)
    torch.testing.assert_close(age, torch.tensor([[8.0]]))


def test_footprint_refinement_cannot_create_evidence() -> None:
    logits = torch.tensor([[[3.0, 0.0], [-1.0, 0.0]]])
    footprint = torch.tensor([[0.2, 0.8]])
    valid = torch.ones_like(footprint, dtype=torch.bool)
    original = footprint_evidence(logits, footprint, valid)

    split_logits = torch.tensor([[[3.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]])
    split_footprint = torch.tensor([[0.1, 0.1, 0.8]])
    split = footprint_evidence(split_logits, split_footprint, torch.ones(1, 3, dtype=torch.bool))
    torch.testing.assert_close(split.support, original.support)
    torch.testing.assert_close(
        split.robust_log_likelihood_ratio,
        original.robust_log_likelihood_ratio,
    )


def test_small_strong_region_is_not_erased_by_global_area_average() -> None:
    logits = torch.tensor([[[8.0, 0.0], [-0.1, 0.0]]])
    evidence = footprint_evidence(
        logits,
        torch.tensor([[0.02, 0.98]]),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert evidence.support.item() < 0.5
    assert evidence.robust_log_likelihood_ratio.item() > 0.0


def test_generalized_ci_counts_prior_once_and_is_split_invariant() -> None:
    prior_mean = torch.tensor([[[0.0, 0.0]]])
    prior_information = torch.eye(2).reshape(1, 1, 2, 2)
    observation = torch.tensor([[[[2.0, -1.0]]]])
    increment = (2 * torch.eye(2)).reshape(1, 1, 1, 2, 2)
    one = generalized_covariance_intersection(
        prior_mean,
        prior_information,
        observation,
        increment,
        torch.tensor([[[0.5, 0.5]]]),
    )
    duplicate = generalized_covariance_intersection(
        prior_mean,
        prior_information,
        observation.expand(1, 1, 2, 2),
        increment.expand(1, 1, 2, 2, 2),
        torch.tensor([[[0.5, 0.25, 0.25]]]),
    )
    torch.testing.assert_close(duplicate.information, one.information)
    torch.testing.assert_close(duplicate.mean, one.mean)

    bf16 = generalized_covariance_intersection(
        prior_mean.to(torch.bfloat16),
        prior_information.to(torch.bfloat16),
        observation.to(torch.bfloat16),
        increment.to(torch.bfloat16),
        torch.tensor([[[0.5, 0.5]]], dtype=torch.bfloat16),
    )
    assert bf16.mean.dtype == torch.bfloat16
    assert bf16.information.dtype == torch.bfloat16


def test_generalized_ci_zero_likelihood_preserves_a_singular_prior_exactly() -> None:
    prior_mean = torch.tensor([[[3.0, -7.0]]])
    prior_information = torch.zeros(1, 1, 2, 2)
    observations = torch.tensor([[[[100.0, 200.0]]]])
    increments = torch.zeros(1, 1, 1, 2, 2)
    fused = generalized_covariance_intersection(
        prior_mean,
        prior_information,
        observations,
        increments,
        torch.tensor([[[0.5, 0.5]]]),
    )
    torch.testing.assert_close(fused.mean, prior_mean)
    torch.testing.assert_close(fused.information, prior_information)


def test_generalized_ci_preserves_unobserved_nullspace_under_partial_evidence() -> None:
    prior_mean = torch.tensor([[[3.0, -7.0]]])
    prior_information = torch.diag(torch.tensor([1.0, 0.0])).reshape(1, 1, 2, 2)
    observations = torch.tensor([[[[5.0, 100.0]]]])
    increments = torch.diag(torch.tensor([1.0, 0.0])).reshape(1, 1, 1, 2, 2)
    fused = generalized_covariance_intersection(
        prior_mean,
        prior_information,
        observations,
        increments,
        torch.tensor([[[0.5, 0.5]]]),
    )
    torch.testing.assert_close(fused.mean[..., 0], torch.tensor([[11.0 / 3.0]]))
    torch.testing.assert_close(fused.mean[..., 1], prior_mean[..., 1])


def test_deterministic_ci_masks_missing_modalities_and_prefers_information() -> None:
    prior = torch.eye(2).reshape(1, 1, 2, 2)
    increments = torch.stack((0.1 * torch.eye(2), 4.0 * torch.eye(2)), dim=0).reshape(1, 1, 2, 2, 2)
    weights = deterministic_logdet_ci_weights(
        prior,
        increments,
        torch.tensor([[[True, True]]]),
        iterations=32,
    )
    torch.testing.assert_close(weights.sum(-1), torch.ones(1, 1))
    assert weights[..., 2].item() > weights[..., 1].item()

    missing = deterministic_logdet_ci_weights(
        prior,
        increments,
        torch.tensor([[[False, False]]]),
    )
    torch.testing.assert_close(missing, torch.tensor([[[1.0, 0.0, 0.0]]]))


def test_posterior_fusion_operators_fail_closed_on_invalid_numeric_inputs() -> None:
    with pytest.raises(ValueError, match="finite"):
        categorical_lifecycle_prior(torch.tensor([float("nan")]), torch.tensor([0.1]))
    with pytest.raises(TypeError, match="Python int"):
        deterministic_logdet_ci_weights(
            torch.eye(1).reshape(1, 1, 1, 1),
            torch.ones(1, 1, 1, 1, 1),
            torch.ones(1, 1, 1, dtype=torch.bool),
            iterations=True,
        )
    with pytest.raises(ValueError, match="finite"):
        footprint_evidence(
            torch.tensor([[[float("inf"), 0.0]]]),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        posterior_expected_age(
            torch.ones(1, 1),
            torch.log_softmax(torch.ones(1, 1, 3), dim=-1),
            elapsed_time=float("nan"),
        )


def test_shared_qk_coreference_supports_gqa_and_row_permutation() -> None:
    torch.manual_seed(4)
    queries = torch.randn(1, 4, 4, 3)  # three beliefs plus context
    keys = torch.randn(1, 5, 2, 3)  # two KV heads, repeated as in GQA
    footprint = torch.full((1, 5), 0.2)
    valid = torch.ones(1, 5, dtype=torch.bool)
    output = shared_qk_coreference(queries, keys, footprint, valid)
    torch.testing.assert_close(
        torch.cat(
            (output.evidence.responsibilities, output.evidence.context_probability.unsqueeze(-1)),
            dim=-1,
        ).sum(-1),
        torch.ones(1, 5),
    )

    permutation = torch.tensor([2, 0, 1])
    moved_queries = torch.cat((queries[:, permutation], queries[:, -1:]), dim=1)
    moved = shared_qk_coreference(moved_queries, keys, footprint, valid)
    torch.testing.assert_close(
        moved.evidence.responsibilities,
        output.evidence.responsibilities[..., permutation],
    )
    torch.testing.assert_close(
        moved.evidence.context_probability,
        output.evidence.context_probability,
    )


def test_relation_message_uses_existing_value_head_space() -> None:
    responsibilities = torch.tensor([[[0.75, 0.25], [0.25, 0.75]]])
    values = torch.tensor([[[[2.0], [10.0]], [[6.0], [14.0]]]])
    message = responsibility_weighted_message(
        responsibilities,
        values,
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    assert message.shape == (1, 2, 2, 1)
    torch.testing.assert_close(message[0, 0, :, 0], torch.tensor([3.0, 11.0]))
    torch.testing.assert_close(message[0, 1, :, 0], torch.tensor([5.0, 13.0]))


def test_grouped_relation_evidence_is_modality_typed_and_refinement_invariant() -> None:
    logits = torch.tensor([[[3.0, 0.0], [-1.0, 0.0]]])
    responsibilities = torch.softmax(logits, -1)[..., :1]
    values = torch.tensor([[[[2.0]], [[8.0]]]])
    original = grouped_relation_evidence(
        logits,
        responsibilities,
        values,
        torch.tensor([[0.2, 0.8]]),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1]]),
        modality_count=2,
    )
    assert original.message.shape == (1, 2, 1, 1, 1)
    torch.testing.assert_close(original.message[0, :, 0, 0, 0], torch.tensor([2.0, 8.0]))
    torch.testing.assert_close(original.available, torch.tensor([[True, True]]))

    split_logits = torch.tensor([[[3.0, 0.0], [3.0, 0.0], [-1.0, 0.0]]])
    split = grouped_relation_evidence(
        split_logits,
        torch.softmax(split_logits, -1)[..., :1],
        torch.tensor([[[[2.0]], [[2.0]], [[8.0]]]]),
        torch.tensor([[0.1, 0.1, 0.8]]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[0, 0, 1]]),
        modality_count=2,
    )
    torch.testing.assert_close(split.support, original.support)
    torch.testing.assert_close(
        split.robust_log_likelihood_ratio,
        original.robust_log_likelihood_ratio,
    )
    torch.testing.assert_close(split.message, original.message)


def test_same_entity_relation_preserves_modality_private_messages() -> None:
    logits = torch.tensor([[[8.0, 0.0], [8.0, 0.0]]])
    responsibilities = torch.softmax(logits, dim=-1)[..., :1]
    evidence = grouped_relation_evidence(
        logits,
        responsibilities,
        torch.tensor([[[[2.0], [-1.0]], [[9.0], [4.0]]]]),
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        torch.tensor([[0, 1]]),
        modality_count=2,
    )

    torch.testing.assert_close(responsibilities[:, 0], responsibilities[:, 1])
    assert responsibilities[0, 0, 0] > 0.99
    torch.testing.assert_close(
        evidence.message[0, 0, 0, :, 0],
        torch.tensor([2.0, -1.0]),
    )
    torch.testing.assert_close(
        evidence.message[0, 1, 0, :, 0],
        torch.tensor([9.0, 4.0]),
    )
    assert not torch.equal(evidence.message[0, 0], evidence.message[0, 1])


def test_relation_evidence_rejects_nonprobabilistic_responsibilities() -> None:
    with pytest.raises(ValueError, match="sub-probability"):
        responsibility_weighted_message(
            torch.tensor([[[1.1]]]),
            torch.ones(1, 1, 1, 1),
            torch.ones(1, 1),
            torch.ones(1, 1, dtype=torch.bool),
        )

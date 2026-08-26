from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from picf_next.unified.objective import combine_objective  # noqa: E402
from picf_next.unified.state import UnifiedBeliefState  # noqa: E402
from picf_next.unified.supervision import (  # noqa: E402
    BeliefSetTarget,
    belief_set_supervision_terms,
)
from picf_next.unified.temporal import assert_deploy_payload_is_causal  # noqa: E402


def _state() -> UnifiedBeliefState:
    return UnifiedBeliefState(
        content=torch.zeros(1, 2, 3),
        lifecycle_log_probs=torch.tensor(
            [[[0.90, 0.05, 0.05], [0.80, 0.10, 0.10]]],
        ).log(),
        geometry_mean=torch.tensor([[[0.0, 0.0], [2.0, 2.0]]]),
        geometry_information=torch.eye(2).expand(1, 2, 2, 2).clone(),
        geometry_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        content_log_variance=torch.zeros(1, 2, 1),
        expected_age=torch.ones(1, 2),
        evidence_age=torch.ones(1, 2),
    )


def _target() -> BeliefSetTarget:
    return BeliefSetTarget(
        sample_valid=torch.tensor([True]),
        exhaustive=torch.tensor([True]),
        object_valid=torch.tensor([[True, True]]),
        # Target order deliberately differs from belief-row order.
        geometry=torch.tensor([[[2.0, 2.0], [0.0, 0.0]]]),
        geometry_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        token_owner=torch.tensor(
            [
                [
                    [0.8, 0.0, 0.2],
                    [0.0, 0.7, 0.3],
                    [0.0, 0.0, 1.0],
                ]
            ]
        ),
        token_valid=torch.ones(1, 3, dtype=torch.bool),
    )


def test_set_supervision_is_row_permutation_invariant_and_soft_overlap_safe() -> None:
    state = _state()
    logits = torch.tensor(
        [[[-2.0, 4.0, 0.0], [4.0, -2.0, 0.0], [-3.0, -3.0, 4.0]]],
        requires_grad=True,
    )
    target = _target()
    objective = combine_objective(belief_set_supervision_terms(state, logits, target))
    assert objective.valid_counts == {
        "set/lifecycle": 2,
        "set/geometry": 2,
        "set/assignment": 3,
    }

    permutation = torch.tensor([1, 0])
    permuted_logits = torch.cat(
        (logits[..., :2].index_select(-1, permutation), logits[..., -1:]),
        dim=-1,
    )
    permuted = combine_objective(
        belief_set_supervision_terms(
            state.permute_rows(permutation),
            permuted_logits,
            target,
        )
    )
    torch.testing.assert_close(permuted.total, objective.total)
    for name in objective.normalized_terms:
        torch.testing.assert_close(
            permuted.normalized_terms[name],
            objective.normalized_terms[name],
        )
    objective.total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_unlabelled_sample_contributes_no_set_denominator() -> None:
    target = BeliefSetTarget(
        sample_valid=torch.tensor([False]),
        exhaustive=torch.tensor([False]),
        object_valid=torch.tensor([[False]]),
        geometry=torch.zeros(1, 1, 2),
        geometry_valid=torch.zeros(1, 1, 2, dtype=torch.bool),
        token_owner=torch.zeros(1, 3, 2),
        token_valid=torch.zeros(1, 3, dtype=torch.bool),
    )
    terms = belief_set_supervision_terms(
        _state(),
        torch.zeros(1, 3, 3),
        target,
    )
    objective = combine_objective(terms)
    assert objective.valid_counts == {
        "set/lifecycle": 0,
        "set/geometry": 0,
        "set/assignment": 0,
    }
    torch.testing.assert_close(objective.total, torch.tensor(0.0))
    with pytest.raises(ValueError, match="BeliefSetTarget"):
        assert_deploy_payload_is_causal({"metadata": target})


def test_set_target_fails_closed_on_invalid_probability_or_gradient() -> None:
    target = _target()
    with pytest.raises(ValueError, match="sum to one"):
        BeliefSetTarget(
            sample_valid=target.sample_valid,
            exhaustive=target.exhaustive,
            object_valid=target.object_valid,
            geometry=target.geometry,
            geometry_valid=target.geometry_valid,
            token_owner=target.token_owner * 0.5,
            token_valid=target.token_valid,
        )
    with pytest.raises(ValueError, match="stop-gradient"):
        BeliefSetTarget(
            sample_valid=target.sample_valid,
            exhaustive=target.exhaustive,
            object_valid=target.object_valid,
            geometry=target.geometry.clone().requires_grad_(),
            geometry_valid=target.geometry_valid,
            token_owner=target.token_owner,
            token_valid=target.token_valid,
        )


def test_partial_set_labels_never_supervise_unmatched_rows_as_empty() -> None:
    target = BeliefSetTarget(
        sample_valid=torch.tensor([True]),
        exhaustive=torch.tensor([False]),
        object_valid=torch.tensor([[True]]),
        geometry=torch.tensor([[[0.0, 0.0]]]),
        geometry_valid=torch.ones(1, 1, 2, dtype=torch.bool),
        # The second valid token is outside the partial annotation. It must not
        # become a false context/negative target until the set is exhaustive.
        token_owner=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        token_valid=torch.tensor([[True, True]]),
    )
    partial = combine_objective(
        belief_set_supervision_terms(
            _state(),
            torch.zeros(1, 2, 3),
            target,
        )
    )
    assert partial.valid_counts["set/lifecycle"] == 1
    assert partial.valid_counts["set/assignment"] == 1

    exhaustive = BeliefSetTarget(
        sample_valid=target.sample_valid,
        exhaustive=torch.tensor([True]),
        object_valid=target.object_valid,
        geometry=target.geometry,
        geometry_valid=target.geometry_valid,
        token_owner=target.token_owner,
        token_valid=target.token_valid,
    )
    complete = combine_objective(
        belief_set_supervision_terms(
            _state(),
            torch.zeros(1, 2, 3),
            exhaustive,
        )
    )
    assert complete.valid_counts["set/lifecycle"] == 2
    assert complete.valid_counts["set/assignment"] == 2

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from picf_next.models.discovery import ObjectExistenceCalibration  # noqa: E402
from tools.audit_molmoact2_m2_existence_ownership_coupling import (  # noqa: E402
    _couple_existence_and_ownership,
)


def test_neutral_physical_existence_posterior_is_an_exact_noop() -> None:
    calibration = ObjectExistenceCalibration(unmatched_query_weight=0.1)
    ownership_logits = torch.tensor([[[0.2, -0.4, 0.1], [1.0, 0.5, -0.5]]])
    existence_logits = torch.full(
        (1, 2),
        calibration.training_logit_at_half_posterior,
    )

    coupled, ownership = _couple_existence_and_ownership(
        ownership_logits,
        existence_logits,
        calibration,
    )

    assert torch.equal(coupled, ownership_logits)
    assert torch.equal(ownership, torch.softmax(ownership_logits, dim=-1))


def test_absent_query_loses_mass_to_supported_query_and_context() -> None:
    calibration = ObjectExistenceCalibration(unmatched_query_weight=0.1)
    ownership_logits = torch.zeros(1, 1, 3)
    physical_probabilities = torch.tensor([[0.01, 0.5]])
    posterior_log_odds = torch.logit(physical_probabilities)
    training_logits = posterior_log_odds - math.log(calibration.unmatched_query_weight)

    _coupled, ownership = _couple_existence_and_ownership(
        ownership_logits,
        training_logits,
        calibration,
    )

    assert ownership[0, 0, 0] < ownership[0, 0, 1]
    assert ownership[0, 0, 0] < ownership[0, 0, 2]
    assert ownership[0, 0, 1] == pytest.approx(ownership[0, 0, 2])


def test_ownership_likelihood_sends_gradient_to_existence_logit() -> None:
    calibration = ObjectExistenceCalibration(unmatched_query_weight=0.1)
    ownership_logits = torch.zeros(1, 1, 2)
    existence_logits = torch.tensor(
        [[calibration.training_logit_at_half_posterior]],
        requires_grad=True,
    )

    _coupled, ownership = _couple_existence_and_ownership(
        ownership_logits,
        existence_logits,
        calibration,
    )
    loss = -ownership[0, 0, 0].log()
    loss.backward()

    assert existence_logits.grad is not None
    assert existence_logits.grad.item() < 0.0


def test_query_permutation_equivariance_includes_existence_prior() -> None:
    calibration = ObjectExistenceCalibration(unmatched_query_weight=0.2)
    ownership_logits = torch.tensor([[[0.3, -0.7, 1.2, -0.2]]])
    existence_logits = torch.tensor([[0.1, 1.5, -0.4]])
    permutation = torch.tensor([2, 0, 1])

    coupled, ownership = _couple_existence_and_ownership(
        ownership_logits,
        existence_logits,
        calibration,
    )
    permuted_logits = torch.cat(
        (ownership_logits[..., :-1][..., permutation], ownership_logits[..., -1:]),
        dim=-1,
    )
    permuted_coupled, permuted_ownership = _couple_existence_and_ownership(
        permuted_logits,
        existence_logits[:, permutation],
        calibration,
    )

    assert torch.allclose(permuted_coupled[..., :-1], coupled[..., :-1][..., permutation])
    assert torch.allclose(permuted_coupled[..., -1], coupled[..., -1])
    assert torch.allclose(permuted_ownership[..., :-1], ownership[..., :-1][..., permutation])
    assert torch.allclose(permuted_ownership[..., -1], ownership[..., -1])


@pytest.mark.parametrize(
    ("ownership_shape", "existence_shape", "message"),
    [
        ((2, 3), (2, 1), "batch-by-token-by-category"),
        ((2, 3, 4), (2, 3, 1), "batch-by-query"),
        ((2, 3, 4), (1, 3), "batch sizes differ"),
        ((2, 3, 4), (2, 2), "queries plus context"),
    ],
)
def test_coupling_rejects_shape_drift(
    ownership_shape: tuple[int, ...],
    existence_shape: tuple[int, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _couple_existence_and_ownership(
            torch.zeros(ownership_shape),
            torch.zeros(existence_shape),
            ObjectExistenceCalibration(),
        )

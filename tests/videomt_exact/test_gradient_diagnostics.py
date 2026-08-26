from __future__ import annotations

import pytest
import torch

from picf_next.videomt_exact.gradient_diagnostics import objective_gradient_moments


def test_objective_gradient_moments_recovers_shared_surface_gram_matrix() -> None:
    surface = torch.tensor([[1.0, 2.0]], requires_grad=True)
    action = surface.sum()
    world = (2.0 * surface).sum()
    host = action + 0.1 * world
    source = -surface[0, 0] + surface[0, 1]

    moments = objective_gradient_moments(
        surface=surface,
        losses={
            "action": action,
            "host": host,
            "source": source,
            "world": world,
        },
    )

    assert moments.elements == 2
    assert moments.squared_norms["action"].item() == pytest.approx(2.0)
    assert moments.squared_norms["host"].item() == pytest.approx(2.88)
    assert moments.squared_norms["source"].item() == pytest.approx(2.0)
    assert moments.squared_norms["world"].item() == pytest.approx(8.0)
    assert moments.pairwise_dots["action__source"].item() == pytest.approx(0.0)
    assert moments.pairwise_dots["action__world"].item() == pytest.approx(4.0)
    assert moments.pairwise_dots["host__world"].item() == pytest.approx(4.8)
    assert surface.grad is None

    (action + world + source).backward()
    assert torch.equal(surface.grad, torch.tensor([[2.0, 4.0]]))


def test_objective_gradient_moments_rejects_detached_surface() -> None:
    surface = torch.ones(1, 2)
    losses = {
        name: torch.ones((), requires_grad=True) for name in ("action", "host", "source", "world")
    }
    with pytest.raises(RuntimeError, match="detached"):
        objective_gradient_moments(surface=surface, losses=losses)

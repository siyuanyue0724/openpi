"""Read-only gradient moments at VidEoMT's released prediction-query surface."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations

import torch

from picf_next.videomt_exact.lingbot_joint import CompleteNativeVidEoMTLingBotStep


@dataclass(frozen=True, slots=True)
class SharedQueryGradientMoments:
    """Local sufficient statistics for distributed objective-gradient alignment."""

    squared_norms: Mapping[str, torch.Tensor]
    pairwise_dots: Mapping[str, torch.Tensor]
    elements: int

    def __post_init__(self) -> None:
        names = tuple(sorted(self.squared_norms))
        expected_pairs = tuple(f"{left}__{right}" for left, right in combinations(names, 2))
        if names != ("action", "host", "source", "world"):
            raise ValueError("shared-query diagnostic objective inventory changed")
        if tuple(sorted(self.pairwise_dots)) != expected_pairs:
            raise ValueError("shared-query diagnostic pair inventory changed")
        values = (*self.squared_norms.values(), *self.pairwise_dots.values())
        if (
            isinstance(self.elements, bool)
            or not isinstance(self.elements, int)
            or self.elements <= 0
            or any(
                not isinstance(value, torch.Tensor)
                or value.shape != ()
                or value.dtype != torch.float64
                or not torch.isfinite(value)
                for value in values
            )
            or any(value < 0 for value in self.squared_norms.values())
        ):
            raise ValueError("shared-query gradient moments are invalid")


def objective_gradient_moments(
    *,
    surface: torch.Tensor,
    losses: Mapping[str, torch.Tensor],
) -> SharedQueryGradientMoments:
    """Differentiate four named objectives to one unchanged shared surface."""

    if surface.numel() <= 0 or not surface.requires_grad or not torch.isfinite(surface).all():
        raise RuntimeError("shared-query surface is empty, detached, or non-finite")
    if tuple(sorted(losses)) != ("action", "host", "source", "world"):
        raise RuntimeError("shared-query diagnostic objective inventory changed")
    if any(
        not isinstance(loss, torch.Tensor)
        or loss.shape != ()
        or not loss.requires_grad
        or not torch.isfinite(loss)
        for loss in losses.values()
    ):
        raise RuntimeError("shared-query diagnostic received an invalid objective")

    gradients: dict[str, torch.Tensor] = {}
    for name in sorted(losses):
        gradient = torch.autograd.grad(
            losses[name],
            surface,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        if (
            gradient is None
            or gradient.shape != surface.shape
            or not torch.isfinite(gradient).all()
        ):
            raise RuntimeError(f"{name} loss produced an invalid shared-query gradient")
        gradients[name] = gradient.detach().to(dtype=torch.float64)

    squared = {name: gradient.square().sum() for name, gradient in gradients.items()}
    dots = {
        f"{left}__{right}": (gradients[left] * gradients[right]).sum()
        for left, right in combinations(sorted(gradients), 2)
    }
    return SharedQueryGradientMoments(
        squared_norms=squared,
        pairwise_dots=dots,
        elements=surface.numel(),
    )


def shared_query_gradient_moments(
    result: CompleteNativeVidEoMTLingBotStep,
) -> SharedQueryGradientMoments:
    """Differentiate unchanged source/action/world losses to one shared donor surface.

    The surface is the exact tensor consumed by VidEoMT's released class and mask
    heads and, through a view, by the LingBot posterior projection.  Measuring here
    requires no parameter copy, learned head, loss reweighting, or optimizer update.
    """

    if not isinstance(result, CompleteNativeVidEoMTLingBotStep):
        raise TypeError("shared-query gradient diagnostic requires one native joint step")
    surface = result.source.current_output.prediction_query_surface
    if surface is None or not surface.requires_grad:
        raise RuntimeError("released prediction-query surface is absent or detached")
    metrics = result.policy.backend_metrics
    if metrics is None or "loss_world" not in metrics:
        raise RuntimeError("shared-query gradient diagnostic requires WLA action and world losses")
    world_loss = metrics["loss_world"]
    return objective_gradient_moments(
        surface=surface,
        losses={
            "action": result.policy.official_action_loss,
            "host": result.policy.official_total_loss,
            "source": result.source.source_objective.total,
            "world": world_loss,
        },
    )

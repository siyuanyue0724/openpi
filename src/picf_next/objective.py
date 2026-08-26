"""Valid-support-normalized objective primitives shared by independent arms."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch


@dataclass(frozen=True, slots=True)
class ObjectiveSupport:
    """Detached normalization support for an objective term not yet materialized."""

    name: str
    valid: torch.Tensor
    weight: float
    sample_weight: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("objective support name must be non-empty")
        if self.valid.dtype != torch.bool or self.valid.requires_grad:
            raise ValueError("objective support validity must be detached boolean")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("objective support weight must be real-valued")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError("objective support weight must be finite and non-negative")
        if self.sample_weight is not None:
            if (
                self.sample_weight.shape != self.valid.shape
                or not self.sample_weight.is_floating_point()
                or self.sample_weight.device != self.valid.device
            ):
                raise ValueError("objective support weights must match validity")
            if (
                self.sample_weight.requires_grad
                or not torch.isfinite(self.sample_weight).all()
                or (self.sample_weight < 0).any()
                or (self.sample_weight.masked_select(~self.valid) != 0).any()
            ):
                raise ValueError(
                    "objective support weights must be detached, finite, non-negative "
                    "and zero when invalid"
                )

    @property
    def valid_count(self) -> int:
        return int(self.valid.sum().detach().to(device="cpu", dtype=torch.int64).item())

    def denominator(self) -> float:
        if self.sample_weight is None:
            return float(self.valid_count)
        return float(
            self.sample_weight.masked_fill(~self.valid, 0)
            .sum()
            .detach()
            .to(device="cpu", dtype=torch.float64)
            .item()
        )


@dataclass(frozen=True, slots=True)
class ObjectiveTerm:
    name: str
    values: torch.Tensor
    valid: torch.Tensor
    weight: float
    sample_weight: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("objective term name must be non-empty")
        if self.valid.shape != self.values.shape or self.valid.dtype != torch.bool:
            raise ValueError("objective valid mask must be boolean and match values")
        if self.valid.device != self.values.device:
            raise ValueError("objective values and validity must share one device")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise TypeError("objective weight must be real-valued")
        if self.weight < 0 or not math.isfinite(self.weight):
            raise ValueError("objective weight must be finite and non-negative")
        if not self.values.is_floating_point():
            raise TypeError("objective values must be floating point")
        if not torch.isfinite(self.values).all():
            raise ValueError("objective values must be finite")
        if self.sample_weight is not None:
            if (
                self.sample_weight.shape != self.values.shape
                or not self.sample_weight.is_floating_point()
                or self.sample_weight.device != self.values.device
            ):
                raise ValueError("objective sample weights must be floating point and match values")
            if (
                not torch.isfinite(self.sample_weight).all()
                or (self.sample_weight < 0).any()
                or (self.sample_weight.masked_select(~self.valid) != 0).any()
                or self.sample_weight.requires_grad
            ):
                raise ValueError(
                    "objective sample weights must be detached, finite, non-negative "
                    "and zero when invalid"
                )

    def normalized(self) -> torch.Tensor:
        if self.sample_weight is None:
            denominator = self.valid.sum().to(self.values.dtype)
            numerator = self.values.masked_fill(~self.valid, 0).sum()
        else:
            effective_weight = self.sample_weight.masked_fill(~self.valid, 0)
            denominator = effective_weight.sum()
            numerator = (self.values * effective_weight).sum()
        return numerator / denominator.clamp_min(torch.finfo(numerator.dtype).tiny)

    def support(self) -> ObjectiveSupport:
        return ObjectiveSupport(
            name=self.name,
            valid=self.valid.detach(),
            weight=self.weight,
            sample_weight=(
                None if self.sample_weight is None else self.sample_weight.detach()
            ),
        )


@dataclass(frozen=True, slots=True)
class UnifiedObjective:
    total: torch.Tensor
    normalized_terms: dict[str, torch.Tensor]
    valid_counts: dict[str, int]
    family_terms: dict[str, torch.Tensor] = field(default_factory=dict)


def combine_objective(terms: tuple[ObjectiveTerm, ...]) -> UnifiedObjective:
    if not terms:
        raise ValueError("at least one objective term is required")
    names = [term.name for term in terms]
    if len(set(names)) != len(names):
        raise ValueError("objective term names must be unique")
    device = terms[0].values.device
    if any(term.values.device != device or term.valid.device != device for term in terms):
        raise ValueError("all objective terms must share one device")
    normalized = {term.name: term.normalized() for term in terms}
    total = sum((normalized[term.name] * term.weight for term in terms), terms[0].values.sum() * 0)
    packed_counts = torch.stack(tuple(term.valid.sum() for term in terms))
    valid_counts = packed_counts.detach().to(device="cpu", dtype=torch.int64).tolist()
    return UnifiedObjective(
        total=total,
        normalized_terms=normalized,
        valid_counts={
            term.name: int(count) for term, count in zip(terms, valid_counts, strict=True)
        },
    )


def normalized_scalar_term(name: str, value: torch.Tensor, *, weight: float = 1.0) -> ObjectiveTerm:
    """Wrap one already-normalized host loss without re-normalizing it."""

    if value.ndim != 0:
        raise ValueError("an official normalized loss must be scalar")
    return ObjectiveTerm(
        name=name,
        values=value.reshape(1),
        valid=torch.ones(1, dtype=torch.bool, device=value.device),
        weight=weight,
    )

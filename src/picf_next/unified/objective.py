"""Compatibility surface for the historical unified-arm objective."""

from __future__ import annotations

from dataclasses import dataclass

from picf_next.objective import (
    ObjectiveTerm,
    UnifiedObjective,
    combine_objective,
    normalized_scalar_term,
)

__all__ = [
    "DeclaredUnifiedObjective",
    "ObjectiveTerm",
    "UnifiedObjective",
    "combine_declared_objective",
    "combine_objective",
    "normalized_scalar_term",
]


@dataclass(frozen=True, slots=True)
class DeclaredUnifiedObjective:
    """The single owner-approved PICF training law.

    LingBot's action loss is mandatory. Host-native regularizers remain
    explicit rather than being hidden inside an opaque total, while each PICF
    family can be absent when its typed supervision is unavailable.
    """

    action: ObjectiveTerm
    host_regularization: tuple[ObjectiveTerm, ...] = ()
    set_supervision: tuple[ObjectiveTerm, ...] = ()
    cross_modal_prediction: tuple[ObjectiveTerm, ...] = ()
    future_prediction: tuple[ObjectiveTerm, ...] = ()
    overshooting: tuple[ObjectiveTerm, ...] = ()

    def __post_init__(self) -> None:
        if self.action.name != "action" or self.action.weight != 1.0:
            raise ValueError("the mandatory action term must be named action with unit weight")
        if int(self.action.valid.sum().item()) == 0:
            raise ValueError("the mandatory action term requires at least one valid element")
        families = (
            ("host/", self.host_regularization),
            ("set/", self.set_supervision),
            ("xmod/", self.cross_modal_prediction),
            ("future/", self.future_prediction),
            ("over/", self.overshooting),
        )
        for prefix, terms in families:
            if any(not term.name.startswith(prefix) for term in terms):
                raise ValueError(f"objective terms in this family must start with {prefix}")

    @property
    def terms(self) -> tuple[ObjectiveTerm, ...]:
        return (
            self.action,
            *self.host_regularization,
            *self.set_supervision,
            *self.cross_modal_prediction,
            *self.future_prediction,
            *self.overshooting,
        )


def combine_declared_objective(declaration: DeclaredUnifiedObjective) -> UnifiedObjective:
    """Combine the declared law while retaining every family as a named term."""

    return combine_objective(declaration.terms)

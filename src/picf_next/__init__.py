"""Host-neutral PICF-Next public API.

The legacy probabilistic utilities remain import-compatible, but they are
loaded only when a caller requests one of their public names.  This keeps the
LingBot-native runtime from importing historical association and lifecycle
code as a side effect of importing ``picf_next``.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AssociationResult": ("picf_next.association", "AssociationResult"),
    "BIRTH_EVENT": ("picf_next.posterior", "BIRTH_EVENT"),
    "DEATH_EVENT": ("picf_next.posterior", "DEATH_EVENT"),
    "DenseEvidence": ("picf_next.contracts", "DenseEvidence"),
    "GaussianCorrection": ("picf_next.posterior", "GaussianCorrection"),
    "LifecycleAssociationResult": ("picf_next.association", "LifecycleAssociationResult"),
    "MATCH_EVENT": ("picf_next.posterior", "MATCH_EVENT"),
    "MISS_EVENT": ("picf_next.posterior", "MISS_EVENT"),
    "ObjectAssociationProblem": ("picf_next.posterior", "ObjectAssociationProblem"),
    "ObjectBeliefSet": ("picf_next.contracts", "ObjectBeliefSet"),
    "ObjectObservationSet": ("picf_next.contracts", "ObjectObservationSet"),
    "PICFContext": ("picf_next.contracts", "PICFContext"),
    "PhysicalGeometryContract": ("picf_next.geometry", "PhysicalGeometryContract"),
    "PosteriorCapacityError": ("picf_next.posterior", "PosteriorCapacityError"),
    "PosteriorUpdate": ("picf_next.posterior", "PosteriorUpdate"),
    "UNUSED_EVENT": ("picf_next.posterior", "UNUSED_EVENT"),
    "associate": ("picf_next.association", "associate"),
    "associate_lifecycle": ("picf_next.association", "associate_lifecycle"),
    "build_object_association_problem": (
        "picf_next.posterior",
        "build_object_association_problem",
    ),
    "diagonal_gaussian_correct": ("picf_next.posterior", "diagonal_gaussian_correct"),
    "diagonal_gaussian_predict": ("picf_next.posterior", "diagonal_gaussian_predict"),
    "pack_dynamic": ("picf_next.posterior", "pack_dynamic"),
    "pack_state": ("picf_next.posterior", "pack_state"),
    "pairwise_cosine_distance": ("picf_next.posterior", "pairwise_cosine_distance"),
    "pairwise_diagonal_gaussian_nll": (
        "picf_next.posterior",
        "pairwise_diagonal_gaussian_nll",
    ),
    "predict_object_belief": ("picf_next.posterior", "predict_object_belief"),
    "update_object_posterior": ("picf_next.posterior", "update_object_posterior"),
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute_name = _LAZY_EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))

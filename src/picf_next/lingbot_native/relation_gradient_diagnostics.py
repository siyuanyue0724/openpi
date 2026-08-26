"""Read-only objective-component gradients on existing relation surfaces."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

LEGACY_RELATION_SURFACE_GRADIENT_KEYS = (
    "task@match_embeddings",
    "task_dense@match_embeddings",
    "task_dense@row_embeddings",
    "ownership@row_embeddings",
)

LEGACY_RELATION_SURFACE_GRADIENT_PAIRS = (
    "task__task_dense@match_embeddings",
    "task_dense__ownership@row_embeddings",
)

FACTORIZED_RELATION_SURFACE_GRADIENT_KEYS = (
    "task_row@match_embeddings",
    "ownership@row_embeddings",
)

FACTORIZED_RELATION_SURFACE_GRADIENT_PAIRS: tuple[str, ...] = ()

# Backward-compatible names for readers of legacy reports.
RELATION_SURFACE_GRADIENT_KEYS = LEGACY_RELATION_SURFACE_GRADIENT_KEYS
RELATION_SURFACE_GRADIENT_PAIRS = LEGACY_RELATION_SURFACE_GRADIENT_PAIRS


def relation_surface_gradient_contract(
    names: object,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    """Resolve the exact legacy or factorized diagnostic geometry from its keys."""

    if isinstance(names, (str, bytes)) or not isinstance(names, Iterable):
        raise RuntimeError("relation-surface gradient keys are not iterable")
    observed = set(names)
    if any(not isinstance(name, str) for name in observed):
        raise RuntimeError("relation-surface gradient keys must be strings")
    if observed == set(LEGACY_RELATION_SURFACE_GRADIENT_KEYS):
        return LEGACY_RELATION_SURFACE_GRADIENT_KEYS, (
            (
                "task__task_dense@match_embeddings",
                "task@match_embeddings",
                "task_dense@match_embeddings",
            ),
            (
                "task_dense__ownership@row_embeddings",
                "task_dense@row_embeddings",
                "ownership@row_embeddings",
            ),
        )
    if observed == set(FACTORIZED_RELATION_SURFACE_GRADIENT_KEYS):
        return FACTORIZED_RELATION_SURFACE_GRADIENT_KEYS, ()
    raise RuntimeError("relation-surface gradient probe coverage is incomplete")


def relation_surface_component_gradients(
    result: Any,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Differentiate existing structural terms only to their final relation surfaces."""

    relation = result.final_relation
    structural_terms = result.objective.structural_terms
    term_by_name = {term.name: term for term in structural_terms}
    if len(term_by_name) != len(structural_terms):
        raise RuntimeError("native structural objective repeats a component name")
    factorized = "set/task_row" in term_by_name
    component_terms = (
        {
            "task_row": "set/task_row",
            "ownership": "set/ownership",
        }
        if factorized
        else {
            "task": "set/task",
            "task_dense": "set/task_dense",
            "ownership": "set/ownership",
        }
    )
    if not set(component_terms.values()) <= set(term_by_name):
        raise RuntimeError("native structural objective omitted a relation-gradient component")
    if factorized:
        dense_term = term_by_name.get("set/task_dense")
        if dense_term is not None and dense_term.weight != 0:
            raise RuntimeError(
                "factorized relation objective retained an independent dense-task weight"
            )
    normalized_terms = result.objective.objective.normalized_terms
    if not set(component_terms.values()) <= set(normalized_terms):
        raise RuntimeError("native objective omitted a normalized relation-gradient component")

    surfaces = {
        "match_embeddings": relation.match_embeddings,
        "row_embeddings": relation.row_embeddings,
    }
    if any(
        not torch_module.is_tensor(surface) or surface.numel() <= 0 or not surface.requires_grad
        for surface in surfaces.values()
    ):
        raise RuntimeError("final relation gradient surface is detached or empty")
    requested = (
        {
            "task_row": ("match_embeddings",),
            "ownership": ("row_embeddings",),
        }
        if factorized
        else {
            "task": ("match_embeddings",),
            "task_dense": ("match_embeddings", "row_embeddings"),
            "ownership": ("row_embeddings",),
        }
    )
    gradients: dict[str, Any] = {}
    for component, surface_names in requested.items():
        term_name = component_terms[component]
        term = term_by_name[term_name]
        scalar = normalized_terms[term_name] * term.weight
        if (
            not torch_module.is_tensor(scalar)
            or scalar.ndim != 0
            or not scalar.requires_grad
            or not bool(torch_module.isfinite(scalar).item())
        ):
            raise RuntimeError(f"{term_name} is not a finite attached diagnostic scalar")
        selected_surfaces = tuple(surfaces[name] for name in surface_names)
        measured = torch_module.autograd.grad(
            scalar,
            selected_surfaces,
            retain_graph=True,
            allow_unused=False,
        )
        for surface_name, _surface, gradient in zip(
            surface_names,
            selected_surfaces,
            measured,
            strict=True,
        ):
            if gradient is None:
                raise RuntimeError(f"{term_name} did not reach its declared {surface_name} surface")
            if not bool(torch_module.isfinite(gradient).all().item()):
                raise FloatingPointError(
                    f"{term_name} produced a non-finite {surface_name} gradient"
                )
            gradients[f"{component}@{surface_name}"] = gradient.detach().float().clone()
    expected, _pairs = relation_surface_gradient_contract(gradients)
    if tuple(gradients) != expected:
        raise RuntimeError("relation-surface gradient probe produced another component order")
    return gradients

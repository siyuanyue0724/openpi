"""Canonical loss-side identity-to-row binding contracts.

Bindings remove a supervision permutation ambiguity inside one episode. They
are labels, not model inputs or inference state.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

RowBindings = tuple[tuple[str, int], ...]


def normalize_row_bindings(
    value: Mapping[str, int] | Sequence[tuple[str, int]],
    *,
    capacity: int,
) -> RowBindings:
    """Validate and canonicalize one episode's loss-side row gauge."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("row-binding capacity must be a positive integer")
    items = tuple(value.items()) if isinstance(value, Mapping) else tuple(value)
    normalized: list[tuple[str, int]] = []
    identities: set[str] = set()
    rows: set[int] = set()
    for item in items:
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError("row bindings must contain identity-row pairs")
        identity, row = item
        if not isinstance(identity, str) or not identity:
            raise ValueError("row-binding identities must be non-empty strings")
        if isinstance(row, bool) or not isinstance(row, int) or not 0 <= row < capacity:
            raise ValueError("row-binding indexes must lie inside capacity")
        if identity in identities:
            raise ValueError("row bindings contain a duplicate identity")
        if row in rows:
            raise ValueError("row bindings contain a duplicate row")
        identities.add(identity)
        rows.add(row)
        normalized.append((identity, row))
    return tuple(sorted(normalized))


def row_binding_map(bindings: RowBindings, *, capacity: int) -> dict[str, int]:
    """Return a mutable copy after revalidating its canonical contract."""

    canonical = normalize_row_bindings(bindings, capacity=capacity)
    if canonical != bindings:
        raise ValueError("row bindings are not in canonical order")
    return dict(canonical)

"""Fail-closed loss-side resolution for one task-address entity row."""

from __future__ import annotations

from collections.abc import Sequence


def resolve_task_address_target_row(
    *,
    target_identity: str | None,
    identity_keys: Sequence[str],
    eligible_track_indices: Sequence[int],
    bindings: Sequence[tuple[str, int]],
    allow_unobservable: bool,
) -> tuple[int | None, str]:
    """Resolve one target row without inventing current-frame visibility evidence."""

    if target_identity is None:
        return None, "no-singleton-source-target"
    if not isinstance(target_identity, str) or not target_identity:
        raise TypeError("task-address target identity must be a non-empty string or absent")
    normalized_identities = tuple(identity_keys)
    if (
        not normalized_identities
        or len(set(normalized_identities)) != len(normalized_identities)
        or any(not isinstance(value, str) or not value for value in normalized_identities)
    ):
        raise ValueError("task-address physical inventory identities are invalid")
    binding_map = dict(bindings)
    if target_identity in binding_map:
        return int(binding_map[target_identity]), "bound-current-frame-target"
    if target_identity not in normalized_identities:
        raise RuntimeError(
            f"task-address target identity is absent from inventory: {target_identity}"
        )
    target_track = normalized_identities.index(target_identity)
    eligible = {int(value) for value in eligible_track_indices}
    if target_track in eligible:
        raise RuntimeError(
            "task-address eligible target identity is absent from row bindings: "
            f"{target_identity}"
        )
    if not allow_unobservable:
        raise RuntimeError(f"task-address target identity is unbound: {target_identity}")
    return None, "unobservable-current-frame-target"

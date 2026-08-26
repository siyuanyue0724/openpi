from __future__ import annotations

import pytest

from picf_next.lingbot_native.task_address_target import resolve_task_address_target_row


def test_resolves_bound_and_unobservable_rows_without_fabricating_visibility() -> None:
    identities = ("visible", "hidden")
    bindings = (("visible", 3),)

    assert resolve_task_address_target_row(
        target_identity="visible",
        identity_keys=identities,
        eligible_track_indices=(0,),
        bindings=bindings,
        allow_unobservable=True,
    ) == (3, "bound-current-frame-target")
    assert resolve_task_address_target_row(
        target_identity="hidden",
        identity_keys=identities,
        eligible_track_indices=(0,),
        bindings=bindings,
        allow_unobservable=True,
    ) == (None, "unobservable-current-frame-target")
    assert resolve_task_address_target_row(
        target_identity=None,
        identity_keys=identities,
        eligible_track_indices=(0,),
        bindings=bindings,
        allow_unobservable=True,
    ) == (None, "no-singleton-source-target")


def test_rejects_absent_or_eligible_but_unbound_targets() -> None:
    with pytest.raises(RuntimeError, match="absent from inventory"):
        resolve_task_address_target_row(
            target_identity="missing",
            identity_keys=("visible",),
            eligible_track_indices=(0,),
            bindings=(("visible", 2),),
            allow_unobservable=True,
        )
    with pytest.raises(RuntimeError, match="eligible target identity"):
        resolve_task_address_target_row(
            target_identity="visible",
            identity_keys=("visible",),
            eligible_track_indices=(0,),
            bindings=(),
            allow_unobservable=True,
        )
    with pytest.raises(RuntimeError, match="target identity is unbound"):
        resolve_task_address_target_row(
            target_identity="hidden",
            identity_keys=("visible", "hidden"),
            eligible_track_indices=(0,),
            bindings=(("visible", 2),),
            allow_unobservable=False,
        )

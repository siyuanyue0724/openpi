"""Loss/evaluation-only CALVIN task-to-physical-entity protocol.

This module is deliberately outside model inputs, recurrent state and data
selection.  It turns the pinned CALVIN task key into an independently specified
physical entity set only after a deploy-visible forward has completed.  The
protocol may provide an optional exact task-grounding loss or make causal action
interventions identifiable; it must never select production optimization
samples or enter inference.  A provenance-bound diagnostic that cannot publish
a checkpoint may use exactness only to preregister an auditable evaluation
stratum before loading model weights.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

CALVIN_TASK_CONFIG_SHA256 = "6e905de3ca05118efdd8a51f8a7756ec6e61ffdb2b9b6a2843f0b7e0e9e51dcf"
CALVIN_SCENE_CONFIG_SHA256 = {
    "calvin_scene_A": "e91f76b7af0950828ad9bc426768c4600c8badcdbd41a719dcce326e12e7b05d",
    "calvin_scene_B": "7577625ef8dc40918936875697806b3dc1fa53a7ba59c4d0950cd2937a47baa0",
    "calvin_scene_C": "a7cf204d27465ae80ec41aaa3a62ea4fc30346dda8413d6702969fc1e70a7ddc",
    "calvin_scene_D": "f4515e439dd6b2edb55369ed421d2e47088543655970e049e5ffe760c1448c11",
}
CALVIN_TABLE_URDF_SHA256 = {
    "calvin_table_A": "b2b84edb36450bc7cf807b2c6c7e9a7838aa03d17951b9188bd739b07a233bc9",
    "calvin_table_B": "c92dab7a62cb042aec29b8de3976a6483ece9ca53fc7ae4018a4ddc4c89aa2d6",
    "calvin_table_C": "ed191f3f480b9bf8319f7111ac39fd444f62b21b27279b2501e8875c90fb64fe",
    "calvin_table_D": "b1068f81ec0af69cc784e9d42610fb83ea63ec150997d74a8a9b4e2c87d057f3",
}
CALVIN_TASK_PROTOCOL_SOURCE_SHA256 = {
    "calvin_env/calvin_env/envs/tasks.py": (
        "bca84af6249b2fd2404d1bd17318c7f3cf640acc329d3a26306b29149fc00e1a"
    ),
    **{
        f"calvin_env/conf/scene/{scene}.yaml": digest
        for scene, digest in CALVIN_SCENE_CONFIG_SHA256.items()
    },
    **{
        f"calvin_env/data/{table}/urdf/{table}.urdf": digest
        for table, digest in CALVIN_TABLE_URDF_SHA256.items()
    },
    "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml": (
        CALVIN_TASK_CONFIG_SHA256
    ),
}


@dataclass(frozen=True, slots=True)
class CalvinTaskPhysicalRelevance:
    """Physical participants justified by one pinned CALVIN task definition."""

    task_key: str
    action_target_identity_keys: tuple[str, ...]
    outcome_identity_keys: tuple[str, ...] = ()
    known_participant_identity_keys: tuple[str, ...] = ()
    exact_action_target: bool = True
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        groups = (
            self.action_target_identity_keys,
            self.outcome_identity_keys,
            self.known_participant_identity_keys,
        )
        if not isinstance(self.task_key, str) or not self.task_key:
            raise ValueError("CALVIN task relevance requires a nonempty task key")
        if any(
            not isinstance(group, tuple)
            or any(not isinstance(key, str) or not key for key in group)
            or len(set(group)) != len(group)
            for group in groups
        ):
            raise ValueError("CALVIN task relevance identity groups must be unique strings")
        if self.exact_action_target:
            if not self.action_target_identity_keys or self.exclusion_reason is not None:
                raise ValueError("exact CALVIN tasks require targets and no exclusion reason")
        elif self.action_target_identity_keys or not self.exclusion_reason:
            raise ValueError("inexact CALVIN tasks must fail closed with an explicit reason")


@dataclass(frozen=True, slots=True)
class CalvinHiddenTaskRowSelection:
    """Loss-side attribution of witnessed, currently hidden task rows."""

    task_key: str
    action_target_identity_keys: tuple[str, ...]
    row_indices: tuple[int, ...]
    row_identity_keys: tuple[str, ...]
    exact_action_target: bool
    reason: str

    @property
    def eligible(self) -> bool:
        return bool(self.row_indices)


@dataclass(frozen=True, slots=True)
class CalvinWitnessedTaskRowSelection:
    """Loss-side attribution of task targets witnessed before an intervention."""

    task_key: str
    action_target_identity_keys: tuple[str, ...]
    row_indices: tuple[int, ...]
    row_identity_keys: tuple[str, ...]
    exact_action_target: bool
    reason: str

    @property
    def eligible(self) -> bool:
        return bool(self.row_indices)


def _exact(
    task_key: str,
    *action_targets: str,
    outcomes: tuple[str, ...] = (),
) -> CalvinTaskPhysicalRelevance:
    return CalvinTaskPhysicalRelevance(
        task_key=task_key,
        action_target_identity_keys=tuple(action_targets),
        outcome_identity_keys=outcomes,
    )


def _inexact(
    task_key: str,
    *,
    known_participants: tuple[str, ...],
    reason: str,
) -> CalvinTaskPhysicalRelevance:
    return CalvinTaskPhysicalRelevance(
        task_key=task_key,
        action_target_identity_keys=(),
        known_participant_identity_keys=known_participants,
        exact_action_target=False,
        exclusion_reason=reason,
    )


_BLOCK_RED = "movable/block_red"
_BLOCK_BLUE = "movable/block_blue"
_BLOCK_PINK = "movable/block_pink"
_SLIDER_DOOR = "part/table/slide_link"
_SLIDER_SURFACE = "part/table/plank_link"
_DRAWER = "part/table/drawer_link"
_BUTTON = "part/table/button_link"
_LED = "part/table/led_link"
_SWITCH = "part/table/switch_link"
_LIGHT = "part/table/light_link"


# This is a benchmark evaluator protocol, not a runtime object vocabulary.  The
# entries are an explicit review surface for the task definitions pinned by
# ``CALVIN_TASK_CONFIG_SHA256``.  Ambiguous predicates fail closed rather than
# guessing a currently manipulated block from language or scene heuristics.
_CALVIN_TASK_RELEVANCE = {
    # Rotation.
    "rotate_red_block_right": _exact("rotate_red_block_right", _BLOCK_RED),
    "rotate_red_block_left": _exact("rotate_red_block_left", _BLOCK_RED),
    "rotate_blue_block_right": _exact("rotate_blue_block_right", _BLOCK_BLUE),
    "rotate_blue_block_left": _exact("rotate_blue_block_left", _BLOCK_BLUE),
    "rotate_pink_block_right": _exact("rotate_pink_block_right", _BLOCK_PINK),
    "rotate_pink_block_left": _exact("rotate_pink_block_left", _BLOCK_PINK),
    # Pushing.
    "push_red_block_right": _exact("push_red_block_right", _BLOCK_RED),
    "push_red_block_left": _exact("push_red_block_left", _BLOCK_RED),
    "push_blue_block_right": _exact("push_blue_block_right", _BLOCK_BLUE),
    "push_blue_block_left": _exact("push_blue_block_left", _BLOCK_BLUE),
    "push_pink_block_right": _exact("push_pink_block_right", _BLOCK_PINK),
    "push_pink_block_left": _exact("push_pink_block_left", _BLOCK_PINK),
    # Articulated fixtures.
    "move_slider_left": _exact("move_slider_left", _SLIDER_DOOR),
    "move_slider_right": _exact("move_slider_right", _SLIDER_DOOR),
    "open_drawer": _exact("open_drawer", _DRAWER),
    "close_drawer": _exact("close_drawer", _DRAWER),
    # Lifting.  The support surface is a predicate condition, while the block is
    # the physical entity whose hidden state determines the demonstrated action.
    "lift_red_block_table": _exact("lift_red_block_table", _BLOCK_RED),
    "lift_red_block_slider": _exact("lift_red_block_slider", _BLOCK_RED),
    "lift_red_block_drawer": _exact("lift_red_block_drawer", _BLOCK_RED),
    "lift_blue_block_table": _exact("lift_blue_block_table", _BLOCK_BLUE),
    "lift_blue_block_slider": _exact("lift_blue_block_slider", _BLOCK_BLUE),
    "lift_blue_block_drawer": _exact("lift_blue_block_drawer", _BLOCK_BLUE),
    "lift_pink_block_table": _exact("lift_pink_block_table", _BLOCK_PINK),
    "lift_pink_block_slider": _exact("lift_pink_block_slider", _BLOCK_PINK),
    "lift_pink_block_drawer": _exact("lift_pink_block_drawer", _BLOCK_PINK),
    # The action targets the control; the light is an independently useful
    # outcome identity but is not selected for the hidden-action-row gate.
    "turn_on_lightbulb": _exact("turn_on_lightbulb", _SWITCH, outcomes=(_LIGHT,)),
    "turn_off_lightbulb": _exact("turn_off_lightbulb", _SWITCH, outcomes=(_LIGHT,)),
    "turn_on_led": _exact("turn_on_led", _BUTTON, outcomes=(_LED,)),
    "turn_off_led": _exact("turn_off_led", _BUTTON, outcomes=(_LED,)),
    # These predicates do not identify every directly manipulated block from
    # task identity alone.  Known destinations are recorded for audit prose but
    # are not sufficient to authorize a strict row intervention.
    "place_in_slider": _inexact(
        "place_in_slider",
        known_participants=(_SLIDER_SURFACE,),
        reason="held block is not identified by the task definition",
    ),
    "place_in_drawer": _inexact(
        "place_in_drawer",
        known_participants=(_DRAWER,),
        reason="held block is not identified by the task definition",
    ),
    "stack_block": _inexact(
        "stack_block",
        known_participants=(_BLOCK_RED, _BLOCK_BLUE, _BLOCK_PINK),
        reason="ordered source and destination blocks are state dependent",
    ),
    "unstack_block": _inexact(
        "unstack_block",
        known_participants=(_BLOCK_RED, _BLOCK_BLUE, _BLOCK_PINK),
        reason="ordered source and destination blocks are state dependent",
    ),
    "push_into_drawer": _inexact(
        "push_into_drawer",
        known_participants=(_BLOCK_RED, _BLOCK_BLUE, _BLOCK_PINK, _DRAWER),
        reason="the task definition permits any one of three source blocks",
    ),
}


def calvin_task_physical_relevance(task_key: str) -> CalvinTaskPhysicalRelevance:
    """Return the frozen evaluator semantics for one official task key."""

    if not isinstance(task_key, str) or not task_key:
        raise ValueError("CALVIN task key must be a nonempty string")
    try:
        return _CALVIN_TASK_RELEVANCE[task_key]
    except KeyError as error:
        raise KeyError(
            f"task key is absent from the pinned CALVIN protocol: {task_key!r}"
        ) from error


def calvin_task_physical_relevance_inventory() -> tuple[CalvinTaskPhysicalRelevance, ...]:
    """Return the complete frozen protocol in canonical task-key order."""

    return tuple(_CALVIN_TASK_RELEVANCE[key] for key in sorted(_CALVIN_TASK_RELEVANCE))


def calvin_exact_task_loss_identities(task_key: str) -> tuple[str, ...] | None:
    """Return exact post-forward loss identities, or unknown for ambiguous tasks."""

    relevance = calvin_task_physical_relevance(task_key)
    if not relevance.exact_action_target:
        return None
    return relevance.action_target_identity_keys


def validate_calvin_task_protocol_source(path: str | Path) -> str:
    """Fail if the reviewed official CALVIN task configuration has drifted."""

    source = Path(path).expanduser().resolve()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != CALVIN_TASK_CONFIG_SHA256:
        raise ValueError(
            "official CALVIN task configuration differs from the reviewed evaluator protocol"
        )
    return digest


def validate_calvin_task_protocol_sources(calvin_checkout: str | Path) -> dict[str, str]:
    """Validate every official source used to interpret task participants."""

    root = Path(calvin_checkout).expanduser().resolve()
    observed = {}
    for relative_path, expected in CALVIN_TASK_PROTOCOL_SOURCE_SHA256.items():
        digest = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(
                "official CALVIN task protocol source differs from the reviewed revision: "
                f"{relative_path}"
            )
        observed[relative_path] = digest
    return observed


def validate_calvin_task_protocol_inventory(identity_keys: Sequence[str]) -> tuple[str, ...]:
    """Require the sidecar ontology to contain every protocol participant."""

    inventory = tuple(identity_keys)
    if any(not isinstance(key, str) or not key for key in inventory):
        raise ValueError("CALVIN physical inventory must contain nonempty identity strings")
    if len(set(inventory)) != len(inventory):
        raise ValueError("CALVIN physical inventory identities must be unique")
    required = {
        key
        for relevance in _CALVIN_TASK_RELEVANCE.values()
        for group in (
            relevance.action_target_identity_keys,
            relevance.outcome_identity_keys,
            relevance.known_participant_identity_keys,
        )
        for key in group
    }
    missing = sorted(required - set(inventory))
    if missing:
        raise ValueError(f"CALVIN task protocol identities are absent from the sidecar: {missing}")
    return tuple(sorted(required))


def _validated_identity_rows(
    identity_keys_by_row: Sequence[str | None],
    row_valid: Sequence[bool],
) -> tuple[tuple[str | None, ...], tuple[bool, ...], tuple[str, ...]]:
    keys = tuple(identity_keys_by_row)
    valid = tuple(row_valid)
    if len(keys) != len(valid):
        raise ValueError("CALVIN row identities and validity must share one posterior capacity")
    present = tuple(key for key in keys if key is not None)
    if any(not isinstance(key, str) or not key for key in present):
        raise ValueError("loss-side row identities must be nonempty strings or None")
    if len(set(present)) != len(present):
        raise ValueError("one physical identity cannot occupy two posterior rows")
    if any(not isinstance(value, bool) for value in valid):
        raise TypeError("posterior row validity must contain booleans")
    if any(key is not None and not is_valid for key, is_valid in zip(keys, valid, strict=True)):
        raise ValueError("a loss-side identity cannot name an invalid posterior row")
    return keys, valid, present


def select_witnessed_task_rows(
    *,
    task_key: str,
    identity_keys_by_row: Sequence[str | None],
    row_valid: Sequence[bool],
) -> CalvinWitnessedTaskRowSelection:
    """Select exact task-target identities already bound to posterior rows.

    The row map is a loss-side audit annotation from a completed earlier frame.
    This function is intended for evaluator-controlled perturbations only; its
    output must never enter training sample selection or runtime inference.
    """

    relevance = calvin_task_physical_relevance(task_key)
    keys, valid, present = _validated_identity_rows(identity_keys_by_row, row_valid)
    if not relevance.exact_action_target:
        return CalvinWitnessedTaskRowSelection(
            task_key=task_key,
            action_target_identity_keys=(),
            row_indices=(),
            row_identity_keys=(),
            exact_action_target=False,
            reason=f"task protocol is inexact: {relevance.exclusion_reason}",
        )

    target_set = set(relevance.action_target_identity_keys)
    selected = tuple(
        (row, key)
        for row, (key, is_valid) in enumerate(zip(keys, valid, strict=True))
        if is_valid and key in target_set
    )
    reason = (
        "task target was witnessed on a valid posterior row"
        if selected
        else "no action-target identity has yet been witnessed on a posterior row"
    )
    if selected and not target_set.issubset(set(present)):
        raise RuntimeError("partial multi-target witness selection is not an exact intervention")
    return CalvinWitnessedTaskRowSelection(
        task_key=task_key,
        action_target_identity_keys=relevance.action_target_identity_keys,
        row_indices=tuple(row for row, _key in selected),
        row_identity_keys=tuple(key for _row, key in selected),
        exact_action_target=True,
        reason=reason,
    )


def select_hidden_task_rows(
    *,
    task_key: str,
    identity_keys_by_row: Sequence[str | None],
    row_valid: Sequence[bool],
    measurement_age_s: Sequence[float],
    currently_measurable_identity_keys: Sequence[str],
    reference_delta_t_s: float,
) -> CalvinHiddenTaskRowSelection:
    """Select witnessed action-target rows that are currently unmeasurable.

    Every input except ``task_key`` is produced after the deploy-visible model
    forward.  ``identity_keys_by_row`` and current measurability are loss-side
    annotations and therefore may only be used by a read-only evaluator.
    """

    relevance = calvin_task_physical_relevance(task_key)
    keys, valid, present = _validated_identity_rows(identity_keys_by_row, row_valid)
    ages = tuple(measurement_age_s)
    measurable = tuple(currently_measurable_identity_keys)
    if len(keys) != len(ages):
        raise ValueError("CALVIN hidden-row inputs must share one posterior capacity")
    if any(
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0.0
        for value in ages
    ):
        raise ValueError("posterior measurement ages must be finite and nonnegative")
    if (
        isinstance(reference_delta_t_s, bool)
        or not isinstance(reference_delta_t_s, int | float)
        or not math.isfinite(float(reference_delta_t_s))
        or reference_delta_t_s <= 0.0
    ):
        raise ValueError("reference delta time must be finite and positive")
    if any(not isinstance(key, str) or not key for key in measurable):
        raise ValueError("measurable identities must be nonempty strings")
    if len(set(measurable)) != len(measurable):
        raise ValueError("measurable identities must be unique")
    if not relevance.exact_action_target:
        return CalvinHiddenTaskRowSelection(
            task_key=task_key,
            action_target_identity_keys=(),
            row_indices=(),
            row_identity_keys=(),
            exact_action_target=False,
            reason=f"task protocol is inexact: {relevance.exclusion_reason}",
        )

    measurable_set = set(measurable)
    target_set = set(relevance.action_target_identity_keys)
    selected = tuple(
        (row, key)
        for row, (key, is_valid, age) in enumerate(zip(keys, valid, ages, strict=True))
        if is_valid
        and key in target_set
        and key not in measurable_set
        and float(age) >= float(reference_delta_t_s)
    )
    if selected:
        reason = "witnessed task target is currently unmeasurable and at least one frame old"
    elif not target_set.intersection(present):
        reason = "no action-target identity has yet been witnessed on a posterior row"
    elif target_set.intersection(measurable_set):
        reason = "witnessed action-target identity is currently measurable"
    else:
        reason = "witnessed hidden action-target row is younger than one reference frame"
    return CalvinHiddenTaskRowSelection(
        task_key=task_key,
        action_target_identity_keys=relevance.action_target_identity_keys,
        row_indices=tuple(row for row, _key in selected),
        row_identity_keys=tuple(key for _row, key in selected),
        exact_action_target=True,
        reason=reason,
    )

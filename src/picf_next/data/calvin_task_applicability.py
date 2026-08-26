"""Offline CALVIN task applicability for same-observation grounding data.

This module is privileged dataset tooling. Simulator state is used only to
prove that an official instruction is valid for an archived observation. None
of the state, identity, visibility, or applicability values are model inputs.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
)
from picf_next.eval.calvin_task_relevance import (
    CALVIN_TASK_CONFIG_SHA256,
    calvin_exact_task_loss_identities,
)

CALVIN_OFFICIAL_ANNOTATIONS_SHA256: Final = (
    "c37dcb23fa8d57614e6e4f53e9dea6b8829e3621d8eb1aef43244729eedfbf9b"
)
CALVIN_OFFICIAL_TASKS_SHA256: Final = CALVIN_TASK_CONFIG_SHA256
CALVIN_TASK_APPLICABILITY_SCHEMA: Final = "picf-next.calvin-task-applicability.v1"
_MAX_JOINT_LIMIT_RESIDUAL_FRACTION: Final = 0.01

_DOOR_TASKS: Final = {
    "base__slide": (
        ("move_slider_left", 0.15),
        ("move_slider_right", -0.15),
    ),
    "base__drawer": (
        ("open_drawer", 0.12),
        ("close_drawer", -0.12),
    ),
}
_LIGHT_TASKS: Final = {
    "led": ("turn_on_led", "turn_off_led"),
    "lightbulb": ("turn_on_lightbulb", "turn_off_lightbulb"),
}
_LIFT_TASKS: Final = {
    ("block_red", "base_link"): "lift_red_block_table",
    ("block_red", "plank_link"): "lift_red_block_slider",
    ("block_red", "drawer_link"): "lift_red_block_drawer",
    ("block_blue", "base_link"): "lift_blue_block_table",
    ("block_blue", "plank_link"): "lift_blue_block_slider",
    ("block_blue", "drawer_link"): "lift_blue_block_drawer",
    ("block_pink", "base_link"): "lift_pink_block_table",
    ("block_pink", "plank_link"): "lift_pink_block_slider",
    ("block_pink", "drawer_link"): "lift_pink_block_drawer",
}


def _finite_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"{name} must be a finite number")
    return result


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, name: str) -> str:
    result = _nonempty_text(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ContractError(f"{name} must be one lowercase SHA-256")
    return result


def _source_global_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError("same-observation source index must be non-negative")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    return value


def _mapping_entry(
    values: Mapping[object, object],
    key: str,
    name: str,
) -> Mapping[object, object]:
    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} is missing or malformed")
    return value


def _record_sequence(value: object, name: str) -> Sequence[object]:
    if isinstance(value, str | bytes) or not isinstance(value, Sequence):
        raise ContractError(f"{name} must be a record sequence")
    return value


@dataclass(frozen=True, slots=True)
class CalvinJointState:
    name: str
    position: float
    lower_limit: float
    upper_limit: float

    def __post_init__(self) -> None:
        _nonempty_text(self.name, "CALVIN joint name")
        position = _finite_float(self.position, "CALVIN joint position")
        lower = _finite_float(self.lower_limit, "CALVIN joint lower limit")
        upper = _finite_float(self.upper_limit, "CALVIN joint upper limit")
        if lower >= upper:
            raise ContractError("CALVIN joint state lies outside a valid finite range")
        residual_tolerance = max(
            1e-6,
            (upper - lower) * _MAX_JOINT_LIMIT_RESIDUAL_FRACTION,
        )
        if position < lower - residual_tolerance or position > upper + residual_tolerance:
            raise ContractError("CALVIN joint state lies outside a valid finite range")

    @property
    def feasible_position(self) -> float:
        """Project bounded simulator constraint residuals onto the feasible interval."""

        return min(max(float(self.position), float(self.lower_limit)), float(self.upper_limit))


@dataclass(frozen=True, slots=True)
class CalvinTaskApplicabilityState:
    """Minimal loss-side state needed to prove official task preconditions."""

    doors: tuple[CalvinJointState, ...]
    light_states: tuple[tuple[str, int], ...]
    block_support_links: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.doors, tuple) or any(
            not isinstance(item, CalvinJointState) for item in self.doors
        ):
            raise ContractError("CALVIN applicability doors must be joint-state records")
        door_names = tuple(item.name for item in self.doors)
        if len(set(door_names)) != len(door_names) or set(door_names) != set(_DOOR_TASKS):
            raise ContractError("CALVIN applicability requires the pinned drawer and slider")
        if not isinstance(self.light_states, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not item[0]
            or isinstance(item[1], bool)
            or not isinstance(item[1], int)
            for item in self.light_states
        ):
            raise ContractError("CALVIN applicability light-state records are malformed")
        light_names = tuple(name for name, _ in self.light_states)
        if len(set(light_names)) != len(light_names) or set(light_names) != set(_LIGHT_TASKS):
            raise ContractError("CALVIN applicability requires the pinned LED and lightbulb")
        if any(state not in (0, 1) for _, state in self.light_states):
            raise ContractError("CALVIN logical light states must be binary")
        if not isinstance(self.block_support_links, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, str) or not value for value in item)
            for item in self.block_support_links
        ):
            raise ContractError("CALVIN block support contacts are malformed")
        if len(set(self.block_support_links)) != len(self.block_support_links):
            raise ContractError("CALVIN block support contacts must be unique")
        if any(item not in _LIFT_TASKS for item in self.block_support_links):
            raise ContractError("CALVIN block support contact is outside the pinned task protocol")


@dataclass(frozen=True, slots=True)
class CalvinApplicableTask:
    task_key: str
    target_identity_key: str
    proof: str

    def __post_init__(self) -> None:
        _nonempty_text(self.task_key, "applicable CALVIN task key")
        _nonempty_text(self.target_identity_key, "applicable CALVIN target identity")
        _nonempty_text(self.proof, "applicable CALVIN task proof")
        targets = calvin_exact_task_loss_identities(self.task_key)
        if targets != (self.target_identity_key,):
            raise ContractError("applicable CALVIN task disagrees with the pinned target protocol")


@dataclass(frozen=True, slots=True)
class CalvinVisibleIdentitySupport:
    identity_key: str
    camera_pixel_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        _nonempty_text(self.identity_key, "visible CALVIN identity")
        camera_names = tuple(name for name, _ in self.camera_pixel_counts)
        if (
            not self.camera_pixel_counts
            or len(set(camera_names)) != len(camera_names)
            or any(
                not isinstance(name, str)
                or not name
                or isinstance(count, bool)
                or not isinstance(count, int)
                or count < 0
                for name, count in self.camera_pixel_counts
            )
            or self.total_pixel_count <= 0
        ):
            raise ContractError("visible CALVIN identity support is malformed")

    @property
    def total_pixel_count(self) -> int:
        return sum(count for _, count in self.camera_pixel_counts)


@dataclass(frozen=True, slots=True)
class CalvinSameObservationVariant:
    task_key: str
    instruction: str
    instruction_sha256: str
    target_identity_key: str
    proof: str

    def __post_init__(self) -> None:
        _nonempty_text(self.task_key, "same-observation task key")
        instruction = _nonempty_text(self.instruction, "same-observation instruction")
        _sha256(self.instruction_sha256, "same-observation instruction SHA-256")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != self.instruction_sha256:
            raise ContractError("same-observation instruction SHA-256 changed")
        CalvinApplicableTask(self.task_key, self.target_identity_key, self.proof)


@dataclass(frozen=True, slots=True)
class CalvinSameObservationGroup:
    """At least two true prompts with different targets at exactly fixed X."""

    source_global_index: int
    source_state_sha256: str
    variants: tuple[CalvinSameObservationVariant, ...]
    schema: str = CALVIN_TASK_APPLICABILITY_SCHEMA

    def __post_init__(self) -> None:
        _source_global_index(self.source_global_index)
        _sha256(self.source_state_sha256, "same-observation source-state SHA-256")
        if self.schema != CALVIN_TASK_APPLICABILITY_SCHEMA:
            raise ContractError("same-observation applicability schema changed")
        if (
            not isinstance(self.variants, tuple)
            or any(not isinstance(item, CalvinSameObservationVariant) for item in self.variants)
            or len(self.variants) < 2
        ):
            raise ContractError("same-observation group requires at least two variants")
        targets = tuple(item.target_identity_key for item in self.variants)
        tasks = tuple(item.task_key for item in self.variants)
        if len(set(targets)) != len(targets) or len(set(tasks)) != len(tasks):
            raise ContractError("same-observation variants require distinct tasks and targets")

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_global_index": self.source_global_index,
            "source_state_sha256": self.source_state_sha256,
            "model_input_contains_simulator_state_or_identity": False,
            "variants": [
                {
                    "instruction": item.instruction,
                    "instruction_sha256": item.instruction_sha256,
                    "proof": item.proof,
                    "target_identity_key": item.target_identity_key,
                    "task_key": item.task_key,
                }
                for item in self.variants
            ],
        }


def calvin_state_applicable_tasks(
    state: CalvinTaskApplicabilityState,
) -> tuple[CalvinApplicableTask, ...]:
    """Return only official tasks whose start-state predicates are proven."""

    if not isinstance(state, CalvinTaskApplicabilityState):
        raise TypeError("state must be CalvinTaskApplicabilityState")
    tasks: list[CalvinApplicableTask] = []
    for door in sorted(state.doors, key=lambda item: item.name):
        feasible_position = door.feasible_position
        for task_key, delta in _DOOR_TASKS[door.name]:
            endpoint = feasible_position + delta
            if door.lower_limit - 1e-6 <= endpoint <= door.upper_limit + 1e-6:
                target = calvin_exact_task_loss_identities(task_key)
                if target is None or len(target) != 1:
                    raise RuntimeError("pinned articulated task lost its exact target")
                tasks.append(
                    CalvinApplicableTask(
                        task_key=task_key,
                        target_identity_key=target[0],
                        proof=(
                            f"joint-range:{door.name}:{feasible_position:.9g}"
                            f"{delta:+.9g}->{endpoint:.9g}"
                        ),
                    )
                )
    for light_name, logical_state in sorted(state.light_states):
        task_key = _LIGHT_TASKS[light_name][logical_state]
        target = calvin_exact_task_loss_identities(task_key)
        if target is None or len(target) != 1:
            raise RuntimeError("pinned light task lost its exact action target")
        tasks.append(
            CalvinApplicableTask(
                task_key=task_key,
                target_identity_key=target[0],
                proof=f"logical-state:{light_name}:{logical_state}",
            )
        )
    for block_name, support_link in sorted(state.block_support_links):
        task_key = _LIFT_TASKS[block_name, support_link]
        target = calvin_exact_task_loss_identities(task_key)
        if target is None or len(target) != 1:
            raise RuntimeError("pinned lift task lost its exact target")
        tasks.append(
            CalvinApplicableTask(
                task_key=task_key,
                target_identity_key=target[0],
                proof=f"contact:table/{support_link}->{block_name}",
            )
        )
    return tuple(sorted(tasks, key=lambda item: (item.target_identity_key, item.task_key)))


def calvin_visible_supervised_identity_support(
    frame: CalvinPhysicalSupervisionFrame,
) -> tuple[CalvinVisibleIdentitySupport, ...]:
    """Count only depth-verified owner pixels in deploy-visible cameras."""

    if not isinstance(frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("frame must be CalvinPhysicalSupervisionFrame")
    camera_names = tuple(camera.camera_name for camera in frame.cameras)
    if len(set(camera_names)) != len(camera_names):
        raise ContractError("CALVIN physical frame contains duplicate cameras")
    output = []
    for owner_index, identity_key in enumerate(frame.identity_keys, start=1):
        camera_counts = tuple(
            (
                camera.camera_name,
                int(((camera.owner_index == owner_index) & camera.owner_supervised).sum()),
            )
            for camera in frame.cameras
        )
        if sum(count for _, count in camera_counts) > 0:
            output.append(
                CalvinVisibleIdentitySupport(
                    identity_key=identity_key,
                    camera_pixel_counts=camera_counts,
                )
            )
    return tuple(output)


def extract_calvin_task_applicability_state(environment: Any) -> CalvinTaskApplicabilityState:
    """Extract exact preconditions after ``restore_calvin_archived_state``."""

    if not all(hasattr(environment, name) for name in ("scene", "p", "cid", "get_info")):
        raise ContractError("CALVIN applicability environment is malformed")
    client_id = _integer(environment.cid, "CALVIN physics client ID")
    info = environment.get_info()
    if not isinstance(info, Mapping):
        raise ContractError("CALVIN environment info must be a mapping")
    scene_info = info.get("scene_info")
    if not isinstance(scene_info, Mapping):
        raise ContractError("CALVIN environment omitted scene_info")
    raw_doors = scene_info.get("doors")
    raw_lights = scene_info.get("lights")
    raw_fixed = scene_info.get("fixed_objects")
    raw_movable = scene_info.get("movable_objects")
    if (
        not isinstance(raw_doors, Mapping)
        or not isinstance(raw_lights, Mapping)
        or not isinstance(raw_fixed, Mapping)
        or not isinstance(raw_movable, Mapping)
    ):
        raise ContractError("CALVIN scene_info is missing task applicability fields")

    if not hasattr(environment.scene, "doors"):
        raise ContractError("CALVIN simulator omitted its door inventory")
    try:
        door_objects = tuple(environment.scene.doors)
    except TypeError as error:
        raise ContractError("CALVIN simulator door inventory is malformed") from error
    doors_by_name: dict[str, object] = {}
    for item in door_objects:
        name = _nonempty_text(getattr(item, "name", None), "CALVIN simulator door name")
        if name in doors_by_name:
            raise ContractError("CALVIN simulator door inventory contains duplicates")
        doors_by_name[name] = item
    if set(doors_by_name) != set(_DOOR_TASKS):
        raise ContractError("CALVIN simulator door inventory changed")
    doors = []
    for name in sorted(_DOOR_TASKS):
        item = doors_by_name[name]
        uid = _integer(getattr(item, "uid", None), f"{name} body ID")
        joint_index = _integer(getattr(item, "joint_index", None), f"{name} joint index")
        joint_info = environment.p.getJointInfo(
            uid,
            joint_index,
            physicsClientId=client_id,
        )
        if not isinstance(joint_info, Sequence) or len(joint_info) < 10:
            raise ContractError("CALVIN joint metadata is malformed")
        door_info = _mapping_entry(raw_doors, name, f"{name} scene info")
        doors.append(
            CalvinJointState(
                name=name,
                position=_finite_float(door_info.get("current_state"), f"{name} current state"),
                lower_limit=_finite_float(joint_info[8], f"{name} lower limit"),
                upper_limit=_finite_float(joint_info[9], f"{name} upper limit"),
            )
        )

    light_states = []
    for name in sorted(_LIGHT_TASKS):
        light_info = _mapping_entry(raw_lights, name, f"{name} scene info")
        logical_state = light_info.get("logical_state")
        if not isinstance(logical_state, int) or isinstance(logical_state, bool):
            raise ContractError("CALVIN logical light state is malformed")
        light_states.append((name, logical_state))

    table_info = _mapping_entry(raw_fixed, "table", "CALVIN table scene info")
    table_uid = table_info.get("uid")
    table_links = table_info.get("links")
    if (
        not isinstance(table_uid, int)
        or isinstance(table_uid, bool)
        or not isinstance(table_links, Mapping)
    ):
        raise ContractError("CALVIN table identity metadata is malformed")
    support_by_link = {}
    for link_name in ("base_link", "plank_link", "drawer_link"):
        link_index = _integer(
            table_links.get(link_name),
            f"CALVIN table {link_name} index",
        )
        if link_index in support_by_link:
            raise ContractError("CALVIN table support links share one index")
        support_by_link[link_index] = link_name

    support_contacts = []
    for block_name in ("block_blue", "block_pink", "block_red"):
        block_info = _mapping_entry(raw_movable, block_name, f"{block_name} scene info")
        contacts = _record_sequence(
            block_info.get("contacts"),
            f"{block_name} contacts",
        )
        for raw_contact in contacts:
            contact = _record_sequence(raw_contact, f"{block_name} contact")
            if len(contact) < 5:
                raise ContractError("CALVIN contact record is malformed")
            contacted_body = _integer(contact[2], f"{block_name} contact body ID")
            contacted_link = _integer(contact[4], f"{block_name} contact link index")
            if contacted_body == table_uid and contacted_link in support_by_link:
                support_contacts.append((block_name, support_by_link[contacted_link]))
    return CalvinTaskApplicabilityState(
        doors=tuple(doors),
        light_states=tuple(light_states),
        block_support_links=tuple(sorted(set(support_contacts))),
    )


def load_official_calvin_annotations(path: str | Path) -> Mapping[str, tuple[str, ...]]:
    """Load the content-pinned official CALVIN language annotations."""

    source = Path(path)
    payload = source.read_bytes()
    if hashlib.sha256(payload).hexdigest() != CALVIN_OFFICIAL_ANNOTATIONS_SHA256:
        raise ContractError("official CALVIN annotation SHA-256 changed")
    decoded = yaml.safe_load(payload)
    if not isinstance(decoded, Mapping):
        raise ContractError("official CALVIN annotations must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for task_key, raw_instructions in decoded.items():
        if (
            not isinstance(task_key, str)
            or not task_key
            or not isinstance(raw_instructions, list)
            or not raw_instructions
        ):
            raise ContractError("official CALVIN annotation entry is malformed")
        instructions = tuple(
            _nonempty_text(item, f"official instruction for {task_key}")
            for item in raw_instructions
        )
        if len(set(instructions)) != len(instructions):
            raise ContractError("official CALVIN annotations contain duplicate instructions")
        result[task_key] = instructions
    required_tasks = {
        *(task_key for tasks in _DOOR_TASKS.values() for task_key, _ in tasks),
        *(task_key for tasks in _LIGHT_TASKS.values() for task_key in tasks),
        *_LIFT_TASKS.values(),
    }
    if not required_tasks <= result.keys():
        raise ContractError("official CALVIN annotations omit an applicability task")
    return result


def verify_official_calvin_task_config(path: str | Path) -> str:
    """Require the exact official task-to-predicate configuration."""

    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != CALVIN_OFFICIAL_TASKS_SHA256:
        raise ContractError("official CALVIN task configuration SHA-256 changed")
    return digest


def build_same_observation_group(
    *,
    source_global_index: int,
    source_state_sha256: str,
    visible_identity_keys: Sequence[str],
    applicable_tasks: Sequence[CalvinApplicableTask],
    annotations: Mapping[str, Sequence[str]],
    maximum_variants: int,
) -> CalvinSameObservationGroup | None:
    """Build a deterministic fixed-X group or skip an unidentifiable frame."""

    if (
        isinstance(maximum_variants, bool)
        or not isinstance(maximum_variants, int)
        or maximum_variants < 2
    ):
        raise ContractError("same-observation maximum_variants must be at least two")
    if isinstance(visible_identity_keys, str | bytes) or not isinstance(
        visible_identity_keys, Sequence
    ):
        raise ContractError("visible identity keys must be a sequence")
    if isinstance(applicable_tasks, str | bytes) or not isinstance(applicable_tasks, Sequence):
        raise ContractError("applicable tasks must be a sequence")
    if not isinstance(annotations, Mapping):
        raise ContractError("official CALVIN annotations must be a mapping")
    validated_source_index = _source_global_index(source_global_index)
    state_digest = _sha256(source_state_sha256, "same-observation source-state SHA-256")
    visible = tuple(_nonempty_text(item, "visible identity key") for item in visible_identity_keys)
    if len(set(visible)) != len(visible):
        raise ContractError("visible identity keys must be unique")
    visible_set = set(visible)

    by_target: dict[str, list[CalvinApplicableTask]] = {}
    for task in applicable_tasks:
        if not isinstance(task, CalvinApplicableTask):
            raise TypeError("applicable_tasks must contain CalvinApplicableTask values")
        if task.target_identity_key in visible_set:
            by_target.setdefault(task.target_identity_key, []).append(task)
    if len(by_target) < 2:
        return None

    selected = []
    for target, tasks in sorted(by_target.items()):
        task = min(
            tasks,
            key=lambda item: hashlib.sha256(
                f"{state_digest}\0{target}\0{item.task_key}".encode()
            ).digest(),
        )
        raw_instructions = annotations.get(task.task_key)
        if not isinstance(raw_instructions, Sequence) or isinstance(raw_instructions, str):
            raise ContractError("applicable task is absent from official CALVIN annotations")
        instructions = tuple(
            _nonempty_text(item, f"official instruction for {task.task_key}")
            for item in raw_instructions
        )
        if not instructions:
            raise ContractError("applicable task has no official CALVIN instruction")
        instruction = min(
            instructions,
            key=lambda item: hashlib.sha256(
                f"{state_digest}\0{task.task_key}\0{item}".encode()
            ).digest(),
        )
        selected.append(
            CalvinSameObservationVariant(
                task_key=task.task_key,
                instruction=instruction,
                instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                target_identity_key=target,
                proof=task.proof,
            )
        )
    ordered = sorted(
        selected,
        key=lambda item: hashlib.sha256(
            f"{state_digest}\0{item.target_identity_key}\0{item.task_key}".encode()
        ).digest(),
    )[:maximum_variants]
    return CalvinSameObservationGroup(
        source_global_index=validated_source_index,
        source_state_sha256=state_digest,
        variants=tuple(ordered),
    )

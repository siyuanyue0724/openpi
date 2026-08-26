"""Diagnostic CALVIN behavior support for task-grounding data audits.

This module measures whether an official language segment contains physical
evidence consistent with its reviewed direct-manipulation target.  It is not a
task-success evaluator and may not select training examples or enter model
inputs.  In particular, AABB-centre motion cannot certify rotational outcomes.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinLanguageSegment
from picf_next.eval.calvin_task_relevance import (
    CALVIN_SCENE_CONFIG_SHA256,
    calvin_task_physical_relevance,
)

CALVIN_ROBOT_BASE_POSITION_M = (-0.34, -0.46, 0.24)
CALVIN_ROBOT_BASE_EULER_RAD = (0.0, 0.0, 0.0)
# Compatibility names retained for earlier diagnostic reports and tests.
CALVIN_SCENE_D_CONFIG_SHA256 = CALVIN_SCENE_CONFIG_SHA256["calvin_scene_D"]
CALVIN_SCENE_D_ROBOT_BASE_POSITION_M = CALVIN_ROBOT_BASE_POSITION_M
CALVIN_SCENE_D_ROBOT_BASE_EULER_RAD = CALVIN_ROBOT_BASE_EULER_RAD


def select_calvin_behavior_segments(
    segments: Sequence[CalvinLanguageSegment],
    *,
    samples_per_task_scene: int,
    scene_by_segment_index: Mapping[int, str],
) -> tuple[CalvinLanguageSegment, ...]:
    """Select deterministic task/scene strata without consulting target evidence."""

    if (
        isinstance(samples_per_task_scene, bool)
        or not isinstance(samples_per_task_scene, int)
        or samples_per_task_scene <= 0
    ):
        raise ContractError("behavior audit samples_per_task_scene must be positive")
    grouped: dict[tuple[str, str], list[CalvinLanguageSegment]] = {}
    seen_indices: set[int] = set()
    for segment in segments:
        if not isinstance(segment, CalvinLanguageSegment):
            raise TypeError("behavior audit requires CALVIN language segments")
        if segment.index in seen_indices:
            raise ContractError("behavior audit segment indices must be unique")
        seen_indices.add(segment.index)
        try:
            scene = scene_by_segment_index[segment.index]
        except KeyError as error:
            raise ContractError("behavior audit segment has no scene assignment") from error
        if scene not in CALVIN_SCENE_CONFIG_SHA256:
            raise ContractError("behavior audit scene assignment is unsupported")
        grouped.setdefault((segment.task_key, scene), []).append(segment)
    if not grouped:
        raise ContractError("behavior audit requires at least one language segment")
    if set(scene_by_segment_index) != seen_indices:
        raise ContractError("behavior audit scene assignments differ from segment inventory")

    selected: list[CalvinLanguageSegment] = []
    for task_scene in sorted(grouped):
        task_segments = sorted(grouped[task_scene], key=lambda item: item.index)
        count = min(samples_per_task_scene, len(task_segments))
        if count == 1:
            positions = (len(task_segments) // 2,)
        else:
            positions = tuple(
                round(rank * (len(task_segments) - 1) / (count - 1)) for rank in range(count)
            )
        if len(set(positions)) != count:
            raise RuntimeError("behavior audit quantile selection produced duplicate segments")
        selected.extend(task_segments[position] for position in positions)
    return tuple(selected)


def _finite_array(
    value: object,
    *,
    name: str,
    ndim: int,
    trailing_shape: tuple[int, ...],
) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != ndim or array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ContractError(f"{name} must have trailing shape {trailing_shape}")
    if not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")
    return array


def calvin_tcp_robot_base_position(
    tcp_position_world_m: object,
) -> NDArray[np.float64]:
    """Transform archived TCP world positions into the common CALVIN robot-base chart."""

    tcp = _finite_array(
        tcp_position_world_m,
        name="CALVIN TCP world position",
        ndim=2,
        trailing_shape=(3,),
    )
    # Official scene A/B/C/D configs share this base and zero orientation.
    # Keep that pinned fact explicit rather than adding an untested general transform.
    base = np.asarray(CALVIN_ROBOT_BASE_POSITION_M, dtype=np.float64)
    return tcp - base


def calvin_scene_d_tcp_robot_base_position(
    tcp_position_world_m: object,
) -> NDArray[np.float64]:
    """Compatibility alias for the common CALVIN robot-base transform."""

    return calvin_tcp_robot_base_position(tcp_position_world_m)


@dataclass(frozen=True, slots=True)
class CalvinBehaviorSupportSummary:
    task_key: str
    target_identity_key: str
    global_indices: tuple[int, ...]
    target_net_displacement_m: float
    target_max_displacement_m: float
    target_motion_rank: int
    physical_identity_count: int
    minimum_tcp_target_distance_m: float
    initial_tcp_target_distance_m: float
    final_tcp_target_distance_m: float
    closest_global_index: int
    maximum_displacement_global_index: int
    mean_action_motion_norm: float
    camera_visible_frame_counts: tuple[tuple[str, int], ...]
    camera_max_visible_pixels: tuple[tuple[str, int], ...]
    identity_max_displacements_m: tuple[tuple[str, float], ...]
    maximum_motion_identity_key: str
    maximum_identity_displacement_m: float
    target_motion_margin_to_best_other_m: float
    geometry_observation_scope: str = "aabb-centre-translation-only"
    task_success_certified: bool = False
    training_authorized: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "camera_max_visible_pixels": dict(self.camera_max_visible_pixels),
            "camera_visible_frame_counts": dict(self.camera_visible_frame_counts),
            "closest_global_index": self.closest_global_index,
            "final_tcp_target_distance_m": self.final_tcp_target_distance_m,
            "geometry_observation_scope": self.geometry_observation_scope,
            "global_indices": list(self.global_indices),
            "initial_tcp_target_distance_m": self.initial_tcp_target_distance_m,
            "identity_max_displacements_m": dict(self.identity_max_displacements_m),
            "maximum_identity_displacement_m": self.maximum_identity_displacement_m,
            "maximum_motion_identity_key": self.maximum_motion_identity_key,
            "maximum_displacement_global_index": self.maximum_displacement_global_index,
            "mean_action_motion_norm": self.mean_action_motion_norm,
            "minimum_tcp_target_distance_m": self.minimum_tcp_target_distance_m,
            "physical_identity_count": self.physical_identity_count,
            "target_identity_key": self.target_identity_key,
            "target_max_displacement_m": self.target_max_displacement_m,
            "target_motion_rank": self.target_motion_rank,
            "target_motion_margin_to_best_other_m": (self.target_motion_margin_to_best_other_m),
            "target_net_displacement_m": self.target_net_displacement_m,
            "task_key": self.task_key,
            "task_success_certified": self.task_success_certified,
            "training_authorized": self.training_authorized,
        }


def summarize_calvin_behavior_support(
    *,
    task_key: str,
    target_identity_key: str,
    global_indices: Sequence[int],
    identity_keys: Sequence[str],
    geometry_robot_base_m: object,
    tcp_position_world_m: object,
    actions: object,
    visible_target_pixels: Mapping[str, Sequence[int]],
) -> CalvinBehaviorSupportSummary:
    """Summarize one real language segment without defining a success threshold."""

    relevance = calvin_task_physical_relevance(task_key)
    if not relevance.exact_action_target:
        raise ContractError("behavior support requires a reviewed exact action target")
    if relevance.action_target_identity_keys != (target_identity_key,):
        raise ContractError("behavior target differs from the reviewed CALVIN protocol")

    indices = tuple(global_indices)
    if (
        len(indices) < 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in indices)
        or any(right != left + 1 for left, right in zip(indices, indices[1:], strict=False))
    ):
        raise ContractError("behavior support requires at least two contiguous source frames")
    identities = tuple(identity_keys)
    if (
        not identities
        or len(set(identities)) != len(identities)
        or any(not isinstance(value, str) or not value for value in identities)
    ):
        raise ContractError("behavior support requires a unique physical inventory")
    try:
        target_row = identities.index(target_identity_key)
    except ValueError as error:
        raise ContractError("behavior target is absent from the physical inventory") from error

    geometry = _finite_array(
        geometry_robot_base_m,
        name="CALVIN physical geometry",
        ndim=3,
        trailing_shape=(len(identities), 3),
    )
    if geometry.shape[0] != len(indices):
        raise ContractError("behavior geometry and source indices have different lengths")
    tcp = calvin_tcp_robot_base_position(tcp_position_world_m)
    if tcp.shape[0] != len(indices):
        raise ContractError("behavior TCP and source indices have different lengths")
    action_array = _finite_array(
        actions,
        name="CALVIN relative actions",
        ndim=2,
        trailing_shape=(7,),
    )
    if action_array.shape[0] not in {len(indices) - 1, len(indices)}:
        raise ContractError("behavior actions must align to segment transitions or frames")

    target = geometry[:, target_row]
    displacement_by_frame = np.linalg.norm(target - target[0], axis=1)
    net_displacement = float(np.linalg.norm(target[-1] - target[0]))
    max_displacement = float(displacement_by_frame.max())
    motion_by_identity = np.linalg.norm(geometry - geometry[0:1], axis=2).max(axis=0)
    # Equal displacement shares one rank; float noise below one nanometre is immaterial.
    target_motion_rank = 1 + int(np.sum(motion_by_identity > max_displacement + 1e-9))
    maximum_motion_row = int(np.argmax(motion_by_identity))
    other_motion = np.delete(motion_by_identity, target_row)
    best_other_motion = float(other_motion.max()) if len(other_motion) else 0.0
    tcp_distance = np.linalg.norm(tcp - target, axis=1)
    closest_row = int(np.argmin(tcp_distance))
    maximum_displacement_row = int(np.argmax(displacement_by_frame))

    camera_counts: list[tuple[str, int]] = []
    camera_maxima: list[tuple[str, int]] = []
    if set(visible_target_pixels) != {"static", "gripper"}:
        raise ContractError("behavior support requires the two pinned CALVIN cameras")
    for camera_name in sorted(visible_target_pixels):
        values = tuple(visible_target_pixels[camera_name])
        if len(values) != len(indices) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values
        ):
            raise ContractError("visible target pixels must be nonnegative frame-aligned integers")
        camera_counts.append((camera_name, sum(value > 0 for value in values)))
        camera_maxima.append((camera_name, max(values)))
    return CalvinBehaviorSupportSummary(
        task_key=task_key,
        target_identity_key=target_identity_key,
        global_indices=indices,
        target_net_displacement_m=net_displacement,
        target_max_displacement_m=max_displacement,
        target_motion_rank=target_motion_rank,
        physical_identity_count=len(identities),
        minimum_tcp_target_distance_m=float(tcp_distance[closest_row]),
        initial_tcp_target_distance_m=float(tcp_distance[0]),
        final_tcp_target_distance_m=float(tcp_distance[-1]),
        closest_global_index=indices[closest_row],
        maximum_displacement_global_index=indices[maximum_displacement_row],
        mean_action_motion_norm=float(np.linalg.norm(action_array[:, :6], axis=1).mean()),
        camera_visible_frame_counts=tuple(camera_counts),
        camera_max_visible_pixels=tuple(camera_maxima),
        identity_max_displacements_m=tuple(
            (identity, float(displacement))
            for identity, displacement in zip(identities, motion_by_identity, strict=True)
        ),
        maximum_motion_identity_key=identities[maximum_motion_row],
        maximum_identity_displacement_m=float(motion_by_identity[maximum_motion_row]),
        target_motion_margin_to_best_other_m=max_displacement - best_other_motion,
    )


def calvin_behavior_review_keyframes(
    summary: CalvinBehaviorSupportSummary,
) -> tuple[int, ...]:
    """Return chronological evidence-derived review frames without model output."""

    ordered = (
        summary.global_indices[0],
        summary.closest_global_index,
        summary.maximum_displacement_global_index,
        summary.global_indices[-1],
    )
    return tuple(sorted(set(ordered)))

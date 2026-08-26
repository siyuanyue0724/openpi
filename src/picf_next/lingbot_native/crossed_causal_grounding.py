"""Fail-closed support audits for task-conditioned crossed grounding data."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Any

import networkx as nx

from picf_next.data.calvin_qwen_grounding import qwen3vl_normalized_bbox
from picf_next.lingbot_native.fixed_observation import FixedObservationGroup

CROSSED_PHYSICAL_SUPPORT_SCHEMA = "picf-next.crossed-grounding-physical-support.v1"
CROSSED_PARTITION_SUPPORT_SCHEMA = "picf-next.crossed-grounding-partition-support.v1"
CROSSED_EPISODE_SPLIT_SCHEMA = "picf-next.crossed-grounding-episode-split.v1"
NATIVE_VL_SCENE_AUDIT_SCHEMA = "picf-next.native-vl-scene-curriculum-audit.v2"
CALVIN_GROUNDING_CAMERAS = ("static", "gripper")

_CAMERA_EXTENTS = {
    "static": (200, 200),
    "gripper": (84, 84),
}


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("crossed-grounding evidence is not canonical finite JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return result


def _git_revision(value: object, *, name: str) -> str:
    result = _text(value, name=name)
    if len(result) != 40 or any(character not in "0123456789abcdef" for character in result):
        raise ValueError(f"{name} must be one lowercase Git revision")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _bbox(value: object, *, camera_name: str, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain four integers")
    width, height = _CAMERA_EXTENTS[camera_name]
    result = (value[0], value[1], value[2], value[3])
    if not (0 <= result[0] < result[2] <= width and 0 <= result[1] < result[3] <= height):
        raise ValueError(f"{name} lies outside the source image")
    return result


def _centre(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _contains(box: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def boxes_are_mutually_centre_exclusive(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> bool:
    """Return whether neither target box contains the other target's centre."""

    return not _contains(first, _centre(second)) and not _contains(second, _centre(first))


@dataclass(frozen=True, slots=True)
class SceneTargetEvidence:
    group_index: int
    global_index: int
    camera_name: str
    source_rgb_sha256: str
    identity_key: str
    state: str
    bbox_qwen_xyxy: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        _nonnegative_int(self.group_index, name="scene target group index")
        _nonnegative_int(self.global_index, name="scene target global index")
        if self.camera_name not in CALVIN_GROUNDING_CAMERAS:
            raise ValueError("scene target camera differs from the CALVIN contract")
        _sha256(self.source_rgb_sha256, name="scene target RGB SHA-256")
        _text(self.identity_key, name="scene target identity")
        if self.state not in {"supervised", "subpatch", "absent"}:
            raise ValueError("scene target observability state is unsupported")
        if (self.state == "supervised") != (self.bbox_qwen_xyxy is not None):
            raise ValueError("only supervised scene targets may carry a training box")


@dataclass(frozen=True, slots=True)
class CrossedVariantViewEvidence:
    """One audited prompt/target joined to one source camera view."""

    group_index: int
    global_index: int
    source_episode_index: int
    source_state_sha256: str
    camera_name: str
    source_rgb_sha256: str
    task_key: str
    instruction_sha256: str
    target_identity_key: str
    state: str
    bbox_qwen_xyxy: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        _nonnegative_int(self.group_index, name="crossed variant group index")
        _nonnegative_int(self.global_index, name="crossed variant global index")
        _nonnegative_int(self.source_episode_index, name="crossed variant source episode")
        _sha256(self.source_state_sha256, name="crossed variant source-state SHA-256")
        if self.camera_name not in CALVIN_GROUNDING_CAMERAS:
            raise ValueError("crossed variant camera differs from the CALVIN contract")
        _sha256(self.source_rgb_sha256, name="crossed variant RGB SHA-256")
        _text(self.task_key, name="crossed variant task")
        _sha256(self.instruction_sha256, name="crossed variant instruction SHA-256")
        _text(self.target_identity_key, name="crossed variant target")
        if self.state not in {"supervised", "subpatch", "absent"}:
            raise ValueError("crossed variant observability state is unsupported")
        if (self.state == "supervised") != (self.bbox_qwen_xyxy is not None):
            raise ValueError("only supervised crossed variants may carry a training box")


def _identity_keys(values: Sequence[str]) -> tuple[str, ...]:
    identities = tuple(values)
    if not identities or any(not isinstance(value, str) or not value for value in identities):
        raise ValueError("crossed-grounding target identities must be nonempty strings")
    if len(set(identities)) != len(identities):
        raise ValueError("crossed-grounding target identities must be unique")
    return tuple(sorted(identities))


def _task_keys(values: Sequence[str]) -> tuple[str, ...]:
    tasks = tuple(values)
    if not tasks or any(not isinstance(value, str) or not value for value in tasks):
        raise ValueError("crossed-grounding expected tasks must be nonempty strings")
    if len(set(tasks)) != len(tasks):
        raise ValueError("crossed-grounding expected tasks must be unique")
    return tuple(sorted(tasks))


def _parse_identity_rows(
    rows: object,
    *,
    state: str,
    camera_name: str,
) -> dict[str, tuple[int, int, int, int] | None]:
    if state == "absent":
        if not isinstance(rows, list) or any(
            not isinstance(value, str) or not value for value in rows
        ):
            raise ValueError("scene absent identities are malformed")
        if len(set(rows)) != len(rows):
            raise ValueError("scene absent identities contain duplicates")
        return {value: None for value in rows}
    if not isinstance(rows, list):
        raise ValueError(f"scene {state} objects are malformed")
    parsed: dict[str, tuple[int, int, int, int] | None] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"scene {state} object is malformed")
        identity = _text(row.get("identity_key"), name=f"scene {state} identity")
        if identity in parsed:
            raise ValueError(f"scene {state} identities contain duplicates")
        raw_bbox = _bbox(
            row.get("bbox_xyxy"),
            camera_name=camera_name,
            name=f"scene {state} bbox {index}",
        )
        if state == "supervised":
            width, height = _CAMERA_EXTENTS[camera_name]
            parsed[identity] = qwen3vl_normalized_bbox(
                raw_bbox,
                width=width,
                height=height,
            )
        else:
            parsed[identity] = None
    return parsed


def parse_scene_target_evidence(
    scene_audit: Mapping[str, object],
    *,
    target_identity_keys: Sequence[str],
    expected_curriculum_artifact_sha256: str | None = None,
) -> tuple[SceneTargetEvidence, ...]:
    """Parse all target/camera observability states from one signed scene audit."""

    identities = _identity_keys(target_identity_keys)
    artifact_sha256 = _sha256(
        scene_audit.get("artifact_sha256"),
        name="scene audit artifact SHA-256",
    )
    unsigned_scene_audit = {
        key: value for key, value in scene_audit.items() if key != "artifact_sha256"
    }
    if _canonical_sha256(unsigned_scene_audit) != artifact_sha256:
        raise ValueError("scene audit artifact SHA-256 changed")
    if scene_audit.get("schema") != NATIVE_VL_SCENE_AUDIT_SCHEMA:
        raise ValueError("scene audit schema differs from the crossed-grounding contract")
    if scene_audit.get("status") != "PASS":
        raise ValueError("crossed grounding requires a passing scene audit")
    curriculum_artifact = _sha256(
        scene_audit.get("curriculum_artifact_sha256"),
        name="scene audit curriculum artifact SHA-256",
    )
    if expected_curriculum_artifact_sha256 is not None and curriculum_artifact != _sha256(
        expected_curriculum_artifact_sha256,
        name="expected curriculum artifact SHA-256",
    ):
        raise ValueError("scene audit belongs to another grounding curriculum")
    views = scene_audit.get("scene_views")
    if not isinstance(views, list) or not views:
        raise ValueError("scene audit contains no camera views")

    evidence: list[SceneTargetEvidence] = []
    view_keys: set[tuple[int, str]] = set()
    group_sources: dict[int, int] = {}
    for view_index, view in enumerate(views):
        if not isinstance(view, Mapping):
            raise ValueError("scene audit view is malformed")
        group_index = _nonnegative_int(
            view.get("group_index"),
            name=f"scene view {view_index} group index",
        )
        global_index = _nonnegative_int(
            view.get("global_index"),
            name=f"scene view {view_index} global index",
        )
        camera_name = _text(
            view.get("camera_name"),
            name=f"scene view {view_index} camera",
        )
        if camera_name not in CALVIN_GROUNDING_CAMERAS:
            raise ValueError("scene audit contains an unsupported camera")
        view_key = (group_index, camera_name)
        if view_key in view_keys:
            raise ValueError("scene audit repeats one group/camera view")
        view_keys.add(view_key)
        if group_index in group_sources and group_sources[group_index] != global_index:
            raise ValueError("scene audit group changes source frame across cameras")
        group_sources[group_index] = global_index
        source_rgb_sha256 = _sha256(
            view.get("source_rgb_sha256"),
            name=f"scene view {view_index} RGB SHA-256",
        )
        partitions = {
            "supervised": _parse_identity_rows(
                view.get("objects"),
                state="supervised",
                camera_name=camera_name,
            ),
            "subpatch": _parse_identity_rows(
                view.get("subpatch_objects"),
                state="subpatch",
                camera_name=camera_name,
            ),
            "absent": _parse_identity_rows(
                view.get("absent_identity_keys"),
                state="absent",
                camera_name=camera_name,
            ),
        }
        occurrences: Counter[str] = Counter(
            identity for rows in partitions.values() for identity in rows
        )
        if any(count != 1 for count in occurrences.values()):
            raise ValueError("scene view visibility partitions overlap")
        for identity in identities:
            if occurrences[identity] != 1:
                raise ValueError("scene view does not uniquely partition every target identity")
            state = next(state for state, rows in partitions.items() if identity in rows)
            evidence.append(
                SceneTargetEvidence(
                    group_index=group_index,
                    global_index=global_index,
                    camera_name=camera_name,
                    source_rgb_sha256=source_rgb_sha256,
                    identity_key=identity,
                    state=state,
                    bbox_qwen_xyxy=partitions[state][identity],
                )
            )

    cameras = {item.camera_name for item in evidence}
    if cameras != set(CALVIN_GROUNDING_CAMERAS):
        raise ValueError("scene audit does not cover both CALVIN cameras")
    group_indices = set(group_sources)
    if any(
        (group_index, camera) not in view_keys
        for group_index in group_indices
        for camera in cameras
    ):
        raise ValueError("scene audit does not contain both cameras for every group")
    return tuple(evidence)


def _physical_identity_camera_summary(
    rows: Sequence[SceneTargetEvidence],
) -> dict[str, object]:
    supervised = tuple(row for row in rows if row.state == "supervised")
    pair_count = 0
    representative: dict[str, object] | None = None
    maximum_distance = 0.0
    for index, first in enumerate(supervised):
        if first.bbox_qwen_xyxy is None:
            raise RuntimeError("supervised physical evidence lost its bounding box")
        for second in supervised[index + 1 :]:
            if second.bbox_qwen_xyxy is None:
                raise RuntimeError("supervised physical evidence lost its bounding box")
            if (
                first.group_index == second.group_index
                or first.global_index == second.global_index
                or first.source_rgb_sha256 == second.source_rgb_sha256
                or not boxes_are_mutually_centre_exclusive(
                    first.bbox_qwen_xyxy,
                    second.bbox_qwen_xyxy,
                )
            ):
                continue
            pair_count += 1
            distance = math.dist(_centre(first.bbox_qwen_xyxy), _centre(second.bbox_qwen_xyxy))
            maximum_distance = max(maximum_distance, distance)
            candidate = {
                "first_bbox_qwen_xyxy": list(first.bbox_qwen_xyxy),
                "first_global_index": first.global_index,
                "first_group_index": first.group_index,
                "second_bbox_qwen_xyxy": list(second.bbox_qwen_xyxy),
                "second_global_index": second.global_index,
                "second_group_index": second.group_index,
            }
            if representative is None or _canonical_sha256(candidate) < _canonical_sha256(
                representative
            ):
                representative = candidate
    return {
        "absent_count": sum(row.state == "absent" for row in rows),
        "maximum_center_displacement_qwen": maximum_distance,
        "mutually_center_exclusive_source_pair_count": pair_count,
        "representative_pair": representative,
        "source_count": len({row.group_index for row in rows}),
        "subpatch_count": sum(row.state == "subpatch" for row in rows),
        "supervised_count": len(supervised),
        "unique_supervised_bbox_count": len(
            {row.bbox_qwen_xyxy for row in supervised if row.bbox_qwen_xyxy is not None}
        ),
    }


def build_crossed_physical_support_report(
    scene_audit: Mapping[str, object],
    *,
    scene_audit_file_sha256: str,
    target_identity_keys: Sequence[str],
    expected_curriculum_artifact_sha256: str | None = None,
) -> dict[str, object]:
    """Build a canonical pre-model audit of physical pixel-crossing support."""

    identities = _identity_keys(target_identity_keys)
    evidence = parse_scene_target_evidence(
        scene_audit,
        target_identity_keys=identities,
        expected_curriculum_artifact_sha256=expected_curriculum_artifact_sha256,
    )
    summaries: dict[str, dict[str, object]] = {}
    failures: list[str] = []
    for identity in identities:
        by_camera = {}
        for camera in CALVIN_GROUNDING_CAMERAS:
            rows = tuple(
                row
                for row in evidence
                if row.identity_key == identity and row.camera_name == camera
            )
            summary = _physical_identity_camera_summary(rows)
            by_camera[camera] = summary
        summaries[identity] = by_camera
        if not any(
            int(by_camera[camera]["mutually_center_exclusive_source_pair_count"]) > 0
            for camera in CALVIN_GROUNDING_CAMERAS
        ):
            failures.append(f"target {identity} has no physical pixel-causal pair")

    arm_steps = scene_audit.get("arm_steps")
    if not isinstance(arm_steps, list):
        raise ValueError("scene audit arm-step inventory is malformed")
    arm_camera_histogram: Counter[str] = Counter()
    for row in arm_steps:
        if not isinstance(row, Mapping):
            raise ValueError("scene audit arm step is malformed")
        camera = _text(row.get("camera_name"), name="scene audit arm camera")
        if camera not in CALVIN_GROUNDING_CAMERAS:
            raise ValueError("scene audit arm step contains an unsupported camera")
        arm_camera_histogram[camera] += 1

    content: dict[str, object] = {
        "arm_camera_histogram": {
            camera: arm_camera_histogram[camera] for camera in CALVIN_GROUNDING_CAMERAS
        },
        "curriculum_artifact_sha256": _sha256(
            scene_audit.get("curriculum_artifact_sha256"),
            name="scene audit curriculum artifact SHA-256",
        ),
        "failures": failures,
        "identity_camera_support": summaries,
        "scene_audit_artifact_sha256": _sha256(
            scene_audit.get("artifact_sha256"),
            name="scene audit artifact SHA-256",
        ),
        "scene_audit_file_sha256": _sha256(
            scene_audit_file_sha256,
            name="scene audit file SHA-256",
        ),
        "schema": CROSSED_PHYSICAL_SUPPORT_SCHEMA,
        "scope": "physical-pixel-support-only",
        "status": "PASS" if not failures else "FAIL",
        "target_identity_keys": list(identities),
        "training_authorized": False,
        "view_count": len({(row.group_index, row.camera_name) for row in evidence}),
    }
    return {**content, "artifact_sha256": _canonical_sha256(content)}


def materialize_crossed_variant_views(
    groups: Sequence[FixedObservationGroup],
    scene_audit: Mapping[str, object],
    *,
    expected_curriculum_artifact_sha256: str,
) -> tuple[CrossedVariantViewEvidence, ...]:
    """Strictly join every curriculum variant to both audited camera states."""

    typed_groups = tuple(groups)
    if not typed_groups or any(not isinstance(group, FixedObservationGroup) for group in groups):
        raise ValueError("crossed partition requires typed curriculum groups")
    identities = tuple(
        sorted(
            {variant.target_identity_key for group in typed_groups for variant in group.variants}
        )
    )
    scene_rows = parse_scene_target_evidence(
        scene_audit,
        target_identity_keys=identities,
        expected_curriculum_artifact_sha256=expected_curriculum_artifact_sha256,
    )
    scene_index = {(row.group_index, row.camera_name, row.identity_key): row for row in scene_rows}
    scene_group_indices = {row.group_index for row in scene_rows}
    if scene_group_indices != set(range(len(typed_groups))):
        raise ValueError("scene audit group indices differ from the curriculum")

    joined: list[CrossedVariantViewEvidence] = []
    for group_index, group in enumerate(typed_groups):
        sensor_hashes = group.source_sensor_hash_by_field
        for variant in group.variants:
            for camera_name in CALVIN_GROUNDING_CAMERAS:
                scene = scene_index[(group_index, camera_name, variant.target_identity_key)]
                if scene.global_index != group.source_global_index:
                    raise ValueError("scene audit global index differs from the curriculum")
                if scene.source_rgb_sha256 != sensor_hashes[f"rgb_{camera_name}"]:
                    raise ValueError("scene audit RGB digest differs from the curriculum")
                joined.append(
                    CrossedVariantViewEvidence(
                        group_index=group_index,
                        global_index=group.source_global_index,
                        source_episode_index=group.source_episode_index,
                        source_state_sha256=group.source_state_sha256,
                        camera_name=camera_name,
                        source_rgb_sha256=scene.source_rgb_sha256,
                        task_key=variant.task_key,
                        instruction_sha256=variant.instruction_sha256,
                        target_identity_key=variant.target_identity_key,
                        state=scene.state,
                        bbox_qwen_xyxy=scene.bbox_qwen_xyxy,
                    )
                )
    return tuple(joined)


def crossed_variant_views_are_source_disjoint(
    first: CrossedVariantViewEvidence,
    second: CrossedVariantViewEvidence,
) -> bool:
    """Return whether two audited views differ on every registered source identity."""

    return (
        first.group_index != second.group_index
        and first.global_index != second.global_index
        and first.source_episode_index != second.source_episode_index
        and first.source_state_sha256 != second.source_state_sha256
        and first.source_rgb_sha256 != second.source_rgb_sha256
    )


def _representative_variant_pair(
    first: CrossedVariantViewEvidence,
    second: CrossedVariantViewEvidence,
) -> dict[str, object]:
    return {
        "camera_name": first.camera_name,
        "first_bbox_qwen_xyxy": (
            list(first.bbox_qwen_xyxy) if first.bbox_qwen_xyxy is not None else None
        ),
        "first_global_index": first.global_index,
        "first_group_index": first.group_index,
        "first_instruction_sha256": first.instruction_sha256,
        "first_state": first.state,
        "second_bbox_qwen_xyxy": (
            list(second.bbox_qwen_xyxy) if second.bbox_qwen_xyxy is not None else None
        ),
        "second_global_index": second.global_index,
        "second_group_index": second.group_index,
        "second_instruction_sha256": second.instruction_sha256,
        "second_state": second.state,
        "target_identity_key": first.target_identity_key,
        "task_key": first.task_key,
    }


def _summarize_source_pairs(
    rows: Sequence[CrossedVariantViewEvidence],
    *,
    exact_instruction: bool,
    visibility_pair: bool,
) -> dict[str, object]:
    strata: dict[tuple[str, ...], list[CrossedVariantViewEvidence]] = {}
    for row in rows:
        key = (
            row.task_key,
            row.target_identity_key,
            row.camera_name,
            *((row.instruction_sha256,) if exact_instruction else ()),
        )
        strata.setdefault(key, []).append(row)

    pair_count = 0
    representative: dict[str, object] | None = None
    covered_tasks: set[str] = set()
    covered_targets: set[str] = set()
    covered_target_cameras: set[tuple[str, str]] = set()
    covered_instructions: set[str] = set()
    for stratum_rows in strata.values():
        for first, second in combinations(stratum_rows, 2):
            if not crossed_variant_views_are_source_disjoint(first, second):
                continue
            states = {first.state, second.state}
            if visibility_pair:
                if states != {"supervised", "absent"}:
                    continue
            else:
                if states != {"supervised"}:
                    continue
                if first.bbox_qwen_xyxy is None or second.bbox_qwen_xyxy is None:
                    raise RuntimeError("supervised crossed evidence lost its bounding box")
                if not boxes_are_mutually_centre_exclusive(
                    first.bbox_qwen_xyxy,
                    second.bbox_qwen_xyxy,
                ):
                    continue
            pair_count += 1
            covered_tasks.add(first.task_key)
            covered_targets.add(first.target_identity_key)
            covered_target_cameras.add((first.target_identity_key, first.camera_name))
            if first.instruction_sha256 == second.instruction_sha256:
                covered_instructions.add(first.instruction_sha256)
            candidate = _representative_variant_pair(first, second)
            if representative is None or _canonical_sha256(candidate) < _canonical_sha256(
                representative
            ):
                representative = candidate
    return {
        "covered_instruction_count": len(covered_instructions),
        "covered_target_cameras": [
            {"camera_name": camera, "target_identity_key": target}
            for target, camera in sorted(covered_target_cameras)
        ],
        "covered_target_identity_keys": sorted(covered_targets),
        "covered_task_keys": sorted(covered_tasks),
        "pair_count": pair_count,
        "representative_pair": representative,
        "stratum_count": len(strata),
    }


def _prompt_causal_summary(
    rows: Sequence[CrossedVariantViewEvidence],
) -> dict[str, object]:
    strata: dict[tuple[int, str], list[CrossedVariantViewEvidence]] = {}
    for row in rows:
        strata.setdefault((row.group_index, row.camera_name), []).append(row)
    pair_count = 0
    tasks: set[str] = set()
    targets: set[str] = set()
    for stratum_rows in strata.values():
        for first, second in combinations(stratum_rows, 2):
            if (
                first.state != "supervised"
                or second.state != "supervised"
                or first.task_key == second.task_key
                or first.instruction_sha256 == second.instruction_sha256
                or first.target_identity_key == second.target_identity_key
            ):
                continue
            if first.bbox_qwen_xyxy is None or second.bbox_qwen_xyxy is None:
                raise RuntimeError("supervised prompt evidence lost its bounding box")
            if not boxes_are_mutually_centre_exclusive(
                first.bbox_qwen_xyxy,
                second.bbox_qwen_xyxy,
            ):
                continue
            pair_count += 1
            tasks.update((first.task_key, second.task_key))
            targets.update((first.target_identity_key, second.target_identity_key))
    return {
        "covered_target_identity_keys": sorted(targets),
        "covered_task_keys": sorted(tasks),
        "pair_count": pair_count,
        "source_camera_stratum_count": len(strata),
    }


def _camera_transfer_summary(
    rows: Sequence[CrossedVariantViewEvidence],
) -> dict[str, object]:
    strata: dict[tuple[int, str, str, str], set[str]] = {}
    for row in rows:
        if row.state == "supervised":
            key = (
                row.group_index,
                row.task_key,
                row.instruction_sha256,
                row.target_identity_key,
            )
            strata.setdefault(key, set()).add(row.camera_name)
    pairs = tuple(
        key for key, cameras in strata.items() if cameras == set(CALVIN_GROUNDING_CAMERAS)
    )
    return {
        "covered_target_identity_keys": sorted({key[3] for key in pairs}),
        "covered_task_keys": sorted({key[1] for key in pairs}),
        "pair_count": len(pairs),
    }


def _exact_x_episode_graph_summary(
    rows: Sequence[CrossedVariantViewEvidence],
) -> dict[str, object]:
    """Summarize independent episode-pair support without selecting model outputs."""

    strata: dict[tuple[str, str, str, str], list[CrossedVariantViewEvidence]] = {}
    for row in rows:
        strata.setdefault(
            (
                row.task_key,
                row.target_identity_key,
                row.camera_name,
                row.instruction_sha256,
            ),
            [],
        ).append(row)

    task_edges: dict[str, set[tuple[int, int]]] = {}
    target_camera_edges: dict[tuple[str, str], set[tuple[int, int]]] = {}
    for stratum_rows in strata.values():
        for first, second in combinations(stratum_rows, 2):
            if (
                first.state != "supervised"
                or second.state != "supervised"
                or first.bbox_qwen_xyxy is None
                or second.bbox_qwen_xyxy is None
                or not crossed_variant_views_are_source_disjoint(first, second)
                or not boxes_are_mutually_centre_exclusive(
                    first.bbox_qwen_xyxy,
                    second.bbox_qwen_xyxy,
                )
            ):
                continue
            first_episode, second_episode = sorted(
                (first.source_episode_index, second.source_episode_index)
            )
            edge = (first_episode, second_episode)
            task_edges.setdefault(first.task_key, set()).add(edge)
            target_camera_edges.setdefault(
                (first.target_identity_key, first.camera_name),
                set(),
            ).add(edge)

    def rows_for(graphs: Mapping[Any, set[tuple[int, int]]]) -> list[dict[str, object]]:
        summaries = []
        for key, edges in sorted(graphs.items(), key=lambda item: str(item[0])):
            graph = nx.Graph()
            graph.add_edges_from(edges)
            matching = nx.max_weight_matching(graph, maxcardinality=True)
            row: dict[str, object] = {
                "episode_pair_count": len(edges),
                "maximum_source_disjoint_partition_count": len(matching),
                "source_episode_count": len({episode for edge in edges for episode in edge}),
            }
            if isinstance(key, tuple):
                row.update({"camera_name": key[1], "target_identity_key": key[0]})
            else:
                row["task_key"] = key
            summaries.append(row)
        return summaries

    return {
        "by_target_camera": rows_for(target_camera_edges),
        "by_task": rows_for(task_edges),
    }


def _crossed_partition_summary(
    rows: Sequence[CrossedVariantViewEvidence],
) -> dict[str, object]:
    return {
        "camera_transfer_cells": _camera_transfer_summary(rows),
        "null_cells_exact_instruction": _summarize_source_pairs(
            rows,
            exact_instruction=True,
            visibility_pair=True,
        ),
        "pixel_causal_cells_exact_instruction": _summarize_source_pairs(
            rows,
            exact_instruction=True,
            visibility_pair=False,
        ),
        "prompt_causal_cells": _prompt_causal_summary(rows),
        "source_episode_count": len({row.source_episode_index for row in rows}),
        "source_group_count": len({row.group_index for row in rows}),
        "state_histogram": dict(sorted(Counter(row.state for row in rows).items())),
        "variant_view_count": len(rows),
    }


def _summary_coverage(
    summary: Mapping[str, object],
    *,
    cell_name: str,
) -> tuple[set[str], set[str]]:
    cell = summary.get(cell_name)
    if not isinstance(cell, Mapping):
        raise ValueError(f"crossed split {cell_name} summary is malformed")
    return (
        _summary_string_set(
            cell.get("covered_task_keys"),
            name=f"crossed split {cell_name} tasks",
        ),
        _summary_string_set(
            cell.get("covered_target_identity_keys"),
            name=f"crossed split {cell_name} targets",
        ),
    )


def build_crossed_episode_split_report(
    groups: Sequence[FixedObservationGroup],
    scene_audit: Mapping[str, object],
    crossed_support_report: Mapping[str, object],
    *,
    curriculum_artifact_sha256: str,
    curriculum_file_sha256: str,
    scene_audit_file_sha256: str,
    crossed_support_report_file_sha256: str,
    picf_code_revision: str,
    expected_task_keys: Sequence[str],
    expected_target_identity_keys: Sequence[str],
    heldout_source_episode_indices: Sequence[int],
) -> dict[str, object]:
    """Audit one episode-disjoint train/heldout split before schedule selection."""

    expected_tasks = set(_task_keys(expected_task_keys))
    expected_targets = set(_identity_keys(expected_target_identity_keys))
    curriculum_artifact = _sha256(
        curriculum_artifact_sha256,
        name="crossed split curriculum artifact SHA-256",
    )
    support_artifact = _sha256(
        crossed_support_report.get("artifact_sha256"),
        name="crossed split support artifact SHA-256",
    )
    support_content = {
        key: value for key, value in crossed_support_report.items() if key != "artifact_sha256"
    }
    if (
        _canonical_sha256(support_content) != support_artifact
        or crossed_support_report.get("schema") != CROSSED_PARTITION_SUPPORT_SCHEMA
        or crossed_support_report.get("status") != "PASS"
        or crossed_support_report.get("partition") != "training"
        or crossed_support_report.get("curriculum_artifact_sha256") != curriculum_artifact
        or crossed_support_report.get("scene_audit_artifact_sha256")
        != scene_audit.get("artifact_sha256")
    ):
        raise ValueError("crossed split support report differs from its passing provenance")

    heldout = tuple(heldout_source_episode_indices)
    if (
        not heldout
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in heldout
        )
        or tuple(sorted(set(heldout))) != heldout
    ):
        raise ValueError("crossed heldout source episodes must be sorted, unique and nonempty")

    joined = materialize_crossed_variant_views(
        groups,
        scene_audit,
        expected_curriculum_artifact_sha256=curriculum_artifact,
    )
    source_episodes = {row.source_episode_index for row in joined}
    heldout_set = set(heldout)
    if not heldout_set < source_episodes:
        raise ValueError("crossed heldout episodes must be a strict subset of source episodes")
    partitions = {
        "training": tuple(row for row in joined if row.source_episode_index not in heldout_set),
        "heldout": tuple(row for row in joined if row.source_episode_index in heldout_set),
    }

    source_fields = {
        "source_episode_indices": lambda row: row.source_episode_index,
        "source_global_indices": lambda row: row.global_index,
        "source_group_indices": lambda row: row.group_index,
        "source_rgb_sha256s": lambda row: row.source_rgb_sha256,
        "source_state_sha256s": lambda row: row.source_state_sha256,
    }
    disjointness = {}
    for field, getter in source_fields.items():
        training_values = {getter(row) for row in partitions["training"]}
        heldout_values = {getter(row) for row in partitions["heldout"]}
        intersection = training_values & heldout_values
        disjointness[field] = {
            "heldout_count": len(heldout_values),
            "intersection_count": len(intersection),
            "training_count": len(training_values),
        }
        if intersection:
            raise ValueError(f"crossed split leaks {field} across training and heldout")

    summaries = {name: _crossed_partition_summary(rows) for name, rows in partitions.items()}
    failures: list[str] = []
    for partition, required_cells in {
        "training": (
            "prompt_causal_cells",
            "pixel_causal_cells_exact_instruction",
            "camera_transfer_cells",
        ),
        "heldout": ("prompt_causal_cells", "pixel_causal_cells_exact_instruction"),
    }.items():
        for cell_name in required_cells:
            tasks, targets = _summary_coverage(summaries[partition], cell_name=cell_name)
            if tasks != expected_tasks:
                failures.append(
                    f"{partition} {cell_name} misses tasks: "
                    + ",".join(sorted(expected_tasks - tasks))
                )
            if targets != expected_targets:
                failures.append(
                    f"{partition} {cell_name} misses targets: "
                    + ",".join(sorted(expected_targets - targets))
                )

    full_exact = _summarize_source_pairs(
        joined,
        exact_instruction=True,
        visibility_pair=False,
    )
    support_exact = crossed_support_report.get("pixel_causal_cells_exact_instruction")
    if support_exact != full_exact:
        raise ValueError("crossed split recomputation differs from the support report")
    required_target_cameras = _summary_target_cameras(full_exact["covered_target_cameras"])
    training_exact = summaries["training"]["pixel_causal_cells_exact_instruction"]
    if not isinstance(training_exact, Mapping):
        raise ValueError("crossed split training exact-X summary is malformed")
    training_target_cameras = _summary_target_cameras(training_exact.get("covered_target_cameras"))
    missing_training_target_cameras = required_target_cameras - training_target_cameras
    if missing_training_target_cameras:
        failures.append(
            "training exact-X misses globally supported target/cameras: "
            + ",".join(
                f"{target}@{camera}" for target, camera in sorted(missing_training_target_cameras)
            )
        )

    heldout_camera = summaries["heldout"]["camera_transfer_cells"]
    if not isinstance(heldout_camera, Mapping):
        raise ValueError("crossed split heldout camera summary is malformed")
    heldout_camera_tasks = _summary_string_set(
        heldout_camera.get("covered_task_keys"),
        name="crossed split heldout camera tasks",
    )
    heldout_camera_targets = _summary_string_set(
        heldout_camera.get("covered_target_identity_keys"),
        name="crossed split heldout camera targets",
    )
    if heldout_camera_targets != expected_targets:
        failures.append(
            "heldout camera-transfer cells miss targets: "
            + ",".join(sorted(expected_targets - heldout_camera_targets))
        )

    episode_graph = _exact_x_episode_graph_summary(joined)
    content: dict[str, object] = {
        "crossed_support_artifact_sha256": support_artifact,
        "crossed_support_file_sha256": _sha256(
            crossed_support_report_file_sha256,
            name="crossed split support file SHA-256",
        ),
        "curriculum_artifact_sha256": curriculum_artifact,
        "curriculum_file_sha256": _sha256(
            curriculum_file_sha256,
            name="crossed split curriculum file SHA-256",
        ),
        "disjointness": disjointness,
        "exact_x_episode_graph": episode_graph,
        "failures": failures,
        "heldout_camera_transfer_missing_task_keys": sorted(expected_tasks - heldout_camera_tasks),
        "heldout_source_episode_indices": list(heldout),
        "partitions": summaries,
        "picf_code_revision": _git_revision(
            picf_code_revision,
            name="crossed split PICF code revision",
        ),
        "scene_audit_artifact_sha256": _sha256(
            scene_audit.get("artifact_sha256"),
            name="crossed split scene artifact SHA-256",
        ),
        "scene_audit_file_sha256": _sha256(
            scene_audit_file_sha256,
            name="crossed split scene file SHA-256",
        ),
        "schema": CROSSED_EPISODE_SPLIT_SCHEMA,
        "selection_basis": "predeclared-data-only-source-episode-partition",
        "status": "PASS" if not failures else "FAIL",
        "training_authorized": False,
        "validation_claim": "no-complete-third-calvin-exact-x-partition",
    }
    return {**content, "artifact_sha256": _canonical_sha256(content)}


def _summary_string_set(value: object, *, name: str) -> set[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise ValueError(f"{name} is malformed")
    return set(value)


def _summary_target_cameras(value: object) -> set[tuple[str, str]]:
    if not isinstance(value, list):
        raise ValueError("crossed target-camera summary is malformed")
    result: set[tuple[str, str]] = set()
    for row in value:
        if not isinstance(row, Mapping) or set(row) != {
            "camera_name",
            "target_identity_key",
        }:
            raise ValueError("crossed target-camera row is malformed")
        camera = _text(row["camera_name"], name="crossed summary camera")
        if camera not in CALVIN_GROUNDING_CAMERAS:
            raise ValueError("crossed summary camera is unsupported")
        target = _text(row["target_identity_key"], name="crossed summary target")
        result.add((target, camera))
    return result


def build_crossed_partition_support_report(
    groups: Sequence[FixedObservationGroup],
    scene_audit: Mapping[str, object],
    *,
    curriculum_artifact_sha256: str,
    curriculum_file_sha256: str,
    scene_audit_file_sha256: str,
    expected_task_keys: Sequence[str],
    expected_target_identity_keys: Sequence[str],
) -> dict[str, object]:
    """Audit observed P/X/N/C support in one immutable training curriculum."""

    expected_tasks = _task_keys(expected_task_keys)
    expected_targets = _identity_keys(expected_target_identity_keys)
    curriculum_artifact = _sha256(
        curriculum_artifact_sha256,
        name="crossed curriculum artifact SHA-256",
    )
    joined = materialize_crossed_variant_views(
        groups,
        scene_audit,
        expected_curriculum_artifact_sha256=curriculum_artifact,
    )
    task_keys = tuple(sorted({row.task_key for row in joined}))
    target_keys = tuple(sorted({row.target_identity_key for row in joined}))
    failures: list[str] = []
    if task_keys != expected_tasks:
        missing = sorted(set(expected_tasks) - set(task_keys))
        unexpected = sorted(set(task_keys) - set(expected_tasks))
        failures.append(
            "task inventory differs from the declared contract: "
            f"missing={','.join(missing)};unexpected={','.join(unexpected)}"
        )
    if target_keys != expected_targets:
        failures.append("target inventory differs from the declared direct identities")

    prompt = _prompt_causal_summary(joined)
    pixel_exact = _summarize_source_pairs(
        joined,
        exact_instruction=True,
        visibility_pair=False,
    )
    pixel_task_semantic = _summarize_source_pairs(
        joined,
        exact_instruction=False,
        visibility_pair=False,
    )
    null_exact = _summarize_source_pairs(
        joined,
        exact_instruction=True,
        visibility_pair=True,
    )
    null_task_semantic = _summarize_source_pairs(
        joined,
        exact_instruction=False,
        visibility_pair=True,
    )
    camera_transfer = _camera_transfer_summary(joined)

    prompt_tasks = _summary_string_set(
        prompt["covered_task_keys"],
        name="prompt-causal covered tasks",
    )
    prompt_targets = _summary_string_set(
        prompt["covered_target_identity_keys"],
        name="prompt-causal covered targets",
    )
    exact_tasks = _summary_string_set(
        pixel_exact["covered_task_keys"],
        name="pixel-causal covered tasks",
    )
    if prompt_tasks != set(task_keys):
        failures.append("prompt-causal P cells do not cover every task")
    if prompt_targets != set(expected_targets):
        failures.append("prompt-causal P cells do not cover every target")
    if exact_tasks != set(task_keys):
        missing = sorted(set(task_keys) - exact_tasks)
        failures.append(f"strict exact-instruction X cells miss tasks: {','.join(missing)}")
    camera_tasks = _summary_string_set(
        camera_transfer["covered_task_keys"],
        name="camera-transfer covered tasks",
    )
    camera_targets = _summary_string_set(
        camera_transfer["covered_target_identity_keys"],
        name="camera-transfer covered targets",
    )
    if camera_tasks != set(task_keys):
        missing = sorted(set(task_keys) - camera_tasks)
        failures.append(f"camera-transfer C cells miss tasks: {','.join(missing)}")
    if camera_targets != set(expected_targets):
        missing = sorted(set(expected_targets) - camera_targets)
        failures.append(f"camera-transfer C cells miss targets: {','.join(missing)}")

    physical = build_crossed_physical_support_report(
        scene_audit,
        scene_audit_file_sha256=scene_audit_file_sha256,
        target_identity_keys=expected_targets,
        expected_curriculum_artifact_sha256=curriculum_artifact,
    )
    if physical["status"] != "PASS":
        physical_failures = physical["failures"]
        if not isinstance(physical_failures, list) or any(
            not isinstance(value, str) for value in physical_failures
        ):
            raise ValueError("physical crossed-support failures are malformed")
        failures.extend(f"physical support: {value}" for value in physical_failures)
    physical_support = physical["identity_camera_support"]
    if not isinstance(physical_support, Mapping):
        raise ValueError("physical target-camera support is malformed")
    feasible_target_cameras = {
        (identity, camera)
        for identity, cameras in physical_support.items()
        if isinstance(identity, str) and isinstance(cameras, Mapping)
        for camera, summary in cameras.items()
        if isinstance(camera, str)
        and isinstance(summary, Mapping)
        and isinstance(summary.get("mutually_center_exclusive_source_pair_count"), int)
        and summary["mutually_center_exclusive_source_pair_count"] > 0
    }
    exact_target_cameras = _summary_target_cameras(pixel_exact["covered_target_cameras"])
    if not feasible_target_cameras.issubset(exact_target_cameras):
        missing_pairs = sorted(feasible_target_cameras - exact_target_cameras)
        failures.append(
            "strict exact-instruction X cells miss physically feasible target/cameras: "
            + ",".join(f"{target}@{camera}" for target, camera in missing_pairs)
        )

    state_histogram = Counter(row.state for row in joined)
    content: dict[str, object] = {
        "camera_transfer_cells": camera_transfer,
        "curriculum_artifact_sha256": curriculum_artifact,
        "curriculum_file_sha256": _sha256(
            curriculum_file_sha256,
            name="crossed curriculum file SHA-256",
        ),
        "expected_target_identity_keys": list(expected_targets),
        "expected_task_keys": list(expected_tasks),
        "failures": failures,
        "null_cells_exact_instruction": null_exact,
        "null_cells_task_semantic_only": null_task_semantic,
        "partition": "training",
        "physical_support_artifact_sha256": physical["artifact_sha256"],
        "pixel_causal_cells_exact_instruction": pixel_exact,
        "pixel_causal_cells_task_semantic_only": pixel_task_semantic,
        "prompt_causal_cells": prompt,
        "scene_audit_artifact_sha256": physical["scene_audit_artifact_sha256"],
        "scene_audit_file_sha256": physical["scene_audit_file_sha256"],
        "schema": CROSSED_PARTITION_SUPPORT_SCHEMA,
        "scope": "training-partition-observed-support",
        "state_histogram": dict(sorted(state_histogram.items())),
        "status": "PASS" if not failures else "FAIL",
        "target_identity_keys": list(target_keys),
        "task_keys": list(task_keys),
        "training_authorized": False,
        "variant_view_count": len(joined),
    }
    return {**content, "artifact_sha256": _canonical_sha256(content)}


def crossed_support_report_bytes(report: Mapping[str, Any]) -> bytes:
    """Serialize one already-built report without weakening its artifact hash."""

    if report.get("schema") not in {
        CROSSED_EPISODE_SPLIT_SCHEMA,
        CROSSED_PARTITION_SUPPORT_SCHEMA,
        CROSSED_PHYSICAL_SUPPORT_SCHEMA,
    }:
        raise ValueError("crossed-support report schema differs from the writer")
    artifact = _sha256(report.get("artifact_sha256"), name="crossed-support artifact SHA-256")
    content = {key: value for key, value in report.items() if key != "artifact_sha256"}
    if _canonical_sha256(content) != artifact:
        raise ValueError("crossed-support report artifact SHA-256 changed")
    return _canonical_bytes(dict(report)) + b"\n"

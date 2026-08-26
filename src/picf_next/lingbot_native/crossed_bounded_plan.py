"""Frozen matched candidate/control schedule for crossed native grounding."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import Bounds, LinearConstraint, linear_sum_assignment, milp
from scipy.sparse import lil_matrix

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.crossed_causal_grounding import (
    CALVIN_GROUNDING_CAMERAS,
    CROSSED_EPISODE_SPLIT_SCHEMA,
    CrossedVariantViewEvidence,
    boxes_are_mutually_centre_exclusive,
    crossed_variant_views_are_source_disjoint,
    materialize_crossed_variant_views,
)
from picf_next.lingbot_native.fixed_observation import (
    FixedObservationGroup,
    FixedObservationVariant,
)
from picf_next.lingbot_native.vl_curriculum import NativeVLGroundingCurriculumPlan

CROSSED_BOUNDED_PLAN_SCHEMA = "picf-next.crossed-grounding-bounded-plan.v1"
CROSSED_BOUNDED_PLAN_ALGORITHM = "matched-milp-p32-x32-exact-vs-unique-prompt.v1"
CROSSED_BOUNDED_PLAN_MAXIMUM_BYTES = 8 * 1024 * 1024
CROSSED_BOUNDED_TOTAL_STEPS = 64
CROSSED_BOUNDED_CELL_STEPS = 32
CROSSED_BOUNDED_CAMERA_STEPS = 16

CrossedArm = Literal["candidate", "control"]
CrossedCell = Literal["P", "X"]


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
        raise ValueError("crossed bounded plan is not canonical finite JSON") from error


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _git_revision(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase Git revision")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: object, *, name: str) -> int:
    result = _nonnegative_int(value, name=name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def _sorted_unique_text(values: Sequence[str], *, name: str) -> tuple[str, ...]:
    result = tuple(values)
    if (
        not result
        or any(not isinstance(value, str) or not value for value in result)
        or tuple(sorted(set(result))) != result
    ):
        raise ValueError(f"{name} must be sorted, unique and nonempty")
    return result


def _qwen_bbox(value: object, *, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, str | bytes)
        or len(value) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in value)
    ):
        raise ValueError(f"{name} must contain four integers")
    result = (value[0], value[1], value[2], value[3])
    if not (0 <= result[0] < result[2] <= 1000 and 0 <= result[1] < result[3] <= 1000):
        raise ValueError(f"{name} is outside Qwen's coordinate grid")
    return result


@dataclass(frozen=True, slots=True)
class CrossedBoundedRecord:
    """One schedule row rebound to immutable curriculum and scene evidence."""

    group_index: int
    variant_index: int
    global_index: int
    source_episode_index: int
    source_state_sha256: str
    camera_name: str
    source_rgb_sha256: str
    task_key: str
    instruction_sha256: str
    target_identity_key: str
    bbox_qwen_xyxy: tuple[int, int, int, int]

    def __post_init__(self) -> None:
        _nonnegative_int(self.group_index, name="crossed bounded group index")
        _nonnegative_int(self.variant_index, name="crossed bounded variant index")
        _nonnegative_int(self.global_index, name="crossed bounded global index")
        _nonnegative_int(self.source_episode_index, name="crossed bounded source episode")
        _sha256(self.source_state_sha256, name="crossed bounded source-state SHA-256")
        if self.camera_name not in CALVIN_GROUNDING_CAMERAS:
            raise ValueError("crossed bounded record has an unsupported camera")
        _sha256(self.source_rgb_sha256, name="crossed bounded RGB SHA-256")
        _text(self.task_key, name="crossed bounded task")
        _sha256(self.instruction_sha256, name="crossed bounded instruction SHA-256")
        _text(self.target_identity_key, name="crossed bounded target")
        _qwen_bbox(self.bbox_qwen_xyxy, name="crossed bounded bbox")

    @property
    def source_key(self) -> tuple[int, int, int, str, str]:
        return (
            self.group_index,
            self.global_index,
            self.source_episode_index,
            self.source_state_sha256,
            self.source_rgb_sha256,
        )

    @property
    def task_camera(self) -> tuple[str, str]:
        return (self.task_key, self.camera_name)

    def as_dict(self) -> dict[str, object]:
        return {
            "bbox_qwen_xyxy": list(self.bbox_qwen_xyxy),
            "camera_name": self.camera_name,
            "global_index": self.global_index,
            "group_index": self.group_index,
            "instruction_sha256": self.instruction_sha256,
            "source_episode_index": self.source_episode_index,
            "source_rgb_sha256": self.source_rgb_sha256,
            "source_state_sha256": self.source_state_sha256,
            "target_identity_key": self.target_identity_key,
            "task_key": self.task_key,
            "variant_index": self.variant_index,
        }

    @classmethod
    def from_dict(cls, value: object) -> CrossedBoundedRecord:
        expected = {
            "bbox_qwen_xyxy",
            "camera_name",
            "global_index",
            "group_index",
            "instruction_sha256",
            "source_episode_index",
            "source_rgb_sha256",
            "source_state_sha256",
            "target_identity_key",
            "task_key",
            "variant_index",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("crossed bounded record fields differ from schema")
        return cls(
            group_index=_nonnegative_int(value["group_index"], name="group index"),
            variant_index=_nonnegative_int(value["variant_index"], name="variant index"),
            global_index=_nonnegative_int(value["global_index"], name="global index"),
            source_episode_index=_nonnegative_int(
                value["source_episode_index"], name="source episode"
            ),
            source_state_sha256=_sha256(value["source_state_sha256"], name="state SHA-256"),
            camera_name=_text(value["camera_name"], name="camera"),
            source_rgb_sha256=_sha256(value["source_rgb_sha256"], name="RGB SHA-256"),
            task_key=_text(value["task_key"], name="task"),
            instruction_sha256=_sha256(value["instruction_sha256"], name="instruction SHA-256"),
            target_identity_key=_text(value["target_identity_key"], name="target"),
            bbox_qwen_xyxy=_qwen_bbox(value["bbox_qwen_xyxy"], name="bbox"),
        )


_RecordPair = tuple[CrossedBoundedRecord, CrossedBoundedRecord]


def _record_order(record: CrossedBoundedRecord) -> tuple[object, ...]:
    return (
        record.task_key,
        record.camera_name,
        record.instruction_sha256,
        record.group_index,
        record.variant_index,
    )


def _ordered_record_pair(records: _RecordPair) -> _RecordPair:
    ordered = sorted(records, key=_record_order)
    return ordered[0], ordered[1]


def _reverse_record_pair(records: _RecordPair) -> _RecordPair:
    return records[1], records[0]


def _validate_source_disjoint_pair(
    records: _RecordPair,
) -> None:
    first, second = records
    if (
        first.group_index == second.group_index
        or first.global_index == second.global_index
        or first.source_episode_index == second.source_episode_index
        or first.source_state_sha256 == second.source_state_sha256
        or first.source_rgb_sha256 == second.source_rgb_sha256
    ):
        raise ValueError("crossed bounded X pair is not source-disjoint")
    if not boxes_are_mutually_centre_exclusive(
        first.bbox_qwen_xyxy,
        second.bbox_qwen_xyxy,
    ):
        raise ValueError("crossed bounded X pair boxes are not mutually exclusive")


@dataclass(frozen=True, slots=True)
class CrossedBoundedStep:
    """One two-rank update for both matched experimental arms."""

    optimizer_step: int
    cell: CrossedCell
    candidate_records: _RecordPair
    control_records: _RecordPair

    def __post_init__(self) -> None:
        _nonnegative_int(self.optimizer_step, name="crossed bounded optimizer step")
        if self.cell not in {"P", "X"}:
            raise ValueError("crossed bounded step has an unsupported cell")
        for records in (self.candidate_records, self.control_records):
            if (
                not isinstance(records, tuple)
                or len(records) != 2
                or any(not isinstance(record, CrossedBoundedRecord) for record in records)
            ):
                raise ValueError("crossed bounded step requires two typed records per arm")
        candidate = self.candidate_records
        control = self.control_records
        if self.cell == "P":
            if candidate != control:
                raise ValueError("crossed bounded P records must be identical across arms")
            first, second = candidate
            if (
                first.group_index != second.group_index
                or first.global_index != second.global_index
                or first.source_episode_index != second.source_episode_index
                or first.source_state_sha256 != second.source_state_sha256
                or first.camera_name != second.camera_name
                or first.source_rgb_sha256 != second.source_rgb_sha256
                or first.task_key == second.task_key
                or first.instruction_sha256 == second.instruction_sha256
                or first.target_identity_key == second.target_identity_key
                or not boxes_are_mutually_centre_exclusive(
                    first.bbox_qwen_xyxy,
                    second.bbox_qwen_xyxy,
                )
            ):
                raise ValueError("crossed bounded P pair violates fixed-image prompt causality")
            return
        _validate_source_disjoint_pair(candidate)
        _validate_source_disjoint_pair(control)
        if not (
            candidate[0].task_key
            == candidate[1].task_key
            == control[0].task_key
            == control[1].task_key
            and candidate[0].target_identity_key
            == candidate[1].target_identity_key
            == control[0].target_identity_key
            == control[1].target_identity_key
            and candidate[0].camera_name
            == candidate[1].camera_name
            == control[0].camera_name
            == control[1].camera_name
            and candidate[0].instruction_sha256 == candidate[1].instruction_sha256
            and control[0].instruction_sha256 != control[1].instruction_sha256
        ):
            raise ValueError("crossed bounded X arms are not semantically matched")

    def records_for_arm(
        self,
        arm: CrossedArm,
    ) -> tuple[CrossedBoundedRecord, CrossedBoundedRecord]:
        if arm == "candidate":
            return self.candidate_records
        if arm == "control":
            return self.control_records
        raise ValueError("crossed bounded arm is unsupported")

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_records": [record.as_dict() for record in self.candidate_records],
            "cell": self.cell,
            "control_records": [record.as_dict() for record in self.control_records],
            "optimizer_step": self.optimizer_step,
        }

    @classmethod
    def from_dict(cls, value: object) -> CrossedBoundedStep:
        expected = {"candidate_records", "cell", "control_records", "optimizer_step"}
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("crossed bounded step fields differ from schema")
        candidate = value["candidate_records"]
        control = value["control_records"]
        if not isinstance(candidate, list) or not isinstance(control, list):
            raise ValueError("crossed bounded step records are malformed")
        return cls(
            optimizer_step=_nonnegative_int(value["optimizer_step"], name="optimizer step"),
            cell=_text(value["cell"], name="cell"),  # type: ignore[arg-type]
            candidate_records=tuple(  # type: ignore[arg-type]
                CrossedBoundedRecord.from_dict(record) for record in candidate
            ),
            control_records=tuple(  # type: ignore[arg-type]
                CrossedBoundedRecord.from_dict(record) for record in control
            ),
        )


def _histogram(values: Sequence[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


@dataclass(frozen=True, slots=True)
class CrossedBoundedPlan:
    """A content-addressed bounded experiment; never a long-run authorization."""

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    curriculum_file_sha256: str
    curriculum_artifact_sha256: str
    scene_audit_file_sha256: str
    scene_audit_artifact_sha256: str
    episode_split_file_sha256: str
    episode_split_artifact_sha256: str
    episode_split_picf_code_revision: str
    picf_code_revision: str
    expected_task_keys: tuple[str, ...]
    expected_target_identity_keys: tuple[str, ...]
    expected_x_task_camera_strata: tuple[tuple[str, str], ...]
    heldout_source_episode_indices: tuple[int, ...]
    maximum_control_x_source_group_overlap_count: int
    steps: tuple[CrossedBoundedStep, ...]

    def __post_init__(self) -> None:
        _text(self.dataset_id, name="crossed bounded dataset ID")
        _text(self.dataset_revision, name="crossed bounded dataset revision")
        for name in (
            "dataset_manifest_sha256",
            "curriculum_file_sha256",
            "curriculum_artifact_sha256",
            "scene_audit_file_sha256",
            "scene_audit_artifact_sha256",
            "episode_split_file_sha256",
            "episode_split_artifact_sha256",
        ):
            _sha256(getattr(self, name), name=f"crossed bounded {name}")
        _git_revision(
            self.episode_split_picf_code_revision,
            name="crossed bounded split PICF revision",
        )
        _git_revision(self.picf_code_revision, name="crossed bounded PICF revision")
        tasks = _sorted_unique_text(self.expected_task_keys, name="crossed bounded tasks")
        targets = _sorted_unique_text(
            self.expected_target_identity_keys,
            name="crossed bounded targets",
        )
        if (
            not self.expected_x_task_camera_strata
            or tuple(sorted(set(self.expected_x_task_camera_strata)))
            != self.expected_x_task_camera_strata
            or any(
                task not in tasks or camera not in CALVIN_GROUNDING_CAMERAS
                for task, camera in self.expected_x_task_camera_strata
            )
        ):
            raise ValueError("crossed bounded X strata are malformed")
        if (
            not self.heldout_source_episode_indices
            or tuple(sorted(set(self.heldout_source_episode_indices)))
            != self.heldout_source_episode_indices
        ):
            raise ValueError("crossed bounded heldout episodes are malformed")
        maximum_overlap = _positive_int(
            self.maximum_control_x_source_group_overlap_count,
            name="crossed bounded maximum X source overlap",
        )
        if maximum_overlap > CROSSED_BOUNDED_CELL_STEPS * 2:
            raise ValueError("crossed bounded maximum X source overlap is impossible")
        if (
            not isinstance(self.steps, tuple)
            or len(self.steps) != CROSSED_BOUNDED_TOTAL_STEPS
            or any(not isinstance(step, CrossedBoundedStep) for step in self.steps)
            or tuple(step.optimizer_step for step in self.steps)
            != tuple(range(CROSSED_BOUNDED_TOTAL_STEPS))
            or tuple(step.cell for step in self.steps) != ("P", "X") * CROSSED_BOUNDED_CELL_STEPS
        ):
            raise ValueError("crossed bounded steps differ from the frozen interleave")
        self._validate_measure(tasks=tasks, targets=targets, maximum_overlap=maximum_overlap)

    def _validate_measure(
        self,
        *,
        tasks: tuple[str, ...],
        targets: tuple[str, ...],
        maximum_overlap: int,
    ) -> None:
        p_steps = tuple(step for step in self.steps if step.cell == "P")
        x_steps = tuple(step for step in self.steps if step.cell == "X")
        for steps, name in ((p_steps, "P"), (x_steps, "X")):
            if len(steps) != CROSSED_BOUNDED_CELL_STEPS:
                raise ValueError(f"crossed bounded {name} step count changed")
            cameras = Counter(step.candidate_records[0].camera_name for step in steps)
            expected_cameras = Counter(
                {camera: CROSSED_BOUNDED_CAMERA_STEPS for camera in CALVIN_GROUNDING_CAMERAS}
            )
            if cameras != expected_cameras:
                raise ValueError(f"crossed bounded {name} camera balance changed")

        p_records = tuple(record for step in p_steps for record in step.candidate_records)
        candidate_x = tuple(record for step in x_steps for record in step.candidate_records)
        control_x = tuple(record for step in x_steps for record in step.control_records)
        if {record.task_key for record in p_records} != set(tasks) or {
            record.task_key for record in candidate_x
        } != set(tasks):
            raise ValueError("crossed bounded arms do not cover every expected task")
        if {record.target_identity_key for record in p_records} != set(targets) or {
            record.target_identity_key for record in candidate_x
        } != set(targets):
            raise ValueError("crossed bounded arms do not cover every expected target")
        selected_x_strata = tuple(sorted({record.task_camera for record in candidate_x}))
        if selected_x_strata != self.expected_x_task_camera_strata:
            raise ValueError("crossed bounded candidate omits a supported X stratum")
        if Counter(record.task_camera for record in candidate_x) != Counter(
            record.task_camera for record in control_x
        ):
            raise ValueError("crossed bounded X arms have different semantic histograms")

        p_instructions = Counter(record.instruction_sha256 for record in p_records)
        candidate_instructions = Counter(record.instruction_sha256 for record in candidate_x)
        control_instructions = Counter(record.instruction_sha256 for record in control_x)
        if set(p_instructions.values()) != {1} or len(p_instructions) != 64:
            raise ValueError("crossed bounded P instructions must be globally unique")
        if set(candidate_instructions.values()) != {2} or len(candidate_instructions) != 32:
            raise ValueError("crossed bounded candidate X must repeat exactly 32 prompts")
        if set(control_instructions.values()) != {1} or len(control_instructions) != 64:
            raise ValueError("crossed bounded control X prompts must be globally unique")
        if set(p_instructions) & (set(candidate_instructions) | set(control_instructions)):
            raise ValueError("crossed bounded P and X instruction supports overlap")

        p_groups = {record.group_index for record in p_records}
        candidate_groups = {record.group_index for record in candidate_x}
        control_groups = {record.group_index for record in control_x}
        if (len(p_groups), len(candidate_groups), len(control_groups)) != (32, 64, 64):
            raise ValueError("crossed bounded source-group budgets changed")
        if p_groups & (candidate_groups | control_groups):
            raise ValueError("crossed bounded P and X source groups overlap")
        overlap = len(candidate_groups & control_groups)
        if overlap != maximum_overlap:
            raise ValueError("crossed bounded control does not attain maximum source overlap")
        if len(p_groups | candidate_groups) != 96 or len(p_groups | control_groups) != 96:
            raise ValueError("crossed bounded arm source budgets differ")

        heldout = set(self.heldout_source_episode_indices)
        all_records = tuple(
            record
            for step in self.steps
            for records in (step.candidate_records, step.control_records)
            for record in records
        )
        if any(record.source_episode_index in heldout for record in all_records):
            raise ValueError("crossed bounded plan leaks a heldout source episode")
        source_metadata: dict[tuple[int, str], tuple[int, int, str, str]] = {}
        for record in all_records:
            key = (record.group_index, record.camera_name)
            value = (
                record.global_index,
                record.source_episode_index,
                record.source_state_sha256,
                record.source_rgb_sha256,
            )
            if key in source_metadata and source_metadata[key] != value:
                raise ValueError("crossed bounded source metadata changes within one view")
            source_metadata[key] = value
        for rank in range(2):
            candidate_rank = Counter(
                (
                    step.candidate_records[rank].task_key,
                    step.candidate_records[rank].target_identity_key,
                    step.candidate_records[rank].camera_name,
                )
                for step in self.steps
            )
            control_rank = Counter(
                (
                    step.control_records[rank].task_key,
                    step.control_records[rank].target_identity_key,
                    step.control_records[rank].camera_name,
                )
                for step in self.steps
            )
            if candidate_rank != control_rank:
                raise ValueError("crossed bounded per-rank semantic measures differ")

    @property
    def summary(self) -> dict[str, object]:
        p_steps = tuple(step for step in self.steps if step.cell == "P")
        x_steps = tuple(step for step in self.steps if step.cell == "X")
        candidate_x = tuple(record for step in x_steps for record in step.candidate_records)
        control_x = tuple(record for step in x_steps for record in step.control_records)
        p_records = tuple(record for step in p_steps for record in step.candidate_records)
        candidate_groups = {record.group_index for record in candidate_x}
        control_groups = {record.group_index for record in control_x}
        return {
            "arm_calvin_record_count": CROSSED_BOUNDED_TOTAL_STEPS * 2,
            "arm_unique_source_group_count": 96,
            "candidate_exact_x_instruction_count": 32,
            "candidate_x_source_group_count": len(candidate_groups),
            "cell_camera_step_histogram": {
                cell: _histogram(
                    [
                        step.candidate_records[0].camera_name
                        for step in self.steps
                        if step.cell == cell
                    ]
                )
                for cell in ("P", "X")
            },
            "control_unique_x_instruction_count": len(
                {record.instruction_sha256 for record in control_x}
            ),
            "control_x_source_group_count": len(control_groups),
            "maximum_control_x_source_group_overlap_count": (
                self.maximum_control_x_source_group_overlap_count
            ),
            "p_instruction_count": len({record.instruction_sha256 for record in p_records}),
            "p_source_group_count": len({record.group_index for record in p_records}),
            "selected_control_x_source_group_overlap_count": len(candidate_groups & control_groups),
            "x_task_camera_step_histogram": {
                f"{task}@{camera}": count
                for (task, camera), count in sorted(
                    Counter(step.candidate_records[0].task_camera for step in x_steps).items()
                )
            },
        }

    def as_dict(self) -> dict[str, object]:
        content: dict[str, object] = {
            "algorithm": CROSSED_BOUNDED_PLAN_ALGORITHM,
            "bounded_training_authorized": True,
            "curriculum_artifact_sha256": self.curriculum_artifact_sha256,
            "curriculum_file_sha256": self.curriculum_file_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "episode_split_artifact_sha256": self.episode_split_artifact_sha256,
            "episode_split_file_sha256": self.episode_split_file_sha256,
            "episode_split_picf_code_revision": self.episode_split_picf_code_revision,
            "expected_target_identity_keys": list(self.expected_target_identity_keys),
            "expected_task_keys": list(self.expected_task_keys),
            "expected_x_task_camera_strata": [
                {"camera_name": camera, "task_key": task}
                for task, camera in self.expected_x_task_camera_strata
            ],
            "heldout_source_episode_indices": list(self.heldout_source_episode_indices),
            "long_training_authorized": False,
            "maximum_control_x_source_group_overlap_count": (
                self.maximum_control_x_source_group_overlap_count
            ),
            "picf_code_revision": self.picf_code_revision,
            "scene_audit_artifact_sha256": self.scene_audit_artifact_sha256,
            "scene_audit_file_sha256": self.scene_audit_file_sha256,
            "schema": CROSSED_BOUNDED_PLAN_SCHEMA,
            "selection_basis": "pre-model-data-only-matched-milp",
            "steps": [step.as_dict() for step in self.steps],
            "summary": self.summary,
        }
        return {**content, "artifact_sha256": _canonical_sha256(content)}

    @property
    def artifact_sha256(self) -> str:
        return str(self.as_dict()["artifact_sha256"])

    def write(self, path: str | Path) -> None:
        write_bytes_durable_exclusive(Path(path), _canonical_bytes(self.as_dict()))

    def resolve_record(
        self,
        groups: Sequence[FixedObservationGroup],
        record: CrossedBoundedRecord,
    ) -> tuple[FixedObservationGroup, FixedObservationVariant]:
        if record.group_index >= len(groups):
            raise ValueError("crossed bounded group index is outside the curriculum")
        group = groups[record.group_index]
        if record.variant_index >= len(group.variants):
            raise ValueError("crossed bounded variant index is outside the source group")
        variant = group.variants[record.variant_index]
        if (
            group.source_global_index != record.global_index
            or group.source_episode_index != record.source_episode_index
            or group.source_state_sha256 != record.source_state_sha256
            or group.source_sensor_hash_by_field[f"rgb_{record.camera_name}"]
            != record.source_rgb_sha256
            or variant.task_key != record.task_key
            or variant.instruction_sha256 != record.instruction_sha256
            or variant.target_identity_key != record.target_identity_key
        ):
            raise ValueError("crossed bounded record differs from its source curriculum")
        return group, variant

    @classmethod
    def from_dict(cls, value: object) -> CrossedBoundedPlan:
        expected = {
            "algorithm",
            "artifact_sha256",
            "bounded_training_authorized",
            "curriculum_artifact_sha256",
            "curriculum_file_sha256",
            "dataset_id",
            "dataset_manifest_sha256",
            "dataset_revision",
            "episode_split_artifact_sha256",
            "episode_split_file_sha256",
            "episode_split_picf_code_revision",
            "expected_target_identity_keys",
            "expected_task_keys",
            "expected_x_task_camera_strata",
            "heldout_source_episode_indices",
            "long_training_authorized",
            "maximum_control_x_source_group_overlap_count",
            "picf_code_revision",
            "scene_audit_artifact_sha256",
            "scene_audit_file_sha256",
            "schema",
            "selection_basis",
            "steps",
            "summary",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("crossed bounded plan fields differ from schema")
        if (
            value["schema"] != CROSSED_BOUNDED_PLAN_SCHEMA
            or value["algorithm"] != CROSSED_BOUNDED_PLAN_ALGORITHM
            or value["selection_basis"] != "pre-model-data-only-matched-milp"
            or value["bounded_training_authorized"] is not True
            or value["long_training_authorized"] is not False
        ):
            raise ValueError("crossed bounded plan contract changed")
        artifact = _sha256(value["artifact_sha256"], name="plan artifact SHA-256")
        content = {key: child for key, child in value.items() if key != "artifact_sha256"}
        if _canonical_sha256(content) != artifact:
            raise ValueError("crossed bounded plan artifact SHA-256 changed")
        raw_tasks = value["expected_task_keys"]
        raw_targets = value["expected_target_identity_keys"]
        raw_strata = value["expected_x_task_camera_strata"]
        raw_heldout = value["heldout_source_episode_indices"]
        raw_steps = value["steps"]
        collections = (raw_tasks, raw_targets, raw_strata, raw_heldout, raw_steps)
        if not all(isinstance(item, list) for item in collections):
            raise ValueError("crossed bounded plan collections are malformed")
        strata: list[tuple[str, str]] = []
        for row in raw_strata:
            if not isinstance(row, Mapping) or set(row) != {"camera_name", "task_key"}:
                raise ValueError("crossed bounded X stratum is malformed")
            strata.append(
                (
                    _text(row["task_key"], name="X stratum task"),
                    _text(row["camera_name"], name="X stratum camera"),
                )
            )
        plan = cls(
            dataset_id=_text(value["dataset_id"], name="dataset ID"),
            dataset_revision=_text(value["dataset_revision"], name="dataset revision"),
            dataset_manifest_sha256=_sha256(
                value["dataset_manifest_sha256"], name="dataset manifest SHA-256"
            ),
            curriculum_file_sha256=_sha256(
                value["curriculum_file_sha256"], name="curriculum file SHA-256"
            ),
            curriculum_artifact_sha256=_sha256(
                value["curriculum_artifact_sha256"], name="curriculum artifact SHA-256"
            ),
            scene_audit_file_sha256=_sha256(
                value["scene_audit_file_sha256"], name="scene file SHA-256"
            ),
            scene_audit_artifact_sha256=_sha256(
                value["scene_audit_artifact_sha256"], name="scene artifact SHA-256"
            ),
            episode_split_file_sha256=_sha256(
                value["episode_split_file_sha256"], name="split file SHA-256"
            ),
            episode_split_artifact_sha256=_sha256(
                value["episode_split_artifact_sha256"], name="split artifact SHA-256"
            ),
            episode_split_picf_code_revision=_git_revision(
                value["episode_split_picf_code_revision"],
                name="split PICF revision",
            ),
            picf_code_revision=_git_revision(value["picf_code_revision"], name="PICF revision"),
            expected_task_keys=tuple(raw_tasks),  # type: ignore[arg-type]
            expected_target_identity_keys=tuple(raw_targets),  # type: ignore[arg-type]
            expected_x_task_camera_strata=tuple(strata),
            heldout_source_episode_indices=tuple(raw_heldout),  # type: ignore[arg-type]
            maximum_control_x_source_group_overlap_count=_positive_int(
                value["maximum_control_x_source_group_overlap_count"],
                name="maximum X source overlap",
            ),
            steps=tuple(CrossedBoundedStep.from_dict(step) for step in raw_steps),
        )
        if plan.summary != value["summary"]:
            raise ValueError("crossed bounded plan summary changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> CrossedBoundedPlan:
        source = Path(path).expanduser().absolute()
        if source.is_symlink() or not source.is_file():
            raise ValueError("crossed bounded plan must be one real file")
        if not 0 < source.stat().st_size <= CROSSED_BOUNDED_PLAN_MAXIMUM_BYTES:
            raise ValueError("crossed bounded plan size is outside the supported contract")
        try:
            value = json.loads(source.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("crossed bounded plan is not valid JSON") from error
        return cls.from_dict(value)


@dataclass(frozen=True, slots=True)
class _Edge:
    records: tuple[CrossedBoundedRecord, CrossedBoundedRecord]

    @property
    def camera_name(self) -> str:
        return self.records[0].camera_name

    @property
    def task_camera(self) -> tuple[str, str]:
        return self.records[0].task_camera

    @property
    def group_indices(self) -> frozenset[int]:
        return frozenset(record.group_index for record in self.records)

    @property
    def instruction_sha256s(self) -> frozenset[str]:
        return frozenset(record.instruction_sha256 for record in self.records)

    @property
    def key(self) -> tuple[object, ...]:
        return tuple(item for record in self.records for item in _record_order(record))


def _edge_tie_objective(edges: Sequence[_Edge]) -> NDArray[np.float64]:
    values = []
    for edge in edges:
        digest = hashlib.sha256(repr(edge.key).encode("ascii")).hexdigest()
        values.append(int(digest[:12], 16) / float(16**12))
    return np.asarray(values, dtype=np.float64)


def _solve_selection(
    edges: Sequence[_Edge],
    constraints: Sequence[tuple[Sequence[int] | NDArray[np.float64], float, float]],
    objective: NDArray[np.float64],
    *,
    name: str,
) -> tuple[_Edge, ...]:
    count = len(edges)
    if count == 0 or objective.shape != (count,):
        raise ValueError(f"crossed bounded {name} has no feasible edge inventory")
    matrix = lil_matrix((len(constraints), count), dtype=np.float64)
    lower: list[float] = []
    upper: list[float] = []
    for row_index, (indices_or_coefficients, minimum, maximum) in enumerate(constraints):
        if isinstance(indices_or_coefficients, np.ndarray):
            if indices_or_coefficients.shape != (count,):
                raise ValueError(f"crossed bounded {name} coefficient row is malformed")
            matrix[row_index, :] = indices_or_coefficients.astype(np.float64)
        else:
            matrix[row_index, list(indices_or_coefficients)] = 1.0
        lower.append(minimum)
        upper.append(maximum)
    result = milp(
        objective,
        integrality=np.ones(count, dtype=np.int8),
        bounds=Bounds(0.0, 1.0),
        # SciPy accepts vector bounds; its current inline type narrows both to scalars.
        constraints=LinearConstraint(
            matrix.tocsr(),
            lower,  # pyright: ignore[reportArgumentType]
            upper,  # pyright: ignore[reportArgumentType]
        ),
        options={"time_limit": 60.0},
    )
    if not result.success or result.x is None:
        raise ValueError(f"crossed bounded {name} MILP is infeasible: {result.message}")
    if any(not (value <= 1e-7 or value >= 1.0 - 1e-7) for value in result.x):
        raise ValueError(f"crossed bounded {name} MILP returned a fractional selection")
    return tuple(edge for edge, selected in zip(edges, result.x, strict=True) if selected > 0.5)


def _selection_constraints(
    edges: Sequence[_Edge],
    *,
    task_camera_counts: Mapping[tuple[str, str], int] | None,
    cover_task_cameras: Sequence[tuple[str, str]] = (),
    cover_tasks: Sequence[str] = (),
    cover_targets: Sequence[str] = (),
) -> list[tuple[Sequence[int] | NDArray[np.float64], float, float]]:
    constraints: list[tuple[Sequence[int] | NDArray[np.float64], float, float]] = [
        (tuple(range(len(edges))), CROSSED_BOUNDED_CELL_STEPS, CROSSED_BOUNDED_CELL_STEPS)
    ]
    for camera in CALVIN_GROUNDING_CAMERAS:
        indices = tuple(index for index, edge in enumerate(edges) if edge.camera_name == camera)
        constraints.append((indices, CROSSED_BOUNDED_CAMERA_STEPS, CROSSED_BOUNDED_CAMERA_STEPS))
    for group_index in sorted({group for edge in edges for group in edge.group_indices}):
        constraints.append(
            (
                tuple(
                    index for index, edge in enumerate(edges) if group_index in edge.group_indices
                ),
                0,
                1,
            )
        )
    for instruction in sorted(
        {instruction for edge in edges for instruction in edge.instruction_sha256s}
    ):
        constraints.append(
            (
                tuple(
                    index
                    for index, edge in enumerate(edges)
                    if instruction in edge.instruction_sha256s
                ),
                0,
                1,
            )
        )
    if task_camera_counts is not None:
        for stratum, count in sorted(task_camera_counts.items()):
            indices = tuple(
                index for index, edge in enumerate(edges) if edge.task_camera == stratum
            )
            constraints.append((indices, count, count))
    for stratum in cover_task_cameras:
        indices = tuple(index for index, edge in enumerate(edges) if edge.task_camera == stratum)
        constraints.append((indices, 1, np.inf))
    for task in cover_tasks:
        indices = tuple(
            index
            for index, edge in enumerate(edges)
            if task in {record.task_key for record in edge.records}
        )
        constraints.append((indices, 1, np.inf))
    for target in cover_targets:
        indices = tuple(
            index
            for index, edge in enumerate(edges)
            if target in {record.target_identity_key for record in edge.records}
        )
        constraints.append((indices, 1, np.inf))
    return constraints


def _record_from_evidence(
    row: CrossedVariantViewEvidence,
    groups: Sequence[FixedObservationGroup],
) -> CrossedBoundedRecord:
    if row.bbox_qwen_xyxy is None:
        raise ValueError("crossed bounded schedule cannot materialize an unsupervised row")
    group = groups[row.group_index]
    matches = tuple(
        index
        for index, variant in enumerate(group.variants)
        if variant.task_key == row.task_key
        and variant.instruction_sha256 == row.instruction_sha256
        and variant.target_identity_key == row.target_identity_key
    )
    if len(matches) != 1:
        raise ValueError("crossed bounded evidence does not resolve one curriculum variant")
    return CrossedBoundedRecord(
        group_index=row.group_index,
        variant_index=matches[0],
        global_index=row.global_index,
        source_episode_index=row.source_episode_index,
        source_state_sha256=row.source_state_sha256,
        camera_name=row.camera_name,
        source_rgb_sha256=row.source_rgb_sha256,
        task_key=row.task_key,
        instruction_sha256=row.instruction_sha256,
        target_identity_key=row.target_identity_key,
        bbox_qwen_xyxy=row.bbox_qwen_xyxy,
    )


def _build_edge_inventories(
    rows: Sequence[CrossedVariantViewEvidence],
    groups: Sequence[FixedObservationGroup],
) -> tuple[tuple[_Edge, ...], tuple[_Edge, ...], tuple[_Edge, ...]]:
    by_group_camera: dict[tuple[int, str], list[CrossedVariantViewEvidence]] = defaultdict(list)
    by_exact: dict[tuple[str, str, str, str], list[CrossedVariantViewEvidence]] = defaultdict(list)
    by_semantic: dict[tuple[str, str, str], list[CrossedVariantViewEvidence]] = defaultdict(list)
    for row in rows:
        by_group_camera[(row.group_index, row.camera_name)].append(row)
        exact_key = (
            row.task_key,
            row.target_identity_key,
            row.camera_name,
            row.instruction_sha256,
        )
        by_exact[exact_key].append(row)
        by_semantic[(row.task_key, row.target_identity_key, row.camera_name)].append(row)

    p_edges: list[_Edge] = []
    for stratum_rows in by_group_camera.values():
        for first, second in combinations(stratum_rows, 2):
            if (
                first.task_key == second.task_key
                or first.instruction_sha256 == second.instruction_sha256
                or first.target_identity_key == second.target_identity_key
                or first.bbox_qwen_xyxy is None
                or second.bbox_qwen_xyxy is None
                or not boxes_are_mutually_centre_exclusive(
                    first.bbox_qwen_xyxy, second.bbox_qwen_xyxy
                )
            ):
                continue
            records = tuple(
                sorted(
                    (_record_from_evidence(first, groups), _record_from_evidence(second, groups)),
                    key=_record_order,
                )
            )
            p_edges.append(_Edge(records=records))  # type: ignore[arg-type]

    def source_edges(
        strata: Mapping[tuple[str, ...], list[CrossedVariantViewEvidence]],
        *,
        require_different_instruction: bool,
    ) -> list[_Edge]:
        result = []
        for stratum_rows in strata.values():
            for first, second in combinations(stratum_rows, 2):
                if (
                    first.bbox_qwen_xyxy is None
                    or second.bbox_qwen_xyxy is None
                    or (
                        require_different_instruction
                        and first.instruction_sha256 == second.instruction_sha256
                    )
                    or not crossed_variant_views_are_source_disjoint(first, second)
                    or not boxes_are_mutually_centre_exclusive(
                        first.bbox_qwen_xyxy, second.bbox_qwen_xyxy
                    )
                ):
                    continue
                records = tuple(
                    sorted(
                        (
                            _record_from_evidence(first, groups),
                            _record_from_evidence(second, groups),
                        ),
                        key=_record_order,
                    )
                )
                result.append(_Edge(records=records))  # type: ignore[arg-type]
        return result

    exact_edges = source_edges(by_exact, require_different_instruction=False)
    control_edges = source_edges(by_semantic, require_different_instruction=True)
    return (
        tuple(sorted(p_edges, key=lambda edge: edge.key)),
        tuple(sorted(exact_edges, key=lambda edge: edge.key)),
        tuple(sorted(control_edges, key=lambda edge: edge.key)),
    )


def _match_x_edges(
    candidate: Sequence[_Edge],
    control: Sequence[_Edge],
) -> tuple[tuple[_Edge, _Edge], ...]:
    candidate_by_stratum: dict[tuple[str, str], list[_Edge]] = defaultdict(list)
    control_by_stratum: dict[tuple[str, str], list[_Edge]] = defaultdict(list)
    for edge in candidate:
        candidate_by_stratum[edge.task_camera].append(edge)
    for edge in control:
        control_by_stratum[edge.task_camera].append(edge)
    if {key: len(value) for key, value in candidate_by_stratum.items()} != {
        key: len(value) for key, value in control_by_stratum.items()
    }:
        raise ValueError("crossed bounded X selections cannot be matched by stratum")
    matched = []
    for stratum in sorted(candidate_by_stratum):
        left = sorted(candidate_by_stratum[stratum], key=lambda edge: edge.key)
        right = sorted(control_by_stratum[stratum], key=lambda edge: edge.key)
        size = len(left)
        cost = np.zeros((size, size), dtype=np.float64)
        for left_index, candidate_edge in enumerate(left):
            for right_index, control_edge in enumerate(right):
                overlap = len(candidate_edge.group_indices & control_edge.group_indices)
                cost[left_index, right_index] = -1000.0 * overlap + right_index
        row_indices, column_indices = linear_sum_assignment(cost)
        matched.extend(
            (left[row], right[column])
            for row, column in zip(row_indices, column_indices, strict=True)
        )
    return tuple(matched)


def _orient_matched_x(
    candidate: _Edge,
    control: _Edge,
) -> tuple[_RecordPair, _RecordPair]:
    candidate_records = _ordered_record_pair(candidate.records)
    ordered_control = _ordered_record_pair(control.records)
    control_options = (ordered_control, _reverse_record_pair(ordered_control))
    scored = []
    for records in control_options:
        overlap = sum(
            left.group_index == right.group_index
            for left, right in zip(candidate_records, records, strict=True)
        )
        scored.append((-overlap, tuple(_record_order(record) for record in records), records))
    return candidate_records, min(scored, key=lambda item: item[:2])[2]


def build_crossed_bounded_plan(
    curriculum: NativeVLGroundingCurriculumPlan,
    scene_audit: Mapping[str, object],
    episode_split: Mapping[str, object],
    *,
    curriculum_file_sha256: str,
    scene_audit_file_sha256: str,
    episode_split_file_sha256: str,
    picf_code_revision: str,
    expected_task_keys: Sequence[str],
    expected_target_identity_keys: Sequence[str],
) -> CrossedBoundedPlan:
    """Build the only data-authorized 64-update ADR-128 comparison."""

    if not isinstance(curriculum, NativeVLGroundingCurriculumPlan):
        raise TypeError("crossed bounded plan requires a typed curriculum")
    tasks = _sorted_unique_text(tuple(expected_task_keys), name="expected tasks")
    targets = _sorted_unique_text(tuple(expected_target_identity_keys), name="expected targets")
    curriculum_file = _sha256(curriculum_file_sha256, name="curriculum file SHA-256")
    scene_file = _sha256(scene_audit_file_sha256, name="scene audit file SHA-256")
    split_file = _sha256(episode_split_file_sha256, name="episode split file SHA-256")
    code_revision = _git_revision(picf_code_revision, name="PICF revision")
    split_artifact = _sha256(
        episode_split.get("artifact_sha256"), name="episode split artifact SHA-256"
    )
    split_content = {key: value for key, value in episode_split.items() if key != "artifact_sha256"}
    if (
        _canonical_sha256(split_content) != split_artifact
        or episode_split.get("schema") != CROSSED_EPISODE_SPLIT_SCHEMA
        or episode_split.get("status") != "PASS"
        or episode_split.get("training_authorized") is not False
        or episode_split.get("curriculum_artifact_sha256") != curriculum.artifact_sha256
        or episode_split.get("curriculum_file_sha256") != curriculum_file
        or episode_split.get("scene_audit_artifact_sha256") != scene_audit.get("artifact_sha256")
        or episode_split.get("scene_audit_file_sha256") != scene_file
    ):
        raise ValueError("crossed bounded plan requires its exact passing episode split")
    split_revision = _git_revision(
        episode_split.get("picf_code_revision"),
        name="episode split PICF revision",
    )
    heldout_value = episode_split.get("heldout_source_episode_indices")
    if not isinstance(heldout_value, list):
        raise ValueError("crossed bounded episode split has no heldout source list")
    heldout = tuple(heldout_value)
    if (
        not heldout
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in heldout
        )
        or tuple(sorted(set(heldout))) != heldout
    ):
        raise ValueError("crossed bounded heldout episodes are malformed")

    joined = materialize_crossed_variant_views(
        curriculum.groups,
        scene_audit,
        expected_curriculum_artifact_sha256=curriculum.artifact_sha256,
    )
    training = tuple(
        row
        for row in joined
        if row.source_episode_index not in set(heldout) and row.state == "supervised"
    )
    p_inventory, exact_inventory, control_inventory = _build_edge_inventories(
        training, curriculum.groups
    )
    supported_strata = tuple(sorted({edge.task_camera for edge in exact_inventory}))
    if {task for task, _camera in supported_strata} != set(tasks):
        raise ValueError("crossed bounded exact-X support misses an expected task")

    candidate_constraints = _selection_constraints(
        exact_inventory,
        task_camera_counts=None,
        cover_task_cameras=supported_strata,
    )
    candidate = _solve_selection(
        exact_inventory,
        candidate_constraints,
        _edge_tie_objective(exact_inventory),
        name="candidate X",
    )
    candidate_histogram = Counter(edge.task_camera for edge in candidate)
    candidate_groups = {group for edge in candidate for group in edge.group_indices}

    control_constraints = _selection_constraints(
        control_inventory,
        task_camera_counts=candidate_histogram,
    )
    overlap_coefficients = np.asarray(
        [len(edge.group_indices & candidate_groups) for edge in control_inventory],
        dtype=np.float64,
    )
    maximum_control = _solve_selection(
        control_inventory,
        control_constraints,
        -overlap_coefficients,
        name="maximum-overlap control X",
    )
    maximum_overlap = sum(len(edge.group_indices & candidate_groups) for edge in maximum_control)
    control_constraints.append((overlap_coefficients, maximum_overlap, maximum_overlap))
    control = _solve_selection(
        control_inventory,
        control_constraints,
        _edge_tie_objective(control_inventory),
        name="tie-broken control X",
    )

    x_groups = {group for edge in (*candidate, *control) for group in edge.group_indices}
    x_instructions = {
        instruction for edge in (*candidate, *control) for instruction in edge.instruction_sha256s
    }
    eligible_p = tuple(
        edge
        for edge in p_inventory
        if not edge.group_indices & x_groups and not edge.instruction_sha256s & x_instructions
    )
    p_constraints = _selection_constraints(
        eligible_p,
        task_camera_counts=None,
        cover_tasks=tasks,
        cover_targets=targets,
    )
    prompt = _solve_selection(
        eligible_p,
        p_constraints,
        _edge_tie_objective(eligible_p),
        name="shared P",
    )

    prompt_ordered = tuple(sorted(prompt, key=lambda edge: edge.key))
    x_matched = tuple(
        sorted(
            _match_x_edges(candidate, control),
            key=lambda pair: (pair[0].task_camera, pair[0].key, pair[1].key),
        )
    )
    steps = []
    for pair_index in range(CROSSED_BOUNDED_CELL_STEPS):
        p_records = prompt_ordered[pair_index].records
        if pair_index % 2:
            p_records = _reverse_record_pair(p_records)
        steps.append(
            CrossedBoundedStep(
                optimizer_step=len(steps),
                cell="P",
                candidate_records=p_records,
                control_records=p_records,
            )
        )
        candidate_records, control_records = _orient_matched_x(*x_matched[pair_index])
        if pair_index % 2:
            candidate_records = _reverse_record_pair(candidate_records)
            control_records = _reverse_record_pair(control_records)
        steps.append(
            CrossedBoundedStep(
                optimizer_step=len(steps),
                cell="X",
                candidate_records=candidate_records,
                control_records=control_records,
            )
        )

    return CrossedBoundedPlan(
        dataset_id=curriculum.dataset_id,
        dataset_revision=curriculum.dataset_revision,
        dataset_manifest_sha256=curriculum.dataset_manifest_sha256,
        curriculum_file_sha256=curriculum_file,
        curriculum_artifact_sha256=curriculum.artifact_sha256,
        scene_audit_file_sha256=scene_file,
        scene_audit_artifact_sha256=_sha256(
            scene_audit.get("artifact_sha256"), name="scene audit artifact SHA-256"
        ),
        episode_split_file_sha256=split_file,
        episode_split_artifact_sha256=split_artifact,
        episode_split_picf_code_revision=split_revision,
        picf_code_revision=code_revision,
        expected_task_keys=tasks,
        expected_target_identity_keys=targets,
        expected_x_task_camera_strata=supported_strata,
        heldout_source_episode_indices=heldout,
        maximum_control_x_source_group_overlap_count=maximum_overlap,
        steps=tuple(steps),
    )

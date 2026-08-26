"""Content-addressed held-out exact-prompt pixel-causality evaluation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.crossed_bounded_plan import CrossedBoundedRecord
from picf_next.lingbot_native.crossed_causal_grounding import (
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

CROSSED_EVALUATION_SCHEMA = "picf-next.crossed-grounding-heldout-evaluation-plan.v1"
CROSSED_EVALUATION_MAXIMUM_BYTES = 4 * 1024 * 1024


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
        raise ValueError("crossed evaluation is not canonical finite JSON") from error


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
        raise ValueError(f"{name} must be one nonempty string")
    return value


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be one positive integer")
    return value


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be one nonnegative integer")
    return value


def _record_order(record: CrossedBoundedRecord) -> tuple[object, ...]:
    return (
        record.task_key,
        record.target_identity_key,
        record.camera_name,
        record.instruction_sha256,
        record.group_index,
        record.variant_index,
    )


@dataclass(frozen=True, slots=True)
class CrossedEvaluationPair:
    """One complete-prompt held-out intervention over two physical sources."""

    first: CrossedBoundedRecord
    second: CrossedBoundedRecord

    def __post_init__(self) -> None:
        if not isinstance(self.first, CrossedBoundedRecord) or not isinstance(
            self.second, CrossedBoundedRecord
        ):
            raise ValueError("crossed evaluation pair requires two typed records")
        if _record_order(self.first) >= _record_order(self.second):
            raise ValueError("crossed evaluation pair order is not canonical")
        if (
            self.first.task_key != self.second.task_key
            or self.first.target_identity_key != self.second.target_identity_key
            or self.first.camera_name != self.second.camera_name
            or self.first.instruction_sha256 != self.second.instruction_sha256
            or self.first.group_index == self.second.group_index
            or self.first.global_index == self.second.global_index
            or self.first.source_episode_index == self.second.source_episode_index
            or self.first.source_state_sha256 == self.second.source_state_sha256
            or self.first.source_rgb_sha256 == self.second.source_rgb_sha256
            or not boxes_are_mutually_centre_exclusive(
                self.first.bbox_qwen_xyxy,
                self.second.bbox_qwen_xyxy,
            )
        ):
            raise ValueError("crossed evaluation pair differs from strict exact-X")

    @property
    def key(self) -> str:
        return _canonical_sha256(self.as_dict())

    def as_dict(self) -> dict[str, object]:
        return {"first": self.first.as_dict(), "second": self.second.as_dict()}

    @classmethod
    def from_dict(cls, value: object) -> CrossedEvaluationPair:
        if not isinstance(value, Mapping) or set(value) != {"first", "second"}:
            raise ValueError("crossed evaluation pair fields differ from schema")
        return cls(
            first=CrossedBoundedRecord.from_dict(value["first"]),
            second=CrossedBoundedRecord.from_dict(value["second"]),
        )


@dataclass(frozen=True, slots=True)
class CrossedEvaluationPlan:
    """All strict exact-X pairs in the frozen one-shot held-out episodes."""

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
    heldout_source_episode_indices: tuple[int, ...]
    expected_pair_count: int
    pairs: tuple[CrossedEvaluationPair, ...]

    def __post_init__(self) -> None:
        _text(self.dataset_id, name="crossed evaluation dataset ID")
        _text(self.dataset_revision, name="crossed evaluation dataset revision")
        for name in (
            "dataset_manifest_sha256",
            "curriculum_file_sha256",
            "curriculum_artifact_sha256",
            "scene_audit_file_sha256",
            "scene_audit_artifact_sha256",
            "episode_split_file_sha256",
            "episode_split_artifact_sha256",
        ):
            _sha256(getattr(self, name), name=f"crossed evaluation {name}")
        _git_revision(
            self.episode_split_picf_code_revision,
            name="crossed evaluation split revision",
        )
        _git_revision(self.picf_code_revision, name="crossed evaluation PICF revision")
        tasks = tuple(sorted(self.expected_task_keys))
        targets = tuple(sorted(self.expected_target_identity_keys))
        heldout = tuple(sorted(self.heldout_source_episode_indices))
        if (
            not tasks
            or tasks != self.expected_task_keys
            or len(set(tasks)) != len(tasks)
            or not targets
            or targets != self.expected_target_identity_keys
            or len(set(targets)) != len(targets)
            or not heldout
            or heldout != self.heldout_source_episode_indices
            or any(_nonnegative_int(value, name="heldout episode") != value for value in heldout)
        ):
            raise ValueError("crossed evaluation coverage contracts are malformed")
        expected = _positive_int(self.expected_pair_count, name="crossed evaluation pair count")
        if (
            not isinstance(self.pairs, tuple)
            or len(self.pairs) != expected
            or any(not isinstance(pair, CrossedEvaluationPair) for pair in self.pairs)
            or tuple(sorted(self.pairs, key=lambda pair: pair.key)) != self.pairs
            or len({pair.key for pair in self.pairs}) != len(self.pairs)
        ):
            raise ValueError("crossed evaluation pairs changed")
        records = {record for pair in self.pairs for record in (pair.first, pair.second)}
        if (
            {record.task_key for record in records} != set(tasks)
            or {record.target_identity_key for record in records} != set(targets)
            or {record.source_episode_index for record in records} != set(heldout)
        ):
            raise ValueError("crossed evaluation record coverage changed")

    @property
    def unique_records(self) -> tuple[CrossedBoundedRecord, ...]:
        return tuple(
            sorted(
                {record for pair in self.pairs for record in (pair.first, pair.second)},
                key=_record_order,
            )
        )

    @property
    def summary(self) -> dict[str, object]:
        records = self.unique_records
        return {
            "camera_record_histogram": dict(
                sorted(Counter(record.camera_name for record in records).items())
            ),
            "covered_instruction_count": len({record.instruction_sha256 for record in records}),
            "covered_target_cameras": [
                {"camera_name": camera, "target_identity_key": target}
                for target, camera in sorted(
                    {(record.target_identity_key, record.camera_name) for record in records}
                )
            ],
            "covered_target_identity_keys": sorted(
                {record.target_identity_key for record in records}
            ),
            "covered_task_keys": sorted({record.task_key for record in records}),
            "pair_count": len(self.pairs),
            "unique_record_count": len(records),
        }

    def as_dict(self) -> dict[str, object]:
        content: dict[str, object] = {
            "checkpoint_selection_authorized": False,
            "curriculum_artifact_sha256": self.curriculum_artifact_sha256,
            "curriculum_file_sha256": self.curriculum_file_sha256,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "episode_split_artifact_sha256": self.episode_split_artifact_sha256,
            "episode_split_file_sha256": self.episode_split_file_sha256,
            "episode_split_picf_code_revision": self.episode_split_picf_code_revision,
            "expected_pair_count": self.expected_pair_count,
            "expected_target_identity_keys": list(self.expected_target_identity_keys),
            "expected_task_keys": list(self.expected_task_keys),
            "heldout_source_episode_indices": list(self.heldout_source_episode_indices),
            "pairs": [pair.as_dict() for pair in self.pairs],
            "picf_code_revision": self.picf_code_revision,
            "scene_audit_artifact_sha256": self.scene_audit_artifact_sha256,
            "scene_audit_file_sha256": self.scene_audit_file_sha256,
            "schema": CROSSED_EVALUATION_SCHEMA,
            "selection_basis": "all-strict-exact-x-pairs-in-frozen-heldout-episodes",
            "summary": self.summary,
            "training_authorized": False,
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
            raise ValueError("crossed evaluation group is outside the curriculum")
        group = groups[record.group_index]
        if record.variant_index >= len(group.variants):
            raise ValueError("crossed evaluation variant is outside its group")
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
            raise ValueError("crossed evaluation record differs from its curriculum")
        return group, variant

    @classmethod
    def from_dict(cls, value: object) -> CrossedEvaluationPlan:
        expected = {
            "artifact_sha256",
            "checkpoint_selection_authorized",
            "curriculum_artifact_sha256",
            "curriculum_file_sha256",
            "dataset_id",
            "dataset_manifest_sha256",
            "dataset_revision",
            "episode_split_artifact_sha256",
            "episode_split_file_sha256",
            "episode_split_picf_code_revision",
            "expected_pair_count",
            "expected_target_identity_keys",
            "expected_task_keys",
            "heldout_source_episode_indices",
            "pairs",
            "picf_code_revision",
            "scene_audit_artifact_sha256",
            "scene_audit_file_sha256",
            "schema",
            "selection_basis",
            "summary",
            "training_authorized",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise ValueError("crossed evaluation plan fields differ from schema")
        if (
            value["schema"] != CROSSED_EVALUATION_SCHEMA
            or value["selection_basis"] != "all-strict-exact-x-pairs-in-frozen-heldout-episodes"
            or value["training_authorized"] is not False
            or value["checkpoint_selection_authorized"] is not False
        ):
            raise ValueError("crossed evaluation plan contract changed")
        artifact = _sha256(value["artifact_sha256"], name="evaluation artifact SHA-256")
        content = {key: child for key, child in value.items() if key != "artifact_sha256"}
        if _canonical_sha256(content) != artifact:
            raise ValueError("crossed evaluation artifact SHA-256 changed")
        tasks = value["expected_task_keys"]
        targets = value["expected_target_identity_keys"]
        heldout = value["heldout_source_episode_indices"]
        pairs = value["pairs"]
        if not all(isinstance(item, list) for item in (tasks, targets, heldout, pairs)):
            raise ValueError("crossed evaluation collections are malformed")
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
                value["episode_split_picf_code_revision"], name="split revision"
            ),
            picf_code_revision=_git_revision(value["picf_code_revision"], name="PICF revision"),
            expected_task_keys=tuple(tasks),  # type: ignore[arg-type]
            expected_target_identity_keys=tuple(targets),  # type: ignore[arg-type]
            heldout_source_episode_indices=tuple(heldout),  # type: ignore[arg-type]
            expected_pair_count=_positive_int(
                value["expected_pair_count"], name="expected pair count"
            ),
            pairs=tuple(CrossedEvaluationPair.from_dict(pair) for pair in pairs),
        )
        if plan.summary != value["summary"]:
            raise ValueError("crossed evaluation summary changed")
        return plan

    @classmethod
    def load(cls, path: str | Path) -> CrossedEvaluationPlan:
        source = Path(path).expanduser().absolute()
        if source.is_symlink() or not source.is_file():
            raise ValueError("crossed evaluation plan must be one real file")
        if not 0 < source.stat().st_size <= CROSSED_EVALUATION_MAXIMUM_BYTES:
            raise ValueError("crossed evaluation plan size is outside the contract")
        try:
            value = json.loads(source.read_bytes())
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("crossed evaluation plan is not valid JSON") from error
        return cls.from_dict(value)


def _row_order(row: CrossedVariantViewEvidence) -> tuple[object, ...]:
    return (
        row.task_key,
        row.target_identity_key,
        row.camera_name,
        row.instruction_sha256,
        row.group_index,
    )


def _record_from_row(
    curriculum: NativeVLGroundingCurriculumPlan,
    row: CrossedVariantViewEvidence,
) -> CrossedBoundedRecord:
    group = curriculum.groups[row.group_index]
    matches = [
        index
        for index, variant in enumerate(group.variants)
        if (
            variant.task_key,
            variant.instruction_sha256,
            variant.target_identity_key,
        )
        == (row.task_key, row.instruction_sha256, row.target_identity_key)
    ]
    if len(matches) != 1 or row.state != "supervised" or row.bbox_qwen_xyxy is None:
        raise ValueError("crossed evaluation row does not resolve one supervised variant")
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


def build_crossed_evaluation_plan(
    curriculum: NativeVLGroundingCurriculumPlan,
    scene_audit: Mapping[str, object],
    episode_split: Mapping[str, object],
    *,
    curriculum_file_sha256: str,
    scene_audit_file_sha256: str,
    episode_split_file_sha256: str,
    picf_code_revision: str,
) -> CrossedEvaluationPlan:
    """Freeze every legal held-out exact-X pair without consulting model output."""

    curriculum_file = _sha256(curriculum_file_sha256, name="curriculum file SHA-256")
    scene_file = _sha256(scene_audit_file_sha256, name="scene file SHA-256")
    split_file = _sha256(episode_split_file_sha256, name="split file SHA-256")
    split_artifact = _sha256(episode_split.get("artifact_sha256"), name="split artifact")
    split_content = {key: value for key, value in episode_split.items() if key != "artifact_sha256"}
    if _canonical_sha256(split_content) != split_artifact:
        raise ValueError("crossed evaluation split artifact changed")
    if (
        episode_split.get("schema") != CROSSED_EPISODE_SPLIT_SCHEMA
        or episode_split.get("status") != "PASS"
        or episode_split.get("training_authorized") is not False
        or episode_split.get("curriculum_file_sha256") != curriculum_file
        or episode_split.get("curriculum_artifact_sha256") != curriculum.artifact_sha256
        or episode_split.get("scene_audit_file_sha256") != scene_file
        or episode_split.get("scene_audit_artifact_sha256") != scene_audit.get("artifact_sha256")
    ):
        raise ValueError("crossed evaluation inputs differ from the passing split")
    heldout_value = episode_split.get("heldout_source_episode_indices")
    partitions = episode_split.get("partitions")
    if not isinstance(heldout_value, list) or not isinstance(partitions, Mapping):
        raise ValueError("crossed evaluation split coverage is malformed")
    heldout = tuple(
        sorted(_nonnegative_int(value, name="heldout episode") for value in heldout_value)
    )
    heldout_set = set(heldout)
    heldout_summary = partitions.get("heldout")
    if not isinstance(heldout_summary, Mapping):
        raise ValueError("crossed evaluation heldout summary is malformed")
    exact_summary = heldout_summary.get("pixel_causal_cells_exact_instruction")
    if not isinstance(exact_summary, Mapping):
        raise ValueError("crossed evaluation exact-X summary is malformed")
    expected_pair_count = _positive_int(exact_summary.get("pair_count"), name="exact-X pairs")

    rows = materialize_crossed_variant_views(
        curriculum.groups,
        scene_audit,
        expected_curriculum_artifact_sha256=curriculum.artifact_sha256,
    )
    strata: dict[tuple[str, str, str, str], list[CrossedVariantViewEvidence]] = defaultdict(list)
    for row in rows:
        if row.source_episode_index in heldout_set and row.state == "supervised":
            strata[
                (row.task_key, row.target_identity_key, row.camera_name, row.instruction_sha256)
            ].append(row)
    pairs = []
    for values in strata.values():
        for first, second in combinations(sorted(values, key=_row_order), 2):
            if not crossed_variant_views_are_source_disjoint(first, second):
                continue
            if first.bbox_qwen_xyxy is None or second.bbox_qwen_xyxy is None:
                raise RuntimeError("crossed evaluation lost a supervised box")
            if not boxes_are_mutually_centre_exclusive(
                first.bbox_qwen_xyxy,
                second.bbox_qwen_xyxy,
            ):
                continue
            records = (_record_from_row(curriculum, first), _record_from_row(curriculum, second))
            if _record_order(records[0]) > _record_order(records[1]):
                records = (records[1], records[0])
            pairs.append(CrossedEvaluationPair(first=records[0], second=records[1]))
    pairs = sorted(pairs, key=lambda pair: pair.key)
    plan = CrossedEvaluationPlan(
        dataset_id=curriculum.dataset_id,
        dataset_revision=curriculum.dataset_revision,
        dataset_manifest_sha256=curriculum.dataset_manifest_sha256,
        curriculum_file_sha256=curriculum_file,
        curriculum_artifact_sha256=curriculum.artifact_sha256,
        scene_audit_file_sha256=scene_file,
        scene_audit_artifact_sha256=_sha256(
            scene_audit.get("artifact_sha256"), name="scene artifact"
        ),
        episode_split_file_sha256=split_file,
        episode_split_artifact_sha256=split_artifact,
        episode_split_picf_code_revision=_git_revision(
            episode_split.get("picf_code_revision"), name="split revision"
        ),
        picf_code_revision=_git_revision(picf_code_revision, name="PICF revision"),
        expected_task_keys=tuple(exact_summary.get("covered_task_keys", ())),
        expected_target_identity_keys=tuple(exact_summary.get("covered_target_identity_keys", ())),
        heldout_source_episode_indices=heldout,
        expected_pair_count=expected_pair_count,
        pairs=tuple(pairs),
    )
    expected_summary = {
        "covered_instruction_count": exact_summary.get("covered_instruction_count"),
        "covered_target_cameras": exact_summary.get("covered_target_cameras"),
        "covered_target_identity_keys": exact_summary.get("covered_target_identity_keys"),
        "covered_task_keys": exact_summary.get("covered_task_keys"),
        "pair_count": expected_pair_count,
    }
    if any(plan.summary[key] != value for key, value in expected_summary.items()):
        raise ValueError("crossed evaluation recomputation differs from the split summary")
    return plan

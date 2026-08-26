#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Measure native-Qwen same-image prompt causality on source-disjoint CALVIN."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    _text = str(_path)
    while _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenGroundingRecord,
    CalvinQwenSceneGroundingRecord,
    build_calvin_qwen_grounding_records,
    build_calvin_qwen_scene_grounding_record,
    qwen_grounding_label,
)
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_FAMILIES,
    PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY,
    PublicNativeVLRetentionManifest,
    load_frozen_public_native_vl_retention_gate,
)
from picf_next.lingbot_native.fixed_observation import (
    validate_fixed_observation_group_source_index,
)
from picf_next.lingbot_native.fixed_observation_evaluation import (
    FIXED_OBSERVATION_EVALUATION_PARTITIONS,
    FixedObservationEvaluationItem,
    FixedObservationEvaluationPlan,
)
from picf_next.lingbot_native.crossed_bounded_plan import CrossedBoundedRecord
from picf_next.lingbot_native.crossed_evaluation import (
    CrossedEvaluationPair,
    CrossedEvaluationPlan,
)
from picf_next.lingbot_native.lattice_feasibility import (
    LATTICE_BASELINE,
    configure_native_processor_area_budget,
    configure_native_processor_lattice,
    validate_native_processor_record_grid,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    normalize_native_vl_answer as _normalize_generated_answer,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_fixed_x_pair_geometry_metrics as _pair_geometry_metrics,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_fixed_x_partition_summary as _partition_summary,
)
from picf_next.lingbot_native.native_vl_scene_metrics import (
    native_vl_scene_bank_summary,
    native_vl_scene_order_pair_metrics,
    normalize_scene_label,
)
from picf_next.lingbot_native.runtime_provenance import (
    adr127_runtime_python_trees_contract,
)
from picf_next.lingbot_native.vl_curriculum import NativeVLGroundingCurriculumPlan
from picf_next.lingbot_native.vl_cotraining import (
    NATIVE_VL_GENERATION_MAX_NEW_TOKENS,
    NativeVLGeneratedGrounding,
    NativeVLGeneratedSceneGrounding,
    parse_native_vl_grounding_answer,
    parse_native_vl_scene_grounding_answer,
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
)
from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MODEL_SOURCE,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native_vl import (
    NATIVE_VL_PATCH_RELATIVE_PATH,
    NATIVE_VL_PATCHED_MODEL_SHA256,
    _validate_native_vl_model,
    detect_native_vl_patch_state,
    verify_native_vl_patch,
)
from tools.lingbot_vla2_runtime_helpers import (
    _merge_qwen_config,
    _resolve_training_config,
    load_lingbot_training_config,
    strip_targetless_alignment_teacher_heads,
)
from tools.probe_lingbot_native_vl_grounding import (
    _validate_optional_qwen_restore,
    _validate_qwen_restore_load_result,
)
from tools.probe_qwen3vl_grounding_baseline import _model_hashes

OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-fixed-x-g0.v8"
CROSSED_OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-fixed-x-g0.v9"
SCENE_AUDIT_SCHEMA = "picf-next.native-vl-scene-curriculum-audit.v2"
SCENE_MAX_NEW_TOKENS_LIMIT = NATIVE_VL_GENERATION_MAX_NEW_TOKENS


@dataclass(frozen=True, slots=True)
class _FixedXRecordPair:
    item: FixedObservationEvaluationItem
    records: tuple[CalvinQwenGroundingRecord, CalvinQwenGroundingRecord]


@dataclass(frozen=True, slots=True)
class _SceneRecordPair:
    bank_index: int
    group_index: int
    task_keys: tuple[str, ...]
    records: tuple[CalvinQwenSceneGroundingRecord, CalvinQwenSceneGroundingRecord]


@dataclass(frozen=True, slots=True)
class _CrossedXRuntimeRecord:
    evidence: CrossedBoundedRecord
    record: CalvinQwenGroundingRecord


def _scene_generation_budget_contract(
    supervised_token_counts: Sequence[int],
    *,
    max_new_tokens: int,
) -> dict[str, int]:
    """Bind free generation to the longest legal scene answer."""

    if (
        isinstance(max_new_tokens, bool)
        or not isinstance(max_new_tokens, int)
        or not 1 <= max_new_tokens <= SCENE_MAX_NEW_TOKENS_LIMIT
    ):
        raise ContractError("fixed-X scene generation budget is invalid")
    counts = tuple(supervised_token_counts)
    if not counts or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in counts
    ):
        raise ContractError("fixed-X scene target token counts are invalid")
    required = max(counts)
    if max_new_tokens < required:
        raise ContractError(
            "fixed-X scene generation budget cannot emit the longest legal target: "
            f"configured={max_new_tokens}, required={required}"
        )
    return {
        "configured_max_new_tokens": max_new_tokens,
        "headroom_tokens": max_new_tokens - required,
        "maximum_target_supervised_tokens": required,
        "minimum_target_supervised_tokens": min(counts),
        "target_record_count": len(counts),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_git_revision(value: str, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"fixed-X native VL {name} must be one lowercase Git commit")
    return value


def _validate_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"fixed-X {name} must be one lowercase SHA-256")
    return value


def _scene_audit_canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ContractError("fixed-X scene audit is not canonical JSON") from error


def _load_scene_audit_report(path: Path, *, expected_file_sha256: str) -> dict[str, Any]:
    expected_digest = _validate_sha256(
        expected_file_sha256,
        name="scene audit file digest",
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ContractError("fixed-X scene audit file binding changed") from error
    payload = bytearray()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ContractError("fixed-X scene audit file binding changed")
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
        if len(payload) != metadata.st_size:
            raise ContractError("fixed-X scene audit file changed while reading")
    finally:
        os.close(descriptor)
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ContractError("fixed-X scene audit file binding changed")

    def strict_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output = {}
        for key, value in pairs:
            if key in output:
                raise ContractError("fixed-X scene audit contains a duplicate JSON key")
            output[key] = value
        return output

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON constant: {value}")
            ),
        )
    except ContractError:
        raise
    except (UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise ContractError("fixed-X scene audit is invalid JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != SCENE_AUDIT_SCHEMA
        or value.get("status") != "PASS"
    ):
        raise ContractError("fixed-X scene audit did not pass")
    artifact_sha256 = value.get("artifact_sha256")
    artifact_sha256 = _validate_sha256(
        artifact_sha256,
        name="scene audit artifact digest",
    )
    content = {key: item for key, item in value.items() if key != "artifact_sha256"}
    if hashlib.sha256(_scene_audit_canonical_bytes(content)).hexdigest() != artifact_sha256:
        raise ContractError("fixed-X scene audit artifact digest changed")
    bank = value.get("source_disjoint_scene_bank")
    arm = value.get("arm_steps")
    if not isinstance(bank, list) or len(bank) != 32 or not isinstance(arm, list) or len(arm) != 64:
        raise ContractError("fixed-X scene audit bank dimensions changed")
    return cast(dict[str, Any], value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--public-vl-retention-manifest", type=Path)
    parser.add_argument("--public-vl-retention-manifest-sha256")
    parser.add_argument("--public-vl-retention-root", type=Path)
    parser.add_argument("--public-vl-heldout-limit-per-family", type=int, default=0)
    parser.add_argument("--scene-audit-report", type=Path)
    parser.add_argument("--scene-audit-report-sha256")
    parser.add_argument("--scene-max-new-tokens", type=int, default=512)
    parser.add_argument("--crossed-evaluation-plan", type=Path)
    parser.add_argument("--crossed-evaluation-plan-sha256")
    parser.add_argument("--crossed-curriculum-plan", type=Path)
    parser.add_argument("--crossed-curriculum-plan-sha256")
    parser.add_argument("--crossed-scene-audit-report", type=Path)
    parser.add_argument("--crossed-scene-audit-report-sha256")
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument(
        "--partition",
        choices=("all", *FIXED_OBSERVATION_EVALUATION_PARTITIONS),
        default="all",
    )
    parser.add_argument("--item-limit-per-partition", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--visual-lattice", type=int, default=LATTICE_BASELINE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--restore-qwen-dir", type=Path)
    parser.add_argument("--restore-qwen-revision")
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> Path:
    for path in (
        args.training_config,
        args.dataset_manifest,
        args.evaluation_plan,
        args.source_checkout / MODEL_SOURCE,
        _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.physical_sidecar_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    partial = args.output_dir.with_name(f"{args.output_dir.name}.partial")
    for path in (args.output_dir, partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    integers = (
        args.item_limit_per_partition,
        args.max_new_tokens,
        args.public_vl_heldout_limit_per_family,
        args.scene_max_new_tokens,
        args.visual_lattice,
        args.seed,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise ContractError("fixed-X native VL integer arguments are invalid")
    if (
        args.item_limit_per_partition < 0
        or args.public_vl_heldout_limit_per_family < 0
        or args.public_vl_heldout_limit_per_family > PUBLIC_NATIVE_VL_HELDOUT_RECORDS_PER_FAMILY
        or not 1 <= args.max_new_tokens <= 256
        or not 1 <= args.scene_max_new_tokens <= SCENE_MAX_NEW_TOKENS_LIMIT
        or args.visual_lattice <= 0
        or args.seed < 0
        or not args.device.startswith("cuda:")
    ):
        raise ContractError("fixed-X native VL runtime arguments are invalid")
    _validate_git_revision(args.picf_code_revision, name="PICF code revision")
    retention_values = (
        args.public_vl_retention_manifest,
        args.public_vl_retention_manifest_sha256,
        args.public_vl_retention_root,
    )
    if any(value is not None for value in retention_values):
        if any(value is None for value in retention_values):
            raise ContractError("fixed-X public retention arguments must be all present")
        if not isinstance(args.public_vl_retention_manifest, Path):
            raise ContractError("fixed-X public retention manifest path is missing")
        if not isinstance(args.public_vl_retention_manifest_sha256, str):
            raise ContractError("fixed-X public retention manifest SHA-256 is missing")
        if not isinstance(args.public_vl_retention_root, Path):
            raise ContractError("fixed-X public retention root is missing")
        args.public_vl_retention_manifest_object = load_frozen_public_native_vl_retention_gate(
            manifest_path=args.public_vl_retention_manifest,
            manifest_file_sha256=args.public_vl_retention_manifest_sha256,
            artifact_root=args.public_vl_retention_root,
            max_steps=1,
        )
    else:
        if args.public_vl_heldout_limit_per_family != 0:
            raise ContractError("fixed-X public retention limit requires a manifest")
        args.public_vl_retention_manifest_object = None
    scene_values = (args.scene_audit_report, args.scene_audit_report_sha256)
    if any(value is not None for value in scene_values):
        if any(value is None for value in scene_values):
            raise ContractError("fixed-X scene audit arguments must be all present")
        if not isinstance(args.scene_audit_report, Path) or not isinstance(
            args.scene_audit_report_sha256, str
        ):
            raise ContractError("fixed-X scene audit arguments are invalid")
        args.scene_audit_report_object = _load_scene_audit_report(
            args.scene_audit_report,
            expected_file_sha256=args.scene_audit_report_sha256,
        )
        if args.visual_lattice != LATTICE_BASELINE:
            raise ContractError("fixed-X scene evaluation requires the baseline lattice")
    else:
        args.scene_audit_report_object = None
    crossed_values = (
        args.crossed_evaluation_plan,
        args.crossed_evaluation_plan_sha256,
        args.crossed_curriculum_plan,
        args.crossed_curriculum_plan_sha256,
        args.crossed_scene_audit_report,
        args.crossed_scene_audit_report_sha256,
    )
    if any(value is not None for value in crossed_values):
        if any(value is None for value in crossed_values):
            raise ContractError("fixed-X crossed evaluation arguments must be all present")
        for path in (
            args.crossed_evaluation_plan,
            args.crossed_curriculum_plan,
            args.crossed_scene_audit_report,
        ):
            if not isinstance(path, Path) or not path.is_file():
                raise FileNotFoundError(path)
        evaluation_file_sha256 = _validate_sha256(
            args.crossed_evaluation_plan_sha256,
            name="crossed evaluation-plan file digest",
        )
        curriculum_file_sha256 = _validate_sha256(
            args.crossed_curriculum_plan_sha256,
            name="crossed curriculum file digest",
        )
        crossed_scene_file_sha256 = _validate_sha256(
            args.crossed_scene_audit_report_sha256,
            name="crossed scene-audit file digest",
        )
        if _sha256(args.crossed_evaluation_plan) != evaluation_file_sha256:
            raise ContractError("fixed-X crossed evaluation-plan file binding changed")
        if _sha256(args.crossed_curriculum_plan) != curriculum_file_sha256:
            raise ContractError("fixed-X crossed curriculum file binding changed")
        crossed_scene_audit = _load_scene_audit_report(
            args.crossed_scene_audit_report,
            expected_file_sha256=crossed_scene_file_sha256,
        )
        if (
            args.scene_audit_report_object is not None
            and args.scene_audit_report_object != crossed_scene_audit
        ):
            raise ContractError("fixed-X scene evaluation and crossed evidence differ")
        crossed_plan = CrossedEvaluationPlan.load(args.crossed_evaluation_plan)
        crossed_curriculum = NativeVLGroundingCurriculumPlan.load(args.crossed_curriculum_plan)
        if (
            crossed_plan.picf_code_revision != args.picf_code_revision
            or crossed_plan.curriculum_file_sha256 != curriculum_file_sha256
            or crossed_plan.curriculum_artifact_sha256 != crossed_curriculum.artifact_sha256
            or crossed_plan.scene_audit_file_sha256 != crossed_scene_file_sha256
            or crossed_plan.scene_audit_artifact_sha256 != crossed_scene_audit["artifact_sha256"]
        ):
            raise ContractError("fixed-X crossed evaluation provenance changed")
        args.crossed_evaluation_plan_object = crossed_plan
        args.crossed_curriculum_plan_object = crossed_curriculum
        args.crossed_scene_audit_report_object = crossed_scene_audit
    else:
        args.crossed_evaluation_plan_object = None
        args.crossed_curriculum_plan_object = None
        args.crossed_scene_audit_report_object = None
    _validate_optional_qwen_restore(args.restore_qwen_dir, args.restore_qwen_revision)
    return partial


def _source_images(arrays: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation.images.image": arrays["rgb_static"],
        "observation.images.wrist_image": arrays["rgb_gripper"],
    }


def _camera_records(
    records: tuple[CalvinQwenGroundingRecord, ...],
) -> dict[str, CalvinQwenGroundingRecord]:
    result = {record.camera_name: record for record in records}
    if len(result) != len(records):
        raise ContractError("fixed-X grounding records repeat one camera")
    return result


def _materialize_pair(
    *,
    index: Any,
    sidecar: Any,
    item: FixedObservationEvaluationItem,
) -> _FixedXRecordPair | None:
    validate_fixed_observation_group_source_index(index, item.group)
    global_index = item.group.source_global_index
    arrays = dict(
        index.validated_source_frame_arrays(
            global_index,
            fields=("rgb_gripper", "rgb_static"),
        )
    )
    physical = sidecar.source_frame(global_index)
    by_variant = []
    for variant in item.variants:
        records = build_calvin_qwen_grounding_records(
            global_index=global_index,
            task_key=variant.task_key,
            instruction=variant.instruction,
            observation_images=_source_images(arrays),
            physical_frame=physical,
        )
        if any(record.target_identity_key != variant.target_identity_key for record in records):
            raise ContractError("fixed-X native target differs from its audited variant")
        by_variant.append(_camera_records(records))
    common = set(by_variant[0]).intersection(by_variant[1])
    if not common:
        return None
    camera_name = "static" if "static" in common else sorted(common)[0]
    pair = (by_variant[0][camera_name], by_variant[1][camera_name])
    if pair[0].source_rgb_sha256 != pair[1].source_rgb_sha256 or not np.array_equal(
        pair[0].image,
        pair[1].image,
    ):
        raise ContractError("fixed-X variants do not share byte-identical source pixels")
    return _FixedXRecordPair(item=item, records=pair)


def _validate_crossed_runtime_record(
    evidence: CrossedBoundedRecord,
    record: CalvinQwenGroundingRecord,
) -> None:
    if (
        record.global_index != evidence.global_index
        or record.task_key != evidence.task_key
        or hashlib.sha256(record.instruction.encode("utf-8")).hexdigest()
        != evidence.instruction_sha256
        or record.target_identity_key != evidence.target_identity_key
        or record.camera_name != evidence.camera_name
        or record.source_rgb_sha256 != evidence.source_rgb_sha256
        or record.qwen_bbox_xyxy != evidence.bbox_qwen_xyxy
    ):
        raise ContractError("fixed-X crossed runtime record differs from frozen evidence")


def _materialize_crossed_x_records(
    *,
    index: Any,
    sidecar: Any,
    plan: CrossedEvaluationPlan,
    curriculum: NativeVLGroundingCurriculumPlan,
    materialize_record: Any,
) -> tuple[_CrossedXRuntimeRecord, ...]:
    records = []
    for evidence in plan.unique_records:
        group, variant = plan.resolve_record(curriculum.groups, evidence)
        record = materialize_record(
            index=index,
            sidecar=sidecar,
            group=group,
            variant=variant,
            expected_camera_name=evidence.camera_name,
        )
        if not isinstance(record, CalvinQwenGroundingRecord):
            raise TypeError("fixed-X crossed materialization returned an invalid record")
        _validate_crossed_runtime_record(evidence, record)
        records.append(_CrossedXRuntimeRecord(evidence=evidence, record=record))
    if len(records) != len(plan.unique_records):
        raise RuntimeError("fixed-X crossed materialization lost records")
    return tuple(records)


def _materialize_scene_bank(
    *,
    audit: Mapping[str, object],
    index: Any,
    sidecar: Any,
    dataset_tree_sha256: str,
    picf_code_revision: str,
) -> tuple[_SceneRecordPair, ...]:
    """Rebuild every frozen scene-bank item from the bound RGB and sidecar."""

    if audit.get("dataset_tree_sha256") != dataset_tree_sha256:
        raise ContractError("fixed-X scene audit belongs to another dataset tree")
    if audit.get("physical_sidecar_manifest_sha256") != sidecar.manifest_sha256:
        raise ContractError("fixed-X scene audit belongs to another physical sidecar")
    if audit.get("picf_code_revision") != picf_code_revision:
        raise ContractError("fixed-X scene audit belongs to another PICF revision")
    if audit.get("visual_lattice") != LATTICE_BASELINE:
        raise ContractError("fixed-X scene audit lattice changed")
    arm = audit.get("arm_steps")
    bank = audit.get("source_disjoint_scene_bank")
    if not isinstance(arm, list) or len(arm) != 64 or not isinstance(bank, list) or len(bank) != 32:
        raise ContractError("fixed-X scene audit dimensions changed")

    arm_groups: set[int] = set()
    arm_global_indices: set[int] = set()
    arm_rgb_sha256: set[str] = set()
    for row in arm:
        if not isinstance(row, Mapping):
            raise ContractError("fixed-X scene arm row is malformed")
        group_index = row.get("group_index")
        if isinstance(group_index, bool) or not isinstance(group_index, int) or group_index < 0:
            raise ContractError("fixed-X scene arm group index is invalid")
        global_index = row.get("global_index")
        if isinstance(global_index, bool) or not isinstance(global_index, int) or global_index < 0:
            raise ContractError("fixed-X scene arm source index is invalid")
        arm_groups.add(group_index)
        arm_global_indices.add(global_index)
        arm_rgb_sha256.add(
            _validate_sha256(row.get("source_rgb_sha256"), name="scene arm RGB digest")
        )

    pairs = []
    bank_groups: set[int] = set()
    bank_global_indices: set[int] = set()
    bank_rgb_sha256: set[str] = set()
    for expected_bank_index, row in enumerate(bank):
        if not isinstance(row, Mapping):
            raise ContractError("fixed-X scene bank row is malformed")
        bank_index = row.get("bank_index")
        group_index = row.get("group_index")
        global_index = row.get("global_index")
        camera_name = row.get("camera_name")
        task_keys = row.get("task_keys")
        object_identity_keys = row.get("object_identity_keys")
        if bank_index != expected_bank_index:
            raise ContractError("fixed-X scene bank order changed")
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or group_index < 0
            or group_index in arm_groups
            or group_index in bank_groups
        ):
            raise ContractError("fixed-X scene bank is not source-disjoint")
        if isinstance(global_index, bool) or not isinstance(global_index, int) or global_index < 0:
            raise ContractError("fixed-X scene bank source index is invalid")
        source_rgb_sha256 = _validate_sha256(
            row.get("source_rgb_sha256"),
            name="scene bank RGB digest",
        )
        if (
            global_index in arm_global_indices
            or source_rgb_sha256 in arm_rgb_sha256
            or global_index in bank_global_indices
            or source_rgb_sha256 in bank_rgb_sha256
        ):
            raise ContractError("fixed-X scene bank reuses one source observation")
        if camera_name not in ("static", "gripper"):
            raise ContractError("fixed-X scene bank camera is invalid")
        validated_task_keys = _validate_scene_bank_task_keys(task_keys)
        if not isinstance(object_identity_keys, list) or any(
            not isinstance(value, str) for value in object_identity_keys
        ):
            raise ContractError("fixed-X scene bank object identities are invalid")
        if len(set(object_identity_keys)) != len(object_identity_keys) or not set(
            object_identity_keys
        ).issubset(CALVIN_QWEN_SCENE_IDENTITY_ORDER):
            raise ContractError("fixed-X scene bank object identities changed")
        bank_groups.add(group_index)
        bank_global_indices.add(global_index)
        bank_rgb_sha256.add(source_rgb_sha256)

        arrays = dict(
            index.validated_source_frame_arrays(
                global_index,
                fields=("rgb_gripper", "rgb_static"),
            )
        )
        image = arrays["rgb_static" if camera_name == "static" else "rgb_gripper"]
        physical = sidecar.source_frame(global_index)
        canonical = build_calvin_qwen_scene_grounding_record(
            global_index=global_index,
            camera_name=camera_name,
            image=image,
            physical_frame=physical,
            category_identity_order=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
            visual_lattice=LATTICE_BASELINE,
        )
        reverse = build_calvin_qwen_scene_grounding_record(
            global_index=global_index,
            camera_name=camera_name,
            image=image,
            physical_frame=physical,
            category_identity_order=tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER)),
            visual_lattice=LATTICE_BASELINE,
        )
        if (
            canonical.source_rgb_sha256 != reverse.source_rgb_sha256
            or not np.array_equal(canonical.image, reverse.image)
            or canonical.source_rgb_sha256 != source_rgb_sha256
            or [item.identity_key for item in canonical.objects] != object_identity_keys
            or hashlib.sha256(canonical.assistant_text.encode("utf-8")).hexdigest()
            != row.get("canonical_answer_sha256")
            or hashlib.sha256(reverse.assistant_text.encode("utf-8")).hexdigest()
            != row.get("reverse_answer_sha256")
        ):
            raise ContractError("fixed-X scene bank differs from reconstructed supervision")
        canonical_evidence = {
            item.identity_key: (
                item.bbox_xyxy,
                item.visible_owner_pixels,
                item.projected_target_mass,
                item.positive_visual_token_count,
            )
            for item in (*canonical.objects, *canonical.subpatch_objects)
        }
        reverse_evidence = {
            item.identity_key: (
                item.bbox_xyxy,
                item.visible_owner_pixels,
                item.projected_target_mass,
                item.positive_visual_token_count,
            )
            for item in (*reverse.objects, *reverse.subpatch_objects)
        }
        if canonical_evidence != reverse_evidence:
            raise ContractError("fixed-X scene order pair changed its physical evidence")
        pairs.append(
            _SceneRecordPair(
                bank_index=expected_bank_index,
                group_index=group_index,
                task_keys=validated_task_keys,
                records=(canonical, reverse),
            )
        )
    return tuple(pairs)


def _validate_scene_bank_task_keys(value: object) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(task_key, str) or not task_key for task_key in value)
        or len(set(value)) != len(value)
    ):
        raise ContractError("fixed-X scene bank task keys are invalid")
    return tuple(value)


def _select_pairs(
    pairs: tuple[_FixedXRecordPair, ...],
    *,
    partition: str,
    limit_per_partition: int,
) -> tuple[_FixedXRecordPair, ...]:
    partitions = FIXED_OBSERVATION_EVALUATION_PARTITIONS if partition == "all" else (partition,)
    selected = []
    for name in partitions:
        values = tuple(pair for pair in pairs if pair.item.partition == name)
        if not values:
            raise ContractError(f"fixed-X native VL has no eligible {name} pairs")
        if limit_per_partition > len(values):
            raise ContractError(f"fixed-X native VL has too few eligible {name} pairs")
        selected.extend(values if limit_per_partition == 0 else values[:limit_per_partition])
    return tuple(selected)


def _normalized_to_pixel_bbox(
    bbox: tuple[int, int, int, int] | None,
    *,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    if bbox is None:
        return None
    values = tuple(
        int(round(value * extent / 1000.0))
        for value, extent in zip(bbox, (width, height, width, height), strict=True)
    )
    return (
        max(0, min(width - 1, values[0])),
        max(0, min(height - 1, values[1])),
        max(1, min(width, values[2])),
        max(1, min(height, values[3])),
    )


def _draw_scaled_box(
    draw: Any,
    box: tuple[int, int, int, int] | None,
    *,
    image_left: int,
    header_height: int,
    scale: int,
    color: str,
    width: int,
) -> None:
    if box is None:
        return
    x0, y0, x1, y1 = box
    draw.rectangle(
        (
            image_left + x0 * scale,
            header_height + y0 * scale,
            image_left + max(x0 * scale, x1 * scale - 1),
            header_height + max(y0 * scale, y1 * scale - 1),
        ),
        outline=color,
        width=width,
    )


def _render_pair(
    pair: _FixedXRecordPair,
    predictions: tuple[
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
    ],
    output: Path,
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    header_height = 118
    panel_width = 520
    source_height, source_width = pair.records[0].image.shape[:2]
    canvas = Image.new("RGB", (panel_width * 2, header_height + source_height * scale), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    for index, (record, prediction) in enumerate(zip(pair.records, predictions, strict=True)):
        left = index * panel_width
        image = Image.fromarray(record.image).resize(
            (source_width * scale, source_height * scale),
            resample=Image.Resampling.NEAREST,
        )
        image_left = left + (panel_width - image.width) // 2
        canvas.paste(image, (image_left, header_height))
        draw = ImageDraw.Draw(canvas)
        variant = pair.item.variants[index]
        alternate = pair.item.variants[1 - index]
        lines = (
            f"{pair.item.partition} item={pair.item.ordinal} step={record.global_index}",
            f"task={variant.task_key} target={variant.target_identity_key}",
            f"instruction={variant.instruction[:68]}",
            f"prediction={prediction}",
            f"green=target red={alternate.target_identity_key} cyan=prediction",
        )
        for line_index, line in enumerate(lines):
            draw.text((left + 5, 4 + 21 * line_index), line, fill="black", font=font)

        _draw_scaled_box(
            draw,
            pair.records[1 - index].bbox_xyxy,
            image_left=image_left,
            header_height=header_height,
            scale=scale,
            color="red",
            width=2,
        )
        _draw_scaled_box(
            draw,
            record.bbox_xyxy,
            image_left=image_left,
            header_height=header_height,
            scale=scale,
            color="lime",
            width=3,
        )
        _draw_scaled_box(
            draw,
            _normalized_to_pixel_bbox(prediction, width=source_width, height=source_height),
            image_left=image_left,
            header_height=header_height,
            scale=scale,
            color="cyan",
            width=2,
        )
    canvas.save(output, format="PNG")
    return _sha256(output)


def _render_crossed_x_pair(
    *,
    pair_index: int,
    pair: CrossedEvaluationPair,
    records: tuple[CalvinQwenGroundingRecord, CalvinQwenGroundingRecord],
    predictions: tuple[
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
    ],
    output: Path,
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    source_shapes = tuple(record.image.shape[:2] for record in records)
    if source_shapes[0] != source_shapes[1]:
        raise ContractError("crossed exact-X pair changed camera geometry")
    source_height, source_width = source_shapes[0]
    scale = 2
    header_height = 138
    panel_width = max(520, source_width * scale + 20)
    canvas = Image.new("RGB", (panel_width * 2, header_height + source_height * scale), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    evidence = (pair.first, pair.second)
    for index, (frozen, record, prediction) in enumerate(
        zip(evidence, records, predictions, strict=True)
    ):
        left = index * panel_width
        image = Image.fromarray(record.image).resize(
            (source_width * scale, source_height * scale),
            resample=Image.Resampling.NEAREST,
        )
        image_left = left + (panel_width - image.width) // 2
        canvas.paste(image, (image_left, header_height))
        draw = ImageDraw.Draw(canvas)
        other_target = evidence[1 - index].bbox_qwen_xyxy
        lines = (
            f"heldout exact-X pair={pair_index:03d} source={record.global_index}",
            f"task={record.task_key} target={record.target_identity_key}",
            f"camera={record.camera_name} episode={frozen.source_episode_index}",
            f"instruction={record.instruction[:68]}",
            f"prediction={prediction}",
            "green=own target | red=other-source coordinate | cyan=prediction",
        )
        for line_index, line in enumerate(lines):
            draw.text((left + 5, 4 + 21 * line_index), line, fill="black", font=font)
        _draw_scaled_box(
            draw,
            _normalized_to_pixel_bbox(
                other_target,
                width=source_width,
                height=source_height,
            ),
            image_left=image_left,
            header_height=header_height,
            scale=scale,
            color="red",
            width=2,
        )
        _draw_scaled_box(
            draw,
            record.bbox_xyxy,
            image_left=image_left,
            header_height=header_height,
            scale=scale,
            color="lime",
            width=3,
        )
        _draw_scaled_box(
            draw,
            _normalized_to_pixel_bbox(
                prediction,
                width=source_width,
                height=source_height,
            ),
            image_left=image_left,
            header_height=header_height,
            scale=scale,
            color="cyan",
            width=2,
        )
    canvas.save(output, format="PNG")
    return _sha256(output)


def _render_scene_pair(
    pair: _SceneRecordPair,
    generated: tuple[NativeVLGeneratedSceneGrounding, NativeVLGeneratedSceneGrounding],
    output: Path,
) -> str:
    """Render expected and generated label-addressed boxes for both order prompts."""

    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    header_height = 142
    panel_width = 520
    source_height, source_width = pair.records[0].image.shape[:2]
    canvas = Image.new("RGB", (panel_width * 2, header_height + source_height * scale), "white")
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 11)
    except OSError:
        font = ImageFont.load_default()
    for variant_index, (record, prediction) in enumerate(zip(pair.records, generated, strict=True)):
        left = variant_index * panel_width
        image = Image.fromarray(record.image).resize(
            (source_width * scale, source_height * scale),
            resample=Image.Resampling.NEAREST,
        )
        image_left = left + (panel_width - image.width) // 2
        canvas.paste(image, (image_left, header_height))
        draw = ImageDraw.Draw(canvas)
        order_labels = [
            qwen_grounding_label(identity_key) for identity_key in record.category_identity_order
        ]
        order_variant = "canonical" if variant_index == 0 else "reverse"
        task_summary = " | ".join(pair.task_keys[:2])
        if len(pair.task_keys) > 2:
            task_summary = f"{task_summary} | +{len(pair.task_keys) - 2}"
        lines = (
            f"scene-bank={pair.bank_index} group={pair.group_index} step={record.global_index}",
            f"camera={record.camera_name} variant={order_variant}",
            f"tasks={task_summary}",
            "order=" + ",".join(order_labels[:5]),
            "      " + ",".join(order_labels[5:]),
            (
                f"schema={prediction.schema_valid} expected={len(record.objects)} "
                f"generated={len(prediction.objects)}"
            ),
            "green=target cyan=generated",
        )
        for line_index, line in enumerate(lines):
            draw.text((left + 5, 4 + 19 * line_index), line, fill="black", font=font)

        for item in record.objects:
            _draw_scaled_box(
                draw,
                item.bbox_xyxy,
                image_left=image_left,
                header_height=header_height,
                scale=scale,
                color="lime",
                width=3,
            )
            x0, y0, _x1, _y1 = item.bbox_xyxy
            draw.text(
                (image_left + x0 * scale, header_height + y0 * scale),
                qwen_grounding_label(item.identity_key),
                fill="lime",
                stroke_fill="black",
                stroke_width=2,
                font=font,
            )
        expected_labels = {
            normalize_scene_label(qwen_grounding_label(item.identity_key))
            for item in record.objects
        }
        for item in prediction.objects:
            pixel_box = _normalized_to_pixel_bbox(
                item.bbox_qwen_xyxy,
                width=source_width,
                height=source_height,
            )
            _draw_scaled_box(
                draw,
                pixel_box,
                image_left=image_left,
                header_height=header_height,
                scale=scale,
                color="cyan",
                width=2,
            )
            if pixel_box is not None:
                x0, _y0, _x1, y1 = pixel_box
                label = normalize_scene_label(item.label)
                draw.text(
                    (image_left + x0 * scale, header_height + max(0, y1 * scale - 14)),
                    item.label,
                    fill="cyan" if label in expected_labels else "magenta",
                    stroke_fill="black",
                    stroke_width=2,
                    font=font,
                )
    canvas.save(output, format="PNG")
    return _sha256(output)


def _public_retention_summary(rows: list[dict[str, Any]]) -> dict[str, object]:
    if not rows:
        raise ContractError("fixed-X public retention evaluation has no rows")
    summary = {}
    for family in PUBLIC_NATIVE_VL_FAMILIES:
        selected = [row for row in rows if row["family"] == family]
        if not selected:
            raise ContractError(f"fixed-X public retention evaluation omits {family}")
        supervised_tokens = sum(int(row["supervised_token_count"]) for row in selected)
        if supervised_tokens <= 0:
            raise ContractError("fixed-X public retention has no supervised tokens")
        values: dict[str, object] = {
            "mean_record_nll": sum(float(row["mean_token_nll"]) for row in selected)
            / len(selected),
            "record_count": len(selected),
            "supervised_token_count": supervised_tokens,
            "token_weighted_mean_nll": sum(
                float(row["mean_token_nll"]) * int(row["supervised_token_count"])
                for row in selected
            )
            / supervised_tokens,
        }
        if family == "referring":
            values.update(
                {
                    "generated_bbox_count": sum(
                        row["generated_bbox_qwen_xyxy"] is not None for row in selected
                    ),
                    "generated_bbox_schema_valid_count": sum(
                        bool(row["generated_bbox_schema_valid"]) for row in selected
                    ),
                    "mean_target_iou": sum(float(row["target_iou"]) for row in selected)
                    / len(selected),
                    "target_center_hit_count": sum(
                        bool(row["target_center_hit"]) for row in selected
                    ),
                }
            )
        else:
            values["normalized_exact_match_count"] = sum(
                bool(row["normalized_exact_match"]) for row in selected
            )
        summary[family] = values
    return summary


def _text_sha256(value: str, *, name: str) -> str:
    if not isinstance(value, str) or not value or "\0" in value:
        raise ContractError(f"fixed-X {name} must be nonempty text")
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _calvin_semantic_evidence(
    record: CalvinQwenGroundingRecord,
    generated: NativeVLGeneratedGrounding,
) -> dict[str, object]:
    """Bind free-generated semantics to the exact assistant-only CALVIN target."""

    if not isinstance(record, CalvinQwenGroundingRecord):
        raise TypeError("fixed-X semantic evidence requires a CALVIN grounding record")
    if not isinstance(generated, NativeVLGeneratedGrounding):
        raise TypeError("fixed-X semantic evidence requires parsed native VL grounding")
    grounding_request = record.grounding_request
    target_answer = record.assistant_text
    target = parse_native_vl_grounding_answer(target_answer)
    if (
        not target.schema_valid
        or target.bbox_qwen_xyxy != record.qwen_bbox_xyxy
        or not target.label_present
        or not target.label_schema_valid
        or target.generated_label != record.qwen_label
    ):
        raise ContractError("fixed-X CALVIN semantic target serialization is malformed")
    if target.generated_label is None:
        raise RuntimeError("validated CALVIN semantic target lost its label")
    normalized_exact_match = (
        generated.label_schema_valid
        and generated.generated_label is not None
        and _normalize_generated_answer(generated.generated_label)
        == _normalize_generated_answer(target.generated_label)
    )
    return {
        "generated_label": generated.generated_label,
        "generated_label_present": generated.label_present,
        "generated_label_schema_valid": generated.label_schema_valid,
        "grounding_request": grounding_request,
        "grounding_request_sha256": _text_sha256(
            grounding_request,
            name="CALVIN grounding request",
        ),
        "normalized_label_exact_match": normalized_exact_match,
        "target_answer": target_answer,
        "target_answer_sha256": _text_sha256(
            target_answer,
            name="CALVIN target answer",
        ),
        "target_label": target.generated_label,
    }


def _semantic_partition_summary(groups: list[dict[str, Any]]) -> dict[str, int]:
    """Recompute fail-closed CALVIN label counts from raw fixed-X evidence."""

    if not isinstance(groups, list) or not groups:
        raise ContractError("fixed-X semantic partition summary requires pair rows")
    variants: list[Mapping[str, object]] = []
    for group in groups:
        if not isinstance(group, Mapping):
            raise ContractError("fixed-X semantic partition group is malformed")
        group_variants = group.get("variants")
        if (
            not isinstance(group_variants, list)
            or len(group_variants) != 2
            or any(not isinstance(variant, Mapping) for variant in group_variants)
        ):
            raise ContractError("fixed-X semantic partition variants are malformed")
        variants.extend(group_variants)

    present_count = 0
    schema_valid_count = 0
    exact_match_count = 0
    for variant in variants:
        generated_text = variant.get("generated_text")
        generated_label = variant.get("generated_label")
        target_label = variant.get("target_label")
        if not isinstance(generated_text, str):
            raise ContractError("fixed-X generated answer must be text")
        if generated_label is not None and (
            not isinstance(generated_label, str)
            or not generated_label
            or generated_label != generated_label.strip()
        ):
            raise ContractError("fixed-X generated label is malformed")
        if (
            not isinstance(target_label, str)
            or not target_label
            or target_label != target_label.strip()
        ):
            raise ContractError("fixed-X target label is malformed")
        flags = tuple(
            variant.get(field)
            for field in (
                "generated_label_present",
                "generated_label_schema_valid",
                "normalized_label_exact_match",
            )
        )
        if any(not isinstance(value, bool) for value in flags):
            raise ContractError("fixed-X semantic flags must be boolean")
        label_present, label_schema_valid, normalized_exact_match = cast(
            tuple[bool, bool, bool], flags
        )
        parsed_generated = parse_native_vl_grounding_answer(generated_text)
        if (
            parsed_generated.generated_label != generated_label
            or parsed_generated.label_present != label_present
            or parsed_generated.label_schema_valid != label_schema_valid
        ):
            raise ContractError("fixed-X generated text and label evidence differ")
        expected_exact_match = (
            label_schema_valid
            and generated_label is not None
            and _normalize_generated_answer(generated_label)
            == _normalize_generated_answer(target_label)
        )
        if normalized_exact_match != expected_exact_match:
            raise ContractError("fixed-X normalized label exact match is inconsistent")

        target_answer = variant.get("target_answer")
        target_bbox = variant.get("target_bbox_qwen_xyxy")
        if not isinstance(target_answer, str) or not target_answer or "\0" in target_answer:
            raise ContractError("fixed-X target answer must be nonempty text")
        parsed_target = parse_native_vl_grounding_answer(target_answer)
        if (
            not parsed_target.schema_valid
            or not parsed_target.label_schema_valid
            or parsed_target.generated_label != target_label
            or not isinstance(target_bbox, list)
            or parsed_target.bbox_qwen_xyxy != tuple(target_bbox)
        ):
            raise ContractError("fixed-X target answer and semantic evidence differ")
        for text_field in ("grounding_request", "target_answer"):
            text_value = variant.get(text_field)
            digest_value = variant.get(f"{text_field}_sha256")
            if not isinstance(text_value, str):
                raise ContractError(f"fixed-X {text_field} must be text")
            if digest_value != _text_sha256(text_value, name=text_field):
                raise ContractError(f"fixed-X {text_field} digest differs")

        present_count += int(label_present)
        schema_valid_count += int(label_schema_valid)
        exact_match_count += int(normalized_exact_match)

    return {
        "generated_label_present_count": present_count,
        "generated_label_schema_valid_count": schema_valid_count,
        "item_count": len(groups),
        "normalized_label_exact_match_count": exact_match_count,
        "variant_count": len(variants),
    }


def _render_public_referring_prediction(
    *,
    image: np.ndarray,
    record_id: str,
    user_text: str,
    target: tuple[int, int, int, int],
    prediction: tuple[int, int, int, int] | None,
    output: Path,
) -> str:
    from PIL import Image, ImageDraw, ImageFont

    scale = 2
    header_height = 96
    source_height, source_width = image.shape[:2]
    canvas = Image.new(
        "RGB",
        (source_width * scale, header_height + source_height * scale),
        "white",
    )
    source = Image.fromarray(image).resize(
        (source_width * scale, source_height * scale),
        resample=Image.Resampling.NEAREST,
    )
    canvas.paste(source, (0, header_height))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)
    lines = (
        f"record={record_id}",
        f"prompt={user_text[:100]}",
        f"target={target}",
        f"prediction={prediction} green=target cyan=prediction",
    )
    for line_index, line in enumerate(lines):
        draw.text((5, 4 + 21 * line_index), line, fill="black", font=font)
    _draw_scaled_box(
        draw,
        _normalized_to_pixel_bbox(target, width=source_width, height=source_height),
        image_left=0,
        header_height=header_height,
        scale=scale,
        color="lime",
        width=3,
    )
    _draw_scaled_box(
        draw,
        _normalized_to_pixel_bbox(prediction, width=source_width, height=source_height),
        image_left=0,
        header_height=header_height,
        scale=scale,
        color="cyan",
        width=2,
    )
    canvas.save(output, format="PNG")
    return _sha256(output)


def main() -> None:
    args = _parse_args()
    partial = _validate_args(args)
    patch_report = verify_native_vl_patch(root=_ROOT.resolve(), checkout=args.source_checkout)
    overlay = _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH
    if detect_native_vl_patch_state(args.source_checkout, overlay) != "applied":
        raise RuntimeError("fixed-X native VL source overlay is not applied")
    if _validate_native_vl_model(args.source_checkout / MODEL_SOURCE) != (
        NATIVE_VL_PATCHED_MODEL_SHA256
    ):
        raise RuntimeError("fixed-X native VL source digest differs")
    commit = subprocess.run(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("fixed-X native VL source commit differs")
    picf_commit = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if picf_commit != args.picf_code_revision:
        raise RuntimeError("fixed-X native VL checkout differs from its declared PICF revision")
    runtime_python_trees = adr127_runtime_python_trees_contract(
        repo_root=_ROOT,
        revision=args.picf_code_revision,
        source_checkout=args.source_checkout,
    )
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    checkpoint_model_file_sha256 = _model_hashes(args.checkpoint_dir)

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.name,
    )
    plan = FixedObservationEvaluationPlan.load(args.evaluation_plan)
    if plan.dataset_tree_sha256 != manifest.tree_sha256:
        raise ContractError("fixed-X evaluation plan belongs to another dataset tree")
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root, index)
    crossed_plan = args.crossed_evaluation_plan_object
    crossed_curriculum = args.crossed_curriculum_plan_object
    crossed_scene_audit = args.crossed_scene_audit_report_object
    crossed_presence = tuple(
        value is not None for value in (crossed_plan, crossed_curriculum, crossed_scene_audit)
    )
    if any(crossed_presence) and not all(crossed_presence):
        raise RuntimeError("fixed-X crossed evaluation lost one typed input")
    if crossed_plan is not None:
        if (
            not isinstance(crossed_plan, CrossedEvaluationPlan)
            or not isinstance(crossed_curriculum, NativeVLGroundingCurriculumPlan)
            or not isinstance(crossed_scene_audit, Mapping)
        ):
            raise RuntimeError("fixed-X crossed evaluation lost typed inputs")
        if (
            crossed_plan.dataset_manifest_sha256 != manifest.tree_sha256
            or crossed_curriculum.dataset_manifest_sha256 != manifest.tree_sha256
            or crossed_plan.dataset_id != manifest.dataset_id
            or crossed_plan.dataset_revision != manifest.dataset_revision
        ):
            raise ContractError("fixed-X crossed evaluation belongs to another dataset tree")
    scene_audit = args.scene_audit_report_object
    if scene_audit is not None and not isinstance(scene_audit, Mapping):
        raise RuntimeError("fixed-X native VL lost its typed scene audit")
    if (
        scene_audit is not None
        and scene_audit.get("runtime_python_tree") != runtime_python_trees["picf"]
    ):
        raise ContractError("fixed-X scene audit runtime tree differs from evaluation")
    if (
        crossed_scene_audit is not None
        and crossed_scene_audit.get("runtime_python_tree") != runtime_python_trees["picf"]
    ):
        raise ContractError("fixed-X crossed scene audit runtime tree differs from evaluation")
    scene_pairs = (
        ()
        if scene_audit is None
        else _materialize_scene_bank(
            audit=scene_audit,
            index=index,
            sidecar=sidecar,
            dataset_tree_sha256=manifest.tree_sha256,
            picf_code_revision=args.picf_code_revision,
        )
    )
    eligible = []
    excluded = []
    for item in plan.items:
        pair = _materialize_pair(index=index, sidecar=sidecar, item=item)
        if pair is None:
            excluded.append(
                {
                    "ordinal": item.ordinal,
                    "partition": item.partition,
                    "reason": "no-common-visible-camera",
                    "source_global_index": item.group.source_global_index,
                }
            )
        else:
            eligible.append(pair)
    selected = _select_pairs(
        tuple(eligible),
        partition=args.partition,
        limit_per_partition=args.item_limit_per_partition,
    )
    public_retention_manifest = args.public_vl_retention_manifest_object
    if public_retention_manifest is not None and not isinstance(
        public_retention_manifest,
        PublicNativeVLRetentionManifest,
    ):
        raise RuntimeError("fixed-X native VL lost its typed public retention manifest")

    sys.path.insert(0, str(args.source_checkout.resolve()))
    import torch
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from transformers import AutoConfig
    from transformers.modeling_utils import load_sharded_checkpoint, no_init_weights

    from picf_next.lingbot_native.vl_cotraining import (
        build_native_vl_generation_batch,
        build_native_vl_grounding_batch,
        generate_native_vl_answer,
        materialize_fixed_observation_native_vl_record,
        retie_and_validate_native_qwen_lm_head,
        run_native_vl_grounding_forward,
    )

    crossed_runtime_records = (
        ()
        if crossed_plan is None or crossed_curriculum is None
        else _materialize_crossed_x_records(
            index=index,
            sidecar=sidecar,
            plan=crossed_plan,
            curriculum=crossed_curriculum,
            materialize_record=materialize_fixed_observation_native_vl_record,
        )
    )

    device = torch.device(args.device)
    dtype = torch.bfloat16
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    training = load_lingbot_training_config(args.training_config)
    merged, _ = _resolve_training_config(
        training,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        num_steps=1,
    )
    merged.update(
        {
            "attention_implementation": "eager",
            "use_cache": False,
            "use_compile": False,
            "use_lm_head": True,
            "vit_attn_implementation": "eager",
        }
    )
    config = LingbotVLAV2Config(**merged)
    for key, value in merged.items():
        if not hasattr(config, key):
            setattr(config, key, value)
    # QWEN_PROCESSOR_REVISION is an exact commit and this load is local-only.
    qwen_config = AutoConfig.from_pretrained(  # nosec B615
        args.processor_dir,
        revision=QWEN_PROCESSOR_REVISION,
        local_files_only=True,
    )
    _merge_qwen_config(config, qwen_config)
    config.tokenizer_path = str(args.processor_dir.resolve())
    config.use_lm_head = True

    qwen_restore = None
    if args.restore_qwen_dir is not None:
        qwen_restore = {
            "model_dir": str(args.restore_qwen_dir.resolve()),
            "model_file_sha256": _model_hashes(args.restore_qwen_dir),
            "model_revision": args.restore_qwen_revision,
        }

    processor = build_processor(str(args.processor_dir.resolve()))
    processor_lattice = configure_native_processor_lattice(
        processor,
        args.visual_lattice,
    )
    scene_generation_budget = None
    if scene_pairs:
        scene_supervised_token_counts = []
        for pair in scene_pairs:
            for record in pair.records:
                supervised_batch = build_native_vl_grounding_batch(record, processor)
                scene_supervised_token_counts.append(supervised_batch.supervised_token_count)
                del supervised_batch
        scene_generation_budget = _scene_generation_budget_contract(
            scene_supervised_token_counts,
            max_new_tokens=args.scene_max_new_tokens,
        )
    public_processor = None
    public_processor_contract = None
    if public_retention_manifest is not None:
        public_processor = build_processor(str(args.processor_dir.resolve()))
        public_processor_contract = configure_native_processor_area_budget(
            public_processor,
            LATTICE_BASELINE,
        )
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    load_started = time.perf_counter()
    with init_empty_weights(), no_init_weights():
        policy = LingbotVlaV2Policy(config=config, eval=True).to(dtype)
    preload_tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
    load_model_weights(
        policy,
        str(args.checkpoint_dir.resolve()),
        str(device),
        post_training=True,
        adanorm_time=bool(config.adanorm_time),
    )
    tied_parameter_name = retie_and_validate_native_qwen_lm_head(policy)
    if tied_parameter_name != preload_tied_parameter_name:
        raise ContractError("fixed-X Qwen tied parameter name drifted during released load")
    if args.restore_qwen_dir is not None and qwen_restore is not None:
        qwen_restore["load_result"] = _validate_qwen_restore_load_result(
            load_sharded_checkpoint(
                policy.model.qwenvl_with_expert.qwenvl,
                args.restore_qwen_dir,
                strict=False,
                prefer_safe=True,
            )
        )
        restored_name = retie_and_validate_native_qwen_lm_head(policy)
        if restored_name != tied_parameter_name:
            raise ContractError("fixed-X Qwen tied parameter name drifted during restoration")
    teacher_prune = strip_targetless_alignment_teacher_heads(policy)
    policy.eval()
    load_seconds = time.perf_counter() - load_started

    partial.mkdir(parents=True)
    visual_dir = partial / "visuals"
    visual_dir.mkdir()
    public_visual_dir = None
    if public_retention_manifest is not None:
        public_visual_dir = partial / "public_retention_visuals"
        public_visual_dir.mkdir()
    scene_visual_dir = None
    if scene_audit is not None:
        scene_visual_dir = partial / "scene_visuals"
        scene_visual_dir.mkdir()
    crossed_visual_dir = None
    if crossed_plan is not None:
        crossed_visual_dir = partial / "crossed_exact_x_visuals"
        crossed_visual_dir.mkdir()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    groups = []
    public_retention_rows = []
    public_retention_started = None
    scene_rows = []
    scene_started = None
    scene_elapsed = None
    crossed_rows = []
    crossed_started = None
    crossed_elapsed = None
    with torch.inference_mode():
        for pair in selected:
            variant_values = []
            predictions = []
            for record in pair.records:
                batch = build_native_vl_generation_batch(record, processor).to(
                    device,
                    pixel_dtype=dtype,
                )
                generated_text = generate_native_vl_answer(
                    policy.model.qwenvl_with_expert.qwenvl,
                    batch,
                    processor.tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
                generated = parse_native_vl_grounding_answer(generated_text)
                semantic_evidence = _calvin_semantic_evidence(record, generated)
                predictions.append(generated.bbox_qwen_xyxy)
                variant_values.append(
                    {
                        "camera_name": record.camera_name,
                        "generated_bbox_qwen_xyxy": (
                            list(generated.bbox_qwen_xyxy)
                            if generated.bbox_qwen_xyxy is not None
                            else None
                        ),
                        "generated_bbox_schema_valid": generated.schema_valid,
                        "generated_text": generated_text,
                        "instruction": record.instruction,
                        "instruction_sha256": hashlib.sha256(
                            record.instruction.encode("utf-8")
                        ).hexdigest(),
                        **semantic_evidence,
                        "target_bbox_qwen_xyxy": list(record.qwen_bbox_xyxy),
                        "target_identity_key": record.target_identity_key,
                        "task_key": record.task_key,
                    }
                )
            prediction_pair = (predictions[0], predictions[1])
            metrics = _pair_geometry_metrics(
                prediction_pair,
                (pair.records[0].qwen_bbox_xyxy, pair.records[1].qwen_bbox_xyxy),
            )
            metric_variants = metrics.get("variants")
            if not isinstance(metric_variants, list) or any(
                not isinstance(metric, dict) for metric in metric_variants
            ):
                raise RuntimeError("fixed-X pair geometry variants are malformed")
            for value, metric in zip(variant_values, metric_variants, strict=True):
                value.update(metric)
            visual_name = (
                f"{pair.item.partition}-{pair.item.ordinal:03d}_"
                f"step-{pair.item.group.source_global_index}_"
                f"{pair.records[0].task_key}__{pair.records[1].task_key}.png"
            )
            visual_sha256 = _render_pair(pair, prediction_pair, visual_dir / visual_name)
            groups.append(
                {
                    "ordinal": pair.item.ordinal,
                    "pair_metrics": {
                        key: value for key, value in metrics.items() if key != "variants"
                    },
                    "partition": pair.item.partition,
                    "source_global_index": pair.item.group.source_global_index,
                    "source_rgb_sha256": pair.records[0].source_rgb_sha256,
                    "source_state_sha256": pair.item.group.source_state_sha256,
                    "variants": variant_values,
                    "visual": {"file": f"visuals/{visual_name}", "sha256": visual_sha256},
                }
            )
        if crossed_plan is not None:
            if crossed_visual_dir is None:
                raise RuntimeError("fixed-X crossed visual directory is missing")
            crossed_started = time.perf_counter()
            generated_by_evidence = {}
            runtime_by_evidence = {}
            for runtime in crossed_runtime_records:
                batch = build_native_vl_generation_batch(runtime.record, processor).to(
                    device,
                    pixel_dtype=dtype,
                )
                generated_text = generate_native_vl_answer(
                    policy.model.qwenvl_with_expert.qwenvl,
                    batch,
                    processor.tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
                generated = parse_native_vl_grounding_answer(generated_text)
                generated_by_evidence[runtime.evidence] = (generated_text, generated)
                runtime_by_evidence[runtime.evidence] = runtime.record
                del batch
            for pair_index, pair in enumerate(crossed_plan.pairs):
                evidence_pair = (pair.first, pair.second)
                record_pair = (
                    runtime_by_evidence[pair.first],
                    runtime_by_evidence[pair.second],
                )
                generated_pair = (
                    generated_by_evidence[pair.first],
                    generated_by_evidence[pair.second],
                )
                predictions = (
                    generated_pair[0][1].bbox_qwen_xyxy,
                    generated_pair[1][1].bbox_qwen_xyxy,
                )
                metrics = _pair_geometry_metrics(
                    predictions,
                    (pair.first.bbox_qwen_xyxy, pair.second.bbox_qwen_xyxy),
                )
                metric_variants = metrics.get("variants")
                if not isinstance(metric_variants, list) or any(
                    not isinstance(metric, dict) for metric in metric_variants
                ):
                    raise RuntimeError("fixed-X crossed geometry variants are malformed")
                variants = []
                for frozen, record, generated_value, metric in zip(
                    evidence_pair,
                    record_pair,
                    generated_pair,
                    metric_variants,
                    strict=True,
                ):
                    generated_text, generated = generated_value
                    variants.append(
                        {
                            "camera_name": record.camera_name,
                            "generated_bbox_qwen_xyxy": (
                                None
                                if generated.bbox_qwen_xyxy is None
                                else list(generated.bbox_qwen_xyxy)
                            ),
                            "generated_bbox_schema_valid": generated.schema_valid,
                            "generated_text": generated_text,
                            "instruction": record.instruction,
                            "instruction_sha256": frozen.instruction_sha256,
                            **_calvin_semantic_evidence(record, generated),
                            **metric,
                            "source_episode_index": frozen.source_episode_index,
                            "source_global_index": frozen.global_index,
                            "source_rgb_sha256": frozen.source_rgb_sha256,
                            "source_state_sha256": frozen.source_state_sha256,
                            "target_bbox_qwen_xyxy": list(record.qwen_bbox_xyxy),
                            "target_identity_key": record.target_identity_key,
                            "task_key": record.task_key,
                        }
                    )
                visual_name = f"exact-x-{pair_index:03d}_{pair.key[:12]}.png"
                visual_sha256 = _render_crossed_x_pair(
                    pair_index=pair_index,
                    pair=pair,
                    records=record_pair,
                    predictions=predictions,
                    output=crossed_visual_dir / visual_name,
                )
                crossed_rows.append(
                    {
                        "pair_key": pair.key,
                        "pair_metrics": {
                            key: value for key, value in metrics.items() if key != "variants"
                        },
                        "variants": variants,
                        "visual": {
                            "file": f"crossed_exact_x_visuals/{visual_name}",
                            "sha256": visual_sha256,
                        },
                    }
                )
            torch.cuda.synchronize(device)
            crossed_elapsed = time.perf_counter() - crossed_started
        if scene_audit is not None:
            if scene_visual_dir is None:
                raise RuntimeError("fixed-X scene visual directory is missing")
            scene_started = time.perf_counter()
            for pair in scene_pairs:
                generated_pair = []
                generated_texts = []
                for record in pair.records:
                    batch = build_native_vl_generation_batch(record, processor).to(
                        device,
                        pixel_dtype=dtype,
                    )
                    generated_text = generate_native_vl_answer(
                        policy.model.qwenvl_with_expert.qwenvl,
                        batch,
                        processor.tokenizer,
                        max_new_tokens=args.scene_max_new_tokens,
                    )
                    generated_pair.append(parse_native_vl_scene_grounding_answer(generated_text))
                    generated_texts.append(generated_text)
                    del batch
                typed_generated_pair = (generated_pair[0], generated_pair[1])
                pair_metrics = native_vl_scene_order_pair_metrics(
                    pair.records,
                    typed_generated_pair,
                )
                metric_variants = pair_metrics.get("variants")
                if (
                    not isinstance(metric_variants, list)
                    or len(metric_variants) != 2
                    or any(not isinstance(value, dict) for value in metric_variants)
                ):
                    raise RuntimeError("fixed-X scene pair metric variants are malformed")
                for variant_index, (metric, record, generated_text) in enumerate(
                    zip(metric_variants, pair.records, generated_texts, strict=True)
                ):
                    metric.update(
                        {
                            "category_identity_order": list(record.category_identity_order),
                            "generated_text": generated_text,
                            "generated_text_sha256": hashlib.sha256(
                                generated_text.encode("utf-8")
                            ).hexdigest(),
                            "grounding_request": record.grounding_request,
                            "grounding_request_sha256": hashlib.sha256(
                                record.grounding_request.encode("utf-8")
                            ).hexdigest(),
                            "order_variant": "canonical" if variant_index == 0 else "reverse",
                            "target_answer": record.assistant_text,
                            "target_answer_sha256": hashlib.sha256(
                                record.assistant_text.encode("utf-8")
                            ).hexdigest(),
                        }
                    )
                visual_name = (
                    f"scene-{pair.bank_index:02d}_group-{pair.group_index}_"
                    f"step-{pair.records[0].global_index}_{pair.records[0].camera_name}.png"
                )
                visual_sha256 = _render_scene_pair(
                    pair,
                    typed_generated_pair,
                    scene_visual_dir / visual_name,
                )
                scene_rows.append(
                    {
                        "bank_index": pair.bank_index,
                        "group_index": pair.group_index,
                        "pair_metrics": pair_metrics,
                        "source_global_index": pair.records[0].global_index,
                        "source_rgb_sha256": pair.records[0].source_rgb_sha256,
                        "task_keys": list(pair.task_keys),
                        "visual": {
                            "file": f"scene_visuals/{visual_name}",
                            "sha256": visual_sha256,
                        },
                    }
                )
            torch.cuda.synchronize(device)
            scene_elapsed = time.perf_counter() - scene_started
        if public_retention_manifest is not None:
            if not isinstance(args.public_vl_retention_root, Path):
                raise RuntimeError("fixed-X native VL lost its public retention root")
            if public_processor is None or public_processor_contract is None:
                raise RuntimeError("fixed-X native VL lost its public retention processor")
            public_retention_started = time.perf_counter()
            for family in PUBLIC_NATIVE_VL_FAMILIES:
                descriptors = public_retention_manifest.records_for(family, "heldout")
                if args.public_vl_heldout_limit_per_family:
                    descriptors = descriptors[: args.public_vl_heldout_limit_per_family]
                for descriptor in descriptors:
                    runtime_record = public_retention_manifest.materialize_record(
                        descriptor,
                        artifact_root=args.public_vl_retention_root,
                    )
                    supervised_batch = build_native_vl_grounding_batch(
                        runtime_record,
                        public_processor,
                    )
                    generation_batch = build_native_vl_generation_batch(
                        runtime_record,
                        public_processor,
                    )
                    supervised_grid_thw = supervised_batch.image_grid_thw.detach().cpu().tolist()
                    generation_grid_thw = generation_batch.image_grid_thw.detach().cpu().tolist()
                    if supervised_grid_thw != generation_grid_thw:
                        raise RuntimeError("fixed-X public supervised/generation grids differ")
                    grid_budget = validate_native_processor_record_grid(
                        supervised_grid_thw,
                        image_height=descriptor.height,
                        image_width=descriptor.width,
                        lattice=LATTICE_BASELINE,
                    )
                    supervised_batch = supervised_batch.to(
                        device,
                        pixel_dtype=dtype,
                    )
                    mean_token_nll = float(
                        run_native_vl_grounding_forward(policy, supervised_batch)
                        .detach()
                        .float()
                        .item()
                    )
                    generation_batch = generation_batch.to(
                        device,
                        pixel_dtype=dtype,
                    )
                    generated_text = generate_native_vl_answer(
                        policy.model.qwenvl_with_expert.qwenvl,
                        generation_batch,
                        public_processor.tokenizer,
                        max_new_tokens=args.max_new_tokens,
                    )
                    row: dict[str, Any] = {
                        "family": descriptor.family,
                        "generated_text": generated_text,
                        "grid_budget": grid_budget,
                        "image_height": descriptor.height,
                        "image_rgb_sha256": descriptor.image_rgb_sha256,
                        "image_grid_thw": supervised_grid_thw,
                        "image_width": descriptor.width,
                        "mean_token_nll": mean_token_nll,
                        "record_id": descriptor.record_id,
                        "record_sha256": descriptor.record_sha256,
                        "source_row_index": descriptor.source_row_index,
                        "source_subindex": descriptor.source_subindex,
                        "supervised_token_count": supervised_batch.supervised_token_count,
                        "target_answer": descriptor.assistant_text,
                        "target_answer_sha256": hashlib.sha256(
                            descriptor.assistant_text.encode("utf-8")
                        ).hexdigest(),
                        "user_text": descriptor.user_text,
                        "user_text_sha256": hashlib.sha256(
                            descriptor.user_text.encode("utf-8")
                        ).hexdigest(),
                    }
                    if descriptor.family == "referring":
                        target = parse_native_vl_grounding_answer(descriptor.assistant_text)
                        if target.bbox_qwen_xyxy is None or not target.schema_valid:
                            raise ContractError(
                                "fixed-X public referring target answer is malformed"
                            )
                        generated = parse_native_vl_grounding_answer(generated_text)
                        prediction = generated.bbox_qwen_xyxy
                        row.update(
                            {
                                "generated_bbox_qwen_xyxy": (
                                    None if prediction is None else list(prediction)
                                ),
                                "generated_bbox_schema_valid": generated.schema_valid,
                                "target_bbox_qwen_xyxy": list(target.bbox_qwen_xyxy),
                                "target_center_hit": (
                                    False
                                    if prediction is None
                                    else qwen_target_center_in_bbox(
                                        prediction,
                                        target.bbox_qwen_xyxy,
                                    )
                                ),
                                "target_iou": (
                                    0.0
                                    if prediction is None
                                    else qwen_grounding_bbox_iou(
                                        prediction,
                                        target.bbox_qwen_xyxy,
                                    )
                                ),
                            }
                        )
                        if public_visual_dir is None:
                            raise RuntimeError(
                                "fixed-X public retention visual directory is missing"
                            )
                        visual_name = f"{descriptor.record_id}.png"
                        row["visual"] = {
                            "file": f"public_retention_visuals/{visual_name}",
                            "sha256": _render_public_referring_prediction(
                                image=runtime_record.image,
                                record_id=descriptor.record_id,
                                user_text=descriptor.user_text,
                                target=target.bbox_qwen_xyxy,
                                prediction=prediction,
                                output=public_visual_dir / visual_name,
                            ),
                        }
                    else:
                        row["normalized_exact_match"] = _normalize_generated_answer(
                            generated_text
                        ) == _normalize_generated_answer(descriptor.assistant_text)
                    public_retention_rows.append(row)
                    del generation_batch, runtime_record, supervised_batch
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    public_retention_elapsed = (
        None if public_retention_started is None else time.perf_counter() - public_retention_started
    )
    summaries = {
        partition: _partition_summary(
            [group for group in groups if group["partition"] == partition]
        )
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
        if any(group["partition"] == partition for group in groups)
    }
    semantic_label_summaries = {
        partition: _semantic_partition_summary(
            [group for group in groups if group["partition"] == partition]
        )
        for partition in FIXED_OBSERVATION_EVALUATION_PARTITIONS
        if any(group["partition"] == partition for group in groups)
    }
    if (
        adr127_runtime_python_trees_contract(
            repo_root=_ROOT,
            revision=args.picf_code_revision,
            source_checkout=args.source_checkout,
        )
        != runtime_python_trees
    ):
        raise ContractError("fixed-X native VL runtime source changed during execution")
    report = {
        "checkpoint_dir": str(args.checkpoint_dir.resolve()),
        "checkpoint_model_file_sha256": checkpoint_model_file_sha256,
        "crossed_exact_x_evaluation": (
            {"enabled": False}
            if crossed_plan is None
            else {
                "enabled": True,
                "elapsed_seconds": crossed_elapsed,
                "evaluation_plan_artifact_sha256": crossed_plan.artifact_sha256,
                "evaluation_plan_file_sha256": args.crossed_evaluation_plan_sha256,
                "results": crossed_rows,
                "summary": _partition_summary(crossed_rows),
                "unique_record_count": len(crossed_runtime_records),
            }
        ),
        "dataset_manifest_sha256": manifest.tree_sha256,
        "elapsed_seconds": elapsed,
        "eligible_item_count": len(eligible),
        "evaluation_plan_artifact_sha256": plan.artifact_sha256,
        "evaluation_plan_file_sha256": _sha256(args.evaluation_plan),
        "excluded_items": excluded,
        "item_limit_per_partition": args.item_limit_per_partition,
        "load_seconds": load_seconds,
        "max_new_tokens": args.max_new_tokens,
        "native_vl_patch_sha256": patch_report["native_vl_patch_sha256"],
        "partition": args.partition,
        "picf_code_revision": args.picf_code_revision,
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        "preload_tied_parameter_name": preload_tied_parameter_name,
        "processor_lattice": processor_lattice,
        "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
        "public_vl_retention": (
            {"enabled": False}
            if public_retention_manifest is None
            else {
                "artifact_root": str(args.public_vl_retention_root.resolve()),
                "artifact_sha256": public_retention_manifest.artifact_sha256,
                "enabled": True,
                "elapsed_seconds": public_retention_elapsed,
                "family_partition_counts": (public_retention_manifest.family_partition_counts),
                "heldout_limit_per_family": args.public_vl_heldout_limit_per_family,
                "manifest_file_sha256": args.public_vl_retention_manifest_sha256,
                "quality_exclusions": [
                    item.to_dict() for item in public_retention_manifest.quality_exclusions
                ],
                "processor": public_processor_contract,
                "results": public_retention_rows,
                "sources": {
                    key: public_retention_manifest.sources[key].to_dict()
                    for key in sorted(public_retention_manifest.sources)
                },
                "summaries": _public_retention_summary(public_retention_rows),
            }
        ),
        "qwen_restore": qwen_restore,
        "results": groups,
        "runtime_python_trees": runtime_python_trees,
        "schema": CROSSED_OUTPUT_SCHEMA if crossed_plan is not None else OUTPUT_SCHEMA,
        "scene_evaluation": (
            {"enabled": False}
            if scene_audit is None
            else {
                "audit_artifact_sha256": scene_audit["artifact_sha256"],
                "audit_file_sha256": args.scene_audit_report_sha256,
                "audit_report": str(args.scene_audit_report.resolve()),
                "enabled": True,
                "elapsed_seconds": scene_elapsed,
                "generation_budget": scene_generation_budget,
                "max_new_tokens": args.scene_max_new_tokens,
                "results": scene_rows,
                "source_disjoint_scene_bank_count": len(scene_pairs),
                "summary": native_vl_scene_bank_summary(scene_rows),
            }
        ),
        "seed": args.seed,
        "selected_item_count": len(selected),
        "semantic_label_summaries": semantic_label_summaries,
        "source_commit": commit,
        "summaries": summaries,
        "teacher_prune": teacher_prune,
        "tied_parameter_name": tied_parameter_name,
    }
    write_text_durable_exclusive(
        partial / "report.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    os.replace(partial, args.output_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

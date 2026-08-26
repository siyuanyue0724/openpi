#!/usr/bin/env python3
"""Audit and visualize the complete ADR-127 CALVIN scene curriculum."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import cast

import numpy as np
from PIL import Image, ImageDraw

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot native VL scene curriculum audit",
)

from picf_next.artifact_io import write_bytes_durable_exclusive  # noqa: E402
from picf_next.contracts import ContractError  # noqa: E402
from picf_next.data.calvin import CalvinDatasetIndex  # noqa: E402
from picf_next.data.calvin_physical_supervision_sidecar import (  # noqa: E402
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_qwen_grounding import (  # noqa: E402
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenSceneGroundingRecord,
    CalvinQwenSceneObject,
    build_calvin_qwen_scene_grounding_record,
    qwen_grounding_label,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.runtime_provenance import (  # noqa: E402
    revision_bound_python_source_tree_contract,
)
from picf_next.lingbot_native.vl_cotraining import (  # noqa: E402
    build_counterfactual_scene_grounding_records,
    materialize_fixed_observation_native_vl_records,
)
from picf_next.lingbot_native.vl_curriculum import (  # noqa: E402
    NativeVLGroundingCurriculumPlan,
)

SCHEMA = "picf-next.native-vl-scene-curriculum-audit.v2"
VISUAL_LATTICE = 8
ADR127_ARM_STEPS = 64
ADR127_ARM_UNIQUE_GROUPS = 32
SCENE_EVALUATION_BANK_SIZE = 32
_PANEL_SIZE = (540, 400)
_PANELS_PER_PAGE = 16
_CAMERA_NAMES = ("static", "gripper")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_CONTACT_SHEET_PREFIX_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")

_COLORS = {
    identity_key: color
    for identity_key, color in zip(
        CALVIN_QWEN_SCENE_IDENTITY_ORDER,
        (
            "#0072B2",
            "#CC79A7",
            "#D55E00",
            "#000000",
            "#009E73",
            "#E69F00",
            "#F0E442",
            "#56B4E9",
            "#6A3D9A",
            "#8C564B",
        ),
        strict=True,
    )
}


def _canonical_bytes(value: object) -> bytes:
    def validate(item: object) -> None:
        if item is None or type(item) in {bool, int, str}:
            return
        if type(item) is float:
            if not math.isfinite(item):
                raise ValueError("scene curriculum audit contains a non-finite float")
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                validate(child)
            return
        if isinstance(item, dict):
            if any(type(key) is not str for key in item):
                raise ValueError("scene curriculum audit object keys must be strings")
            for child in item.values():
                validate(child)
            return
        raise ValueError(f"scene curriculum audit contains unsupported JSON type {type(item)!r}")

    try:
        validate(value)
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError) as error:
        raise ValueError("scene curriculum audit is not canonical JSON") from error


def _artifact_payload(content: dict[str, object]) -> tuple[str, bytes]:
    if "artifact_sha256" in content:
        raise ValueError("unsigned scene curriculum content already contains artifact_sha256")
    artifact_sha256 = hashlib.sha256(_canonical_bytes(content)).hexdigest()
    payload = _canonical_bytes({**content, "artifact_sha256": artifact_sha256}) + b"\n"
    return artifact_sha256, payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: object, *, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} SHA-256 is invalid")
    return value


def _verified_sha256_file(path: Path, expected: str, *, name: str) -> str:
    expected = _require_sha256(expected, name=f"{name} expected")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{name} must be one real file") from error
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{name} must be one real file")
        while chunk := os.read(descriptor, 1024 * 1024):
            digest.update(chunk)
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(f"{name} SHA-256 changed")
    return observed


def _validated_checkout_revision(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise ValueError("scene curriculum audit requires a clean revision-bound checkout")
    return revision


def _bbox_intersection_area(
    left: tuple[int, int, int, int],
    right: tuple[int, int, int, int],
) -> int:
    width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    return width * height


def _object_payload(item: CalvinQwenSceneObject) -> dict[str, object]:
    return {
        "bbox_xyxy": list(item.bbox_xyxy),
        "identity_key": item.identity_key,
        "label": qwen_grounding_label(item.identity_key),
        "positive_visual_token_count": item.positive_visual_token_count,
        "projected_target_mass": item.projected_target_mass,
        "visible_owner_pixels": item.visible_owner_pixels,
    }


def _scene_evidence_map(
    record: CalvinQwenSceneGroundingRecord,
) -> dict[str, tuple[tuple[int, int, int, int], int, float, int, str]]:
    output = {
        item.identity_key: (
            item.bbox_xyxy,
            item.visible_owner_pixels,
            item.projected_target_mass,
            item.positive_visual_token_count,
            "object",
        )
        for item in record.objects
    }
    output.update(
        {
            item.identity_key: (
                item.bbox_xyxy,
                item.visible_owner_pixels,
                item.projected_target_mass,
                item.positive_visual_token_count,
                "subpatch",
            )
            for item in record.subpatch_objects
        }
    )
    return output


def _validate_scene_pair(
    canonical: CalvinQwenSceneGroundingRecord,
    reverse: CalvinQwenSceneGroundingRecord,
) -> None:
    if (
        canonical.global_index != reverse.global_index
        or canonical.camera_name != reverse.camera_name
        or canonical.source_rgb_sha256 != reverse.source_rgb_sha256
        or not np.array_equal(canonical.image, reverse.image)
        or canonical.category_identity_order != CALVIN_QWEN_SCENE_IDENTITY_ORDER
        or reverse.category_identity_order != tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
        or _scene_evidence_map(canonical) != _scene_evidence_map(reverse)
        or set(canonical.absent_identity_keys) != set(reverse.absent_identity_keys)
    ):
        raise ContractError("scene curriculum category-order counterfactual changed evidence")
    if tuple(item.identity_key for item in canonical.objects) != tuple(
        reversed(tuple(item.identity_key for item in reverse.objects))
    ):
        raise ContractError("scene curriculum answer did not reverse its visible object order")
    forbidden = (*CALVIN_QWEN_SCENE_IDENTITY_ORDER, str(canonical.global_index))
    for record in (canonical, reverse):
        request = record.grounding_request
        if any(value in request for value in forbidden):
            raise ContractError("scene curriculum request contains privileged metadata")
        if not all(
            qwen_grounding_label(key) in request for key in CALVIN_QWEN_SCENE_IDENTITY_ORDER
        ):
            raise ContractError("scene curriculum request omits a global natural category")


def _overlap_payload(record: CalvinQwenSceneGroundingRecord) -> list[dict[str, object]]:
    visible = (*record.objects, *record.subpatch_objects)
    overlaps = []
    for index, left in enumerate(visible):
        for right in visible[index + 1 :]:
            area = _bbox_intersection_area(left.bbox_xyxy, right.bbox_xyxy)
            if area:
                overlaps.append(
                    {
                        "intersection_pixels": area,
                        "left_identity_key": left.identity_key,
                        "right_identity_key": right.identity_key,
                    }
                )
    return overlaps


def _scene_payload(
    canonical: CalvinQwenSceneGroundingRecord,
    reverse: CalvinQwenSceneGroundingRecord,
    *,
    group_index: int,
) -> dict[str, object]:
    return {
        "absent_identity_keys": list(canonical.absent_identity_keys),
        "camera_name": canonical.camera_name,
        "canonical_answer_sha256": hashlib.sha256(
            canonical.assistant_text.encode("utf-8")
        ).hexdigest(),
        "global_index": canonical.global_index,
        "group_index": group_index,
        "image_grid_thw": list(canonical.image_grid_thw),
        "minimum_projected_target_mass": canonical.minimum_projected_target_mass,
        "objects": [_object_payload(item) for item in canonical.objects],
        "overlaps": _overlap_payload(canonical),
        "reverse_answer_sha256": hashlib.sha256(reverse.assistant_text.encode("utf-8")).hexdigest(),
        "source_rgb_sha256": canonical.source_rgb_sha256,
        "subpatch_objects": [_object_payload(item) for item in canonical.subpatch_objects],
    }


def _draw_scene_panel(
    record: CalvinQwenSceneGroundingRecord,
    *,
    title_lines: tuple[str, ...],
) -> Image.Image:
    source = Image.fromarray(record.image)
    canvas = Image.new("RGB", _PANEL_SIZE, "white")
    resized = source.resize((320, 320), Image.Resampling.NEAREST)
    canvas.paste(resized, (20, 72))
    draw = ImageDraw.Draw(canvas)
    scale_x = 320 / source.width
    scale_y = 320 / source.height
    legend_items: list[tuple[str, CalvinQwenSceneObject]] = []
    for status, objects in (("object", record.objects), ("subpatch", record.subpatch_objects)):
        for item in objects:
            legend_items.append((status, item))
            color = _COLORS[item.identity_key]
            width = 3 if status == "object" else 1
            x0, y0, x1, y1 = item.bbox_xyxy
            box = (
                20 + round(x0 * scale_x),
                72 + round(y0 * scale_y),
                20 + round(x1 * scale_x) - 1,
                72 + round(y1 * scale_y) - 1,
            )
            draw.rectangle(box, outline="white", width=width + 2)
            draw.rectangle(box, outline=color, width=width)
            marker = str(len(legend_items))
            marker_box = draw.textbbox((box[0] + 2, box[1] + 2), marker)
            draw.rectangle(marker_box, fill="white", outline="black")
            draw.text((box[0] + 2, box[1] + 2), marker, fill="black")
    for legend_index, (status, item) in enumerate(legend_items, start=1):
        legend_y = 74 + (legend_index - 1) * 30
        color = _COLORS[item.identity_key]
        draw.rectangle((352, legend_y + 2, 362, legend_y + 12), fill=color, outline="black")
        draw.text(
            (368, legend_y),
            f"{legend_index} {qwen_grounding_label(item.identity_key)}"[:27],
            fill="black",
        )
        draw.text(
            (352, legend_y + 14),
            f"{status} mass={item.projected_target_mass:.3f} "
            f"tokens={item.positive_visual_token_count}",
            fill="#303030",
        )
    for line_index, line in enumerate(title_lines[:4]):
        draw.text((8, 4 + 16 * line_index), line[:58], fill="black")
    return canvas


def _missing_panel(*, title_lines: tuple[str, ...]) -> Image.Image:
    canvas = Image.new("RGB", _PANEL_SIZE, "white")
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(title_lines[:4]):
        draw.text((8, 4 + 16 * line_index), line[:58], fill="black")
    draw.text((90, 190), "NO VISIBLE SOURCE EVIDENCE", fill="#D55E00")
    return canvas


def _write_png(path: Path, image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False)
    payload = stream.getvalue()
    write_bytes_durable_exclusive(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _write_contact_sheet_pages(
    directory: Path,
    *,
    prefix: str,
    panels: list[Image.Image],
    columns: int = 4,
) -> list[dict[str, object]]:
    if not panels:
        raise ValueError("contact sheet requires at least one panel")
    if (
        not isinstance(prefix, str)
        or _CONTACT_SHEET_PREFIX_PATTERN.fullmatch(prefix) is None
        or prefix in {".", ".."}
    ):
        raise ValueError("contact sheet prefix is invalid")
    if (
        isinstance(columns, bool)
        or not isinstance(columns, int)
        or not 1 <= columns <= _PANELS_PER_PAGE
    ):
        raise ValueError("contact sheet columns are invalid")
    contact_sheet_directory = directory / "contact_sheets"
    if (
        directory.is_symlink()
        or not directory.is_dir()
        or contact_sheet_directory.is_symlink()
        or not contact_sheet_directory.is_dir()
    ):
        raise ValueError("contact sheet output directories must be real directories")
    if any(panel.mode != "RGB" or panel.size != _PANEL_SIZE for panel in panels):
        raise ValueError("contact sheet panels must have the canonical RGB panel shape")
    outputs = []
    for page_index, start in enumerate(range(0, len(panels), _PANELS_PER_PAGE)):
        page_panels = panels[start : start + _PANELS_PER_PAGE]
        rows = (len(page_panels) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (_PANEL_SIZE[0] * columns, _PANEL_SIZE[1] * rows),
            "#D0D0D0",
        )
        for panel_index, panel in enumerate(page_panels):
            x = (panel_index % columns) * _PANEL_SIZE[0]
            y = (panel_index // columns) * _PANEL_SIZE[1]
            sheet.paste(panel, (x, y))
        relative = Path("contact_sheets") / f"{prefix}_{page_index:02d}.png"
        digest = _write_png(directory / relative, sheet)
        outputs.append(
            {
                "panel_count": len(page_panels),
                "path": relative.as_posix(),
                "sha256": digest,
            }
        )
    return outputs


def _select_source_disjoint_scene_bank(
    records: dict[tuple[int, str], CalvinQwenSceneGroundingRecord],
    *,
    excluded_group_indices: set[int],
    curriculum_artifact_sha256: str,
    bank_size: int = SCENE_EVALUATION_BANK_SIZE,
) -> list[tuple[int, CalvinQwenSceneGroundingRecord]]:
    if isinstance(bank_size, bool) or not isinstance(bank_size, int) or bank_size <= 0:
        raise ValueError("scene evaluation bank size must be positive")
    _require_sha256(curriculum_artifact_sha256, name="curriculum artifact")
    if not isinstance(excluded_group_indices, set) or any(
        isinstance(group_index, bool) or not isinstance(group_index, int) or group_index < 0
        for group_index in excluded_group_indices
    ):
        raise ValueError("excluded scene-bank group indices are invalid")
    if not isinstance(records, dict) or not records:
        raise ValueError("scene evaluation bank records must be a non-empty dictionary")

    group_to_global_index: dict[int, int] = {}
    for key, record in records.items():
        if (
            not isinstance(key, tuple)
            or len(key) != 2
            or isinstance(key[0], bool)
            or not isinstance(key[0], int)
            or key[0] < 0
            or key[1] not in _CAMERA_NAMES
        ):
            raise ContractError("scene evaluation bank record key is invalid")
        group_index, camera_name = key
        if not isinstance(record, CalvinQwenSceneGroundingRecord):
            raise ContractError("scene evaluation bank record has the wrong type")
        if record.camera_name != camera_name:
            raise ContractError("scene evaluation bank key disagrees with record camera")
        if record.category_identity_order != CALVIN_QWEN_SCENE_IDENTITY_ORDER:
            raise ContractError("scene evaluation bank requires canonical category order")
        _require_sha256(record.source_rgb_sha256, name="scene source RGB")
        previous_global_index = group_to_global_index.setdefault(group_index, record.global_index)
        if previous_global_index != record.global_index:
            raise ContractError("one scene group maps to multiple source frames")

    global_index_to_group: dict[int, int] = {}
    for group_index, global_index in group_to_global_index.items():
        previous_group = global_index_to_group.setdefault(global_index, group_index)
        if previous_group != group_index:
            raise ContractError("scene groups are not source-frame disjoint")
    if not excluded_group_indices.issubset(group_to_global_index):
        raise ContractError("scene evaluation bank exclusion references an unknown group")

    excluded_global_indices = {
        record.global_index
        for (group_index, _camera_name), record in records.items()
        if group_index in excluded_group_indices
    }
    excluded_rgb_sha256 = {
        record.source_rgb_sha256
        for (group_index, _camera_name), record in records.items()
        if group_index in excluded_group_indices
    }
    candidates = [
        (group_index, record)
        for (group_index, _camera_name), record in records.items()
        if group_index not in excluded_group_indices
        and record.global_index not in excluded_global_indices
        and record.source_rgb_sha256 not in excluded_rgb_sha256
    ]
    if len({record.global_index for _group_index, record in candidates}) < bank_size:
        raise ContractError("scene evaluation bank has too few source-disjoint groups")

    def digest(item: tuple[int, CalvinQwenSceneGroundingRecord]) -> str:
        group_index, record = item
        payload = (
            f"{curriculum_artifact_sha256}\0{group_index}\0{record.camera_name}\0"
            f"{record.source_rgb_sha256}"
        ).encode("ascii")
        return hashlib.sha256(b"picf-next.adr127-scene-bank.v1\0" + payload).hexdigest()

    candidates.sort(key=digest)
    available_units: set[tuple[str, ...]] = {
        ("identity_camera", item.identity_key, record.camera_name)
        for _group_index, record in candidates
        for item in record.objects
    }
    available_units.update(("camera", record.camera_name) for _group_index, record in candidates)
    units_by_candidate = {
        digest(candidate): {
            ("identity_camera", item.identity_key, candidate[1].camera_name)
            for item in candidate[1].objects
        }
        | {("camera", candidate[1].camera_name)}
        for candidate in candidates
    }
    if len(units_by_candidate) != len(candidates):
        raise ContractError("scene evaluation bank candidate digest collision")

    def eligible_candidates(
        selected_groups: frozenset[int],
        selected_global_indices: frozenset[int],
        selected_rgb_sha256: frozenset[str],
    ) -> list[tuple[int, CalvinQwenSceneGroundingRecord]]:
        return [
            item
            for item in candidates
            if item[0] not in selected_groups
            and item[1].global_index not in selected_global_indices
            and item[1].source_rgb_sha256 not in selected_rgb_sha256
        ]

    def cover(
        uncovered: frozenset[tuple[str, ...]],
        selected: tuple[tuple[int, CalvinQwenSceneGroundingRecord], ...],
        selected_groups: frozenset[int],
        selected_global_indices: frozenset[int],
        selected_rgb_sha256: frozenset[str],
    ) -> tuple[tuple[int, CalvinQwenSceneGroundingRecord], ...] | None:
        if not uncovered:
            return selected
        if len(selected) >= bank_size:
            return None
        eligible = eligible_candidates(
            selected_groups,
            selected_global_indices,
            selected_rgb_sha256,
        )
        candidate_counts = {
            unit: sum(unit in units_by_candidate[digest(item)] for item in eligible)
            for unit in uncovered
        }
        if not candidate_counts or min(candidate_counts.values()) == 0:
            return None
        pivot = min(uncovered, key=lambda unit: (candidate_counts[unit], unit))
        options = [item for item in eligible if pivot in units_by_candidate[digest(item)]]
        options.sort(
            key=lambda item: (
                -len(units_by_candidate[digest(item)] & uncovered),
                -len(_overlap_payload(item[1])),
                -len(item[1].objects),
                digest(item),
            )
        )
        for item in options:
            result = cover(
                uncovered - units_by_candidate[digest(item)],
                (*selected, item),
                selected_groups | {item[0]},
                selected_global_indices | {item[1].global_index},
                selected_rgb_sha256 | {item[1].source_rgb_sha256},
            )
            if result is not None:
                return result
        return None

    covered = cover(
        frozenset(available_units),
        (),
        frozenset(),
        frozenset(),
        frozenset(),
    )
    if covered is None:
        raise ContractError(
            "scene evaluation bank cannot jointly cover attainable cameras and identities"
        )
    selected = list(covered)
    selected_groups = {item[0] for item in selected}
    selected_global_indices = {item[1].global_index for item in selected}
    selected_rgb_sha256 = {item[1].source_rgb_sha256 for item in selected}
    while len(selected) < bank_size:
        eligible = eligible_candidates(
            frozenset(selected_groups),
            frozenset(selected_global_indices),
            frozenset(selected_rgb_sha256),
        )
        if not eligible:
            raise ContractError("scene evaluation bank exhausted source-disjoint candidates")
        best = min(
            eligible,
            key=lambda item: (
                -len(_overlap_payload(item[1])),
                -len(item[1].objects),
                digest(item),
            ),
        )
        selected.append(best)
        selected_groups.add(best[0])
        selected_global_indices.add(best[1].global_index)
        selected_rgb_sha256.add(best[1].source_rgb_sha256)
    if (
        len(selected_groups) != bank_size
        or len(selected_global_indices) != bank_size
        or len(selected_rgb_sha256) != bank_size
        or selected_groups & excluded_group_indices
        or selected_global_indices & excluded_global_indices
        or selected_rgb_sha256 & excluded_rgb_sha256
    ):
        raise ContractError("scene evaluation bank violated source-disjoint selection")
    return selected


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum-plan", type=Path, required=True)
    parser.add_argument("--curriculum-plan-sha256", required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--expected-group-count", type=int, default=216)
    parser.add_argument("--expected-step-count", type=int, default=432)
    parser.add_argument("--arm-step-count", type=int, default=ADR127_ARM_STEPS)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    checkout_revision = _validated_checkout_revision(repo_root)
    if args.picf_code_revision != checkout_revision:
        raise ValueError("scene curriculum checkout differs from declared PICF revision")
    runtime_python_tree = revision_bound_python_source_tree_contract(
        repo_root=repo_root,
        revision=checkout_revision,
        roots={"src": repo_root / "src", "tools": repo_root / "tools"},
    )
    curriculum_file_sha256 = _verified_sha256_file(
        args.curriculum_plan,
        args.curriculum_plan_sha256,
        name="scene curriculum",
    )
    plan = NativeVLGroundingCurriculumPlan.load(args.curriculum_plan)
    _verified_sha256_file(
        args.curriculum_plan,
        curriculum_file_sha256,
        name="scene curriculum after parsing",
    )
    if (
        args.expected_group_count != len(plan.groups)
        or args.expected_step_count != len(plan.steps)
        or args.arm_step_count != ADR127_ARM_STEPS
        or args.arm_step_count > len(plan.steps)
    ):
        raise ValueError("scene curriculum expected dimensions changed")

    dataset_manifest_file_sha256 = _file_sha256(args.dataset_manifest)
    _verified_sha256_file(
        args.dataset_manifest,
        dataset_manifest_file_sha256,
        name="dataset manifest before parsing",
    )
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    _verified_sha256_file(
        args.dataset_manifest,
        dataset_manifest_file_sha256,
        name="dataset manifest after parsing",
    )
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.name,
    )
    if (
        plan.dataset_id,
        plan.dataset_revision,
        plan.dataset_manifest_sha256,
    ) != (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256):
        raise ContractError("scene curriculum belongs to another dataset")
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root, index)

    output = args.output_dir.expanduser().absolute()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(output)
    partial.mkdir(parents=False)
    (partial / "contact_sheets").mkdir()

    scene_rows = []
    scene_by_group_camera: dict[tuple[int, str], CalvinQwenSceneGroundingRecord] = {}
    identity_examples: dict[tuple[str, str], tuple[float, CalvinQwenSceneGroundingRecord]] = {}
    overlap_examples: list[tuple[int, CalvinQwenSceneGroundingRecord]] = []
    status_histogram: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    camera_object_histogram: dict[str, Counter[int]] = defaultdict(Counter)
    for group_index, group in enumerate(plan.groups):
        arrays = dict(
            index.validated_source_frame_arrays(
                group.source_global_index,
                fields=("rgb_gripper", "rgb_static"),
            )
        )
        images = {
            "static": arrays["rgb_static"],
            "gripper": arrays["rgb_gripper"],
        }
        frame = sidecar.source_frame(group.source_global_index)
        if frame.identity_keys != CALVIN_QWEN_SCENE_IDENTITY_ORDER:
            raise ContractError("scene curriculum physical identity order changed")
        for camera_name, image in images.items():
            canonical = build_calvin_qwen_scene_grounding_record(
                global_index=group.source_global_index,
                camera_name=camera_name,
                image=image,
                physical_frame=frame,
                category_identity_order=CALVIN_QWEN_SCENE_IDENTITY_ORDER,
                visual_lattice=VISUAL_LATTICE,
            )
            reverse = build_calvin_qwen_scene_grounding_record(
                global_index=group.source_global_index,
                camera_name=camera_name,
                image=image,
                physical_frame=frame,
                category_identity_order=tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER)),
                visual_lattice=VISUAL_LATTICE,
            )
            _validate_scene_pair(canonical, reverse)
            row = _scene_payload(canonical, reverse, group_index=group_index)
            scene_rows.append(row)
            scene_by_group_camera[(group_index, camera_name)] = canonical
            camera_object_histogram[camera_name][len(canonical.objects)] += 1
            object_keys = {item.identity_key for item in canonical.objects}
            subpatch_keys = {item.identity_key for item in canonical.subpatch_objects}
            for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER:
                status = (
                    "object"
                    if identity_key in object_keys
                    else "subpatch"
                    if identity_key in subpatch_keys
                    else "absent"
                )
                status_histogram[(identity_key, camera_name)][status] += 1
            for item in (*canonical.objects, *canonical.subpatch_objects):
                key = (item.identity_key, camera_name)
                score = item.projected_target_mass
                if key not in identity_examples or score < identity_examples[key][0]:
                    identity_examples[key] = (score, canonical)
            overlap_score = sum(
                cast(int, item["intersection_pixels"]) for item in _overlap_payload(canonical)
            )
            if overlap_score:
                overlap_examples.append((overlap_score, canonical))

    arm_rows = []
    arm_panels = []
    arm_group_indices = []
    for step in plan.steps[: args.arm_step_count]:
        group, batches = plan.resolve_step(step.optimizer_step)
        visual_lattice, camera_name, variants = batches[0]
        if visual_lattice != VISUAL_LATTICE:
            raise ContractError("scene curriculum arm lattice changed")
        targets = materialize_fixed_observation_native_vl_records(
            index=index,
            sidecar=sidecar,
            group=group,
            variants=variants,
            expected_camera_name=camera_name,
        )
        scenes = build_counterfactual_scene_grounding_records(
            targets,
            sidecar.source_frame(group.source_global_index),
            visual_lattice=VISUAL_LATTICE,
        )
        _validate_scene_pair(*scenes)
        group_index = step.group_index
        exhaustive_scene = scene_by_group_camera[(group_index, camera_name)]
        if _scene_evidence_map(scenes[0]) != _scene_evidence_map(exhaustive_scene):
            raise ContractError("scene curriculum arm differs from exhaustive camera audit")
        task_keys = tuple(variant.task_key for variant in variants)
        arm_group_indices.append(group_index)
        arm_rows.append(
            {
                "camera_name": camera_name,
                "global_index": group.source_global_index,
                "group_index": group_index,
                "object_identity_keys": [item.identity_key for item in scenes[0].objects],
                "optimizer_step": step.optimizer_step,
                "rank_category_orders": [list(scene.category_identity_order) for scene in scenes],
                "rank_target_identity_keys": [target.target_identity_key for target in targets],
                "source_rgb_sha256": scenes[0].source_rgb_sha256,
                "task_keys": list(task_keys),
            }
        )
        arm_panels.append(
            _draw_scene_panel(
                scenes[0],
                title_lines=(
                    f"step={step.optimizer_step} group={group_index}",
                    f"global={group.source_global_index} camera={camera_name}",
                    task_keys[0],
                    task_keys[1],
                ),
            )
        )

    if len(arm_rows) != ADR127_ARM_STEPS or len(set(arm_group_indices)) != ADR127_ARM_UNIQUE_GROUPS:
        raise ContractError("scene curriculum arm no longer covers 64 steps and 32 source groups")

    identity_panels = []
    identity_coverage = []
    for identity_key in CALVIN_QWEN_SCENE_IDENTITY_ORDER:
        for camera_name in ("static", "gripper"):
            key = (identity_key, camera_name)
            counts = status_histogram[key]
            identity_coverage.append(
                {
                    "absent_count": counts["absent"],
                    "camera_name": camera_name,
                    "identity_key": identity_key,
                    "object_count": counts["object"],
                    "subpatch_count": counts["subpatch"],
                }
            )
            example = identity_examples.get(key)
            title = (
                qwen_grounding_label(identity_key),
                f"camera={camera_name}",
                f"object={counts['object']} subpatch={counts['subpatch']}",
                f"absent={counts['absent']}",
            )
            identity_panels.append(
                _missing_panel(title_lines=title)
                if example is None
                else _draw_scene_panel(example[1], title_lines=title)
            )

    overlap_examples.sort(key=lambda item: (-item[0], item[1].global_index, item[1].camera_name))
    overlap_panels = [
        _draw_scene_panel(
            record,
            title_lines=(
                f"overlap_pixels={score}",
                f"global={record.global_index} camera={record.camera_name}",
                "green-width=eligible; thin=subpatch",
            ),
        )
        for score, record in overlap_examples[:_PANELS_PER_PAGE]
    ]
    if not overlap_panels:
        overlap_panels = [_missing_panel(title_lines=("No overlapping tight boxes",))]

    scene_bank = _select_source_disjoint_scene_bank(
        scene_by_group_camera,
        excluded_group_indices=set(arm_group_indices),
        curriculum_artifact_sha256=plan.artifact_sha256,
    )
    scene_bank_rows = []
    scene_bank_panels = []
    arm_group_set = set(arm_group_indices)
    arm_global_indices = {
        record.global_index
        for (group_index, _camera_name), record in scene_by_group_camera.items()
        if group_index in arm_group_set
    }
    arm_rgb_sha256 = {
        record.source_rgb_sha256
        for (group_index, _camera_name), record in scene_by_group_camera.items()
        if group_index in arm_group_set
    }
    bank_group_indices = {group_index for group_index, _record in scene_bank}
    bank_global_indices = {record.global_index for _group_index, record in scene_bank}
    bank_rgb_sha256 = {record.source_rgb_sha256 for _group_index, record in scene_bank}
    if (
        len(scene_bank) != SCENE_EVALUATION_BANK_SIZE
        or len(bank_group_indices) != SCENE_EVALUATION_BANK_SIZE
        or len(bank_global_indices) != SCENE_EVALUATION_BANK_SIZE
        or len(bank_rgb_sha256) != SCENE_EVALUATION_BANK_SIZE
        or bank_group_indices & arm_group_set
        or bank_global_indices & arm_global_indices
        or bank_rgb_sha256 & arm_rgb_sha256
    ):
        raise ContractError("scene evaluation bank is not disjoint from the 64-step arm")
    row_by_group_camera = {
        (int(row["group_index"]), str(row["camera_name"])): row for row in scene_rows
    }
    for bank_index, (group_index, record) in enumerate(scene_bank):
        group = plan.groups[group_index]
        source_row = row_by_group_camera[(group_index, record.camera_name)]
        task_keys = [variant.task_key for variant in group.variants]
        scene_bank_rows.append(
            {
                "bank_index": bank_index,
                "camera_name": record.camera_name,
                "canonical_answer_sha256": source_row["canonical_answer_sha256"],
                "global_index": record.global_index,
                "group_index": group_index,
                "object_identity_keys": [item.identity_key for item in record.objects],
                "reverse_answer_sha256": source_row["reverse_answer_sha256"],
                "source_rgb_sha256": record.source_rgb_sha256,
                "task_keys": task_keys,
            }
        )
        scene_bank_panels.append(
            _draw_scene_panel(
                record,
                title_lines=(
                    f"scene-bank={bank_index} group={group_index}",
                    f"global={record.global_index} camera={record.camera_name}",
                    ", ".join(task_keys[:2]),
                    f"objects={len(record.objects)} subpatch={len(record.subpatch_objects)}",
                ),
            )
        )

    bank_camera_names = sorted({record.camera_name for _group_index, record in scene_bank})
    bank_identity_camera_pairs = sorted(
        {
            (item.identity_key, record.camera_name)
            for _group_index, record in scene_bank
            for item in record.objects
        }
    )
    attainable_identity_camera_pairs = sorted(
        {
            (item.identity_key, record.camera_name)
            for (group_index, _camera_name), record in scene_by_group_camera.items()
            if group_index not in arm_group_set
            and record.global_index not in arm_global_indices
            and record.source_rgb_sha256 not in arm_rgb_sha256
            for item in record.objects
        }
    )
    if (
        bank_camera_names != sorted(_CAMERA_NAMES)
        or bank_identity_camera_pairs != attainable_identity_camera_pairs
    ):
        raise ContractError("scene evaluation bank coverage changed after selection")

    sheets = {
        "adr127_arm": _write_contact_sheet_pages(
            partial,
            prefix="adr127_arm_steps",
            panels=arm_panels,
        ),
        "identity_camera_low_support": _write_contact_sheet_pages(
            partial,
            prefix="identity_camera_low_support",
            panels=identity_panels,
            columns=4,
        ),
        "overlap": _write_contact_sheet_pages(
            partial,
            prefix="overlap_examples",
            panels=overlap_panels,
        ),
        "source_disjoint_scene_bank": _write_contact_sheet_pages(
            partial,
            prefix="source_disjoint_scene_bank",
            panels=scene_bank_panels,
        ),
    }
    summary = {
        "arm_excluded_group_count": len(arm_group_set),
        "arm_step_count": len(arm_rows),
        "arm_unique_group_count": len(arm_group_set),
        "camera_object_count_histograms": {
            camera: {str(count): frequency for count, frequency in sorted(histogram.items())}
            for camera, histogram in sorted(camera_object_histogram.items())
        },
        "group_count": len(plan.groups),
        "identity_camera_coverage": identity_coverage,
        "overlap_scene_count": len(overlap_examples),
        "scene_view_count": len(scene_rows),
        "source_disjoint_scene_bank_camera_names": bank_camera_names,
        "source_disjoint_scene_bank_count": len(scene_bank_rows),
        "source_disjoint_scene_bank_identity_camera_pairs": [
            {"camera_name": camera_name, "identity_key": identity_key}
            for identity_key, camera_name in bank_identity_camera_pairs
        ],
        "source_disjoint_scene_bank_unique_global_count": len(bank_global_indices),
        "source_disjoint_scene_bank_unique_rgb_count": len(bank_rgb_sha256),
    }
    if _validated_checkout_revision(Path(__file__).resolve().parents[1]) != checkout_revision:
        raise ValueError("scene curriculum checkout revision changed during audit")
    if (
        revision_bound_python_source_tree_contract(
            repo_root=repo_root,
            revision=checkout_revision,
            roots={"src": repo_root / "src", "tools": repo_root / "tools"},
        )
        != runtime_python_tree
    ):
        raise ContractError("scene curriculum runtime source changed during audit")
    _verified_sha256_file(
        args.curriculum_plan,
        curriculum_file_sha256,
        name="scene curriculum before report signing",
    )
    _verified_sha256_file(
        args.dataset_manifest,
        dataset_manifest_file_sha256,
        name="dataset manifest before report signing",
    )
    content = {
        "arm_steps": arm_rows,
        "curriculum_artifact_sha256": plan.artifact_sha256,
        "curriculum_file_sha256": curriculum_file_sha256,
        "dataset_manifest_file_sha256": dataset_manifest_file_sha256,
        "dataset_tree_sha256": manifest.tree_sha256,
        "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
        "picf_code_revision": checkout_revision,
        "runtime_python_tree": runtime_python_tree,
        "scene_views": scene_rows,
        "source_disjoint_scene_bank": scene_bank_rows,
        "schema": SCHEMA,
        "status": "PASS",
        "summary": summary,
        "visual_lattice": VISUAL_LATTICE,
        "contact_sheets": sheets,
    }
    artifact_sha256, payload = _artifact_payload(content)
    write_bytes_durable_exclusive(partial / "report.json", payload)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "output_dir": str(output),
                **summary,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

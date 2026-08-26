#!/usr/bin/env python3
"""Render deterministic P/X/N/C review cells from one crossed curriculum."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import stat
import subprocess
import textwrap
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tools.repository_import import bind_entrypoint_to_own_repository
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot crossed-grounding visual review renderer",
)

from picf_next.artifact_io import write_bytes_durable_exclusive  # noqa: E402
from picf_next.contracts import ContractError  # noqa: E402
from picf_next.data.calvin import CalvinDatasetIndex  # noqa: E402
from picf_next.data.calvin_physical_supervision_schema import (  # noqa: E402
    source_array_sha256,
)
from picf_next.data.calvin_qwen_grounding import (  # noqa: E402
    qwen3vl_normalized_bbox,
    qwen_grounding_label,
)
from picf_next.data.dataset_manifest import (  # noqa: E402
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.crossed_causal_grounding import (  # noqa: E402
    CALVIN_GROUNDING_CAMERAS,
    CrossedVariantViewEvidence,
    boxes_are_mutually_centre_exclusive,
    build_crossed_partition_support_report,
    crossed_support_report_bytes,
    crossed_variant_views_are_source_disjoint,
    materialize_crossed_variant_views,
)
from picf_next.lingbot_native.runtime_provenance import (  # noqa: E402
    revision_bound_python_source_tree_contract,
)
from picf_next.lingbot_native.vl_curriculum import (  # noqa: E402
    NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
    NativeVLGroundingCurriculumPlan,
)

SCHEMA = "picf-next.crossed-grounding-visual-review.v1"
_CELL_KINDS = ("P", "X", "N", "C")
_CAMERA_EXTENTS = {"static": (200, 200), "gripper": (84, 84)}
_PANEL_SIZE = (640, 650)
_PANELS_PER_PAGE = 16


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
        raise ValueError("crossed visual-review evidence is not canonical finite JSON") from error


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


def _verified_file_sha256(
    path: Path,
    *,
    expected_sha256: str | None,
    name: str,
) -> str:
    source = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
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
    if expected_sha256 is not None and observed != _sha256(
        expected_sha256,
        name=f"{name} expected SHA-256",
    ):
        raise ValueError(f"{name} file SHA-256 changed")
    return observed


def _load_verified_json(
    path: Path,
    *,
    expected_sha256: str,
    maximum_bytes: int,
    name: str,
) -> tuple[Mapping[str, Any], str]:
    expected = _sha256(expected_sha256, name=f"{name} expected SHA-256")
    source = path.expanduser().absolute()
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ValueError(f"{name} must be one real file") from error
    payload = bytearray()
    digest = hashlib.sha256()
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum_bytes:
            raise ValueError(f"{name} file size is outside the supported contract")
        while chunk := os.read(descriptor, 1024 * 1024):
            payload.extend(chunk)
            digest.update(chunk)
    finally:
        os.close(descriptor)
    observed = digest.hexdigest()
    if observed != expected:
        raise ValueError(f"{name} file SHA-256 changed")

    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} repeats JSON key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            bytes(payload).decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                ValueError(f"{name} contains non-finite JSON constant {constant}")
            ),
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain one JSON object")
    return value, observed


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
        raise ValueError("crossed visual review requires a clean revision-bound checkout")
    return revision


def _box_area(box: tuple[int, int, int, int] | None) -> int:
    if box is None:
        return 0
    return (box[2] - box[0]) * (box[3] - box[1])


def _box_centre(box: tuple[int, int, int, int]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _row_key(row: CrossedVariantViewEvidence) -> tuple[object, ...]:
    return (
        row.group_index,
        row.global_index,
        row.camera_name,
        row.task_key,
        row.instruction_sha256,
        row.target_identity_key,
        row.state,
    )


@dataclass(frozen=True, slots=True)
class CrossedReviewCell:
    """One output-independent causal pair selected for original-image review."""

    kind: str
    first: CrossedVariantViewEvidence
    second: CrossedVariantViewEvidence

    def __post_init__(self) -> None:
        if self.kind not in _CELL_KINDS or _row_key(self.first) >= _row_key(self.second):
            raise ValueError("crossed review cell kind or row order is invalid")
        first, second = self.first, self.second
        first_box = first.bbox_qwen_xyxy
        second_box = second.bbox_qwen_xyxy
        if self.kind == "P":
            if (
                (first.group_index, first.global_index, first.camera_name, first.source_rgb_sha256)
                != (
                    second.group_index,
                    second.global_index,
                    second.camera_name,
                    second.source_rgb_sha256,
                )
                or first.state != "supervised"
                or second.state != "supervised"
                or first_box is None
                or second_box is None
                or first.task_key == second.task_key
                or first.instruction_sha256 == second.instruction_sha256
                or first.target_identity_key == second.target_identity_key
                or not boxes_are_mutually_centre_exclusive(first_box, second_box)
            ):
                raise ValueError("P review cell differs from the prompt-causal contract")
        elif self.kind in {"X", "N"}:
            if (
                (first.task_key, first.target_identity_key, first.camera_name)
                != (second.task_key, second.target_identity_key, second.camera_name)
                or first.instruction_sha256 != second.instruction_sha256
                or not crossed_variant_views_are_source_disjoint(first, second)
            ):
                raise ValueError(
                    f"{self.kind} review cell differs from the crossed-source contract"
                )
            states = {first.state, second.state}
            if self.kind == "X" and (
                states != {"supervised"}
                or first_box is None
                or second_box is None
                or not boxes_are_mutually_centre_exclusive(first_box, second_box)
            ):
                raise ValueError("X review cell differs from the pixel-causal contract")
            if self.kind == "N" and states != {"supervised", "absent"}:
                raise ValueError("N review cell differs from the visibility-causal contract")
        elif (
            (
                first.group_index,
                first.global_index,
                first.source_episode_index,
                first.source_state_sha256,
                first.task_key,
                first.instruction_sha256,
                first.target_identity_key,
            )
            != (
                second.group_index,
                second.global_index,
                second.source_episode_index,
                second.source_state_sha256,
                second.task_key,
                second.instruction_sha256,
                second.target_identity_key,
            )
            or {first.camera_name, second.camera_name} != set(CALVIN_GROUNDING_CAMERAS)
            or first.state != "supervised"
            or second.state != "supervised"
        ):
            raise ValueError("C review cell differs from the camera-transfer contract")

    @property
    def key(self) -> str:
        return _canonical_sha256(
            {
                "first": _row_key(self.first),
                "kind": self.kind,
                "second": _row_key(self.second),
            }
        )

    @property
    def task_keys(self) -> frozenset[str]:
        return frozenset((self.first.task_key, self.second.task_key))

    @property
    def target_identity_keys(self) -> frozenset[str]:
        return frozenset((self.first.target_identity_key, self.second.target_identity_key))


def _candidate_rank(cell: CrossedReviewCell) -> tuple[float, float, str]:
    boxes = tuple(
        box for box in (cell.first.bbox_qwen_xyxy, cell.second.bbox_qwen_xyxy) if box is not None
    )
    minimum_area = min(_box_area(box) for box in boxes)
    displacement = 0.0
    if len(boxes) == 2:
        displacement = math.dist(_box_centre(boxes[0]), _box_centre(boxes[1]))
    return (-float(minimum_area), -displacement, cell.key)


def _ordered_cell(
    kind: str, first: CrossedVariantViewEvidence, second: CrossedVariantViewEvidence
) -> CrossedReviewCell:
    if _row_key(first) > _row_key(second):
        first, second = second, first
    return CrossedReviewCell(kind=kind, first=first, second=second)


def _review_candidates(
    rows: Sequence[CrossedVariantViewEvidence],
) -> dict[str, tuple[CrossedReviewCell, ...]]:
    candidates: dict[str, list[CrossedReviewCell]] = {kind: [] for kind in _CELL_KINDS}

    prompt_strata: dict[tuple[int, str], list[CrossedVariantViewEvidence]] = defaultdict(list)
    source_strata: dict[tuple[str, str, str, str], list[CrossedVariantViewEvidence]] = defaultdict(
        list
    )
    camera_strata: dict[tuple[int, str, str, str], list[CrossedVariantViewEvidence]] = defaultdict(
        list
    )
    for row in rows:
        prompt_strata[(row.group_index, row.camera_name)].append(row)
        source_strata[
            (row.task_key, row.target_identity_key, row.camera_name, row.instruction_sha256)
        ].append(row)
        camera_strata[
            (row.group_index, row.task_key, row.instruction_sha256, row.target_identity_key)
        ].append(row)

    for stratum in prompt_strata.values():
        for first, second in combinations(stratum, 2):
            try:
                candidates["P"].append(_ordered_cell("P", first, second))
            except ValueError:
                continue
    for stratum in source_strata.values():
        for first, second in combinations(stratum, 2):
            for kind in ("X", "N"):
                try:
                    candidates[kind].append(_ordered_cell(kind, first, second))
                except ValueError:
                    continue
    for stratum in camera_strata.values():
        if len(stratum) != 2:
            continue
        try:
            candidates["C"].append(_ordered_cell("C", stratum[0], stratum[1]))
        except ValueError:
            continue
    return {
        kind: tuple(sorted(values, key=lambda cell: cell.key))
        for kind, values in candidates.items()
    }


def select_crossed_review_cells(
    rows: Sequence[CrossedVariantViewEvidence],
    *,
    expected_task_keys: Sequence[str],
    expected_target_identity_keys: Sequence[str],
) -> tuple[CrossedReviewCell, ...]:
    """Select a deterministic, geometry-only review cover for every causal cell kind."""

    tasks = tuple(sorted(expected_task_keys))
    targets = tuple(sorted(expected_target_identity_keys))
    if (
        not tasks
        or len(set(tasks)) != len(tasks)
        or not targets
        or len(set(targets)) != len(targets)
    ):
        raise ValueError("crossed review task and target contracts must be nonempty and unique")
    candidates = _review_candidates(rows)
    selected: dict[str, CrossedReviewCell] = {}
    for kind in _CELL_KINDS:
        values = candidates[kind]
        for task in tasks:
            eligible = tuple(cell for cell in values if task in cell.task_keys)
            if not eligible:
                raise ValueError(f"{kind} review cells do not cover task {task}")
            best = min(eligible, key=_candidate_rank)
            selected[best.key] = best
        for target in targets:
            eligible = tuple(cell for cell in values if target in cell.target_identity_keys)
            if not eligible:
                raise ValueError(f"{kind} review cells do not cover target {target}")
            best = min(eligible, key=_candidate_rank)
            selected[best.key] = best
        if kind in {"X", "N"}:
            target_cameras = sorted(
                {(cell.first.target_identity_key, cell.first.camera_name) for cell in values}
            )
            for target, camera in target_cameras:
                eligible = tuple(
                    cell
                    for cell in values
                    if (cell.first.target_identity_key, cell.first.camera_name) == (target, camera)
                )
                best = min(eligible, key=_candidate_rank)
                selected[best.key] = best
    return tuple(
        sorted(
            selected.values(),
            key=lambda cell: (
                _CELL_KINDS.index(cell.kind),
                sorted(cell.task_keys),
                sorted(cell.target_identity_keys),
                cell.key,
            ),
        )
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curriculum", required=True, type=Path)
    parser.add_argument("--curriculum-sha256", required=True)
    parser.add_argument("--expected-curriculum-artifact-sha256", required=True)
    parser.add_argument("--scene-audit", required=True, type=Path)
    parser.add_argument("--scene-audit-sha256", required=True)
    parser.add_argument("--support-report", required=True, type=Path)
    parser.add_argument("--support-report-sha256", required=True)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument(
        "--expected-task-key",
        action="append",
        dest="expected_task_keys",
        required=True,
    )
    parser.add_argument(
        "--target-identity",
        action="append",
        dest="target_identities",
        required=True,
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _write_png(path: Path, image: Image.Image) -> str:
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=False)
    payload = stream.getvalue()
    write_bytes_durable_exclusive(path, payload)
    return hashlib.sha256(payload).hexdigest()


def _write_contact_sheet_pages(
    directory: Path,
    *,
    kind: str,
    panels: Sequence[Image.Image],
    cell_ids: Sequence[str],
) -> list[dict[str, object]]:
    if kind not in _CELL_KINDS or not panels or len(panels) != 2 * len(cell_ids):
        raise ValueError("crossed review contact-sheet inputs are inconsistent")
    outputs = []
    for page_index, start in enumerate(range(0, len(panels), _PANELS_PER_PAGE)):
        page_panels = panels[start : start + _PANELS_PER_PAGE]
        columns = 4
        rows = (len(page_panels) + columns - 1) // columns
        sheet = Image.new(
            "RGB",
            (_PANEL_SIZE[0] * columns, _PANEL_SIZE[1] * rows),
            "#D0D0D0",
        )
        for index, panel in enumerate(page_panels):
            if panel.mode != "RGB" or panel.size != _PANEL_SIZE:
                raise ValueError("crossed review panel shape changed")
            sheet.paste(
                panel, ((index % columns) * _PANEL_SIZE[0], (index // columns) * _PANEL_SIZE[1])
            )
        relative = Path("contact_sheets") / f"{kind.lower()}_cells_{page_index:02d}.png"
        digest = _write_png(directory / relative, sheet)
        cell_start = start // 2
        cell_end = (start + len(page_panels)) // 2
        outputs.append(
            {
                "cell_ids": list(cell_ids[cell_start:cell_end]),
                "panel_count": len(page_panels),
                "path": relative.as_posix(),
                "sha256": digest,
            }
        )
    return outputs


def _scene_raw_bbox_index(
    scene_audit: Mapping[str, Any],
) -> dict[tuple[int, str, str], tuple[int, int, int, int] | None]:
    views = scene_audit.get("scene_views")
    if not isinstance(views, list):
        raise ValueError("scene audit views are malformed")
    output: dict[tuple[int, str, str], tuple[int, int, int, int] | None] = {}
    for view in views:
        if not isinstance(view, Mapping):
            raise ValueError("scene audit view is malformed")
        group_index = view.get("group_index")
        camera = view.get("camera_name")
        if (
            isinstance(group_index, bool)
            or not isinstance(group_index, int)
            or camera not in _CAMERA_EXTENTS
        ):
            raise ValueError("scene audit view identity is malformed")
        for state, field in (
            ("supervised", "objects"),
            ("subpatch", "subpatch_objects"),
        ):
            rows = view.get(field)
            if not isinstance(rows, list):
                raise ValueError("scene audit object partition is malformed")
            for row in rows:
                if not isinstance(row, Mapping) or not isinstance(row.get("identity_key"), str):
                    raise ValueError("scene audit object row is malformed")
                bbox = row.get("bbox_xyxy")
                if (
                    not isinstance(bbox, list)
                    or len(bbox) != 4
                    or any(isinstance(value, bool) or not isinstance(value, int) for value in bbox)
                ):
                    raise ValueError("scene audit object bbox is malformed")
                key = (group_index, camera, row["identity_key"])
                if key in output:
                    raise ValueError("scene audit repeats one target partition")
                output[key] = tuple(bbox) if state == "supervised" else None
        absent = view.get("absent_identity_keys")
        if not isinstance(absent, list) or any(not isinstance(value, str) for value in absent):
            raise ValueError("scene audit absent partition is malformed")
        for identity in absent:
            key = (group_index, camera, identity)
            if key in output:
                raise ValueError("scene audit target partitions overlap")
            output[key] = None
    return output


def _draw_panel(
    image: np.ndarray,
    *,
    cell_id: str,
    kind: str,
    side: str,
    row: CrossedVariantViewEvidence,
    instruction: str,
    raw_bbox: tuple[int, int, int, int] | None,
) -> Image.Image:
    panel = Image.new("RGB", _PANEL_SIZE, "white")
    draw = ImageDraw.Draw(panel)
    title_font = _font(16)
    body_font = _font(14)
    source = Image.fromarray(image)
    image_size = 460
    image_xy = (90, 104)
    panel.paste(source.resize((image_size, image_size), Image.Resampling.NEAREST), image_xy)
    draw.text(
        (8, 6),
        f"{kind} cell={cell_id[:12]} side={side} camera={row.camera_name}",
        fill="black",
        font=title_font,
    )
    draw.text((8, 29), f"task={row.task_key}", fill="black", font=body_font)
    draw.text(
        (8, 49),
        f"target={row.target_identity_key} state={row.state}",
        fill="black",
        font=body_font,
    )
    draw.text(
        (8, 69),
        f"global={row.global_index} group={row.group_index}",
        fill="black",
        font=body_font,
    )
    if raw_bbox is not None:
        width, height = _CAMERA_EXTENTS[row.camera_name]
        x0, y0, x1, y1 = raw_bbox
        scaled = (
            image_xy[0] + round(x0 * image_size / width),
            image_xy[1] + round(y0 * image_size / height),
            image_xy[0] + round(x1 * image_size / width) - 1,
            image_xy[1] + round(y1 * image_size / height) - 1,
        )
        draw.rectangle(scaled, outline="white", width=8)
        draw.rectangle(scaled, outline="#D00000", width=5)
        label = qwen_grounding_label(row.target_identity_key)
        label_box = draw.textbbox((scaled[0] + 4, scaled[1] + 4), label, font=body_font)
        draw.rectangle(label_box, fill="white", outline="#D00000", width=2)
        draw.text((scaled[0] + 4, scaled[1] + 4), label, fill="#A00000", font=body_font)
    else:
        message = "TRUE ABSENT: no target bbox"
        message_box = draw.textbbox((0, 0), message, font=title_font)
        x = (_PANEL_SIZE[0] - (message_box[2] - message_box[0])) // 2
        y = image_xy[1] + image_size // 2
        draw.rectangle(
            (x - 8, y - 6, x + message_box[2] + 8, y + 24), fill="white", outline="#D00000", width=3
        )
        draw.text((x, y), message, fill="#A00000", font=title_font)
    prompt_lines = textwrap.wrap(f"prompt: {instruction}", width=72)[:3]
    for line_index, line in enumerate(prompt_lines):
        draw.text((8, 574 + 20 * line_index), line, fill="black", font=body_font)
    return panel


def _row_payload(
    row: CrossedVariantViewEvidence,
    *,
    instruction: str,
    raw_bbox: tuple[int, int, int, int] | None,
) -> dict[str, object]:
    return {
        "bbox_qwen_xyxy": list(row.bbox_qwen_xyxy) if row.bbox_qwen_xyxy is not None else None,
        "bbox_raw_xyxy": list(raw_bbox) if raw_bbox is not None else None,
        "camera_name": row.camera_name,
        "global_index": row.global_index,
        "group_index": row.group_index,
        "instruction": instruction,
        "instruction_sha256": row.instruction_sha256,
        "source_episode_index": row.source_episode_index,
        "source_rgb_sha256": row.source_rgb_sha256,
        "source_state_sha256": row.source_state_sha256,
        "state": row.state,
        "target_identity_key": row.target_identity_key,
        "task_key": row.task_key,
    }


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    revision = _validated_checkout_revision(repo_root)
    runtime_python_tree = revision_bound_python_source_tree_contract(
        repo_root=repo_root,
        revision=revision,
        roots={"src": repo_root / "src", "tools": repo_root / "tools"},
    )
    curriculum_json, curriculum_file_sha256 = _load_verified_json(
        args.curriculum,
        expected_sha256=args.curriculum_sha256,
        maximum_bytes=NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
        name="native VL curriculum",
    )
    plan = NativeVLGroundingCurriculumPlan.from_dict(curriculum_json)
    expected_curriculum_artifact = _sha256(
        args.expected_curriculum_artifact_sha256,
        name="expected curriculum artifact SHA-256",
    )
    if plan.artifact_sha256 != expected_curriculum_artifact:
        raise ValueError("native VL curriculum artifact SHA-256 changed")
    scene_audit, scene_audit_file_sha256 = _load_verified_json(
        args.scene_audit,
        expected_sha256=args.scene_audit_sha256,
        maximum_bytes=64 * 1024 * 1024,
        name="scene audit",
    )
    support_report, support_report_file_sha256 = _load_verified_json(
        args.support_report,
        expected_sha256=args.support_report_sha256,
        maximum_bytes=64 * 1024 * 1024,
        name="crossed support report",
    )
    crossed_support_report_bytes(support_report)
    rebuilt_support = build_crossed_partition_support_report(
        plan.groups,
        scene_audit,
        curriculum_artifact_sha256=plan.artifact_sha256,
        curriculum_file_sha256=curriculum_file_sha256,
        scene_audit_file_sha256=scene_audit_file_sha256,
        expected_task_keys=args.expected_task_keys,
        expected_target_identity_keys=args.target_identities,
    )
    if dict(support_report) != rebuilt_support or support_report.get("status") != "PASS":
        raise ValueError("crossed support report differs from the verified inputs")

    manifest_path = args.dataset_manifest.expanduser().absolute()
    dataset_manifest_file_sha256 = _verified_file_sha256(
        manifest_path,
        expected_sha256=None,
        name="dataset manifest",
    )
    manifest = load_dataset_file_manifest(manifest_path)
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
        raise ContractError("crossed review curriculum belongs to another dataset")
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    joined = materialize_crossed_variant_views(
        plan.groups,
        scene_audit,
        expected_curriculum_artifact_sha256=plan.artifact_sha256,
    )
    cells = select_crossed_review_cells(
        joined,
        expected_task_keys=args.expected_task_keys,
        expected_target_identity_keys=args.target_identities,
    )
    raw_bbox_index = _scene_raw_bbox_index(scene_audit)
    instruction_index = {
        (
            group_index,
            variant.task_key,
            variant.instruction_sha256,
            variant.target_identity_key,
        ): variant.instruction
        for group_index, group in enumerate(plan.groups)
        for variant in group.variants
    }

    output = args.output_dir.expanduser().absolute()
    partial = output.with_name(f"{output.name}.partial")
    if output.exists() or output.is_symlink() or partial.exists() or partial.is_symlink():
        raise FileExistsError(output)
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise ValueError("crossed review output parent must be one real directory")
    partial.mkdir()
    (partial / "contact_sheets").mkdir()

    image_cache: dict[tuple[int, str], np.ndarray] = {}
    panels_by_kind: dict[str, list[Image.Image]] = {kind: [] for kind in _CELL_KINDS}
    cell_ids_by_kind: dict[str, list[str]] = {kind: [] for kind in _CELL_KINDS}
    cell_payloads = []
    for cell in cells:
        member_payloads = []
        for side, row in (("A", cell.first), ("B", cell.second)):
            image_key = (row.group_index, row.camera_name)
            if image_key not in image_cache:
                field = f"rgb_{row.camera_name}"
                arrays = dict(
                    index.validated_source_frame_arrays(row.global_index, fields=(field,))
                )
                image = np.asarray(arrays[field])
                if source_array_sha256(field, image) != row.source_rgb_sha256:
                    raise ContractError("crossed review source RGB differs from audited evidence")
                image_cache[image_key] = image
            image = image_cache[image_key]
            raw_bbox = raw_bbox_index[
                (
                    row.group_index,
                    row.camera_name,
                    row.target_identity_key,
                )
            ]
            if raw_bbox is not None:
                width, height = _CAMERA_EXTENTS[row.camera_name]
                if (
                    qwen3vl_normalized_bbox(raw_bbox, width=width, height=height)
                    != row.bbox_qwen_xyxy
                ):
                    raise ContractError("crossed review raw and normalized boxes disagree")
            elif row.bbox_qwen_xyxy is not None:
                raise ContractError("crossed review lost one supervised raw box")
            instruction = instruction_index[
                (
                    row.group_index,
                    row.task_key,
                    row.instruction_sha256,
                    row.target_identity_key,
                )
            ]
            panels_by_kind[cell.kind].append(
                _draw_panel(
                    image,
                    cell_id=cell.key,
                    kind=cell.kind,
                    side=side,
                    row=row,
                    instruction=instruction,
                    raw_bbox=raw_bbox,
                )
            )
            member_payloads.append(_row_payload(row, instruction=instruction, raw_bbox=raw_bbox))
        cell_ids_by_kind[cell.kind].append(cell.key)
        cell_payloads.append(
            {
                "cell_id": cell.key,
                "first": member_payloads[0],
                "kind": cell.kind,
                "second": member_payloads[1],
            }
        )

    sheets = {
        kind: _write_contact_sheet_pages(
            partial,
            kind=kind,
            panels=panels_by_kind[kind],
            cell_ids=cell_ids_by_kind[kind],
        )
        for kind in _CELL_KINDS
    }
    cell_count_by_kind = {kind: sum(cell.kind == kind for cell in cells) for kind in _CELL_KINDS}
    if _validated_checkout_revision(repo_root) != revision:
        raise ValueError("crossed visual-review checkout revision changed during rendering")
    if (
        revision_bound_python_source_tree_contract(
            repo_root=repo_root,
            revision=revision,
            roots={"src": repo_root / "src", "tools": repo_root / "tools"},
        )
        != runtime_python_tree
    ):
        raise ContractError("crossed visual-review runtime source changed during rendering")
    _load_verified_json(
        args.curriculum,
        expected_sha256=curriculum_file_sha256,
        maximum_bytes=NATIVE_VL_CURRICULUM_MAXIMUM_BYTES,
        name="native VL curriculum before report signing",
    )
    _load_verified_json(
        args.scene_audit,
        expected_sha256=scene_audit_file_sha256,
        maximum_bytes=64 * 1024 * 1024,
        name="scene audit before report signing",
    )
    _load_verified_json(
        args.support_report,
        expected_sha256=support_report_file_sha256,
        maximum_bytes=64 * 1024 * 1024,
        name="crossed support report before report signing",
    )
    _verified_file_sha256(
        manifest_path,
        expected_sha256=dataset_manifest_file_sha256,
        name="dataset manifest before report signing",
    )
    content: dict[str, object] = {
        "cells": cell_payloads,
        "contact_sheets": sheets,
        "crossed_support_artifact_sha256": support_report["artifact_sha256"],
        "crossed_support_file_sha256": support_report_file_sha256,
        "curriculum_artifact_sha256": plan.artifact_sha256,
        "curriculum_file_sha256": curriculum_file_sha256,
        "dataset_manifest_file_sha256": dataset_manifest_file_sha256,
        "dataset_tree_sha256": manifest.tree_sha256,
        "manual_review_status": "PENDING",
        "picf_code_revision": revision,
        "runtime_python_tree": runtime_python_tree,
        "scene_audit_artifact_sha256": scene_audit["artifact_sha256"],
        "scene_audit_file_sha256": scene_audit_file_sha256,
        "schema": SCHEMA,
        "status": "PASS",
        "summary": {
            "cell_count": len(cells),
            "cell_count_by_kind": cell_count_by_kind,
            "panel_count": 2 * len(cells),
            "target_identity_keys": sorted(args.target_identities),
            "task_keys": sorted(args.expected_task_keys),
        },
        "training_authorized": False,
    }
    artifact_sha256 = _canonical_sha256(content)
    payload = _canonical_bytes({**content, "artifact_sha256": artifact_sha256}) + b"\n"
    write_bytes_durable_exclusive(partial / "report.json", payload)
    os.replace(partial, output)
    print(
        json.dumps(
            {
                "artifact_sha256": artifact_sha256,
                "cell_count_by_kind": cell_count_by_kind,
                "file_sha256": hashlib.sha256(payload).hexdigest(),
                "output_dir": str(output),
                "status": "PASS",
                "training_authorized": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

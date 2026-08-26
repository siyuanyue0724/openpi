#!/usr/bin/env python3
"""Build immutable same-renderer CALVIN factual/object-removed probes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_counterfactual_plan import (
    CalvinCounterfactualPairPlan,
    CalvinCounterfactualPairRequest,
    load_calvin_counterfactual_pair_plan,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CALIBRATION_LIMITS,
    CALVIN_CAMERA_SPECS,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_simulator_geometry import (
    CalvinObjectRemovalPair,
    build_calvin_geometry_environment,
    build_calvin_object_removal_pair,
    close_calvin_geometry_environment,
    load_calvin_scene_ranges,
    scene_for_global_index,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_files,
)

MANUAL_SCHEMA = "picf-next.calvin-object-removal-probe.v1"
PLANNED_SCHEMA = "picf-next.calvin-object-removal-bank.v2"


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    global_index: int
    target_identity_key: str


def _parse_probe(value: str) -> ProbeRequest:
    frame, separator, identity = value.partition(":")
    if not separator or not frame.isdigit() or not identity:
        raise argparse.ArgumentTypeError("probe must be GLOBAL_INDEX:IDENTITY_KEY")
    global_index = int(frame)
    if global_index < 0 or any(character.isspace() for character in identity):
        raise argparse.ArgumentTypeError("probe frame or identity is invalid")
    return ProbeRequest(global_index=global_index, target_identity_key=identity)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    request = parser.add_mutually_exclusive_group(required=True)
    request.add_argument(
        "--probe",
        action="append",
        type=_parse_probe,
        help="Repeatable GLOBAL_INDEX:IDENTITY_KEY request.",
    )
    request.add_argument("--plan", type=Path)
    parser.add_argument("--plan-sha256")
    parser.add_argument("--source-sidecar-root", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_records(index: CalvinDatasetIndex, global_index: int) -> list[dict[str, object]]:
    records = [
        {
            "instruction": segment.instruction,
            "segment_end": segment.end,
            "segment_index": segment.index,
            "segment_start": segment.start,
            "task_key": segment.task_key,
        }
        for segment in index.segments
        if segment.start <= global_index <= segment.end
    ]
    if not records:
        raise ContractError(f"CALVIN probe frame {global_index} has no language annotation")
    return records


def _planned_requests(
    args: argparse.Namespace,
) -> tuple[
    tuple[ProbeRequest, ...],
    CalvinCounterfactualPairPlan | None,
    dict[tuple[int, str], CalvinCounterfactualPairRequest],
]:
    if args.plan is None:
        if args.plan_sha256 is not None or args.source_sidecar_root is not None:
            raise ValueError("plan hash and source sidecar require --plan")
        requests = tuple(args.probe or ())
        return requests, None, {}
    if args.probe is not None:
        raise ValueError("manual probes and a pair plan cannot be combined")
    if args.plan_sha256 is None or args.source_sidecar_root is None:
        raise ValueError("planned pair generation requires plan hash and source sidecar")
    plan = load_calvin_counterfactual_pair_plan(
        args.plan,
        expected_sha256=args.plan_sha256,
    )
    by_key = {request.key: request for request in plan.requests}
    requests = tuple(
        ProbeRequest(
            global_index=request.global_index,
            target_identity_key=request.target_identity_key,
        )
        for request in plan.requests
    )
    return requests, plan, by_key


def _verify_planned_request(
    request: CalvinCounterfactualPairRequest,
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
) -> None:
    if not 0 <= request.source_segment_index < len(index.segments):
        raise ContractError("planned removal request references an unknown segment")
    segment = index.segments[request.source_segment_index]
    if (
        not segment.start <= request.global_index < segment.end
        or segment.task_key != request.task_key
        or segment.instruction != request.instruction
    ):
        raise ContractError("planned removal request source segment changed")
    physical = sidecar.source_frame(request.global_index)
    try:
        owner = physical.identity_keys.index(request.target_identity_key) + 1
    except ValueError as error:
        raise ContractError("planned removal target left the physical inventory") from error
    counts = {
        camera.camera_name: int(((camera.owner_index == owner) & camera.owner_supervised).sum())
        for camera in physical.cameras
    }
    if counts != {
        "static": request.static_visible_pixels,
        "gripper": request.gripper_visible_pixels,
    }:
        raise ContractError("planned removal visibility support changed")


def _slug(request: ProbeRequest, tasks: list[dict[str, object]]) -> str:
    task_keys = sorted({str(task["task_key"]) for task in tasks})
    target = request.target_identity_key.replace("/", "-")
    return f"frame{request.global_index:07d}_{'-'.join(task_keys)}_{target}"


def _calibration_record(
    *,
    source_rgb: np.ndarray,
    source_depth: np.ndarray,
    factual_rgb: np.ndarray,
    factual_depth: np.ndarray,
) -> dict[str, float]:
    rgb_delta = np.abs(factual_rgb.astype(np.float32) - source_rgb.astype(np.float32))
    depth_delta = np.abs(factual_depth.astype(np.float32) - source_depth.astype(np.float32))
    record = {
        "depth_mae_m": float(depth_delta.mean()),
        "depth_p95_m": float(np.quantile(depth_delta, 0.95)),
        "rgb_mae": float(rgb_delta.mean()),
    }
    if (
        record["rgb_mae"] > CALVIN_CALIBRATION_LIMITS["maximum_rgb_mae"]
        or record["depth_mae_m"] > CALVIN_CALIBRATION_LIMITS["maximum_depth_mean_absolute_error_m"]
        or record["depth_p95_m"] > CALVIN_CALIBRATION_LIMITS["maximum_depth_p95_absolute_error_m"]
    ):
        raise ContractError("CALVIN object-removal factual render failed source calibration")
    return record


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    interior = (
        padded[1:-1, 1:-1]
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return mask & ~interior


def _target_overlay(rgb: np.ndarray, owner: np.ndarray, target_owner: int) -> np.ndarray:
    output = rgb.copy()
    target = owner == target_owner
    if np.any(target):
        fill = np.asarray((255, 32, 32), dtype=np.float32)
        output[target] = np.rint(0.55 * output[target] + 0.45 * fill).astype(np.uint8)
        output[_mask_boundary(target)] = np.asarray((0, 255, 255), dtype=np.uint8)
    return output


def _rgb_difference(factual: np.ndarray, removed: np.ndarray) -> np.ndarray:
    delta = np.abs(factual.astype(np.int16) - removed.astype(np.int16))
    return np.clip(delta * 4, 0, 255).astype(np.uint8)


def _resized_panel(value: np.ndarray, *, nearest: bool = False) -> Image.Image:
    image = Image.fromarray(value)
    return image.resize(
        (200, 200),
        Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR,
    )


def _write_contact_sheet(
    path: Path,
    *,
    request: ProbeRequest,
    tasks: list[dict[str, object]],
    pair: CalvinObjectRemovalPair,
    source_arrays: dict[str, np.ndarray],
) -> None:
    columns = ("archived", "factual", "target owner", "removed", "|f-r| x4")
    title_height = 64
    label_height = 18
    width = len(columns) * 200
    height = title_height + len(pair.cameras) * (label_height + 200)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    task_text = " | ".join(f"{task['task_key']}: {task['instruction']}" for task in tasks)
    draw.text(
        (4, 4),
        f"frame={request.global_index} target={request.target_identity_key}",
        fill="black",
    )
    draw.text((4, 22), task_text[:170], fill="black")
    for column, name in enumerate(columns):
        draw.text((column * 200 + 4, 46), name, fill="black")

    spec_by_name = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    for row, camera in enumerate(pair.cameras):
        y = title_height + row * (label_height + 200)
        draw.text((4, y + 2), camera.camera_name, fill="black")
        spec = spec_by_name[camera.camera_name]
        source = np.asarray(source_arrays[str(spec["source_rgb_field"])])
        panels = (
            _resized_panel(source),
            _resized_panel(camera.factual.rgb),
            _resized_panel(
                _target_overlay(
                    camera.factual.rgb,
                    camera.factual.owner_index,
                    pair.target_owner_index,
                ),
                nearest=True,
            ),
            _resized_panel(camera.removed.rgb),
            _resized_panel(_rgb_difference(camera.factual.rgb, camera.removed.rgb), nearest=True),
        )
        for column, panel in enumerate(panels):
            canvas.paste(panel, (column * 200, y + label_height))
    canvas.save(path)


def _write_probe(
    output_dir: Path,
    *,
    request: ProbeRequest,
    tasks: list[dict[str, object]],
    pair: CalvinObjectRemovalPair,
    source_arrays: dict[str, np.ndarray],
    plan_request: CalvinCounterfactualPairRequest | None = None,
) -> dict[str, Any]:
    slug = _slug(request, tasks)
    arrays: dict[str, np.ndarray] = {}
    calibration: dict[str, dict[str, float]] = {}
    spec_by_name = {str(spec["camera_name"]): spec for spec in CALVIN_CAMERA_SPECS}
    for camera in pair.cameras:
        spec = spec_by_name[camera.camera_name]
        source_rgb = np.asarray(source_arrays[str(spec["source_rgb_field"])])
        source_depth = np.asarray(source_arrays[str(spec["source_depth_field"])])
        calibration[camera.camera_name] = _calibration_record(
            source_rgb=source_rgb,
            source_depth=source_depth,
            factual_rgb=camera.factual.rgb,
            factual_depth=camera.factual.depth_m,
        )
        prefix = camera.camera_name
        arrays.update(
            {
                f"{prefix}_archived_rgb": source_rgb,
                f"{prefix}_factual_depth_m": camera.factual.depth_m,
                f"{prefix}_factual_owner": camera.factual.owner_index,
                f"{prefix}_factual_rgb": camera.factual.rgb,
                f"{prefix}_removed_depth_m": camera.removed.depth_m,
                f"{prefix}_removed_owner": camera.removed.owner_index,
                f"{prefix}_removed_rgb": camera.removed.rgb,
            }
        )
        Image.fromarray(source_rgb).save(output_dir / f"{slug}_{prefix}_archived.png")
        Image.fromarray(camera.factual.rgb).save(output_dir / f"{slug}_{prefix}_factual.png")
        Image.fromarray(camera.removed.rgb).save(output_dir / f"{slug}_{prefix}_removed.png")
        Image.fromarray(
            _target_overlay(
                camera.factual.rgb,
                camera.factual.owner_index,
                pair.target_owner_index,
            ),
            mode="RGB",
        ).save(output_dir / f"{slug}_{prefix}_target-overlay.png")
        Image.fromarray(
            _rgb_difference(camera.factual.rgb, camera.removed.rgb),
            mode="RGB",
        ).save(output_dir / f"{slug}_{prefix}_difference-x4.png")

    archive_path = output_dir / f"{slug}.npz"
    np.savez_compressed(archive_path, **arrays)
    contact_sheet_path = output_dir / f"{slug}_contact-sheet.png"
    _write_contact_sheet(
        contact_sheet_path,
        request=request,
        tasks=tasks,
        pair=pair,
        source_arrays=source_arrays,
    )
    result: dict[str, Any] = {
        "array_archive": archive_path.name,
        "array_archive_sha256": _sha256(archive_path),
        "calibration": calibration,
        "contact_sheet": contact_sheet_path.name,
        "contact_sheet_sha256": _sha256(contact_sheet_path),
        "pair": pair.contract_dict(),
        "tasks": tasks,
    }
    if plan_request is not None:
        result["plan_request"] = plan_request.to_dict()
    return result


def main() -> None:
    args = _parse_args()
    requests, plan, planned_by_key = _planned_requests(args)
    if len(set(requests)) != len(requests):
        raise ValueError("CALVIN object-removal probe requests must be unique")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite probe evidence: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))

    environment: Any | None = None
    try:
        split_root = args.split_root.resolve()
        manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
        validate_dataset_files(
            manifest,
            split_root,
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            split_name=split_root.name,
            verify_hashes=True,
        )
        index = CalvinDatasetIndex.load(
            split_root,
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=True,
            dataset_manifest=manifest,
        )
        source_sidecar = None
        if plan is not None:
            if (
                plan.dataset_id != index.dataset_id
                or plan.dataset_revision != index.dataset_revision
                or plan.split_name != index.split_root.name
            ):
                raise ContractError("counterfactual pair plan dataset identity changed")
            source_sidecar_root = args.source_sidecar_root.resolve()
            if _sha256(source_sidecar_root / "manifest.json") != (
                plan.source_sidecar_manifest_sha256
            ):
                raise ContractError("counterfactual pair plan source sidecar changed")
            source_sidecar = CalvinPhysicalSupervisionSidecar(
                source_sidecar_root,
                index,
                verify_hashes=True,
                cache_shards=24,
            )
        scene_ranges = load_calvin_scene_ranges(split_root, dataset_manifest=manifest)
        records = []
        active_scene: str | None = None
        for request in requests:
            plan_request = planned_by_key.get((request.global_index, request.target_identity_key))
            if plan is not None:
                if plan_request is None or source_sidecar is None:
                    raise RuntimeError("planned removal request mapping is incomplete")
                _verify_planned_request(
                    plan_request,
                    index=index,
                    sidecar=source_sidecar,
                )
            scene = scene_for_global_index(scene_ranges, request.global_index)
            if plan_request is not None and scene != plan_request.scene:
                raise ContractError("planned removal request scene assignment changed")
            if scene != active_scene:
                if environment is not None:
                    close_calvin_geometry_environment(environment)
                    environment = None
                environment = build_calvin_geometry_environment(
                    args.calvin_env_root.resolve(),
                    scene=scene,
                    include_cameras=True,
                )
                active_scene = scene
            source_arrays = dict(
                index.validated_source_frame_arrays(
                    request.global_index,
                    fields=(
                        "depth_gripper",
                        "depth_static",
                        "rgb_gripper",
                        "rgb_static",
                        "robot_obs",
                        "scene_obs",
                    ),
                )
            )
            pair = build_calvin_object_removal_pair(
                environment,
                scene_obs=source_arrays["scene_obs"],
                robot_obs=source_arrays["robot_obs"],
                source_global_index=request.global_index,
                target_identity_key=request.target_identity_key,
            )
            records.append(
                _write_probe(
                    temporary_dir,
                    request=request,
                    tasks=_task_records(index, request.global_index),
                    pair=pair,
                    source_arrays=source_arrays,
                    plan_request=plan_request,
                )
            )
        if environment is not None:
            close_calvin_geometry_environment(environment)
            environment = None
        summary = {
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "probe_count": len(records),
            "probes": records,
            "schema": PLANNED_SCHEMA if plan is not None else MANUAL_SCHEMA,
        }
        if plan is not None:
            summary.update(
                {
                    "pair_plan": str(plan.path),
                    "pair_plan_sha256": plan.file_sha256,
                    "source_sidecar_manifest_sha256": (plan.source_sidecar_manifest_sha256),
                }
            )
        summary_path = temporary_dir / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_dir, output_dir)
        print(json.dumps({"output_dir": str(output_dir), **summary}, sort_keys=True))
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise
    finally:
        if environment is not None:
            close_calvin_geometry_environment(environment)


if __name__ == "__main__":
    main()

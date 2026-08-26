#!/usr/bin/env python3
"""Numerically and visually audit a CALVIN physical-geometry sidecar."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pybullet
from PIL import Image, ImageDraw

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import (
    CALVIN_GEOMETRY_SIDECAR_SCHEMA,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_STATE_RESTORATION,
    calvin_source_state_sha256,
    sha256_file,
)
from picf_next.data.calvin_simulator_geometry import (
    build_calvin_geometry_environment,
    calvin_segmentation_identity_map,
    close_calvin_geometry_environment,
    extract_robot_base_aabb_centres,
    load_calvin_scene_ranges,
    scene_for_global_index,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_files,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--sidecar-dir", required=True, type=Path)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--max-segments", type=int)
    return parser.parse_args()


def _load_sidecar_frames(root: Path, manifest: dict[str, Any]) -> dict[int, dict[str, Any]]:
    frames: dict[int, dict[str, Any]] = {}
    for metadata in manifest["shards"]:
        path = root / metadata["path"]
        if sha256_file(path) != metadata["sha256"]:
            raise ContractError(f"CALVIN geometry audit found corrupt shard {path.name}")
        with np.load(path, allow_pickle=False) as archive:
            indices = archive["global_indices"]
            hashes = archive["source_state_sha256"]
            offsets = archive["frame_offsets"]
            keys = archive["identity_keys"]
            geometry = archive["geometry"]
        for row, raw_index in enumerate(indices.tolist()):
            global_index = int(raw_index)
            if global_index in frames:
                raise ContractError("CALVIN geometry audit found duplicate frame")
            start, stop = int(offsets[row]), int(offsets[row + 1])
            frames[global_index] = {
                "source_state_sha256": str(hashes[row]),
                "identity_keys": tuple(str(key) for key in keys[start:stop].tolist()),
                "geometry": geometry[start:stop].copy(),
            }
    return frames


def _phase_indices(start: int, end: int) -> tuple[tuple[str, int], ...]:
    return (("start", start), ("mid", (start + end) // 2), ("end", end))


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")[:64] or "unknown"


def _project(camera: Any, point: tuple[float, float, float]) -> tuple[int, int]:
    x, y = camera.project(np.asarray((*point, 1.0)))
    return int(x), int(y)


def _nearest_visible_distance(mask: np.ndarray, x: int, y: int) -> float | None:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return None
    return float(np.sqrt(np.min((columns - x) ** 2 + (rows - y) ** 2)))


def _panel(
    reference: np.ndarray,
    *,
    title: str,
    points: tuple[dict[str, Any], ...],
) -> Image.Image:
    source = Image.fromarray(reference).convert("RGB")
    overlay = source.copy()
    draw = ImageDraw.Draw(overlay)
    for item in points:
        x, y = item["pixel"]
        status = item["status"]
        color = {
            "inside_visible_mask": (20, 230, 80),
            "visible_but_center_occluded_or_nonconvex": (255, 180, 20),
            "not_visible_or_offscreen": (160, 160, 160),
        }[status]
        if 0 <= x < reference.shape[1] and 0 <= y < reference.shape[0]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color, outline=(0, 0, 0))
            draw.text((x + 5, y - 7), item["key"].split("/")[-1], fill=color)
    header = 126
    canvas = Image.new("RGB", (2 * source.width, source.height + header), (8, 8, 8))
    canvas.paste(source, (0, header))
    canvas.paste(overlay, (source.width, header))
    canvas_draw = ImageDraw.Draw(canvas)
    canvas_draw.text((8, 6), title, fill=(255, 255, 255))
    canvas_draw.text(
        (8, 24),
        "green=center pixel owns object; amber=object visible but physical center is "
        "occluded/nonconvex; gray=not visible/offscreen",
        fill=(220, 220, 220),
    )
    legend = " | ".join(
        f"{item['key']}:{item['status']}:d={item['nearest_visible_distance_px']}" for item in points
    )
    for row, offset in enumerate(range(0, len(legend), 150)):
        canvas_draw.text((8, 43 + 14 * row), legend[offset : offset + 150], fill=(190, 190, 190))
        if row >= 4:
            break
    canvas_draw.text((3, header + 2), "archived RGB", fill=(255, 255, 255))
    canvas_draw.text((source.width + 3, header + 2), "physical AABB centers", fill=(255, 255, 255))
    return canvas


def main() -> None:
    args = _arguments()
    if args.max_segments is not None and args.max_segments <= 0:
        raise ValueError("max segments must be positive")
    split_root = args.split_root.resolve()
    sidecar_root = args.sidecar_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    validate_dataset_files(
        dataset_manifest,
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=split_root.name,
        verify_hashes=True,
    )
    manifest = json.loads((sidecar_root / "manifest.json").read_text())
    if (
        manifest.get("schema") != CALVIN_GEOMETRY_SIDECAR_SCHEMA
        or manifest.get("dataset_id") != args.dataset_id
        or manifest.get("dataset_revision") != args.dataset_revision
        or manifest.get("geometry_contract_sha256") != CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint
        or manifest.get("state_restoration") != CALVIN_STATE_RESTORATION
        or manifest.get("scene_info_sha256") != dataset_manifest.record_for("scene_info.npy").sha256
    ):
        raise ContractError("CALVIN geometry audit manifest identity differs from inputs")
    frames = _load_sidecar_frames(sidecar_root, manifest)
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        verify_files=True,
        dataset_manifest=dataset_manifest,
    )
    ranges = load_calvin_scene_ranges(split_root, dataset_manifest=dataset_manifest)
    environments: dict[str, Any] = {}
    records = []
    maximum_geometry_error = 0.0
    try:
        segments = index.segments[: args.max_segments]
        for segment in segments:
            for phase, global_index in _phase_indices(segment.start, segment.end):
                source = index.validated_source_frame_arrays(
                    global_index,
                    fields=("scene_obs", "robot_obs", "rgb_static"),
                )
                scene_obs = source["scene_obs"]
                robot_obs = source["robot_obs"]
                reference = source["rgb_static"]
                stored = frames.get(global_index)
                if stored is None:
                    raise ContractError(f"sidecar does not cover audited frame {global_index}")
                source_hash = calvin_source_state_sha256(scene_obs, robot_obs)
                if stored["source_state_sha256"] != source_hash:
                    raise ContractError(f"source-state hash mismatch at frame {global_index}")
                scene = scene_for_global_index(ranges, global_index)
                environment = environments.get(scene)
                if environment is None:
                    environment = build_calvin_geometry_environment(
                        args.calvin_env_root.resolve(),
                        scene=scene,
                        include_cameras=True,
                    )
                    environments[scene] = environment
                keys, regenerated = extract_robot_base_aabb_centres(
                    environment,
                    scene_obs=scene_obs,
                    robot_obs=robot_obs,
                )
                if keys != stored["identity_keys"]:
                    raise ContractError(f"physical identity order mismatch at frame {global_index}")
                geometry_error = float(np.max(np.abs(regenerated - stored["geometry"])))
                maximum_geometry_error = max(maximum_geometry_error, geometry_error)
                if geometry_error > 1e-6:
                    raise ContractError(f"physical geometry mismatch at frame {global_index}")

                camera = environment.cameras[0]
                raw = environment.p.getCameraImage(
                    width=camera.width,
                    height=camera.height,
                    viewMatrix=camera.viewMatrix,
                    projectionMatrix=camera.projectionMatrix,
                    flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                    physicsClientId=environment.cid,
                )
                segmentation = np.asarray(raw[4]).reshape(camera.height, camera.width)
                encoded_by_key = {
                    key: encoded
                    for encoded, key in calvin_segmentation_identity_map(
                        environment.get_info()["scene_info"]
                    ).items()
                }
                base_position, base_orientation = environment.p.getBasePositionAndOrientation(
                    int(environment.robot.robot_uid),
                    physicsClientId=environment.cid,
                )
                point_records = []
                for key, normalized in zip(keys, regenerated, strict=True):
                    local = CALVIN_OBJECT_GEOMETRY_CONTRACT.denormalize_values(
                        tuple(float(value) for value in normalized)
                    )
                    world, _ = environment.p.multiplyTransforms(
                        base_position,
                        base_orientation,
                        local,
                        (0.0, 0.0, 0.0, 1.0),
                    )
                    x, y = _project(camera, tuple(float(value) for value in world))
                    mask = segmentation == encoded_by_key[key]
                    inside = 0 <= x < camera.width and 0 <= y < camera.height and bool(mask[y, x])
                    nearest = _nearest_visible_distance(mask, x, y)
                    status = (
                        "inside_visible_mask"
                        if inside
                        else (
                            "visible_but_center_occluded_or_nonconvex"
                            if nearest is not None
                            else "not_visible_or_offscreen"
                        )
                    )
                    point_records.append(
                        {
                            "key": key,
                            "pixel": [x, y],
                            "status": status,
                            "nearest_visible_distance_px": (
                                None if nearest is None else round(nearest, 3)
                            ),
                        }
                    )
                title = (
                    f"split={split_root.name} segment={segment.index} phase={phase} "
                    f"step={global_index} task={segment.task_key} "
                    f"geometry_error={geometry_error:.2e}"
                )
                filename = (
                    f"segment{segment.index:03d}_{_slug(segment.task_key)}_"
                    f"{phase}_step{global_index:07d}.png"
                )
                panel_path = output_dir / filename
                _panel(reference, title=title, points=tuple(point_records)).save(panel_path)
                records.append(
                    {
                        "segment_index": segment.index,
                        "phase": phase,
                        "global_index": global_index,
                        "task": segment.task_key,
                        "source_state_sha256": source_hash,
                        "geometry_max_abs_error": geometry_error,
                        "panel": filename,
                        "panel_sha256": sha256_file(panel_path),
                        "points": point_records,
                    }
                )
    finally:
        for environment in environments.values():
            close_calvin_geometry_environment(environment)
    report = {
        "schema": "picf-next.calvin-geometry-visual-audit.v1",
        "sidecar_manifest_sha256": sha256_file(sidecar_root / "manifest.json"),
        "maximum_geometry_abs_error": maximum_geometry_error,
        "audited_frames": len(records),
        "records": records,
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: report[key] for key in report if key != "records"}, sort_keys=True))


if __name__ == "__main__":
    main()

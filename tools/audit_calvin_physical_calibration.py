#!/usr/bin/env python3
"""Render named CALVIN frames for physical-sidecar calibration diagnosis."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_calibration import (
    calvin_depth_consistent_supervision,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS,
    CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION,
)
from picf_next.data.calvin_simulator_geometry import (
    build_calvin_geometry_environment,
    close_calvin_geometry_environment,
    extract_robot_base_aabb_centres,
    load_calvin_scene_ranges,
    render_calvin_camera_ownership,
    scene_for_global_index,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--global-index", action="append", type=int)
    selection.add_argument("--all-source-frames", action="store_true")
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def _selected_indices(args: argparse.Namespace, index: CalvinDatasetIndex) -> tuple[int, ...]:
    if args.global_index is not None:
        return tuple(args.global_index)
    values = tuple(
        global_index
        for episode in index.episodes
        for global_index in range(episode.start, episode.end + 1)
    )
    start = len(values) * args.partition_index // args.partition_count
    stop = len(values) * (args.partition_index + 1) // args.partition_count
    return values[start:stop]


def _load_bound_index(
    args: argparse.Namespace,
) -> tuple[Path, DatasetFileManifest, CalvinDatasetIndex, dict[str, object]]:
    """Bind the manifest once; selected source reads verify their own bytes."""

    split_root = args.split_root.resolve()
    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    binding = validate_dataset_runtime_binding(
        manifest,
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=split_root.name,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        dataset_manifest=manifest,
        verify_files=False,
    )
    return split_root, manifest, index, binding


def _color(key: str) -> np.ndarray:
    raw = hashlib.blake2b(key.encode("utf-8"), digest_size=3).digest()
    return np.asarray([64 + value % 176 for value in raw], dtype=np.float32)


def _owner_overlay(
    rgb: np.ndarray,
    owner: np.ndarray,
    keys: tuple[str, ...],
) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    for index, key in enumerate(keys, start=1):
        mask = owner == index
        if mask.any():
            output[mask] = 0.4 * output[mask] + 0.6 * _color(key)
    return np.clip(output, 0, 255).astype(np.uint8)


def _depth_error_image(source: np.ndarray, rendered: np.ndarray) -> np.ndarray:
    error = np.abs(source.astype(np.float32) - rendered.astype(np.float32))
    scaled = np.clip(error / 0.05, 0.0, 1.0)
    return np.stack(
        (
            255.0 * scaled,
            255.0 * np.sqrt(scaled),
            255.0 * (1.0 - scaled),
        ),
        axis=-1,
    ).astype(np.uint8)


def _unknown_overlay(rgb: np.ndarray, supervised: np.ndarray) -> np.ndarray:
    output = rgb.astype(np.float32).copy()
    output[~supervised] = 0.35 * output[~supervised] + 0.65 * np.asarray([255.0, 32.0, 32.0])
    return np.clip(output, 0, 255).astype(np.uint8)


def _owner_interior(owner: np.ndarray) -> np.ndarray:
    """Select non-context pixels whose 3x3 neighbourhood has one owner."""

    if owner.ndim != 2 or owner.dtype != np.uint8:
        raise ValueError("owner raster must be a two-dimensional uint8 array")
    padded = np.pad(owner, 1, mode="constant", constant_values=0)
    interior = owner > 0
    for y_offset in range(3):
        for x_offset in range(3):
            interior &= (
                padded[
                    y_offset : y_offset + owner.shape[0],
                    x_offset : x_offset + owner.shape[1],
                ]
                == owner
            )
    return interior


def _panel(
    *,
    global_index: int,
    keys: tuple[str, ...],
    rows: list[tuple[str, tuple[np.ndarray, ...], dict[str, Any]]],
) -> Image.Image:
    tile = (280, 280)
    header = 92
    labels = (
        "source RGB",
        "rendered RGB",
        "raw owners on source",
        "depth abs error (0-5 cm)",
        "depth-inconsistent unknown",
        "filtered owners on source",
    )
    canvas = Image.new("RGB", (len(labels) * tile[0], header + len(rows) * tile[1]), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), f"CALVIN physical calibration | frame={global_index}", fill="black")
    draw.text((8, 26), "objects=" + " | ".join(keys), fill="black")
    for row_index, (camera, images, metrics) in enumerate(rows):
        y = header + row_index * tile[1]
        metric_text = (
            f"{camera} rgb_mae={metrics['rgb_mae']:.4f} "
            f"depth_mean={metrics['depth_mae_m']:.5f}m "
            f"depth_p95={metrics['depth_p95_m']:.5f}m"
        )
        draw.text((8, y - 17), metric_text, fill="black")
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            x = column * tile[0]
            resized = Image.fromarray(image).resize(tile, resample=Image.Resampling.NEAREST)
            canvas.paste(resized, (x, y))
            draw.rectangle((x, y, x + 210, y + 16), fill="black")
            draw.text((x + 3, y + 2), f"{camera}: {label}", fill="white")
    return canvas


def main() -> None:
    args = _parse_args()
    for name in ("partition_count", "progress_every"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index must lie inside partition count")
    if args.global_index is not None and any(index < 0 for index in args.global_index):
        raise ValueError("CALVIN global indices must be non-negative")
    if args.global_index is not None and (args.partition_count != 1 or args.partition_index != 0):
        raise ValueError("explicit frame selection cannot also be partitioned")
    split_root, manifest, index, dataset_runtime_binding = _load_bound_index(args)
    selected_indices = _selected_indices(args, index)
    if not selected_indices:
        raise ValueError("CALVIN calibration audit selected no frames")
    scene_ranges = load_calvin_scene_ranges(split_root, dataset_manifest=manifest)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    environments: dict[str, Any] = {}
    records = []
    try:
        for ordinal, global_index in enumerate(selected_indices, start=1):
            scene = scene_for_global_index(scene_ranges, global_index)
            environment = environments.get(scene)
            if environment is None:
                environment = build_calvin_geometry_environment(
                    args.calvin_env_root.resolve(),
                    scene=scene,
                    include_cameras=True,
                )
                environments[scene] = environment
            required = {
                "scene_obs",
                "robot_obs",
                *(str(spec["source_rgb_field"]) for spec in CALVIN_CAMERA_SPECS),
                *(str(spec["source_depth_field"]) for spec in CALVIN_CAMERA_SPECS),
            }
            frame = index.validated_source_frame_arrays(
                global_index,
                fields=tuple(sorted(required)),
            )
            keys, _geometry = extract_robot_base_aabb_centres(
                environment,
                scene_obs=frame["scene_obs"],
                robot_obs=frame["robot_obs"],
            )
            renders = render_calvin_camera_ownership(environment, identity_keys=keys)
            rows = []
            metrics_by_camera = {}
            for spec, render in zip(CALVIN_CAMERA_SPECS, renders, strict=True):
                camera = str(spec["camera_name"])
                source_rgb = np.asarray(frame[str(spec["source_rgb_field"])])
                source_depth = np.asarray(frame[str(spec["source_depth_field"])])
                rgb_error = np.abs(source_rgb.astype(np.float32) - render.rgb.astype(np.float32))
                depth_error = np.abs(
                    source_depth.astype(np.float32) - render.depth_m.astype(np.float32)
                )
                owner_interior = _owner_interior(render.owner_index)
                if not owner_interior.any():
                    raise RuntimeError(
                        f"frame {global_index}/{camera} has no visible-owner interior"
                    )
                owner_depth_error = depth_error[owner_interior]
                owner_metrics = {}
                signed_depth_delta = render.depth_m.astype(np.float32) - source_depth.astype(
                    np.float32
                )
                production_supervised = calvin_depth_consistent_supervision(
                    source_depth,
                    render.depth_m,
                )
                consistency_thresholds_m = (0.005, 0.01, 0.02)
                for owner_index, identity_key in enumerate(keys, start=1):
                    support = owner_interior & (render.owner_index == owner_index)
                    if support.any():
                        values = depth_error[support]
                        signed_values = signed_depth_delta[support]
                        owner_metrics[identity_key] = {
                            "pixel_count": int(support.sum()),
                            "depth_mae_m": float(values.mean()),
                            "depth_p95_m": float(np.quantile(values, 0.95)),
                            "depth_max_m": float(values.max()),
                            "signed_depth_mean_m": float(signed_values.mean()),
                            "signed_depth_median_m": float(np.median(signed_values)),
                            "signed_depth_p05_m": float(np.quantile(signed_values, 0.05)),
                            "signed_depth_p95_m": float(np.quantile(signed_values, 0.95)),
                            "depth_consistent_fraction": {
                                f"{threshold_m:.3f}": float((values <= threshold_m).mean())
                                for threshold_m in consistency_thresholds_m
                            },
                        }
                metrics = {
                    "rgb_mae": float(rgb_error.mean()),
                    "depth_mae_m": float(depth_error.mean()),
                    "depth_p95_m": float(np.quantile(depth_error, 0.95)),
                    "depth_max_m": float(depth_error.max()),
                    "owner_pixel_fraction": float((render.owner_index > 0).mean()),
                    "owner_interior_pixel_fraction": float(owner_interior.mean()),
                    "owner_interior_depth_mae_m": float(owner_depth_error.mean()),
                    "owner_interior_depth_p95_m": float(np.quantile(owner_depth_error, 0.95)),
                    "owner_interior_depth_max_m": float(owner_depth_error.max()),
                    "owner_interior_by_identity": owner_metrics,
                    "depth_consistent_pixel_fraction": {
                        f"{threshold_m:.3f}": float((depth_error <= threshold_m).mean())
                        for threshold_m in consistency_thresholds_m
                    },
                    "owner_interior_depth_consistent_fraction": {
                        f"{threshold_m:.3f}": float((owner_depth_error <= threshold_m).mean())
                        for threshold_m in consistency_thresholds_m
                    },
                    "production_depth_consistent_fraction": float(production_supervised.mean()),
                }
                metrics_by_camera[camera] = metrics
                rows.append(
                    (
                        camera,
                        (
                            source_rgb.astype(np.uint8, copy=False),
                            render.rgb.astype(np.uint8, copy=False),
                            _owner_overlay(source_rgb, render.owner_index, keys),
                            _depth_error_image(source_depth, render.depth_m),
                            _unknown_overlay(source_rgb, production_supervised),
                            _owner_overlay(
                                source_rgb,
                                np.where(production_supervised, render.owner_index, 0).astype(
                                    np.uint8
                                ),
                                keys,
                            ),
                        ),
                        metrics,
                    )
                )
            filename = None
            if args.global_index is not None:
                filename = f"frame_{global_index:07d}_physical_calibration.png"
                _panel(global_index=global_index, keys=keys, rows=rows).save(
                    output_dir / filename,
                    optimize=True,
                )
            records.append(
                {
                    "global_index": global_index,
                    "scene": scene,
                    "identity_keys": list(keys),
                    "metrics": metrics_by_camera,
                    "panel": filename,
                }
            )
            if ordinal % args.progress_every == 0 or ordinal == len(selected_indices):
                print(
                    json.dumps(
                        {
                            "partition": args.partition_index,
                            "processed": ordinal,
                            "total": len(selected_indices),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        for environment in environments.values():
            close_calvin_geometry_environment(environment)
    (output_dir / "report.json").write_text(
        json.dumps(
            {
                "schema": "picf-next.calvin-physical-calibration-audit.v3",
                "runtime_input": False,
                "dataset_runtime_binding": dataset_runtime_binding,
                "frame_diagnostics": CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS,
                "owner_supervision": CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION,
                "partition_count": args.partition_count,
                "partition_index": args.partition_index,
                "records": records,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()

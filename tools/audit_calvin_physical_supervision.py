#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Audit unified CALVIN owner/geometry sidecars and emit named visual panels."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import textwrap
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="physical supervision audit",
)

import numpy as np
from PIL import Image, ImageDraw

from picf_next.artifact_io import (
    publish_prepared_directory_durable_exclusive,
    write_bytes_durable_exclusive,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import sha256_file
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (
    file_sha256,
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
    validate_dataset_runtime_binding,
)
from picf_next.data.lingbot_calvin_projection import (
    load_lingbot_calvin_projection_contract,
    projection_payload_sha256,
)
from picf_next.data.qwen3vl_raster import project_qwen3vl_segmentation
from picf_next.data.raster_targets import ProjectedRasterMembership, regular_grid_pixel_boxes
from picf_next.data.token_supervision_policy import (
    build_known_pixel_token_supervision_policy,
    token_supervision_policy_sha256,
)

_CAMERAS = ("static", "gripper")
_MAXIMUM_SIDECAR_MANIFEST_BYTES = 64 * 1024 * 1024
_CALIBRATION_METRICS = ("rgb_mae", "depth_mae_m", "depth_p95_m")
_AUDIT_QUANTILES = (
    ("minimum", 0.0),
    ("p001", 0.001),
    ("p01", 0.01),
    ("p05", 0.05),
    ("p50", 0.50),
    ("p95", 0.95),
    ("p99", 0.99),
    ("p999", 0.999),
    ("maximum", 1.0),
)
_TAIL_DIRECTIONS = {
    "rgb_mae": ("high",),
    "depth_mae_m": ("high",),
    "depth_p95_m": ("high",),
    "known_pixel_fraction": ("low",),
    "raw_object_pixel_fraction": ("high",),
    "known_object_pixel_fraction": ("low", "high"),
    "known_owner_retention": ("low",),
}


@dataclass(frozen=True, slots=True)
class _FullTailScan:
    global_indices: np.ndarray
    series: dict[str, dict[str, np.ndarray]]
    distributions: dict[str, dict[str, dict[str, float | int | None]]]
    recomputed_manifest_summary: dict[str, float]


def _color(key: str) -> tuple[int, int, int]:
    raw = hashlib.blake2b(key.encode("utf-8"), digest_size=3).digest()
    return (
        int(64 + raw[0] % 176),
        int(64 + raw[1] % 176),
        int(64 + raw[2] % 176),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")[:64] or "unknown"


def _json_bytes(payload: object) -> bytes:
    return json.dumps(payload, allow_nan=False, indent=2, sort_keys=True).encode("ascii") + b"\n"


def _png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _publish_audit_directory(
    output_dir: Path,
    prepare: Callable[[Path], dict[str, object]],
) -> dict[str, object]:
    """Build one audit under a hidden path, then publish it exactly once."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    try:
        summary = prepare(partial)
        publish_prepared_directory_durable_exclusive(partial, output_dir)
        return summary
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _owner_overlay(
    rgb: np.ndarray,
    owner: np.ndarray,
    supervised: np.ndarray,
    keys: tuple[str, ...],
) -> np.ndarray:
    if supervised.dtype != np.bool_ or supervised.shape != owner.shape:
        raise ValueError("owner supervision must be a bool raster aligned to owner labels")
    output = rgb.astype(np.float32).copy()
    for index, key in enumerate(keys, start=1):
        mask = (owner == index) & supervised
        if mask.any():
            color = np.asarray(_color(key), dtype=np.float32)
            output[mask] = 0.4 * output[mask] + 0.6 * color
    unknown = ~supervised
    if unknown.any():
        y, x = np.indices(owner.shape)
        magenta = unknown & (((y // 4) + (x // 4)) % 2 == 0)
        dark = unknown & ~magenta
        output[magenta] = np.asarray((255, 0, 255), dtype=np.float32)
        output[dark] *= 0.15
    return np.clip(output, 0, 255).astype(np.uint8)


def _token_target(
    owner: np.ndarray,
    supervised: np.ndarray,
    keys: tuple[str, ...],
    *,
    projection: dict[str, Any],
    camera_name: str,
    minimum_observed_fraction: float = 0.0,
) -> ProjectedRasterMembership:
    view = projection["views"][camera_name]
    expected_shape = tuple(int(value) for value in view["source_shape"][:2])
    if owner.shape != expected_shape or supervised.shape != expected_shape:
        raise ContractError("owner raster differs from the measured Qwen source geometry")
    return project_qwen3vl_segmentation(
        owner.astype(np.int64, copy=False),
        instance_ids=tuple(range(1, len(keys) + 1)),
        image_grid_thw=np.asarray(view["image_grid_thw"], dtype=np.int64),
        patch_size=int(projection["patch_size"]),
        merge_size=int(projection["merge_size"]),
        pixel_supervised=supervised,
        minimum_supervised_fraction=minimum_observed_fraction,
    ).merged


def _token_overlay(
    rgb: np.ndarray,
    owner: np.ndarray,
    supervised: np.ndarray,
    keys: tuple[str, ...],
    *,
    projection: dict[str, Any],
    camera_name: str,
    minimum_observed_fraction: float = 0.0,
) -> np.ndarray:
    target = _token_target(
        owner,
        supervised,
        keys,
        projection=projection,
        camera_name=camera_name,
        minimum_observed_fraction=minimum_observed_fraction,
    )
    rows, columns = projection["views"][camera_name]["merged_grid_hw"]
    boxes = regular_grid_pixel_boxes(
        height=rgb.shape[0],
        width=rgb.shape[1],
        rows=int(rows),
        columns=int(columns),
    )
    image = Image.fromarray(rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for row, (y0, x0, y1, x1) in enumerate(boxes.tolist()):
        if not target.supervised[row]:
            token_row, token_column = divmod(row, int(columns))
            fill = (255, 0, 255, 165) if (token_row + token_column) % 2 == 0 else (0, 0, 0, 210)
            draw.rectangle(
                (x0, y0, x1 - 1, y1 - 1),
                fill=fill,
                outline=(255, 0, 255, 230),
            )
            continue
        probabilities = target.object_probability[row]
        context = float(target.context_probability[row])
        observed_fraction = float(target.observed_fraction[row])
        segments = [
            (float(probability), _color(keys[source_owner - 1]))
            for probability, source_owner in zip(
                probabilities,
                target.instance_ids,
                strict=True,
            )
            if float(probability) > 0
        ]
        if context > 0:
            segments.append((context, (96, 96, 96)))
        alpha = int(45 + 145 * observed_fraction)
        cumulative = 0.0
        width = x1 - x0
        for segment_index, (probability, color) in enumerate(segments):
            segment_start = x0 + round(cumulative * width)
            cumulative = min(1.0, cumulative + probability)
            segment_stop = (
                x1 if segment_index == len(segments) - 1 else x0 + round(cumulative * width)
            )
            if segment_stop > segment_start:
                draw.rectangle(
                    (segment_start, y0, segment_stop - 1, y1 - 1),
                    fill=(*color, alpha),
                )
        partial = observed_fraction < 1.0 - 1e-6
        draw.rectangle(
            (x0, y0, x1 - 1, y1 - 1),
            outline=(255, 255, 255, 75),
        )
        if partial:
            missing_width = max(1, min(width, int(np.ceil((1.0 - observed_fraction) * width))))
            draw.line(
                (x0, y0, x0 + missing_width - 1, y0),
                fill=(255, 0, 255, 230),
                width=1,
            )
    return np.asarray(image, dtype=np.uint8)


def _known_visible_owner_indices(owner: np.ndarray, supervised: np.ndarray) -> tuple[int, ...]:
    if supervised.dtype != np.bool_ or supervised.shape != owner.shape:
        raise ValueError("owner supervision must be a bool raster aligned to owner labels")
    return tuple(
        sorted(int(value) for value in np.unique(owner[supervised]).tolist() if int(value) > 0)
    )


def _validate_projection_source_samples(
    index: CalvinDatasetIndex,
    projection: dict[str, Any],
) -> None:
    sample_indices = projection["sample_global_indices"]
    views = projection["views"]
    for camera_name in _CAMERAS:
        view = views[camera_name]
        source_field = view["source_field"]
        expected_hashes = view["source_rgb_sha256"]
        for global_index, expected_sha256 in zip(
            sample_indices,
            expected_hashes,
            strict=True,
        ):
            source = index.validated_source_frame_arrays(
                int(global_index),
                fields=(source_field,),
            )
            actual = source_array_sha256(source_field, source[source_field])
            if actual != expected_sha256:
                raise ContractError(
                    "LingBot projection source image differs from the CALVIN dataset"
                )


def _resize(image: np.ndarray, size: tuple[int, int]) -> Image.Image:
    return Image.fromarray(image).resize(size, resample=Image.Resampling.NEAREST)


def _panel(
    *,
    static: tuple[np.ndarray, np.ndarray, np.ndarray],
    gripper: tuple[np.ndarray, np.ndarray, np.ndarray],
    title: str,
    legend: str,
) -> Image.Image:
    tile = (300, 300)
    title_lines = textwrap.wrap(title, width=132, break_long_words=False) or [title]
    legend_lines = textwrap.wrap(legend, width=132, break_long_words=False) or [legend]
    header = 12 + 15 * (len(title_lines) + len(legend_lines)) + 8
    canvas = Image.new("RGB", (3 * tile[0], header + 2 * tile[1]), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    y = 7
    for line in title_lines:
        draw.text((8, y), line, fill=(255, 255, 255))
        y += 15
    for line in legend_lines:
        draw.text((8, y), line, fill=(210, 210, 210))
        y += 15
    labels = (
        "source RGB",
        "physical owners / checker=unknown",
        "Qwen 8x8 mixture / magenta=missing",
    )
    for row, (camera_name, images) in enumerate((("static", static), ("gripper", gripper))):
        for column, (label, image) in enumerate(zip(labels, images, strict=True)):
            x = column * tile[0]
            y = header + row * tile[1]
            canvas.paste(_resize(image, tile), (x, y))
            draw.rectangle((x, y, x + 235, y + 16), fill=(0, 0, 0))
            draw.text((x + 3, y + 2), f"{camera_name}: {label}", fill=(255, 255, 255))
    return canvas


def _distribution(values: np.ndarray) -> dict[str, float | int | None]:
    if values.ndim != 1:
        raise ValueError("audit metric series must be one-dimensional")
    finite = values[np.isfinite(values)]
    result: dict[str, float | int | None] = {
        "count": int(values.size),
        "finite_count": int(finite.size),
        "missing_count": int(values.size - finite.size),
    }
    for name, quantile in _AUDIT_QUANTILES:
        result[name] = (
            None if not finite.size else float(np.quantile(finite, quantile, method="linear"))
        )
    return result


def _select_extreme_indices(
    global_indices: np.ndarray,
    values: np.ndarray,
    *,
    count: int,
    direction: str,
) -> tuple[int, ...]:
    if (
        global_indices.dtype != np.int64
        or global_indices.ndim != 1
        or values.ndim != 1
        or values.shape != global_indices.shape
    ):
        raise ValueError("audit extrema require aligned int64 frame indices and metric values")
    if count <= 0 or direction not in {"low", "high"}:
        raise ValueError("audit extrema count/direction is invalid")
    finite = np.isfinite(values)
    candidate_indices = global_indices[finite]
    candidate_values = values[finite]
    if not candidate_indices.size:
        return ()
    primary = candidate_values if direction == "low" else -candidate_values
    order = np.lexsort((candidate_indices, primary))
    return tuple(int(value) for value in candidate_indices[order[:count]].tolist())


def _recomputed_manifest_summary(
    series: dict[str, dict[str, np.ndarray]],
) -> dict[str, float]:
    summary: dict[str, float] = {}
    for camera in _CAMERAS:
        for metric in _CALIBRATION_METRICS:
            summary[f"maximum_{camera}_{metric}"] = float(series[camera][metric].max())
        values = series[camera]["known_pixel_fraction"]
        summary[f"minimum_{camera}_depth_consistent_fraction"] = float(values.min())
        for name, quantile in (("p01", 0.01), ("p05", 0.05), ("p50", 0.50)):
            summary[f"{name}_{camera}_depth_consistent_fraction"] = float(
                np.quantile(values, quantile, method="linear")
            )
    return summary


def _validate_recomputed_manifest_summary(
    expected: dict[str, Any],
    measured: dict[str, float],
) -> dict[str, float]:
    if set(expected) != set(measured):
        raise ContractError("CALVIN audit and sidecar calibration summary fields differ")
    absolute_error = {}
    for name, measured_value in measured.items():
        expected_value = expected[name]
        if (
            isinstance(expected_value, bool)
            or not isinstance(expected_value, int | float)
            or not np.isfinite(expected_value)
        ):
            raise ContractError("CALVIN sidecar calibration summary is not numeric")
        error = abs(float(expected_value) - measured_value)
        absolute_error[name] = error
        if not np.isclose(
            float(expected_value),
            measured_value,
            rtol=0.0,
            atol=1e-12,
        ):
            raise ContractError(
                f"CALVIN full-tail summary mismatch for {name}: "
                f"manifest={expected_value}, measured={measured_value}"
            )
    return absolute_error


def _scan_full_tail(
    sidecar: CalvinPhysicalSupervisionSidecar,
    manifest: dict[str, Any],
) -> _FullTailScan:
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise ContractError("full-tail audit requires all-source physical supervision")
    index_parts: list[np.ndarray] = []
    parts: dict[str, dict[str, list[np.ndarray]]] = {
        camera: defaultdict(list) for camera in _CAMERAS
    }
    for shard_index, metadata in enumerate(sidecar.shards):
        # Use the canonical sidecar loader so every array receives exactly the
        # same schema/value validation used by training before audit statistics
        # are derived from its already validated arrays.
        loaded = sidecar._load_shard(shard_index)  # noqa: SLF001
        if (
            loaded.global_indices.shape != (metadata.frame_count,)
            or int(loaded.global_indices[0]) != metadata.first_global_index
            or int(loaded.global_indices[-1]) != metadata.last_global_index
        ):
            raise ContractError("CALVIN full-tail shard metadata differs from validated arrays")
        index_parts.append(loaded.global_indices.copy())
        for camera in _CAMERAS:
            arrays = loaded.camera_arrays[camera]
            owner = arrays["owner_index"]
            supervised = arrays["owner_supervised"]
            pixel_count = owner.shape[1] * owner.shape[2]
            raw_object_pixels = np.count_nonzero(owner, axis=(1, 2))
            known_object_pixels = np.count_nonzero((owner > 0) & supervised, axis=(1, 2))
            retention = np.full(metadata.frame_count, np.nan, dtype=np.float64)
            np.divide(
                known_object_pixels,
                raw_object_pixels,
                out=retention,
                where=raw_object_pixels > 0,
            )
            for metric in _CALIBRATION_METRICS:
                parts[camera][metric].append(np.asarray(arrays[metric], dtype=np.float64).copy())
            parts[camera]["known_pixel_fraction"].append(
                np.asarray(arrays["depth_consistent_fraction"], dtype=np.float64).copy()
            )
            parts[camera]["raw_object_pixel_fraction"].append(
                raw_object_pixels.astype(np.float64) / pixel_count
            )
            parts[camera]["known_object_pixel_fraction"].append(
                known_object_pixels.astype(np.float64) / pixel_count
            )
            parts[camera]["known_owner_retention"].append(retention)
    sidecar.clear_cache()
    global_indices = np.concatenate(index_parts)
    if (
        global_indices.dtype != np.int64
        or global_indices.ndim != 1
        or np.any(global_indices[1:] <= global_indices[:-1])
        or global_indices.size != manifest.get("frame_count")
    ):
        raise ContractError("CALVIN full-tail audit did not scan exact manifest coverage")
    series = {
        camera: {
            metric: np.concatenate(metric_parts) for metric, metric_parts in camera_parts.items()
        }
        for camera, camera_parts in parts.items()
    }
    for camera, camera_series in series.items():
        if set(camera_series) != set(_TAIL_DIRECTIONS):
            raise ContractError(f"CALVIN full-tail metric set is incomplete for {camera}")
        if any(values.shape != global_indices.shape for values in camera_series.values()):
            raise ContractError("CALVIN full-tail metric coverage differs from frame coverage")
    distributions = {
        camera: {metric: _distribution(values) for metric, values in camera_series.items()}
        for camera, camera_series in series.items()
    }
    return _FullTailScan(
        global_indices=global_indices,
        series=series,
        distributions=distributions,
        recomputed_manifest_summary=_recomputed_manifest_summary(series),
    )


def _add_selection(selections: dict[int, set[str]], global_index: int, reason: str) -> None:
    selections.setdefault(global_index, set()).add(reason)


def _is_covered(global_indices: np.ndarray, global_index: int) -> bool:
    position = int(np.searchsorted(global_indices, global_index))
    return position < global_indices.size and int(global_indices[position]) == global_index


def _full_tail_selections(
    index: CalvinDatasetIndex,
    scan: _FullTailScan,
    *,
    tail_per_metric: int,
    temporal_strata: int,
) -> dict[int, tuple[str, ...]]:
    if tail_per_metric <= 0 or temporal_strata <= 0:
        raise ValueError("full-tail selection counts must be positive")
    selections: dict[int, set[str]] = {}
    for camera in _CAMERAS:
        for metric, directions in _TAIL_DIRECTIONS.items():
            values = scan.series[camera][metric]
            for direction in directions:
                for global_index in _select_extreme_indices(
                    scan.global_indices,
                    values,
                    count=tail_per_metric,
                    direction=direction,
                ):
                    _add_selection(
                        selections,
                        global_index,
                        f"metric_tail:{camera}:{metric}:{direction}",
                    )
    positions = np.rint(
        np.linspace(
            0,
            scan.global_indices.size - 1,
            num=min(temporal_strata, scan.global_indices.size),
        )
    ).astype(np.int64)
    for stratum, position in enumerate(np.unique(positions).tolist()):
        _add_selection(
            selections,
            int(scan.global_indices[position]),
            f"temporal_stratum:{stratum:03d}",
        )
    segments_by_task: dict[str, list[Any]] = defaultdict(list)
    for segment in index.segments:
        segments_by_task[segment.task_key].append(segment)
    for task_key in sorted(segments_by_task):
        segments = segments_by_task[task_key]
        segment = segments[(len(segments) - 1) // 2]
        global_index = (segment.start + segment.end) // 2
        if not _is_covered(scan.global_indices, global_index):
            raise ContractError("CALVIN task-stratified audit frame is outside sidecar coverage")
        _add_selection(selections, global_index, f"task_stratum:{task_key}")
    return {
        global_index: tuple(sorted(reasons)) for global_index, reasons in sorted(selections.items())
    }


def _frame_annotations(index: CalvinDatasetIndex, global_index: int) -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "segment_index": segment.index,
            "task_key": segment.task_key,
            "instruction": segment.instruction,
        }
        for segment in index.segments
        if segment.start <= global_index <= segment.end
    )


def _camera_support_metrics(
    owner: np.ndarray,
    supervised: np.ndarray,
) -> dict[str, float | None]:
    pixels = owner.size
    raw_object_pixels = int(np.count_nonzero(owner))
    known_object_pixels = int(np.count_nonzero((owner > 0) & supervised))
    return {
        "known_pixel_fraction": float(supervised.mean()),
        "raw_object_pixel_fraction": raw_object_pixels / pixels,
        "known_object_pixel_fraction": known_object_pixels / pixels,
        "known_owner_retention": (
            None if raw_object_pixels == 0 else known_object_pixels / raw_object_pixels
        ),
    }


def _run_full_tail_audit(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    sidecar_manifest_bytes: bytes,
    dataset_manifest_path: Path,
    projection: dict[str, Any],
    projection_contract_sha256: str,
    output: Path,
    tail_per_metric: int,
    temporal_strata: int,
) -> dict[str, object]:
    supervision_policy = build_known_pixel_token_supervision_policy()
    supervision_policy_sha256 = token_supervision_policy_sha256(supervision_policy)
    minimum_observed_fraction = float.fromhex(
        str(supervision_policy["minimum_observed_fraction_hex"])
    )
    if hashlib.sha256(sidecar_manifest_bytes).hexdigest() != sidecar.manifest_sha256:
        raise ContractError("CALVIN sidecar manifest changed after verification")
    manifest = json.loads(sidecar_manifest_bytes)
    if not isinstance(manifest, dict):
        raise ContractError("CALVIN sidecar manifest must be a mapping")
    scan = _scan_full_tail(sidecar, manifest)
    summary_error = _validate_recomputed_manifest_summary(
        manifest["calibration_summary"],
        scan.recomputed_manifest_summary,
    )
    selections = _full_tail_selections(
        index,
        scan,
        tail_per_metric=tail_per_metric,
        temporal_strata=temporal_strata,
    )
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for global_index, reasons in selections.items():
        physical = sidecar.source_frame(global_index)
        source = index.validated_source_frame_arrays(
            global_index,
            fields=("rgb_static", "depth_static", "rgb_gripper", "depth_gripper"),
        )
        camera_images = []
        camera_records = {}
        visible = set()
        for camera_name in _CAMERAS:
            camera = next(value for value in physical.cameras if value.camera_name == camera_name)
            rgb = np.asarray(source[f"rgb_{camera_name}"], dtype=np.uint8)
            depth = np.asarray(source[f"depth_{camera_name}"])
            if source_array_sha256(f"rgb_{camera_name}", rgb) != camera.source_rgb_sha256:
                raise ContractError("CALVIN physical source RGB hash mismatch")
            if source_array_sha256(f"depth_{camera_name}", depth) != camera.source_depth_sha256:
                raise ContractError("CALVIN physical source depth hash mismatch")
            visible.update(
                _known_visible_owner_indices(camera.owner_index, camera.owner_supervised)
            )
            token_target = _token_target(
                camera.owner_index,
                camera.owner_supervised,
                physical.identity_keys,
                projection=projection,
                camera_name=camera_name,
                minimum_observed_fraction=minimum_observed_fraction,
            )
            camera_images.append(
                (
                    rgb,
                    _owner_overlay(
                        rgb,
                        camera.owner_index,
                        camera.owner_supervised,
                        physical.identity_keys,
                    ),
                    _token_overlay(
                        rgb,
                        camera.owner_index,
                        camera.owner_supervised,
                        physical.identity_keys,
                        projection=projection,
                        camera_name=camera_name,
                        minimum_observed_fraction=minimum_observed_fraction,
                    ),
                )
            )
            camera_records[camera_name] = {
                "source_rgb_sha256": camera.source_rgb_sha256,
                "source_depth_sha256": camera.source_depth_sha256,
                "rgb_mae": camera.rgb_mae,
                "depth_mae_m": camera.depth_mae_m,
                "depth_p95_m": camera.depth_p95_m,
                **_camera_support_metrics(
                    camera.owner_index,
                    camera.owner_supervised,
                ),
                "training_token_nonzero_observed_fraction": float(token_target.supervised.mean()),
                "training_token_mean_observed_fraction": float(
                    token_target.observed_fraction.mean()
                ),
                "training_token_min_positive_observed_fraction": (
                    None
                    if not token_target.supervised.any()
                    else float(token_target.observed_fraction[token_target.supervised].min())
                ),
            }
        annotations = _frame_annotations(index, global_index)
        task_label = (
            "all-source unlabeled"
            if not annotations
            else " | ".join(f"{value['task_key']}: {value['instruction']}" for value in annotations)
        )
        visible_keys = tuple(
            physical.identity_keys[owner_index - 1] for owner_index in sorted(visible)
        )
        title = f"step={global_index} task={task_label}"
        legend = (
            "reasons=" + " | ".join(reasons) + "; visible=" + (" | ".join(visible_keys) or "none")
        )
        filename = f"step{global_index:07d}_{_slug(task_label)}.png"
        path = output / filename
        write_bytes_durable_exclusive(
            path,
            _png_bytes(
                _panel(
                    static=camera_images[0],
                    gripper=camera_images[1],
                    title=title,
                    legend=legend,
                )
            ),
        )
        row = int(np.searchsorted(scan.global_indices, global_index))
        if row >= scan.global_indices.size or int(scan.global_indices[row]) != global_index:
            raise ContractError("CALVIN selected audit frame left full-scan coverage")
        records.append(
            {
                "global_index": global_index,
                "selection_reasons": reasons,
                "task_annotations": annotations,
                "identity_keys": physical.identity_keys,
                "visible_identity_keys": visible_keys,
                "panel": filename,
                "panel_sha256": sha256_file(path),
                "cameras": camera_records,
                "scanned_metrics": {
                    camera: {
                        metric: (
                            None
                            if not np.isfinite(scan.series[camera][metric][row])
                            else float(scan.series[camera][metric][row])
                        )
                        for metric in scan.series[camera]
                    }
                    for camera in _CAMERAS
                },
            }
        )
        print(json.dumps(records[-1], sort_keys=True), flush=True)
    report = {
        "format": "picf-next.calvin-physical-supervision-audit.v5",
        "mode": "full_tail",
        "runtime_input": False,
        "task_used_for_owner_selection": False,
        "task_used_for_audit_selection": True,
        "selection_affects_training": False,
        "coverage": sidecar.coverage,
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "sidecar_manifest_sha256": sidecar.manifest_sha256,
        "training_projection_contract_sha256": projection_contract_sha256,
        "training_projection_payload_sha256": projection_payload_sha256(projection),
        "training_projection": projection,
        "training_supervision_policy_sha256": supervision_policy_sha256,
        "training_supervision_policy": supervision_policy,
        "frame_count": int(scan.global_indices.size),
        "first_global_index": int(scan.global_indices[0]),
        "last_global_index": int(scan.global_indices[-1]),
        "full_shard_schema_validation": True,
        "manifest_summary_match": True,
        "manifest_summary_absolute_error": summary_error,
        "distributions": scan.distributions,
        "selection_contract": {
            "tail_per_metric": tail_per_metric,
            "tail_directions": _TAIL_DIRECTIONS,
            "temporal_strata": temporal_strata,
            "one_median_occurrence_midpoint_per_task": True,
            "deduplicated": True,
        },
        "record_count": len(records),
        "records": records,
    }
    manifest_path = output / "audit_manifest.json"
    write_bytes_durable_exclusive(manifest_path, _json_bytes(report))
    return {
        "audit_manifest": manifest_path.name,
        "audit_manifest_sha256": sha256_file(manifest_path),
        "frame_count": report["frame_count"],
        "record_count": report["record_count"],
    }


def _run_sampled_audit(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    projection: dict[str, Any],
    projection_contract_sha256: str,
    dataset_manifest_sha256: str,
    supervision_policy: dict[str, object],
    minimum_observed_fraction: float,
    output: Path,
    max_segments: int,
) -> dict[str, object]:
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for segment in index.segments[:max_segments]:
        for phase, global_index in (
            ("start", segment.start),
            ("mid", (segment.start + segment.end) // 2),
            ("end", segment.end),
        ):
            physical = sidecar(segment.index, global_index)
            source = index.validated_source_frame_arrays(
                global_index,
                fields=("rgb_static", "depth_static", "rgb_gripper", "depth_gripper"),
            )
            camera_images = []
            visible = set()
            for camera in physical.cameras:
                suffix = "static" if camera.camera_name == "static" else "gripper"
                rgb = np.asarray(source[f"rgb_{suffix}"], dtype=np.uint8)
                depth = np.asarray(source[f"depth_{suffix}"])
                if source_array_sha256(f"rgb_{suffix}", rgb) != camera.source_rgb_sha256:
                    raise ValueError("CALVIN physical source RGB hash mismatch")
                if source_array_sha256(f"depth_{suffix}", depth) != camera.source_depth_sha256:
                    raise ValueError("CALVIN physical source depth hash mismatch")
                visible.update(
                    _known_visible_owner_indices(
                        camera.owner_index,
                        camera.owner_supervised,
                    )
                )
                camera_images.append(
                    (
                        rgb,
                        _owner_overlay(
                            rgb,
                            camera.owner_index,
                            camera.owner_supervised,
                            physical.identity_keys,
                        ),
                        _token_overlay(
                            rgb,
                            camera.owner_index,
                            camera.owner_supervised,
                            physical.identity_keys,
                            projection=projection,
                            camera_name=camera.camera_name,
                            minimum_observed_fraction=minimum_observed_fraction,
                        ),
                    )
                )
            visible_keys = tuple(
                physical.identity_keys[owner_index - 1] for owner_index in sorted(visible)
            )
            title = (
                f"segment={segment.index} phase={phase} step={global_index} "
                f"task={segment.task_key} | {segment.instruction}"
            )
            legend = "visible=" + (" | ".join(visible_keys) or "none")
            filename = (
                f"segment{segment.index:03d}_{_slug(segment.task_key)}_"
                f"{phase}_step{global_index:07d}.png"
            )
            path = output / filename
            write_bytes_durable_exclusive(
                path,
                _png_bytes(
                    _panel(
                        static=camera_images[0],
                        gripper=camera_images[1],
                        title=title,
                        legend=legend,
                    )
                ),
            )
            records.append(
                {
                    "segment_index": segment.index,
                    "phase": phase,
                    "global_index": global_index,
                    "task_key": segment.task_key,
                    "instruction": segment.instruction,
                    "panel": filename,
                    "visible_identity_keys": visible_keys,
                    "calibration": {
                        camera.camera_name: {
                            "rgb_mae": camera.rgb_mae,
                            "depth_mae_m": camera.depth_mae_m,
                            "depth_p95_m": camera.depth_p95_m,
                            "known_pixel_fraction": camera.depth_consistent_fraction,
                        }
                        for camera in physical.cameras
                    },
                }
            )
            print(json.dumps(records[-1], sort_keys=True), flush=True)
    report = {
        "format": "picf-next.calvin-physical-supervision-audit.v4",
        "runtime_input": False,
        "task_used_for_owner_selection": False,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "sidecar_manifest_sha256": sidecar.manifest_sha256,
        "training_projection_contract_sha256": projection_contract_sha256,
        "training_projection_payload_sha256": projection_payload_sha256(projection),
        "training_projection": projection,
        "training_supervision_policy_sha256": token_supervision_policy_sha256(supervision_policy),
        "training_supervision_policy": supervision_policy,
        "records": records,
    }
    manifest_path = output / "audit_manifest.json"
    write_bytes_durable_exclusive(manifest_path, _json_bytes(report))
    return {
        "audit_manifest": manifest_path.name,
        "audit_manifest_sha256": sha256_file(manifest_path),
        "record_count": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument(
        "--sidecar-manifest",
        type=Path,
        help="optional immutable manifest view for the existing sidecar shard root",
    )
    parser.add_argument(
        "--sidecar-manifest-sha256",
        help="required immutable SHA-256 when --sidecar-manifest is external",
    )
    parser.add_argument(
        "--training-projection-contract",
        required=True,
        type=Path,
        help="official Qwen geometry measured on real CALVIN images",
    )
    parser.add_argument(
        "--training-projection-contract-sha256",
        required=True,
        help="immutable SHA-256 of --training-projection-contract",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--max-segments", type=int, default=4)
    parser.add_argument(
        "--full-tail",
        action="store_true",
        help="scan every hash-bound shard and render deterministic metric/task/time tails",
    )
    parser.add_argument("--tail-per-metric", type=int, default=4)
    parser.add_argument("--temporal-strata", type=int, default=16)
    args = parser.parse_args()

    split_root = args.split_root.resolve()
    dataset_manifest_path = args.dataset_manifest.resolve()
    dataset_manifest_sha256 = sha256_file(dataset_manifest_path)
    dataset_manifest = load_dataset_file_manifest(dataset_manifest_path)
    projection = load_lingbot_calvin_projection_contract(
        args.training_projection_contract.resolve(),
        expected_sha256=args.training_projection_contract_sha256,
        expected_dataset_manifest_sha256=dataset_manifest_sha256,
    )
    supervision_policy = build_known_pixel_token_supervision_policy()
    minimum_observed_fraction = float.fromhex(
        str(supervision_policy["minimum_observed_fraction_hex"])
    )
    validate_dataset_runtime_binding(
        dataset_manifest,
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=split_root.name,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    _validate_projection_source_samples(index, projection)
    sidecar_manifest_path = (
        args.sidecar_manifest
        if args.sidecar_manifest is not None
        else args.sidecar_root / "manifest.json"
    )
    if args.sidecar_manifest is not None and args.sidecar_manifest_sha256 is None:
        raise ValueError("external sidecar manifest requires its immutable SHA-256")
    sidecar_manifest_sha256 = (
        file_sha256(sidecar_manifest_path)
        if args.sidecar_manifest_sha256 is None
        else args.sidecar_manifest_sha256
    )
    sidecar_manifest_bytes = read_sha256_verified_file_beneath(
        sidecar_manifest_path.parent,
        sidecar_manifest_path.name,
        expected_sha256=sidecar_manifest_sha256,
        maximum_bytes=_MAXIMUM_SIDECAR_MANIFEST_BYTES,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.sidecar_root,
        index,
        manifest_bytes=sidecar_manifest_bytes,
    )
    output_dir = args.output_dir.resolve()
    if args.full_tail:
        summary = _publish_audit_directory(
            output_dir,
            lambda output: _run_full_tail_audit(
                index=index,
                sidecar=sidecar,
                sidecar_manifest_bytes=sidecar_manifest_bytes,
                dataset_manifest_path=dataset_manifest_path,
                projection=projection,
                projection_contract_sha256=args.training_projection_contract_sha256,
                output=output,
                tail_per_metric=args.tail_per_metric,
                temporal_strata=args.temporal_strata,
            ),
        )
        event = "full_tail_audit_complete"
    else:
        summary = _publish_audit_directory(
            output_dir,
            lambda output: _run_sampled_audit(
                index=index,
                sidecar=sidecar,
                projection=projection,
                projection_contract_sha256=args.training_projection_contract_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                supervision_policy=supervision_policy,
                minimum_observed_fraction=minimum_observed_fraction,
                output=output,
                max_segments=args.max_segments,
            ),
        )
        event = "sampled_audit_complete"
    print(
        json.dumps(
            {
                "event": event,
                "output_dir": str(output_dir),
                **summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

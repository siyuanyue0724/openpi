#!/usr/bin/env python3
"""Build task-independent visible-instance golden panels from official CALVIN data.

The task string is written only into audit metadata and panel titles. It never
selects an instance. PyBullet segmentation and simulator identities are
training-only targets; no generated field is a PICF runtime input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_SOURCE_COMMIT,
)
from picf_next.data.calvin_simulator_geometry import (
    build_calvin_geometry_environment,
    calvin_segmentation_identity_map,
    close_calvin_geometry_environment,
    restore_calvin_archived_state,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    load_dataset_file_manifest,
    validate_dataset_files,
)
from picf_next.data.raster_targets import (
    ProjectedRasterMembership,
    project_exclusive_segmentation,
    regular_grid_pixel_boxes,
)

DEBUG_ARCHIVE_SHA256 = "c66d09147e2c806b244f18ea7d61e388d4dac11f828929779437f728d03e1204"


@dataclass(frozen=True, slots=True)
class Segment:
    index: int
    start: int
    end: int
    task: str
    prompt: str


@dataclass(frozen=True, slots=True)
class VisibleInstance:
    key: str
    encoded_id: int
    pixel_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_segments(
    dataset_root: Path,
    split: str,
    manifest: DatasetFileManifest,
) -> tuple[list[Segment], CalvinDatasetIndex]:
    split_root = dataset_root / split
    validate_dataset_files(
        manifest,
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=split,
        verify_hashes=True,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        dataset_manifest=manifest,
    )
    return (
        [
            Segment(
                index=segment.index,
                start=segment.start,
                end=segment.end,
                task=segment.task_key,
                prompt=segment.instruction,
            )
            for segment in index.segments
        ],
        index,
    )


def _scene_instance_map(scene_info: dict[str, Any]) -> dict[int, str]:
    """Map all salient CALVIN scene objects/parts without consulting a task."""

    return calvin_segmentation_identity_map(scene_info)


def _stable_color(key: str) -> np.ndarray:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=3).digest()
    raw = np.frombuffer(digest, dtype=np.uint8).astype(np.int64)
    return (64 + raw % 176).astype(np.uint8)


def _overlay_instances(
    reference_rgb: np.ndarray,
    segmentation: np.ndarray,
    instances: tuple[VisibleInstance, ...],
) -> np.ndarray:
    output = reference_rgb.astype(np.float32).copy()
    for instance in instances:
        mask = segmentation == instance.encoded_id
        color = _stable_color(instance.key).astype(np.float32)
        output[mask] = 0.45 * output[mask] + 0.55 * color
    return np.clip(output, 0, 255).astype(np.uint8)


def _token_overlay(
    reference_rgb: np.ndarray,
    projected: ProjectedRasterMembership,
    boxes: np.ndarray,
    key_by_id: dict[int, str],
) -> np.ndarray:
    image = Image.fromarray(reference_rgb).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    for index, (y0, x0, y1, x1) in enumerate(boxes.tolist()):
        if not projected.supervised[index]:
            draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 255, 255))
            continue
        object_row = projected.object_probability[index]
        context = float(projected.context_probability[index])
        if object_row.size and float(object_row.max()) > context:
            owner = int(object_row.argmax())
            key = key_by_id[projected.instance_ids[owner]]
            color = _stable_color(key)
            alpha = int(48 + 120 * float(object_row[owner]))
            fill = (int(color[0]), int(color[1]), int(color[2]), alpha)
        else:
            fill = (96, 96, 96, int(32 + 96 * context))
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill, outline=(255, 255, 255, 90))
    return np.asarray(image, dtype=np.uint8)


def _panel(
    *,
    reference_rgb: np.ndarray,
    rendered_rgb: np.ndarray,
    instance_overlay: np.ndarray,
    token_overlay: np.ndarray,
    title: str,
    instances: tuple[VisibleInstance, ...],
) -> Image.Image:
    height, width = reference_rgb.shape[:2]
    header_height = 96
    canvas = Image.new("RGB", (4 * width, header_height + height), color=(8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(255, 255, 255))
    legend = (
        " | ".join(f"{item.key}:{item.pixel_count}" for item in instances)
        or "no selected visible instance"
    )
    wrapped = [legend[offset : offset + 125] for offset in range(0, len(legend), 125)]
    for line_index, line in enumerate(wrapped[:5]):
        draw.text((8, 25 + 13 * line_index), line, fill=(210, 210, 210))
    labels = ("reference", "rerender", "visible instances", "14x14 token target")
    images = (reference_rgb, rendered_rgb, instance_overlay, token_overlay)
    for column, (label, array) in enumerate(zip(labels, images, strict=True)):
        x = column * width
        canvas.paste(Image.fromarray(array), (x, header_height))
        draw.rectangle((x, header_height, x + 112, header_height + 14), fill=(0, 0, 0))
        draw.text((x + 3, header_height + 1), label, fill=(255, 255, 255))
    return canvas


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return cleaned[:64] or "unknown"


def _phase_steps(segment: Segment, phases: tuple[str, ...]) -> list[tuple[str, int]]:
    values = {
        "start": segment.start,
        "mid": (segment.start + segment.end) // 2,
        "end": segment.end,
    }
    return [(phase, values[phase]) for phase in phases]


def _build_environment(calvin_env_root: Path):
    import pybullet

    environment = build_calvin_geometry_environment(
        calvin_env_root,
        scene="calvin_scene_D",
        include_cameras=True,
    )
    return environment, pybullet


def _render(environment, pybullet, frame: dict[str, np.ndarray]):
    restore_calvin_archived_state(
        environment,
        scene_obs=frame["scene_obs"],
        robot_obs=frame["robot_obs"],
    )
    camera = environment.cameras[0]
    raw = environment.p.getCameraImage(
        width=camera.width,
        height=camera.height,
        viewMatrix=camera.viewMatrix,
        projectionMatrix=camera.projectionMatrix,
        flags=pybullet.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
        physicsClientId=environment.cid,
    )
    rendered = np.asarray(raw[2]).reshape(camera.height, camera.width, 4)[..., :3]
    segmentation = np.asarray(raw[4]).reshape(camera.height, camera.width)
    return (
        rendered.astype(np.uint8),
        segmentation.astype(np.int64),
        environment.get_info()["scene_info"],
    )


def _close_environment(environment) -> None:
    close_calvin_geometry_environment(environment)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--calvin-env-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--dataset-manifest",
        required=True,
        action="append",
        type=Path,
        help="one content-addressed split manifest; repeat for every selected split",
    )
    parser.add_argument("--split", action="append", choices=("training", "validation"))
    parser.add_argument("--phase", action="append", choices=("start", "mid", "end"))
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--grid-rows", type=int, default=14)
    parser.add_argument("--grid-columns", type=int, default=14)
    parser.add_argument("--maximum-rgb-mae", type=float, default=36.0)
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = tuple(args.split or ("training", "validation"))
    phases = tuple(args.phase or ("start", "mid", "end"))
    manifests: dict[str, DatasetFileManifest] = {}
    for manifest_path in args.dataset_manifest:
        manifest = load_dataset_file_manifest(manifest_path.resolve())
        if manifest.split_name in manifests:
            raise ContractError(f"duplicate dataset manifest for split {manifest.split_name}")
        manifests[manifest.split_name] = manifest
    missing_manifests = sorted(set(splits) - manifests.keys())
    if missing_manifests:
        raise ContractError(f"missing dataset manifests for splits: {missing_manifests}")
    environment, pybullet = _build_environment(args.calvin_env_root.resolve())
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    try:
        for split in splits:
            segments, index = _load_segments(dataset_root, split, manifests[split])
            if args.max_segments is not None:
                segments = segments[: args.max_segments]
            for segment in segments:
                for phase, step in _phase_steps(segment, phases):
                    source = dataset_root / split / f"episode_{step:07d}.npz"
                    frame = index.validated_source_frame_arrays(step)
                    rendered, segmentation, scene_info = _render(environment, pybullet, frame)
                    reference = np.asarray(frame["rgb_static"], dtype=np.uint8)
                    mae = float(
                        np.abs(rendered.astype(np.float32) - reference.astype(np.float32)).mean()
                    )
                    if mae > args.maximum_rgb_mae:
                        failures.append(f"{split}/{step}: rgb_mae={mae:.4f}")
                    instance_map = _scene_instance_map(scene_info)
                    instances = tuple(
                        VisibleInstance(
                            key=key,
                            encoded_id=encoded_id,
                            pixel_count=int((segmentation == encoded_id).sum()),
                        )
                        for encoded_id, key in sorted(
                            instance_map.items(), key=lambda item: item[1]
                        )
                        if (segmentation == encoded_id).any()
                    )
                    boxes = regular_grid_pixel_boxes(
                        height=reference.shape[0],
                        width=reference.shape[1],
                        rows=args.grid_rows,
                        columns=args.grid_columns,
                    )
                    projected = project_exclusive_segmentation(
                        segmentation,
                        instance_ids=tuple(item.encoded_id for item in instances),
                        token_boxes_yxyx=boxes,
                    )
                    key_by_id = {item.encoded_id: item.key for item in instances}
                    object_overlay = _overlay_instances(reference, segmentation, instances)
                    token_overlay = _token_overlay(reference, projected, boxes, key_by_id)
                    simplex = (
                        projected.object_probability.sum(axis=-1) + projected.context_probability
                    )
                    simplex_error = float(
                        np.abs(simplex[projected.supervised] - 1.0).max(initial=0.0)
                    )
                    title = (
                        f"split={split} segment={segment.index} phase={phase} "
                        f"step={step} task={segment.task} rgb_mae={mae:.3f}"
                    )
                    panel = _panel(
                        reference_rgb=reference,
                        rendered_rgb=rendered,
                        instance_overlay=object_overlay,
                        token_overlay=token_overlay,
                        title=title,
                        instances=instances,
                    )
                    filename = (
                        f"{split}_seg{segment.index:03d}_{_slug(segment.task)}_"
                        f"{phase}_step{step:07d}.png"
                    )
                    panel_path = output_dir / filename
                    panel.save(panel_path, optimize=True)
                    records.append(
                        {
                            "split": split,
                            "segment_index": segment.index,
                            "phase": phase,
                            "step": step,
                            "task": segment.task,
                            "prompt": segment.prompt,
                            "source_episode": source.name,
                            "source_sha256": manifests[split].record_for(source.name).sha256,
                            "panel": filename,
                            "panel_sha256": _sha256(panel_path),
                            "render_rgb_mae": mae,
                            "token_simplex_max_error": simplex_error,
                            "visible_instances": [
                                {
                                    "key": item.key,
                                    "encoded_id": item.encoded_id,
                                    "pixel_count": item.pixel_count,
                                }
                                for item in instances
                            ],
                        }
                    )
                    print(json.dumps(records[-1], sort_keys=True), flush=True)
    finally:
        _close_environment(environment)

    manifest = {
        "format": "picf-next.calvin-visible-instance-golden.v1",
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
        "debug_archive_sha256": DEBUG_ARCHIVE_SHA256,
        "task_used_for_instance_selection": False,
        "runtime_input": False,
        "grid": {"rows": args.grid_rows, "columns": args.grid_columns},
        "maximum_rgb_mae": args.maximum_rgb_mae,
        "records": records,
        "failures": failures,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if failures:
        raise RuntimeError(
            f"{len(failures)} rendered frames exceeded RGB calibration threshold; "
            f"see {manifest_path}"
        )


if __name__ == "__main__":
    main()

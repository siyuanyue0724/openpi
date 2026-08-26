#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Audit behavior-aligned CALVIN support without authorizing model training."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import textwrap
from collections import Counter, defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="CALVIN behavior-grounding support audit",
)

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from picf_next.artifact_io import (
    publish_prepared_directory_durable_exclusive,
    write_bytes_durable_exclusive,
)
from picf_next.contracts import ContractError
from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinLanguageSegment,
)
from picf_next.data.calvin_behavior_support import (
    CALVIN_ROBOT_BASE_EULER_RAD,
    CALVIN_ROBOT_BASE_POSITION_M,
    CALVIN_SCENE_CONFIG_SHA256,
    calvin_behavior_review_keyframes,
    select_calvin_behavior_segments,
    summarize_calvin_behavior_support,
)
from picf_next.data.calvin_geometry_schema import calvin_source_state_sha256
from picf_next.data.calvin_official_source import (
    CALVIN_OFFICIAL_DATASET_ID,
    validate_calvin_content_identity_migration,
    validate_calvin_official_source_receipt,
)
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.dataset_manifest import (
    file_sha256,
    load_dataset_file_manifest,
    read_sha256_verified_file_beneath,
    validate_dataset_runtime_binding,
)
from picf_next.data.calvin_simulator_geometry import (
    CalvinSceneRange,
    load_calvin_scene_ranges,
    scene_for_global_index,
)
from picf_next.eval.calvin_task_relevance import (
    calvin_task_physical_relevance,
    calvin_task_physical_relevance_inventory,
    validate_calvin_task_protocol_inventory,
    validate_calvin_task_protocol_sources,
)

if TYPE_CHECKING:
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionFrame,
        CalvinPhysicalSupervisionSidecar,
    )

AUDIT_SCHEMA = "picf-next.calvin-behavior-grounding-support-audit.v3"
SELECTION_ALGORITHM = "per-task-scene-source-order-even-quantiles.v2"
RENDER_SELECTION_ALGORITHM = "per-selected-task-scene-middle-segment-evidence-keyframes.v2"
_CAMERA_FIELDS = {"static": "rgb_static", "gripper": "rgb_gripper"}
_MAXIMUM_SOURCE_RECEIPT_BYTES = 2 * 1024 * 1024
_REPO_ROOT = Path(__file__).resolve().parents[1]
_RUNTIME_SOURCE_ROOTS = {"src": _REPO_ROOT / "src", "tools": _REPO_ROOT / "tools"}


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value).strip("-")


def _validate_expected_full_dataset_identity(dataset_id: str, dataset_revision: str) -> None:
    if dataset_id != CALVIN_OFFICIAL_DATASET_ID:
        raise ContractError("expected full CALVIN dataset identity is not the official source")
    if (
        not isinstance(dataset_revision, str)
        or not dataset_revision.startswith("sha256:")
        or len(dataset_revision) != 71
        or any(character not in "0123456789abcdef" for character in dataset_revision[7:])
    ):
        raise ContractError("expected full CALVIN revision is not a content SHA-256")


def _load_source_receipt(path: Path, expected_sha256: str) -> tuple[dict[str, object], str]:
    payload = read_sha256_verified_file_beneath(
        path.parent,
        path.name,
        expected_sha256=expected_sha256,
        maximum_bytes=_MAXIMUM_SOURCE_RECEIPT_BYTES,
    )
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError("CALVIN source receipt is not valid JSON") from error
    if not isinstance(value, dict):
        raise ContractError("CALVIN source receipt must be a mapping")
    return value, hashlib.sha256(payload).hexdigest()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _denormalized_geometry(frame: CalvinPhysicalSupervisionFrame) -> np.ndarray:
    geometry = frame.geometry.detach().cpu().numpy().astype(np.float64, copy=False)
    supervised = frame.geometry_supervised.detach().cpu().numpy()
    expected = (len(frame.identity_keys), frame.geometry_contract.dimension)
    if geometry.shape != expected or supervised.shape != expected or not supervised.all():
        raise ContractError("behavior audit requires fully supervised physical geometry")
    scale = np.asarray(frame.geometry_contract.normalization_scale, dtype=np.float64)
    offset = np.asarray(frame.geometry_contract.normalization_offset, dtype=np.float64)
    result = geometry * scale + offset
    if not np.isfinite(result).all():
        raise ContractError("denormalized physical geometry is not finite")
    return result


def _target_visible_pixels(
    frame: CalvinPhysicalSupervisionFrame,
    target_identity_key: str,
) -> dict[str, int]:
    try:
        owner_index = frame.identity_keys.index(target_identity_key) + 1
    except ValueError as error:
        raise ContractError("task target is absent from the physical sidecar") from error
    result = {}
    for camera in frame.cameras:
        result[camera.camera_name] = int(
            np.count_nonzero((camera.owner_index == owner_index) & camera.owner_supervised)
        )
    if set(result) != set(_CAMERA_FIELDS):
        raise ContractError("physical sidecar camera inventory differs from CALVIN")
    return result


def _audit_exact_segment(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    segment: CalvinLanguageSegment,
    scene: str,
) -> tuple[dict[str, object], tuple[int, ...]]:
    relevance = calvin_task_physical_relevance(segment.task_key)
    if not relevance.exact_action_target or len(relevance.action_target_identity_keys) != 1:
        raise ContractError("exact behavior audit requires one reviewed direct target")
    target_identity_key = relevance.action_target_identity_keys[0]
    global_indices = tuple(range(segment.start, segment.end + 1))
    geometry = []
    tcp_world = []
    actions = []
    visible = {camera: [] for camera in _CAMERA_FIELDS}
    identity_keys: tuple[str, ...] | None = None
    for global_index in global_indices:
        physical = sidecar.source_frame(global_index)
        if identity_keys is None:
            identity_keys = physical.identity_keys
            validate_calvin_task_protocol_inventory(identity_keys)
        elif physical.identity_keys != identity_keys:
            raise ContractError("physical identity order changed inside a language segment")
        geometry.append(_denormalized_geometry(physical))
        frame_pixels = _target_visible_pixels(physical, target_identity_key)
        for camera_name in visible:
            visible[camera_name].append(frame_pixels[camera_name])

        source = index.validated_source_frame_arrays(
            global_index,
            fields=("scene_obs", "robot_obs", "rel_actions"),
        )
        if calvin_source_state_sha256(source["scene_obs"], source["robot_obs"]) != (
            sidecar.source_state_sha256(global_index)
        ):
            raise ContractError("behavior source state differs from physical sidecar provenance")
        tcp_world.append(np.asarray(source["robot_obs"][:3], dtype=np.float64))
        if global_index < segment.end:
            actions.append(np.asarray(source["rel_actions"], dtype=np.float64))
    if identity_keys is None:
        raise RuntimeError("CALVIN behavior segment unexpectedly contained no frames")

    summary = summarize_calvin_behavior_support(
        task_key=segment.task_key,
        target_identity_key=target_identity_key,
        global_indices=global_indices,
        identity_keys=identity_keys,
        geometry_robot_base_m=np.stack(geometry),
        tcp_position_world_m=np.stack(tcp_world),
        actions=np.stack(actions),
        visible_target_pixels=visible,
    )
    record = {
        "episode_index": segment.episode_index,
        "instruction": segment.instruction,
        "scene": scene,
        "segment_end": segment.end,
        "segment_index": segment.index,
        "segment_start": segment.start,
        "summary": summary.to_dict(),
        "task_key": segment.task_key,
        "transition_count": segment.transition_count,
    }
    return record, calvin_behavior_review_keyframes(summary)


def _load_review_frame(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    global_index: int,
    target_identity_key: str,
) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    physical = sidecar.source_frame(global_index)
    owner_index = physical.identity_keys.index(target_identity_key) + 1
    source = index.validated_source_frame_arrays(
        global_index,
        fields=tuple(_CAMERA_FIELDS.values()),
    )
    result = []
    cameras = {camera.camera_name: camera for camera in physical.cameras}
    for camera_name, source_field in _CAMERA_FIELDS.items():
        camera = cameras[camera_name]
        image = np.asarray(source[source_field], dtype=np.uint8)
        if source_array_sha256(source_field, image) != camera.source_rgb_sha256:
            raise ContractError("behavior review RGB differs from physical sidecar provenance")
        mask = (camera.owner_index == owner_index) & camera.owner_supervised
        result.append((camera_name, image, mask))
    return tuple(result)


def _target_overlay(image: np.ndarray, mask: np.ndarray) -> Image.Image:
    source = Image.fromarray(image).convert("RGBA")
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
    fill = Image.new("RGBA", source.size, (255, 0, 0, 105))
    overlay.alpha_composite(Image.composite(fill, Image.new("RGBA", source.size), mask_image))
    bounds = mask_image.getbbox()
    if bounds is not None:
        ImageDraw.Draw(overlay).rectangle(bounds, outline=(255, 255, 0, 255), width=2)
    return Image.alpha_composite(source, overlay).convert("RGB")


def _render_review(
    *,
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    segment: CalvinLanguageSegment,
    keyframes: tuple[int, ...],
    target_identity_key: str,
) -> bytes:
    tile_size = 260
    label_height = 36
    canvas_width = tile_size * len(keyframes)
    title_lines = textwrap.wrap(
        f"task={segment.task_key} | target={target_identity_key} | {segment.instruction}",
        width=max(24, canvas_width // 9),
        break_long_words=False,
    )
    header_height = 18 + 24 * len(title_lines)
    canvas = Image.new(
        "RGB",
        (canvas_width, header_height + 2 * (tile_size + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for line_index, line in enumerate(title_lines):
        draw.text((10, 8 + line_index * 24), line, fill="black", font=_font(16))
    for column, global_index in enumerate(keyframes):
        for row, (camera_name, image, mask) in enumerate(
            _load_review_frame(
                index=index,
                sidecar=sidecar,
                global_index=global_index,
                target_identity_key=target_identity_key,
            )
        ):
            rendered = _target_overlay(image, mask).resize(
                (tile_size, tile_size),
                resample=Image.Resampling.NEAREST,
            )
            x = column * tile_size
            y = header_height + row * (tile_size + label_height)
            canvas.paste(rendered, (x, y))
            visible = int(np.count_nonzero(mask))
            label = f"{camera_name} step={global_index} pixels={visible}"
            draw.rectangle((x, y, x + tile_size - 1, y + 23), fill=(0, 0, 0))
            draw.text((x + 5, y + 3), label, fill="white", font=_font(14))
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=False)
    return payload.getvalue()


def _numeric_distribution(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ContractError("behavior aggregate requires finite scalar observations")
    return {
        "count": int(len(array)),
        "maximum": float(array.max()),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "minimum": float(array.min()),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
    }


def _aggregate_exact_records(records: list[dict[str, object]]) -> dict[str, object]:
    metrics = defaultdict(list)
    rank_one = 0
    visible_both = 0
    visible_either = 0
    for record in records:
        summary = record["summary"]
        if not isinstance(summary, dict):
            raise RuntimeError("behavior record summary changed type")
        for name in (
            "target_net_displacement_m",
            "target_max_displacement_m",
            "minimum_tcp_target_distance_m",
            "initial_tcp_target_distance_m",
            "final_tcp_target_distance_m",
        ):
            metrics[name].append(float(summary[name]))
        rank_one += int(summary["target_motion_rank"] == 1)
        counts = summary["camera_visible_frame_counts"]
        if not isinstance(counts, dict):
            raise RuntimeError("behavior visibility summary changed type")
        visible_both += int(all(int(counts[name]) > 0 for name in _CAMERA_FIELDS))
        visible_either += int(any(int(counts[name]) > 0 for name in _CAMERA_FIELDS))
    return {
        "both_cameras_visible_segment_count": visible_both,
        "either_camera_visible_segment_count": visible_either,
        "metric_distributions": {
            name: _numeric_distribution(values) for name, values in sorted(metrics.items())
        },
        "segment_count": len(records),
        "target_motion_rank_one_count": rank_one,
    }


def _grouped_exact_aggregates(
    records: list[dict[str, object]],
    *,
    field: str,
) -> dict[str, object]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for record in records:
        value = record.get(field)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"behavior record {field} changed type")
        grouped[value].append(record)
    return {
        value: _aggregate_exact_records(group_records)
        for value, group_records in sorted(grouped.items())
    }


def _task_scene_exact_aggregates(
    records: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        task_key = record.get("task_key")
        scene = record.get("scene")
        if not isinstance(task_key, str) or not isinstance(scene, str):
            raise RuntimeError("behavior task/scene record changed type")
        grouped[task_key][scene].append(record)
    return {
        task_key: {
            scene: _aggregate_exact_records(scene_records)
            for scene, scene_records in sorted(scene_groups.items())
        }
        for task_key, scene_groups in sorted(grouped.items())
    }


def _segment_scene(
    segment: CalvinLanguageSegment,
    scene_ranges: tuple[CalvinSceneRange, ...],
) -> str:
    start_scene = scene_for_global_index(scene_ranges, segment.start)
    end_scene = scene_for_global_index(scene_ranges, segment.end)
    if start_scene != end_scene:
        raise ContractError("CALVIN language segment crosses scene ranges")
    return start_scene


def _publish_artifact_directory(
    *,
    output_dir: Path,
    report: dict[str, object],
    visual_payloads: list[tuple[str, bytes]],
) -> None:
    partial = output_dir.with_name(f".{output_dir.name}.partial-{os.getpid()}")
    partial.mkdir(parents=True, exist_ok=False)
    try:
        for filename, payload in visual_payloads:
            write_bytes_durable_exclusive(partial / filename, payload)
        write_bytes_durable_exclusive(partial / "report.json", _json_bytes(report))
        publish_prepared_directory_durable_exclusive(partial, output_dir)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def _run(args: argparse.Namespace) -> dict[str, object]:
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.lingbot_native.runtime_provenance import (
        revision_bound_python_source_tree_contract,
    )

    split_root = args.split_root.resolve()
    manifest_path = args.dataset_manifest.resolve()
    source_manifest_path = args.source_dataset_manifest.resolve()
    source_receipt_path = args.source_receipt.resolve()
    sidecar_root = args.sidecar_root.resolve()
    sidecar_manifest = args.sidecar_manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    _validate_expected_full_dataset_identity(
        args.expected_dataset_id,
        args.expected_dataset_revision,
    )
    runtime_python_tree = revision_bound_python_source_tree_contract(
        repo_root=_REPO_ROOT,
        revision=args.picf_code_revision,
        roots=_RUNTIME_SOURCE_ROOTS,
    )
    protocol_source_sha256 = validate_calvin_task_protocol_sources(args.calvin_source_checkout)

    manifest_sha256 = file_sha256(manifest_path)
    source_manifest_sha256 = file_sha256(source_manifest_path)
    manifest = load_dataset_file_manifest(manifest_path)
    source_manifest = load_dataset_file_manifest(source_manifest_path)
    if (
        file_sha256(manifest_path) != manifest_sha256
        or file_sha256(source_manifest_path) != source_manifest_sha256
    ):
        raise ContractError("CALVIN dataset manifest changed while loading")
    validate_calvin_content_identity_migration(source_manifest, manifest)
    source_receipt, source_receipt_sha256 = _load_source_receipt(
        source_receipt_path,
        args.source_receipt_sha256,
    )
    validate_calvin_official_source_receipt(
        source_receipt,
        source_manifest=source_manifest,
        source_manifest_sha256=source_manifest_sha256,
        target_manifest=manifest,
        target_manifest_sha256=manifest_sha256,
    )
    expected_identity = (args.expected_dataset_id, args.expected_dataset_revision, split_root.name)
    actual_identity = (manifest.dataset_id, manifest.dataset_revision, manifest.split_name)
    provenance_matches = actual_identity == expected_identity
    if not provenance_matches:
        raise ContractError("dataset identity differs from the expected official CALVIN source")
    runtime_binding = validate_dataset_runtime_binding(
        manifest,
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=manifest.split_name,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    protocol = calvin_task_physical_relevance_inventory()
    protocol_keys = tuple(item.task_key for item in protocol)
    census = Counter(segment.task_key for segment in index.segments)
    if set(census) != set(protocol_keys):
        raise ContractError("CALVIN language task inventory differs from the frozen protocol")
    scene_ranges = load_calvin_scene_ranges(split_root, dataset_manifest=manifest)
    scene_by_segment_index = {
        segment.index: _segment_scene(segment, scene_ranges) for segment in index.segments
    }
    task_scene_census = Counter(
        (segment.task_key, scene_by_segment_index[segment.index]) for segment in index.segments
    )

    print(json.dumps({"event": "sidecar_full_hash_verification_started"}), flush=True)
    sidecar = CalvinPhysicalSupervisionSidecar(
        sidecar_root,
        index,
        manifest_path=sidecar_manifest,
        expected_manifest_sha256=args.sidecar_manifest_sha256,
    )
    print(
        json.dumps(
            {
                "event": "sidecar_full_hash_verification_finished",
                "manifest_sha256": sidecar.manifest_sha256,
                "shard_count": len(sidecar.shards),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    selected = select_calvin_behavior_segments(
        index.segments,
        samples_per_task_scene=args.samples_per_task_scene,
        scene_by_segment_index=scene_by_segment_index,
    )
    selected_by_task_scene: dict[tuple[str, str], list[CalvinLanguageSegment]] = defaultdict(list)
    for segment in selected:
        scene = scene_by_segment_index[segment.index]
        selected_by_task_scene[(segment.task_key, scene)].append(segment)

    exact_records: list[dict[str, object]] = []
    ambiguous_records: list[dict[str, object]] = []
    visuals = []
    visual_payloads: list[tuple[str, bytes]] = []
    for ordinal, segment in enumerate(selected, start=1):
        relevance = calvin_task_physical_relevance(segment.task_key)
        scene = scene_by_segment_index[segment.index]
        if not relevance.exact_action_target:
            ambiguous_records.append(
                {
                    "episode_index": segment.episode_index,
                    "exclusion_reason": relevance.exclusion_reason,
                    "instruction": segment.instruction,
                    "known_participant_identity_keys": list(
                        relevance.known_participant_identity_keys
                    ),
                    "segment_end": segment.end,
                    "segment_index": segment.index,
                    "segment_start": segment.start,
                    "scene": scene,
                    "task_key": segment.task_key,
                    "training_target_generated": False,
                }
            )
        else:
            record, keyframes = _audit_exact_segment(
                index=index,
                sidecar=sidecar,
                segment=segment,
                scene=scene,
            )
            exact_records.append(record)
            task_scene_segments = selected_by_task_scene[(segment.task_key, scene)]
            render_segment = task_scene_segments[len(task_scene_segments) // 2]
            if segment.index == render_segment.index:
                target = relevance.action_target_identity_keys[0]
                payload = _render_review(
                    index=index,
                    sidecar=sidecar,
                    segment=segment,
                    keyframes=keyframes,
                    target_identity_key=target,
                )
                filename = (
                    f"task-{_slug(segment.task_key)}-scene-{_slug(scene)}-"
                    f"segment-{segment.index:05d}.png"
                )
                visual_payloads.append((filename, payload))
                visuals.append(
                    {
                        "file": filename,
                        "global_indices": list(keyframes),
                        "instruction": segment.instruction,
                        "scene": scene,
                        "segment_index": segment.index,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "target_identity_key": target,
                        "task_key": segment.task_key,
                    }
                )
        print(
            json.dumps(
                {
                    "completed_selected_segments": ordinal,
                    "event": "behavior_segment_audited",
                    "scene": scene,
                    "segment_index": segment.index,
                    "task_key": segment.task_key,
                    "total_selected_segments": len(selected),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    report = {
        "ambiguous_task_records": ambiguous_records,
        "audit_schema": AUDIT_SCHEMA,
        "behavior_support_aggregate": _aggregate_exact_records(exact_records),
        "behavior_support_by_scene": _grouped_exact_aggregates(
            exact_records,
            field="scene",
        ),
        "behavior_support_by_task": _grouped_exact_aggregates(
            exact_records,
            field="task_key",
        ),
        "behavior_support_by_task_scene": _task_scene_exact_aggregates(exact_records),
        "dataset": {
            "actual_identity": {
                "dataset_id": manifest.dataset_id,
                "dataset_revision": manifest.dataset_revision,
                "split_name": manifest.split_name,
            },
            "expected_identity": {
                "dataset_id": args.expected_dataset_id,
                "dataset_revision": args.expected_dataset_revision,
                "split_name": split_root.name,
            },
            "file_count": len(manifest.files),
            "language_segment_count": len(index.segments),
            "manifest_sha256": file_sha256(manifest_path),
            "provenance_gate_passed": provenance_matches,
            "runtime_binding": runtime_binding,
            "source_episode_count": len(index.episodes),
            "task_segment_census": dict(sorted(census.items())),
            "task_scene_segment_census": {
                task_key: {
                    scene: task_scene_census[(task_key, scene)]
                    for scene in sorted({item.scene for item in scene_ranges})
                }
                for task_key in sorted(census)
            },
            "total_size_bytes": manifest.total_size_bytes,
            "tree_sha256": manifest.tree_sha256,
        },
        "diagnostic_only": True,
        "exact_task_records": exact_records,
        "limitations": [
            "AABB-centre translation cannot certify rotations or full task success.",
            "Task semantics are used only after source-stratified selection.",
            "This report does not evaluate a model or authorize a checkpoint.",
        ],
        "physical_sidecar": {
            "coverage": sidecar.coverage,
            "manifest_sha256": sidecar.manifest_sha256,
            "shard_count": len(sidecar.shards),
        },
        "picf_code_revision": args.picf_code_revision,
        "protocol": {
            "ambiguous_task_count": sum(not item.exact_action_target for item in protocol),
            "exact_task_count": sum(item.exact_action_target for item in protocol),
            "robot_base_euler_rad": list(CALVIN_ROBOT_BASE_EULER_RAD),
            "robot_base_position_m": list(CALVIN_ROBOT_BASE_POSITION_M),
            "scene_config_sha256": dict(CALVIN_SCENE_CONFIG_SHA256),
            "source_sha256": protocol_source_sha256,
            "task_count": len(protocol),
        },
        "render_selection_algorithm": RENDER_SELECTION_ALGORITHM,
        "runtime_python_tree": runtime_python_tree,
        "source_provenance": {
            "source_manifest_sha256": source_manifest_sha256,
            "source_receipt_sha256": source_receipt_sha256,
        },
        "samples_per_task_scene": args.samples_per_task_scene,
        "selection_algorithm": SELECTION_ALGORITHM,
        "selection_consults_model_output": False,
        "selection_consults_physical_target_or_pixels": False,
        "task_success_certified": False,
        "training_authorized": False,
        "visuals": visuals,
    }
    if (
        revision_bound_python_source_tree_contract(
            repo_root=_REPO_ROOT,
            revision=args.picf_code_revision,
            roots=_RUNTIME_SOURCE_ROOTS,
        )
        != runtime_python_tree
    ):
        raise ContractError("behavior audit runtime source changed during execution")
    if validate_calvin_task_protocol_sources(args.calvin_source_checkout) != (
        protocol_source_sha256
    ):
        raise ContractError("CALVIN task protocol source changed during behavior audit")
    if (
        file_sha256(manifest_path) != manifest_sha256
        or file_sha256(source_manifest_path) != source_manifest_sha256
        or file_sha256(source_receipt_path) != source_receipt_sha256
    ):
        raise ContractError("CALVIN provenance input changed during behavior audit")
    _publish_artifact_directory(
        output_dir=output_dir,
        report=report,
        visual_payloads=visual_payloads,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-dataset-manifest", required=True, type=Path)
    parser.add_argument("--source-receipt", required=True, type=Path)
    parser.add_argument("--source-receipt-sha256", required=True)
    parser.add_argument("--sidecar-root", required=True, type=Path)
    parser.add_argument("--sidecar-manifest", required=True, type=Path)
    parser.add_argument("--sidecar-manifest-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--calvin-source-checkout", required=True, type=Path)
    parser.add_argument("--expected-dataset-id", required=True)
    parser.add_argument("--expected-dataset-revision", required=True)
    parser.add_argument("--samples-per-task-scene", type=int, default=1)
    args = parser.parse_args()
    report = _run(args)
    exact_records = report.get("exact_task_records")
    dataset = report.get("dataset")
    visuals = report.get("visuals")
    provenance_gate_passed = (
        dataset.get("provenance_gate_passed") if isinstance(dataset, dict) else None
    )
    if (
        not isinstance(exact_records, list)
        or not isinstance(dataset, dict)
        or not isinstance(visuals, list)
        or not isinstance(provenance_gate_passed, bool)
    ):
        raise RuntimeError("behavior audit report changed summary field types")
    print(
        json.dumps(
            {
                "event": "calvin_behavior_grounding_support_audit_complete",
                "exact_segment_count": len(exact_records),
                "output_dir": str(args.output_dir.resolve()),
                "provenance_gate_passed": provenance_gate_passed,
                "training_authorized": False,
                "visual_count": len(visuals),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

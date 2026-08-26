#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Audit whether LingBot future targets contain change beyond current-frame copies."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import re
import textwrap
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="predictive temporal audit",
)

import numpy as np
import torch
from PIL import Image, ImageDraw

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.calvin_geometry_schema import sha256_file
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.current_grid_cache import (
    CurrentGridCacheRecord,
    LingBotCurrentGridTargetCache,
)
from picf_next.lingbot_native.predictive_cache import (
    LingBotPredictiveTargetCache,
    PredictiveObjectCacheRecord,
    pool_dino_object_summaries,
)
from picf_next.lingbot_native.predictive_diagnostics import (
    PREDICTIVE_TEMPORAL_AUDIT_SCHEMA,
    PREDICTIVE_TEMPORAL_FEATURE_PAIRING,
    predictive_latent_diagnostics,
    predictive_target_pretraining_readiness,
    predictive_temporal_diagnostics,
    predictive_temporal_pretraining_readiness,
    predictive_visible_support_diagnostics,
)

_SAMPLE_SCHEMA = b"picf-next.lingbot-predictive-temporal-sample/v1\0"
_CURRENT_SAMPLE_SCHEMA = b"picf-next.lingbot-current-correction-target-sample/v1\0"
_VISUAL_AUDIT_SCHEMA = "picf-next.lingbot-teacher-cache-visual-audit/v3"
_VISUAL_SELECTION = (
    "per-observed-horizon:first,temporal-median,last,minimum-total-visible-importance/v1"
)


def _color(key: str) -> tuple[int, int, int]:
    raw = hashlib.blake2b(key.encode("utf-8"), digest_size=3).digest()
    return (
        int(64 + raw[0] % 176),
        int(64 + raw[1] % 176),
        int(64 + raw[2] % 176),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")[:72] or "unknown"


def _owner_overlay(
    rgb: np.ndarray,
    frame: CalvinPhysicalSupervisionFrame,
) -> np.ndarray:
    cameras = tuple(camera for camera in frame.cameras if camera.camera_name == "static")
    if len(cameras) != 1:
        raise ContractError("teacher-cache visual audit requires one static owner raster")
    camera = cameras[0]
    owner = camera.owner_index
    supervised = camera.owner_supervised
    if rgb.shape[:2] != owner.shape or supervised.shape != owner.shape:
        raise ContractError("teacher-cache RGB and physical owner geometry differ")
    output = rgb.astype(np.float32).copy()
    for owner_index, identity in enumerate(frame.identity_keys, start=1):
        mask = (owner == owner_index) & supervised
        if mask.any():
            color = np.asarray(_color(identity), dtype=np.float32)
            output[mask] = 0.4 * output[mask] + 0.6 * color
    unknown = ~supervised
    if unknown.any():
        y, x = np.indices(owner.shape)
        magenta = unknown & (((y // 4) + (x // 4)) % 2 == 0)
        output[magenta] = np.asarray((255, 0, 255), dtype=np.float32)
        output[unknown & ~magenta] *= 0.15
    return np.clip(output, 0, 255).astype(np.uint8)


def _feature_rgb(features: np.ndarray) -> np.ndarray:
    """Show three fixed teacher dimensions without fitting a visual projection."""

    if (
        features.dtype != np.float16
        or features.ndim != 2
        or features.shape[0] <= 0
        or features.shape[1] < 3
        or not np.isfinite(features).all()
    ):
        raise ContractError("teacher-cache visual features have invalid geometry")
    side = int(round(features.shape[0] ** 0.5))
    if side * side != features.shape[0]:
        raise ContractError("teacher-cache visual features are not one square patch grid")
    channels = features[:, :3].astype(np.float32)
    low = np.quantile(channels, 0.02, axis=0)
    high = np.quantile(channels, 0.98, axis=0)
    scale = np.maximum(high - low, np.finfo(np.float32).eps)
    normalized = np.clip((channels - low) / scale, 0.0, 1.0)
    return np.rint(normalized.reshape(side, side, 3) * 255.0).astype(np.uint8)


def _importance_panel(
    record: PredictiveObjectCacheRecord,
    *,
    width: int,
    height: int,
) -> Image.Image:
    panel = Image.new("RGB", (width, height), (18, 18, 18))
    draw = ImageDraw.Draw(panel)
    draw.text((7, 6), "future object rows: visible importance", fill=(255, 255, 255))
    count = len(record.identity_keys)
    row_height = max(18, min(26, (height - 34) // max(count, 1)))
    for row, (identity, importance) in enumerate(
        zip(record.identity_keys, record.importance.tolist(), strict=True)
    ):
        y = 28 + row * row_height
        if y + row_height > height:
            break
        color = _color(identity)
        bar_start = width // 2
        bar_stop = bar_start + round(float(importance) * (width - bar_start - 8))
        draw.text((7, y), identity[:38], fill=color)
        draw.rectangle((bar_start, y + 2, width - 8, y + 12), outline=(92, 92, 92))
        if bar_stop > bar_start:
            draw.rectangle((bar_start, y + 2, bar_stop, y + 12), fill=color)
        draw.text((bar_start, y + 13), f"{float(importance):.5f}", fill=(210, 210, 210))
    return panel


def _tile(image: np.ndarray, *, size: tuple[int, int], nearest: bool = False) -> Image.Image:
    resample = Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR
    return Image.fromarray(image).resize(size, resample=resample)


def _render_visual_panel(
    *,
    source_rgb: np.ndarray,
    source_frame: CalvinPhysicalSupervisionFrame,
    target_rgb: np.ndarray,
    target_frame: CalvinPhysicalSupervisionFrame,
    current_record: CurrentGridCacheRecord,
    future_record: PredictiveObjectCacheRecord,
    task_keys: tuple[str, ...],
    instructions: tuple[str, ...],
    selection_reasons: tuple[str, ...],
) -> Image.Image:
    if not task_keys or any(not value for value in task_keys):
        raise ValueError("teacher-cache visual audit requires task keys")
    if not instructions or any(not value for value in instructions):
        raise ValueError("teacher-cache visual audit requires task instructions")
    tile = (320, 320)
    task_text = " | ".join(task_keys)
    instruction_text = " | ".join(instructions)
    title = (
        f"source={future_record.source_global_index} target={future_record.target_global_index} "
        f"h={future_record.horizon} tasks={task_text} | prompts={instruction_text}"
    )
    legend = (
        f"selection={','.join(selection_reasons)}; cache-only diagnostic, "
        f"matching_tasks={len(task_keys)}, matching_prompts={len(instructions)}; "
        "not model input and not learned-anchor evidence"
    )
    title_lines = textwrap.wrap(title, width=138, break_long_words=False)
    legend_lines = textwrap.wrap(legend, width=138, break_long_words=False)
    header = 12 + 15 * (len(title_lines) + len(legend_lines)) + 8
    canvas = Image.new("RGB", (3 * tile[0], header + 2 * tile[1]), (8, 8, 8))
    draw = ImageDraw.Draw(canvas)
    y = 7
    for line in (*title_lines, *legend_lines):
        fill = (255, 255, 255) if y < 7 + 15 * len(title_lines) else (205, 205, 205)
        draw.text((8, y), line, fill=fill)
        y += 15
    panels = (
        _tile(source_rgb, size=tile),
        _tile(_owner_overlay(source_rgb, source_frame), size=tile, nearest=True),
        _tile(_feature_rgb(current_record.features), size=tile, nearest=True),
        _tile(target_rgb, size=tile),
        _tile(_owner_overlay(target_rgb, target_frame), size=tile, nearest=True),
        _importance_panel(future_record, width=tile[0], height=tile[1]),
    )
    labels = (
        "source RGB",
        "source physical owners / checker=unknown",
        "current DINO fixed dims 0/1/2",
        "target RGB",
        "target physical owners / checker=unknown",
        "future pooled object rows",
    )
    for index, (panel, label) in enumerate(zip(panels, labels, strict=True)):
        row, column = divmod(index, 3)
        x = column * tile[0]
        panel_y = header + row * tile[1]
        canvas.paste(panel, (x, panel_y))
        draw.rectangle((x, panel_y, x + 284, panel_y + 17), fill=(0, 0, 0))
        draw.text((x + 4, panel_y + 3), label, fill=(255, 255, 255))
    return canvas


def _select_visual_records(
    records: Iterable[PredictiveObjectCacheRecord],
    *,
    declared_horizons: tuple[int, ...],
) -> tuple[tuple[PredictiveObjectCacheRecord, tuple[str, ...]], ...]:
    grouped: dict[int, list[PredictiveObjectCacheRecord]] = defaultdict(list)
    for record in records:
        if record.horizon not in declared_horizons:
            raise ContractError("visual audit found an undeclared predictive horizon")
        grouped[record.horizon].append(record)
    selected: dict[tuple[int, int], tuple[PredictiveObjectCacheRecord, set[str]]] = {}
    for horizon in declared_horizons:
        group = grouped.get(horizon, [])
        if not group:
            continue
        if any(
            left.source_global_index >= right.source_global_index
            for left, right in zip(group, group[1:], strict=False)
        ):
            raise ContractError("visual audit predictive rows are not source ordered")
        choices = (
            ("first", group[0]),
            ("temporal_median", group[(len(group) - 1) // 2]),
            ("last", group[-1]),
            (
                "minimum_total_visible_importance",
                min(
                    group,
                    key=lambda value: (
                        float(value.importance.sum()),
                        value.source_global_index,
                    ),
                ),
            ),
        )
        for reason, record in choices:
            key = record.source_global_index, record.horizon
            if key not in selected:
                selected[key] = (record, set())
            selected[key][1].add(reason)
    return tuple(
        (record, tuple(sorted(reasons))) for _key, (record, reasons) in sorted(selected.items())
    )


def _sample_priority(*, source: int, horizon: int, target_digest: str, identity: str) -> int:
    payload = (
        _SAMPLE_SCHEMA
        + source.to_bytes(8, byteorder="big", signed=False)
        + horizon.to_bytes(8, byteorder="big", signed=False)
        + target_digest.encode("ascii")
        + b"\0"
        + identity.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big", signed=False)


def _current_sample_priority(*, source: int, source_digest: str, identity: str) -> int:
    payload = (
        _CURRENT_SAMPLE_SCHEMA
        + source.to_bytes(8, byteorder="big", signed=False)
        + source_digest.encode("ascii")
        + b"\0"
        + identity.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big", signed=False)


def audit_predictive_temporal_content(
    predictive_cache: LingBotPredictiveTargetCache,
    current_cache: LingBotCurrentGridTargetCache,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    *,
    maximum_samples: int,
) -> dict[str, object]:
    """Scan the cache intersection and compare aligned current/future object features."""

    if not isinstance(predictive_cache, LingBotPredictiveTargetCache) or not isinstance(
        current_cache, LingBotCurrentGridTargetCache
    ):
        raise TypeError("temporal target audit requires typed current and future caches")
    if not isinstance(physical_sidecar, CalvinPhysicalSupervisionSidecar):
        raise TypeError("temporal target audit requires a verified physical sidecar")
    if (
        isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples < 2
    ):
        raise ValueError("temporal target audit requires at least two samples")
    future_contract = predictive_cache.contract
    current_contract = current_cache.contract
    shared_contract = (
        future_contract.dataset_tree_sha256 == current_contract.dataset_tree_sha256,
        future_contract.physical_sidecar_manifest_sha256
        == current_contract.physical_sidecar_manifest_sha256
        == physical_sidecar.manifest_sha256,
        future_contract.lingbot_source_commit == current_contract.lingbot_source_commit,
        future_contract.lingbot_checkpoint_revision == current_contract.lingbot_checkpoint_revision,
        future_contract.teacher_config_sha256 == current_contract.teacher_config_sha256,
        future_contract.teacher_checkpoint_sha256 == current_contract.teacher_checkpoint_sha256,
        future_contract.hidden_size == current_contract.hidden_size,
        future_contract.input_size == current_contract.input_size,
        future_contract.patch_tokens == current_contract.patch_tokens,
        future_contract.camera_name == current_contract.camera_name == "static",
    )
    if not all(shared_contract):
        raise ContractError("current/future temporal audit provenance or geometry differs")

    retained: list[tuple[int, int, str, int, np.ndarray, np.ndarray]] = []
    retained_current: list[tuple[int, int, str, str, float, np.ndarray]] = []
    serial = 0
    current_serial = 0
    scanned_current_records = 0
    scanned_current_objects = 0
    supported_current_objects = 0
    current_identity_support: Counter[str] = Counter()
    total_current_importance = 0.0
    minimum_current_importance = float("inf")
    maximum_current_importance = 0.0
    matched_future_records = 0
    supported_pairs = 0
    horizon_pairs: Counter[int] = Counter()
    selection_digest = hashlib.sha256()
    for current_record in current_cache.iter_records():
        scanned_current_records += 1
        frame = physical_sidecar.source_frame(current_record.source_global_index)
        cameras = tuple(
            camera for camera in frame.cameras if camera.camera_name == current_contract.camera_name
        )
        if len(cameras) != 1 or cameras[0].source_rgb_sha256 != current_record.source_rgb_sha256:
            raise ContractError("current temporal target differs from its visible-owner raster")
        current_features, current_importance = pool_dino_object_summaries(
            torch.from_numpy(current_record.features.copy()).float(),
            owner_index=cameras[0].owner_index,
            owner_supervised=cameras[0].owner_supervised,
            identity_keys=frame.identity_keys,
            minimum_visible_fraction=future_contract.minimum_visible_fraction,
            input_size=future_contract.input_size,
        )
        for object_index, identity in enumerate(frame.identity_keys):
            scanned_current_objects += 1
            importance = float(current_importance[object_index])
            if importance <= 0:
                continue
            supported_current_objects += 1
            current_identity_support[identity] += 1
            total_current_importance += importance
            minimum_current_importance = min(minimum_current_importance, importance)
            maximum_current_importance = max(maximum_current_importance, importance)
            priority = _current_sample_priority(
                source=current_record.source_global_index,
                source_digest=current_record.source_rgb_sha256,
                identity=identity,
            )
            current_item = (
                -priority,
                current_serial,
                identity,
                current_record.source_rgb_sha256,
                importance,
                current_features[object_index].astype(np.float32, copy=True),
            )
            current_serial += 1
            if len(retained_current) < maximum_samples:
                heapq.heappush(retained_current, current_item)
            elif current_item[0] > retained_current[0][0]:
                heapq.heapreplace(retained_current, current_item)
        current_positions = {identity: index for index, identity in enumerate(frame.identity_keys)}
        for horizon in future_contract.horizons:
            future_record = predictive_cache.record_for(
                source_global_index=current_record.source_global_index,
                horizon=horizon,
            )
            if future_record is None:
                continue
            matched_future_records += 1
            if future_record.source_rgb_sha256 != current_record.source_rgb_sha256:
                raise ContractError("current/future temporal cache source RGB differs")
            for future_index, identity in enumerate(future_record.identity_keys):
                current_index = current_positions.get(identity)
                if current_index is None:
                    continue
                if (
                    float(current_importance[current_index]) <= 0
                    or float(future_record.importance[future_index]) <= 0
                ):
                    continue
                supported_pairs += 1
                horizon_pairs[horizon] += 1
                priority = _sample_priority(
                    source=current_record.source_global_index,
                    horizon=horizon,
                    target_digest=future_record.target_rgb_sha256,
                    identity=identity,
                )
                item = (
                    -priority,
                    serial,
                    identity,
                    horizon,
                    current_features[current_index].astype(np.float32, copy=True),
                    future_record.features[future_index].astype(np.float32, copy=True),
                )
                serial += 1
                if len(retained) < maximum_samples:
                    heapq.heappush(retained, item)
                elif item[0] > retained[0][0]:
                    heapq.heapreplace(retained, item)

    if scanned_current_records != current_contract.expected_record_count:
        raise RuntimeError("temporal target audit did not scan complete current-cache coverage")
    if supported_current_objects < 2 or len(retained_current) < 2:
        raise RuntimeError("current-correction cache has fewer than two supported object targets")
    if matched_future_records <= 0 or supported_pairs < 2 or len(retained) < 2:
        raise RuntimeError("temporal target audit found fewer than two supported aligned pairs")

    ordered_current = sorted(retained_current, key=lambda value: (-value[0], value[1]))
    current_identities = tuple(value[2] for value in ordered_current)
    current_groups = tuple(value[3] for value in ordered_current)
    current_target_features = torch.from_numpy(
        np.stack(tuple(value[5] for value in ordered_current))
    )
    current_selection_digest = hashlib.sha256()
    for negative_priority, _serial, identity, group, _importance, _feature in ordered_current:
        current_selection_digest.update((-negative_priority).to_bytes(32, byteorder="big"))
        current_selection_digest.update(identity.encode("utf-8") + b"\0")
        current_selection_digest.update(group.encode("ascii") + b"\0")
    current_diagnostics = predictive_latent_diagnostics(
        current_target_features,
        identity_keys=current_identities,
        target_group_keys=current_groups,
    )
    current_ready, current_failures = predictive_target_pretraining_readiness(current_diagnostics)
    current_support_diagnostics = predictive_visible_support_diagnostics(
        torch.tensor(tuple(value[4] for value in ordered_current), dtype=torch.float32),
        supported_count=supported_current_objects,
        total_importance=total_current_importance,
        minimum_importance=minimum_current_importance,
        maximum_importance=maximum_current_importance,
    )

    ordered = sorted(retained, key=lambda value: (-value[0], value[1]))
    identities = tuple(value[2] for value in ordered)
    horizons = tuple(value[3] for value in ordered)
    current = torch.from_numpy(np.stack(tuple(value[4] for value in ordered)))
    future = torch.from_numpy(np.stack(tuple(value[5] for value in ordered)))
    for negative_priority, _serial, identity, horizon, _current, _future in ordered:
        selection_digest.update((-negative_priority).to_bytes(32, byteorder="big"))
        selection_digest.update(identity.encode("utf-8") + b"\0")
        selection_digest.update(horizon.to_bytes(8, byteorder="big", signed=False))
    diagnostics = predictive_temporal_diagnostics(
        current,
        future,
        identity_keys=identities,
        horizons=horizons,
    )
    ready, failures = predictive_temporal_pretraining_readiness(diagnostics)
    aggregate_ready = current_ready and ready
    aggregate_failures = tuple(
        [f"current_correction:{failure}" for failure in current_failures]
        + [f"controlled_future:{failure}" for failure in failures]
    )
    return {
        "current_cache_manifest_sha256": current_cache.manifest_sha256,
        "current_correction_diagnostics": current_diagnostics.as_dict(),
        "current_correction_identity_count": len(current_identity_support),
        "current_correction_sample_selection_sha256": current_selection_digest.hexdigest(),
        "current_correction_sampled_target_count": len(ordered_current),
        "current_correction_scanned_object_target_count": scanned_current_objects,
        "current_correction_supported_object_target_count": supported_current_objects,
        "current_correction_visible_support_diagnostics": (current_support_diagnostics.as_dict()),
        "current_correction_zero_support_object_target_count": (
            scanned_current_objects - supported_current_objects
        ),
        "diagnostics": diagnostics.as_dict(),
        "current_encoder_digest": current_contract.encoder_digest,
        "feature_pairing": PREDICTIVE_TEMPORAL_FEATURE_PAIRING,
        "future_cache_manifest_sha256": predictive_cache.manifest_sha256,
        "future_encoder_digest": future_contract.encoder_digest,
        "horizon_supported_pair_counts": {
            str(horizon): horizon_pairs[horizon] for horizon in future_contract.horizons
        },
        "interpretation": {
            "controlled_future_temporal_pretraining_readiness": ("PASS" if ready else "FAIL"),
            "controlled_future_temporal_pretraining_readiness_failures": list(failures),
            "current_correction_pretraining_readiness": ("PASS" if current_ready else "FAIL"),
            "current_correction_pretraining_readiness_failures": list(current_failures),
            "pretraining_readiness": "PASS" if aggregate_ready else "FAIL",
            "pretraining_readiness_failures": list(aggregate_failures),
            "scientific_acceptance": False,
            "scientific_acceptance_reason": (
                "target-bank statistics do not establish source-conditioned prediction, "
                "action conditioning or action benefit"
            ),
        },
        "matched_future_record_count": matched_future_records,
        "maximum_samples": maximum_samples,
        "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
        "sample_selection": "lowest-sha256-priority-without-replacement/v1",
        "sample_selection_sha256": selection_digest.hexdigest(),
        "sampled_pair_count": len(ordered),
        "scanned_current_record_count": scanned_current_records,
        "schema": PREDICTIVE_TEMPORAL_AUDIT_SCHEMA,
        "supported_aligned_pair_count": supported_pairs,
    }


def _write_json_durable(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    write_bytes_durable_exclusive(path, payload)


def _matching_task_annotations(
    index: CalvinDatasetIndex,
    *,
    source_global_index: int,
    target_global_index: int,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    matches = tuple(
        segment
        for segment in index.segments
        if segment.start <= source_global_index < target_global_index <= segment.end
    )
    if not matches:
        raise ContractError("teacher-cache visual pair is not inside one language segment")
    task_keys = tuple(sorted({segment.task_key for segment in matches}))
    annotations = tuple(
        sorted(
            {
                (int(segment.start), int(segment.end), str(segment.instruction))
                for segment in matches
            }
        )
    )
    instructions = tuple(sorted({instruction for _start, _end, instruction in annotations}))
    if not task_keys or not annotations or not instructions:
        raise ContractError("teacher-cache visual task annotations are empty")
    return task_keys, instructions, annotations


def _static_rgb_and_frame(
    index: CalvinDatasetIndex,
    sidecar: CalvinPhysicalSupervisionSidecar,
    *,
    global_index: int,
    expected_sha256: str,
) -> tuple[np.ndarray, CalvinPhysicalSupervisionFrame]:
    source = index.validated_source_frame_arrays(global_index, fields=("rgb_static",))
    rgb = np.asarray(source["rgb_static"], dtype=np.uint8)
    actual_sha256 = source_array_sha256("rgb_static", rgb)
    if actual_sha256 != expected_sha256:
        raise ContractError("teacher-cache visual RGB differs from cache content hash")
    frame = sidecar.source_frame(global_index)
    cameras = tuple(camera for camera in frame.cameras if camera.camera_name == "static")
    if len(cameras) != 1 or cameras[0].source_rgb_sha256 != actual_sha256:
        raise ContractError("teacher-cache visual RGB differs from physical sidecar hash")
    return rgb, frame


def render_predictive_temporal_visual_audit(
    *,
    output_dir: Path,
    index: CalvinDatasetIndex,
    physical_sidecar: CalvinPhysicalSupervisionSidecar,
    predictive_cache: LingBotPredictiveTargetCache,
    current_cache: LingBotCurrentGridTargetCache,
    temporal_audit_sha256: str,
) -> dict[str, object]:
    """Render deterministic loss-side cache evidence without changing the report ABI."""

    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if len(temporal_audit_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in temporal_audit_sha256
    ):
        raise ValueError("temporal audit SHA-256 is invalid")
    if predictive_cache.contract.dataset_tree_sha256 != current_cache.contract.dataset_tree_sha256:
        raise ContractError("teacher-cache visual current/future dataset identity differs")
    output_dir.mkdir(parents=True)
    records: list[dict[str, object]] = []
    selections = _select_visual_records(
        predictive_cache.iter_records(),
        declared_horizons=predictive_cache.contract.horizons,
    )
    for future_record, reasons in selections:
        current_record = current_cache.record_for(
            source_global_index=future_record.source_global_index
        )
        if current_record is None:
            raise ContractError("teacher-cache visual source is absent from current cache")
        if current_record.source_rgb_sha256 != future_record.source_rgb_sha256:
            raise ContractError("teacher-cache visual current/future source hashes differ")
        source_rgb, source_frame = _static_rgb_and_frame(
            index,
            physical_sidecar,
            global_index=future_record.source_global_index,
            expected_sha256=future_record.source_rgb_sha256,
        )
        target_rgb, target_frame = _static_rgb_and_frame(
            index,
            physical_sidecar,
            global_index=future_record.target_global_index,
            expected_sha256=future_record.target_rgb_sha256,
        )
        if source_frame.identity_keys != future_record.identity_keys:
            raise ContractError("teacher-cache visual source identities differ from future rows")
        if target_frame.identity_keys != future_record.identity_keys:
            raise ContractError("teacher-cache visual target identities differ from future rows")
        task_keys, instructions, matching_segments = _matching_task_annotations(
            index,
            source_global_index=future_record.source_global_index,
            target_global_index=future_record.target_global_index,
        )
        task_slug = "__".join(task_keys)
        filename = (
            f"h{future_record.horizon:02d}_source{future_record.source_global_index:07d}_"
            f"target{future_record.target_global_index:07d}_{_slug(task_slug)}.png"
        )
        path = output_dir / filename
        _render_visual_panel(
            source_rgb=source_rgb,
            source_frame=source_frame,
            target_rgb=target_rgb,
            target_frame=target_frame,
            current_record=current_record,
            future_record=future_record,
            task_keys=task_keys,
            instructions=instructions,
            selection_reasons=reasons,
        ).save(path, format="PNG", optimize=False)
        records.append(
            {
                "current_feature_rms": float(
                    np.sqrt(np.mean(np.square(current_record.features.astype(np.float32))))
                ),
                "future_identity_keys": list(future_record.identity_keys),
                "future_importance": [float(value) for value in future_record.importance.tolist()],
                "horizon": future_record.horizon,
                "instructions": list(instructions),
                "matching_segments": [
                    {
                        "end": end,
                        "instruction": instruction,
                        "start": start,
                    }
                    for start, end, instruction in matching_segments
                ],
                "panel": filename,
                "panel_sha256": sha256_file(path),
                "selection_reasons": list(reasons),
                "source_global_index": future_record.source_global_index,
                "source_rgb_sha256": future_record.source_rgb_sha256,
                "target_global_index": future_record.target_global_index,
                "target_rgb_sha256": future_record.target_rgb_sha256,
                "task_keys": list(task_keys),
            }
        )
    manifest = {
        "current_cache_manifest_sha256": current_cache.manifest_sha256,
        "dataset_tree_sha256": current_cache.contract.dataset_tree_sha256,
        "declared_horizons": list(predictive_cache.contract.horizons),
        "future_cache_manifest_sha256": predictive_cache.manifest_sha256,
        "learned_anchor_evidence": False,
        "loss_side_only": True,
        "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
        "record_count": len(records),
        "records": records,
        "runtime_input": False,
        "schema": _VISUAL_AUDIT_SCHEMA,
        "selection_affects_training": False,
        "selection_contract": _VISUAL_SELECTION,
        "temporal_audit_sha256": temporal_audit_sha256,
    }
    _write_json_durable(output_dir / "visual_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--physical-sidecar-manifest", type=Path)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--predictive-cache-root", required=True, type=Path)
    parser.add_argument("--predictive-cache-manifest-sha256", required=True)
    parser.add_argument("--predictive-query-schema-sha256", required=True)
    parser.add_argument("--predictive-coverage-sha256", required=True)
    parser.add_argument("--current-cache-root", required=True, type=Path)
    parser.add_argument("--current-cache-manifest-sha256", required=True)
    parser.add_argument("--current-coverage-sha256", required=True)
    parser.add_argument("--predictive-encoder-digest", required=True)
    parser.add_argument("--current-encoder-digest", required=True)
    parser.add_argument("--maximum-samples", default=2048, type=int)
    parser.add_argument("--memory-capacity", default=1, type=int)
    parser.add_argument(
        "--visual-output-dir",
        type=Path,
        help="optional immutable cache-alignment panels; never model input or learned evidence",
    )
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    validate_dataset_runtime_binding(
        manifest,
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        split_name=args.dataset_split.resolve().name,
    )
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(
        args.physical_sidecar_root,
        index,
        manifest_path=args.physical_sidecar_manifest,
        expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
    )
    predictive_cache = LingBotPredictiveTargetCache.load(
        args.predictive_cache_root,
        manifest_sha256=args.predictive_cache_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
        physical_sidecar_manifest_sha256=sidecar.manifest_sha256,
        encoder_digest=args.predictive_encoder_digest,
        query_schema_sha256=args.predictive_query_schema_sha256,
        coverage_sha256=args.predictive_coverage_sha256,
        memory_capacity=args.memory_capacity,
    )
    current_cache = LingBotCurrentGridTargetCache.load(
        args.current_cache_root,
        manifest_sha256=args.current_cache_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
        physical_sidecar_manifest_sha256=sidecar.manifest_sha256,
        encoder_digest=args.current_encoder_digest,
        coverage_sha256=args.current_coverage_sha256,
        memory_capacity=args.memory_capacity,
    )
    report = audit_predictive_temporal_content(
        predictive_cache,
        current_cache,
        sidecar,
        maximum_samples=args.maximum_samples,
    )
    report_payload = (
        json.dumps(report, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    report_sha256 = hashlib.sha256(report_payload).hexdigest()
    if args.visual_output_dir is not None:
        visual_manifest = render_predictive_temporal_visual_audit(
            output_dir=args.visual_output_dir,
            index=index,
            physical_sidecar=sidecar,
            predictive_cache=predictive_cache,
            current_cache=current_cache,
            temporal_audit_sha256=report_sha256,
        )
        print(
            json.dumps(
                {
                    "event": "teacher_cache_visual_audit_complete",
                    "record_count": visual_manifest["record_count"],
                    "visual_manifest": str(
                        args.visual_output_dir.resolve() / "visual_manifest.json"
                    ),
                    "visual_manifest_sha256": sha256_file(
                        args.visual_output_dir / "visual_manifest.json"
                    ),
                },
                sort_keys=True,
            )
        )
    _write_json_durable(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    interpretation = report["interpretation"]
    if (
        not isinstance(interpretation, dict)
        or interpretation.get("pretraining_readiness") != "PASS"
    ):
        raise RuntimeError("predictive temporal targets failed pretraining readiness")


if __name__ == "__main__":
    main()

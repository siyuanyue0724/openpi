#!/usr/bin/env python3
"""Probe real CALVIN physical targets through the official MolmoAct2 layout.

The probe does not load model weights or run discovery.  It runs the pinned
official image processor and ``MolmoAct2Model.build_batched_images`` once to
materialize the exact resize-mode two-camera patch contract, then validates
every selected dataset frame against the loss-only physical sidecar.  Source
sensor hashes are recomputed from the materialized training record.
"""

from __future__ import annotations

# ruff: noqa: E402
import argparse
import json
import sys
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from olmo.hf_model.image_processing_molmoact2 import MolmoAct2ImageProcessor
from olmo.hf_model.modeling_molmoact2 import MolmoAct2Model

from picf_next.data.calvin import CALVIN_HOST_IMAGE_KEYS, CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionSidecar,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_files,
)
from picf_next.eval.lifecycle import (
    audit_visibility_target_sequences,
    partition_contiguous_visibility_targets,
)
from picf_next.hosts.molmoact2 import (
    MOLMO_VISION_PATCH_MODALITY,
    _dense_patch_partition,
    _molmoact2_vision_patch_layout,
)
from picf_next.hosts.molmoact2_training import (
    CalvinSourceFrameLossTargetRequest,
    CalvinStatefulLossTargetLayout,
    CalvinStatefulLossTargetRequest,
    CalvinVisibleObjectTargetBuilder,
    calvin_physical_source_hashes,
)
from picf_next.models.evidence import ModalityTokenSpan


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split-root",
        type=Path,
        default=_ROOT / "data/calvin_download/calvin_debug_dataset/training",
    )
    parser.add_argument(
        "--sidecar-root",
        type=Path,
        default=_ROOT / "data/calvin_physical_supervision_v2",
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_ROOT / "evidence/calvin_dataset_audit/training_source_manifest.json",
    )
    parser.add_argument(
        "--processor-config",
        type=Path,
        default=_ROOT / "data/molmoact2_metadata/processor_config.json",
    )
    parser.add_argument(
        "--model-config",
        type=Path,
        default=_ROOT / "data/molmoact2_metadata/config.json",
    )
    parser.add_argument("--query-capacity", type=int, default=16)
    parser.add_argument(
        "--maximum-frames",
        type=int,
        default=0,
        help="Zero validates every sidecar frame.",
    )
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _source_frame_arrays(
    index: CalvinDatasetIndex,
    global_index: int,
) -> Mapping[str, np.ndarray]:
    fields = {
        str(spec[field])
        for spec in CALVIN_CAMERA_SPECS
        for field in ("source_rgb_field", "source_depth_field")
    }
    return index.validated_source_frame_arrays(
        global_index,
        fields=tuple(sorted(fields)),
    )


def _source_frame_hashes(
    values: Mapping[str, np.ndarray],
) -> tuple[tuple[str, str], ...]:
    return tuple((field, source_array_sha256(field, values[field])) for field in sorted(values))


def _sidecar_global_indices(sidecar_root: Path) -> tuple[int, ...]:
    manifest = _read_json(sidecar_root / "manifest.json")
    shards = manifest.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ValueError("physical sidecar manifest has no shards")
    indices: list[int] = []
    for shard in shards:
        if not isinstance(shard, dict) or not isinstance(shard.get("path"), str):
            raise ValueError("physical sidecar shard metadata is malformed")
        with np.load(sidecar_root / shard["path"], allow_pickle=False) as archive:
            values = archive["global_indices"]
            if values.dtype != np.int64 or values.ndim != 1:
                raise ValueError("physical sidecar global indices are malformed")
            indices.extend(int(value) for value in values.tolist())
    frozen = tuple(indices)
    if not frozen or tuple(sorted(set(frozen))) != frozen:
        raise ValueError("physical sidecar global indices must be unique and sorted")
    return frozen


def _segment_for_global_index(index: CalvinDatasetIndex, global_index: int) -> int:
    candidates = tuple(
        segment.index for segment in index.segments if segment.start <= global_index <= segment.end
    )
    if not candidates:
        raise ValueError(f"sidecar frame {global_index} is outside every language segment")
    return candidates[0]


def _official_layout(
    *,
    source_arrays: dict[str, np.ndarray],
    processor_config: dict[str, Any],
    model_config: dict[str, Any],
) -> CalvinStatefulLossTargetLayout:
    image_config = processor_config.get("image_processor")
    if not isinstance(image_config, dict):
        raise ValueError("MolmoAct2 processor config has no image_processor mapping")
    processor = MolmoAct2ImageProcessor(**image_config)
    processed = dict(
        processor.preprocess(
            [
                source_arrays["rgb_static"].copy(),
                source_arrays["rgb_gripper"].copy(),
            ],
            return_tensors="pt",
        )
    )
    image_end_token_id = model_config.get("image_end_token_id")
    if not isinstance(image_end_token_id, int) or isinstance(image_end_token_id, bool):
        raise ValueError("MolmoAct2 model config has no integer image_end_token_id")
    input_ids = torch.full((1, 2), image_end_token_id, dtype=torch.long)
    backbone_stub = SimpleNamespace(config=SimpleNamespace(image_end_token_id=image_end_token_id))
    images, pooling = MolmoAct2Model.build_batched_images(
        backbone_stub,
        input_ids=input_ids,
        pixel_values=processed["pixel_values"],
        image_token_pooling=processed["image_token_pooling"],
        image_grids=processed["image_grids"],
        image_num_crops=processed["image_num_crops"],
    )
    dense_valid = _dense_patch_partition(
        pooling,
        num_crops=images.shape[1],
        patches_per_crop=images.shape[2],
    )
    policy_stub = SimpleNamespace(config=SimpleNamespace(image_keys=list(CALVIN_HOST_IMAGE_KEYS)))
    layout = _molmoact2_vision_patch_layout(
        policy_stub,
        model_inputs={"input_ids": input_ids, **processed},
        images=images,
        batched_token_pooling=pooling,
        dense_valid=dense_valid,
    )
    return CalvinStatefulLossTargetLayout(
        token_valid=dense_valid.detach().clone(),
        spans=(ModalityTokenSpan(MOLMO_VISION_PATCH_MODALITY, 0, dense_valid.shape[1]),),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.float32,
        vision_patch_layout=layout,
    )


def main() -> None:
    args = _parse_args()
    if args.query_capacity <= 0:
        raise ValueError("query capacity must be positive")
    if args.maximum_frames < 0:
        raise ValueError("maximum frames cannot be negative")

    split_root = args.split_root.resolve()
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    validate_dataset_files(
        dataset_manifest,
        split_root,
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        split_name=split_root.name,
        verify_hashes=True,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=True,
        dataset_manifest=dataset_manifest,
    )
    sidecar = CalvinPhysicalSupervisionSidecar(args.sidecar_root, index)
    global_indices = _sidecar_global_indices(args.sidecar_root)
    if args.maximum_frames:
        global_indices = global_indices[: args.maximum_frames]
    first_global_index = global_indices[0]
    first_source_arrays = _source_frame_arrays(index, first_global_index)
    layout = _official_layout(
        source_arrays=first_source_arrays,
        processor_config=_read_json(args.processor_config),
        model_config=_read_json(args.model_config),
    )
    builder = CalvinVisibleObjectTargetBuilder(sidecar)

    visible_histogram: Counter[int] = Counter()
    alive_histogram: Counter[int] = Counter()
    total_supervised_tokens = 0
    supervised_tokens_by_frame: list[int] = []
    object_ownership_masses: list[float] = []
    maximum_simplex_error = 0.0
    maximum_visible = 0
    maximum_alive = 0
    visibility_by_global_index: dict[int, dict[str, int | None]] = {}
    for global_index in global_indices:
        source_arrays = _source_frame_arrays(index, global_index)
        if sidecar.coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            source_request = CalvinSourceFrameLossTargetRequest(
                sample_key=f"source-step-{global_index:07d}",
                source_global_index=global_index,
                augmentation_seed=0,
                source_sensor_sha256=_source_frame_hashes(source_arrays),
            )
            targets = builder.source_frames((source_request,), layout)
        else:
            segment_index = _segment_for_global_index(index, global_index)
            record = index.record(segment_index, global_index)
            request = CalvinStatefulLossTargetRequest(
                sample_key=f"segment-{segment_index:05d}/step-{global_index:07d}",
                segment_index=segment_index,
                source_global_index=global_index,
                augmentation_seed=0,
                source_sensor_sha256=calvin_physical_source_hashes(record),
            )
            targets = builder((request,), layout)
        if targets.set_targets is None or targets.lifecycle_targets is None:
            raise RuntimeError("visible target builder omitted set or lifecycle targets")
        target = targets.set_targets[0]
        lifecycle = targets.lifecycle_targets[0]
        if lifecycle is None:
            raise RuntimeError("visible target builder omitted lifecycle inventory")
        visible = target.num_objects
        alive = len(lifecycle.alive_identity_keys)
        if visible > args.query_capacity or alive > args.query_capacity:
            raise RuntimeError(
                f"posterior/query capacity {args.query_capacity} is below frame "
                f"{global_index}: visible={visible}, alive={alive}"
            )
        visible_keys = set(target.temporal_identity_keys or ())
        if not visible_keys.issubset(lifecycle.alive_identity_keys):
            raise RuntimeError("visible physical set is not a subset of the alive inventory")
        if lifecycle.visibility is None or lifecycle.visibility_supervised is None:
            raise RuntimeError("visible target builder omitted lifecycle visibility labels")
        if lifecycle.visibility.shape != lifecycle.visibility_supervised.shape or (
            lifecycle.visibility.shape != (alive,)
        ):
            raise RuntimeError("lifecycle visibility labels do not align with alive identities")
        visibility_by_global_index[global_index] = {
            identity: (
                int(lifecycle.visibility[index].item())
                if bool(lifecycle.visibility_supervised[index].item())
                else None
            )
            for index, identity in enumerate(lifecycle.alive_identity_keys)
        }
        supervised = target.token_supervised
        if (
            supervised is None
            or (supervised & ~layout.token_valid[0]).any()
            or not supervised.any()
        ):
            raise RuntimeError("physical ownership supervision is empty or outside valid patches")
        sums = target.ownership[supervised].sum(dim=-1)
        simplex_error = float((sums - 1.0).abs().max().item())
        if simplex_error > 1e-6:
            raise RuntimeError(f"ownership simplex error {simplex_error} at frame {global_index}")
        if target.num_objects:
            object_mass = target.ownership[supervised, :-1].sum(dim=0).float()
            if not torch.isfinite(object_mass).all() or not bool(torch.all(object_mass > 0.0)):
                raise RuntimeError(
                    f"frame {global_index} contains a token-visible object without "
                    "positive supervised ownership mass"
                )
            object_ownership_masses.extend(object_mass.cpu().tolist())
        supervised_tokens = int(supervised.sum().item())
        total_supervised_tokens += supervised_tokens
        supervised_tokens_by_frame.append(supervised_tokens)
        maximum_simplex_error = max(maximum_simplex_error, simplex_error)
        maximum_visible = max(maximum_visible, visible)
        maximum_alive = max(maximum_alive, alive)
        visible_histogram[visible] += 1
        alive_histogram[alive] += 1

    vision_layout = layout.vision_patch_layout
    if vision_layout is None:
        raise RuntimeError("official processor probe omitted its vision patch layout")
    ownership_mass = np.asarray(object_ownership_masses, dtype=np.float64)
    if not ownership_mass.size:
        raise RuntimeError("official processor probe found no token-visible physical objects")
    lifecycle_sequences: list[tuple[dict[str, int | None], ...]] = []
    selected_global_indices = set(global_indices)
    for segment in index.segments:
        segment_global_indices = tuple(
            global_index
            for global_index in range(segment.start, segment.end + 1)
            if global_index in selected_global_indices
        )
        if any(current != previous + 1 for previous, current in pairwise(segment_global_indices)):
            raise RuntimeError("selected lifecycle target frames are discontinuous")
        segment_frames = tuple(
            visibility_by_global_index[global_index] for global_index in segment_global_indices
        )
        if segment_frames:
            lifecycle_sequences.append(segment_frames)
    if not lifecycle_sequences:
        raise RuntimeError("target probe found no lifecycle supervision sequences")
    task_segment_transition_census = audit_visibility_target_sequences(lifecycle_sequences)
    source_frame_transition_census = audit_visibility_target_sequences(
        partition_contiguous_visibility_targets(global_indices, visibility_by_global_index)
    )
    report = {
        "schema": "picf-next.calvin-molmoact2-target-probe.v3",
        "status": "PASS",
        "frame_count": len(global_indices),
        "first_global_index": global_indices[0],
        "last_global_index": global_indices[-1],
        "query_capacity": args.query_capacity,
        "maximum_visible_objects": maximum_visible,
        "maximum_alive_objects": maximum_alive,
        "visible_object_histogram": {
            str(key): value for key, value in sorted(visible_histogram.items())
        },
        "alive_object_histogram": {
            str(key): value for key, value in sorted(alive_histogram.items())
        },
        "tokens_per_frame": layout.token_valid.shape[1],
        "valid_tokens_per_frame": int(layout.token_valid[0].sum().item()),
        "total_supervised_tokens": total_supervised_tokens,
        "minimum_supervised_tokens_per_frame": min(supervised_tokens_by_frame),
        "maximum_supervised_tokens_per_frame": max(supervised_tokens_by_frame),
        "mean_supervised_tokens_per_frame": (
            total_supervised_tokens / len(supervised_tokens_by_frame)
        ),
        "minimum_supervised_token_fraction": (
            min(supervised_tokens_by_frame) / int(layout.token_valid[0].sum().item())
        ),
        "mean_supervised_token_fraction": (
            total_supervised_tokens
            / (len(supervised_tokens_by_frame) * int(layout.token_valid[0].sum().item()))
        ),
        "maximum_ownership_simplex_error": maximum_simplex_error,
        "object_ownership_mass": {
            "count": int(ownership_mass.size),
            "minimum": float(ownership_mass.min()),
            "p01": float(np.quantile(ownership_mass, 0.01)),
            "p05": float(np.quantile(ownership_mass, 0.05)),
            "median": float(np.median(ownership_mass)),
            "mean": float(ownership_mass.mean()),
            "maximum": float(ownership_mass.max()),
            "counts_below": {
                "0.01": int((ownership_mass < 0.01).sum()),
                "0.1": int((ownership_mass < 0.1).sum()),
                "1.0": int((ownership_mass < 1.0).sum()),
            },
        },
        "objects_without_supervised_ownership_mass": 0,
        "semantic_image_keys": vision_layout.semantic_image_keys,
        "camera_layout": [
            {
                "image_key": span.image_key,
                "start": span.start,
                "stop": span.stop,
                "image_num_crops": span.image_num_crops,
                "patches_per_crop": span.patches_per_crop,
                "image_grid": list(span.image_grid),
            }
            for span in vision_layout.rows[0]
        ],
        "source_hash_alignment": "all materialized RGB/depth arrays matched sidecar",
        "coverage": sidecar.coverage,
        "unknown_pixels_are_context": False,
        "runtime_target_leakage": False,
        "lifecycle_transition_census": task_segment_transition_census,
        "source_frame_transition_census": source_frame_transition_census,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

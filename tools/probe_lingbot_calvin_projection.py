#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Measure LingBot's exact Qwen token geometry on real CALVIN images."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot CALVIN projection probe",
)

import numpy as np

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    source_array_sha256,
)
from picf_next.data.dataset_manifest import DatasetFileManifest, load_dataset_file_manifest
from picf_next.data.lingbot_calvin_projection import (
    LINGBOT_CALVIN_PROJECTION_SCHEMA,
    processor_assets_sha256,
    validate_lingbot_calvin_projection_payload,
)
from tools.bootstrap_lingbot_vla2 import (
    QWEN_PROCESSOR_ID,
    QWEN_PROCESSOR_REVISION,
    validate_processor,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_stable_dataset_manifest(path: Path) -> tuple[DatasetFileManifest, str]:
    digest = _sha256(path)
    manifest = load_dataset_file_manifest(path)
    if _sha256(path) != digest:
        raise ContractError("CALVIN dataset manifest changed while loading the Qwen projection")
    return manifest, digest


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _config_int(config: object, name: str) -> int:
    value = getattr(config, name, None)
    return _positive_int(value, name=f"Qwen vision config {name}")


def _processor_int(processor: object, name: str) -> int:
    image_processor = getattr(processor, "image_processor", None)
    value = getattr(image_processor, name, None)
    return _positive_int(value, name=f"Qwen image processor {name}")


def _sample_global_indices(index: CalvinDatasetIndex) -> list[int]:
    episodes = index.episodes
    if not episodes or episodes[0].start != 0:
        raise ContractError("CALVIN source episodes must begin at global frame zero")
    for previous, current in zip(episodes, episodes[1:], strict=False):
        if current.start != previous.end + 1:
            raise ContractError("CALVIN source episodes must cover one contiguous split")
    frame_count = episodes[-1].end + 1
    return sorted({0, frame_count // 2, frame_count - 1})


def _tensor_shape(value: object, *, name: str) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise ContractError(f"{name} has no tensor shape")
    result = [int(item) for item in shape]
    if not result or any(item <= 0 for item in result):
        raise ContractError(f"{name} has an invalid tensor shape")
    return result


def _grid(value: object) -> list[int]:
    tensor = cast(Any, value)
    if not callable(getattr(tensor, "detach", None)):
        raise ContractError("Qwen image_grid_thw must be a tensor")
    array = np.asarray(tensor.detach().cpu().numpy())
    if array.shape != (1, 3) or not np.issubdtype(array.dtype, np.integer):
        raise ContractError("Qwen image_grid_thw must be integer shape [1,3]")
    return [int(item) for item in array[0]]


def _measure_view(
    *,
    processor: object,
    index: CalvinDatasetIndex,
    sample_global_indices: list[int],
    source_field: str,
    source_shape: list[int],
    merge_size: int,
) -> dict[str, object]:
    import torch

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None or not callable(image_processor):
        raise ContractError("official Qwen processor has no callable image processor")
    measured_geometry: tuple[list[int], list[int]] | None = None
    source_hashes: list[str] = []
    for global_index in sample_global_indices:
        arrays = index.validated_source_frame_arrays(
            global_index,
            fields=(source_field,),
        )
        image = np.asarray(arrays[source_field])
        if list(image.shape) != source_shape or image.dtype != np.uint8:
            raise ContractError("CALVIN source image differs from the camera contract")
        source_hashes.append(source_array_sha256(source_field, image))
        training_image = torch.from_numpy(image.copy()).permute(2, 0, 1).to(torch.float32)
        result = image_processor(training_image)
        if not isinstance(result, Mapping):
            raise ContractError("Qwen image processor output must be a mapping")
        if set(result) != {"pixel_values", "image_grid_thw"}:
            raise ContractError("Qwen image processor output fields changed")
        image_grid = _grid(result["image_grid_thw"])
        pixel_values_shape = _tensor_shape(
            result["pixel_values"],
            name="Qwen pixel_values",
        )
        geometry = (image_grid, pixel_values_shape)
        if measured_geometry is None:
            measured_geometry = geometry
        elif geometry != measured_geometry:
            raise ContractError("Qwen token geometry varies across sampled CALVIN frames")
    if measured_geometry is None:
        raise RuntimeError("projection probe measured no CALVIN frames")
    image_grid, pixel_values_shape = measured_geometry
    if image_grid[0] != 1 or image_grid[1] % merge_size or image_grid[2] % merge_size:
        raise ContractError("Qwen image grid cannot form merged CALVIN tokens")
    raw_patch_count = int(np.prod(image_grid, dtype=np.int64))
    return {
        "source_field": source_field,
        "source_shape": source_shape,
        "image_grid_thw": image_grid,
        "merged_grid_hw": [
            image_grid[1] // merge_size,
            image_grid[2] // merge_size,
        ],
        "raw_patch_count": raw_patch_count,
        "merged_token_count": raw_patch_count // (merge_size * merge_size),
        "pixel_values_shape": pixel_values_shape,
        "source_rgb_sha256": source_hashes,
    }


def build_projection_contract(
    *,
    split_root: Path,
    dataset_manifest_path: Path,
    processor_dir: Path,
) -> dict[str, Any]:
    """Build one fail-closed contract using the exact production processor."""

    from transformers import AutoConfig, AutoProcessor, __version__ as transformers_version

    processor_report = validate_processor(processor_dir)
    if (
        processor_report["processor_id"] != QWEN_PROCESSOR_ID
        or processor_report["processor_revision"] != QWEN_PROCESSOR_REVISION
    ):
        raise ContractError("validated Qwen processor identity changed")
    dataset_manifest, dataset_manifest_sha256 = _load_stable_dataset_manifest(dataset_manifest_path)
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=dataset_manifest.dataset_id,
        dataset_revision=dataset_manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    sample_global_indices = _sample_global_indices(index)
    source_frame_count = index.episodes[-1].end + 1
    # QWEN_PROCESSOR_REVISION is an exact commit and these loads are local-only.
    processor = AutoProcessor.from_pretrained(  # nosec B615
        processor_dir,
        padding_side="right",
        revision=QWEN_PROCESSOR_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    qwen_config = AutoConfig.from_pretrained(  # nosec B615
        processor_dir,
        revision=QWEN_PROCESSOR_REVISION,
        trust_remote_code=True,
        local_files_only=True,
    )
    vision_config = getattr(qwen_config, "vision_config", None)
    if vision_config is None:
        raise ContractError("Qwen config has no vision_config")
    patch_size = _config_int(vision_config, "patch_size")
    merge_size = _config_int(vision_config, "spatial_merge_size")
    temporal_patch_size = _config_int(vision_config, "temporal_patch_size")
    if (
        patch_size != _processor_int(processor, "patch_size")
        or merge_size != _processor_int(processor, "merge_size")
        or temporal_patch_size != _processor_int(processor, "temporal_patch_size")
    ):
        raise ContractError("Qwen processor and vision config geometry differ")

    assets = processor_report["processor_assets"]
    if not isinstance(assets, list):
        raise ContractError("validated Qwen processor returned no asset manifest")
    assets_by_path = {
        str(item["path"]): str(item["sha256"])
        for item in assets
        if isinstance(item, dict) and set(item) == {"path", "bytes", "sha256"}
    }
    if set(("config.json", "preprocessor_config.json")).difference(assets_by_path):
        raise ContractError("validated Qwen processor omitted projection configuration")
    views = {}
    for spec in CALVIN_CAMERA_SPECS:
        camera_name = str(spec["camera_name"])
        views[camera_name] = _measure_view(
            processor=processor,
            index=index,
            sample_global_indices=sample_global_indices,
            source_field=str(spec["source_rgb_field"]),
            source_shape=[int(spec["height"]), int(spec["width"]), 3],
            merge_size=merge_size,
        )
    payload = {
        "schema": LINGBOT_CALVIN_PROJECTION_SCHEMA,
        "status": "PASS",
        "runtime_input": False,
        "processor_id": QWEN_PROCESSOR_ID,
        "processor_revision": QWEN_PROCESSOR_REVISION,
        "processor_assets_sha256": processor_assets_sha256(assets),
        "processor_config_sha256": assets_by_path["config.json"],
        "processor_preprocessor_config_sha256": assets_by_path["preprocessor_config.json"],
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_tree_sha256": dataset_manifest.tree_sha256,
        "source_frame_count": source_frame_count,
        "sample_global_indices": sample_global_indices,
        "patch_size": patch_size,
        "merge_size": merge_size,
        "temporal_patch_size": temporal_patch_size,
        "views": views,
        "transformers_version": transformers_version,
    }
    if _sha256(dataset_manifest_path) != dataset_manifest_sha256:
        raise ContractError("CALVIN dataset manifest changed during the Qwen projection probe")
    return validate_lingbot_calvin_projection_payload(
        payload,
        expected_dataset_manifest_sha256=payload["dataset_manifest_sha256"],
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = build_projection_contract(
        split_root=args.split_root.resolve(),
        dataset_manifest_path=args.dataset_manifest.resolve(),
        processor_dir=args.processor_dir.resolve(),
    )
    encoded = (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode(
        "ascii"
    )
    write_bytes_durable_exclusive(args.output.resolve(), encoded)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

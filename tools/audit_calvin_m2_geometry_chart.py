#!/usr/bin/env python3
"""Reproduce the train-only visible-object geometry chart used by CALVIN M2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_M2_TRAIN_GEOMETRY_OFFSET,
    CALVIN_M2_TRAIN_GEOMETRY_SCALE,
    CALVIN_M2_TRAIN_VISIBLE_GEOMETRY_ROWS,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    sha256_file,
)
from picf_next.geometry import PhysicalGeometryContract


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache-manifest", required=True, type=Path)
    parser.add_argument("--physical-sidecar-root", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid JSON artifact: {path}") from error
    if not isinstance(value, dict):
        raise ContractError(f"JSON artifact must contain one mapping: {path}")
    return value


def _safe_shard_path(root: Path, metadata: object) -> Path:
    if not isinstance(metadata, dict):
        raise ContractError("CALVIN physical shard metadata must be a mapping")
    relative = metadata.get("path")
    expected = metadata.get("sha256")
    if (
        not isinstance(relative, str)
        or not relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
        or not isinstance(expected, str)
        or len(expected) != 64
    ):
        raise ContractError("CALVIN physical shard path or digest is unsafe")
    path = root / relative
    if not path.is_file() or sha256_file(path) != expected:
        raise ContractError(f"CALVIN physical shard is missing or corrupt: {relative}")
    return path


def _load_feature_records(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _json_mapping(path)
    if manifest.get("schema") != "picf-next.molmoact2-m2-feature-cache.v1":
        raise ContractError("unsupported MolmoAct2 M2 feature-cache schema")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise ContractError("M2 feature cache has no records")
    if manifest.get("sample_count") != len(records):
        raise ContractError("M2 feature-cache sample count differs from its records")
    if manifest.get("records_sha256") != _canonical_sha256(records):
        raise ContractError("M2 feature-cache record fingerprint is invalid")
    typed = []
    sample_keys = set()
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("M2 feature-cache record must be a mapping")
        sample_key = record.get("sample_key")
        split = record.get("split")
        global_index = record.get("global_index")
        sensor_hashes = record.get("source_sensor_sha256")
        if (
            not isinstance(sample_key, str)
            or sample_key in sample_keys
            or split not in {"train", "validation", "heldout"}
            or not isinstance(global_index, int)
            or isinstance(global_index, bool)
            or not isinstance(sensor_hashes, list)
        ):
            raise ContractError("M2 feature-cache record identity is invalid")
        sample_keys.add(sample_key)
        typed.append(record)
    return manifest, typed


def _load_physical_frames(
    root: Path,
) -> tuple[dict[str, Any], PhysicalGeometryContract, dict[int, dict[str, Any]]]:
    manifest_path = root / "manifest.json"
    manifest = _json_mapping(manifest_path)
    if manifest.get("runtime_input") is not False or manifest.get("task_conditioned") is not False:
        raise ContractError("CALVIN physical sidecar must remain loss-only and task-independent")
    contract = PhysicalGeometryContract.from_dict(manifest.get("geometry_contract"))
    if manifest.get("geometry_contract_sha256") != contract.fingerprint:
        raise ContractError("CALVIN physical geometry fingerprint is invalid")
    camera_specs = manifest.get("camera_specs")
    shards = manifest.get("shards")
    if not isinstance(camera_specs, list) or not camera_specs:
        raise ContractError("CALVIN physical manifest has no camera contract")
    if not isinstance(shards, list) or not shards:
        raise ContractError("CALVIN physical manifest has no shards")

    frames: dict[int, dict[str, Any]] = {}
    for metadata in shards:
        path = _safe_shard_path(root, metadata)
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        required = {
            "global_indices",
            "frame_offsets",
            "geometry",
            "geometry_supervised",
        }
        for raw_spec in camera_specs:
            if not isinstance(raw_spec, dict):
                raise ContractError("CALVIN physical camera contract is invalid")
            camera = raw_spec.get("camera_name")
            required.update(
                {
                    f"{camera}_owner_index",
                    f"{camera}_source_rgb_sha256",
                    f"{camera}_source_depth_sha256",
                }
            )
        if not required.issubset(arrays):
            raise ContractError("CALVIN physical shard lacks geometry/owner evidence")
        indices = arrays["global_indices"]
        offsets = arrays["frame_offsets"]
        geometry = arrays["geometry"]
        supervised = arrays["geometry_supervised"]
        if (
            indices.dtype != np.int64
            or indices.ndim != 1
            or offsets.dtype != np.int64
            or offsets.shape != (len(indices) + 1,)
            or geometry.dtype != np.float32
            or geometry.ndim != 2
            or geometry.shape[1] != contract.dimension
            or supervised.dtype != np.bool_
            or supervised.shape != geometry.shape
        ):
            raise ContractError("CALVIN physical geometry arrays differ from their contract")
        for row, raw_global_index in enumerate(indices.tolist()):
            global_index = int(raw_global_index)
            if global_index in frames:
                raise ContractError("CALVIN physical sidecar repeats one global frame")
            start, stop = int(offsets[row]), int(offsets[row + 1])
            if not 0 <= start < stop <= len(geometry):
                raise ContractError("CALVIN physical frame offsets are invalid")
            owners = []
            sensor_hashes = {}
            for raw_spec in camera_specs:
                camera = str(raw_spec["camera_name"])
                owner = arrays[f"{camera}_owner_index"][row]
                if owner.dtype != np.uint8 or owner.ndim != 2 or int(owner.max()) > stop - start:
                    raise ContractError("CALVIN owner raster contains an unknown physical object")
                owners.append(owner)
                sensor_hashes[str(raw_spec["source_rgb_field"])] = str(
                    arrays[f"{camera}_source_rgb_sha256"][row]
                )
                sensor_hashes[str(raw_spec["source_depth_field"])] = str(
                    arrays[f"{camera}_source_depth_sha256"][row]
                )
            visible = sorted(
                {
                    int(value)
                    for owner in owners
                    for value in np.unique(owner).tolist()
                    if int(value) > 0
                }
            )
            selected = np.asarray([start + owner - 1 for owner in visible], dtype=np.int64)
            if len(selected) and not supervised[selected].all():
                raise ContractError("visible CALVIN object has unsupervised geometry")
            frames[global_index] = {
                "geometry": geometry[selected].astype(np.float64),
                "source_sensor_sha256": sensor_hashes,
            }
    if manifest.get("frame_count") != len(frames):
        raise ContractError("CALVIN physical frame count differs from decoded shards")
    return manifest, contract, frames


def _statistics(values: np.ndarray, *, sample_count: int) -> dict[str, Any]:
    if values.ndim != 2 or values.shape[1] != 3 or not len(values):
        raise ContractError("geometry statistics require nonempty xyz rows")
    mean = values.mean(axis=0)
    variance = values.var(axis=0)
    second_moment = np.square(values).mean(axis=0)
    return {
        "sample_count": sample_count,
        "object_rows": len(values),
        "axis": [
            {
                "mean": float(mean[index]),
                "variance_about_mean": float(variance[index]),
                "standard_deviation": float(np.sqrt(variance[index])),
                "second_moment_about_contract_origin": float(second_moment[index]),
            }
            for index in range(3)
        ],
        "scalar_variance_about_global_scalar_mean": float(values.var()),
        "scalar_second_moment_about_contract_origin": float(np.square(values).mean()),
    }


def build_report(feature_manifest_path: Path, physical_root: Path) -> dict[str, Any]:
    feature_manifest, records = _load_feature_records(feature_manifest_path)
    physical_manifest, source_contract, frames = _load_physical_frames(physical_root)
    split_values: dict[str, list[np.ndarray]] = {
        "train": [],
        "validation": [],
        "heldout": [],
    }
    split_samples = {name: 0 for name in split_values}
    for record in records:
        global_index = int(record["global_index"])
        frame = frames.get(global_index)
        if frame is None:
            raise ContractError(f"physical sidecar does not cover M2 frame {global_index}")
        expected_sensor_hashes = dict(record["source_sensor_sha256"])
        if expected_sensor_hashes != frame["source_sensor_sha256"]:
            raise ContractError(f"M2 feature/physical frame mismatch at {global_index}")
        split = str(record["split"])
        normalized = frame["geometry"]
        raw = normalized * np.asarray(
            source_contract.normalization_scale, dtype=np.float64
        ) + np.asarray(source_contract.normalization_offset, dtype=np.float64)
        split_values[split].append(raw)
        split_samples[split] += 1

    raw_statistics = {
        split: _statistics(
            np.concatenate(values, axis=0),
            sample_count=split_samples[split],
        )
        for split, values in split_values.items()
    }
    train_axis = raw_statistics["train"]["axis"]
    observed_offset = tuple(float(axis["mean"]) for axis in train_axis)
    observed_scale = tuple(float(axis["standard_deviation"]) for axis in train_axis)
    offset_error = max(
        abs(left - right)
        for left, right in zip(
            observed_offset,
            CALVIN_M2_TRAIN_GEOMETRY_OFFSET,
            strict=True,
        )
    )
    scale_error = max(
        abs(left - right)
        for left, right in zip(
            observed_scale,
            CALVIN_M2_TRAIN_GEOMETRY_SCALE,
            strict=True,
        )
    )
    if (
        raw_statistics["train"]["object_rows"] != CALVIN_M2_TRAIN_VISIBLE_GEOMETRY_ROWS
        or offset_error > 1e-8
        or scale_error > 1e-8
    ):
        raise ContractError("declared CALVIN M2 geometry chart differs from train-only evidence")

    destination_offset = np.asarray(
        CALVIN_OBJECT_GEOMETRY_CONTRACT.normalization_offset,
        dtype=np.float64,
    )
    destination_scale = np.asarray(
        CALVIN_OBJECT_GEOMETRY_CONTRACT.normalization_scale,
        dtype=np.float64,
    )
    normalized_statistics = {}
    for split, values in split_values.items():
        raw = np.concatenate(values, axis=0)
        normalized_statistics[split] = _statistics(
            (raw - destination_offset) / destination_scale,
            sample_count=split_samples[split],
        )
    return {
        "schema": "picf-next.calvin-m2-geometry-chart-audit.v1",
        "feature_cache_manifest_sha256": sha256_file(feature_manifest_path),
        "feature_cache_records_sha256": feature_manifest["records_sha256"],
        "physical_sidecar_manifest_sha256": sha256_file(physical_root / "manifest.json"),
        "physical_global_indices_sha256": physical_manifest["global_indices_sha256"],
        "all_feature_sensor_hashes_match_physical_sidecar": True,
        "source_geometry_contract": source_contract.to_dict(),
        "destination_geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "raw_geometry_statistics_m": raw_statistics,
        "destination_normalized_statistics": normalized_statistics,
        "declared_train_chart_maximum_offset_error_m": offset_error,
        "declared_train_chart_maximum_scale_error_m": scale_error,
    }


def main() -> None:
    args = _parse_args()
    feature_manifest = args.feature_cache_manifest.expanduser().resolve()
    physical_root = args.physical_sidecar_root.expanduser().resolve()
    report = build_report(feature_manifest, physical_root)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()

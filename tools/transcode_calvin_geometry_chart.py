#!/usr/bin/env python3
"""Transcode verified CALVIN sidecars into the current geometry chart."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import (
    CALVIN_GEOMETRY_SIDECAR_SCHEMA,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    sha256_file,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_SUPERVISION_SCHEMA,
)
from picf_next.geometry import PhysicalGeometryContract


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-physical-root", required=True, type=Path)
    parser.add_argument("--source-geometry-root", required=True, type=Path)
    parser.add_argument("--output-physical-root", required=True, type=Path)
    parser.add_argument("--output-geometry-root", required=True, type=Path)
    return parser.parse_args()


def _load_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ContractError(f"invalid source sidecar manifest: {path}") from error
    if not isinstance(payload, dict):
        raise ContractError("source sidecar manifest must be a mapping")
    if payload.get("runtime_input") is not False:
        raise ContractError("CALVIN geometry transcoding accepts only loss-only sidecars")
    contract = PhysicalGeometryContract.from_dict(payload.get("geometry_contract"))
    if payload.get("geometry_contract_sha256") != contract.fingerprint:
        raise ContractError("source sidecar geometry contract fingerprint is invalid")
    shards = payload.get("shards")
    if not isinstance(shards, list) or not shards:
        raise ContractError("source sidecar manifest has no shards")
    return payload


def _create_output(root: Path) -> None:
    root = root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite output sidecar: {root}")
    root.mkdir(parents=True, exist_ok=True)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _transform_geometry(
    geometry: np.ndarray,
    variance: np.ndarray,
    supervised: np.ndarray,
    *,
    source: PhysicalGeometryContract,
    destination: PhysicalGeometryContract,
) -> tuple[np.ndarray, np.ndarray, float]:
    if (
        geometry.dtype != np.float32
        or variance.dtype != np.float32
        or supervised.dtype != np.bool_
        or geometry.shape != variance.shape
        or geometry.shape != supervised.shape
        or geometry.ndim != 2
        or geometry.shape[1] != source.dimension
        or source.dimension != destination.dimension
    ):
        raise ContractError("CALVIN geometry shard arrays differ from the chart contract")
    if (
        not np.isfinite(geometry).all()
        or not np.isfinite(variance).all()
        or (variance < 0.0).any()
        or (geometry[~supervised] != 0.0).any()
        or (variance[~supervised] != 0.0).any()
    ):
        raise ContractError("CALVIN geometry shard contains invalid supervised values")

    source_offset = np.asarray(source.normalization_offset, dtype=np.float64)
    source_scale = np.asarray(source.normalization_scale, dtype=np.float64)
    destination_offset = np.asarray(destination.normalization_offset, dtype=np.float64)
    destination_scale = np.asarray(destination.normalization_scale, dtype=np.float64)

    raw_geometry = geometry.astype(np.float64) * source_scale + source_offset
    raw_variance = variance.astype(np.float64) * np.square(source_scale)
    transformed = (raw_geometry - destination_offset) / destination_scale
    transformed_variance = raw_variance / np.square(destination_scale)
    transformed = np.where(supervised, transformed, 0.0).astype(np.float32)
    transformed_variance = np.where(
        supervised,
        transformed_variance,
        0.0,
    ).astype(np.float32)

    recovered = transformed.astype(np.float64) * destination_scale + destination_offset
    active = supervised
    maximum_roundtrip_error = (
        float(np.max(np.abs(recovered[active] - raw_geometry[active]))) if active.any() else 0.0
    )
    if maximum_roundtrip_error > 1e-6:
        raise ContractError("CALVIN geometry chart transcode exceeded float32 tolerance")
    return transformed, transformed_variance, maximum_roundtrip_error


def _transcode_root(
    source_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
    *,
    expected_schema: str,
) -> dict[str, Any]:
    source_contract = PhysicalGeometryContract.from_dict(manifest["geometry_contract"])
    destination_contract = CALVIN_OBJECT_GEOMETRY_CONTRACT
    output_shards = []
    maximum_roundtrip_error = 0.0

    for raw_metadata in manifest["shards"]:
        if not isinstance(raw_metadata, dict):
            raise ContractError("CALVIN sidecar shard metadata must be a mapping")
        relative = raw_metadata.get("path")
        expected_sha256 = raw_metadata.get("sha256")
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(expected_sha256, str)
        ):
            raise ContractError("CALVIN sidecar shard path or digest is unsafe")
        source_path = source_root / relative
        if not source_path.is_file() or sha256_file(source_path) != expected_sha256:
            raise ContractError(f"CALVIN source sidecar shard is missing or corrupt: {relative}")
        with np.load(source_path, allow_pickle=False) as archive:
            arrays = {name: archive[name].copy() for name in archive.files}
        required = {"geometry", "geometry_variance", "geometry_supervised"}
        if not required.issubset(arrays):
            raise ContractError("CALVIN sidecar shard lacks geometry arrays")
        geometry, variance, roundtrip_error = _transform_geometry(
            arrays["geometry"],
            arrays["geometry_variance"],
            arrays["geometry_supervised"],
            source=source_contract,
            destination=destination_contract,
        )
        arrays["geometry"] = geometry
        arrays["geometry_variance"] = variance
        maximum_roundtrip_error = max(maximum_roundtrip_error, roundtrip_error)

        destination_path = output_root / relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination_path)
        metadata = dict(raw_metadata)
        metadata["sha256"] = sha256_file(destination_path)
        output_shards.append(metadata)

    output_manifest = dict(manifest)
    output_manifest["schema"] = expected_schema
    output_manifest["geometry_contract"] = destination_contract.to_dict()
    output_manifest["geometry_contract_sha256"] = destination_contract.fingerprint
    output_manifest["shards"] = output_shards
    _atomic_json(output_root / "manifest.json", output_manifest)
    return {
        "source_contract_sha256": source_contract.fingerprint,
        "destination_contract_sha256": destination_contract.fingerprint,
        "frame_count": output_manifest.get("frame_count"),
        "object_record_count": output_manifest.get("object_record_count"),
        "shard_count": len(output_shards),
        "maximum_raw_geometry_roundtrip_error_m": maximum_roundtrip_error,
        "manifest_sha256": sha256_file(output_root / "manifest.json"),
    }


def main() -> None:
    args = _parse_args()
    source_physical = args.source_physical_root.resolve()
    source_geometry = args.source_geometry_root.resolve()
    output_physical = args.output_physical_root.resolve()
    output_geometry = args.output_geometry_root.resolve()
    physical_manifest = _load_manifest(source_physical)
    geometry_manifest = _load_manifest(source_geometry)
    if (
        physical_manifest["geometry_contract"] != geometry_manifest["geometry_contract"]
        or physical_manifest.get("global_indices_sha256")
        != geometry_manifest.get("global_indices_sha256")
        or physical_manifest.get("frame_count") != geometry_manifest.get("frame_count")
        or physical_manifest.get("object_record_count")
        != geometry_manifest.get("object_record_count")
    ):
        raise ContractError("CALVIN physical and geometry source sidecars are not aligned")

    _create_output(output_physical)
    try:
        _create_output(output_geometry)
    except Exception:
        shutil.rmtree(output_physical)
        raise
    try:
        report = {
            "schema": "picf-next.calvin-geometry-chart-transcode.v1",
            "physical": _transcode_root(
                source_physical,
                output_physical,
                physical_manifest,
                expected_schema=CALVIN_PHYSICAL_SUPERVISION_SCHEMA,
            ),
            "geometry": _transcode_root(
                source_geometry,
                output_geometry,
                geometry_manifest,
                expected_schema=CALVIN_GEOMETRY_SIDECAR_SCHEMA,
            ),
        }
    except Exception:
        shutil.rmtree(output_physical)
        shutil.rmtree(output_geometry)
        raise
    if (
        report["physical"]["destination_contract_sha256"]
        != report["geometry"]["destination_contract_sha256"]
    ):
        raise ContractError("CALVIN output sidecars use different destination geometry charts")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build a hash-verified loss-only CALVIN object-geometry sidecar.

The extractor is deterministically partitionable across CPU workers or cloud
jobs.  It reads only archived ``scene_obs``/``robot_obs`` and task-independent
simulator object metadata.  Task strings, model predictions and action targets
cannot influence object selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
    CalvinGeometryShard,
    calvin_source_state_sha256,
    geometry_manifest_payload,
    sha256_file,
)
from picf_next.data.calvin_simulator_geometry import (
    build_calvin_geometry_environment,
    close_calvin_geometry_environment,
    extract_robot_base_aabb_centres,
    load_calvin_scene_ranges,
    scene_for_global_index,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    load_dataset_file_manifest,
    validate_dataset_files,
)

PARTITION_SCHEMA = "picf-next.calvin-physical-geometry-partition.v4"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", required=True, type=Path)
    parser.add_argument("--calvin-env-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--partition-count", type=int, default=1)
    parser.add_argument("--partition-index", type=int, default=0)
    parser.add_argument("--shard-size", type=int, default=2048)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--finalize-only", action="store_true")
    return parser.parse_args()


def _required_indices(index: CalvinDatasetIndex) -> np.ndarray:
    return np.asarray(
        sorted(
            {
                global_index
                for segment in index.segments
                for global_index in range(segment.start, segment.end + 1)
            }
        ),
        dtype=np.int64,
    )


def _partition(indices: np.ndarray, count: int, partition_index: int) -> np.ndarray:
    start = len(indices) * partition_index // count
    stop = len(indices) * (partition_index + 1) // count
    return indices[start:stop]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _shard_payload(shard: CalvinGeometryShard) -> dict[str, Any]:
    return {
        "path": shard.path,
        "sha256": shard.sha256,
        "first_global_index": shard.first_global_index,
        "last_global_index": shard.last_global_index,
        "frame_count": shard.frame_count,
        "object_record_count": shard.object_record_count,
    }


def _write_shard(
    output_dir: Path,
    *,
    partition_index: int,
    shard_index: int,
    records: list[tuple[int, str, tuple[str, ...], np.ndarray]],
) -> CalvinGeometryShard:
    if not records:
        raise ContractError("cannot write an empty CALVIN geometry shard")
    global_indices = np.asarray([record[0] for record in records], dtype=np.int64)
    if np.any(global_indices[1:] <= global_indices[:-1]):
        raise ContractError("CALVIN geometry extraction records must be ordered")
    source_hashes = np.asarray([record[1] for record in records], dtype=np.str_)
    keys = [key for record in records for key in record[2]]
    if not keys:
        raise ContractError("CALVIN geometry shard contains no physical object")
    identity_keys = np.asarray(keys, dtype=np.str_)
    frame_lengths = np.asarray([len(record[2]) for record in records], dtype=np.int64)
    frame_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(frame_lengths)))
    geometry = np.concatenate([record[3] for record in records], axis=0).astype(
        np.float32,
        copy=False,
    )
    variance = np.zeros_like(geometry)
    supervised = np.ones_like(geometry, dtype=np.bool_)
    filename = f"part{partition_index:05d}_shard{shard_index:06d}.npz"
    destination = output_dir / filename
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w+b",
        dir=output_dir,
        prefix=f".{filename}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        np.savez_compressed(
            handle,
            global_indices=global_indices,
            source_state_sha256=source_hashes,
            frame_offsets=frame_offsets,
            identity_keys=identity_keys,
            geometry=geometry,
            geometry_variance=variance,
            geometry_supervised=supervised,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return CalvinGeometryShard(
        path=filename,
        sha256=sha256_file(destination),
        first_global_index=int(global_indices[0]),
        last_global_index=int(global_indices[-1]),
        frame_count=len(records),
        object_record_count=len(identity_keys),
    )


def _extract_partition(
    index: CalvinDatasetIndex,
    *,
    calvin_env_root: Path,
    output_dir: Path,
    partition_count: int,
    partition_index: int,
    shard_size: int,
    progress_every: int,
    scene_info_sha256: str,
    dataset_manifest: DatasetFileManifest,
) -> None:
    all_indices = _required_indices(index)
    selected = _partition(all_indices, partition_count, partition_index)
    if not len(selected):
        raise ContractError("CALVIN geometry partition contains no frame")
    scene_ranges = load_calvin_scene_ranges(
        index.split_root,
        dataset_manifest=dataset_manifest,
    )
    scenes = tuple(scene_for_global_index(scene_ranges, int(value)) for value in selected)
    environments: dict[str, Any] = {}
    shards: list[CalvinGeometryShard] = []
    records: list[tuple[int, str, tuple[str, ...], np.ndarray]] = []
    expected_inventory: tuple[str, ...] | None = None
    try:
        for ordinal, raw_index in enumerate(selected.tolist(), start=1):
            global_index = int(raw_index)
            scene = scenes[ordinal - 1]
            environment = environments.get(scene)
            if environment is None:
                environment = build_calvin_geometry_environment(
                    calvin_env_root,
                    scene=scene,
                )
                environments[scene] = environment
            source = index.validated_source_frame_arrays(
                global_index,
                fields=("scene_obs", "robot_obs"),
            )
            scene_obs = source["scene_obs"]
            robot_obs = source["robot_obs"]
            keys, geometry = extract_robot_base_aabb_centres(
                environment,
                scene_obs=scene_obs,
                robot_obs=robot_obs,
            )
            if expected_inventory is None:
                expected_inventory = keys
            elif keys != expected_inventory:
                raise ContractError(f"CALVIN physical inventory changed at frame {global_index}")
            records.append(
                (
                    global_index,
                    calvin_source_state_sha256(scene_obs, robot_obs),
                    keys,
                    geometry,
                )
            )
            if len(records) == shard_size or ordinal == len(selected):
                shards.append(
                    _write_shard(
                        output_dir,
                        partition_index=partition_index,
                        shard_index=len(shards),
                        records=records,
                    )
                )
                records = []
            if ordinal % progress_every == 0 or ordinal == len(selected):
                print(
                    json.dumps(
                        {
                            "partition_count": partition_count,
                            "partition_index": partition_index,
                            "processed": ordinal,
                            "total": len(selected),
                            "global_index": global_index,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        for environment in environments.values():
            close_calvin_geometry_environment(environment)

    partition_payload = {
        "schema": PARTITION_SCHEMA,
        "dataset_id": index.dataset_id,
        "dataset_revision": index.dataset_revision,
        "split_name": index.split_root.name,
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
        "state_restoration": CALVIN_STATE_RESTORATION,
        "geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "geometry_contract_sha256": CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint,
        "scene_info_sha256": scene_info_sha256,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "frame_count": int(len(selected)),
        "global_indices_sha256": hashlib.sha256(selected.tobytes(order="C")).hexdigest(),
        "shards": [_shard_payload(shard) for shard in shards],
    }
    _atomic_json(output_dir / f"partition_{partition_index:05d}.json", partition_payload)


def _load_partition_manifest(
    output_dir: Path,
    *,
    partition_count: int,
    partition_index: int,
    index: CalvinDatasetIndex,
    expected_indices: np.ndarray,
    scene_info_sha256: str,
) -> tuple[CalvinGeometryShard, ...]:
    path = output_dir / f"partition_{partition_index:05d}.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text())
    expected_fields = {
        "schema",
        "dataset_id",
        "dataset_revision",
        "split_name",
        "calvin_commit",
        "calvin_env_commit",
        "state_restoration",
        "geometry_contract",
        "geometry_contract_sha256",
        "scene_info_sha256",
        "partition_count",
        "partition_index",
        "frame_count",
        "global_indices_sha256",
        "shards",
    }
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ContractError("CALVIN geometry partition manifest differs from schema v4")
    selected = _partition(expected_indices, partition_count, partition_index)
    expected_values = {
        "schema": PARTITION_SCHEMA,
        "dataset_id": index.dataset_id,
        "dataset_revision": index.dataset_revision,
        "split_name": index.split_root.name,
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
        "state_restoration": CALVIN_STATE_RESTORATION,
        "geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "geometry_contract_sha256": CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint,
        "scene_info_sha256": scene_info_sha256,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "frame_count": len(selected),
        "global_indices_sha256": hashlib.sha256(selected.tobytes(order="C")).hexdigest(),
    }
    if any(payload[name] != value for name, value in expected_values.items()):
        raise ContractError("CALVIN geometry partition manifest identity or coverage drifted")
    raw_shards = payload["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ContractError("CALVIN geometry partition has no shards")
    shards = tuple(CalvinGeometryShard.from_dict(value) for value in raw_shards)
    cursor = 0
    for shard in shards:
        shard_path = output_dir / shard.path
        if not shard_path.is_file() or sha256_file(shard_path) != shard.sha256:
            raise ContractError(
                f"CALVIN geometry partition shard is missing or corrupt: {shard.path}"
            )
        with np.load(shard_path, allow_pickle=False) as archive:
            indices = archive["global_indices"]
        if not np.array_equal(indices, selected[cursor : cursor + shard.frame_count]):
            raise ContractError("CALVIN geometry partition shard coverage drifted")
        cursor += shard.frame_count
    if cursor != len(selected):
        raise ContractError("CALVIN geometry partition coverage is incomplete")
    return shards


def _finalize(
    index: CalvinDatasetIndex,
    *,
    output_dir: Path,
    partition_count: int,
    scene_info_sha256: str,
) -> None:
    expected_indices = _required_indices(index)
    shards = tuple(
        shard
        for partition_index in range(partition_count)
        for shard in _load_partition_manifest(
            output_dir,
            partition_count=partition_count,
            partition_index=partition_index,
            index=index,
            expected_indices=expected_indices,
            scene_info_sha256=scene_info_sha256,
        )
    )
    payload = geometry_manifest_payload(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        calvin_commit=CALVIN_SOURCE_COMMIT,
        calvin_env_commit=CALVIN_ENV_SOURCE_COMMIT,
        scene_info_sha256=scene_info_sha256,
        global_indices=expected_indices,
        shards=shards,
    )
    if payload["frame_count"] != len(expected_indices):
        raise ContractError("CALVIN geometry final manifest frame count is incomplete")
    _atomic_json(output_dir / "manifest.json", payload)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "frame_count": payload["frame_count"],
                "object_record_count": payload["object_record_count"],
                "shard_count": len(shards),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    integers = {
        "partition_count": args.partition_count,
        "shard_size": args.shard_size,
        "progress_every": args.progress_every,
    }
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in integers.values()
    ):
        raise ValueError("partition count, shard size and progress interval must be positive")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index must lie inside partition count")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    validate_dataset_files(
        dataset_manifest,
        args.split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=args.split_root.resolve().name,
        verify_hashes=True,
    )
    index = CalvinDatasetIndex.load(
        args.split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        verify_files=True,
        dataset_manifest=dataset_manifest,
    )
    scene_info_sha256 = dataset_manifest.record_for("scene_info.npy").sha256
    scene_ranges = load_calvin_scene_ranges(
        index.split_root,
        dataset_manifest=dataset_manifest,
    )
    required_indices = _required_indices(index)
    for global_index in required_indices.tolist():
        scene_for_global_index(scene_ranges, int(global_index))
    if not args.finalize_only:
        if args.calvin_env_root is None:
            raise ValueError("extraction requires --calvin-env-root")
        _extract_partition(
            index,
            calvin_env_root=args.calvin_env_root.resolve(),
            output_dir=output_dir,
            partition_count=args.partition_count,
            partition_index=args.partition_index,
            shard_size=args.shard_size,
            progress_every=args.progress_every,
            scene_info_sha256=scene_info_sha256,
            dataset_manifest=dataset_manifest,
        )
    partition_manifests_ready = all(
        (output_dir / f"partition_{partition_index:05d}.json").is_file()
        for partition_index in range(args.partition_count)
    )
    if args.finalize_only or partition_manifests_ready:
        _finalize(
            index,
            output_dir=output_dir,
            partition_count=args.partition_count,
            scene_info_sha256=scene_info_sha256,
        )


if __name__ == "__main__":
    main()

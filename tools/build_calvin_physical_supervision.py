#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Build unified loss-only CALVIN geometry and visible-owner sidecars."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="physical supervision builder",
)

import numpy as np

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_geometry_schema import (
    CALVIN_ENV_SOURCE_COMMIT,
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
    CALVIN_SOURCE_COMMIT,
    CALVIN_STATE_RESTORATION,
    calvin_source_state_sha256,
    sha256_file,
)
from picf_next.data.calvin_physical_calibration import (
    calvin_depth_consistent_fraction,
    calvin_depth_consistent_supervision,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CALIBRATION_LIMITS,
    CALVIN_CAMERA_SPECS,
    CALVIN_DEPTH_CONSISTENT_DIAGNOSTIC_QUANTILES,
    CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS,
    CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT,
    CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION,
    CALVIN_OWNER_CONTRACT,
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA,
    CALVIN_PHYSICAL_SUPERVISION_PARTITION_SCHEMA,
    CalvinPhysicalSupervisionShard,
    physical_supervision_manifest_payload,
    source_array_sha256,
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


@dataclass(frozen=True, slots=True)
class _CameraRecord:
    source_rgb_sha256: str
    source_depth_sha256: str
    owner_index: np.ndarray
    owner_supervised: np.ndarray
    rgb_mae: float
    depth_mae_m: float
    depth_p95_m: float
    depth_consistent_fraction: float


@dataclass(frozen=True, slots=True)
class _FrameRecord:
    global_index: int
    source_state_sha256: str
    identity_keys: tuple[str, ...]
    geometry: np.ndarray
    cameras: tuple[_CameraRecord, ...]


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
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument(
        "--coverage",
        choices=(
            CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
            CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
        ),
        default=CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES,
    )
    parser.add_argument("--finalize-only", action="store_true")
    parser.add_argument("--defer-finalize", action="store_true")
    parser.add_argument("--resume-completed-partition", action="store_true")
    return parser.parse_args()


def _required_indices(index: CalvinDatasetIndex, coverage: str) -> np.ndarray:
    if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES:
        ranges = [
            np.arange(segment.start, segment.end + 1, dtype=np.int64) for segment in index.segments
        ]
    elif coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        ranges = [
            np.arange(episode.start, episode.end + 1, dtype=np.int64) for episode in index.episodes
        ]
    else:
        raise ContractError("CALVIN physical supervision coverage is unsupported")
    if not ranges:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(ranges))


def _partition(indices: np.ndarray, count: int, index: int) -> np.ndarray:
    return indices[len(indices) * index // count : len(indices) * (index + 1) // count]


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    serialized = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    write_bytes_durable_exclusive(path, serialized)


def _shard_payload(shard: CalvinPhysicalSupervisionShard) -> dict[str, Any]:
    return {
        "path": shard.path,
        "sha256": shard.sha256,
        "first_global_index": shard.first_global_index,
        "last_global_index": shard.last_global_index,
        "frame_count": shard.frame_count,
        "object_record_count": shard.object_record_count,
    }


def _validate_calibration(
    camera_name: str,
    record: _CameraRecord,
    global_index: int,
    *,
    coverage: str,
) -> None:
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        # The all-source contract accepts labels per pixel. Aggregate frame
        # errors include unowned robot/background support and are diagnostics,
        # not evidence that an already depth-verified owner label is invalid.
        return
    if coverage != CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES:
        raise ContractError("CALVIN physical supervision coverage is unsupported")
    failures = []
    limits = CALVIN_CALIBRATION_LIMITS
    if record.rgb_mae > limits["maximum_rgb_mae"]:
        failures.append(f"rgb_mae={record.rgb_mae:.6f}")
    if record.depth_mae_m > limits["maximum_depth_mean_absolute_error_m"]:
        failures.append(f"depth_mae_m={record.depth_mae_m:.6f}")
    if record.depth_p95_m > CALVIN_CALIBRATION_LIMITS["maximum_depth_p95_absolute_error_m"]:
        failures.append(f"depth_p95_m={record.depth_p95_m:.6f}")
    if failures:
        raise ContractError(
            f"CALVIN camera calibration failed at frame {global_index}/{camera_name}: "
            + ", ".join(failures)
        )


def _extract_record(
    index: CalvinDatasetIndex,
    environment: Any,
    global_index: int,
    *,
    coverage: str,
) -> _FrameRecord:
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
    keys, geometry = extract_robot_base_aabb_centres(
        environment,
        scene_obs=frame["scene_obs"],
        robot_obs=frame["robot_obs"],
    )
    renders = render_calvin_camera_ownership(environment, identity_keys=keys)
    if tuple(render.camera_name for render in renders) != tuple(
        str(spec["camera_name"]) for spec in CALVIN_CAMERA_SPECS
    ):
        raise ContractError("CALVIN physical camera output order drifted")
    camera_records = []
    for spec, render in zip(CALVIN_CAMERA_SPECS, renders, strict=True):
        rgb_field = str(spec["source_rgb_field"])
        depth_field = str(spec["source_depth_field"])
        source_rgb = np.asarray(frame[rgb_field])
        source_depth = np.asarray(frame[depth_field])
        if source_rgb.shape != render.rgb.shape or source_depth.shape != render.depth_m.shape:
            raise ContractError("CALVIN rendered and archived camera shapes differ")
        rgb_delta = np.abs(render.rgb.astype(np.float32) - source_rgb.astype(np.float32))
        depth_delta = np.abs(render.depth_m.astype(np.float32) - source_depth.astype(np.float32))
        if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            owner_supervised = calvin_depth_consistent_supervision(
                source_depth,
                render.depth_m,
            )
            depth_consistent_fraction = calvin_depth_consistent_fraction(owner_supervised)
        else:
            owner_supervised = np.ones(render.owner_index.shape, dtype=np.bool_)
            depth_consistent_fraction = 1.0
        record = _CameraRecord(
            source_rgb_sha256=source_array_sha256(rgb_field, source_rgb),
            source_depth_sha256=source_array_sha256(depth_field, source_depth),
            owner_index=render.owner_index,
            owner_supervised=owner_supervised,
            rgb_mae=float(rgb_delta.mean()),
            depth_mae_m=float(depth_delta.mean()),
            depth_p95_m=float(np.quantile(depth_delta, 0.95)),
            depth_consistent_fraction=depth_consistent_fraction,
        )
        _validate_calibration(
            str(spec["camera_name"]),
            record,
            global_index,
            coverage=coverage,
        )
        camera_records.append(record)
    return _FrameRecord(
        global_index=global_index,
        source_state_sha256=calvin_source_state_sha256(frame["scene_obs"], frame["robot_obs"]),
        identity_keys=keys,
        geometry=geometry,
        cameras=tuple(camera_records),
    )


def _write_shard(
    output_dir: Path,
    *,
    partition_index: int,
    shard_index: int,
    records: list[_FrameRecord],
    coverage: str,
) -> CalvinPhysicalSupervisionShard:
    if not records:
        raise ContractError("cannot write an empty CALVIN physical shard")
    global_indices = np.asarray([record.global_index for record in records], dtype=np.int64)
    if np.any(global_indices[1:] <= global_indices[:-1]):
        raise ContractError("CALVIN physical extraction records must be ordered")
    frame_lengths = np.asarray([len(record.identity_keys) for record in records], dtype=np.int64)
    frame_offsets = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(frame_lengths)))
    identity_keys = np.asarray(
        [key for record in records for key in record.identity_keys], dtype=np.str_
    )
    geometry = np.concatenate([record.geometry for record in records], axis=0).astype(
        np.float32, copy=False
    )
    arrays: dict[str, np.ndarray] = {
        "global_indices": global_indices,
        "source_state_sha256": np.asarray(
            [record.source_state_sha256 for record in records], dtype=np.str_
        ),
        "frame_offsets": frame_offsets,
        "identity_keys": identity_keys,
        "geometry": geometry,
        "geometry_variance": np.zeros_like(geometry),
        "geometry_supervised": np.ones_like(geometry, dtype=np.bool_),
    }
    for camera_index, spec in enumerate(CALVIN_CAMERA_SPECS):
        name = str(spec["camera_name"])
        values = [record.cameras[camera_index] for record in records]
        arrays.update(
            {
                f"{name}_source_rgb_sha256": np.asarray(
                    [value.source_rgb_sha256 for value in values], dtype=np.str_
                ),
                f"{name}_source_depth_sha256": np.asarray(
                    [value.source_depth_sha256 for value in values], dtype=np.str_
                ),
                f"{name}_owner_index": np.stack(
                    [value.owner_index for value in values], axis=0
                ).astype(np.uint8, copy=False),
                f"{name}_rgb_mae": np.asarray(
                    [value.rgb_mae for value in values], dtype=np.float32
                ),
                f"{name}_depth_mae_m": np.asarray(
                    [value.depth_mae_m for value in values], dtype=np.float32
                ),
                f"{name}_depth_p95_m": np.asarray(
                    [value.depth_p95_m for value in values], dtype=np.float32
                ),
            }
        )
        if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            arrays.update(
                {
                    f"{name}_owner_supervised": np.stack(
                        [value.owner_supervised for value in values], axis=0
                    ).astype(np.bool_, copy=False),
                    f"{name}_depth_consistent_fraction": np.asarray(
                        [value.depth_consistent_fraction for value in values],
                        dtype=np.float32,
                    ),
                }
            )
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
        np.savez_compressed(handle, **cast(dict[str, Any], arrays))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return CalvinPhysicalSupervisionShard(
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
    required_indices: np.ndarray,
    calvin_env_root: Path,
    output_dir: Path,
    partition_count: int,
    partition_index: int,
    shard_size: int,
    progress_every: int,
    scene_info_sha256: str,
    dataset_manifest: DatasetFileManifest,
    coverage: str,
) -> None:
    selected = _partition(required_indices, partition_count, partition_index)
    if not len(selected):
        raise ContractError("CALVIN physical partition contains no frame")
    scene_ranges = load_calvin_scene_ranges(
        index.split_root,
        dataset_manifest=dataset_manifest,
    )
    environments: dict[str, Any] = {}
    shards: list[CalvinPhysicalSupervisionShard] = []
    records: list[_FrameRecord] = []
    try:
        for ordinal, raw_index in enumerate(selected, start=1):
            global_index = int(raw_index)
            scene = scene_for_global_index(scene_ranges, global_index)
            environment = environments.get(scene)
            if environment is None:
                environment = build_calvin_geometry_environment(
                    calvin_env_root, scene=scene, include_cameras=True
                )
                environments[scene] = environment
            records.append(
                _extract_record(
                    index,
                    environment,
                    global_index,
                    coverage=coverage,
                )
            )
            if len(records) == shard_size or ordinal == len(selected):
                shards.append(
                    _write_shard(
                        output_dir,
                        partition_index=partition_index,
                        shard_index=len(shards),
                        records=records,
                        coverage=coverage,
                    )
                )
                records = []
            if ordinal % progress_every == 0 or ordinal == len(selected):
                print(
                    json.dumps(
                        {
                            "partition": partition_index,
                            "processed": ordinal,
                            "total": len(selected),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    finally:
        for environment in environments.values():
            close_calvin_geometry_environment(environment)
    manifest = {
        "schema": (
            CALVIN_PHYSICAL_SUPERVISION_PARTITION_SCHEMA
            if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            else CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA
        ),
        "dataset_id": index.dataset_id,
        "dataset_revision": index.dataset_revision,
        "split_name": index.split_root.name,
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
        "state_restoration": CALVIN_STATE_RESTORATION,
        "geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "geometry_contract_sha256": CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint,
        "owner_contract": (
            CALVIN_OWNER_CONTRACT
            if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            else CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT
        ),
        "scene_info_sha256": scene_info_sha256,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "frame_count": len(selected),
        "global_indices_sha256": hashlib.sha256(selected.tobytes(order="C")).hexdigest(),
        "shards": [_shard_payload(shard) for shard in shards],
    }
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        manifest["coverage"] = coverage
        manifest["owner_supervision"] = dict(CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION)
        manifest["frame_diagnostics"] = dict(CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS)
    _atomic_json(output_dir / f"partition_{partition_index:05d}.json", manifest)


def _load_partition_shards(
    output_dir: Path,
    *,
    partition_count: int,
    partition_index: int,
    index: CalvinDatasetIndex,
    expected_indices: np.ndarray,
    scene_info_sha256: str,
    coverage: str,
) -> tuple[CalvinPhysicalSupervisionShard, ...]:
    path = output_dir / f"partition_{partition_index:05d}.json"
    if path.is_symlink():
        raise ContractError(f"invalid CALVIN physical partition manifest: {path}")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"invalid CALVIN physical partition manifest: {path}") from error
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
        "owner_contract",
        "scene_info_sha256",
        "partition_count",
        "partition_index",
        "frame_count",
        "global_indices_sha256",
        "shards",
    }
    expected_schema = (
        CALVIN_PHYSICAL_SUPERVISION_PARTITION_SCHEMA
        if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
        else CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA
    )
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        expected_fields.update(("coverage", "owner_supervision", "frame_diagnostics"))
    if not isinstance(payload, dict) or set(payload) != expected_fields:
        raise ContractError("CALVIN physical partition manifest differs from schema v2")
    selected = _partition(expected_indices, partition_count, partition_index)
    expected_values = {
        "schema": expected_schema,
        "dataset_id": index.dataset_id,
        "dataset_revision": index.dataset_revision,
        "split_name": index.split_root.name,
        "calvin_commit": CALVIN_SOURCE_COMMIT,
        "calvin_env_commit": CALVIN_ENV_SOURCE_COMMIT,
        "state_restoration": CALVIN_STATE_RESTORATION,
        "geometry_contract": CALVIN_OBJECT_GEOMETRY_CONTRACT.to_dict(),
        "geometry_contract_sha256": CALVIN_OBJECT_GEOMETRY_CONTRACT.fingerprint,
        "owner_contract": (
            CALVIN_OWNER_CONTRACT
            if coverage == CALVIN_PHYSICAL_COVERAGE_LANGUAGE_FRAMES
            else CALVIN_DEPTH_CONSISTENT_OWNER_CONTRACT
        ),
        "scene_info_sha256": scene_info_sha256,
        "partition_count": partition_count,
        "partition_index": partition_index,
        "frame_count": len(selected),
        "global_indices_sha256": hashlib.sha256(selected.tobytes(order="C")).hexdigest(),
    }
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        expected_values["coverage"] = coverage
        expected_values["owner_supervision"] = dict(CALVIN_DEPTH_CONSISTENT_OWNER_SUPERVISION)
        expected_values["frame_diagnostics"] = dict(CALVIN_DEPTH_CONSISTENT_FRAME_DIAGNOSTICS)
    if any(payload[name] != value for name, value in expected_values.items()):
        raise ContractError("CALVIN physical partition identity or coverage drifted")
    raw_shards = payload["shards"]
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ContractError("CALVIN physical partition has no shards")
    shards = tuple(CalvinPhysicalSupervisionShard.from_dict(value) for value in raw_shards)
    cursor = 0
    for shard in shards:
        shard_path = output_dir / shard.path
        if (
            shard_path.is_symlink()
            or not shard_path.is_file()
            or sha256_file(shard_path) != shard.sha256
        ):
            raise ContractError(f"CALVIN physical shard is missing or corrupt: {shard.path}")
        with np.load(shard_path, allow_pickle=False) as archive:
            indices = archive["global_indices"]
        if not np.array_equal(indices, selected[cursor : cursor + shard.frame_count]):
            raise ContractError("CALVIN physical partition shard coverage drifted")
        cursor += shard.frame_count
    if cursor != len(selected):
        raise ContractError("CALVIN physical partition coverage is incomplete")
    return shards


def _calibration_summary(
    output_dir: Path,
    shards: tuple[CalvinPhysicalSupervisionShard, ...],
    *,
    coverage: str,
) -> dict[str, float]:
    summary = {
        f"maximum_{camera}_{metric}": 0.0
        for camera in ("static", "gripper")
        for metric in ("rgb_mae", "depth_mae_m", "depth_p95_m")
    }
    consistent_values: dict[str, list[np.ndarray]] = {
        camera: [] for camera in ("static", "gripper")
    }
    for shard in shards:
        with np.load(output_dir / shard.path, allow_pickle=False) as archive:
            for camera in ("static", "gripper"):
                for metric in ("rgb_mae", "depth_mae_m", "depth_p95_m"):
                    name = f"maximum_{camera}_{metric}"
                    summary[name] = max(
                        summary[name],
                        float(archive[f"{camera}_{metric}"].max()),
                    )
                if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
                    consistent_values[camera].append(
                        np.asarray(
                            archive[f"{camera}_depth_consistent_fraction"],
                            dtype=np.float64,
                        ).copy()
                    )
    if coverage == CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        for camera, arrays in consistent_values.items():
            values = np.concatenate(arrays)
            summary[f"minimum_{camera}_depth_consistent_fraction"] = float(values.min())
            for name, quantile in CALVIN_DEPTH_CONSISTENT_DIAGNOSTIC_QUANTILES:
                summary[f"{name}_{camera}_depth_consistent_fraction"] = float(
                    np.quantile(values, quantile, method="linear")
                )
    return summary


def _validate_output_membership(
    output_dir: Path,
    *,
    partition_count: int,
    shards: tuple[CalvinPhysicalSupervisionShard, ...],
) -> None:
    expected_partitions = {
        f"partition_{partition_index:05d}.json" for partition_index in range(partition_count)
    }
    actual_partitions = {path.name for path in output_dir.glob("partition_*.json")}
    if actual_partitions != expected_partitions:
        raise ContractError("CALVIN physical output has stale or missing partition manifests")
    expected_shards = {shard.path for shard in shards}
    actual_shards = {path.name for path in output_dir.glob("part*_shard*.npz")}
    if actual_shards != expected_shards:
        raise ContractError("CALVIN physical output has stale or missing shard files")


def _finalize(
    index: CalvinDatasetIndex,
    *,
    expected_indices: np.ndarray,
    output_dir: Path,
    partition_count: int,
    scene_info_sha256: str,
    coverage: str,
) -> None:
    shards = tuple(
        shard
        for partition_index in range(partition_count)
        for shard in _load_partition_shards(
            output_dir,
            partition_count=partition_count,
            partition_index=partition_index,
            index=index,
            expected_indices=expected_indices,
            scene_info_sha256=scene_info_sha256,
            coverage=coverage,
        )
    )
    _validate_output_membership(
        output_dir,
        partition_count=partition_count,
        shards=shards,
    )
    payload = physical_supervision_manifest_payload(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        split_name=index.split_root.name,
        scene_info_sha256=scene_info_sha256,
        global_indices=expected_indices,
        shards=shards,
        calibration_summary=_calibration_summary(
            output_dir,
            shards,
            coverage=coverage,
        ),
        coverage=coverage,
    )
    _atomic_json(output_dir / "manifest.json", payload)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / "manifest.json"),
                "frame_count": payload["frame_count"],
                "object_record_count": payload["object_record_count"],
                "calibration_summary": payload["calibration_summary"],
                "shard_count": len(shards),
                "coverage": coverage,
            },
            sort_keys=True,
        ),
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    for name in ("partition_count", "shard_size", "progress_every"):
        value = getattr(args, name)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    if not 0 <= args.partition_index < args.partition_count:
        raise ValueError("partition index must lie inside partition count")
    if args.finalize_only and args.defer_finalize:
        raise ValueError("--finalize-only and --defer-finalize are mutually exclusive")
    if args.finalize_only and args.resume_completed_partition:
        raise ValueError("--finalize-only and --resume-completed-partition are mutually exclusive")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_manifest = output_dir / "manifest.json"
    if aggregate_manifest.exists() or aggregate_manifest.is_symlink():
        raise FileExistsError(
            f"{aggregate_manifest} already exists; completed physical artifacts are immutable"
        )
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    validate_dataset_runtime_binding(
        dataset_manifest,
        args.split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=args.split_root.resolve().name,
    )
    index = CalvinDatasetIndex.load(
        args.split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    scene_info_sha256 = dataset_manifest.record_for("scene_info.npy").sha256
    scene_ranges = load_calvin_scene_ranges(
        index.split_root,
        dataset_manifest=dataset_manifest,
    )
    required_indices = _required_indices(index, args.coverage)
    for global_index in required_indices:
        scene_for_global_index(scene_ranges, int(global_index))
    if not args.finalize_only:
        partition_manifest = output_dir / f"partition_{args.partition_index:05d}.json"
        if partition_manifest.is_symlink() or (
            partition_manifest.exists() and not partition_manifest.is_file()
        ):
            raise ContractError(
                f"CALVIN physical partition manifest is not a regular file: {partition_manifest}"
            )
        if partition_manifest.is_file():
            if not args.resume_completed_partition:
                raise FileExistsError(
                    f"{partition_manifest} already exists; use "
                    "--resume-completed-partition to verify and retain it"
                )
            _load_partition_shards(
                output_dir,
                partition_count=args.partition_count,
                partition_index=args.partition_index,
                index=index,
                expected_indices=required_indices,
                scene_info_sha256=scene_info_sha256,
                coverage=args.coverage,
            )
            print(
                json.dumps(
                    {
                        "partition": args.partition_index,
                        "status": "verified_complete",
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        else:
            if args.calvin_env_root is None:
                raise ValueError("extraction requires --calvin-env-root")
            _extract_partition(
                index,
                required_indices=required_indices,
                calvin_env_root=args.calvin_env_root.resolve(),
                output_dir=output_dir,
                partition_count=args.partition_count,
                partition_index=args.partition_index,
                shard_size=args.shard_size,
                progress_every=args.progress_every,
                scene_info_sha256=scene_info_sha256,
                dataset_manifest=dataset_manifest,
                coverage=args.coverage,
            )
    ready = all(
        (output_dir / f"partition_{partition_index:05d}.json").is_file()
        for partition_index in range(args.partition_count)
    )
    if args.finalize_only or (ready and not args.defer_finalize):
        _finalize(
            index,
            expected_indices=required_indices,
            output_dir=output_dir,
            partition_count=args.partition_count,
            scene_info_sha256=scene_info_sha256,
            coverage=args.coverage,
        )


if __name__ == "__main__":
    main()

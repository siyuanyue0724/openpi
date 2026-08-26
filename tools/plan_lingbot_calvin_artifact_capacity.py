#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Project persistent storage for one frozen LingBot-native CALVIN plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="CALVIN artifact capacity planner",
)

import numpy as np

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA,
)
from picf_next.data.calvin_simulator_geometry import (
    load_calvin_scene_ranges,
    scene_for_global_index,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)
from picf_next.lingbot_native.artifact_capacity import (
    LingBotCalvinArtifactCapacity,
    PhysicalSidecarStorageSample,
)
from picf_next.lingbot_native.calvin import build_native_calvin_training_stream_plan
from picf_next.lingbot_native.predictive_plan import (
    build_native_current_grid_coverage_plan,
    build_native_predictive_coverage_plan,
)
from picf_next.lingbot_native.representation_split import (
    REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA,
    RepresentationTrialSplit,
)
from picf_next.lingbot_native.stream_plan import (
    add_reset_mixture_arguments,
    reset_mixture_values,
    validate_stream_optimizer_lag,
)
from picf_next.lingbot_native.temporal import TemporalEstimatorConfig


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_physical_samples(
    root: Path,
    *,
    index: CalvinDatasetIndex,
    dataset_manifest: DatasetFileManifest,
    required_partition_indices: tuple[int, ...],
) -> tuple[
    tuple[str, ...],
    tuple[PhysicalSidecarStorageSample, ...],
    tuple[dict[str, object], ...],
    str,
]:
    if (
        len(required_partition_indices) < 3
        or tuple(sorted(set(required_partition_indices))) != required_partition_indices
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in required_partition_indices
        )
    ):
        raise ContractError(
            "physical capacity required partition indices must be sorted, unique, and "
            "contain at least three strata"
        )
    if root.is_symlink():
        raise ContractError("physical capacity sample root cannot be a symlink")
    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(resolved)
    scene_ranges = load_calvin_scene_ranges(
        index.split_root,
        dataset_manifest=dataset_manifest,
    )
    required_scenes = tuple(sorted(value.scene for value in scene_ranges))
    required_indices = np.unique(
        np.concatenate(
            [
                np.arange(episode.start, episode.end + 1, dtype=np.int64)
                for episode in index.episodes
            ]
        )
    )
    manifest_paths = tuple(sorted(resolved.glob("partition_*.json")))
    if len(manifest_paths) != len(required_partition_indices):
        raise ContractError("physical capacity partition indices differ from the frozen strata")
    samples: list[PhysicalSidecarStorageSample] = []
    evidence: list[dict[str, object]] = []
    seen_partitions: set[int] = set()
    seen_indices: set[int] = set()
    frozen_partition_count: int | None = None
    for manifest_path in manifest_paths:
        if manifest_path.is_symlink() or not manifest_path.is_file():
            raise ContractError("physical capacity partition manifest is invalid")
        try:
            payload = json.loads(manifest_path.read_text(encoding="ascii"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ContractError("physical capacity partition manifest is invalid") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema") != CALVIN_PHYSICAL_SUPERVISION_ALL_SOURCE_PARTITION_SCHEMA
            or payload.get("coverage") != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES
            or payload.get("dataset_id") != index.dataset_id
            or payload.get("dataset_revision") != index.dataset_revision
            or payload.get("split_name") != index.split_root.name
        ):
            raise ContractError("physical capacity partition provenance differs")
        partition_index = payload.get("partition_index")
        partition_count = payload.get("partition_count")
        frame_count = payload.get("frame_count")
        shards = payload.get("shards")
        if (
            isinstance(partition_index, bool)
            or not isinstance(partition_index, int)
            or partition_index < 0
            or isinstance(partition_count, bool)
            or not isinstance(partition_count, int)
            or partition_count <= 0
            or partition_index >= partition_count
            or partition_index in seen_partitions
            or isinstance(frame_count, bool)
            or not isinstance(frame_count, int)
            or frame_count <= 0
            or not isinstance(shards, list)
            or not shards
        ):
            raise ContractError("physical capacity partition structure is invalid")
        if manifest_path.name != f"partition_{partition_index:05d}.json" or (
            frozen_partition_count is not None and partition_count != frozen_partition_count
        ):
            raise ContractError("physical capacity partition topology differs")
        frozen_partition_count = partition_count
        seen_partitions.add(partition_index)
        stratum_indices: list[int] = []
        stratum_scenes: set[str] = set()
        stratum_shard_bytes = 0
        maximum_shard_bytes_per_frame = 0.0
        shard_evidence: list[dict[str, object]] = []
        maximum_object_count = 0
        maximum_identity_key_characters = 0
        for item in shards:
            if not isinstance(item, dict):
                raise ContractError("physical capacity shard descriptor is invalid")
            relative = item.get("path")
            expected_sha256 = item.get("sha256")
            expected_frames = item.get("frame_count")
            if (
                not isinstance(relative, str)
                or not relative
                or Path(relative).name != relative
                or not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or isinstance(expected_frames, bool)
                or not isinstance(expected_frames, int)
                or expected_frames <= 0
            ):
                raise ContractError("physical capacity shard descriptor is invalid")
            candidate = resolved / relative
            shard = candidate.resolve()
            if (
                shard.parent != resolved
                or candidate.is_symlink()
                or not shard.is_file()
                or _sha256_file(shard) != expected_sha256
            ):
                raise ContractError("physical capacity shard is missing or corrupt")
            with np.load(shard, allow_pickle=False) as archive:
                indices = np.asarray(archive["global_indices"])
                offsets = np.asarray(archive["frame_offsets"])
                identity_keys = np.asarray(archive["identity_keys"])
            if (
                indices.dtype != np.int64
                or indices.ndim != 1
                or len(indices) != expected_frames
                or offsets.dtype != np.int64
                or offsets.shape != (expected_frames + 1,)
                or int(offsets[0]) != 0
                or np.any(offsets[1:] <= offsets[:-1])
                or not np.issubdtype(identity_keys.dtype, np.str_)
                or identity_keys.shape != (int(offsets[-1]),)
                or any(not str(value) for value in identity_keys.tolist())
            ):
                raise ContractError("physical capacity shard arrays are malformed")
            object_counts = np.diff(offsets)
            maximum_object_count = max(
                maximum_object_count,
                int(object_counts.max(initial=0)),
            )
            maximum_identity_key_characters = max(
                maximum_identity_key_characters,
                max(len(str(value)) for value in identity_keys.tolist()),
            )
            for raw_index in indices.tolist():
                global_index = int(raw_index)
                if global_index in seen_indices:
                    raise ContractError("physical capacity strata overlap")
                seen_indices.add(global_index)
                stratum_indices.append(global_index)
                stratum_scenes.add(scene_for_global_index(scene_ranges, global_index))
            shard_size = os.stat(shard, follow_symlinks=False).st_size
            stratum_shard_bytes += shard_size
            maximum_shard_bytes_per_frame = max(
                maximum_shard_bytes_per_frame,
                shard_size / expected_frames,
            )
            shard_evidence.append(
                {
                    "frame_count": expected_frames,
                    "path": relative,
                    "sha256": expected_sha256,
                    "size_bytes": shard_size,
                }
            )
        if len(stratum_indices) != frame_count:
            raise ContractError("physical capacity partition frame count differs")
        expected_indices = required_indices[
            len(required_indices) * partition_index // partition_count : len(required_indices)
            * (partition_index + 1)
            // partition_count
        ]
        if not np.array_equal(
            np.asarray(stratum_indices, dtype=np.int64),
            expected_indices,
        ):
            raise ContractError("physical capacity partition coverage differs")
        observed_index_digest = hashlib.sha256(
            np.asarray(stratum_indices, dtype=np.int64).tobytes(order="C")
        ).hexdigest()
        if payload.get("global_indices_sha256") != observed_index_digest:
            raise ContractError("physical capacity partition index digest differs")
        samples.append(
            PhysicalSidecarStorageSample(
                frame_count=frame_count,
                shard_bytes=stratum_shard_bytes,
                maximum_shard_bytes_per_frame=maximum_shard_bytes_per_frame,
                scenes=tuple(sorted(stratum_scenes)),
                maximum_object_count=maximum_object_count,
                maximum_identity_key_characters=maximum_identity_key_characters,
            )
        )
        evidence.append(
            {
                "manifest": manifest_path.name,
                "manifest_sha256": _sha256_file(manifest_path),
                "frame_count": frame_count,
                "partition_count": partition_count,
                "partition_index": partition_index,
                "shards": shard_evidence,
            }
        )
    if tuple(sorted(seen_partitions)) != required_partition_indices:
        raise ContractError("physical capacity partition indices differ from the frozen strata")
    evidence_sha256 = hashlib.sha256(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest()
    return required_scenes, tuple(samples), tuple(evidence), evidence_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--plan-seed", required=True, type=int)
    parser.add_argument("--global-batch-size", required=True, type=int)
    parser.add_argument("--total-steps", required=True, type=int)
    parser.add_argument("--local-bptt-probability", required=True, type=float)
    parser.add_argument("--overshoot-probability", required=True, type=float)
    parser.add_argument("--source-mask-probability", required=True, type=float)
    parser.add_argument("--maximum-optimizer-lag", required=True, type=int)
    parser.add_argument("--lane-interleave-factor", type=int, default=1)
    add_reset_mixture_arguments(parser)
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument(
        "--required-future-horizons",
        type=int,
        nargs="*",
        default=(),
        help="Sorted future horizons required for every planned training sample.",
    )
    parser.add_argument("--physical-sample-root", required=True, type=Path)
    parser.add_argument(
        "--required-partition-indices",
        required=True,
        type=int,
        nargs="+",
        help="Exact sorted physical probe strata; extra or missing partitions fail closed.",
    )
    parser.add_argument("--checkpoint-reserve-bytes", required=True, type=int)
    parser.add_argument(
        "--minimum-headroom-bytes",
        type=int,
        default=20 * 1024**3,
    )
    parser.add_argument("--physical-safety-factor", type=float, default=1.25)
    parser.add_argument("--storage-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reset_mixture = reset_mixture_values(args)
    validate_stream_optimizer_lag(
        reset_mixture=reset_mixture,
        lane_interleave_factor=args.lane_interleave_factor,
        maximum_optimizer_lag=args.maximum_optimizer_lag,
    )
    if (args.representation_split is None) != (args.representation_split_sha256 is None):
        raise ValueError("representation split path and SHA-256 must be provided together")
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
    (
        required_scenes,
        physical_samples,
        physical_sample_evidence,
        physical_sample_evidence_sha256,
    ) = _load_physical_samples(
        args.physical_sample_root,
        index=index,
        dataset_manifest=manifest,
        required_partition_indices=tuple(args.required_partition_indices),
    )
    dataset = CalvinStatefulTransitionDataset(index, action_horizon=1)
    representation_split: RepresentationTrialSplit | None = None
    if args.representation_split is not None:
        if _sha256_file(args.representation_split) != args.representation_split_sha256:
            raise ValueError("representation split file SHA-256 differs")
        representation_split = RepresentationTrialSplit.load(args.representation_split)
    stream_plan = build_native_calvin_training_stream_plan(
        dataset,
        comparison_id=args.comparison_id,
        seed=args.plan_seed,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
        lane_interleave_factor=args.lane_interleave_factor,
        excluded_source_episode_indices=(
            representation_split.evaluation_source_episode_indices
            if representation_split is not None
            and representation_split.schema == REPRESENTATION_REFERENCE_TRIAL_SPLIT_SCHEMA
            else ()
        ),
        reset_numerator=(None if reset_mixture is None else reset_mixture[0]),
        reset_denominator=(None if reset_mixture is None else reset_mixture[1]),
    )
    if (
        representation_split is not None
        and representation_split.stream_plan_sha256 != stream_plan.plan_sha256
    ):
        raise ValueError("representation split differs from capacity stream")
    temporal = TemporalEstimatorConfig(
        local_bptt_probability=args.local_bptt_probability,
        overshoot_probability=args.overshoot_probability,
        source_mask_probability=args.source_mask_probability,
        maximum_optimizer_lag=args.maximum_optimizer_lag,
    )
    predictive = build_native_predictive_coverage_plan(
        stream_plan,
        temporal,
        source_global_index_for_sample=dataset.source_global_index_by_key,
        required_horizons=tuple(args.required_future_horizons),
    )
    current = build_native_current_grid_coverage_plan(
        stream_plan,
        temporal,
        source_global_index_for_sample=dataset.source_global_index_by_key,
    )
    storage_root = args.storage_root.resolve()
    if not storage_root.is_dir():
        raise FileNotFoundError(storage_root)
    projection = LingBotCalvinArtifactCapacity(
        free_bytes=shutil.disk_usage(storage_root).free,
        checkpoint_reserve_bytes=args.checkpoint_reserve_bytes,
        minimum_headroom_bytes=args.minimum_headroom_bytes,
        physical_total_frames=sum(episode.end - episode.start + 1 for episode in index.episodes),
        required_scenes=required_scenes,
        physical_samples=physical_samples,
        current_grid_record_count=len(current.source_global_indices),
        predictive_record_count=len(predictive.pairs),
        physical_safety_factor=args.physical_safety_factor,
    )
    report = {
        **projection.as_dict(),
        "dataset": {
            "dataset_id": manifest.dataset_id,
            "dataset_revision": manifest.dataset_revision,
            "dataset_tree_sha256": manifest.tree_sha256,
            "split_name": manifest.split_name,
        },
        "plan": {
            "comparison_id": stream_plan.comparison_id,
            "plan_sha256": stream_plan.plan_sha256,
            "seed": stream_plan.seed,
            "total_steps": stream_plan.total_steps,
            "global_batch_size": stream_plan.global_batch_size,
            "temporal_estimator_sha256": temporal.digest,
            "current_grid_coverage_sha256": current.coverage_sha256,
            "predictive_coverage_sha256": predictive.coverage_sha256,
        },
        "physical_sample_evidence": {
            "files": list(physical_sample_evidence),
            "required_partition_indices": list(args.required_partition_indices),
            "root": str(args.physical_sample_root.resolve()),
            "sha256": physical_sample_evidence_sha256,
        },
    }
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True).encode("ascii") + b"\n"
    write_bytes_durable_exclusive(args.output.resolve(), payload)
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    if projection.status != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

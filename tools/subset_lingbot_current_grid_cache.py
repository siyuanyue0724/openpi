#!/usr/bin/env python3
"""Publish an exact current-grid cache as a subset of one verified donor.

Current-grid rows are deterministic frozen DINO features indexed by source
frames.  A different distributed topology changes the stream-bound coverage
contract, but it must not require re-encoding rows already present in a verified
source bank.  This utility publishes a new exact manifest, reuses complete
donor shards through a shared shard root, and writes only boundary subsets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.lingbot_native.calvin import build_native_calvin_physical_stream_plan
from picf_next.lingbot_native.current_grid_cache import (
    LINGBOT_CURRENT_GRID_CACHE_SCHEMA,
    CurrentGridCacheContract,
    LingBotCurrentGridTargetCache,
)
from picf_next.lingbot_native.predictive_plan import build_native_current_grid_coverage_plan
from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
from picf_next.lingbot_native.temporal import TemporalEstimatorConfig


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_file_sha256(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError(f"SHA-256 mismatch: {path}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--plan-seed", type=int, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--total-steps", type=int, required=True)
    parser.add_argument("--lane-interleave-factor", type=int, default=1)
    parser.add_argument("--local-bptt-probability", type=float, required=True)
    parser.add_argument("--overshoot-probability", type=float, required=True)
    parser.add_argument("--source-mask-probability", type=float, required=True)
    parser.add_argument("--maximum-optimizer-lag", type=int, required=True)
    parser.add_argument("--donor-cache-root", type=Path, required=True)
    parser.add_argument("--donor-cache-manifest-sha256", required=True)
    parser.add_argument("--donor-build-report", type=Path, required=True)
    parser.add_argument("--donor-build-report-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--build-report-output", type=Path)
    parser.add_argument("--verify-reused-shards", action="store_true")
    return parser


def _write_partial_shard(
    *,
    donor: LingBotCurrentGridTargetCache,
    donor_shard_index: int,
    selected_rows: np.ndarray,
    shared_root: Path,
    coverage_sha256: str,
) -> dict[str, object]:
    loaded = donor._load_shard(donor_shard_index)  # noqa: SLF001 - verified cache primitive.
    indices = loaded.source_global_indices[selected_rows]
    hashes = loaded.source_rgb_sha256[selected_rows]
    features = loaded.features[selected_rows]
    subset_root = shared_root / "subsets" / coverage_sha256
    subset_root.mkdir(parents=True, exist_ok=True)
    temporary = subset_root / f".partial-{donor_shard_index:06d}-{os.getpid()}.npz"
    np.savez(
        temporary,
        source_global_indices=indices,
        source_rgb_sha256=hashes,
        features=features,
    )
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    digest = _file_sha256(temporary)
    destination = subset_root / f"{digest}.npz"
    if destination.exists():
        if _file_sha256(destination) != digest:
            raise ContractError("existing current-grid subset shard changed content")
        temporary.unlink()
    else:
        os.replace(temporary, destination)
    return {
        "path": destination.relative_to(shared_root).as_posix(),
        "sha256": digest,
        "row_count": int(indices.shape[0]),
        "first_source_global_index": int(indices[0]),
        "last_source_global_index": int(indices[-1]),
    }


def main() -> None:
    args = _parser().parse_args()
    for name in ("global_batch_size", "total_steps", "lane_interleave_factor"):
        value = getattr(args, name)
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(args.output_root)
    if _file_sha256(args.representation_split) != args.representation_split_sha256:
        raise ContractError("representation split SHA-256 differs")

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    representation_split = RepresentationTrialSplit.load(args.representation_split)
    stream = build_native_calvin_physical_stream_plan(
        dataset,
        comparison_id=args.comparison_id,
        seed=args.plan_seed,
        global_batch_size=args.global_batch_size,
        total_steps=args.total_steps,
        lane_interleave_factor=args.lane_interleave_factor,
        excluded_source_episode_indices=(
            representation_split.evaluation_source_episode_indices
        ),
    )
    if stream.plan_sha256 != representation_split.stream_plan_sha256:
        raise ContractError("rebuilt stream differs from representation split")
    temporal = TemporalEstimatorConfig(
        local_bptt_probability=args.local_bptt_probability,
        overshoot_probability=args.overshoot_probability,
        source_mask_probability=args.source_mask_probability,
        maximum_optimizer_lag=args.maximum_optimizer_lag,
    )
    coverage = build_native_current_grid_coverage_plan(
        stream,
        temporal,
        source_global_index_for_sample=dataset.source_global_index_by_key,
        required_future_offsets=(1,),
    )

    donor_manifest_bytes = _canonical_file_sha256(
        args.donor_cache_root / "manifest.json",
        args.donor_cache_manifest_sha256,
    )
    donor_manifest = json.loads(donor_manifest_bytes)
    donor_contract = CurrentGridCacheContract.from_mapping(donor_manifest["contract"])
    donor = LingBotCurrentGridTargetCache.load(
        args.donor_cache_root,
        manifest_sha256=args.donor_cache_manifest_sha256,
        dataset_tree_sha256=donor_contract.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=donor_contract.physical_sidecar_manifest_sha256,
        encoder_digest=donor_contract.encoder_digest,
        coverage_sha256=donor_contract.coverage_sha256,
        memory_capacity=1,
    )
    if donor_contract.dataset_tree_sha256 != manifest.tree_sha256:
        raise ContractError("donor current-grid cache belongs to another dataset tree")

    target_sources = tuple(coverage.source_global_indices)
    target_set = set(target_sources)
    if len(target_set) != len(target_sources):
        raise ContractError("target current-grid coverage is not unique")
    missing = target_set.difference(donor.locator)
    if missing:
        raise ContractError(f"donor current-grid cache misses {len(missing)} target rows")
    target_contract = replace(
        donor_contract,
        stream_plan_sha256=coverage.stream_plan_sha256,
        temporal_estimator_sha256=coverage.temporal_estimator_sha256,
        source_keys_sha256=coverage.source_keys_sha256,
        coverage_sha256=coverage.coverage_sha256,
        expected_record_count=len(target_sources),
    )

    shared_root = args.donor_cache_root.resolve()
    shard_metadata: list[dict[str, object]] = []
    cursor = 0
    reused_shards = 0
    subset_shards = 0
    reused_rows = 0
    subset_rows = 0
    for shard_index, shard in enumerate(donor.shards):
        donor_sources = donor.source_global_indices[cursor : cursor + shard.row_count]
        cursor += shard.row_count
        selected_rows = np.fromiter(
            (row for row, source in enumerate(donor_sources) if source in target_set),
            dtype=np.int64,
        )
        if selected_rows.size == 0:
            continue
        if selected_rows.size == shard.row_count:
            path = shared_root / shard.path
            if path.is_symlink() or not path.is_file():
                raise ContractError("reused current-grid shard is absent or indirect")
            if args.verify_reused_shards and _file_sha256(path) != shard.sha256:
                raise ContractError("reused current-grid shard SHA-256 differs")
            shard_metadata.append(
                {
                    "path": shard.path,
                    "sha256": shard.sha256,
                    "row_count": shard.row_count,
                    "first_source_global_index": shard.first_source_global_index,
                    "last_source_global_index": shard.last_source_global_index,
                }
            )
            reused_shards += 1
            reused_rows += shard.row_count
        else:
            shard_metadata.append(
                _write_partial_shard(
                    donor=donor,
                    donor_shard_index=shard_index,
                    selected_rows=selected_rows,
                    shared_root=shared_root,
                    coverage_sha256=coverage.coverage_sha256,
                )
            )
            subset_shards += 1
            subset_rows += int(selected_rows.size)
    if cursor != len(donor.source_global_indices):
        raise ContractError("donor current-grid shard coverage is incomplete")
    if sum(int(shard["row_count"]) for shard in shard_metadata) != len(target_sources):
        raise ContractError("published current-grid subset row count differs")

    target_manifest = {
        "schema": LINGBOT_CURRENT_GRID_CACHE_SCHEMA,
        "contract": target_contract.to_dict(),
        "source_global_indices": list(target_sources),
        "shards": shard_metadata,
    }
    manifest_bytes = (
        json.dumps(target_manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    staging = args.output_root.with_name(f".{args.output_root.name}.{os.getpid()}.incomplete")
    staging.mkdir(parents=True)
    try:
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream_handle:
            stream_handle.write(manifest_bytes)
            stream_handle.flush()
            os.fsync(stream_handle.fileno())
        os.replace(staging, args.output_root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    target_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

    restored = LingBotCurrentGridTargetCache.load(
        args.output_root,
        shard_root=shared_root,
        manifest_sha256=target_manifest_sha256,
        dataset_tree_sha256=target_contract.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=target_contract.physical_sidecar_manifest_sha256,
        encoder_digest=target_contract.encoder_digest,
        coverage_sha256=target_contract.coverage_sha256,
        memory_capacity=1,
    )
    if restored.source_global_indices != target_sources:
        raise RuntimeError("restored current-grid cache differs from target coverage")

    donor_report_bytes = _canonical_file_sha256(
        args.donor_build_report,
        args.donor_build_report_sha256,
    )
    donor_report = json.loads(donor_report_bytes)
    subset_donor = {
        "donor_cache_manifest_sha256": args.donor_cache_manifest_sha256,
        "donor_build_report_sha256": args.donor_build_report_sha256,
        "shard_root": str(shared_root),
        "reused_shards": reused_shards,
        "reused_rows": reused_rows,
        "subset_shards": subset_shards,
        "subset_rows": subset_rows,
        "all_target_rows_present": True,
        "frozen_teacher_unchanged": True,
    }
    report = {
        "cache_manifest_sha256": target_manifest_sha256,
        "coverage_sha256": target_contract.coverage_sha256,
        "expected_record_count": target_contract.expected_record_count,
        "output_root": str(args.output_root.resolve()),
        "patch_sha256": donor_report["patch_sha256"],
        "physical_visual_acceptance_sha256": donor_report[
            "physical_visual_acceptance_sha256"
        ],
        "source_keys_sha256": target_contract.source_keys_sha256,
        "stream_plan_sha256": target_contract.stream_plan_sha256,
        "teacher_encoder_digest": target_contract.encoder_digest,
        "temporal_estimator_sha256": target_contract.temporal_estimator_sha256,
    }
    report_path = args.build_report_output or args.output_root.with_suffix(
        ".build_report.json"
    )
    report_bytes = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    with report_path.open("xb") as stream_handle:
        stream_handle.write(report_bytes)
        stream_handle.flush()
        os.fsync(stream_handle.fileno())
    receipt_path = report_path.with_name(f"{args.output_root.name}.subset_receipt.json")
    receipt_bytes = (
        json.dumps(subset_donor, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    with receipt_path.open("xb") as stream_handle:
        stream_handle.write(receipt_bytes)
        stream_handle.flush()
        os.fsync(stream_handle.fileno())
    print(
        json.dumps(
            {
                **subset_donor,
                "build_report": str(report_path.resolve()),
                "build_report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                "manifest_sha256": target_manifest_sha256,
                "output_root": str(args.output_root.resolve()),
                "record_count": len(target_sources),
                "subset_receipt": str(receipt_path.resolve()),
                "subset_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

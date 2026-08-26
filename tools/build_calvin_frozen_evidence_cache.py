#!/usr/bin/env python3
"""Build one resumable, content-addressed CALVIN frozen-evidence cache."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.calvin_frozen_evidence import (
    CalvinAnyTouch2EvidenceBuilder,
    CalvinSonataEvidenceBuilder,
    CalvinVjepa21EvidenceBuilder,
)
from picf_next.data.calvin_pointcloud import CalvinCalibratedPointCloudBuilder
from picf_next.data.calvin_tactile_calibration import load_calvin_tactile_backgrounds
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.data.dense_evidence_cache import publish_dense_evidence_cache_resumable
from picf_next.data.dense_evidence_coverage import DenseEvidenceCoveragePlan
from picf_next.encoders.anytouch2 import AnyTouch2DenseEncoder
from picf_next.encoders.spatiallm_sonata import SpatialLMSonataDenseEncoder
from picf_next.encoders.vjepa21 import Vjepa21DenseEncoder

CALVIN_FROZEN_EVIDENCE_PUBLICATION_SCHEMA = "picf-next.calvin-frozen-evidence-publication/v2"


def _canonical_json(payload: object) -> bytes:
    return (
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    if temporary.exists():
        temporary.unlink()
    try:
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument("--asset-manifest", required=True, type=Path)
    parser.add_argument("--modality", required=True, choices=("vjepa", "sonata", "anytouch"))
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--coverage-plan", required=True, type=Path)
    parser.add_argument("--coverage-plan-sha256", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--token-dtype", default="float16", choices=("float16", "float32"))
    parser.add_argument("--shard-rows", default=64, type=int)
    parser.add_argument("--encoder-batch-size", default=1, type=int)
    parser.add_argument("--partition-count", default=1, type=int)
    parser.add_argument("--partition-index", default=0, type=int)
    parser.add_argument("--action-horizon", default=1, type=int)
    parser.add_argument("--verify-all-dataset-files", action="store_true")
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--point-pixel-stride", default=2, type=int)
    parser.add_argument("--point-budget", default=4096, type=int)
    parser.add_argument("--tactile-calibration-archive", type=Path)
    parser.add_argument("--tactile-calibration-receipt", type=Path)
    parser.add_argument("--tactile-calibration-receipt-sha256")
    return parser


def _partition_bounds(
    record_count: int,
    partition_count: int,
    partition_index: int,
) -> tuple[int, int]:
    if (
        isinstance(record_count, bool)
        or not isinstance(record_count, int)
        or record_count <= 0
        or isinstance(partition_count, bool)
        or not isinstance(partition_count, int)
        or partition_count <= 0
        or partition_count > record_count
    ):
        raise ValueError("cache partition count must be positive and cannot exceed coverage")
    if (
        isinstance(partition_index, bool)
        or not isinstance(partition_index, int)
        or not 0 <= partition_index < partition_count
    ):
        raise ValueError("cache partition index is outside its declared partition count")
    return (
        record_count * partition_index // partition_count,
        record_count * (partition_index + 1) // partition_count,
    )


def _builder(
    args: argparse.Namespace,
    dataset: CalvinPhysicalTransitionDataset,
    *,
    coverage_plan_sha256: str,
) -> Any:
    if args.modality == "vjepa":
        encoder = Vjepa21DenseEncoder.from_manifest(
            args.asset_manifest,
            device=args.device,
            verify_asset=True,
        )
        return CalvinVjepa21EvidenceBuilder(
            dataset=dataset,
            encoder=encoder,
            coverage_plan_sha256=coverage_plan_sha256,
            token_dtype=args.token_dtype,
        )
    if args.modality == "sonata":
        if args.camera_calibration is None:
            raise ValueError("Sonata publication requires --camera-calibration")
        point_builder = CalvinCalibratedPointCloudBuilder(
            args.camera_calibration,
            pixel_stride=args.point_pixel_stride,
            maximum_points=args.point_budget,
        )
        encoder = SpatialLMSonataDenseEncoder.from_manifest(
            args.asset_manifest,
            device=args.device,
            verify_asset=True,
        )
        return CalvinSonataEvidenceBuilder(
            dataset=dataset,
            point_builder=point_builder,
            encoder=encoder,
            coverage_plan_sha256=coverage_plan_sha256,
            token_dtype=args.token_dtype,
        )
    required = {
        "--tactile-calibration-archive": args.tactile_calibration_archive,
        "--tactile-calibration-receipt": args.tactile_calibration_receipt,
        "--tactile-calibration-receipt-sha256": (args.tactile_calibration_receipt_sha256),
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise ValueError(f"AnyTouch publication omitted required arguments: {missing}")
    manifest = dataset.index.dataset_manifest
    if manifest is None:
        raise RuntimeError("AnyTouch publication lost the dataset manifest")
    calibration = load_calvin_tactile_backgrounds(
        args.tactile_calibration_archive,
        args.tactile_calibration_receipt,
        receipt_sha256=args.tactile_calibration_receipt_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
    )
    encoder = AnyTouch2DenseEncoder.from_manifest(
        args.asset_manifest,
        device=args.device,
        verify_asset=True,
    )
    return CalvinAnyTouch2EvidenceBuilder(
        dataset=dataset,
        calibration=calibration,
        encoder=encoder,
        coverage_plan_sha256=coverage_plan_sha256,
        token_dtype=args.token_dtype,
    )


def main() -> None:
    args = _parser().parse_args()
    for value, name in (
        (args.shard_rows, "shard rows"),
        (args.encoder_batch_size, "encoder batch size"),
        (args.action_horizon, "action horizon"),
        (args.point_pixel_stride, "point pixel stride"),
        (args.point_budget, "point budget"),
    ):
        if isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be positive")
    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        (args.dataset_root / args.split).resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=args.verify_all_dataset_files,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=args.action_horizon)
    coverage_payload = args.coverage_plan.read_bytes()
    if _sha256(coverage_payload) != args.coverage_plan_sha256:
        raise ValueError("coverage plan file SHA-256 differs")
    coverage = DenseEvidenceCoveragePlan.load(args.coverage_plan)
    if (
        coverage.dataset_id,
        coverage.dataset_revision,
        coverage.dataset_tree_sha256,
    ) != (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256):
        raise ValueError("coverage plan belongs to another dataset")
    expected_record_count = len(coverage.records)
    partition_start, partition_stop = _partition_bounds(
        expected_record_count,
        args.partition_count,
        args.partition_index,
    )
    partition_record_count = partition_stop - partition_start
    builder = _builder(
        args,
        dataset,
        coverage_plan_sha256=coverage.artifact_sha256,
    )
    output_root = args.output_root.expanduser().resolve()
    started = time.perf_counter()

    def records_from(completed: int):
        pending = coverage.records[partition_start + completed : partition_stop]
        for start in range(0, len(pending), args.encoder_batch_size):
            selected = tuple(
                record.sample_key for record in pending[start : start + args.encoder_batch_size]
            )
            batch_method = getattr(builder, "records_for_sample_keys", None)
            if callable(batch_method):
                yield from batch_method(selected)
            else:
                yield from (builder.record(sample_key) for sample_key in selected)

    manifest_sha256 = publish_dense_evidence_cache_resumable(
        output_root,
        contract=builder.cache_contract,
        expected_record_count=partition_record_count,
        record_factory=records_from,
        shard_rows=args.shard_rows,
    )
    elapsed = time.perf_counter() - started
    receipt = {
        "cache": {
            "contract": builder.cache_contract.payload(),
            "manifest_sha256": manifest_sha256,
            "output_root": str(output_root),
            "record_count": partition_record_count,
        },
        "coverage_plan": {
            "artifact_sha256": coverage.artifact_sha256,
            "file_sha256": args.coverage_plan_sha256,
            "path": str(args.coverage_plan.resolve()),
            "records_sha256": coverage.records_sha256,
        },
        "dataset_manifest_path": str(args.dataset_manifest.resolve()),
        "elapsed_seconds": elapsed,
        "partition": {
            "count": args.partition_count,
            "index": args.partition_index,
            "record_count": partition_record_count,
            "start": partition_start,
            "stop": partition_stop,
        },
        "records_per_second": partition_record_count / elapsed,
        "schema": CALVIN_FROZEN_EVIDENCE_PUBLICATION_SCHEMA,
    }
    receipt_payload = _canonical_json(receipt)
    receipt_output = (
        args.receipt_output.expanduser().resolve()
        if args.receipt_output is not None
        else output_root.with_name(f"{output_root.name}.receipt.json")
    )
    _write_atomic(receipt_output, receipt_payload)
    print(
        json.dumps(
            {
                **receipt,
                "receipt_output": str(receipt_output),
                "receipt_sha256": _sha256(receipt_payload),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

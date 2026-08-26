#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Probe one pinned full-modal encoder on authenticated CALVIN events."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="CALVIN full-modal encoder probe",
)

from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.calvin_frozen_evidence import (
    CalvinAnyTouch2EvidenceBuilder,
    CalvinSonataEvidenceBuilder,
    CalvinVjepa21EvidenceBuilder,
)
from picf_next.data.calvin_pointcloud import CalvinCalibratedPointCloudBuilder
from picf_next.data.calvin_tactile_calibration import load_calvin_tactile_backgrounds
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.encoders.anytouch2 import AnyTouch2DenseEncoder
from picf_next.encoders.spatiallm_sonata import SpatialLMSonataDenseEncoder
from picf_next.encoders.vjepa21 import Vjepa21DenseEncoder
from picf_next.content_addressing import ndarray_sha256

PROBE_SCHEMA = "picf-next.calvin-full-modal-encoder-probe/v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
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
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--coverage-plan-artifact-sha256", required=True)
    parser.add_argument("--modality", choices=("anytouch", "sonata", "vjepa"), required=True)
    parser.add_argument("--source-global-index", type=int, action="append", required=True)
    parser.add_argument("--encoder-batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--tactile-calibration-archive", type=Path)
    parser.add_argument("--tactile-calibration-receipt", type=Path)
    parser.add_argument("--tactile-calibration-receipt-sha256")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sample_key(dataset: CalvinPhysicalTransitionDataset, source_index: int) -> str:
    if source_index < 0:
        raise ValueError("source global indices must be nonnegative")
    episode = next(
        (
            episode
            for episode in dataset.index.episodes
            if episode.start <= source_index <= episode.end
        ),
        None,
    )
    if episode is None:
        raise KeyError(f"source index {source_index} is outside CALVIN")
    sample_key = f"calvin-source-episode-{episode.index:08d}/frame-{source_index:08d}"
    dataset.event_by_key(sample_key)
    return sample_key


def main() -> None:
    args = _parser().parse_args()
    if args.encoder_batch_size <= 0:
        raise ValueError("encoder batch size must be positive")
    if args.modality != "vjepa" and args.encoder_batch_size != 1:
        raise ValueError("only the V-JEPA2.1 probe currently supports encoder batching")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)

    import torch

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("production full-modal probes require CUDA")
    if args.modality == "vjepa":
        encoder = Vjepa21DenseEncoder.from_manifest(
            args.asset_manifest,
            device=args.device,
            verify_asset=True,
        )
        builder = CalvinVjepa21EvidenceBuilder(
            dataset,
            encoder,
            args.coverage_plan_artifact_sha256,
        )
    elif args.modality == "sonata":
        if args.camera_calibration is None:
            raise ValueError("Sonata probe requires --camera-calibration")
        encoder = SpatialLMSonataDenseEncoder.from_manifest(
            args.asset_manifest,
            device=args.device,
            verify_asset=True,
        )
        point_builder = CalvinCalibratedPointCloudBuilder(
            args.camera_calibration,
            pixel_stride=2,
            maximum_points=encoder.config.maximum_points,
        )
        builder = CalvinSonataEvidenceBuilder(
            dataset,
            point_builder,
            encoder,
            args.coverage_plan_artifact_sha256,
        )
    else:
        required = (
            args.tactile_calibration_archive,
            args.tactile_calibration_receipt,
            args.tactile_calibration_receipt_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("AnyTouch probe requires the complete tactile calibration receipt")
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
        builder = CalvinAnyTouch2EvidenceBuilder(
            dataset,
            calibration,
            encoder,
            args.coverage_plan_artifact_sha256,
        )

    torch.cuda.synchronize(args.device)
    torch.cuda.reset_peak_memory_stats(args.device)
    started = time.perf_counter()
    records = []
    batches = []
    source_indices = tuple(args.source_global_index)
    for batch_index, offset in enumerate(range(0, len(source_indices), args.encoder_batch_size)):
        current_indices = source_indices[offset : offset + args.encoder_batch_size]
        sample_keys = tuple(_sample_key(dataset, source_index) for source_index in current_indices)
        before = time.perf_counter()
        if args.modality == "vjepa":
            current_records = builder.records_for_sample_keys(sample_keys)
        else:
            current_records = tuple(builder.record(sample_key) for sample_key in sample_keys)
        torch.cuda.synchronize(args.device)
        batch_elapsed_seconds = time.perf_counter() - before
        batches.append(
            {
                "batch_index": batch_index,
                "elapsed_seconds": batch_elapsed_seconds,
                "sample_count": len(current_records),
            }
        )
        for source_index, sample_key, record in zip(
            current_indices, sample_keys, current_records, strict=True
        ):
            evidence = record.evidence
            records.append(
                {
                    "available": evidence.available,
                    "batch_index": batch_index,
                    "finite": bool(np.isfinite(evidence.tokens).all()),
                    "sample_key": sample_key,
                    "source_global_index": source_index,
                    "source_input_sha256": record.source_input_sha256,
                    "token_count": evidence.token_count,
                    "token_mean": float(evidence.tokens.mean()) if evidence.token_count else None,
                    "token_sha256": (
                        ndarray_sha256("dense-evidence-tokens", evidence.tokens)
                        if evidence.token_count
                        else None
                    ),
                    "token_std": float(evidence.tokens.std()) if evidence.token_count else None,
                    "token_width": evidence.tokens.shape[1],
                }
            )
    payload = {
        "asset_manifest_file_sha256": _sha256(args.asset_manifest),
        "cache_contract": builder.cache_contract.payload(),
        "cuda_device_name": torch.cuda.get_device_name(args.device),
        "dataset_manifest_file_sha256": _sha256(args.dataset_manifest),
        "elapsed_seconds": time.perf_counter() - started,
        "encoder_batch_size": args.encoder_batch_size,
        "batches": batches,
        "modality": args.modality,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(args.device),
        "records": records,
        "schema": PROBE_SCHEMA,
        "torch_version": torch.__version__,
    }
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    _write_atomic(args.output, encoded)
    print(
        json.dumps(
            {**payload, "output": str(args.output), "output_sha256": _sha256(args.output)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

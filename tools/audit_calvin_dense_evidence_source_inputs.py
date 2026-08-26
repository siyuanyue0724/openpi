#!/usr/bin/env python3
"""Recompute frozen CALVIN dense-cache source identities without inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.calvin_dense_evidence_source_audit import (
    audit_calvin_dense_evidence_source_inputs,
)
from picf_next.data.calvin_tactile_calibration import load_calvin_tactile_backgrounds
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.data.dense_evidence_cache import FrozenDenseEvidenceCacheBank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--coverage-plan-sha256", required=True)
    parser.add_argument("--anytouch-cache", required=True, type=Path)
    parser.add_argument("--anytouch-manifest-sha256", required=True)
    parser.add_argument("--sonata-cache", required=True, type=Path)
    parser.add_argument("--sonata-manifest-sha256", required=True)
    parser.add_argument("--vjepa-cache", required=True, type=Path)
    parser.add_argument("--vjepa-manifest-sha256", required=True)
    parser.add_argument("--tactile-calibration-archive", required=True, type=Path)
    parser.add_argument("--tactile-calibration-receipt", required=True, type=Path)
    parser.add_argument("--tactile-calibration-receipt-sha256", required=True)
    parser.add_argument("--record-start", default=0, type=int)
    parser.add_argument("--record-stop", type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--progress-every", default=128, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.progress_every <= 0:
        raise ContractError("progress interval must be positive")
    manifest = load_dataset_file_manifest(args.dataset_manifest)
    index = CalvinDatasetIndex.load(
        args.dataset_root / args.split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    roots = (args.anytouch_cache, args.sonata_cache, args.vjepa_cache)
    manifest_sha256s = (
        args.anytouch_manifest_sha256,
        args.sonata_manifest_sha256,
        args.vjepa_manifest_sha256,
    )
    bank = FrozenDenseEvidenceCacheBank.load(
        roots,
        manifest_sha256s=manifest_sha256s,
        dataset_tree_sha256=manifest.tree_sha256,
        memory_capacity=1,
    )
    if bank.coverage_plan_sha256 != args.coverage_plan_sha256:
        raise ContractError("source audit cache coverage differs from the pinned plan")
    calibration = load_calvin_tactile_backgrounds(
        args.tactile_calibration_archive,
        args.tactile_calibration_receipt,
        receipt_sha256=args.tactile_calibration_receipt_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
    )
    started = time.perf_counter()

    def progress(completed: int, total: int) -> None:
        if completed == total or completed % args.progress_every == 0:
            elapsed = time.perf_counter() - started
            print(
                json.dumps(
                    {
                        "completed": completed,
                        "elapsed_s": elapsed,
                        "records_per_second": completed / max(elapsed, 1e-12),
                        "total": total,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    report = audit_calvin_dense_evidence_source_inputs(
        dataset,
        bank,
        cache_manifest_sha256_by_modality={
            "anytouch": args.anytouch_manifest_sha256,
            "sonata": args.sonata_manifest_sha256,
            "vjepa": args.vjepa_manifest_sha256,
        },
        calibration=calibration,
        record_start=args.record_start,
        record_stop=args.record_stop,
        workers=args.workers,
        progress=progress,
    )
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

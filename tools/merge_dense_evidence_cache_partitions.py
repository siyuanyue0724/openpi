#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Merge authenticated dense-evidence cache partitions into canonical coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="dense-evidence cache partition merger",
)

from picf_next.data.dense_evidence_cache import merge_dense_evidence_cache_partitions
from picf_next.data.dense_evidence_coverage import DenseEvidenceCoveragePlan

MERGE_RECEIPT_SCHEMA = "picf-next.dense-evidence-cache-partition-merge/v2"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.staging-{os.getpid()}")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--partition-root", type=Path, action="append", required=True)
    parser.add_argument("--partition-manifest-sha256", action="append", required=True)
    parser.add_argument("--coverage-plan", type=Path, required=True)
    parser.add_argument("--coverage-plan-sha256", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--link-shards", action="store_true")
    parser.add_argument("--reference-partitions", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if len(args.partition_root) != len(args.partition_manifest_sha256):
        raise ValueError("partition roots and manifest hashes must have equal length")
    coverage_payload = args.coverage_plan.read_bytes()
    if _sha256(coverage_payload) != args.coverage_plan_sha256:
        raise ValueError("coverage plan file SHA-256 differs")
    coverage = DenseEvidenceCoveragePlan.load(args.coverage_plan)
    started = time.perf_counter()
    manifest_sha256 = merge_dense_evidence_cache_partitions(
        args.output_root,
        partition_roots=args.partition_root,
        manifest_sha256s=args.partition_manifest_sha256,
        dataset_tree_sha256=coverage.dataset_tree_sha256,
        coverage_plan_sha256=coverage.artifact_sha256,
        expected_records=tuple(
            (record.source_global_index, record.sample_key) for record in coverage.records
        ),
        shard_rows=args.shard_rows,
        link_shards=args.link_shards,
        reference_partitions=args.reference_partitions,
    )
    receipt = {
        "coverage_plan": {
            "artifact_sha256": coverage.artifact_sha256,
            "file_sha256": args.coverage_plan_sha256,
            "records_sha256": coverage.records_sha256,
        },
        "elapsed_seconds": time.perf_counter() - started,
        "output": {
            "linked_partition_shards": args.link_shards,
            "manifest_sha256": manifest_sha256,
            "record_count": len(coverage.records),
            "referenced_partitions": args.reference_partitions,
            "root": str(args.output_root.resolve()),
        },
        "partitions": [
            {"manifest_sha256": digest, "root": str(root.resolve())}
            for root, digest in zip(
                args.partition_root,
                args.partition_manifest_sha256,
                strict=True,
            )
        ],
        "schema": MERGE_RECEIPT_SCHEMA,
    }
    encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("ascii")
    receipt_output = (
        args.receipt_output.resolve()
        if args.receipt_output is not None
        else args.output_root.with_name(f"{args.output_root.name}.merge-receipt.json").resolve()
    )
    _write_atomic(receipt_output, encoded)
    print(
        json.dumps(
            {
                **receipt,
                "receipt_output": str(receipt_output),
                "receipt_sha256": _sha256(encoded),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

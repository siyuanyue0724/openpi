#!/usr/bin/env python3
"""Republish one CALVIN evidence cache while encoding only donor misses.

This is a bounded execution utility for topology changes.  It authenticates the
donor manifest and every donor shard that contributes a requested record, but
does not scan unrelated donor rows.  Missing target identities are encoded with
the same frozen upstream encoder used by the canonical cache builder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex, CalvinPhysicalTransitionDataset
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.data.dense_evidence_cache import (
    DenseEvidenceCacheRecord,
    FrozenDenseEvidenceCacheBank,
    publish_dense_evidence_cache_resumable,
)
from picf_next.data.dense_evidence_coverage import (
    DenseEvidenceCoveragePlan,
    DenseEvidenceCoverageRecord,
)
from tools.build_calvin_frozen_evidence_cache import _builder

_TECHNICAL_CONTRACT_FIELDS = (
    "encoder_contract",
    "geometry_width",
    "has_group_ids",
    "maximum_tokens",
    "modality",
    "token_dtype",
    "token_width",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--split", choices=("training", "validation"), default="training")
    parser.add_argument("--coverage-plan", type=Path, required=True)
    parser.add_argument("--coverage-plan-sha256", required=True)
    parser.add_argument("--donor-cache-root", type=Path, required=True)
    parser.add_argument("--donor-cache-manifest-sha256", required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--modality", choices=("vjepa", "sonata", "anytouch"), required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--token-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--shard-rows", type=int, default=64)
    parser.add_argument("--encoder-batch-size", type=int, default=1)
    parser.add_argument("--action-horizon", type=int, default=30)
    parser.add_argument("--camera-calibration", type=Path)
    parser.add_argument("--point-pixel-stride", type=int, default=2)
    parser.add_argument("--point-budget", type=int, default=4096)
    parser.add_argument("--tactile-calibration-archive", type=Path)
    parser.add_argument("--tactile-calibration-receipt", type=Path)
    parser.add_argument("--tactile-calibration-receipt-sha256")
    return parser


def _encoded_records(
    builder: Any,
    expected: Sequence[DenseEvidenceCoverageRecord],
) -> tuple[DenseEvidenceCacheRecord, ...]:
    sample_keys = tuple(record.sample_key for record in expected)
    batch_method = getattr(builder, "records_for_sample_keys", None)
    encoded = (
        tuple(batch_method(sample_keys))
        if callable(batch_method)
        else tuple(builder.record(sample_key) for sample_key in sample_keys)
    )
    expected_identities = tuple(
        (record.source_global_index, record.sample_key) for record in expected
    )
    observed_identities = tuple(
        (record.source_global_index, record.sample_key) for record in encoded
    )
    if observed_identities != expected_identities:
        raise ContractError("frozen encoder changed requested coverage identity/order")
    return encoded


def main() -> None:
    args = _parser().parse_args()
    if args.shard_rows <= 0 or args.encoder_batch_size <= 0 or args.action_horizon <= 0:
        raise ValueError("shard rows, encoder batch size, and action horizon must be positive")
    coverage_bytes = args.coverage_plan.read_bytes()
    if hashlib.sha256(coverage_bytes).hexdigest() != args.coverage_plan_sha256:
        raise ValueError("coverage plan file SHA-256 differs")

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    index = CalvinDatasetIndex.load(
        (args.dataset_root / args.split).resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    dataset = CalvinPhysicalTransitionDataset(index, action_horizon=args.action_horizon)
    coverage = DenseEvidenceCoveragePlan.load(args.coverage_plan)
    if (
        coverage.dataset_id,
        coverage.dataset_revision,
        coverage.dataset_tree_sha256,
    ) != (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256):
        raise ValueError("coverage plan belongs to another dataset")

    donor = FrozenDenseEvidenceCacheBank.load(
        (args.donor_cache_root,),
        manifest_sha256s=(args.donor_cache_manifest_sha256,),
        dataset_tree_sha256=manifest.tree_sha256,
        memory_capacity=2,
    ).caches[0]
    if donor.contract.modality != args.modality:
        raise ValueError("donor modality differs from requested modality")

    builder = _builder(
        args,
        dataset,
        coverage_plan_sha256=coverage.artifact_sha256,
    )
    for field in _TECHNICAL_CONTRACT_FIELDS:
        if getattr(builder.cache_contract, field) != getattr(donor.contract, field):
            raise ValueError(f"donor and target encoder contracts differ at {field}")

    donor_by_index = {record.source_global_index: record for record in donor.records}
    donated_count = sum(
        (location := donor_by_index.get(record.source_global_index)) is not None
        and location.sample_key == record.sample_key
        for record in coverage.records
    )
    encoded_count = len(coverage.records) - donated_count

    def records_from(completed: int) -> Iterable[DenseEvidenceCacheRecord]:
        if not 0 <= completed <= len(coverage.records):
            raise ValueError("resume offset lies outside target coverage")
        missing: list[DenseEvidenceCoverageRecord] = []

        def flush_missing() -> Iterable[DenseEvidenceCacheRecord]:
            if not missing:
                return ()
            encoded = _encoded_records(builder, tuple(missing))
            missing.clear()
            return encoded

        for target in coverage.records[completed:]:
            location = donor_by_index.get(target.source_global_index)
            if location is not None and location.sample_key == target.sample_key:
                yield from flush_missing()
                evidence = donor.evidence_for(
                    source_global_index=location.source_global_index,
                    sample_key=location.sample_key,
                    source_input_sha256=location.source_input_sha256,
                )
                yield DenseEvidenceCacheRecord(
                    source_global_index=location.source_global_index,
                    sample_key=location.sample_key,
                    source_input_sha256=location.source_input_sha256,
                    evidence=evidence,
                )
                continue
            missing.append(target)
            if len(missing) == args.encoder_batch_size:
                yield from flush_missing()
        yield from flush_missing()

    manifest_sha256 = publish_dense_evidence_cache_resumable(
        args.output_root,
        contract=builder.cache_contract,
        expected_record_count=len(coverage.records),
        record_factory=records_from,
        shard_rows=args.shard_rows,
    )
    restored = FrozenDenseEvidenceCacheBank.load(
        (args.output_root,),
        manifest_sha256s=(manifest_sha256,),
        dataset_tree_sha256=manifest.tree_sha256,
        memory_capacity=1,
    ).caches[0]
    restored_identities = tuple(
        (record.source_global_index, record.sample_key) for record in restored.records
    )
    if restored_identities != coverage.record_identities:
        raise RuntimeError("published cache differs from target coverage")

    receipt = {
        "coverage_artifact_sha256": coverage.artifact_sha256,
        "coverage_file_sha256": args.coverage_plan_sha256,
        "donated_record_count": donated_count,
        "donor_manifest_sha256": args.donor_cache_manifest_sha256,
        "encoded_record_count": encoded_count,
        "manifest_sha256": manifest_sha256,
        "modality": args.modality,
        "output_root": str(args.output_root.resolve()),
        "record_count": len(coverage.records),
    }
    encoded_receipt = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("ascii")
    if args.receipt_output is not None:
        write_bytes_durable_exclusive(args.receipt_output, encoded_receipt)
    manifest_file_sha256 = _file_sha256(args.output_root / "manifest.json")
    print(
        json.dumps(
            {**receipt, "manifest_file_sha256": manifest_file_sha256},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Publish a manifest-only semantic audit of frozen CALVIN dense evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin_dense_evidence_audit import (
    audit_calvin_dense_evidence_cache_bank,
)
from picf_next.data.dense_evidence_cache import FrozenDenseEvidenceCacheBank


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-tree-sha256", required=True)
    parser.add_argument("--coverage-plan-sha256", required=True)
    parser.add_argument("--anytouch-cache", required=True, type=Path)
    parser.add_argument("--anytouch-manifest-sha256", required=True)
    parser.add_argument("--sonata-cache", required=True, type=Path)
    parser.add_argument("--sonata-manifest-sha256", required=True)
    parser.add_argument("--vjepa-cache", required=True, type=Path)
    parser.add_argument("--vjepa-manifest-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    bank = FrozenDenseEvidenceCacheBank.load(
        (args.anytouch_cache, args.sonata_cache, args.vjepa_cache),
        manifest_sha256s=(
            args.anytouch_manifest_sha256,
            args.sonata_manifest_sha256,
            args.vjepa_manifest_sha256,
        ),
        dataset_tree_sha256=args.dataset_tree_sha256,
        memory_capacity=1,
    )
    if bank.coverage_plan_sha256 != args.coverage_plan_sha256:
        raise ContractError("semantic audit cache coverage differs from the pinned plan")
    report = audit_calvin_dense_evidence_cache_bank(bank)
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

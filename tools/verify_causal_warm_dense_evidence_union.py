#!/usr/bin/env python3
"""Verify every full-modal row consumed by a causal-warm CALVIN evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from picf_next.data.calvin import (
    CalvinDatasetIndex,
    CalvinPhysicalTransitionDataset,
    CalvinStatefulTransitionDataset,
)
from picf_next.data.calvin_multimodal import validate_calvin_evidence_timestamps
from picf_next.data.dataset_manifest import load_dataset_file_manifest
from picf_next.data.dense_evidence_cache import (
    FrozenDenseEvidenceCacheBank,
    FrozenDenseEvidenceCacheView,
    compose_dense_evidence_cache_banks,
)
from picf_next.data.dense_evidence_coverage import DenseEvidenceCoveragePlan
from picf_next.lingbot_native.dense_modalities import (
    NativeDenseModalityBinding,
    native_modalities_from_dense_evidence,
)
from picf_next.lingbot_native.entity_evaluation_plan import EntityEvaluationPlan


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-split", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--evaluation-plan", required=True, type=Path)
    parser.add_argument("--coverage-plan", required=True, type=Path)
    parser.add_argument("--primary-root", action="append", required=True, type=Path)
    parser.add_argument("--primary-manifest-sha256", action="append", required=True)
    parser.add_argument("--supplement-root", action="append", required=True, type=Path)
    parser.add_argument("--supplement-manifest-sha256", action="append", required=True)
    parser.add_argument("--history-transitions", type=int, default=4)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.history_transitions <= 0:
        raise ValueError("history transitions must be positive")
    if not (
        len(args.primary_root)
        == len(args.primary_manifest_sha256)
        == len(args.supplement_root)
        == len(args.supplement_manifest_sha256)
        == 3
    ):
        raise ValueError("causal-warm verification requires three primary and supplement caches")

    manifest = load_dataset_file_manifest(args.dataset_manifest)
    index = CalvinDatasetIndex.load(
        args.dataset_split,
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    physical = CalvinPhysicalTransitionDataset(index, action_horizon=1)
    evaluation = CalvinStatefulTransitionDataset(index, action_horizon=1)
    coverage = DenseEvidenceCoveragePlan.load(args.coverage_plan)
    plan = EntityEvaluationPlan.load(args.evaluation_plan)
    if coverage.evaluation_plan_sha256 != plan.artifact_sha256:
        raise RuntimeError("coverage and evaluation plan artifacts differ")
    if coverage.evaluation_history_transition_count != args.history_transitions:
        raise RuntimeError("coverage history length differs from requested verification")

    primary = FrozenDenseEvidenceCacheBank.load(
        args.primary_root,
        manifest_sha256s=args.primary_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
    )
    supplement = FrozenDenseEvidenceCacheBank.load(
        args.supplement_root,
        manifest_sha256s=args.supplement_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
    )
    union = compose_dense_evidence_cache_banks(
        (primary, supplement),
        record_identities=coverage.record_identities,
        coverage_plan_sha256=coverage.artifact_sha256,
    )
    bindings = tuple(
        NativeDenseModalityBinding(
            name=contract.modality,
            encoder_contract=contract.encoder_contract,
            token_width=contract.token_width,
            maximum_tokens=contract.maximum_tokens,
            geometry_width=contract.geometry_width,
        )
        for contract in union.contracts
    )
    canonical_key_by_source = dict(coverage.record_identities)
    visit_count = 0
    unique_sources: set[int] = set()
    token_counts: Counter[str] = Counter()
    available_counts: Counter[str] = Counter()
    current_counts: Counter[str] = Counter()

    for item in plan.items:
        if item.transition_index < args.history_transitions:
            continue
        history_keys = evaluation.history_sample_keys(item.sample_key)[
            -args.history_transitions:
        ]
        sample_keys = (*history_keys, item.sample_key)
        source_indices = tuple(
            evaluation.source_global_index_by_key(sample_key) for sample_key in sample_keys
        )
        if source_indices != tuple(
            range(item.source_global_index - args.history_transitions, item.source_global_index + 1)
        ):
            raise RuntimeError("causal-warm verification found a nonconsecutive prefix")
        for source_global_index in source_indices:
            canonical_key = canonical_key_by_source[source_global_index]
            evidence_row = union.evidence_for(
                source_global_index=source_global_index,
                sample_key=canonical_key,
            )
            validate_calvin_evidence_timestamps(
                evidence_row,
                observation_timestamp_s=physical.timestamp_s_by_key(canonical_key),
            )
            native = native_modalities_from_dense_evidence(
                (evidence_row,),
                bindings,
                device="cpu",
                dtype=torch.float32,
            )
            native.validate_against(tuple(binding.native_spec for binding in bindings))
            for evidence in evidence_row:
                token_counts[evidence.modality] += evidence.token_count
                available_counts[evidence.modality] += int(evidence.available)
                current_counts[evidence.modality] += int(
                    evidence.effective_current_measurement_valid.any()
                )
            visit_count += 1
            unique_sources.add(source_global_index)

    expected_visit_count = coverage.evaluation_history_visit_count + sum(
        item.transition_index >= args.history_transitions for item in plan.items
    )
    if visit_count != expected_visit_count:
        raise RuntimeError("verified visit count differs from causal-warm coverage contract")
    source_counts = {
        cache.contract.modality: list(cache.source_record_counts)
        for cache in union.caches
        if isinstance(cache, FrozenDenseEvidenceCacheView)
    }
    expected_source_counts = [
        len(primary.caches[0].records),
        len(supplement.caches[0].records),
    ]
    if any(counts != expected_source_counts for counts in source_counts.values()):
        raise RuntimeError("dense union selected unexpected primary/supplement record counts")

    report = {
        "available_visit_count_by_modality": dict(sorted(available_counts.items())),
        "coverage_artifact_sha256": coverage.artifact_sha256,
        "coverage_file_sha256": _sha256(args.coverage_plan),
        "current_measurement_visit_count_by_modality": dict(sorted(current_counts.items())),
        "eligible_evaluation_item_count": sum(
            item.transition_index >= args.history_transitions for item in plan.items
        ),
        "history_transitions": args.history_transitions,
        "selected_record_count_by_modality_and_source": source_counts,
        "token_count_by_modality": dict(sorted(token_counts.items())),
        "unique_verified_source_count": len(unique_sources),
        "verified_visit_count": visit_count,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

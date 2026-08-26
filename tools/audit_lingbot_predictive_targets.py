#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Audit LingBot predictive target diversity before model training."""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="predictive target audit",
)

import numpy as np
import torch

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.lingbot_native.predictive_cache import LingBotPredictiveTargetCache
from picf_next.lingbot_native.predictive_diagnostics import (
    PREDICTIVE_TARGET_AUDIT_SCHEMA,
    predictive_latent_diagnostics,
    predictive_target_pretraining_readiness,
    predictive_visible_support_diagnostics,
)

_SAMPLE_SCHEMA = b"picf-next.lingbot-predictive-target-sample/v1\0"


def _sample_priority(*, source: int, horizon: int, target_digest: str, identity: str) -> int:
    payload = (
        _SAMPLE_SCHEMA
        + source.to_bytes(8, byteorder="big", signed=False)
        + horizon.to_bytes(8, byteorder="big", signed=False)
        + target_digest.encode("ascii")
        + b"\0"
        + identity.encode("utf-8")
    )
    return int.from_bytes(hashlib.sha256(payload).digest(), byteorder="big", signed=False)


def audit_predictive_target_cache(
    cache: LingBotPredictiveTargetCache,
    *,
    maximum_samples: int,
) -> dict[str, object]:
    """Scan complete coverage and diagnose one deterministic bounded sample."""

    if not isinstance(cache, LingBotPredictiveTargetCache):
        raise TypeError("predictive target audit requires a loaded typed cache")
    if (
        isinstance(maximum_samples, bool)
        or not isinstance(maximum_samples, int)
        or maximum_samples < 2
    ):
        raise ValueError("predictive target audit requires at least two samples")

    # The heap retains the lowest content-derived priorities without depending
    # on shard size or machine RNG state.
    retained: list[tuple[int, int, str, str, float, np.ndarray]] = []
    serial = 0
    scanned_records = 0
    scanned_objects = 0
    supported_objects = 0
    total_positive_importance = 0.0
    minimum_positive_importance = float("inf")
    maximum_positive_importance = 0.0
    horizon_records: Counter[int] = Counter()
    identity_support: Counter[str] = Counter()
    selection_digest = hashlib.sha256()
    for record in cache.iter_records():
        scanned_records += 1
        horizon_records[record.horizon] += 1
        for object_index, identity in enumerate(record.identity_keys):
            scanned_objects += 1
            importance = float(record.importance[object_index])
            if importance <= 0:
                continue
            supported_objects += 1
            total_positive_importance += importance
            minimum_positive_importance = min(minimum_positive_importance, importance)
            maximum_positive_importance = max(maximum_positive_importance, importance)
            identity_support[identity] += 1
            priority = _sample_priority(
                source=record.source_global_index,
                horizon=record.horizon,
                target_digest=record.target_rgb_sha256,
                identity=identity,
            )
            item = (
                -priority,
                serial,
                identity,
                record.target_rgb_sha256,
                importance,
                record.features[object_index].astype(np.float32, copy=True),
            )
            serial += 1
            if len(retained) < maximum_samples:
                heapq.heappush(retained, item)
            elif item[0] > retained[0][0]:
                heapq.heapreplace(retained, item)

    if scanned_records != cache.contract.expected_record_count or not set(horizon_records).issubset(
        cache.contract.horizons
    ):
        raise RuntimeError("predictive target audit did not scan complete cache coverage")
    if supported_objects < 2 or len(retained) < 2:
        raise RuntimeError("predictive target cache has fewer than two supported object targets")

    ordered = sorted(retained, key=lambda value: (-value[0], value[1]))
    identities = tuple(value[2] for value in ordered)
    groups = tuple(value[3] for value in ordered)
    features = torch.from_numpy(np.stack(tuple(value[5] for value in ordered)))
    importance_diagnostics = predictive_visible_support_diagnostics(
        torch.tensor(tuple(value[4] for value in ordered), dtype=torch.float32),
        supported_count=supported_objects,
        total_importance=total_positive_importance,
        minimum_importance=minimum_positive_importance,
        maximum_importance=maximum_positive_importance,
    )
    for negative_priority, _serial, identity, group, _importance, _feature in ordered:
        selection_digest.update((-negative_priority).to_bytes(32, byteorder="big"))
        selection_digest.update(identity.encode("utf-8") + b"\0")
        selection_digest.update(group.encode("ascii") + b"\0")
    diagnostics = predictive_latent_diagnostics(
        features,
        identity_keys=identities,
        target_group_keys=groups,
    )
    ready, readiness_failures = predictive_target_pretraining_readiness(diagnostics)
    numerical_status = (
        "obvious_target_collapse"
        if diagnostics.obvious_numerical_collapse
        else "no_obvious_numerical_collapse"
    )
    return {
        "cache_contract": asdict(cache.contract),
        "cache_manifest_sha256": cache.manifest_sha256,
        "diagnostics": diagnostics.as_dict(),
        "encoder_digest": cache.contract.encoder_digest,
        "horizon_record_counts": {
            str(horizon): horizon_records[horizon] for horizon in cache.contract.horizons
        },
        "identity_count": len(identity_support),
        "visible_support_diagnostics": importance_diagnostics.as_dict(),
        "interpretation": {
            "numerical_status": numerical_status,
            "pretraining_readiness": "PASS" if ready else "FAIL",
            "pretraining_readiness_failures": list(readiness_failures),
            "retrieval_is_computable": diagnostics.retrieval_query_count > 0,
            "scientific_acceptance": False,
            "scientific_acceptance_reason": (
                "target statistics cannot establish source-conditioned learnability, "
                "shared-host gradient reach, object semantics or action benefit"
            ),
        },
        "maximum_samples": maximum_samples,
        "sample_selection": "lowest-sha256-priority-without-replacement/v1",
        "sample_selection_sha256": selection_digest.hexdigest(),
        "sampled_target_count": len(ordered),
        "scanned_object_target_count": scanned_objects,
        "scanned_record_count": scanned_records,
        "schema": PREDICTIVE_TARGET_AUDIT_SCHEMA,
        "supported_object_target_count": supported_objects,
        "zero_support_object_target_count": scanned_objects - supported_objects,
    }


def _write_json_durable(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, allow_nan=False, ensure_ascii=True, indent=2, sort_keys=True) + "\n"
    ).encode("ascii")
    write_bytes_durable_exclusive(path, payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--dataset-tree-sha256", required=True)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--encoder-digest", required=True)
    parser.add_argument("--query-schema-sha256", required=True)
    parser.add_argument("--coverage-sha256", required=True)
    parser.add_argument("--maximum-samples", default=2048, type=int)
    parser.add_argument("--memory-capacity", default=1, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cache = LingBotPredictiveTargetCache.load(
        args.cache_root,
        manifest_sha256=args.cache_manifest_sha256,
        dataset_tree_sha256=args.dataset_tree_sha256,
        physical_sidecar_manifest_sha256=args.physical_sidecar_manifest_sha256,
        encoder_digest=args.encoder_digest,
        query_schema_sha256=args.query_schema_sha256,
        coverage_sha256=args.coverage_sha256,
        memory_capacity=args.memory_capacity,
    )
    report = audit_predictive_target_cache(cache, maximum_samples=args.maximum_samples)
    _write_json_durable(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

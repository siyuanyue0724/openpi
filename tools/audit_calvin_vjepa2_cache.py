#!/usr/bin/env python3
"""Audit every frozen V-JEPA2 cache entry through its production consumer."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
from picf_next.data.causal_video import build_calvin_causal_video_clip
from picf_next.data.dataset_manifest import DatasetFileManifest, load_dataset_file_manifest
from picf_next.data.vjepa2_cache import VJEPA2_CONTEXT_SENSORS, Vjepa2FeatureCache

AUDIT_SCHEMA = "picf-next.vjepa2-causal-token-cache-audit/v1"


def _write_json_atomic(path: Path, payload: object) -> None:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def audit_cache(
    *,
    dataset: CalvinStatefulTransitionDataset,
    cache: Vjepa2FeatureCache,
    cache_manifest_sha256: str,
) -> dict[str, object]:
    """Rebuild source clips and force a verified read of every cached tensor."""

    sample_keys = tuple(sorted(dataset.sample_keys))
    if tuple(cache.entries) != sample_keys:
        raise RuntimeError("V-JEPA2 cache keys differ from the complete dataset sample plan")

    started = time.perf_counter()
    frame_histogram: Counter[int] = Counter()
    token_histogram: Counter[int] = Counter()
    modality_tokens: Counter[str] = Counter()
    artifact_reads = 0
    total_token_rows = 0
    current_measurement_rows = 0

    for sample_key in sample_keys:
        prefix = dataset.evidence_prefix_by_key(
            sample_key,
            maximum_source_frames=cache.maximum_frames,
        )
        clips = {
            sensor_key: build_calvin_causal_video_clip(
                prefix,
                sensor_key=sensor_key,
                maximum_frames=cache.maximum_frames,
                tubelet_size=cache.tubelet_size,
            )
            for sensor_key, _modality in VJEPA2_CONTEXT_SENSORS
        }
        evidence = cache.evidence_for(sample_key, clips)
        if len(evidence) != len(VJEPA2_CONTEXT_SENSORS):
            raise RuntimeError("V-JEPA2 cache consumer omitted one configured camera")

        sample_tokens = 0
        for item, (sensor_key, modality) in zip(
            evidence,
            VJEPA2_CONTEXT_SENSORS,
            strict=True,
        ):
            clip = clips[sensor_key]
            frame_count = 0 if clip is None else len(clip.images)
            frame_histogram[frame_count] += 1
            if item.modality != modality or item.encoder_contract != cache.encoder_contract:
                raise RuntimeError("V-JEPA2 evidence identity differs from its cache contract")
            if item.available != (clip is not None):
                raise RuntimeError("V-JEPA2 availability differs from its causal source clip")
            token_count = int(item.tokens.shape[0])
            expected_count = (
                frame_count // cache.tubelet_size * (cache.image_size // cache.patch_size) ** 2
            )
            if item.tokens.shape != (expected_count, cache.hidden_size):
                raise RuntimeError("V-JEPA2 token tensor differs from its causal clip geometry")
            if token_count != expected_count:
                raise RuntimeError("V-JEPA2 token count differs from its causal clip")
            if item.geometry is None or item.geometry.shape != (token_count, 3):
                raise RuntimeError("V-JEPA2 dense geometry is absent or misaligned")
            if item.timestamps.shape != (token_count,) or item.confidence.shape != (token_count,):
                raise RuntimeError("V-JEPA2 timestamp/confidence rows are misaligned")
            if item.effective_current_measurement_valid.shape != (token_count,):
                raise RuntimeError("V-JEPA2 evidence-role rows are misaligned")
            numeric = (item.tokens, item.geometry, item.timestamps, item.confidence)
            if any(not np.isfinite(np.asarray(value)).all() for value in numeric):
                raise RuntimeError("V-JEPA2 consumer produced non-finite evidence")
            if clip is not None and token_count:
                if float(item.timestamps.max()) > clip.current_timestamp_s + 1e-7:
                    raise RuntimeError("V-JEPA2 token timestamp is later than its action time")
                artifact_reads += 1
            current_rows = int(item.effective_current_measurement_valid.sum())
            if current_rows:
                raise RuntimeError("causal V-JEPA2 context leaked into current measurement updates")
            current_measurement_rows += current_rows
            sample_tokens += token_count
            modality_tokens[modality] += token_count
            total_token_rows += token_count
        token_histogram[sample_tokens] += 1

    return {
        "artifact_reads": artifact_reads,
        "cache_manifest_sha256": cache_manifest_sha256,
        "complete": True,
        "current_measurement_rows": current_measurement_rows,
        "dataset_tree_sha256": cache.dataset_tree_sha256,
        "elapsed_seconds": time.perf_counter() - started,
        "encoder_contract": cache.encoder_contract,
        "frame_count_histogram_across_sensors": {
            str(count): frequency for count, frequency in sorted(frame_histogram.items())
        },
        "modality_token_rows": dict(sorted(modality_tokens.items())),
        "samples": len(sample_keys),
        "schema": AUDIT_SCHEMA,
        "token_count_histogram_per_sample": {
            str(count): frequency for count, frequency in sorted(token_histogram.items())
        },
        "total_token_rows": total_token_rows,
    }


def load_audit_dataset(
    *,
    dataset_root: Path,
    split: str,
    manifest: DatasetFileManifest,
) -> CalvinStatefulTransitionDataset:
    """Load the manifest-bound sample plan without an O(dataset) path scan.

    Every frame consumed by :func:`audit_cache` is still read and hashed through
    the dataset manifest. Repeating ``Path.is_file`` for all 1.8M CALVIN files
    adds no content validation and is prohibitive on persistent object storage.
    """

    index = CalvinDatasetIndex.load(
        (dataset_root / split).resolve(),
        dataset_id=manifest.dataset_id,
        dataset_revision=manifest.dataset_revision,
        verify_files=False,
        dataset_manifest=manifest,
    )
    return CalvinStatefulTransitionDataset(index, action_horizon=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--dataset-manifest", required=True, type=Path)
    parser.add_argument("--split", default="training", choices=("training", "validation"))
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--cache-manifest-sha256", required=True)
    parser.add_argument("--memory-capacity", default=1, type=int)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    dataset = load_audit_dataset(
        dataset_root=args.dataset_root.resolve(),
        split=args.split,
        manifest=manifest,
    )
    cache = Vjepa2FeatureCache.load(
        args.cache_root,
        manifest_sha256=args.cache_manifest_sha256,
        dataset_tree_sha256=manifest.tree_sha256,
        memory_capacity=args.memory_capacity,
    )
    report = audit_cache(
        dataset=dataset,
        cache=cache,
        cache_manifest_sha256=args.cache_manifest_sha256,
    )
    if args.output is not None:
        _write_json_atomic(args.output.resolve(), report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Audit CALVIN language data through the production transition decoder."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from picf_next.data.calvin import (
    CALVIN_DEBUG_DATASET_ID,
    CALVIN_DEBUG_REVISION,
    CalvinDatasetIndex,
    CalvinMolmoAct2Dataset,
    CalvinPosteriorWindowDataset,
)
from picf_next.data.dataset_manifest import (
    DatasetFileManifest,
    load_dataset_file_manifest,
    validate_dataset_files,
)


def _audit_split(
    split_root: Path,
    *,
    dataset_id: str,
    dataset_revision: str,
    dataset_manifest: DatasetFileManifest,
) -> dict:
    split_root = split_root.resolve()
    validate_dataset_files(
        dataset_manifest,
        split_root,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        split_name=split_root.name,
        verify_hashes=True,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        verify_files=True,
        dataset_manifest=dataset_manifest,
    )
    task_counts = Counter(segment.task_key for segment in index.segments)
    unique_steps: set[int] = set()
    source_action_min = np.full(7, np.inf, dtype=np.float64)
    source_action_max = np.full(7, -np.inf, dtype=np.float64)
    source_tactile_nonzero_frames = 0
    validated_source_frames = 0
    for episode in index.episodes:
        for step in range(episode.start, episode.end + 1):
            frame = index.validated_source_frame_arrays(
                step,
                fields=("rel_actions", "depth_tactile"),
            )
            validated_source_frames += 1
            source_action_min = np.minimum(source_action_min, frame["rel_actions"])
            source_action_max = np.maximum(source_action_max, frame["rel_actions"])
            source_tactile_nonzero_frames += int(np.count_nonzero(frame["depth_tactile"]) > 0)

    decoded_records = 0
    for segment in index.segments:
        for record in index.iter_segment(segment.index):
            decoded_records += 1
            unique_steps.add(record.global_index)

    host_dataset = CalvinMolmoAct2Dataset(index, action_horizon=30)
    posterior_dataset = CalvinPosteriorWindowDataset(index, sequence_length=4)
    return {
        "split": split_root.name,
        "dataset_tree_sha256": dataset_manifest.tree_sha256,
        "source_episodes": len(index.episodes),
        "source_frames": sum(episode.length for episode in index.episodes),
        "validated_source_frames": validated_source_frames,
        "language_segments": len(index.segments),
        "language_records_with_overlap": decoded_records,
        "unique_language_frames": len(unique_steps),
        "language_transitions": len(host_dataset),
        "posterior_windows_2_context_2_gradient": len(posterior_dataset),
        "task_segment_counts": dict(sorted(task_counts.items())),
        "source_action_min": source_action_min.tolist(),
        "source_action_max": source_action_max.tolist(),
        "source_frames_with_nonzero_tactile_depth": source_tactile_nonzero_frames,
        "tactile_note": (
            "nonzero deformation is reported as source observability only; "
            "it is not promoted to an audited contact-active label"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", action="append", choices=("training", "validation"))
    parser.add_argument(
        "--dataset-manifest",
        action="append",
        required=True,
        type=Path,
        help="content-addressed split manifest; repeat for every selected split",
    )
    parser.add_argument("--dataset-id", default=CALVIN_DEBUG_DATASET_ID)
    parser.add_argument("--dataset-revision", default=CALVIN_DEBUG_REVISION)
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    splits = tuple(args.split or ("training", "validation"))
    manifests: dict[str, DatasetFileManifest] = {}
    for path in args.dataset_manifest:
        manifest = load_dataset_file_manifest(path.resolve())
        if manifest.split_name in manifests:
            raise ValueError(f"duplicate dataset manifest for split {manifest.split_name}")
        manifests[manifest.split_name] = manifest
    missing = sorted(set(splits) - manifests.keys())
    if missing:
        raise ValueError(f"missing dataset manifests for splits: {missing}")
    records = [
        _audit_split(
            root / split,
            dataset_id=args.dataset_id,
            dataset_revision=args.dataset_revision,
            dataset_manifest=manifests[split],
        )
        for split in splits
    ]
    report = {
        "format": "picf-next.calvin-dataset-audit/v2",
        "dataset_id": args.dataset_id,
        "dataset_revision": args.dataset_revision,
        "runtime_target_fields": [],
        "splits": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

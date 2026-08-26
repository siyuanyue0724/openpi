#!/usr/bin/env python3
"""Build the exact tree-bound CALVIN q01-q99 normalization artifact."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_normalization import (
    build_calvin_normalization_artifact,
    write_calvin_normalization_artifact,
)
from picf_next.data.dataset_manifest import (
    load_dataset_file_manifest,
    validate_dataset_runtime_binding,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    started = time.monotonic()
    next_report = 10_000

    def report_progress(completed: int, total: int) -> None:
        nonlocal next_report
        if completed < next_report and completed != total:
            return
        elapsed = max(time.monotonic() - started, 1e-9)
        print(
            json.dumps(
                {
                    "completed_unique_source_frames": completed,
                    "elapsed_seconds": round(elapsed, 3),
                    "event": "calvin_normalization_progress",
                    "frames_per_second": round(completed / elapsed, 3),
                    "total_unique_source_frames": total,
                    "workers": args.workers,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        while next_report <= completed:
            next_report += 10_000

    split_root = args.split_root.resolve()
    dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    validate_dataset_runtime_binding(
        dataset_manifest,
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=split_root.name,
    )
    index = CalvinDatasetIndex.load(
        split_root,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        verify_files=False,
        dataset_manifest=dataset_manifest,
    )
    payload = build_calvin_normalization_artifact(
        index,
        maximum_workers=args.workers,
        progress_callback=report_progress,
    )
    write_calvin_normalization_artifact(payload, args.output)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

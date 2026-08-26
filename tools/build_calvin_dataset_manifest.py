#!/usr/bin/env python3
"""Build a content-addressed manifest for the official CALVIN source split."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import build_dataset_file_manifest

_EPISODE_NAME = re.compile(r"episode_[0-9]{7}\.npz")
_CALVIN_METADATA_PATHS = (
    ".hydra/merged_config.yaml",
    "ep_lens.npy",
    "ep_start_end_ids.npy",
    "lang_annotations/auto_lang_ann.npy",
    "scene_info.npy",
)


def _atomic_write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    write_bytes_durable_exclusive(path, encoded)


def _calvin_source_relative_paths(split: Path) -> tuple[str, ...]:
    """Inventory CALVIN source bytes without decoding object-array metadata."""

    episode_names = []
    for candidate in split.glob("episode_*.npz"):
        if _EPISODE_NAME.fullmatch(candidate.name) is None:
            raise ContractError(f"CALVIN episode filename is not canonical: {candidate.name}")
        episode_names.append(candidate.name)
    if not episode_names:
        raise ContractError("CALVIN split contains no canonical episode archives")
    return (*_CALVIN_METADATA_PATHS, *sorted(episode_names))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    split = args.split_root.resolve()
    started = time.monotonic()
    next_report = 50_000

    def report_progress(completed: int, total: int) -> None:
        nonlocal next_report
        if completed < next_report and completed != total:
            return
        elapsed = max(time.monotonic() - started, 1e-9)
        print(
            json.dumps(
                {
                    "completed_files": completed,
                    "elapsed_seconds": round(elapsed, 3),
                    "event": "calvin_manifest_progress",
                    "files_per_second": round(completed / elapsed, 3),
                    "total_files": total,
                    "workers": args.workers,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=sys.stderr,
            flush=True,
        )
        while next_report <= completed:
            next_report += 50_000

    manifest = build_dataset_file_manifest(
        split,
        dataset_id=args.dataset_id,
        dataset_revision=args.dataset_revision,
        split_name=split.name,
        relative_paths=_calvin_source_relative_paths(split),
        maximum_workers=args.workers,
        progress_callback=report_progress,
    )
    output = args.output.resolve()
    _atomic_write_json(output, manifest.to_dict())
    print(
        json.dumps(
            {
                "file_count": len(manifest.files),
                "output": str(output),
                "total_size_bytes": manifest.total_size_bytes,
                "tree_sha256": manifest.tree_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


if __name__ == "__main__":
    main()

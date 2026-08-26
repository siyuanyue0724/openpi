#!/usr/bin/env python3
"""Translate an audited CALVIN training artifact to LingBot normalization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picf_next.artifact_io import write_bytes_durable_exclusive
from picf_next.data.calvin_normalization import (
    load_calvin_normalization_artifact,
    official_lingbot_calvin_norm_stats,
)
from picf_next.data.dataset_manifest import load_dataset_file_manifest


def _write_atomic(payload: dict[str, object], destination: Path) -> None:
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("ascii") + b"\n"
    write_bytes_durable_exclusive(destination, encoded)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-normalization", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = load_calvin_normalization_artifact(args.calvin_normalization.resolve())
    manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
    if (
        source["dataset_id"] != manifest.dataset_id
        or source["dataset_revision"] != manifest.dataset_revision
        or source.get("dataset_tree_sha256") != manifest.tree_sha256
    ):
        raise ValueError("CALVIN normalization and dataset manifest provenance differs")
    payload = official_lingbot_calvin_norm_stats(
        source,
        dataset_tree_sha256=manifest.tree_sha256,
    )
    _write_atomic(payload, args.output)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# ruff: noqa: E402
"""Measure how much a label-only row permutation recovers a representation snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(__file__, entrypoint_name="row permutation audit")

from picf_next.lingbot_native.representation_row_permutation import (
    audit_representation_row_permutation_snapshot,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    with args.snapshot.open(encoding="ascii") as stream:
        snapshot = json.load(stream)
    result = audit_representation_row_permutation_snapshot(snapshot)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii")
    print(json.dumps({"artifact_sha256": result["artifact_sha256"], "output": str(args.output)}))


if __name__ == "__main__":
    main()

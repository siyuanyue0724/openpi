#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Finalize full-tail CALVIN physical supervision after complete image review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

_REPOSITORY_ROOT = bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="physical visual finalizer",
)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.data.calvin_physical_visual_acceptance import (
    build_calvin_physical_visual_acceptance,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--audit-manifest-sha256", required=True)
    parser.add_argument("--dataset-manifest-sha256", required=True)
    parser.add_argument("--sidecar-manifest-sha256", required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--review-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    acceptance = build_calvin_physical_visual_acceptance(
        audit_manifest_path=args.audit_manifest,
        audit_manifest_sha256=args.audit_manifest_sha256,
        dataset_manifest_sha256=args.dataset_manifest_sha256,
        sidecar_manifest_sha256=args.sidecar_manifest_sha256,
        review_path=args.review,
        review_sha256=args.review_sha256,
        require_pass=args.require_pass,
    )
    write_text_durable_exclusive(
        args.output,
        json.dumps(acceptance, indent=2, sort_keys=True) + "\n",
        encoding="ascii",
    )
    print(json.dumps(acceptance, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

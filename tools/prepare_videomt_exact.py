#!/usr/bin/env python3
"""Prepare the exact local DINOv3 constructor bundle for released VidEoMT."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from picf_next.videomt_exact.checkpoint import (
    build_local_dinov3_bundle,
    inspect_published_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    receipt = inspect_published_checkpoint(args.checkpoint)
    output = build_local_dinov3_bundle(
        args.checkpoint,
        args.output,
        force=args.force,
    )
    print(json.dumps({"checkpoint": asdict(receipt), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()

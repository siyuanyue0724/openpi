#!/usr/bin/env python3
"""Recompute immutable G2-G6 observations from one producer bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from picf_next.lingbot_native.empirical_producers import (
    build_empirical_observations_from_producer,
)

try:
    from tools.run_lingbot_vla2_native_g0 import _write_text_durable
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from run_lingbot_vla2_native_g0 import _write_text_durable  # type: ignore[no-redef]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--producer", type=Path, required=True)
    parser.add_argument("--producer-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    observations = build_empirical_observations_from_producer(
        args.producer,
        expected_sha256=args.producer_sha256,
    )
    payload = json.dumps(observations, allow_nan=False, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(
        json.dumps(
            {
                "gate": observations["gate"],
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build one G2-G6 report from hash-bound paired episode observations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from picf_next.lingbot_native.empirical_statistics import (
    EMPIRICAL_REPORT_SCHEMAS,
    build_empirical_gate_report_from_observations,
)

try:
    from tools.run_lingbot_vla2_native_g0 import _write_text_durable
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from run_lingbot_vla2_native_g0 import _write_text_durable  # type: ignore[no-redef]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", choices=tuple(EMPIRICAL_REPORT_SCHEMAS), required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--observations-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-pass",
        action="store_true",
        help="Refuse to publish a report whose preregistered comparisons fail.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    report = build_empirical_gate_report_from_observations(
        args.observations,
        report_schema=EMPIRICAL_REPORT_SCHEMAS[args.gate],
        expected_sha256=args.observations_sha256,
    )
    if report["gate"] != args.gate:
        raise ValueError("requested gate differs from the empirical observations")
    if args.require_pass and report["status"] != "PASS":
        raise RuntimeError(f"{args.gate} failed its preregistered empirical criteria")
    payload = json.dumps(report, allow_nan=False, indent=2, sort_keys=True) + "\n"
    _write_text_durable(args.output, payload)
    print(
        json.dumps(
            {
                "gate": args.gate,
                "output": str(args.output.resolve()),
                "sha256": hashlib.sha256(payload.encode("ascii")).hexdigest(),
                "status": report["status"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

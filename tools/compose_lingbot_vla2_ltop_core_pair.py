#!/usr/bin/env python3
"""Compose exact factual/blocked LTOP core-pilot evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.ltop_core_pair import compose_ltop_core_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factual-report", type=Path, required=True)
    parser.add_argument("--blocked-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be one absent direct path")
    report = compose_ltop_core_pair(
        factual_path=args.factual_report,
        blocked_path=args.blocked_report,
    )
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps({"output": str(args.output.resolve()), "status": "PASS"}, sort_keys=True))


if __name__ == "__main__":
    main()

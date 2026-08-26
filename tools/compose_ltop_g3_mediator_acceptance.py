#!/usr/bin/env python3
"""Compose the four immutable ADR165 mediator G3 acceptance artifacts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.ltop_g3_mediator_acceptance import (
    compose_ltop_g3_mediator_acceptance,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--arm-validation", type=Path, required=True)
    parser.add_argument("--action-evaluation-report", type=Path, required=True)
    parser.add_argument("--retention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must be one absent direct path")
    report = compose_ltop_g3_mediator_acceptance(
        training_path=args.training_report,
        arm_validation_path=args.arm_validation,
        action_evaluation_path=args.action_evaluation_report,
        retention_path=args.retention_report,
    )
    write_text_durable_exclusive(
        args.output,
        json.dumps(
            report,
            allow_nan=False,
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="ascii",
    )
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "schema": report["schema"],
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Summarize lifecycle coverage/calibration from exact M3 temporal replays."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from picf_next.eval.lifecycle import audit_lifecycle_reports  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    reports = []
    for path in args.report:
        try:
            payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid temporal report: {path}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"temporal report must be a JSON object: {path}")
        reports.append(payload)
    result = audit_lifecycle_reports(reports)
    result["source_reports"] = [str(path.expanduser().resolve()) for path in args.report]
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

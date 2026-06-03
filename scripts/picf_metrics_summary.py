#!/usr/bin/env python3
"""Summarize key PICF metrics from one or more metrics.jsonl files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_FIELDS = (
    "loss_action_default_equiv",
    "loss_action_active7",
    "loss_total",
    "loss_total_minus_action",
    "loss_anchor_pv",
    "loss_anchor_object_pull",
    "loss_mapg_routing",
    "loss_slot_jepa",
    "grad_norm",
    "logical_batch_distinct_bucket_count",
    "aqr_active_same_role_support_overlap_max",
    "aqr_downstream_same_role_support_overlap_max",
)


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _summarize(path: Path, fields: tuple[str, ...]) -> None:
    rows = _load_rows(path)
    print(f"== {path} ==")
    print(f"rows={len(rows)}")
    if not rows:
        return
    step_key = "step" if "step" in rows[-1] else None
    if step_key is not None:
        print(f"step_first={rows[0].get(step_key)} step_last={rows[-1].get(step_key)}")
    for field in fields:
        series = [value for row in rows if (value := _to_float(row.get(field))) is not None]
        if not series:
            continue
        first = series[0]
        last = series[-1]
        min_value = min(series)
        max_value = max(series)
        tail = series[-min(5, len(series)) :]
        print(
            f"{field}: first={first:.6g} last={last:.6g} min={min_value:.6g} "
            f"max={max_value:.6g} tail_mean={mean(tail):.6g} delta={last - first:+.6g}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path)
    parser.add_argument("--fields", default=",".join(DEFAULT_FIELDS))
    args = parser.parse_args()
    fields = tuple(part.strip() for part in str(args.fields).split(",") if part.strip())
    for path in args.metrics:
        _summarize(path, fields)


if __name__ == "__main__":
    main()


#!/usr/bin/env python3
"""Summarize PICF anchor overlay JSON artifacts.

The trainer writes one JSON sidecar for each anchor overlay snapshot. This
script keeps the visual inspection honest by separating fixed reserve capacity
from action-visible active posterior files.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        out = float(value)
        return out if math.isfinite(out) else default
    except (TypeError, ValueError):
        return default


def _pixel(record: dict[str, Any]) -> tuple[float, float] | None:
    xy = record.get("pixel_xy")
    if not isinstance(xy, list | tuple) or len(xy) < 2:
        return None
    x = _as_float(xy[0], default=float("nan"))
    y = _as_float(xy[1], default=float("nan"))
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def _min_pair_distance(records: list[dict[str, Any]]) -> float | None:
    coords = [_pixel(r) for r in records if _pixel(r) is not None]
    if len(coords) < 2:
        return None
    best = float("inf")
    for i, (x0, y0) in enumerate(coords):
        for x1, y1 in coords[i + 1 :]:
            best = min(best, math.hypot(x0 - x1, y0 - y1))
    return best if math.isfinite(best) else None


def summarize_overlay(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    anchors = payload.get("anchors", [])
    if not isinstance(anchors, list):
        anchors = []
    posterior = [r for r in anchors if isinstance(r, dict) and r.get("source") == "posterior"]
    task = [r for r in anchors if isinstance(r, dict) and r.get("source") == "task"]
    visible = [r for r in posterior if bool(r.get("visible"))]
    active = [r for r in visible if _as_float(r.get("active"), default=1.0) > 0.5]
    inactive = [r for r in visible if _as_float(r.get("active"), default=1.0) <= 0.5]
    demoted = [r for r in visible if _as_float(r.get("file_competition_demoted_mass"), default=0.0) > 1.0e-6]
    proposals = payload.get("proposals", [])
    if not isinstance(proposals, list):
        proposals = []
    static_proposals = [r for r in proposals if isinstance(r, dict) and int(r.get("view_id", -1)) == 0]
    wrist_proposals = [r for r in proposals if isinstance(r, dict) and int(r.get("view_id", -1)) == 1]

    by_role: dict[int, list[dict[str, Any]]] = defaultdict(list)
    inactive_by_role: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for rec in active:
        by_role[int(rec.get("role", -1))].append(rec)
    for rec in inactive:
        inactive_by_role[int(rec.get("role", -1))].append(rec)

    role_summary: dict[str, Any] = {}
    for role, rows in sorted(by_role.items()):
        role_summary[str(role)] = {
            "active_visible": len(rows),
            "inactive_visible": len(inactive_by_role.get(role, [])),
            "min_active_pixel_distance": _min_pair_distance(rows),
            "active_indices": [int(r.get("index", -1)) for r in rows],
        }

    return {
        "path": str(path),
        "step": payload.get("step"),
        "prompt": payload.get("prompt"),
        "image_variants": payload.get("image_variants", {}),
        "posterior_visible": len(visible),
        "posterior_active_visible": len(active),
        "posterior_inactive_visible": len(inactive),
        "posterior_demoted_visible": len(demoted),
        "task_visible": len([r for r in task if bool(r.get("visible"))]),
        "proposal_static_count": len(static_proposals),
        "proposal_wrist_count": len(wrist_proposals),
        "proposal_objectness_top3": sorted([_as_float(r.get("objectness")) for r in proposals], reverse=True)[:3],
        "min_active_pixel_distance_all_roles": _min_pair_distance(active),
        "role_summary": role_summary,
        "debug": payload.get("debug", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="Overlay JSON files or directories containing step_*.json.")
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    json_paths: list[Path] = []
    for path in args.paths:
        if path.is_dir():
            json_paths.extend(sorted(path.glob("step_*.json")))
        else:
            json_paths.append(path)
    summaries = [summarize_overlay(path) for path in json_paths]
    print(json.dumps(summaries if len(summaries) != 1 else summaries[0], indent=args.indent, sort_keys=True))


if __name__ == "__main__":
    main()

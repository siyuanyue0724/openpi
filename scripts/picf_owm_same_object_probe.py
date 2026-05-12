#!/usr/bin/env python3
"""Offline weak IsSameObject probe for PICF anchor debug artifacts.

This script intentionally uses only exported debug artifacts. It is a diagnostic
probe, not a training loss. The goal is to decide whether current AQR/MVTrack
evidence already contains a decodable same-object signal, or whether the next
engineering step must add stronger temporal evidence such as real tracklets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def _finite_vec(xs: Any, n: int) -> list[float] | None:
    if not isinstance(xs, list) or len(xs) < n:
        return None
    out: list[float] = []
    for x in xs[:n]:
        if not isinstance(x, (int, float)) or not math.isfinite(float(x)):
            return None
        out.append(float(x))
    return out


def _dist(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b, strict=True)))


def _sparse(items: Any) -> dict[int, float]:
    out: dict[int, float] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        weight = item.get("weight")
        if isinstance(idx, int) and isinstance(weight, (int, float)) and math.isfinite(float(weight)):
            out[idx] = out.get(idx, 0.0) + max(float(weight), 0.0)
    return out


def _cosine(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    common = set(a).intersection(b)
    dot = sum(a[k] * b[k] for k in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (na * nb)


def _topk_jaccard(a: dict[int, float], b: dict[int, float]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa.intersection(sb)) / max(1, len(sa.union(sb)))


def _auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = 0.0
    total = len(pos) * len(neg)
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / total


def _anchor_rows(row: dict[str, Any]) -> list[dict[str, Any]]:
    ad = row.get("anchor_debug", {})
    obs = ad.get("observation", {})
    mapg = ad.get("mapg", {})
    xyzs = obs.get("xyz") or []
    pixels = obs.get("pixel") or []
    roles = obs.get("role_ids") or []
    visual_topk = ((mapg.get("visual_priors") or {}).get("topk") or [])
    point_topk = ((mapg.get("point_priors") or {}).get("topk") or [])
    anchors: list[dict[str, Any]] = []
    count = min(len(xyzs), len(pixels), len(roles), len(visual_topk), len(point_topk))
    for idx in range(count):
        xyz = _finite_vec(xyzs[idx], 3)
        pixel = _finite_vec(pixels[idx], 2)
        role = roles[idx]
        if xyz is None or pixel is None or not isinstance(role, int):
            continue
        anchors.append(
            {
                "index": idx,
                "role": role,
                "xyz": xyz,
                "pixel": pixel,
                "visual": _sparse(visual_topk[idx]),
                "point": _sparse(point_topk[idx]),
            }
        )
    return anchors


def _combined_score(a: dict[str, Any], b: dict[str, Any]) -> float:
    visual = _cosine(a["visual"], b["visual"])
    point = _cosine(a["point"], b["point"])
    xyz_d = _dist(a["xyz"], b["xyz"])
    pixel_d = _dist(a["pixel"], b["pixel"])
    geom = math.exp(-xyz_d / 0.08) * math.exp(-pixel_d / 30.0)
    return 0.35 * visual + 0.35 * point + 0.30 * geom


def run_probe(
    path: Path,
    *,
    pos_xyz_m: float,
    neg_xyz_m: float,
    pos_px: float,
    neg_px: float,
) -> dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    frames = [(row.get("episode"), row.get("step"), row.get("goal"), _anchor_rows(row)) for row in rows]

    scores: dict[str, dict[str, list[float]]] = {
        "visual_cos": {"pos": [], "neg": []},
        "point_cos": {"pos": [], "neg": []},
        "support_mean": {"pos": [], "neg": []},
        "geometry": {"pos": [], "neg": []},
        "combined": {"pos": [], "neg": []},
    }
    examples = {"positive": 0, "negative": 0, "ambiguous": 0}

    for (ep_a, _step_a, _goal_a, anchors_a), (ep_b, _step_b, _goal_b, anchors_b) in zip(frames, frames[1:], strict=False):
        if ep_a != ep_b:
            continue
        for a in anchors_a:
            for b in anchors_b:
                if a["role"] != b["role"]:
                    continue
                xyz_d = _dist(a["xyz"], b["xyz"])
                pixel_d = _dist(a["pixel"], b["pixel"])
                is_pos = xyz_d <= pos_xyz_m and pixel_d <= pos_px
                is_neg = xyz_d >= neg_xyz_m or pixel_d >= neg_px
                if not is_pos and not is_neg:
                    examples["ambiguous"] += 1
                    continue
                bucket = "pos" if is_pos else "neg"
                examples["positive" if is_pos else "negative"] += 1
                visual = _cosine(a["visual"], b["visual"])
                point = _cosine(a["point"], b["point"])
                geom = math.exp(-xyz_d / 0.08) * math.exp(-pixel_d / 30.0)
                scores["visual_cos"][bucket].append(visual)
                scores["point_cos"][bucket].append(point)
                scores["support_mean"][bucket].append(0.5 * (visual + point))
                scores["geometry"][bucket].append(geom)
                scores["combined"][bucket].append(_combined_score(a, b))

    duplicate_pairs = 0
    same_role_pairs = 0
    duplicate_visual_cos: list[float] = []
    duplicate_point_cos: list[float] = []
    for _ep, _step, _goal, anchors in frames:
        for i, a in enumerate(anchors):
            for b in anchors[i + 1 :]:
                if a["role"] != b["role"]:
                    continue
                same_role_pairs += 1
                xyz_d = _dist(a["xyz"], b["xyz"])
                pixel_d = _dist(a["pixel"], b["pixel"])
                if xyz_d <= pos_xyz_m and pixel_d <= pos_px:
                    duplicate_pairs += 1
                    duplicate_visual_cos.append(_cosine(a["visual"], b["visual"]))
                    duplicate_point_cos.append(_cosine(a["point"], b["point"]))

    metrics: dict[str, Any] = {
        "path": str(path),
        "frames": len(rows),
        "pair_examples": examples,
        "same_role_pairs_within_frame": same_role_pairs,
        "duplicate_candidate_pairs_within_frame": duplicate_pairs,
        "duplicate_candidate_fraction_within_frame": duplicate_pairs / same_role_pairs if same_role_pairs else None,
        "thresholds": {
            "pos_xyz_m": pos_xyz_m,
            "neg_xyz_m": neg_xyz_m,
            "pos_px": pos_px,
            "neg_px": neg_px,
        },
    }
    for name, buckets in scores.items():
        metrics[f"{name}_auc"] = _auc(buckets["pos"], buckets["neg"])
        metrics[f"{name}_pos_mean"] = sum(buckets["pos"]) / len(buckets["pos"]) if buckets["pos"] else None
        metrics[f"{name}_neg_mean"] = sum(buckets["neg"]) / len(buckets["neg"]) if buckets["neg"] else None
    metrics["duplicate_visual_cos_mean"] = (
        sum(duplicate_visual_cos) / len(duplicate_visual_cos) if duplicate_visual_cos else None
    )
    metrics["duplicate_point_cos_mean"] = (
        sum(duplicate_point_cos) / len(duplicate_point_cos) if duplicate_point_cos else None
    )

    combined_auc = metrics.get("combined_auc")
    dup_frac = metrics.get("duplicate_candidate_fraction_within_frame")
    if isinstance(combined_auc, float) and combined_auc >= 0.75 and isinstance(dup_frac, float) and dup_frac >= 0.20:
        decision = "same_object_signal_decodable_but_assignment_duplicates_candidates"
    elif isinstance(combined_auc, float) and combined_auc >= 0.75:
        decision = "same_object_signal_decodable"
    else:
        decision = "same_object_signal_weak_or_debug_features_insufficient"
    metrics["decision"] = decision
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--anchor-debug", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pos-xyz-m", type=float, default=0.04)
    parser.add_argument("--neg-xyz-m", type=float, default=0.12)
    parser.add_argument("--pos-px", type=float, default=12.0)
    parser.add_argument("--neg-px", type=float, default=35.0)
    args = parser.parse_args()

    metrics = run_probe(
        args.anchor_debug,
        pos_xyz_m=args.pos_xyz_m,
        neg_xyz_m=args.neg_xyz_m,
        pos_px=args.pos_px,
        neg_px=args.neg_px,
    )
    text = json.dumps(metrics, indent=2, sort_keys=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()

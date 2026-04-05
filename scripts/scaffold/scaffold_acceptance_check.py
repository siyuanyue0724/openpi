from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .scaffold_invariant_audit import run_invariant_audit
    from .scaffold_replay_smoke import run_smoke
    from .scaffold_stability_eval import run_stability_eval
except ImportError:
    from scaffold_invariant_audit import run_invariant_audit
    from scaffold_replay_smoke import run_smoke
    from scaffold_stability_eval import run_stability_eval


def run_acceptance_check(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 2,
    max_points: int = 2048,
    enable_rgb_identity: bool = False,
) -> dict:
    smoke = run_smoke(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        enable_rgb_identity=enable_rgb_identity,
    )
    invariants = run_invariant_audit(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        enable_rgb_identity=enable_rgb_identity,
    )
    stability = run_stability_eval(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
        enable_rgb_identity=enable_rgb_identity,
    )

    checks = {
        "invariants_pass": bool(invariants["all_pass"]),
        "reindex_failure_lt_5pct": bool(stability["baseline"]["reindex_failure_rate"] < 0.05),
        "stillness_jump_le_25pct": bool(stability["baseline"]["active_jump_rate"] <= 0.25),
        "photo_ratio_le_1p2x": bool(stability["photometric_vs_baseline_reindex_ratio"] <= 1.2),
        "no_birth_flood_under_dropout": bool(stability["point_dropout"]["birth_explosion_rate"] <= 0.25),
        "stale_timeout_bounded": bool(stability["invalid_depth_burst"]["stale_timeout_count"] <= 1),
        "smoke_no_hold": bool(smoke["hold_count"] == 0),
    }

    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "smoke": smoke,
        "invariants": invariants,
        "stability": stability,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineering acceptance gate for deterministic PICF scaffold.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--enable-rgb-identity", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_acceptance_check(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        num_segments=args.segments,
        stride=args.stride,
        max_points=args.max_points,
        enable_rgb_identity=args.enable_rgb_identity,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

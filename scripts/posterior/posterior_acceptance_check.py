from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .posterior_invariant_audit import run_invariant_audit
    from .posterior_replay_smoke import run_smoke
except ImportError:
    from posterior_invariant_audit import run_invariant_audit
    from posterior_replay_smoke import run_smoke


def run_acceptance_check(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 2048,
) -> dict:
    smoke = run_smoke(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
    )
    invariants = run_invariant_audit(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
    )
    checks = {
        "invariants_pass": bool(invariants["all_pass"]),
        "no_nans": bool(smoke["nan_count_total"] == 0),
        "point_gate_nonzero": bool(smoke["mean_point_gate_ratio"] > 0.0),
        "precision_gain_nonzero": bool(smoke["mean_precision_gain_count"] > 0.0),
        "var_bounds_valid": bool(smoke["min_var_block"] >= 1e-4 and smoke["max_var_block"] <= 10.0),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "smoke": smoke,
        "invariants": invariants,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Engineering acceptance gate for PICF posterior.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_acceptance_check(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        num_segments=args.segments,
        stride=args.stride,
        max_points=args.max_points,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

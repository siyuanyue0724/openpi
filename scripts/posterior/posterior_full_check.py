from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .posterior_acceptance_check import run_acceptance_check
    from .posterior_invariant_audit import run_invariant_audit
    from .posterior_replay_smoke import run_smoke
    from .posterior_spec_audit import run_spec_audit
except ImportError:
    from posterior_acceptance_check import run_acceptance_check
    from posterior_invariant_audit import run_invariant_audit
    from posterior_replay_smoke import run_smoke
    from posterior_spec_audit import run_spec_audit


def run_full_check(
    *,
    repo_root: str,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 256,
) -> dict:
    spec = run_spec_audit(repo_root)
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
    acceptance = run_acceptance_check(
        calvin_root=calvin_root,
        split=split,
        backend=backend,
        num_segments=num_segments,
        stride=stride,
        max_points=max_points,
    )
    checks = {
        "spec_pass": bool(spec["all_pass"]),
        "smoke_has_precision_gain": bool(smoke["mean_precision_gain_count"] > 0.0),
        "invariants_pass": bool(invariants["all_pass"]),
        "acceptance_pass": bool(acceptance["all_pass"]),
    }
    return {
        "checks": checks,
        "all_pass": all(checks.values()),
        "spec": spec,
        "smoke": smoke,
        "invariants": invariants,
        "acceptance": acceptance,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full posterior stage check for PICF.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=256)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_full_check(
        repo_root=args.repo_root,
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

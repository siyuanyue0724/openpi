#!/usr/bin/env python3
"""Run the deterministic 100-1000 step PICF posterior acceptance probe."""

from __future__ import annotations

import argparse
import json

import torch

from picf_next.eval.posterior_probe import run_long_horizon_posterior_probe


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_long_horizon_posterior_probe(
        steps=args.steps,
        seed=args.seed,
        dtype=getattr(torch, args.dtype),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build and audit one strict PICF training recipe without loading host weights."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from picf_next.training.recipe import load_training_recipe, write_preflight_report

_ROOT = Path(__file__).resolve().parents[1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("recipe", type=Path)
    parser.add_argument("--optimizer-steps", type=int, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--root", type=Path, default=_ROOT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    recipe = load_training_recipe(args.recipe)
    recipe.assert_optimizer_steps_authorized(args.optimizer_steps)
    report = recipe.local_preflight_report(args.root)
    report["requested_optimizer_steps"] = args.optimizer_steps
    if args.report is not None:
        write_preflight_report(recipe, args.report, root=args.root)
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()

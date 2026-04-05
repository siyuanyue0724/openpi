from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline import PointOnlyPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline


def run_smoke(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 2048,
) -> dict:
    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=None if num_segments is None else list(range(num_segments)),
    )
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    scaffold = DeterministicScaffoldPipeline(builder)
    posterior = PointOnlyPosteriorPipeline()

    scaffold_state = None
    posterior_state = None
    states = []
    for observation in replay:
        scaffold_state = scaffold.step(observation, scaffold_state)
        posterior_state = posterior.step(observation, scaffold_state, posterior_state)
        states.append(posterior_state)
    replay.close()
    if not states:
        raise RuntimeError("No posterior states were produced.")

    return {
        "frames": len(states),
        "mean_point_gate_ratio": sum(state.debug.point_gate_ratio for state in states) / len(states),
        "mean_precision_gain_count": sum(state.debug.precision_gain_count for state in states) / len(states),
        "max_abs_mu": max(state.debug.max_abs_mu for state in states),
        "min_var_block": min(state.debug.min_var_block for state in states),
        "max_var_block": max(state.debug.max_var_block for state in states),
        "stale_equal_count": sum(1 for state in states if state.debug.posterior_prior_equal_on_stale),
        "nan_count_total": sum(state.debug.nan_count for state in states),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential replay smoke test for PICF posterior.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_smoke(
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

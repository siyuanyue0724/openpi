from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline


def run_invariant_audit(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 1024,
    enable_rgb_identity: bool = True,
) -> dict:
    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=None if num_segments is None else list(range(num_segments)),
    )
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    config = DeterministicScaffoldConfig(v_rgb_identity=enable_rgb_identity)
    pipeline = DeterministicScaffoldPipeline(builder, config=config)

    previous = None
    frames = 0
    row_sum_violations = 0
    radius_violations = 0
    normal_violations = 0
    predecessor_violations = 0
    birth_violations = 0
    stale_birth_violations = 0
    worst_row_sum_error = 0.0
    worst_normal_error = 0.0

    for observation in replay:
        state = pipeline.step(observation, previous)
        frames += 1

        populated = np.sum(state.pi_geom, axis=1) > 0
        if np.any(populated):
            row_sums = state.pi_geom[populated].sum(axis=1)
            row_errors = np.abs(row_sums - 1.0)
            row_sum_violations += int(np.sum(row_errors > 1e-4))
            worst_row_sum_error = max(worst_row_sum_error, float(row_errors.max(initial=0.0)))

        if np.any(state.active_mask):
            radii = state.r[state.active_mask]
            radius_violations += int(np.sum((radii < config.r_min_m) | (radii > config.r_max_m)))
            normal_norms = np.linalg.norm(state.n[state.active_mask], axis=1)
            normal_errors = np.abs(normal_norms - 1.0)
            normal_violations += int(np.sum(normal_errors > 1e-4))
            worst_normal_error = max(worst_normal_error, float(normal_errors.max(initial=0.0)))

        if np.any(state.matched_mask):
            pred = state.pred_idx[state.matched_mask]
            predecessor_violations += int(np.sum(pred < 0))

        invalid_birth = state.birth_mask & (~state.active_mask | state.matched_mask)
        birth_violations += int(np.sum(invalid_birth))
        if previous is not None and not state.debug.fresh_scaffold:
            stale_birth_violations += int(np.sum(state.birth_mask))

        previous = state

    replay.close()
    return {
        "frames": frames,
        "row_sum_violations": row_sum_violations,
        "radius_violations": radius_violations,
        "normal_violations": normal_violations,
        "predecessor_violations": predecessor_violations,
        "birth_violations": birth_violations,
        "stale_birth_violations": stale_birth_violations,
        "worst_row_sum_error": worst_row_sum_error,
        "worst_normal_error": worst_normal_error,
        "all_pass": all(
            value == 0
            for value in (
                row_sum_violations,
                radius_violations,
                normal_violations,
                predecessor_violations,
                birth_violations,
                stale_birth_violations,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mathematical invariant audit for deterministic PICF scaffold.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--enable-rgb-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_invariant_audit(
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

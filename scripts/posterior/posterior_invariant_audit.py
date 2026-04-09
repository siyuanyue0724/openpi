from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline import PointOnlyPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.sonata.config import SonataPointConfig
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor


def run_invariant_audit(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 1024,
    sonata_ckpt_path: str | None = None,
    point_device: str | None = None,
) -> dict:
    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=None if num_segments is None else list(range(num_segments)),
    )
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    point_extractor = SonataPointFeatureExtractor(
        SonataPointConfig(
            checkpoint_path=sonata_ckpt_path,
            device=point_device,
            allow_random_init=True,
        )
    )
    scaffold = DeterministicScaffoldPipeline(builder, point_feature_extractor=point_extractor)
    posterior = PointOnlyPosteriorPipeline(point_feature_extractor=point_extractor)

    scaffold_state = None
    posterior_state = None
    frames = 0
    stale_equal_violations = 0
    var_clip_violations = 0
    precision_gain_violations = 0
    nan_violations = 0
    gate_count_violations = 0

    for observation in replay:
        scaffold_state = scaffold.step(observation, scaffold_state)
        posterior_state = posterior.step(observation, scaffold_state, posterior_state)
        frames += 1

        if posterior_state.debug.nan_count != 0:
            nan_violations += posterior_state.debug.nan_count
        if np.any(posterior_state.var_block < posterior.posterior_config.sigma_min2) or np.any(
            posterior_state.var_block > posterior.posterior_config.sigma_max2
        ):
            var_clip_violations += 1
        if not scaffold_state.debug.fresh_scaffold and not posterior_state.debug.posterior_prior_equal_on_stale:
            stale_equal_violations += 1
        if posterior_state.debug.precision_gain_count != int(np.sum(posterior_state.point.gate)):
            precision_gain_violations += 1
        if np.any(posterior_state.point.gate & (posterior_state.point.anchor_count < posterior.posterior_config.n_min_anchors)):
            gate_count_violations += 1

    replay.close()
    return {
        "frames": frames,
        "stale_equal_violations": stale_equal_violations,
        "var_clip_violations": var_clip_violations,
        "precision_gain_violations": precision_gain_violations,
        "nan_violations": nan_violations,
        "gate_count_violations": gate_count_violations,
        "all_pass": all(
            value == 0
            for value in (
                stale_equal_violations,
                var_clip_violations,
                precision_gain_violations,
                nan_violations,
                gate_count_violations,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Mathematical invariant audit for PICF posterior.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--sonata-ckpt-path", default=None)
    parser.add_argument("--point-device", default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_invariant_audit(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        num_segments=args.segments,
        stride=args.stride,
        max_points=args.max_points,
        sonata_ckpt_path=args.sonata_ckpt_path,
        point_device=args.point_device,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

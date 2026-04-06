from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline_visual import PointVisualPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.vjepa.config import VjepaVisualConfig

try:
    from .posterior_visual_replay_smoke import _make_visual_config
    from .posterior_visual_replay_smoke import _mode_flags
except ImportError:
    from posterior_visual_replay_smoke import _make_visual_config
    from posterior_visual_replay_smoke import _mode_flags


def run_visual_invariant_audit(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    mode: str = "point_visual",
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 2048,
    checkpoint_path: str | None = None,
    model_name: str = "vjepa2_1_vit_base_384",
    arch_name_override: str | None = None,
    img_size: int = 384,
    num_frames: int = 64,
    patch_size: int = 16,
    tubelet_size: int = 2,
    device: str | None = None,
    dtype: str = "bfloat16",
    use_last_two_mean: bool = False,
) -> dict:
    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=None if num_segments is None else list(range(num_segments)),
    )
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    scaffold = DeterministicScaffoldPipeline(builder)
    enable_point_expert, enable_visual_expert = _mode_flags(mode)
    visual_config: VjepaVisualConfig = _make_visual_config(
        calvin_root=calvin_root,
        checkpoint_path=checkpoint_path,
        model_name=model_name,
        arch_name_override=arch_name_override,
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        device=device,
        dtype=dtype,
        use_last_two_mean=use_last_two_mean,
    )
    posterior = PointVisualPosteriorPipeline(
        visual_config=visual_config,
        enable_point_expert=enable_point_expert,
        enable_visual_expert=enable_visual_expert,
    )

    scaffold_state = None
    posterior_state = None
    frames = 0
    stale_no_measurement_equal_violations = 0
    var_clip_violations = 0
    point_precision_gain_violations = 0
    visual_precision_gain_violations = 0
    visual_depth_gate_violations = 0
    nan_violations = 0

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
        if posterior_state.debug.point_precision_gain_count != int(np.sum(posterior_state.point.gate)):
            point_precision_gain_violations += 1
        visual = posterior_state.visual
        if visual is not None:
            if posterior_state.debug.visual_precision_gain_count != int(np.sum(visual.gate)):
                visual_precision_gain_violations += 1
            visual_depth_gate_violations += int(
                np.sum(visual.gate & visual.depth_available & (visual.depth_residual >= visual_config.tau_z_m))
            )
            no_measurement = int(np.sum(posterior_state.point.gate)) == 0 and int(np.sum(visual.gate)) == 0
        else:
            no_measurement = int(np.sum(posterior_state.point.gate)) == 0
        if not scaffold_state.debug.fresh_scaffold and no_measurement and not posterior_state.debug.posterior_prior_equal_on_stale:
            stale_no_measurement_equal_violations += 1

    replay.close()
    return {
        "mode": mode,
        "frames": frames,
        "stale_no_measurement_equal_violations": stale_no_measurement_equal_violations,
        "var_clip_violations": var_clip_violations,
        "point_precision_gain_violations": point_precision_gain_violations,
        "visual_precision_gain_violations": visual_precision_gain_violations,
        "visual_depth_gate_violations": visual_depth_gate_violations,
        "nan_violations": nan_violations,
        "all_pass": all(
            value == 0
            for value in (
                stale_no_measurement_equal_violations,
                var_clip_violations,
                point_precision_gain_violations,
                visual_precision_gain_violations,
                visual_depth_gate_violations,
                nan_violations,
            )
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Invariant audit for PICF posterior visual stage-1.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--mode", default="point_visual", choices=["point_only", "visual_only", "point_visual"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--model-name", default="vjepa2_1_vit_base_384")
    parser.add_argument("--arch-name-override", default=None)
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--num-frames", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--tubelet-size", type=int, default=2)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--use-last-two-mean", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_visual_invariant_audit(
        calvin_root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        mode=args.mode,
        num_segments=args.segments,
        stride=args.stride,
        max_points=args.max_points,
        checkpoint_path=args.checkpoint_path,
        model_name=args.model_name,
        arch_name_override=args.arch_name_override,
        img_size=args.img_size,
        num_frames=args.num_frames,
        patch_size=args.patch_size,
        tubelet_size=args.tubelet_size,
        device=args.device,
        dtype=args.dtype,
        use_last_two_mean=args.use_last_two_mean,
    )
    if args.output_json is not None:
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

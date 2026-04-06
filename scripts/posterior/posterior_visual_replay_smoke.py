from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline_visual import PointVisualPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.vjepa.config import VjepaVisualConfig


def _mode_flags(mode: str) -> tuple[bool, bool]:
    if mode == "point_only":
        return True, False
    if mode == "visual_only":
        return False, True
    if mode == "point_visual":
        return True, True
    raise ValueError(f"Unsupported mode {mode!r}")


def _make_visual_config(
    *,
    calvin_root: str,
    checkpoint_path: str | None,
    model_name: str,
    arch_name_override: str | None,
    img_size: int,
    num_frames: int,
    patch_size: int,
    tubelet_size: int,
    device: str | None,
    dtype: str,
    use_last_two_mean: bool,
) -> VjepaVisualConfig:
    return VjepaVisualConfig(
        model_name=model_name,
        arch_name_override=arch_name_override,
        checkpoint_path=checkpoint_path,
        camera_json_path=calvin_root,
        camera_name="static",
        img_size=img_size,
        num_frames=num_frames,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        device=device,
        dtype=dtype,
        use_last_two_mean=use_last_two_mean,
    )


def run_visual_smoke(
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
    visual_config = _make_visual_config(
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
    states = []
    for observation in replay:
        scaffold_state = scaffold.step(observation, scaffold_state)
        posterior_state = posterior.step(observation, scaffold_state, posterior_state)
        states.append(posterior_state)
    replay.close()
    if not states:
        raise RuntimeError("No posterior states were produced.")

    return {
        "mode": mode,
        "frames": len(states),
        "checkpoint_loaded": bool(
            posterior.visual_encoder.checkpoint_loaded if posterior.visual_encoder is not None else False
        ),
        "mean_point_gate_ratio": sum(state.debug.point_gate_ratio for state in states) / len(states),
        "mean_visual_gate_ratio": sum(state.debug.visual_gate_ratio for state in states) / len(states),
        "mean_precision_gain_count": sum(state.debug.precision_gain_count for state in states) / len(states),
        "mean_point_precision_gain_count": sum(state.debug.point_precision_gain_count for state in states) / len(states),
        "mean_visual_precision_gain_count": sum(state.debug.visual_precision_gain_count for state in states)
        / len(states),
        "max_abs_mu": max(state.debug.max_abs_mu for state in states),
        "min_var_block": min(state.debug.min_var_block for state in states),
        "max_var_block": max(state.debug.max_var_block for state in states),
        "stale_equal_count": sum(1 for state in states if state.debug.posterior_prior_equal_on_stale),
        "nan_count_total": sum(state.debug.nan_count for state in states),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential replay smoke test for PICF visual posterior.")
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

    summary = run_visual_smoke(
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

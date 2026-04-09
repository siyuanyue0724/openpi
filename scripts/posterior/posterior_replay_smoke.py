from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.posterior.pipeline import PointOnlyPosteriorPipeline
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline
from openpi.picf.sonata.config import SonataPointConfig
from openpi.picf.sonata.wrapper import sonata_runtime_available
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor


def _require_posterior_point_runtime(point_device: str | None) -> None:
    device = point_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not str(device).startswith("cuda"):
        raise RuntimeError(
            "PICF posterior smoke requires CUDA for SonataPointFeatureExtractor. "
            "CPU fallback has been removed; rerun on a GPU host with --point-device cuda."
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "PICF posterior smoke requires a visible CUDA device, but torch.cuda.is_available() is False."
        )
    if not sonata_runtime_available():
        raise RuntimeError(
            "PICF posterior smoke requires the Sonata runtime stack (including torch_scatter)."
        )


def run_smoke(
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
    _require_posterior_point_runtime(point_device)
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
        "point_backbone_checkpoint_loaded": bool(point_extractor.checkpoint_loaded),
        "point_backbone_checkpoint_path": str(point_extractor.checkpoint_path) if point_extractor.checkpoint_path else None,
        "point_backbone_cpu_fallback": bool(point_extractor.cpu_fallback),
        "mean_point_gate_ratio": sum(state.debug.point_gate_ratio for state in states) / len(states),
        "mean_precision_gain_count": sum(state.debug.precision_gain_count for state in states) / len(states),
        "max_abs_mu": max(state.debug.max_abs_mu for state in states),
        "min_var_block": min(state.debug.min_var_block for state in states),
        "max_var_block": max(state.debug.max_var_block for state in states),
        "stale_equal_count": sum(
            1 for state in states if (not state.debug.fresh_scaffold) and state.debug.posterior_prior_equal_on_stale
        ),
        "nan_count_total": sum(state.debug.nan_count for state in states),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential replay smoke test for the legacy point-only PICF posterior path.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--sonata-ckpt-path", default=None)
    parser.add_argument("--point-device", default=None, help="CUDA device only, e.g. cuda or cuda:0.")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_smoke(
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

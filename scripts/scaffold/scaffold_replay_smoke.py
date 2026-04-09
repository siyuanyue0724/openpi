from __future__ import annotations

import argparse
import json
from pathlib import Path

from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline


def run_smoke(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 1,
    max_points: int = 1024,
    enable_rgb_identity: bool = True,
) -> dict:
    segment_indices = None if num_segments is None else list(range(num_segments))
    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=segment_indices,
    )
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    pipeline = DeterministicScaffoldPipeline(
        builder,
        config=DeterministicScaffoldConfig(v_rgb_identity=enable_rgb_identity),
    )
    states = []
    previous = None
    for observation in replay:
        previous = pipeline.step(observation, previous)
        states.append(previous)
    replay.close()
    if not states:
        raise RuntimeError("No scaffold states were produced.")

    mean_num_active = sum(s.debug.num_active for s in states) / len(states)
    mean_num_birth = sum(s.debug.num_birth for s in states) / len(states)
    mean_match_ratio = sum(s.debug.match_ratio for s in states) / len(states)
    mean_radius = sum(s.debug.mean_radius for s in states) / len(states)
    mean_normal_fallback_ratio = sum(s.debug.normal_fallback_ratio for s in states) / len(states)
    mean_empty_support_ratio = sum(s.debug.empty_support_ratio for s in states) / len(states)
    hold_count = sum(1 for s in states if s.debug.hold_triggered)
    return {
        "frames": len(states),
        "mean_num_active": mean_num_active,
        "mean_num_birth": mean_num_birth,
        "mean_match_ratio": mean_match_ratio,
        "mean_radius": mean_radius,
        "mean_normal_fallback_ratio": mean_normal_fallback_ratio,
        "mean_empty_support_ratio": mean_empty_support_ratio,
        "hold_count": hold_count,
        "last_state": {
            "step_id": states[-1].step_id,
            "segment_id": states[-1].segment_id,
            "num_active": states[-1].debug.num_active,
            "num_birth": states[-1].debug.num_birth,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential replay smoke test for PICF scaffold.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--enable-rgb-identity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_smoke(
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

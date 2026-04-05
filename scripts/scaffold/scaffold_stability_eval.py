from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from openpi.picf.contracts import PicfObservation
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import CalvinSequentialReplay
from openpi.picf.scaffold.pipeline import DeterministicScaffoldConfig
from openpi.picf.scaffold.pipeline import DeterministicScaffoldPipeline


def _load_frames(calvin_root: str, split: str, backend: str, num_segments: int | None) -> list[PicfObservation]:
    replay = CalvinSequentialReplay(
        calvin_root,
        split=split,
        backend=backend,
        segment_indices=None if num_segments is None else list(range(num_segments)),
    )
    frames = list(replay)
    replay.close()
    return frames


def _photometric_jitter(image: np.ndarray) -> np.ndarray:
    jittered = image.astype(np.int16)
    jittered[..., 0] = np.clip(jittered[..., 0] + 18, 0, 255)
    jittered[..., 1] = np.clip(jittered[..., 1] - 12, 0, 255)
    jittered[..., 2] = np.clip(jittered[..., 2] + 8, 0, 255)
    return jittered.astype(np.uint8)


def _mutate_frames(frames: list[PicfObservation], scenario: str) -> list[PicfObservation]:
    mutated: list[PicfObservation] = []
    for index, frame in enumerate(frames):
        clone = dataclasses.replace(frame, point_set=None, runtime_meta=None)
        if scenario == "photometric_jitter":
            clone.rgb_static = _photometric_jitter(clone.rgb_static)
        elif (scenario == "point_dropout" and index % 3 == 2) or (
            scenario == "invalid_depth_burst" and index in {2, 3, 6}
        ):
            clone.depth_static = np.zeros_like(clone.depth_static, dtype=np.float32)
        mutated.append(clone)
    return mutated


def _run_frames(
    frames: list[PicfObservation],
    builder: CalvinDepthToPicfPointCloud,
    *,
    enable_rgb_identity: bool,
) -> list:
    pipeline = DeterministicScaffoldPipeline(
        builder,
        config=DeterministicScaffoldConfig(v_rgb_identity=enable_rgb_identity),
    )
    states = []
    previous = None
    for frame in frames:
        previous = pipeline.step(frame, previous)
        states.append(previous)
    return states


def _stillness_metrics(frames: list[PicfObservation], states: list) -> tuple[int, int]:
    jump_events = 0
    still_pairs = 0
    for idx in range(1, min(len(frames), len(states))):
        dp = np.linalg.norm(frames[idx].robot_obs[0:3] - frames[idx - 1].robot_obs[0:3])
        dr = np.linalg.norm(frames[idx].robot_obs[3:6] - frames[idx - 1].robot_obs[3:6])
        if dp < 0.002 and dr < 0.02:
            still_pairs += 1
            prev_active = max(states[idx - 1].debug.num_active, 1)
            if abs(states[idx].debug.num_active - states[idx - 1].debug.num_active) > 0.25 * prev_active:
                jump_events += 1
    return still_pairs, jump_events


def _summarize(frames: list[PicfObservation], states: list) -> dict:
    still_pairs, jump_events = _stillness_metrics(frames, states)
    fresh_states = [s for s in states if s.debug.fresh_scaffold]
    non_reset = [s for frame, s in zip(frames, states, strict=False) if not frame.reset_scaffold]
    return {
        "frames": len(states),
        "reindex_failure_rate": float(
            sum(s.debug.reindex_failure_rate for s in fresh_states) / max(len(fresh_states), 1)
        ),
        "birth_explosion_rate": float(
            sum(1 for s in non_reset if s.debug.num_birth > max(1, int(0.75 * 12))) / max(len(non_reset), 1)
        ),
        "normal_flip_ratio": float(sum(s.debug.normal_flip_ratio for s in fresh_states) / max(len(fresh_states), 1)),
        "stale_timeout_count": int(sum(1 for s in states if s.debug.hold_reason == "scaffold_stale_timeout")),
        "still_pairs": int(still_pairs),
        "active_jump_rate": float(jump_events / max(still_pairs, 1)),
    }


def run_stability_eval(
    *,
    calvin_root: str,
    split: str,
    backend: str,
    num_segments: int | None = None,
    stride: int = 2,
    max_points: int = 2048,
    enable_rgb_identity: bool = False,
) -> dict:
    frames = _load_frames(calvin_root, split, backend, num_segments)
    builder = CalvinDepthToPicfPointCloud(calvin_root, stride=stride, max_points=max_points)
    baseline_states = _run_frames(
        _mutate_frames(frames, "baseline"),
        builder,
        enable_rgb_identity=enable_rgb_identity,
    )
    jitter_states = _run_frames(
        _mutate_frames(frames, "photometric_jitter"),
        builder,
        enable_rgb_identity=enable_rgb_identity,
    )
    dropout_states = _run_frames(
        _mutate_frames(frames, "point_dropout"),
        builder,
        enable_rgb_identity=enable_rgb_identity,
    )
    burst_states = _run_frames(
        _mutate_frames(frames, "invalid_depth_burst"),
        builder,
        enable_rgb_identity=enable_rgb_identity,
    )
    baseline = _summarize(frames, baseline_states)
    jitter = _summarize(frames, jitter_states)
    dropout = _summarize(frames, dropout_states)
    burst = _summarize(frames, burst_states)
    return {
        "baseline": baseline,
        "photometric_jitter": jitter,
        "point_dropout": dropout,
        "invalid_depth_burst": burst,
        "photometric_vs_baseline_reindex_ratio": jitter["reindex_failure_rate"] / max(
            baseline["reindex_failure_rate"], 1e-6
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Stability evaluation for deterministic PICF scaffold.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="zip", choices=["zip", "dir"])
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-points", type=int, default=2048)
    parser.add_argument("--enable-rgb-identity", action="store_true")
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    summary = run_stability_eval(
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

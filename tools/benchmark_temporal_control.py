#!/usr/bin/env python3
"""Measure stream-plan and detached-posterior control overhead.

This intentionally excludes the encoder, action host and PICF neural core. It
is a lower-level regression guard, not evidence for end-to-end throughput.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import fields
from typing import Protocol

import torch

from picf_next.geometry import PhysicalGeometryContract
from picf_next.models.temporal import ObjectBeliefBatch, TemporalFilterConfig
from picf_next.training.control import (
    EpisodeSampleSequence,
    FrozenEpisodeStreamPlan,
    FrozenSamplePlan,
)
from picf_next.training.stream_state import PosteriorStreamState

BENCHMARK_GEOMETRY = PhysicalGeometryContract(
    name="picf.synthetic-workspace-position.v1",
    quantity="object_position",
    reference_frame="synthetic_robot_base",
    axes=("x", "y", "z"),
    units=("m", "m", "m"),
    normalization_offset=(0.0, 0.0, 0.0),
    normalization_scale=(1.0, 1.0, 1.0),
)


class _BatchPlan(Protocol):
    def global_batch(self, optimizer_step: int) -> object: ...


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--global-lanes", type=int, default=64)
    parser.add_argument("--total-steps", type=int, default=30000)
    parser.add_argument("--plan-iterations", type=int, default=1000)
    parser.add_argument("--state-iterations", type=int, default=1000)
    return parser.parse_args()


def _median_p95_us(values: list[float]) -> tuple[float, float]:
    return statistics.median(values), statistics.quantiles(values, n=100)[94]


def _time_plan(plan: _BatchPlan, steps: list[int]) -> tuple[float, float]:
    plan.global_batch(steps[0])
    values = []
    for step in steps:
        start = time.perf_counter_ns()
        plan.global_batch(step)
        values.append((time.perf_counter_ns() - start) / 1000.0)
    return _median_p95_us(values)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    args = _parse_args()
    positive = {
        "episodes": args.episodes,
        "global_lanes": args.global_lanes,
        "total_steps": args.total_steps,
        "plan_iterations": args.plan_iterations,
        "state_iterations": args.state_iterations,
    }
    if any(value <= 0 for value in positive.values()):
        raise ValueError("all benchmark sizes must be positive")
    if args.episodes < args.global_lanes:
        raise ValueError("episodes must be at least global-lanes")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")

    episodes = tuple(
        EpisodeSampleSequence(
            f"episode-{episode:05d}",
            tuple(f"episode-{episode:05d}/frame-{frame:04d}" for frame in range(80 + episode % 41)),
        )
        for episode in range(args.episodes)
    )
    sample_keys = tuple(key for episode in episodes for key in episode.sample_keys)
    common = {
        "dataset_id": "temporal-control-benchmark",
        "dataset_revision": "generated-v1",
        "dataset_manifest_sha256": "a" * 64,
        "comparison_id": "temporal-control-benchmark-seed-71",
        "seed": 71,
        "global_batch_size": args.global_lanes,
        "total_steps": args.total_steps,
    }
    start = time.perf_counter_ns()
    stream_plan = FrozenEpisodeStreamPlan(episodes=episodes, **common)
    stream_build_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    random_plan = FrozenSamplePlan(sample_keys=sample_keys, **common)
    steps = [step % args.total_steps for step in range(args.plan_iterations)]
    stream_median, stream_p95 = _time_plan(stream_plan, steps)
    random_median, random_p95 = _time_plan(random_plan, steps)

    config = TemporalFilterConfig(
        address_dim=64,
        content_dim=128,
        geometry_dim=3,
        geometry_contract=BENCHMARK_GEOMETRY,
        action_dim=32,
        reference_delta_t_s=0.05,
        hidden_dim=256,
        num_layers=2,
        num_heads=8,
    )
    local_batch = max(1, args.global_lanes // 8)
    stream_state = PosteriorStreamState(
        config,
        lane_ids=tuple(f"local-lane-{index:05d}" for index in range(local_batch)),
        capacity=16,
        device=device,
        dtype=torch.float32,
        max_parameter_lag=1,
    )
    base = stream_state.belief
    final_belief = ObjectBeliefBatch(
        address_mean=torch.ones_like(base.address_mean),
        content_mean=torch.ones_like(base.content_mean),
        geometry_mean=torch.ones_like(base.geometry_mean),
        geometry_covariance_diag=torch.ones_like(base.geometry_covariance_diag),
        existence_logits=torch.ones_like(base.existence_logits),
        visibility_given_existence_logits=torch.ones_like(base.visibility_given_existence_logits),
        measurement_age_s=torch.ones_like(base.measurement_age_s),
        valid=torch.ones_like(base.valid),
        age=torch.ones_like(base.age),
    )
    episode_keys = tuple(f"episode-{index:05d}" for index in range(local_batch))
    state_times = []
    with torch.inference_mode():
        for step in range(args.state_iterations):
            _synchronize(device)
            start = time.perf_counter_ns()
            stream_state.prepare_chunk(
                episode_keys=episode_keys,
                start_transition_indices=(step,) * local_batch,
                current_parameter_version=step,
            )
            stream_state.commit_chunk(
                final_belief,
                transition_count=1,
                state_parameter_version=step,
            )
            _synchronize(device)
            state_times.append((time.perf_counter_ns() - start) / 1000.0)
    state_median, state_p95 = _median_p95_us(state_times)
    posterior_elements = sum(
        getattr(final_belief, field.name).numel() for field in fields(final_belief)
    )

    report = {
        "device": str(device),
        "episode_count": len(episodes),
        "global_lanes": args.global_lanes,
        "plan_iterations": args.plan_iterations,
        "posterior_elements": posterior_elements,
        "random_plan_global_batch_us_median": random_median,
        "random_plan_global_batch_us_p95": random_p95,
        "sample_count": len(sample_keys),
        "schema": "picf-next.temporal-control-benchmark.v1",
        "state_prepare_commit_us_median": state_median,
        "state_prepare_commit_us_p95": state_p95,
        "stream_plan_build_ms": stream_build_ms,
        "stream_plan_global_batch_us_median": stream_median,
        "stream_plan_global_batch_us_p95": stream_p95,
        "stream_plan_minus_random_us_median": stream_median - random_median,
        "total_steps": args.total_steps,
        "warning": "control-plane benchmark only; not end-to-end training throughput",
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

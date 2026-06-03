#!/usr/bin/env python3
"""Audit CALVIN task-bucket sampler distributions without building a model."""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openpi.training.calvin_dataset import CalvinLangSegmentDataset  # noqa: E402


def _metric_key_fragment(value: str) -> str:
    fragment = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    fragment = fragment.strip("_")
    return fragment or "unknown"


def _calvin_prompt_bucket(prompt: str) -> str:
    text = str(prompt).strip().lower().replace("_", " ")
    if not text:
        return "other"
    if "drawer" in text:
        return "drawer"
    if "button" in text or "switch" in text or "light" in text or "led" in text:
        return "switch_button_light"
    if "slider" in text or "slide" in text:
        return "slider"
    if "push" in text and "block" in text:
        return "block_push"
    if ("lift" in text or "grasp" in text or "pick" in text) and "block" in text:
        return "block_lift"
    if "block" in text:
        return "block_other"
    return "other"


def _step_indexed_window_rng(*, seed: int, rank: int, step: int, micro_step: int, retry_count: int = 0) -> np.random.Generator:
    words = (
        int(seed) & 0xFFFFFFFF,
        int(rank) & 0xFFFFFFFF,
        int(step) & 0xFFFFFFFF,
        int(micro_step) & 0xFFFFFFFF,
        int(retry_count) & 0xFFFFFFFF,
        0xA7B5_2026,
    )
    return np.random.default_rng(np.random.SeedSequence(words))


def _parse_bucket_weight_spec(spec: str | None) -> dict[str, float]:
    text = str(spec or "").strip()
    if not text:
        return {}
    result: dict[str, float] = {}
    for raw_part in re.split(r"[,;]", text):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            key, raw_value = part.split("=", 1)
        elif ":" in part:
            key, raw_value = part.split(":", 1)
        else:
            raise ValueError(f"Invalid bucket weight entry {part!r}.")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Invalid bucket weight {value} for {key!r}.")
        result[key.strip()] = value
    return result


def _compute_bucket_sampling_weights(
    *,
    bucket_names: list[str],
    bucket_sizes: dict[str, int],
    mode: str,
    temperature_alpha: float,
    weight_spec: str,
) -> dict[str, float]:
    names = [str(name) for name in bucket_names]
    overrides = _parse_bucket_weight_spec(weight_spec)
    if overrides:
        default_weight = overrides.get("*", overrides.get("__default__"))
        weights: dict[str, float] = {}
        for name in names:
            matched: list[float] = []
            for key, value in overrides.items():
                if key in {"*", "__default__"}:
                    continue
                if key == name or key == _metric_key_fragment(name) or fnmatch.fnmatch(name, key):
                    matched.append(float(value))
            if matched:
                weights[name] = float(matched[-1])
            elif default_weight is not None:
                weights[name] = float(default_weight)
            else:
                raise ValueError(f"Bucket weight spec does not cover bucket {name!r}.")
    elif mode in {"round_robin", "task_uniform"}:
        weights = {name: 1.0 for name in names}
    elif mode == "trajectory":
        weights = {name: float(max(int(bucket_sizes.get(name, 0)), 0)) for name in names}
    elif mode == "temperature":
        alpha = max(float(temperature_alpha), 0.0)
        weights = {name: float(max(int(bucket_sizes.get(name, 0)), 0)) ** alpha for name in names}
    else:
        raise ValueError(f"Unsupported bucket sampling mode {mode!r}.")
    total = float(sum(max(float(value), 0.0) for value in weights.values()))
    if total <= 0.0:
        raise ValueError("Bucket weights sum to zero.")
    return {name: max(float(weights[name]), 0.0) / total for name in names}


def _bucket_sequence_for_logical_step(
    *,
    bucket_names: list[str] | tuple[str, ...],
    target_bucket_weights: dict[str, float],
    mode: str,
    weight_spec: str,
    seed: int,
    step: int,
    world_size: int,
    accum_steps: int,
    without_replacement: bool = True,
) -> tuple[str, ...]:
    names = tuple(str(name) for name in bucket_names)
    if not names:
        return ()
    global_micro_count = max(int(world_size), 1) * max(int(accum_steps), 1)
    if global_micro_count <= 0:
        return ()
    if str(mode) == "round_robin" and not str(weight_spec or "").strip():
        base = int(step) * global_micro_count
        return tuple(names[int(base + offset) % len(names)] for offset in range(global_micro_count))

    raw_weights = np.asarray(
        [max(float(target_bucket_weights.get(str(name), 0.0)), 0.0) for name in names],
        dtype=np.float64,
    )
    positive = raw_weights > 0.0
    if not bool(np.any(positive)):
        raise ValueError(f"Bucket target weights contain no positive mass: {target_bucket_weights!r}.")
    rng = np.random.default_rng(
        np.random.SeedSequence(
            (
                int(seed) & 0xFFFFFFFF,
                int(step) & 0xFFFFFFFF,
                int(global_micro_count) & 0xFFFFFFFF,
                0xB00C_2026,
            )
        )
    )
    eligible_names = np.asarray([names[index] for index, keep in enumerate(positive) if bool(keep)], dtype=object)
    eligible_weights = raw_weights[positive]
    eligible_weights = eligible_weights / float(np.sum(eligible_weights))
    if not bool(without_replacement):
        return tuple(
            str(bucket)
            for bucket in rng.choice(
                eligible_names,
                size=int(global_micro_count),
                replace=True,
                p=eligible_weights,
            )
        )

    sequence: list[str] = []
    while len(sequence) < int(global_micro_count):
        take = min(int(global_micro_count) - len(sequence), int(len(eligible_names)))
        chosen = rng.choice(eligible_names, size=int(take), replace=False, p=eligible_weights)
        sequence.extend(str(bucket) for bucket in np.asarray(chosen, dtype=object).reshape(-1))
    return tuple(sequence)


class _AuditSource:
    def __init__(
        self,
        *,
        root: str,
        split: str,
        backend: str,
        unroll_steps: int,
        action_horizon: int,
        segment_indices: list[int] | None,
        bucket_sampling_mode: str,
        bucket_temperature_alpha: float,
        bucket_weight_spec: str,
        bucket_sample_without_replacement: bool,
    ) -> None:
        self.dataset = CalvinLangSegmentDataset(
            root=root,
            split=split,
            action_horizon=int(action_horizon),
            backend=backend,
            use_wrist_rgb=True,
            sample_within_segment=False,
        )
        self.reader = self.dataset.reader
        self.segments = self.dataset.segments
        self.unroll_steps = int(unroll_steps)
        self.action_horizon = int(action_horizon)
        selected_segment_ids = (
            list(range(len(self.segments))) if segment_indices is None else [int(value) for value in segment_indices]
        )
        self.segment_sampling_slots: list[dict[str, int]] = []
        for segment_id in selected_segment_ids:
            segment = self.segments[int(segment_id)]
            max_start_exclusive = segment.end - (self.unroll_steps + self.action_horizon - 1)
            if segment.start < max_start_exclusive:
                self.segment_sampling_slots.append(
                    {
                        "segment_id": int(segment_id),
                        "first_valid_start_step_id": int(segment.start),
                        "valid_start_exclusive": int(max_start_exclusive),
                    }
                )
        if not self.segment_sampling_slots:
            raise RuntimeError("No valid segment sampling slots found.")
        self.bucket_to_slot_indices: dict[str, list[int]] = {}
        for slot_index, slot in enumerate(self.segment_sampling_slots):
            bucket = _calvin_prompt_bucket(self.segments[int(slot["segment_id"])].lang)
            self.bucket_to_slot_indices.setdefault(bucket, []).append(int(slot_index))
        self.bucket_names = tuple(sorted(bucket for bucket, indices in self.bucket_to_slot_indices.items() if indices))
        self.bucket_segment_counts = {
            str(bucket): int(len(indices)) for bucket, indices in sorted(self.bucket_to_slot_indices.items())
        }
        self.bucket_sampling_mode = str(bucket_sampling_mode)
        self.bucket_temperature_alpha = float(bucket_temperature_alpha)
        self.bucket_weight_spec = str(bucket_weight_spec or "").strip()
        self.bucket_sample_without_replacement = bool(bucket_sample_without_replacement)
        self.bucket_target_weights = _compute_bucket_sampling_weights(
            bucket_names=list(self.bucket_names),
            bucket_sizes=self.bucket_segment_counts,
            mode=self.bucket_sampling_mode,
            temperature_alpha=self.bucket_temperature_alpha,
            weight_spec=self.bucket_weight_spec,
        )

    def balanced_bucket_slot_index(
        self,
        *,
        seed: int,
        rank: int,
        world_size: int,
        step: int,
        micro_step: int,
        accum_steps: int,
        retry_count: int = 0,
    ) -> tuple[int, str, np.random.Generator]:
        sample_rng = _step_indexed_window_rng(
            seed=seed,
            rank=rank,
            step=step,
            micro_step=micro_step,
            retry_count=retry_count,
        )
        bucket_sequence = _bucket_sequence_for_logical_step(
            bucket_names=self.bucket_names,
            target_bucket_weights=self.bucket_target_weights,
            mode=self.bucket_sampling_mode,
            weight_spec=self.bucket_weight_spec,
            seed=int(seed),
            step=int(step),
            world_size=int(world_size),
            accum_steps=int(accum_steps),
            without_replacement=bool(self.bucket_sample_without_replacement),
        )
        global_micro_in_step = int(rank) * max(int(accum_steps), 1) + int(micro_step)
        bucket = str(bucket_sequence[int(global_micro_in_step) % len(bucket_sequence)])
        candidates = self.bucket_to_slot_indices[bucket]
        return int(candidates[int(sample_rng.integers(0, len(candidates)))]), bucket, sample_rng

    def close(self) -> None:
        self.reader.close()


def _parse_segment_indices(value: str) -> list[int] | None:
    text = str(value or "").strip()
    if not text:
        return None
    return [int(part) for part in text.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training")
    parser.add_argument("--backend", default="dir")
    parser.add_argument("--calvin-segment-indices", default="")
    parser.add_argument("--unroll-steps", type=int, default=2)
    parser.add_argument("--action-horizon", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument(
        "--calvin-bucket-sampling-mode",
        choices=("round_robin", "task_uniform", "trajectory", "temperature"),
        default="round_robin",
    )
    parser.add_argument("--calvin-bucket-temperature-alpha", type=float, default=0.0)
    parser.add_argument("--calvin-bucket-weight-spec", default="")
    parser.add_argument("--calvin-bucket-sample-without-replacement", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    source = _AuditSource(
        root=args.calvin_root,
        split=args.split,
        backend=args.backend,
        unroll_steps=args.unroll_steps,
        action_horizon=args.action_horizon,
        segment_indices=_parse_segment_indices(args.calvin_segment_indices),
        bucket_sampling_mode=args.calvin_bucket_sampling_mode,
        bucket_temperature_alpha=args.calvin_bucket_temperature_alpha,
        bucket_weight_spec=args.calvin_bucket_weight_spec,
        bucket_sample_without_replacement=bool(args.calvin_bucket_sample_without_replacement),
    )
    try:
        counts: dict[str, int] = {}
        step_distinct_counts: list[int] = []
        examples: list[dict[str, int | str]] = []
        for step in range(int(args.steps)):
            step_buckets: list[str] = []
            for rank in range(int(args.world_size)):
                for micro_step in range(int(args.accum_steps)):
                    slot_index, bucket, _ = source.balanced_bucket_slot_index(
                        seed=int(args.seed),
                        rank=int(rank),
                        world_size=int(args.world_size),
                        step=int(step),
                        micro_step=int(micro_step),
                        accum_steps=int(args.accum_steps),
                    )
                    counts[str(bucket)] = int(counts.get(str(bucket), 0)) + 1
                    step_buckets.append(str(bucket))
                    if len(examples) < 16:
                        segment_id = int(source.segment_sampling_slots[int(slot_index)]["segment_id"])
                        examples.append(
                            {
                                "step": int(step),
                                "rank": int(rank),
                                "micro_step": int(micro_step),
                                "slot_index": int(slot_index),
                                "segment_id": int(segment_id),
                                "bucket": str(bucket),
                            }
                        )
            step_distinct_counts.append(int(len(set(step_buckets))))
        total = max(sum(counts.values()), 1)
        sample_frequencies = {
            bucket: float(counts.get(bucket, 0)) / float(total) for bucket in source.bucket_names
        }
        max_abs_deviation = max(
            (
                abs(float(sample_frequencies[bucket]) - float(source.bucket_target_weights[bucket]))
                for bucket in source.bucket_names
            ),
            default=0.0,
        )
        kl_empirical_to_target = 0.0
        for bucket in source.bucket_names:
            empirical = float(sample_frequencies[bucket])
            target = max(float(source.bucket_target_weights[bucket]), 1e-12)
            if empirical > 0.0:
                kl_empirical_to_target += empirical * math.log(empirical / target)
        payload = {
            "steps": int(args.steps),
            "world_size": int(args.world_size),
            "accum_steps": int(args.accum_steps),
            "global_micro_count": int(args.steps) * int(args.world_size) * int(args.accum_steps),
            "bucket_sampling_mode": str(source.bucket_sampling_mode),
            "bucket_temperature_alpha": float(source.bucket_temperature_alpha),
            "bucket_weight_spec": str(source.bucket_weight_spec),
            "bucket_sample_without_replacement": bool(source.bucket_sample_without_replacement),
            "bucket_names": list(source.bucket_names),
            "bucket_segment_counts": dict(source.bucket_segment_counts),
            "bucket_target_weights": dict(source.bucket_target_weights),
            "sample_counts": {bucket: int(counts.get(bucket, 0)) for bucket in source.bucket_names},
            "sample_frequencies": sample_frequencies,
            "max_abs_deviation": float(max_abs_deviation),
            "kl_empirical_to_target": float(kl_empirical_to_target),
            "min_distinct_buckets_per_step": int(min(step_distinct_counts, default=0)),
            "max_distinct_buckets_per_step": int(max(step_distinct_counts, default=0)),
            "mean_distinct_buckets_per_step": float(
                np.mean(np.asarray(step_distinct_counts, dtype=np.float64)) if step_distinct_counts else 0.0
            ),
            "examples": examples,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    finally:
        source.close()


if __name__ == "__main__":
    main()

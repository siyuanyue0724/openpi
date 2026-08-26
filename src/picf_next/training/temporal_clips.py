"""Deterministic fixed-parameter clip plans for posterior-core training."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Final

STATIONARY_TEMPORAL_CLIP_ALGORITHM: Final = "sha256-stationary-temporal-clips.v1"
STATIONARY_STATE_CONTRACT: Final = "reset-replay-one-parameter-version.v1"


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _frozen_ranges(ranges: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    if (
        not isinstance(ranges, tuple)
        or not ranges
        or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or any(not isinstance(value, int) or isinstance(value, bool) for value in item)
            or item[0] < 0
            or item[1] <= item[0]
            for item in ranges
        )
    ):
        raise ValueError("source ranges must be nonempty integer half-open intervals")
    if ranges != tuple(sorted(ranges)) or any(
        left[1] > right[0] for left, right in zip(ranges, ranges[1:], strict=False)
    ):
        raise ValueError("source ranges must be sorted and non-overlapping")
    return ranges


@dataclass(frozen=True, slots=True)
class StationaryTemporalClip:
    """One reset/replay clip whose state cannot cross an optimizer update."""

    optimizer_step: int
    source_range_index: int
    start_global_index: int
    prefix_length: int
    train_length: int

    def __post_init__(self) -> None:
        if (
            not isinstance(self.optimizer_step, int)
            or isinstance(self.optimizer_step, bool)
            or self.optimizer_step < 0
        ):
            raise ValueError("optimizer_step must be a non-negative integer")
        if (
            not isinstance(self.source_range_index, int)
            or isinstance(self.source_range_index, bool)
            or self.source_range_index < 0
        ):
            raise ValueError("source_range_index must be a non-negative integer")
        if (
            not isinstance(self.start_global_index, int)
            or isinstance(self.start_global_index, bool)
            or self.start_global_index < 0
        ):
            raise ValueError("start_global_index must be a non-negative integer")
        if (
            not isinstance(self.prefix_length, int)
            or isinstance(self.prefix_length, bool)
            or self.prefix_length < 0
        ):
            raise ValueError("prefix_length must be a non-negative integer")
        _positive_integer(self.train_length, "train_length")

    @property
    def train_start_global_index(self) -> int:
        return self.start_global_index + self.prefix_length

    @property
    def stop_global_index(self) -> int:
        return self.train_start_global_index + self.train_length

    @property
    def total_length(self) -> int:
        return self.prefix_length + self.train_length

    @property
    def prefix_indices(self) -> tuple[int, ...]:
        return tuple(range(self.start_global_index, self.train_start_global_index))

    @property
    def train_indices(self) -> tuple[int, ...]:
        return tuple(range(self.train_start_global_index, self.stop_global_index))

    def to_dict(self) -> dict[str, int]:
        return {
            "optimizer_step": self.optimizer_step,
            "source_range_index": self.source_range_index,
            "start_global_index": self.start_global_index,
            "prefix_length": self.prefix_length,
            "train_length": self.train_length,
            "train_start_global_index": self.train_start_global_index,
            "stop_global_index": self.stop_global_index,
        }


@dataclass(frozen=True, slots=True)
class StationaryTemporalClipPlan:
    """Hash-addressed optimizer plan with no recurrent state in its API."""

    source_ranges: tuple[tuple[int, int], ...]
    prefix_lengths: tuple[int, ...]
    train_length: int
    seed: int
    clips: tuple[StationaryTemporalClip, ...]
    algorithm: str = STATIONARY_TEMPORAL_CLIP_ALGORITHM
    state_contract: str = STATIONARY_STATE_CONTRACT

    def __post_init__(self) -> None:
        ranges = _frozen_ranges(self.source_ranges)
        if self.algorithm != STATIONARY_TEMPORAL_CLIP_ALGORITHM:
            raise ValueError("unsupported stationary temporal clip algorithm")
        if self.state_contract != STATIONARY_STATE_CONTRACT:
            raise ValueError("unsupported stationary temporal state contract")
        if (
            not isinstance(self.prefix_lengths, tuple)
            or not self.prefix_lengths
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in self.prefix_lengths
            )
            or tuple(sorted(set(self.prefix_lengths))) != self.prefix_lengths
        ):
            raise ValueError("prefix_lengths must be unique increasing non-negative integers")
        _positive_integer(self.train_length, "train_length")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not self.clips:
            raise ValueError("stationary temporal clip plan cannot be empty")
        if tuple(clip.optimizer_step for clip in self.clips) != tuple(range(len(self.clips))):
            raise ValueError("stationary temporal clips must cover optimizer steps exactly")
        for clip in self.clips:
            if clip.source_range_index >= len(ranges):
                raise ValueError("stationary temporal clip references an unknown source range")
            start, stop = ranges[clip.source_range_index]
            if clip.prefix_length not in self.prefix_lengths:
                raise ValueError("stationary temporal clip uses an undeclared prefix length")
            if clip.train_length != self.train_length:
                raise ValueError("stationary temporal clip train length changed")
            if clip.start_global_index < start or clip.stop_global_index > stop:
                raise ValueError("stationary temporal clip crosses its source range")

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "picf-next.stationary-temporal-clip-plan.v1",
            "algorithm": self.algorithm,
            "state_contract": self.state_contract,
            "source_ranges": [list(value) for value in self.source_ranges],
            "prefix_lengths": list(self.prefix_lengths),
            "train_length": self.train_length,
            "seed": self.seed,
            "optimizer_steps": len(self.clips),
            "clips": [clip.to_dict() for clip in self.clips],
        }


def _digest_integer(*coordinates: object) -> int:
    encoded = "\0".join(str(value) for value in coordinates).encode("ascii")
    return int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")


def build_stationary_temporal_clip_plan(
    *,
    source_ranges: tuple[tuple[int, int], ...],
    prefix_lengths: tuple[int, ...],
    train_length: int,
    optimizer_steps: int,
    seed: int,
) -> StationaryTemporalClipPlan:
    """Select deterministic clips while keeping every clip inside one range."""

    ranges = _frozen_ranges(source_ranges)
    _positive_integer(train_length, "train_length")
    _positive_integer(optimizer_steps, "optimizer_steps")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(prefix_lengths, tuple)
        or not prefix_lengths
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in prefix_lengths
        )
        or tuple(sorted(set(prefix_lengths))) != prefix_lengths
    ):
        raise ValueError("prefix_lengths must be unique increasing non-negative integers")

    eligible_by_prefix: dict[int, tuple[tuple[int, int, int], ...]] = {}
    for prefix_length in prefix_lengths:
        total_length = prefix_length + train_length
        eligible = tuple(
            (range_index, start, stop)
            for range_index, (start, stop) in enumerate(ranges)
            if stop - start >= total_length
        )
        if not eligible:
            raise ValueError(
                f"no source range can hold prefix={prefix_length}, train={train_length}"
            )
        eligible_by_prefix[prefix_length] = eligible

    prefix_order = tuple(
        sorted(
            prefix_lengths,
            key=lambda value: (_digest_integer(seed, "prefix", value), value),
        )
    )
    clips = []
    for optimizer_step in range(optimizer_steps):
        cycle, cycle_offset = divmod(optimizer_step, len(prefix_order))
        prefix_length = prefix_order[cycle_offset]
        eligible = eligible_by_prefix[prefix_length]
        range_index, range_start, range_stop = eligible[
            _digest_integer(seed, "range", cycle, prefix_length) % len(eligible)
        ]
        total_length = prefix_length + train_length
        start_count = range_stop - range_start - total_length + 1
        start_offset = (
            _digest_integer(
                seed,
                "start",
                cycle,
                prefix_length,
                range_index,
            )
            % start_count
        )
        clips.append(
            StationaryTemporalClip(
                optimizer_step=optimizer_step,
                source_range_index=range_index,
                start_global_index=range_start + start_offset,
                prefix_length=prefix_length,
                train_length=train_length,
            )
        )
    return StationaryTemporalClipPlan(
        source_ranges=ranges,
        prefix_lengths=prefix_lengths,
        train_length=train_length,
        seed=seed,
        clips=tuple(clips),
    )


@dataclass(frozen=True, slots=True)
class DistributedStationaryTemporalClipPlan:
    """One collective-safe reset/replay clip assignment per rank and optimizer step."""

    source_ranges: tuple[tuple[int, int], ...]
    prefix_lengths: tuple[int, ...]
    train_length: int
    required_future_horizon: int
    optimizer_steps: int
    world_size: int
    seed: int
    clips_by_step: tuple[tuple[StationaryTemporalClip, ...], ...]
    algorithm: str = STATIONARY_TEMPORAL_CLIP_ALGORITHM
    state_contract: str = STATIONARY_STATE_CONTRACT

    def __post_init__(self) -> None:
        ranges = _frozen_ranges(self.source_ranges)
        _positive_integer(self.train_length, "train_length")
        _positive_integer(self.required_future_horizon, "required_future_horizon")
        _positive_integer(self.optimizer_steps, "optimizer_steps")
        _positive_integer(self.world_size, "world_size")
        if self.algorithm != STATIONARY_TEMPORAL_CLIP_ALGORITHM:
            raise ValueError("unsupported distributed stationary temporal clip algorithm")
        if self.state_contract != STATIONARY_STATE_CONTRACT:
            raise ValueError("unsupported distributed stationary state contract")
        if (
            not isinstance(self.prefix_lengths, tuple)
            or not self.prefix_lengths
            or tuple(sorted(set(self.prefix_lengths))) != self.prefix_lengths
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in self.prefix_lengths
            )
        ):
            raise ValueError("prefix_lengths must be unique increasing non-negative integers")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if len(self.clips_by_step) != self.optimizer_steps:
            raise ValueError("distributed clips must cover every optimizer step")
        for step, rank_clips in enumerate(self.clips_by_step):
            if len(rank_clips) != self.world_size:
                raise ValueError("distributed clip count differs from world size")
            if any(clip.optimizer_step != step for clip in rank_clips):
                raise ValueError("distributed clip optimizer-step identity changed")
            if len({clip.prefix_length for clip in rank_clips}) != 1:
                raise ValueError("all ranks must replay the same prefix length per step")
            if (
                len({(clip.source_range_index, clip.start_global_index) for clip in rank_clips})
                != self.world_size
            ):
                raise ValueError("distributed ranks cannot consume the same clip in one step")
            for clip in rank_clips:
                if clip.prefix_length not in self.prefix_lengths:
                    raise ValueError("distributed clip uses an undeclared prefix length")
                if clip.train_length != self.train_length:
                    raise ValueError("distributed clip train length changed")
                start, stop = ranges[clip.source_range_index]
                if (
                    clip.start_global_index < start
                    or clip.stop_global_index + self.required_future_horizon > stop
                ):
                    raise ValueError("distributed clip crosses its source range")

    @property
    def plan_sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    def clip(self, optimizer_step: int, rank: int) -> StationaryTemporalClip:
        if not 0 <= optimizer_step < self.optimizer_steps:
            raise IndexError("stationary optimizer step is out of range")
        if not 0 <= rank < self.world_size:
            raise IndexError("stationary process rank is out of range")
        return self.clips_by_step[optimizer_step][rank]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "picf-next.distributed-stationary-temporal-clip-plan.v1",
            "algorithm": self.algorithm,
            "state_contract": self.state_contract,
            "source_ranges": [list(value) for value in self.source_ranges],
            "prefix_lengths": list(self.prefix_lengths),
            "train_length": self.train_length,
            "required_future_horizon": self.required_future_horizon,
            "optimizer_steps": self.optimizer_steps,
            "world_size": self.world_size,
            "seed": self.seed,
            "clips_by_step": [
                [clip.to_dict() for clip in rank_clips] for rank_clips in self.clips_by_step
            ],
        }


def _coprime_stride(seed: int, modulus: int, *coordinates: object) -> int:
    if modulus <= 1:
        return 1
    candidate = 1 + _digest_integer(seed, "stride", *coordinates) % (modulus - 1)
    while math.gcd(candidate, modulus) != 1:
        candidate = 1 + candidate % (modulus - 1)
    return candidate


def build_distributed_stationary_temporal_clip_plan(
    *,
    source_ranges: tuple[tuple[int, int], ...],
    prefix_lengths: tuple[int, ...],
    train_length: int,
    required_future_horizon: int,
    optimizer_steps: int,
    world_size: int,
    seed: int,
) -> DistributedStationaryTemporalClipPlan:
    """Build same-length, distinct-rank clips for deterministic DDP execution."""

    ranges = _frozen_ranges(source_ranges)
    _positive_integer(train_length, "train_length")
    _positive_integer(required_future_horizon, "required_future_horizon")
    _positive_integer(optimizer_steps, "optimizer_steps")
    _positive_integer(world_size, "world_size")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        not isinstance(prefix_lengths, tuple)
        or not prefix_lengths
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in prefix_lengths
        )
        or tuple(sorted(set(prefix_lengths))) != prefix_lengths
    ):
        raise ValueError("prefix_lengths must be unique increasing non-negative integers")

    candidates_by_prefix: dict[int, tuple[tuple[int, int], ...]] = {}
    for prefix_length in prefix_lengths:
        total_length = prefix_length + train_length
        candidates = tuple(
            (range_index, start_global_index)
            for range_index, (range_start, range_stop) in enumerate(ranges)
            for start_global_index in range(
                range_start,
                range_stop - total_length - required_future_horizon + 1,
            )
        )
        if len(candidates) < world_size:
            raise ValueError(
                f"prefix={prefix_length} has fewer distinct clips than distributed ranks"
            )
        candidates_by_prefix[prefix_length] = candidates

    prefix_order = tuple(
        sorted(
            prefix_lengths,
            key=lambda value: (_digest_integer(seed, "prefix", value), value),
        )
    )
    clips_by_step = []
    for optimizer_step in range(optimizer_steps):
        cycle, cycle_offset = divmod(optimizer_step, len(prefix_order))
        prefix_length = prefix_order[cycle_offset]
        candidates = candidates_by_prefix[prefix_length]
        offset = _digest_integer(seed, "offset", cycle, prefix_length) % len(candidates)
        stride = _coprime_stride(seed, len(candidates), cycle, prefix_length)
        rank_clips = tuple(
            StationaryTemporalClip(
                optimizer_step=optimizer_step,
                source_range_index=candidates[(offset + rank * stride) % len(candidates)][0],
                start_global_index=candidates[(offset + rank * stride) % len(candidates)][1],
                prefix_length=prefix_length,
                train_length=train_length,
            )
            for rank in range(world_size)
        )
        clips_by_step.append(rank_clips)
    return DistributedStationaryTemporalClipPlan(
        source_ranges=ranges,
        prefix_lengths=prefix_lengths,
        train_length=train_length,
        required_future_horizon=required_future_horizon,
        optimizer_steps=optimizer_steps,
        world_size=world_size,
        seed=seed,
        clips_by_step=tuple(clips_by_step),
    )

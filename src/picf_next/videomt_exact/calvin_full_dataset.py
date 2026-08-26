"""Lazy all-source CALVIN clips for complete VidEoMT adaptation."""

from __future__ import annotations

import hashlib
import json
import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Literal, Protocol

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinDatasetIndex
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
)

CalvinVidEoMTSplit = Literal["train", "heldout"]
CALVIN_VIDEOMT_EPISODE_SPLIT_IDENTITY = "calvin-raw-episode-modulo-v1"


class CalvinVerifiedRGBIndex(Protocol):
    dataset_id: str
    dataset_revision: str

    def validated_source_frame_arrays(
        self,
        global_index: int,
        *,
        fields: tuple[str, ...] | None = None,
        verify_relative_action: bool = True,
    ) -> Mapping[str, np.ndarray]: ...


class CalvinAllSourcePhysicalSidecar(Protocol):
    coverage: str

    def source_frame(self, global_index: int) -> CalvinPhysicalSupervisionFrame: ...


@dataclass(frozen=True, slots=True)
class CalvinVidEoMTWindowRange:
    """Consecutive valid window starts inside one raw source episode."""

    episode_index: int
    first_start: int
    last_start: int
    split: CalvinVidEoMTSplit

    def __post_init__(self) -> None:
        if (
            isinstance(self.episode_index, bool)
            or not isinstance(self.episode_index, int)
            or self.episode_index < 0
            or isinstance(self.first_start, bool)
            or not isinstance(self.first_start, int)
            or isinstance(self.last_start, bool)
            or not isinstance(self.last_start, int)
            or self.first_start < 0
            or self.last_start < self.first_start
            or self.split not in {"train", "heldout"}
        ):
            raise ContractError("CALVIN VidEoMT window range is invalid")

    @property
    def window_count(self) -> int:
        return self.last_start - self.first_start + 1


@dataclass(frozen=True, slots=True)
class CalvinVidEoMTEpisodeSplitPlan:
    """Compact episode-disjoint plan over all labelled physical events."""

    dataset_id: str
    dataset_revision: str
    clip_length: int
    heldout_modulus: int
    heldout_remainder: int
    ranges: tuple[CalvinVidEoMTWindowRange, ...]
    identity: str = CALVIN_VIDEOMT_EPISODE_SPLIT_IDENTITY

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.dataset_revision:
            raise ContractError("CALVIN VidEoMT split requires dataset identity")
        if (
            isinstance(self.clip_length, bool)
            or not isinstance(self.clip_length, int)
            or self.clip_length <= 0
            or isinstance(self.heldout_modulus, bool)
            or not isinstance(self.heldout_modulus, int)
            or self.heldout_modulus <= 1
            or isinstance(self.heldout_remainder, bool)
            or not isinstance(self.heldout_remainder, int)
            or not 0 <= self.heldout_remainder < self.heldout_modulus
        ):
            raise ContractError("CALVIN VidEoMT episode split controls are invalid")
        if not self.ranges or tuple(sorted(self.ranges, key=_range_sort_key)) != self.ranges:
            raise ContractError("CALVIN VidEoMT ranges must be nonempty and sorted")
        train_episodes = self.episode_indices("train")
        heldout_episodes = self.episode_indices("heldout")
        if not train_episodes or not heldout_episodes:
            raise ContractError("CALVIN VidEoMT split requires train and heldout episodes")
        if set(train_episodes).intersection(heldout_episodes):
            raise ContractError("CALVIN VidEoMT raw episodes cross split boundaries")
        if self.window_count("train") <= 0 or self.window_count("heldout") <= 0:
            raise ContractError("CALVIN VidEoMT split contains an empty window partition")

    def ranges_for(self, split: CalvinVidEoMTSplit) -> tuple[CalvinVidEoMTWindowRange, ...]:
        _validate_split(split)
        return tuple(value for value in self.ranges if value.split == split)

    def episode_indices(self, split: CalvinVidEoMTSplit) -> tuple[int, ...]:
        return tuple(sorted({value.episode_index for value in self.ranges_for(split)}))

    def window_count(self, split: CalvinVidEoMTSplit) -> int:
        return sum(value.window_count for value in self.ranges_for(split))

    def window_at(self, split: CalvinVidEoMTSplit, index: int) -> tuple[int, ...]:
        ranges = self.ranges_for(split)
        count = sum(value.window_count for value in ranges)
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise IndexError(f"CALVIN VidEoMT {split} window index is out of range")
        cumulative: list[int] = []
        total = 0
        for value in ranges:
            total += value.window_count
            cumulative.append(total)
        range_position = bisect_right(cumulative, index)
        prior = 0 if range_position == 0 else cumulative[range_position - 1]
        start = ranges[range_position].first_start + index - prior
        return tuple(range(start, start + self.clip_length))

    @property
    def fingerprint(self) -> str:
        payload = {
            "dataset_id": self.dataset_id,
            "dataset_revision": self.dataset_revision,
            "clip_length": self.clip_length,
            "heldout_modulus": self.heldout_modulus,
            "heldout_remainder": self.heldout_remainder,
            "identity": self.identity,
            "ranges": [asdict(value) for value in self.ranges],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class LazyCalvinVidEoMTClip:
    global_indices: tuple[int, ...]
    rgb_static: tuple[np.ndarray, ...]
    supervision: tuple[CalvinPhysicalSupervisionFrame, ...]

    def __post_init__(self) -> None:
        if (
            not self.global_indices
            or len(self.global_indices) != len(self.rgb_static)
            or len(self.global_indices) != len(self.supervision)
            or any(
                right != left + 1
                for left, right in zip(self.global_indices, self.global_indices[1:], strict=False)
            )
        ):
            raise ContractError("lazy CALVIN VidEoMT clip is not consecutive")
        if any(
            not isinstance(value, np.ndarray)
            or value.shape != (200, 200, 3)
            or value.dtype != np.uint8
            for value in self.rgb_static
        ):
            raise ContractError("lazy CALVIN VidEoMT clip contains invalid static RGB")
        if any(
            not isinstance(value, CalvinPhysicalSupervisionFrame) for value in self.supervision
        ):
            raise ContractError("lazy CALVIN VidEoMT clip contains invalid supervision")


def _validate_split(split: str) -> None:
    if split not in {"train", "heldout"}:
        raise ValueError(f"unsupported CALVIN VidEoMT split {split!r}")


def _range_sort_key(value: CalvinVidEoMTWindowRange) -> tuple[int, int, int, str]:
    return (value.episode_index, value.first_start, value.last_start, value.split)


def _consecutive_runs(values: Sequence[int]) -> tuple[tuple[int, int], ...]:
    if not values:
        return ()
    ordered = tuple(int(value) for value in values)
    if tuple(sorted(set(ordered))) != ordered:
        raise ContractError("CALVIN physical-event indices must be unique and sorted")
    result: list[tuple[int, int]] = []
    start = previous = ordered[0]
    for value in ordered[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return tuple(result)


def build_calvin_videomt_episode_split_plan(
    index: CalvinDatasetIndex,
    *,
    clip_length: int = 5,
    heldout_modulus: int = 5,
    heldout_remainder: int = 4,
) -> CalvinVidEoMTEpisodeSplitPlan:
    """Build a task-blind split before reading any image or owner target."""

    if not isinstance(index, CalvinDatasetIndex):
        raise TypeError("episode split requires a CalvinDatasetIndex")
    if (
        isinstance(clip_length, bool)
        or not isinstance(clip_length, int)
        or clip_length <= 0
        or isinstance(heldout_modulus, bool)
        or not isinstance(heldout_modulus, int)
        or heldout_modulus <= 1
        or isinstance(heldout_remainder, bool)
        or not isinstance(heldout_remainder, int)
        or not 0 <= heldout_remainder < heldout_modulus
    ):
        raise ValueError("invalid CALVIN VidEoMT episode split controls")

    ranges: list[CalvinVidEoMTWindowRange] = []
    for episode in index.episodes:
        event_indices = tuple(
            event.global_index for event in index.physical_episode_manifest(episode.index).events
        )
        split: CalvinVidEoMTSplit = (
            "heldout" if episode.index % heldout_modulus == heldout_remainder else "train"
        )
        for first, last in _consecutive_runs(event_indices):
            last_start = last - clip_length + 1
            if last_start < first:
                continue
            ranges.append(
                CalvinVidEoMTWindowRange(
                    episode_index=episode.index,
                    first_start=first,
                    last_start=last_start,
                    split=split,
                )
            )
    return CalvinVidEoMTEpisodeSplitPlan(
        dataset_id=index.dataset_id,
        dataset_revision=index.dataset_revision,
        clip_length=clip_length,
        heldout_modulus=heldout_modulus,
        heldout_remainder=heldout_remainder,
        ranges=tuple(sorted(ranges, key=_range_sort_key)),
    )


def _affine_permutation_parameters(*, count: int, seed: int, epoch: int) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("permutation count must be positive")
    digest = hashlib.sha256(f"{seed}:{epoch}:{count}".encode("ascii")).digest()
    if count == 1:
        return 0, 0
    multiplier = int.from_bytes(digest[:8], "big") % count
    if multiplier == 0:
        multiplier = 1
    while math.gcd(multiplier, count) != 1:
        multiplier = (multiplier + 1) % count
        if multiplier == 0:
            multiplier = 1
    offset = int.from_bytes(digest[8:16], "big") % count
    return multiplier, offset


def stateless_calvin_videomt_window(
    plan: CalvinVidEoMTEpisodeSplitPlan,
    *,
    split: CalvinVidEoMTSplit,
    visit_index: int,
    seed: int,
) -> tuple[int, ...]:
    """Select every window once per epoch without mutable sampler state."""

    if not isinstance(plan, CalvinVidEoMTEpisodeSplitPlan):
        raise TypeError("stateless selection requires an episode split plan")
    _validate_split(split)
    if isinstance(visit_index, bool) or not isinstance(visit_index, int) or visit_index < 0:
        raise ValueError("visit_index must be non-negative")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    count = plan.window_count(split)
    epoch, position = divmod(visit_index, count)
    multiplier, offset = _affine_permutation_parameters(
        count=count,
        seed=seed,
        epoch=epoch,
    )
    permuted = (multiplier * position + offset) % count
    return plan.window_at(split, permuted)


def materialize_calvin_videomt_clip(
    index: CalvinVerifiedRGBIndex,
    sidecar: CalvinAllSourcePhysicalSidecar,
    global_indices: Sequence[int],
) -> LazyCalvinVidEoMTClip:
    """Read only one requested clip from the manifest and lazy sidecar."""

    values = tuple(int(value) for value in global_indices)
    if not values or any(
        right != left + 1 for left, right in zip(values, values[1:], strict=False)
    ):
        raise ContractError("CALVIN VidEoMT materialization requires consecutive frames")
    if sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
        raise ContractError("complete donor adaptation requires all-source supervision")
    rgb: list[np.ndarray] = []
    supervision: list[CalvinPhysicalSupervisionFrame] = []
    for global_index in values:
        arrays = index.validated_source_frame_arrays(
            global_index,
            fields=("rgb_static",),
            verify_relative_action=False,
        )
        value = np.asarray(arrays["rgb_static"])
        if value.shape != (200, 200, 3) or value.dtype != np.uint8:
            raise ContractError("CALVIN source static RGB contract drifted")
        rgb.append(value)
        supervision.append(sidecar.source_frame(global_index))
    return LazyCalvinVidEoMTClip(values, tuple(rgb), tuple(supervision))

"""Hardware-neutral sample, comparison, and resume contracts.

The model decides what to learn.  This module only fixes which immutable
samples and stochastic inputs each optimizer update consumes.  A global plan
is partitioned after it is generated, so changing data-parallel world size or
gradient accumulation does not silently change the estimator.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import threading
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PLAN_SCHEMA = "picf-next.frozen-sample-plan.v1"
_PLAN_ALGORITHM = "sha256-epoch-sort.v1"
_EPISODE_STREAM_PLAN_SCHEMA = "picf-next.frozen-episode-stream-plan.v1"
_EPISODE_STREAM_PLAN_ALGORITHM = "sha256-episode-availability-heap.v1"
_INTERLEAVED_EPISODE_STREAM_PLAN_SCHEMA = "picf-next.frozen-episode-stream-plan.v2"
_INTERLEAVED_EPISODE_STREAM_PLAN_ALGORITHM = "sha256-episode-availability-heap-lane-interleave.v2"
_RESET_MIXTURE_STREAM_PLAN_SCHEMA = "picf-next.frozen-reset-mixture-stream-plan.v1"
_RESET_MIXTURE_STREAM_PLAN_ALGORITHM = "ceil-rational-reset-first-mixture.v1"
_RANDOMNESS_ALGORITHM = "sha256-stream-seed.v1"
_CONTRACT_SCHEMA = "picf-next.matched-run-contract.v1"
_PROGRESS_SCHEMA = "picf-next.run-progress.v1"
_CHECKPOINT_CONTROL_SCHEMA = "picf-next.checkpoint-control-manifest.v2"
_SHA256_LENGTH = 64

EXPERIMENT_ARMS = frozenset({"vanilla", "full_evidence", "picf"})


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not finite canonical JSON") from exc
    return encoded.encode("ascii")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_state_inventory(checkpoint_dir: Path, control_name: str) -> dict[str, Any]:
    """Hash every published state file except the self-describing control file."""

    if not checkpoint_dir.is_dir():
        raise ValueError(f"checkpoint directory does not exist: {checkpoint_dir}")
    inventory: dict[str, Any] = {}
    for path in sorted(checkpoint_dir.rglob("*")):
        relative = path.relative_to(checkpoint_dir).as_posix()
        if relative == control_name:
            continue
        if path.is_symlink():
            raise ValueError(f"checkpoint state cannot contain symlinks: {relative}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError(f"checkpoint state contains a non-regular file: {relative}")
        inventory[relative] = {
            "sha256": _sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not inventory:
        raise ValueError("checkpoint has no serialized state files")
    return inventory


def _validate_state_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or not value:
        raise ValueError("checkpoint control manifest has no state-file inventory")
    normalized: dict[str, Any] = {}
    for relative, record in value.items():
        if (
            not isinstance(relative, str)
            or not relative
            or relative.startswith("/")
            or "\\" in relative
            or any(part in {"", ".", ".."} for part in relative.split("/"))
        ):
            raise ValueError("checkpoint state inventory contains an unsafe path")
        if not isinstance(record, dict) or set(record) != {"sha256", "size_bytes"}:
            raise ValueError("checkpoint state inventory record is malformed")
        _require_sha256("checkpoint state-file sha256", record["sha256"])
        size = record["size_bytes"]
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError("checkpoint state-file size must be a non-negative integer")
        normalized[relative] = {"sha256": record["sha256"], "size_bytes": size}
    return normalized


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _require_sha256(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase SHA-256")


def _require_git_revision(name: str, value: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase git revision")


def _canonical_json_text(name: str, value: Mapping[str, Any] | str) -> str:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{name} must be canonical JSON") from exc
    elif isinstance(value, Mapping):
        parsed = dict(value)
    else:
        raise ValueError(f"{name} must be a mapping or canonical JSON string")
    if not isinstance(parsed, dict):
        raise ValueError(f"{name} must encode one JSON object")
    return _canonical_json_bytes(parsed).decode("ascii")


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    stale = tuple(path.parent.glob(f".{path.name}.tmp-*"))
    if path.exists() or path.is_symlink() or stale:
        raise FileExistsError(path)
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(_canonical_json_bytes(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        published = True
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        rename_completed = not temporary.exists() and (path.is_file() or path.is_symlink())
        if published or rename_completed:
            path.unlink(missing_ok=True)
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _ordered_sample_keys_sha256(sample_keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(b"picf-next.ordered-sample-keys.v1\0")
    for sample_key in sample_keys:
        encoded = sample_key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def _derive_seed(*parts: str) -> int:
    digest = hashlib.sha256()
    digest.update(b"picf-next.deterministic-seed.v1\0")
    for part in parts:
        encoded = part.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return int.from_bytes(digest.digest()[:8], byteorder="big") & ((1 << 63) - 1)


def derive_subseed(parent_seed: int, *coordinates: str) -> int:
    """Derive an order-independent seed for a view, transition, or target stream."""

    if not isinstance(parent_seed, int) or isinstance(parent_seed, bool) or parent_seed < 0:
        raise ValueError("parent_seed must be a non-negative integer")
    if any(not isinstance(coordinate, str) or not coordinate for coordinate in coordinates):
        raise ValueError("subseed coordinates must be non-empty strings")
    return _derive_seed(_RANDOMNESS_ALGORITHM, str(parent_seed), *coordinates)


@dataclass(frozen=True, slots=True)
class PlannedSample:
    sample_key: str
    sample_index: int
    augmentation_seed: int
    flow_noise_seed: int
    flow_timestep_seed: int


@dataclass(frozen=True, slots=True)
class PlannedGlobalBatch:
    optimizer_step: int
    epoch: int
    samples: tuple[PlannedSample, ...]


@dataclass(frozen=True, slots=True)
class PlannedMicrobatch:
    optimizer_step: int
    accumulation_index: int
    rank: int
    world_size: int
    samples: tuple[PlannedSample, ...]


@dataclass(frozen=True, slots=True)
class FrozenSamplePlan:
    """Random-access global batches independent of process topology.

    The ordered sample-key manifest is external and immutable.  Only its hash
    is serialized in plan metadata, avoiding a second copy of a large dataset
    index while still making reordering or replacement fail closed.
    """

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    sample_keys: tuple[str, ...]
    comparison_id: str
    seed: int
    global_batch_size: int
    total_steps: int
    _sample_keys_sha256: str = field(init=False, repr=False, compare=False)
    _epoch_cache: dict[int, tuple[int, ...]] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _cache_lock: threading.Lock = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        _require_nonempty("dataset_id", self.dataset_id)
        _require_nonempty("dataset_revision", self.dataset_revision)
        _require_sha256("dataset_manifest_sha256", self.dataset_manifest_sha256)
        _require_nonempty("comparison_id", self.comparison_id)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.global_batch_size, int)
            or isinstance(self.global_batch_size, bool)
            or self.global_batch_size <= 0
        ):
            raise ValueError("global_batch_size must be positive")
        if (
            not isinstance(self.total_steps, int)
            or isinstance(self.total_steps, bool)
            or self.total_steps <= 0
        ):
            raise ValueError("total_steps must be positive")
        if not self.sample_keys:
            raise ValueError("sample_keys cannot be empty")
        if len(self.sample_keys) < self.global_batch_size:
            raise ValueError("sample manifest must contain at least one full global batch")
        if any(not isinstance(key, str) or not key for key in self.sample_keys):
            raise ValueError("every sample key must be a non-empty string")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("sample keys must be unique")
        object.__setattr__(
            self,
            "_sample_keys_sha256",
            _ordered_sample_keys_sha256(self.sample_keys),
        )
        object.__setattr__(self, "_epoch_cache", {})
        object.__setattr__(self, "_cache_lock", threading.Lock())

    @property
    def sample_keys_sha256(self) -> str:
        return self._sample_keys_sha256

    @property
    def batches_per_epoch(self) -> int:
        return len(self.sample_keys) // self.global_batch_size

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": _PLAN_ALGORITHM,
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "global_batch_size": self.global_batch_size,
            "randomness_algorithm": _RANDOMNESS_ALGORITHM,
            "sample_count": len(self.sample_keys),
            "sample_keys_sha256": self.sample_keys_sha256,
            "schema": _PLAN_SCHEMA,
            "seed": self.seed,
            "total_steps": self.total_steps,
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_json(self.metadata)

    def _epoch_order(self, epoch: int) -> tuple[int, ...]:
        with self._cache_lock:
            cached = self._epoch_cache.get(epoch)
        if cached is not None:
            return cached

        prefix = f"{_PLAN_ALGORITHM}\0{self.comparison_id}\0{self.seed}\0{epoch}\0".encode()

        def key(index: int) -> tuple[bytes, int]:
            digest = hashlib.sha256(prefix + self.sample_keys[index].encode("utf-8")).digest()
            return digest, index

        order = tuple(sorted(range(len(self.sample_keys)), key=key))
        with self._cache_lock:
            self._epoch_cache.clear()
            self._epoch_cache[epoch] = order
        return order

    def global_batch(self, optimizer_step: int) -> PlannedGlobalBatch:
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or not 0 <= optimizer_step < self.total_steps
        ):
            raise IndexError("optimizer_step is outside the frozen plan")
        epoch, batch_index = divmod(optimizer_step, self.batches_per_epoch)
        order = self._epoch_order(epoch)
        start = batch_index * self.global_batch_size
        indices = order[start : start + self.global_batch_size]
        samples = tuple(
            PlannedSample(
                sample_key=self.sample_keys[index],
                sample_index=index,
                augmentation_seed=_derive_seed(
                    self.comparison_id,
                    str(self.seed),
                    str(optimizer_step),
                    str(slot),
                    self.sample_keys[index],
                    "augmentation",
                ),
                flow_noise_seed=_derive_seed(
                    self.comparison_id,
                    str(self.seed),
                    str(optimizer_step),
                    str(slot),
                    self.sample_keys[index],
                    "flow-noise",
                ),
                flow_timestep_seed=_derive_seed(
                    self.comparison_id,
                    str(self.seed),
                    str(optimizer_step),
                    str(slot),
                    self.sample_keys[index],
                    "flow-timestep",
                ),
            )
            for slot, index in enumerate(indices)
        )
        return PlannedGlobalBatch(optimizer_step=optimizer_step, epoch=epoch, samples=samples)

    def microbatch_for_rank(
        self,
        optimizer_step: int,
        *,
        rank: int,
        world_size: int,
        gradient_accumulation_steps: int,
        accumulation_index: int,
    ) -> PlannedMicrobatch:
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise ValueError("world_size must be positive")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if (
            not isinstance(accumulation_index, int)
            or isinstance(accumulation_index, bool)
            or not (0 <= accumulation_index < gradient_accumulation_steps)
        ):
            raise ValueError("accumulation_index is outside the optimizer step")
        partitions = world_size * gradient_accumulation_steps
        if self.global_batch_size % partitions:
            raise ValueError(
                "global batch must be divisible by world_size * gradient_accumulation_steps"
            )
        local_size = self.global_batch_size // partitions
        partition = accumulation_index * world_size + rank
        start = partition * local_size
        global_batch = self.global_batch(optimizer_step)
        return PlannedMicrobatch(
            optimizer_step=optimizer_step,
            accumulation_index=accumulation_index,
            rank=rank,
            world_size=world_size,
            samples=global_batch.samples[start : start + local_size],
        )

    def write_metadata(self, path: str | Path) -> None:
        _atomic_write_json(
            Path(path),
            {"metadata": self.metadata, "plan_sha256": self.plan_sha256},
        )

    @classmethod
    def from_metadata(
        cls,
        path: str | Path,
        *,
        sample_keys: Sequence[str],
    ) -> FrozenSamplePlan:
        payload = json.loads(Path(path).read_text())
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or metadata.get("schema") != _PLAN_SCHEMA:
            raise ValueError("unsupported frozen sample plan metadata")
        if metadata.get("algorithm") != _PLAN_ALGORITHM:
            raise ValueError("unsupported frozen sample plan algorithm")
        if metadata.get("randomness_algorithm") != _RANDOMNESS_ALGORITHM:
            raise ValueError("unsupported frozen sample randomness algorithm")
        recorded_hash = payload.get("plan_sha256")
        _require_sha256("plan_sha256", recorded_hash)
        if _sha256_json(metadata) != recorded_hash:
            raise ValueError("frozen sample plan metadata hash mismatch")
        plan = cls(
            dataset_id=metadata["dataset_id"],
            dataset_revision=metadata["dataset_revision"],
            dataset_manifest_sha256=metadata["dataset_manifest_sha256"],
            sample_keys=tuple(sample_keys),
            comparison_id=metadata["comparison_id"],
            seed=metadata["seed"],
            global_batch_size=metadata["global_batch_size"],
            total_steps=metadata["total_steps"],
        )
        if plan.plan_sha256 != recorded_hash:
            raise ValueError("current ordered sample manifest differs from the frozen plan")
        return plan


@dataclass(frozen=True, slots=True)
class EpisodeSampleSequence:
    """One immutable ordered episode represented by transition sample keys."""

    episode_key: str
    sample_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty("episode_key", self.episode_key)
        if not self.sample_keys:
            raise ValueError("episode sample keys cannot be empty")
        if any(not isinstance(key, str) or not key for key in self.sample_keys):
            raise ValueError("episode sample keys must be non-empty strings")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ValueError("sample keys must be unique within an episode")


@dataclass(frozen=True, slots=True)
class PlannedStreamTransition:
    """One current-frame sample in a persistent global episode lane."""

    lane_id: str
    episode_key: str
    episode_instance_id: str
    episode_epoch: int
    transition_index: int
    sample: PlannedSample


@dataclass(frozen=True, slots=True)
class PlannedStreamGlobalBatch:
    optimizer_step: int
    transitions: tuple[PlannedStreamTransition, ...]


@dataclass(frozen=True, slots=True)
class PlannedStreamMicrobatch:
    optimizer_step: int
    accumulation_index: int
    rank: int
    world_size: int
    transitions: tuple[PlannedStreamTransition, ...]


@dataclass(frozen=True, slots=True)
class _EpisodeLaneInterval:
    start_visit: int
    end_visit: int
    episode_index: int
    episode_epoch: int


@dataclass(frozen=True, slots=True)
class FrozenEpisodeStreamPlan:
    """Random-access causal episode lanes without history replay.

    Whole episode occurrences are drawn from deterministic epoch permutations
    and assigned to the next globally available lane. An occurrence never
    changes lane, while different epochs may overlap in wall-clock time. This
    preserves transition frequency for unequal episode lengths and keeps every
    optimizer step at one current transition per active global sample slot.
    Optional interleaving rotates multiple detached lanes through each slot.
    """

    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    episodes: tuple[EpisodeSampleSequence, ...]
    comparison_id: str
    seed: int
    global_batch_size: int
    total_steps: int
    lane_interleave_factor: int = 1
    _episode_manifest_sha256: str = field(init=False, repr=False, compare=False)
    _episode_offsets: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _lane_intervals: tuple[tuple[_EpisodeLaneInterval, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _lane_starts: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        _require_nonempty("dataset_id", self.dataset_id)
        _require_nonempty("dataset_revision", self.dataset_revision)
        _require_sha256("dataset_manifest_sha256", self.dataset_manifest_sha256)
        _require_nonempty("comparison_id", self.comparison_id)
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if (
            not isinstance(self.global_batch_size, int)
            or isinstance(self.global_batch_size, bool)
            or self.global_batch_size <= 0
        ):
            raise ValueError("global_batch_size must be positive")
        if (
            not isinstance(self.total_steps, int)
            or isinstance(self.total_steps, bool)
            or self.total_steps <= 0
        ):
            raise ValueError("total_steps must be positive")
        if (
            not isinstance(self.lane_interleave_factor, int)
            or isinstance(self.lane_interleave_factor, bool)
            or self.lane_interleave_factor <= 0
        ):
            raise ValueError("lane_interleave_factor must be positive")
        if self.lane_interleave_factor > self.total_steps:
            raise ValueError("lane_interleave_factor cannot exceed total_steps")
        if not self.episodes:
            raise ValueError("episode manifest cannot be empty")
        if any(not isinstance(episode, EpisodeSampleSequence) for episode in self.episodes):
            raise ValueError("episodes must contain EpisodeSampleSequence values")
        if len(self.episodes) < self.lane_count:
            raise ValueError("episode manifest must initially fill every global lane")
        episode_keys = tuple(episode.episode_key for episode in self.episodes)
        if len(set(episode_keys)) != len(episode_keys):
            raise ValueError("episode keys must be unique")
        flattened_keys = tuple(
            sample_key for episode in self.episodes for sample_key in episode.sample_keys
        )
        if len(set(flattened_keys)) != len(flattened_keys):
            raise ValueError("sample keys must be globally unique across episodes")
        manifest = tuple(
            {
                "episode_key": episode.episode_key,
                "sample_keys_sha256": _ordered_sample_keys_sha256(episode.sample_keys),
                "transition_count": len(episode.sample_keys),
            }
            for episode in self.episodes
        )
        offsets: list[int] = []
        offset = 0
        for episode in self.episodes:
            offsets.append(offset)
            offset += len(episode.sample_keys)
        object.__setattr__(self, "_episode_manifest_sha256", _sha256_json(manifest))
        object.__setattr__(self, "_episode_offsets", tuple(offsets))
        lane_intervals = self._build_lane_intervals()
        object.__setattr__(self, "_lane_intervals", lane_intervals)
        object.__setattr__(
            self,
            "_lane_starts",
            tuple(
                tuple(interval.start_visit for interval in intervals)
                for intervals in lane_intervals
            ),
        )

    @property
    def episode_manifest_sha256(self) -> str:
        return self._episode_manifest_sha256

    @property
    def lane_count(self) -> int:
        return self.global_batch_size * self.lane_interleave_factor

    @property
    def lane_ids(self) -> tuple[str, ...]:
        return tuple(f"global-lane-{index:05d}" for index in range(self.lane_count))

    @property
    def metadata(self) -> dict[str, Any]:
        metadata = {
            "algorithm": _EPISODE_STREAM_PLAN_ALGORITHM,
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "episode_count": len(self.episodes),
            "episode_manifest_sha256": self.episode_manifest_sha256,
            "global_batch_size": self.global_batch_size,
            "randomness_algorithm": _RANDOMNESS_ALGORITHM,
            "sample_count": sum(len(episode.sample_keys) for episode in self.episodes),
            "schema": _EPISODE_STREAM_PLAN_SCHEMA,
            "seed": self.seed,
            "total_steps": self.total_steps,
        }
        if self.lane_interleave_factor == 1:
            return metadata
        return {
            **metadata,
            "algorithm": _INTERLEAVED_EPISODE_STREAM_PLAN_ALGORITHM,
            "episode_order_algorithm": _EPISODE_STREAM_PLAN_ALGORITHM,
            "lane_count": self.lane_count,
            "lane_interleave_factor": self.lane_interleave_factor,
            "schema": _INTERLEAVED_EPISODE_STREAM_PLAN_SCHEMA,
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_json(self.metadata)

    def _epoch_order(self, epoch: int) -> tuple[int, ...]:
        prefix = (
            f"{_EPISODE_STREAM_PLAN_ALGORITHM}\0{self.comparison_id}\0{self.seed}\0{epoch}\0"
        ).encode()

        def key(index: int) -> tuple[bytes, int]:
            digest = hashlib.sha256(
                prefix + self.episodes[index].episode_key.encode("utf-8")
            ).digest()
            return digest, index

        return tuple(sorted(range(len(self.episodes)), key=key))

    def _build_lane_intervals(self) -> tuple[tuple[_EpisodeLaneInterval, ...], ...]:
        lanes: list[list[_EpisodeLaneInterval]] = [[] for _ in range(self.lane_count)]
        occurrence_epoch = 0
        occurrence_order = self._epoch_order(occurrence_epoch)
        occurrence_position = 0

        def next_occurrence() -> tuple[int, int]:
            nonlocal occurrence_epoch, occurrence_order, occurrence_position
            if occurrence_position == len(occurrence_order):
                occurrence_epoch += 1
                occurrence_order = self._epoch_order(occurrence_epoch)
                occurrence_position = 0
            episode_index = occurrence_order[occurrence_position]
            occurrence_position += 1
            return occurrence_epoch, episode_index

        availability: list[tuple[int, int]] = []
        for lane_index in range(self.lane_count):
            epoch, episode_index = next_occurrence()
            end_visit = len(self.episodes[episode_index].sample_keys)
            lanes[lane_index].append(_EpisodeLaneInterval(0, end_visit, episode_index, epoch))
            active_slot = lane_index % self.lane_interleave_factor
            heapq.heappush(
                availability,
                (
                    active_slot + self.lane_interleave_factor * end_visit,
                    lane_index,
                ),
            )

        while availability:
            available_step, lane_index = heapq.heappop(availability)
            if available_step >= self.total_steps:
                continue
            active_slot = lane_index % self.lane_interleave_factor
            start_visit = (available_step - active_slot) // self.lane_interleave_factor
            epoch, episode_index = next_occurrence()
            end_visit = start_visit + len(self.episodes[episode_index].sample_keys)
            lanes[lane_index].append(
                _EpisodeLaneInterval(start_visit, end_visit, episode_index, epoch)
            )
            heapq.heappush(
                availability,
                (
                    active_slot + self.lane_interleave_factor * end_visit,
                    lane_index,
                ),
            )
        return tuple(tuple(intervals) for intervals in lanes)

    def _transition_for_lane(
        self,
        lane_visit: int,
        lane_index: int,
    ) -> PlannedStreamTransition:
        starts = self._lane_starts[lane_index]
        interval_index = bisect_right(starts, lane_visit) - 1
        interval = self._lane_intervals[lane_index][interval_index]
        transition_index = lane_visit - interval.start_visit
        episode = self.episodes[interval.episode_index]
        sample_key = episode.sample_keys[transition_index]
        instance_id = f"epoch-{interval.episode_epoch:08d}/{episode.episode_key}"
        seed_coordinates = (
            self.comparison_id,
            str(self.seed),
            instance_id,
            str(transition_index),
            sample_key,
        )
        sample = PlannedSample(
            sample_key=sample_key,
            sample_index=self._episode_offsets[interval.episode_index] + transition_index,
            augmentation_seed=_derive_seed(*seed_coordinates, "augmentation"),
            flow_noise_seed=_derive_seed(*seed_coordinates, "flow-noise"),
            flow_timestep_seed=_derive_seed(*seed_coordinates, "flow-timestep"),
        )
        return PlannedStreamTransition(
            lane_id=f"global-lane-{lane_index:05d}",
            episode_key=episode.episode_key,
            episode_instance_id=instance_id,
            episode_epoch=interval.episode_epoch,
            transition_index=transition_index,
            sample=sample,
        )

    def global_batch(self, optimizer_step: int) -> PlannedStreamGlobalBatch:
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or not 0 <= optimizer_step < self.total_steps
        ):
            raise IndexError("optimizer_step is outside the frozen stream plan")
        active_slot = optimizer_step % self.lane_interleave_factor
        lane_visit = optimizer_step // self.lane_interleave_factor
        transitions = tuple(
            self._transition_for_lane(
                lane_visit,
                global_slot * self.lane_interleave_factor + active_slot,
            )
            for global_slot in range(self.global_batch_size)
        )
        return PlannedStreamGlobalBatch(
            optimizer_step=optimizer_step,
            transitions=transitions,
        )

    def microbatch_for_rank(
        self,
        optimizer_step: int,
        *,
        rank: int,
        world_size: int,
        gradient_accumulation_steps: int,
        accumulation_index: int,
    ) -> PlannedStreamMicrobatch:
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise ValueError("world_size must be positive")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if (
            not isinstance(accumulation_index, int)
            or isinstance(accumulation_index, bool)
            or not (0 <= accumulation_index < gradient_accumulation_steps)
        ):
            raise ValueError("accumulation_index is outside the optimizer step")
        partitions = world_size * gradient_accumulation_steps
        if self.global_batch_size % partitions:
            raise ValueError(
                "global batch must be divisible by world_size * gradient_accumulation_steps"
            )
        local_size = self.global_batch_size // partitions
        partition = accumulation_index * world_size + rank
        start = partition * local_size
        global_batch = self.global_batch(optimizer_step)
        return PlannedStreamMicrobatch(
            optimizer_step=optimizer_step,
            accumulation_index=accumulation_index,
            rank=rank,
            world_size=world_size,
            transitions=global_batch.transitions[start : start + local_size],
        )

    def write_metadata(self, path: str | Path) -> None:
        _atomic_write_json(
            Path(path),
            {"metadata": self.metadata, "plan_sha256": self.plan_sha256},
        )

    @classmethod
    def from_metadata(
        cls,
        path: str | Path,
        *,
        episodes: Sequence[EpisodeSampleSequence],
    ) -> FrozenEpisodeStreamPlan:
        payload = json.loads(Path(path).read_text())
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("unsupported frozen episode stream plan metadata")
        schema_algorithm = (metadata.get("schema"), metadata.get("algorithm"))
        supported = {
            (_EPISODE_STREAM_PLAN_SCHEMA, _EPISODE_STREAM_PLAN_ALGORITHM),
            (
                _INTERLEAVED_EPISODE_STREAM_PLAN_SCHEMA,
                _INTERLEAVED_EPISODE_STREAM_PLAN_ALGORITHM,
            ),
        }
        if schema_algorithm not in supported:
            raise ValueError("unsupported frozen episode stream plan algorithm")
        interleaved = schema_algorithm[0] == _INTERLEAVED_EPISODE_STREAM_PLAN_SCHEMA
        if interleaved and metadata.get("episode_order_algorithm") != (
            _EPISODE_STREAM_PLAN_ALGORITHM
        ):
            raise ValueError("unsupported frozen episode ordering algorithm")
        if metadata.get("randomness_algorithm") != _RANDOMNESS_ALGORITHM:
            raise ValueError("unsupported frozen sample randomness algorithm")
        recorded_hash = payload.get("plan_sha256")
        _require_sha256("plan_sha256", recorded_hash)
        if _sha256_json(metadata) != recorded_hash:
            raise ValueError("frozen episode stream plan metadata hash mismatch")
        plan = cls(
            dataset_id=metadata["dataset_id"],
            dataset_revision=metadata["dataset_revision"],
            dataset_manifest_sha256=metadata["dataset_manifest_sha256"],
            episodes=tuple(episodes),
            comparison_id=metadata["comparison_id"],
            seed=metadata["seed"],
            global_batch_size=metadata["global_batch_size"],
            total_steps=metadata["total_steps"],
            lane_interleave_factor=(
                metadata.get("lane_interleave_factor", 1) if interleaved else 1
            ),
        )
        if interleaved and metadata.get("lane_count") != plan.lane_count:
            raise ValueError("frozen episode stream lane count differs")
        if plan.plan_sha256 != recorded_hash:
            raise ValueError("current episode manifest differs from the frozen stream plan")
        return plan


@dataclass(frozen=True, slots=True)
class FrozenResetMixtureStreamPlan:
    """Deterministic real-reset/causal mixture with stateless reset samples.

    The causal sub-plan owns every persistent lane. Reset transitions use the
    same typed batch surface, but callers must use :meth:`component_for_step`
    to keep them outside the recurrent state transaction.
    """

    causal_plan: FrozenEpisodeStreamPlan
    reset_sample_keys: tuple[str, ...]
    reset_source_global_indices: tuple[int, ...]
    total_steps: int
    reset_numerator: int = 1
    reset_denominator: int = 2
    _sample_locations: dict[str, tuple[int, int]] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.causal_plan, FrozenEpisodeStreamPlan):
            raise TypeError("reset mixture requires a frozen causal episode plan")
        for name, value in (
            ("total_steps", self.total_steps),
            ("reset_numerator", self.reset_numerator),
            ("reset_denominator", self.reset_denominator),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.reset_numerator >= self.reset_denominator:
            raise ValueError("reset mixture weight must lie strictly between zero and one")
        if (self.total_steps * self.reset_numerator) % self.reset_denominator:
            raise ValueError("reset mixture budget must realize its rational weight exactly")
        if self.causal_plan.total_steps != self.causal_step_count:
            raise ValueError("causal sub-plan step count differs from the mixture schedule")
        if len(self.reset_sample_keys) != self.reset_sample_count:
            raise ValueError("reset sample count differs from the mixture schedule")
        if len(self.reset_source_global_indices) != len(self.reset_sample_keys):
            raise ValueError("reset source identities and sample keys differ")
        if (
            not self.reset_sample_keys
            or any(not isinstance(key, str) or not key for key in self.reset_sample_keys)
            or len(set(self.reset_sample_keys)) != len(self.reset_sample_keys)
        ):
            raise ValueError("reset sample keys must be nonempty and unique")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.reset_source_global_indices
        ) or len(set(self.reset_source_global_indices)) != len(self.reset_source_global_indices):
            raise ValueError("reset source global indices must be unique nonnegative integers")

        sample_locations: dict[str, tuple[int, int]] = {}
        flattened_index = 0
        for episode_index, episode in enumerate(self.episodes):
            for transition_index, sample_key in enumerate(episode.sample_keys):
                sample_locations[sample_key] = (episode_index, flattened_index)
                flattened_index += 1
                if transition_index == 0:
                    continue
        missing = sorted(set(self.reset_sample_keys) - set(sample_locations))
        if missing:
            raise ValueError(f"reset samples are absent from the causal domain: {missing[:3]}")
        first_keys = {episode.sample_keys[0] for episode in self.episodes}
        if any(key not in first_keys for key in self.reset_sample_keys):
            raise ValueError("reset mixture contains a non-transition-zero sample")
        causal_samples = {
            transition.sample.sample_key
            for step in range(self.causal_plan.total_steps)
            for transition in self.causal_plan.global_batch(step).transitions
        }
        overlap = causal_samples.intersection(self.reset_sample_keys)
        if overlap:
            raise ValueError("reset and causal mixture samples overlap")
        object.__setattr__(self, "_sample_locations", sample_locations)

    @property
    def dataset_id(self) -> str:
        return self.causal_plan.dataset_id

    @property
    def dataset_revision(self) -> str:
        return self.causal_plan.dataset_revision

    @property
    def dataset_manifest_sha256(self) -> str:
        return self.causal_plan.dataset_manifest_sha256

    @property
    def comparison_id(self) -> str:
        return self.causal_plan.comparison_id

    @property
    def seed(self) -> int:
        return self.causal_plan.seed

    @property
    def global_batch_size(self) -> int:
        return self.causal_plan.global_batch_size

    @property
    def lane_interleave_factor(self) -> int:
        return self.causal_plan.lane_interleave_factor

    @property
    def lane_count(self) -> int:
        return self.causal_plan.lane_count

    @property
    def lane_ids(self) -> tuple[str, ...]:
        return self.causal_plan.lane_ids

    @property
    def episodes(self) -> tuple[EpisodeSampleSequence, ...]:
        return self.causal_plan.episodes

    @property
    def reset_step_count(self) -> int:
        return self.total_steps * self.reset_numerator // self.reset_denominator

    @property
    def causal_step_count(self) -> int:
        return self.total_steps - self.reset_step_count

    @property
    def reset_sample_count(self) -> int:
        return self.reset_step_count * self.global_batch_size

    @staticmethod
    def _ceil_div(numerator: int, denominator: int) -> int:
        return (numerator + denominator - 1) // denominator

    def _reset_steps_before(self, optimizer_step: int) -> int:
        return self._ceil_div(
            optimizer_step * self.reset_numerator,
            self.reset_denominator,
        )

    def component_for_step(self, optimizer_step: int) -> str:
        self._validate_step(optimizer_step)
        before = self._reset_steps_before(optimizer_step)
        after = self._reset_steps_before(optimizer_step + 1)
        return "reset" if after > before else "causal"

    def component_index_for_step(self, optimizer_step: int) -> int:
        component = self.component_for_step(optimizer_step)
        resets_before = self._reset_steps_before(optimizer_step)
        return resets_before if component == "reset" else optimizer_step - resets_before

    def posterior_committed_for_step(self, optimizer_step: int) -> bool:
        return self.component_for_step(optimizer_step) == "causal"

    @property
    def component_schedule_sha256(self) -> str:
        return _sha256_json([self.component_for_step(step) for step in range(self.total_steps)])

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "algorithm": _RESET_MIXTURE_STREAM_PLAN_ALGORITHM,
            "causal_plan": self.causal_plan.metadata,
            "causal_plan_sha256": self.causal_plan.plan_sha256,
            "component_schedule_sha256": self.component_schedule_sha256,
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "global_batch_size": self.global_batch_size,
            "lane_count": self.lane_count,
            "lane_interleave_factor": self.lane_interleave_factor,
            "randomness_algorithm": _RANDOMNESS_ALGORITHM,
            "reset_denominator": self.reset_denominator,
            "reset_numerator": self.reset_numerator,
            "reset_sample_keys": list(self.reset_sample_keys),
            "reset_source_global_indices": list(self.reset_source_global_indices),
            "schema": _RESET_MIXTURE_STREAM_PLAN_SCHEMA,
            "seed": self.seed,
            "total_steps": self.total_steps,
        }

    @property
    def plan_sha256(self) -> str:
        return _sha256_json(self.metadata)

    def _validate_step(self, optimizer_step: int) -> None:
        if (
            not isinstance(optimizer_step, int)
            or isinstance(optimizer_step, bool)
            or not 0 <= optimizer_step < self.total_steps
        ):
            raise IndexError("optimizer_step is outside the frozen reset mixture plan")

    def _reset_global_batch(
        self,
        optimizer_step: int,
        component_index: int,
    ) -> PlannedStreamGlobalBatch:
        start = component_index * self.global_batch_size
        keys = self.reset_sample_keys[start : start + self.global_batch_size]
        active_slot = component_index % self.lane_interleave_factor
        transitions: list[PlannedStreamTransition] = []
        for global_slot, sample_key in enumerate(keys):
            episode_index, sample_index = self._sample_locations[sample_key]
            episode = self.episodes[episode_index]
            seed_coordinates = (
                self.comparison_id,
                str(self.seed),
                _RESET_MIXTURE_STREAM_PLAN_ALGORITHM,
                str(component_index),
                str(global_slot),
                sample_key,
            )
            transitions.append(
                PlannedStreamTransition(
                    lane_id=(
                        f"global-lane-{global_slot * self.lane_interleave_factor + active_slot:05d}"
                    ),
                    episode_key=episode.episode_key,
                    episode_instance_id=(f"reset-{component_index:08d}/{episode.episode_key}"),
                    episode_epoch=0,
                    transition_index=0,
                    sample=PlannedSample(
                        sample_key=sample_key,
                        sample_index=sample_index,
                        augmentation_seed=_derive_seed(*seed_coordinates, "augmentation"),
                        flow_noise_seed=_derive_seed(*seed_coordinates, "flow-noise"),
                        flow_timestep_seed=_derive_seed(
                            *seed_coordinates,
                            "flow-timestep",
                        ),
                    ),
                )
            )
        if len(transitions) != self.global_batch_size:
            raise RuntimeError("reset mixture schedule exhausted its frozen reset samples")
        return PlannedStreamGlobalBatch(
            optimizer_step=optimizer_step,
            transitions=tuple(transitions),
        )

    def global_batch(self, optimizer_step: int) -> PlannedStreamGlobalBatch:
        self._validate_step(optimizer_step)
        component = self.component_for_step(optimizer_step)
        component_index = self.component_index_for_step(optimizer_step)
        if component == "reset":
            return self._reset_global_batch(optimizer_step, component_index)
        causal = self.causal_plan.global_batch(component_index)
        return PlannedStreamGlobalBatch(
            optimizer_step=optimizer_step,
            transitions=causal.transitions,
        )

    def microbatch_for_rank(
        self,
        optimizer_step: int,
        *,
        rank: int,
        world_size: int,
        gradient_accumulation_steps: int,
        accumulation_index: int,
    ) -> PlannedStreamMicrobatch:
        if not isinstance(world_size, int) or isinstance(world_size, bool) or world_size <= 0:
            raise ValueError("world_size must be positive")
        if not isinstance(rank, int) or isinstance(rank, bool) or not 0 <= rank < world_size:
            raise ValueError("rank must be in [0, world_size)")
        if (
            not isinstance(gradient_accumulation_steps, int)
            or isinstance(gradient_accumulation_steps, bool)
            or gradient_accumulation_steps <= 0
        ):
            raise ValueError("gradient_accumulation_steps must be positive")
        if (
            not isinstance(accumulation_index, int)
            or isinstance(accumulation_index, bool)
            or not 0 <= accumulation_index < gradient_accumulation_steps
        ):
            raise ValueError("accumulation_index is outside the optimizer step")
        partitions = world_size * gradient_accumulation_steps
        if self.global_batch_size % partitions:
            raise ValueError(
                "global batch must be divisible by world_size * gradient_accumulation_steps"
            )
        local_size = self.global_batch_size // partitions
        partition = accumulation_index * world_size + rank
        start = partition * local_size
        global_batch = self.global_batch(optimizer_step)
        return PlannedStreamMicrobatch(
            optimizer_step=optimizer_step,
            accumulation_index=accumulation_index,
            rank=rank,
            world_size=world_size,
            transitions=global_batch.transitions[start : start + local_size],
        )

    def write_metadata(self, path: str | Path) -> None:
        _atomic_write_json(
            Path(path),
            {"metadata": self.metadata, "plan_sha256": self.plan_sha256},
        )

    @classmethod
    def from_metadata(
        cls,
        path: str | Path,
        *,
        episodes: Sequence[EpisodeSampleSequence],
    ) -> FrozenResetMixtureStreamPlan:
        payload = json.loads(Path(path).read_text())
        metadata = payload.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("schema") != _RESET_MIXTURE_STREAM_PLAN_SCHEMA
            or metadata.get("algorithm") != _RESET_MIXTURE_STREAM_PLAN_ALGORITHM
            or metadata.get("randomness_algorithm") != _RANDOMNESS_ALGORITHM
        ):
            raise ValueError("unsupported frozen reset mixture stream plan")
        recorded_hash = payload.get("plan_sha256")
        _require_sha256("plan_sha256", recorded_hash)
        if _sha256_json(metadata) != recorded_hash:
            raise ValueError("frozen reset mixture stream plan metadata hash mismatch")
        causal_metadata = metadata.get("causal_plan")
        if not isinstance(causal_metadata, dict):
            raise ValueError("reset mixture omits its causal sub-plan")
        interleaved = causal_metadata.get("schema") == _INTERLEAVED_EPISODE_STREAM_PLAN_SCHEMA
        causal = FrozenEpisodeStreamPlan(
            dataset_id=causal_metadata["dataset_id"],
            dataset_revision=causal_metadata["dataset_revision"],
            dataset_manifest_sha256=causal_metadata["dataset_manifest_sha256"],
            episodes=tuple(episodes),
            comparison_id=causal_metadata["comparison_id"],
            seed=causal_metadata["seed"],
            global_batch_size=causal_metadata["global_batch_size"],
            total_steps=causal_metadata["total_steps"],
            lane_interleave_factor=(
                causal_metadata.get("lane_interleave_factor", 1) if interleaved else 1
            ),
        )
        if causal.plan_sha256 != metadata.get("causal_plan_sha256"):
            raise ValueError("reset mixture causal sub-plan differs from metadata")
        plan = cls(
            causal_plan=causal,
            reset_sample_keys=tuple(metadata["reset_sample_keys"]),
            reset_source_global_indices=tuple(metadata["reset_source_global_indices"]),
            total_steps=metadata["total_steps"],
            reset_numerator=metadata["reset_numerator"],
            reset_denominator=metadata["reset_denominator"],
        )
        if plan.plan_sha256 != recorded_hash or plan.component_schedule_sha256 != metadata.get(
            "component_schedule_sha256"
        ):
            raise ValueError("current reset mixture domain differs from its frozen metadata")
        return plan


EpisodeStreamPlan = FrozenEpisodeStreamPlan | FrozenResetMixtureStreamPlan
TrainingPlan = FrozenSamplePlan | EpisodeStreamPlan


def load_frozen_episode_stream_plan(
    path: str | Path,
    *,
    episodes: Sequence[EpisodeSampleSequence],
) -> EpisodeStreamPlan:
    """Restore one supported causal or reset-mixture episode plan."""

    try:
        payload = json.loads(Path(path).read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid frozen episode stream plan: {path}") from error
    metadata = payload.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("frozen episode stream plan omits metadata")
    if metadata.get("schema") == _RESET_MIXTURE_STREAM_PLAN_SCHEMA:
        return FrozenResetMixtureStreamPlan.from_metadata(path, episodes=episodes)
    return FrozenEpisodeStreamPlan.from_metadata(path, episodes=episodes)


@dataclass(frozen=True, slots=True)
class ExperimentRunContract:
    """Fail-closed identity for one arm in a matched host experiment."""

    arm: str
    comparison_id: str
    code_revision: str
    host_name: str
    host_source_revision: str
    training_source_revision: str
    foundation_checkpoint_id: str
    foundation_checkpoint_revision: str
    checkpoint_manifest_sha256: str
    dataset_id: str
    dataset_revision: str
    dataset_manifest_sha256: str
    sample_plan_sha256: str
    optimizer_global_batch_size: int
    world_size: int
    gradient_accumulation_steps: int
    precision: str
    action_convention: str
    detached_context_frames: int
    gradient_transitions: int
    trainable_scope: str
    common_config_json: str
    arm_config_json: str

    def __post_init__(self) -> None:
        if self.arm not in EXPERIMENT_ARMS:
            raise ValueError(f"arm must be one of {sorted(EXPERIMENT_ARMS)}")
        for name in (
            "comparison_id",
            "host_name",
            "foundation_checkpoint_id",
            "dataset_id",
            "dataset_revision",
            "precision",
            "action_convention",
            "trainable_scope",
        ):
            _require_nonempty(name, getattr(self, name))
        for name in ("code_revision", "host_source_revision", "training_source_revision"):
            _require_git_revision(name, getattr(self, name))
        _require_nonempty("foundation_checkpoint_revision", self.foundation_checkpoint_revision)
        for name in (
            "checkpoint_manifest_sha256",
            "dataset_manifest_sha256",
            "sample_plan_sha256",
        ):
            _require_sha256(name, getattr(self, name))
        for name in (
            "optimizer_global_batch_size",
            "world_size",
            "gradient_accumulation_steps",
            "gradient_transitions",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            not isinstance(self.detached_context_frames, int)
            or isinstance(self.detached_context_frames, bool)
            or self.detached_context_frames < 0
        ):
            raise ValueError("detached_context_frames cannot be negative")
        partitions = self.world_size * self.gradient_accumulation_steps
        if self.optimizer_global_batch_size % partitions:
            raise ValueError(
                "optimizer_global_batch_size must be divisible by world_size * "
                "gradient_accumulation_steps"
            )
        object.__setattr__(
            self,
            "common_config_json",
            _canonical_json_text("common_config_json", self.common_config_json),
        )
        object.__setattr__(
            self,
            "arm_config_json",
            _canonical_json_text("arm_config_json", self.arm_config_json),
        )

    @classmethod
    def build(
        cls,
        *,
        common_config: Mapping[str, Any],
        arm_config: Mapping[str, Any],
        **fields: Any,
    ) -> ExperimentRunContract:
        return cls(
            common_config_json=_canonical_json_text("common_config", common_config),
            arm_config_json=_canonical_json_text("arm_config", arm_config),
            **fields,
        )

    @property
    def common_config(self) -> dict[str, Any]:
        return json.loads(self.common_config_json)

    @property
    def arm_config(self) -> dict[str, Any]:
        return json.loads(self.arm_config_json)

    @property
    def fairness_payload(self) -> dict[str, Any]:
        return {
            "action_convention": self.action_convention,
            "checkpoint_manifest_sha256": self.checkpoint_manifest_sha256,
            "code_revision": self.code_revision,
            "common_config": self.common_config,
            "comparison_id": self.comparison_id,
            "dataset_id": self.dataset_id,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "dataset_revision": self.dataset_revision,
            "detached_context_frames": self.detached_context_frames,
            "foundation_checkpoint_id": self.foundation_checkpoint_id,
            "foundation_checkpoint_revision": self.foundation_checkpoint_revision,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "gradient_transitions": self.gradient_transitions,
            "host_name": self.host_name,
            "host_source_revision": self.host_source_revision,
            "optimizer_global_batch_size": self.optimizer_global_batch_size,
            "precision": self.precision,
            "sample_plan_sha256": self.sample_plan_sha256,
            "schema": _CONTRACT_SCHEMA,
            "trainable_scope": self.trainable_scope,
            "training_source_revision": self.training_source_revision,
            "world_size": self.world_size,
        }

    @property
    def fairness_sha256(self) -> str:
        return _sha256_json(self.fairness_payload)

    @property
    def payload(self) -> dict[str, Any]:
        return {
            **self.fairness_payload,
            "arm": self.arm,
            "arm_config": self.arm_config,
            "fairness_sha256": self.fairness_sha256,
        }

    @property
    def contract_sha256(self) -> str:
        return _sha256_json(self.payload)

    def validate_plan(self, plan: TrainingPlan) -> None:
        if self.sample_plan_sha256 != plan.plan_sha256:
            raise ValueError("run contract and frozen sample plan hashes differ")
        if self.comparison_id != plan.comparison_id:
            raise ValueError("run contract and frozen sample plan comparison IDs differ")
        if self.dataset_id != plan.dataset_id or self.dataset_revision != plan.dataset_revision:
            raise ValueError("run contract and frozen sample plan datasets differ")
        if self.dataset_manifest_sha256 != plan.dataset_manifest_sha256:
            raise ValueError("run contract and frozen sample plan manifest hashes differ")
        if self.optimizer_global_batch_size != plan.global_batch_size:
            raise ValueError("run contract and frozen sample plan global batch sizes differ")


def validate_matched_abc(contracts: Sequence[ExperimentRunContract]) -> str:
    """Return the shared fairness hash after validating one exact A/B/C triplet."""

    frozen = tuple(contracts)
    if len(frozen) != len(EXPERIMENT_ARMS):
        raise ValueError("matched comparison requires exactly three run contracts")
    if {contract.arm for contract in frozen} != EXPERIMENT_ARMS:
        raise ValueError("matched comparison requires vanilla, full_evidence and picf arms")
    fairness_hashes = {contract.fairness_sha256 for contract in frozen}
    if len(fairness_hashes) != 1:
        raise ValueError("A/B/C common training contracts are not identical")
    return next(iter(fairness_hashes))


class RunProgress:
    """Checkpointable optimizer-attempt cursor with strict contract identity."""

    def __init__(
        self,
        *,
        contract_sha256: str,
        sample_plan_sha256: str,
        optimizer_global_batch_size: int,
    ) -> None:
        _require_sha256("contract_sha256", contract_sha256)
        _require_sha256("sample_plan_sha256", sample_plan_sha256)
        if (
            not isinstance(optimizer_global_batch_size, int)
            or isinstance(optimizer_global_batch_size, bool)
            or optimizer_global_batch_size <= 0
        ):
            raise ValueError("optimizer_global_batch_size must be positive")
        self.contract_sha256 = contract_sha256
        self.sample_plan_sha256 = sample_plan_sha256
        self.optimizer_global_batch_size = optimizer_global_batch_size
        self.attempted_optimizer_steps: int = 0
        self.successful_optimizer_steps: int = 0
        self.consumed_global_samples: int = 0

    @property
    def next_plan_step(self) -> int:
        return self.attempted_optimizer_steps

    def advance_optimizer_step(self, *, optimizer_step_was_skipped: bool) -> None:
        if not isinstance(optimizer_step_was_skipped, bool):
            raise TypeError("optimizer_step_was_skipped must be bool")
        self.attempted_optimizer_steps += 1
        if not optimizer_step_was_skipped:
            self.successful_optimizer_steps += 1
        self.consumed_global_samples += self.optimizer_global_batch_size

    def state_dict(self) -> dict[str, Any]:
        return {
            "attempted_optimizer_steps": self.attempted_optimizer_steps,
            "consumed_global_samples": self.consumed_global_samples,
            "contract_sha256": self.contract_sha256,
            "optimizer_global_batch_size": self.optimizer_global_batch_size,
            "sample_plan_sha256": self.sample_plan_sha256,
            "schema": _PROGRESS_SCHEMA,
            "successful_optimizer_steps": self.successful_optimizer_steps,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != _PROGRESS_SCHEMA:
            raise ValueError("unsupported run progress schema")
        if state.get("contract_sha256") != self.contract_sha256:
            raise ValueError("checkpoint run contract differs from the active run")
        if state.get("sample_plan_sha256") != self.sample_plan_sha256:
            raise ValueError("checkpoint sample plan differs from the active run")
        if state.get("optimizer_global_batch_size") != self.optimizer_global_batch_size:
            raise ValueError("checkpoint global batch size differs from the active run")
        attempted = state.get("attempted_optimizer_steps")
        successful = state.get("successful_optimizer_steps")
        consumed = state.get("consumed_global_samples")
        if (
            not isinstance(attempted, int)
            or isinstance(attempted, bool)
            or not isinstance(successful, int)
            or isinstance(successful, bool)
            or not isinstance(consumed, int)
            or isinstance(consumed, bool)
        ):
            raise ValueError("run progress counters must be integers")
        if attempted < 0 or successful < 0 or successful > attempted:
            raise ValueError("run progress optimizer counters are inconsistent")
        if consumed != attempted * self.optimizer_global_batch_size:
            raise ValueError("run progress consumed-sample count is inconsistent")
        self.attempted_optimizer_steps = attempted
        self.successful_optimizer_steps = successful
        self.consumed_global_samples = consumed

    def validate_capacity(self, plan: TrainingPlan) -> None:
        if self.next_plan_step > plan.total_steps:
            raise ValueError("checkpoint progress exceeds the frozen sample plan")


def write_control_manifest(
    path: str | Path,
    *,
    contract: ExperimentRunContract,
    plan: TrainingPlan,
    progress: RunProgress,
) -> None:
    contract.validate_plan(plan)
    progress.validate_capacity(plan)
    if progress.contract_sha256 != contract.contract_sha256:
        raise ValueError("progress and run contract hashes differ")
    if progress.sample_plan_sha256 != plan.plan_sha256:
        raise ValueError("progress and frozen sample plan hashes differ")
    control_path = Path(path)
    state_files = _checkpoint_state_inventory(control_path.parent, control_path.name)
    payload = {
        "contract": contract.payload,
        "contract_sha256": contract.contract_sha256,
        "plan": plan.metadata,
        "plan_sha256": plan.plan_sha256,
        "progress": progress.state_dict(),
        "progress_sha256": _sha256_json(progress.state_dict()),
        "schema": _CHECKPOINT_CONTROL_SCHEMA,
        "state_files": state_files,
        "state_files_sha256": _sha256_json(state_files),
    }
    payload["manifest_sha256"] = _sha256_json(payload)
    _atomic_write_json(control_path, payload)


def validate_control_manifest(
    path: str | Path,
    *,
    contract: ExperimentRunContract,
    plan: TrainingPlan,
) -> dict[str, Any]:
    control_path = Path(path)
    payload = json.loads(control_path.read_text())
    if payload.get("schema") != _CHECKPOINT_CONTROL_SCHEMA:
        raise ValueError("unsupported checkpoint control manifest")
    manifest_sha256 = payload.get("manifest_sha256")
    _require_sha256("checkpoint control manifest sha256", manifest_sha256)
    manifest_payload = dict(payload)
    del manifest_payload["manifest_sha256"]
    if _sha256_json(manifest_payload) != manifest_sha256:
        raise ValueError("checkpoint control manifest payload is corrupt")
    if payload.get("contract_sha256") != contract.contract_sha256:
        raise ValueError("checkpoint control manifest belongs to another run contract")
    if payload.get("plan_sha256") != plan.plan_sha256:
        raise ValueError("checkpoint control manifest belongs to another sample plan")
    if _sha256_json(payload.get("contract")) != contract.contract_sha256:
        raise ValueError("checkpoint run contract payload is corrupt")
    if _sha256_json(payload.get("plan")) != plan.plan_sha256:
        raise ValueError("checkpoint sample plan payload is corrupt")
    progress = payload.get("progress")
    if not isinstance(progress, dict):
        raise ValueError("checkpoint control manifest has no progress state")
    if _sha256_json(progress) != payload.get("progress_sha256"):
        raise ValueError("checkpoint progress payload is corrupt")
    expected_files = _validate_state_inventory(payload.get("state_files"))
    if _sha256_json(expected_files) != payload.get("state_files_sha256"):
        raise ValueError("checkpoint state-file inventory is corrupt")
    observed_files = _checkpoint_state_inventory(control_path.parent, control_path.name)
    if observed_files != expected_files:
        raise ValueError("checkpoint serialized state files are missing, added, or corrupt")
    return progress

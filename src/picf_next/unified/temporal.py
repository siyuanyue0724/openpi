"""Long-state exposure without long image-backbone BPTT."""

from __future__ import annotations

import json
import math
import re
import struct
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from hashlib import sha256
from typing import Any

import torch

from picf_next.unified.state import UnifiedBeliefState

_LANE_MAGIC = b"PICFLN01"
_LANE_VERSION = 1
_LANE_HEADER = struct.Struct("<8sBI")
_LANE_RECORD = struct.Struct("<qII")


class LaneStateError(RuntimeError):
    """A cached posterior cannot safely represent the requested stream step."""


@dataclass(frozen=True, slots=True)
class StateStamp:
    episode_key: str
    frame_index: int
    schema_digest: str
    model_family_digest: str
    optimizer_step: int

    def __post_init__(self) -> None:
        identifiers = (self.episode_key, self.schema_digest, self.model_family_digest)
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("state stamp identifiers must be non-empty")
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.frame_index, self.optimizer_step)
        ):
            raise TypeError("state stamp indices must be integers")
        if self.frame_index < 0 or self.optimizer_step < 0:
            raise ValueError("state stamp indices must be non-negative")

    def as_dict(self) -> dict[str, str | int]:
        return {
            "episode_key": self.episode_key,
            "frame_index": self.frame_index,
            "schema_digest": self.schema_digest,
            "model_family_digest": self.model_family_digest,
            "optimizer_step": self.optimizer_step,
        }


@dataclass(frozen=True, slots=True)
class StampedBelief:
    state: UnifiedBeliefState
    stamp: StateStamp


class EpisodeLaneBank:
    """One detached posterior per ordered worker lane."""

    def __init__(self) -> None:
        self._lanes: dict[int, StampedBelief] = {}

    def __len__(self) -> int:
        return len(self._lanes)

    def reset(self, lane_id: int) -> None:
        if isinstance(lane_id, bool) or not isinstance(lane_id, int) or lane_id < 0:
            raise ValueError("lane_id must be a non-negative integer")
        self._lanes.pop(lane_id, None)

    def write(
        self,
        lane_id: int,
        state: UnifiedBeliefState,
        stamp: StateStamp,
        *,
        allow_episode_reset: bool = False,
    ) -> None:
        self.write_batch(((lane_id, state, stamp, allow_episode_reset),))

    def write_batch(
        self,
        records: Sequence[tuple[int, UnifiedBeliefState, StateStamp, bool]],
    ) -> None:
        """Atomically publish a batch of lane states after validating every record."""

        if not records:
            raise ValueError("at least one lane record is required")
        lane_ids = [record[0] for record in records]
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("a batch cannot write the same lane twice")
        staged = dict(self._lanes)
        for lane_id, state, stamp, allow_episode_reset in records:
            if isinstance(lane_id, bool) or not isinstance(lane_id, int) or lane_id < 0:
                raise ValueError("lane_id must be a non-negative integer")
            if not isinstance(allow_episode_reset, bool):
                raise TypeError("allow_episode_reset must be boolean")
            if state.batch_size != 1:
                raise ValueError("each lane record must contain exactly one batch item")
            if allow_episode_reset:
                staged.pop(lane_id, None)
            previous = staged.get(lane_id)
            if previous is not None:
                same_episode = previous.stamp.episode_key == stamp.episode_key
                if not same_episode:
                    raise LaneStateError("episode changed without an explicit lane reset")
                if stamp.frame_index != previous.stamp.frame_index + 1:
                    raise LaneStateError("lane frames must advance by exactly one")
                if stamp.schema_digest != previous.stamp.schema_digest:
                    raise LaneStateError("lane schema changed without an explicit reset")
                if stamp.model_family_digest != previous.stamp.model_family_digest:
                    raise LaneStateError("lane model family changed without an explicit reset")
                if stamp.optimizer_step < previous.stamp.optimizer_step:
                    raise LaneStateError("optimizer_step cannot move backwards")
            staged[lane_id] = StampedBelief(state=state.detached(), stamp=stamp)
        self._lanes = staged

    def read_for_next_frame(
        self,
        lane_id: int,
        *,
        episode_key: str,
        frame_index: int,
        schema_digest: str,
        model_family_digest: str,
        optimizer_step: int,
        max_optimizer_lag: int,
    ) -> UnifiedBeliefState | None:
        if isinstance(lane_id, bool) or not isinstance(lane_id, int) or lane_id < 0:
            raise ValueError("lane_id must be a non-negative integer")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (frame_index, optimizer_step, max_optimizer_lag)
        ):
            raise ValueError("lane read indices and staleness must be non-negative integers")
        record = self._lanes.get(lane_id)
        if record is None:
            return None
        stamp = record.stamp
        if stamp.episode_key != episode_key:
            raise LaneStateError("cached lane belongs to a different episode")
        if frame_index != stamp.frame_index + 1:
            raise LaneStateError("requested frame is not contiguous with cached state")
        if stamp.schema_digest != schema_digest:
            raise LaneStateError("cached lane schema is incompatible")
        if stamp.model_family_digest != model_family_digest:
            raise LaneStateError("cached lane model family is incompatible")
        lag = optimizer_step - stamp.optimizer_step
        if lag < 0:
            raise LaneStateError("cached lane comes from a future optimizer step")
        if max_optimizer_lag < 0 or lag > max_optimizer_lag:
            raise LaneStateError("cached lane exceeds the optimizer staleness budget")
        return record.state.detached()

    def snapshot(self) -> bytes:
        payload = bytearray(_LANE_HEADER.pack(_LANE_MAGIC, _LANE_VERSION, len(self._lanes)))
        for lane_id, record in sorted(self._lanes.items()):
            metadata = json.dumps(
                record.stamp.as_dict(),
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            state = record.state.serialize()
            payload.extend(_LANE_RECORD.pack(lane_id, len(metadata), len(state)))
            payload.extend(metadata)
            payload.extend(state)
        return bytes(payload)

    @classmethod
    def from_snapshot(cls, payload: bytes) -> EpisodeLaneBank:
        if len(payload) < _LANE_HEADER.size:
            raise ValueError("lane snapshot is truncated")
        magic, version, count = _LANE_HEADER.unpack_from(payload)
        if magic != _LANE_MAGIC or version != _LANE_VERSION:
            raise ValueError("lane snapshot schema is unsupported")
        cursor = _LANE_HEADER.size
        bank = cls()
        for _ in range(count):
            if cursor + _LANE_RECORD.size > len(payload):
                raise ValueError("lane snapshot record is truncated")
            lane_id, metadata_size, state_size = _LANE_RECORD.unpack_from(payload, cursor)
            cursor += _LANE_RECORD.size
            end_metadata = cursor + metadata_size
            end_state = end_metadata + state_size
            if end_state > len(payload):
                raise ValueError("lane snapshot payload is truncated")
            metadata = json.loads(payload[cursor:end_metadata].decode("utf-8"))
            stamp = StateStamp(**metadata)
            state = UnifiedBeliefState.deserialize(payload[end_metadata:end_state])
            if lane_id < 0:
                raise ValueError("lane snapshot contains a negative lane ID")
            if lane_id in bank._lanes:
                raise ValueError("lane snapshot contains a duplicate lane")
            bank._lanes[lane_id] = StampedBelief(state=state, stamp=stamp)
            cursor = end_state
        if cursor != len(payload):
            raise ValueError("lane snapshot contains trailing bytes")
        return bank

    @property
    def digest(self) -> str:
        return sha256(self.snapshot()).hexdigest()


@dataclass(frozen=True, slots=True)
class SparseBPTTPlan:
    burn_in_steps: int
    differentiable_steps: int
    state_age: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (self.burn_in_steps, self.differentiable_steps, self.state_age)
        ):
            raise TypeError("sparse-BPTT controls must be integers")
        if self.burn_in_steps < 0 or self.state_age < 0:
            raise ValueError("burn-in and state age must be non-negative")
        if not 2 <= self.differentiable_steps <= 4:
            raise ValueError("differentiable window must contain 2-4 steps")

    @property
    def loaded_steps(self) -> int:
        return self.burn_in_steps + self.differentiable_steps


def sparse_bptt_plan(
    *,
    state_age: int,
    draw: float,
    differentiable_probability: float,
    max_burn_in: int = 2,
) -> SparseBPTTPlan | None:
    """Select a low-frequency local-credit window without changing runtime state."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) for value in (state_age, max_burn_in)
    ):
        raise TypeError("state_age and max_burn_in must be integers")
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in (draw, differentiable_probability)
    ):
        raise TypeError("draw and differentiable_probability must be real-valued")
    if not all(math.isfinite(value) for value in (draw, differentiable_probability)):
        raise ValueError("draw and differentiable_probability must be finite")
    if state_age < 0 or max_burn_in < 0:
        raise ValueError("state_age and max_burn_in must be non-negative")
    if not 0 <= draw <= 1 or not 0 <= differentiable_probability <= 1:
        raise ValueError("draw and probability must lie in [0, 1]")
    if draw >= differentiable_probability:
        return None
    differentiable_steps = 2 + min(int(draw * 3 / max(differentiable_probability, 1e-12)), 2)
    return SparseBPTTPlan(
        burn_in_steps=min(state_age, max_burn_in),
        differentiable_steps=differentiable_steps,
        state_age=state_age,
    )


@dataclass(frozen=True, slots=True)
class PackedHorizonPlan:
    horizons: tuple[int, ...]
    source_frame: int
    target_data_digest: str
    target_model_digest: str
    training_only: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.training_only, bool):
            raise TypeError("training_only must be boolean")
        if not self.training_only:
            raise ValueError("packed future horizons are training-only")
        if isinstance(self.source_frame, bool) or not isinstance(self.source_frame, int):
            raise TypeError("source_frame must be an integer")
        if self.source_frame < 0:
            raise ValueError("source_frame must be non-negative")
        for name, value in (
            ("target_data_digest", self.target_data_digest),
            ("target_model_digest", self.target_model_digest),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{name} must be one lowercase SHA-256 digest")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in self.horizons):
            raise TypeError("packed horizons must be integers")
        if not self.horizons or tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be non-empty, sorted and unique")
        if self.horizons[0] <= 0:
            raise ValueError("horizons must be strictly positive")


def logarithmic_horizons(max_horizon: int) -> tuple[int, ...]:
    if isinstance(max_horizon, bool) or not isinstance(max_horizon, int):
        raise TypeError("max_horizon must be an integer")
    if max_horizon <= 0:
        raise ValueError("max_horizon must be positive")
    values = []
    horizon = 1
    while horizon < max_horizon:
        values.append(horizon)
        horizon *= 2
    values.append(max_horizon)
    return tuple(values)


def semigroup_consistency_error(
    direct_prediction: torch.Tensor,
    recursive_prediction: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    if direct_prediction.shape != recursive_prediction.shape:
        raise ValueError("direct and recursive predictions must match")
    if valid.shape != direct_prediction.shape[:-1] or valid.dtype != torch.bool:
        raise ValueError("valid must match prediction axes except feature width")
    squared = (direct_prediction - recursive_prediction).float().square().mean(dim=-1)
    count = valid.sum()
    if count == 0:
        return squared.sum() * 0
    return squared.masked_select(valid).sum() / count


_FORBIDDEN_DEPLOY_FIELDS = frozenset(
    {
        "actions",
        "action",
        "action_target",
        "action_targets",
        "bbox",
        "target_action",
        "target_actions",
        "future",
        "future_features",
        "future_images",
        "future_latents",
        "future_observation",
        "future_target",
        "mask_target",
        "mask_targets",
        "box_target",
        "bbox_target",
        "boxes",
        "segmentation",
        "ground_truth",
        "labels",
        "object_id",
        "oracle",
        "owner_raster",
        "simulator_object_id",
        "sidecar",
        "supervision",
        "teacher_target",
    }
)

_FORBIDDEN_DEPLOY_TYPES = frozenset(
    {
        "BeliefSetTarget",
        "BeliefStateTarget",
        "PackedHorizonPlan",
        "PredictionQueryRequest",
        "PredictiveTarget",
    }
)


def _normalized_field_name(value: str) -> str:
    snake = re.sub(r"(?<!^)(?=[A-Z])", "_", value).lower().replace("-", "_")
    return re.sub(r"_+", "_", snake)


def _is_training_only_field(value: str) -> bool:
    normalized = _normalized_field_name(value)
    parts = set(normalized.split("_"))
    return (
        normalized in _FORBIDDEN_DEPLOY_FIELDS
        or normalized.startswith("future_")
        or normalized.startswith("teacher_")
        or bool(parts.intersection({"target", "targets", "label", "labels", "oracle"}))
    )


def assert_deploy_payload_is_causal(payload: Mapping[str, Any]) -> None:
    """Reject loss-side fields before model-input assembly."""

    offending: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise TypeError("deploy payload mapping keys must be strings")
                normalized = _normalized_field_name(key)
                if _is_training_only_field(key):
                    offending.add(normalized)
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)
        elif is_dataclass(value):
            type_name = type(value).__name__
            if type_name in _FORBIDDEN_DEPLOY_TYPES:
                offending.add(type_name)
            for field in fields(value):
                if _is_training_only_field(field.name):
                    offending.add(_normalized_field_name(field.name))
                visit(getattr(value, field.name))

    visit(payload)
    if offending:
        raise ValueError(f"deploy payload contains training-only fields: {sorted(offending)}")

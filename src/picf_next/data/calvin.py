"""Fail-closed CALVIN transition, language-segment and host-sample adapters.

The adapter reads only deploy-visible observations and the demonstrator action.
``scene_obs``, simulator identities, masks and future observations remain outside
this module. Structural targets are produced by a separate loss-side pipeline.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import threading
from bisect import bisect_right
from collections import OrderedDict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.dataset_manifest import DatasetFileManifest, read_verified_dataset_file
from picf_next.data.robot_record import ArrayObservation, RobotTransitionRecord

CALVIN_CONTRACT = "calvin-transition/v1"
CALVIN_PHYSICAL_EVENT_CONTRACT = "calvin-physical-event/v1"
CALVIN_RAW_ACTION_CONTROL_SPAN_CONTRACT = "calvin-raw-action-control-span/v1"
CALVIN_DEBUG_DATASET_ID = "mees/calvin-debug-dataset"
CALVIN_DEBUG_REVISION = "sha256:c66d09147e2c806b244f18ea7d61e388d4dac11f828929779437f728d03e1204"
CALVIN_CONTROL_HZ = 30
CALVIN_ACTION_CACHE_CAPACITY = 65536

CALVIN_STATE_AXES = (
    "tcp.position.x",
    "tcp.position.y",
    "tcp.position.z",
    "tcp.euler.x",
    "tcp.euler.y",
    "tcp.euler.z",
    "gripper.opening_width",
    *(f"arm.joint.{index}" for index in range(7)),
    "gripper.previous_command",
)
CALVIN_ACTION_AXES = (
    "delta_tcp.position.x",
    "delta_tcp.position.y",
    "delta_tcp.position.z",
    "delta_tcp.euler.x",
    "delta_tcp.euler.y",
    "delta_tcp.euler.z",
    "gripper.command",
)

CALVIN_OBSERVATION_SPECS: tuple[tuple[str, str, tuple[int, ...], np.dtype, str], ...] = (
    (
        "rgb_static",
        "observation.images.rgb_static",
        (200, 200, 3),
        np.dtype(np.uint8),
        "sRGB uint8",
    ),
    (
        "rgb_gripper",
        "observation.images.rgb_gripper",
        (84, 84, 3),
        np.dtype(np.uint8),
        "sRGB uint8",
    ),
    ("depth_static", "observation.depth.static", (200, 200), np.dtype(np.float32), "meters"),
    ("depth_gripper", "observation.depth.gripper", (84, 84), np.dtype(np.float32), "meters"),
    (
        "rgb_tactile",
        "observation.tactile.rgb",
        (160, 120, 6),
        np.dtype(np.uint8),
        "two concatenated DIGIT sRGB uint8 views",
    ),
    (
        "depth_tactile",
        "observation.tactile.depth",
        (160, 120, 2),
        np.dtype(np.float32),
        "two DIGIT deformation-depth views",
    ),
)
CALVIN_HOST_IMAGE_KEYS = (
    "observation.images.image",
    "observation.images.wrist_image",
)

Float32Vector = NDArray[np.float32]
BoolVector = NDArray[np.bool_]
_MAPPING_PROXY_TYPE = type(MappingProxyType({}))


def _readonly_array(value: Any, *, dtype: np.dtype, name: str) -> NDArray:
    array = np.asarray(value)
    if array.dtype != dtype:
        if np.issubdtype(dtype, np.floating) and np.issubdtype(array.dtype, np.floating):
            array = array.astype(dtype)
        else:
            raise ContractError(f"{name} dtype must be {dtype}, got {array.dtype}")
    if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
        raise ContractError(f"{name} contains NaN or infinity")
    contiguous = np.ascontiguousarray(array)
    return np.frombuffer(contiguous.tobytes(order="C"), dtype=dtype).reshape(array.shape)


def _readonly_float32_vector(value: Any, *, width: int, name: str) -> Float32Vector:
    array = np.asarray(value)
    if array.shape != (width,) or not np.issubdtype(array.dtype, np.floating):
        raise ContractError(f"{name} must be a floating vector of shape ({width},)")
    return _readonly_array(array, dtype=np.dtype(np.float32), name=name)


def _readonly_validity(width: int) -> BoolVector:
    return _readonly_array(
        np.ones(width, dtype=np.bool_),
        dtype=np.dtype(np.bool_),
        name="validity",
    )


def _expected_relative_action(absolute_action: NDArray, robot_obs: NDArray) -> NDArray:
    position = np.clip(absolute_action[:3] - robot_obs[:3], -0.02, 0.02) / 0.02
    angle_delta = absolute_action[3:6] - robot_obs[3:6]
    angle_delta = (angle_delta + np.pi) % (2.0 * np.pi) - np.pi
    orientation = np.clip(angle_delta, -0.05, 0.05) / 0.05
    return np.concatenate((position, orientation, absolute_action[-1:]))


def _validated_relative_action(
    frame: Mapping[str, NDArray],
    *,
    verify_relative_action: bool,
) -> Float32Vector:
    """Validate only the state/action triplet and return the relative command."""

    required = {"robot_obs", "actions", "rel_actions"}
    missing = sorted(required.difference(frame))
    if missing:
        raise ContractError(f"CALVIN frame is missing action fields: {missing}")
    robot_obs = np.asarray(frame["robot_obs"])
    absolute_action = np.asarray(frame["actions"])
    relative_action = np.asarray(frame["rel_actions"])
    if robot_obs.shape != (15,) or not np.issubdtype(robot_obs.dtype, np.floating):
        raise ContractError("CALVIN robot_obs must be a floating vector of shape (15,)")
    if (
        absolute_action.shape != (7,)
        or relative_action.shape != (7,)
        or not np.issubdtype(absolute_action.dtype, np.floating)
        or not np.issubdtype(relative_action.dtype, np.floating)
    ):
        raise ContractError(
            "CALVIN absolute and relative actions must be floating shape-(7,) vectors"
        )
    if (
        not np.isfinite(robot_obs).all()
        or not np.isfinite(absolute_action).all()
        or not np.isfinite(relative_action).all()
    ):
        raise ContractError("CALVIN robot state or action contains NaN or infinity")
    if verify_relative_action:
        expected = _expected_relative_action(absolute_action, robot_obs)
        if not np.allclose(relative_action, expected, atol=1e-7, rtol=0.0):
            raise ContractError("CALVIN rel_actions disagrees with the official conversion")
    return _readonly_float32_vector(relative_action, width=7, name="rel_actions")


def validate_calvin_source_frame(
    frame: Mapping[str, NDArray],
    *,
    verify_relative_action: bool = True,
) -> None:
    """Validate one raw CALVIN NPZ payload without assigning task metadata."""

    if not isinstance(verify_relative_action, bool):
        raise ContractError("verify_relative_action must be boolean")

    required = {
        "robot_obs",
        "actions",
        "rel_actions",
        *(spec[0] for spec in CALVIN_OBSERVATION_SPECS),
    }
    missing = sorted(required.difference(frame))
    if missing:
        raise ContractError(f"CALVIN frame is missing fields: {missing}")
    _validated_relative_action(frame, verify_relative_action=verify_relative_action)

    for source_key, _, shape, dtype, _ in CALVIN_OBSERVATION_SPECS:
        value = np.asarray(frame[source_key])
        if value.shape != shape:
            raise ContractError(f"CALVIN {source_key} shape must be {shape}, got {value.shape}")
        if value.dtype != dtype:
            raise ContractError(f"CALVIN {source_key} dtype must be {dtype}, got {value.dtype}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ContractError(f"CALVIN {source_key} contains NaN or infinity")


@dataclass(frozen=True, slots=True)
class CalvinEpisode:
    index: int
    start: int
    end: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool | np.bool_) or not isinstance(value, Integral)
            for value in (self.index, self.start, self.end)
        ):
            raise ContractError("CALVIN episode bounds must be integers")
        if self.index < 0 or self.start < 0 or self.end < self.start:
            raise ContractError("CALVIN episode bounds are invalid")

    @property
    def length(self) -> int:
        return self.end - self.start + 1


@dataclass(frozen=True, slots=True)
class CalvinLanguageSegment:
    index: int
    start: int
    end: int
    task_key: str
    instruction: str
    episode_index: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool | np.bool_) or not isinstance(value, Integral)
            for value in (self.index, self.start, self.end, self.episode_index)
        ):
            raise ContractError("CALVIN language indices must be integers")
        if self.index < 0 or self.start < 0 or self.end <= self.start:
            raise ContractError("CALVIN language segment must contain at least one transition")
        if (
            not isinstance(self.task_key, str)
            or not self.task_key
            or not isinstance(self.instruction, str)
            or not self.instruction
            or self.episode_index < 0
        ):
            raise ContractError("CALVIN language metadata is incomplete")

    @property
    def frame_count(self) -> int:
        return self.end - self.start + 1

    @property
    def transition_count(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class CalvinPhysicalEvent:
    """One unique labelled transition in a raw source episode.

    Every language segment that can supervise the outgoing transition is kept
    in immutable annotation-index order.  The event deliberately has no
    canonical task or instruction; callers must select one candidate explicitly
    before constructing a host sample.
    """

    episode: CalvinEpisode
    global_index: int
    event_index: int
    previous_event_global_index: int | None
    candidate_segments: tuple[CalvinLanguageSegment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, CalvinEpisode):
            raise ContractError("CALVIN physical event requires a source episode")
        if (
            isinstance(self.global_index, bool | np.bool_)
            or not isinstance(self.global_index, Integral)
            or not self.episode.start <= self.global_index < self.episode.end
        ):
            raise ContractError(
                "CALVIN physical event must identify an action-bearing source frame"
            )
        if (
            isinstance(self.event_index, bool | np.bool_)
            or not isinstance(self.event_index, Integral)
            or self.event_index < 0
        ):
            raise ContractError("CALVIN physical event index must be non-negative")
        if self.previous_event_global_index is None:
            if self.event_index != 0:
                raise ContractError("only the first CALVIN physical event can be left-censored")
        elif (
            isinstance(self.previous_event_global_index, bool | np.bool_)
            or not isinstance(self.previous_event_global_index, Integral)
            or not self.episode.start <= self.previous_event_global_index < self.global_index
            or self.event_index == 0
        ):
            raise ContractError("CALVIN previous physical event identity is invalid")
        if not isinstance(self.candidate_segments, tuple) or not self.candidate_segments:
            raise ContractError("CALVIN physical event requires candidate language segments")
        if any(
            not isinstance(segment, CalvinLanguageSegment) for segment in self.candidate_segments
        ):
            raise ContractError("CALVIN physical event candidates must be language segments")
        candidate_indices = tuple(segment.index for segment in self.candidate_segments)
        if candidate_indices != tuple(sorted(candidate_indices)):
            raise ContractError(
                "CALVIN physical event candidates must be sorted by annotation index"
            )
        if len(set(candidate_indices)) != len(candidate_indices):
            raise ContractError("CALVIN physical event candidates contain duplicates")
        if any(
            segment.episode_index != self.episode.index
            or not segment.start <= self.global_index < segment.end
            for segment in self.candidate_segments
        ):
            raise ContractError(
                "CALVIN physical event candidate does not label its outgoing transition"
            )

    @property
    def event_key(self) -> str:
        return f"calvin-source-episode-{self.episode.index:08d}/frame-{self.global_index:08d}"

    @property
    def frame_index(self) -> int:
        return self.global_index - self.episode.start

    @property
    def reset(self) -> bool:
        """Reset once, at the raw boundary before this event's incoming span."""

        return self.previous_event_global_index is None

    @property
    def at_raw_episode_start(self) -> bool:
        """Report whether the labelled observation itself is the boundary frame."""

        return self.global_index == self.episode.start

    @property
    def reset_global_index(self) -> int | None:
        """Locate the reset at the span origin, never at a language boundary."""

        return self.episode.start if self.reset else None

    @property
    def contract(self) -> str:
        return CALVIN_PHYSICAL_EVENT_CONTRACT

    def select_candidate(self, segment_index: int) -> CalvinLanguageSegment:
        """Resolve one explicitly requested prompt without canonicalization."""

        if isinstance(segment_index, bool | np.bool_) or not isinstance(segment_index, Integral):
            raise ContractError("CALVIN physical candidate index must be an integer")
        matching = tuple(
            segment for segment in self.candidate_segments if segment.index == segment_index
        )
        if len(matching) != 1:
            raise ContractError(
                "selected CALVIN language segment is not an exact physical-event candidate"
            )
        return matching[0]


@dataclass(frozen=True, slots=True)
class CalvinPhysicalEpisodeManifest:
    """Exact labelled-event sweep for one raw source episode."""

    episode: CalvinEpisode
    language_segments: tuple[CalvinLanguageSegment, ...]
    events: tuple[CalvinPhysicalEvent, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.episode, CalvinEpisode):
            raise ContractError("CALVIN physical manifest requires a source episode")
        if not isinstance(self.language_segments, tuple) or any(
            not isinstance(segment, CalvinLanguageSegment) for segment in self.language_segments
        ):
            raise ContractError("CALVIN physical manifest segments must be an immutable tuple")
        segment_indices = tuple(segment.index for segment in self.language_segments)
        if segment_indices != tuple(sorted(segment_indices)):
            raise ContractError(
                "CALVIN physical manifest segments must be sorted by annotation index"
            )
        if len(set(segment_indices)) != len(segment_indices):
            raise ContractError("CALVIN physical manifest segments contain duplicates")
        if any(
            segment.episode_index != self.episode.index
            or not self.episode.start <= segment.start < segment.end <= self.episode.end
            for segment in self.language_segments
        ):
            raise ContractError("CALVIN physical manifest segment crossed its source episode")
        if not isinstance(self.events, tuple) or any(
            not isinstance(event, CalvinPhysicalEvent) for event in self.events
        ):
            raise ContractError("CALVIN physical manifest events must be an immutable tuple")

        expected_by_frame: dict[int, list[CalvinLanguageSegment]] = {}
        for segment in self.language_segments:
            for global_index in range(segment.start, segment.end):
                expected_by_frame.setdefault(global_index, []).append(segment)
        expected_indices = tuple(sorted(expected_by_frame))
        actual_indices = tuple(event.global_index for event in self.events)
        if actual_indices != expected_indices:
            raise ContractError(
                "CALVIN physical events contain duplicates, gaps, or source-order drift"
            )
        for event_index, event in enumerate(self.events):
            expected_candidates = tuple(
                sorted(expected_by_frame[event.global_index], key=lambda segment: segment.index)
            )
            expected_previous = (
                None if event_index == 0 else self.events[event_index - 1].global_index
            )
            if (
                event.episode != self.episode
                or event.event_index != event_index
                or event.previous_event_global_index != expected_previous
                or event.candidate_segments != expected_candidates
            ):
                raise ContractError(
                    "CALVIN physical event omitted or reordered source/candidate identity"
                )

    def event_at(self, global_index: int) -> CalvinPhysicalEvent:
        if isinstance(global_index, bool | np.bool_) or not isinstance(global_index, Integral):
            raise ContractError("CALVIN physical event index must be an integer")
        indices = tuple(event.global_index for event in self.events)
        position = bisect_right(indices, global_index) - 1
        if position < 0 or self.events[position].global_index != global_index:
            raise ContractError(
                f"CALVIN source frame {global_index} has no labelled physical event"
            )
        return self.events[position]

    def previous_event(self, global_index: int) -> CalvinPhysicalEvent | None:
        event = self.event_at(global_index)
        indices = tuple(candidate.global_index for candidate in self.events)
        position = bisect_right(indices, event.global_index) - 1
        return None if position == 0 else self.events[position - 1]


def _raw_action_control_span_sha256(
    *,
    dataset_id: str,
    dataset_revision: str,
    episode: CalvinEpisode,
    start_global_index: int,
    end_global_index: int,
    action_global_indices: tuple[int, ...],
    raw_actions: NDArray[np.float32],
    left_censored_start: bool,
) -> str:
    metadata = {
        "action_dtype": "float32-le",
        "action_global_indices": [int(index) for index in action_global_indices],
        "action_shape": list(raw_actions.shape),
        "dataset_id": dataset_id,
        "dataset_revision": dataset_revision,
        "end_global_index": int(end_global_index),
        "episode": {
            "end": int(episode.end),
            "index": int(episode.index),
            "start": int(episode.start),
        },
        "left_censored_start": left_censored_start,
        "schema": CALVIN_RAW_ACTION_CONTROL_SPAN_CONTRACT,
        "start_global_index": int(start_global_index),
    }
    encoded = json.dumps(
        metadata,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    little_endian_actions = np.ascontiguousarray(raw_actions, dtype=np.dtype("<f4"))
    digest = hashlib.sha256()
    digest.update(encoded)
    digest.update(b"\0")
    digest.update(little_endian_actions.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CalvinRawActionControlSpan:
    """Exact incoming source controls between two labelled physical events.

    The first labelled event in an episode has a left-censored receipt rooted at
    the raw episode start.  Its span is empty only when that event is the actual
    episode-start frame; no synthetic zero command is inserted.
    """

    dataset_id: str
    dataset_revision: str
    episode: CalvinEpisode
    start_global_index: int
    end_global_index: int
    action_global_indices: tuple[int, ...]
    raw_actions: NDArray[np.float32]
    left_censored_start: bool
    sha256: str

    def __post_init__(self) -> None:
        if (
            not isinstance(self.dataset_id, str)
            or not self.dataset_id
            or not isinstance(self.dataset_revision, str)
            or not self.dataset_revision
            or not isinstance(self.episode, CalvinEpisode)
        ):
            raise ContractError("CALVIN raw-action span source identity is incomplete")
        if any(
            isinstance(value, bool | np.bool_) or not isinstance(value, Integral)
            for value in (self.start_global_index, self.end_global_index)
        ):
            raise ContractError("CALVIN raw-action span bounds must be integers")
        if not (
            self.episode.start
            <= self.start_global_index
            <= self.end_global_index
            <= self.episode.end
        ):
            raise ContractError("CALVIN raw-action span crossed its source episode")
        if not isinstance(self.left_censored_start, bool):
            raise ContractError("CALVIN raw-action left-censor flag must be boolean")
        if self.left_censored_start:
            if self.start_global_index != self.episode.start:
                raise ContractError(
                    "CALVIN left-censored control span must start at the raw episode boundary"
                )
        elif self.start_global_index == self.end_global_index:
            raise ContractError("CALVIN non-start control span cannot be empty")
        if not isinstance(self.action_global_indices, tuple) or any(
            isinstance(index, bool | np.bool_) or not isinstance(index, Integral)
            for index in self.action_global_indices
        ):
            raise ContractError("CALVIN raw-action indices must be an immutable integer tuple")
        expected_indices = tuple(range(self.start_global_index, self.end_global_index))
        if self.action_global_indices != expected_indices:
            raise ContractError(
                "CALVIN raw-action indices contain duplicates, gaps, or source-order drift"
            )
        if (
            not isinstance(self.raw_actions, np.ndarray)
            or self.raw_actions.shape != (len(expected_indices), 7)
            or self.raw_actions.dtype != np.float32
            or not np.isfinite(self.raw_actions).all()
        ):
            raise ContractError(
                "CALVIN raw-action span must be finite float32 with shape [controls, 7]"
            )
        immutable_actions = _readonly_array(
            self.raw_actions,
            dtype=np.dtype(np.float32),
            name="raw_actions",
        )
        object.__setattr__(self, "raw_actions", immutable_actions)
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ContractError("CALVIN raw-action span digest must be lowercase SHA-256")
        if self.sha256 != self.recomputed_sha256:
            raise ContractError("CALVIN raw-action span digest disagrees with its ordered controls")

    @classmethod
    def from_raw_actions(
        cls,
        *,
        dataset_id: str,
        dataset_revision: str,
        episode: CalvinEpisode,
        start_global_index: int,
        end_global_index: int,
        action_global_indices: tuple[int, ...],
        raw_actions: NDArray[np.float32],
        left_censored_start: bool,
    ) -> CalvinRawActionControlSpan:
        if not isinstance(raw_actions, np.ndarray) or raw_actions.dtype != np.float32:
            raise ContractError("CALVIN raw-action span builder requires a float32 array")
        digest = _raw_action_control_span_sha256(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            episode=episode,
            start_global_index=start_global_index,
            end_global_index=end_global_index,
            action_global_indices=action_global_indices,
            raw_actions=raw_actions,
            left_censored_start=left_censored_start,
        )
        return cls(
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            episode=episode,
            start_global_index=start_global_index,
            end_global_index=end_global_index,
            action_global_indices=action_global_indices,
            raw_actions=raw_actions,
            left_censored_start=left_censored_start,
            sha256=digest,
        )

    @property
    def contract(self) -> str:
        return CALVIN_RAW_ACTION_CONTROL_SPAN_CONTRACT

    @property
    def recomputed_sha256(self) -> str:
        return _raw_action_control_span_sha256(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            episode=self.episode,
            start_global_index=self.start_global_index,
            end_global_index=self.end_global_index,
            action_global_indices=self.action_global_indices,
            raw_actions=self.raw_actions,
            left_censored_start=self.left_censored_start,
        )


@dataclass(frozen=True, slots=True)
class CalvinTrainingWindow:
    """A contiguous, single-instruction posterior window with valid actions."""

    segment: CalvinLanguageSegment
    records: tuple[RobotTransitionRecord, ...]

    def __post_init__(self) -> None:
        if not self.records:
            raise ContractError("CALVIN training window cannot be empty")
        expected = tuple(
            range(
                self.records[0].global_index,
                self.records[0].global_index + len(self.records),
            )
        )
        actual = tuple(record.global_index for record in self.records)
        if actual != expected:
            raise ContractError("CALVIN training window must be contiguous")
        if any(record.task_index != self.segment.index for record in self.records):
            raise ContractError("CALVIN training window crossed a language segment")
        if any(record.task != self.segment.instruction for record in self.records):
            raise ContractError("CALVIN training window changed instruction text")
        if not all(record.transition_valid for record in self.records):
            raise ContractError("CALVIN posterior windows may not include a segment-final frame")

    @property
    def previous_executed_actions(self) -> tuple[Float32Vector, ...]:
        """Return causal action history aligned to each current observation.

        A window is an independently initialized truncated-BPTT sample. Its
        first posterior is empty, so the first prediction receives an explicit
        zero reset command. Every later observation at index ``t`` receives
        exactly the demonstrator command from record ``t - 1``. The current
        record's action remains exclusively an action target for the host.
        """

        reset = np.zeros_like(self.records[0].action)
        reset.setflags(write=False)
        actions = [reset]
        for record in self.records[:-1]:
            previous = np.asarray(record.action, dtype=np.float32).copy()
            previous.setflags(write=False)
            actions.append(previous)
        return tuple(actions)

    @property
    def picf_evidence_frames(self) -> tuple[CalvinPICFEvidenceFrame, ...]:
        """Return deploy-visible, target-free evidence frames for PICF encoders."""

        return tuple(CalvinPICFEvidenceFrame.from_record(record) for record in self.records)


@dataclass(frozen=True, slots=True)
class CalvinPICFSensorObservation:
    """One deploy-visible sensor array without dataset-row identity."""

    key: str
    value: NDArray
    timestamp_s: float
    units: str

    def __post_init__(self) -> None:
        if not self.key or not self.units:
            raise ContractError("PICF sensor semantics must be explicit")
        if not isinstance(self.value, np.ndarray) or self.value.dtype.hasobject:
            raise ContractError("PICF sensor value must be a non-object NumPy array")
        if not self.value.size or self.value.flags.writeable:
            raise ContractError("PICF sensor value must be nonempty and immutable")
        if np.issubdtype(self.value.dtype, np.number) and not np.isfinite(self.value).all():
            raise ContractError("PICF sensor value contains NaN or infinity")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ContractError("PICF sensor timestamp is invalid")

    @classmethod
    def from_array_observation(
        cls,
        observation: ArrayObservation,
    ) -> CalvinPICFSensorObservation:
        return cls(
            key=observation.key,
            value=observation.value,
            timestamp_s=observation.timestamp_s,
            units=observation.units,
        )


@dataclass(frozen=True, slots=True)
class CalvinPICFEvidenceFrame:
    """The only CALVIN payload visible to native PICF evidence encoders.

    Action targets, language, task keys, masks, simulator state and source
    indices are absent by construction. Time is deploy-visible sensor metadata,
    not a persistent identity.
    """

    sensor_observations: tuple[CalvinPICFSensorObservation, ...]
    timestamp_s: float
    delta_t_s: float

    def __post_init__(self) -> None:
        if not self.sensor_observations:
            raise ContractError("CALVIN PICF evidence frame requires sensor observations")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ContractError("CALVIN PICF evidence timestamp is invalid")
        if not np.isfinite(self.delta_t_s) or self.delta_t_s <= 0.0:
            raise ContractError("CALVIN PICF evidence delta_t is invalid")
        if any(
            abs(observation.timestamp_s - self.timestamp_s) > 1e-7
            for observation in self.sensor_observations
        ):
            raise ContractError("PICF sensor arrays and frame timestamp must be synchronous")

    @classmethod
    def from_record(cls, record: RobotTransitionRecord) -> CalvinPICFEvidenceFrame:
        if record.contract != CALVIN_CONTRACT:
            raise ContractError("PICF evidence frame requires a CALVIN transition record")
        return cls(
            sensor_observations=tuple(
                CalvinPICFSensorObservation.from_array_observation(observation)
                for observation in record.array_observations
            ),
            timestamp_s=record.timestamp_s,
            delta_t_s=record.delta_t_s,
        )


@dataclass(frozen=True, slots=True)
class CalvinMolmoAct2SourceObservation:
    """Task-free native-host observation decoded from one source frame.

    This payload contains exactly the two cameras and robot state consumed by
    the official MolmoAct2 processor. Dataset indices, language, actions and
    structural supervision remain outside the model-facing value.
    """

    images: Mapping[str, NDArray]
    state: Float32Vector
    state_valid: BoolVector
    timestamp_s: float
    delta_t_s: float

    def __post_init__(self) -> None:
        if set(self.images) != set(CALVIN_HOST_IMAGE_KEYS):
            raise ContractError("MolmoAct2 source observation camera keys changed")
        expected_shapes = {
            CALVIN_HOST_IMAGE_KEYS[0]: (200, 200, 3),
            CALVIN_HOST_IMAGE_KEYS[1]: (84, 84, 3),
        }
        for key, expected_shape in expected_shapes.items():
            image = self.images[key]
            if (
                not isinstance(image, np.ndarray)
                or image.shape != expected_shape
                or image.dtype != np.uint8
                or image.flags.writeable
            ):
                raise ContractError(
                    "MolmoAct2 source observation image shape, dtype or mutability changed"
                )
        if (
            not isinstance(self.state, np.ndarray)
            or self.state.shape != (15,)
            or self.state.dtype != np.float32
            or self.state.flags.writeable
            or not np.isfinite(self.state).all()
        ):
            raise ContractError("MolmoAct2 source observation state is invalid")
        if (
            not isinstance(self.state_valid, np.ndarray)
            or self.state_valid.shape != (15,)
            or self.state_valid.dtype != np.bool_
            or self.state_valid.flags.writeable
            or not self.state_valid.all()
        ):
            raise ContractError("MolmoAct2 source observation state validity is invalid")
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ContractError("MolmoAct2 source observation timestamp is invalid")
        if not np.isfinite(self.delta_t_s) or self.delta_t_s <= 0.0:
            raise ContractError("MolmoAct2 source observation delta_t is invalid")


@dataclass(frozen=True, slots=True)
class CalvinMolmoAct2Sample:
    """Source-faithful input for the official MolmoAct2 preprocessor.

    The sample deliberately contains only the two native host cameras. Depth
    and tactile arrays remain available in ``record.array_observations`` for
    external PICF encoders; they are not silently converted into RGB host views.
    """

    observation: Mapping[str, NDArray | str]
    action: NDArray[np.float32]
    action_is_pad: BoolVector
    source_global_index: int
    task_key: str

    def __post_init__(self) -> None:
        expected = {*CALVIN_HOST_IMAGE_KEYS, "observation.state", "task"}
        if set(self.observation) != expected:
            raise ContractError(
                "MolmoAct2 CALVIN observation keys differ from the explicit contract"
            )
        if self.action.ndim != 2 or self.action.shape[1] != 7:
            raise ContractError("MolmoAct2 CALVIN action chunk must be time-by-7")
        if self.action.dtype != np.float32 or self.action.flags.writeable:
            raise ContractError("MolmoAct2 CALVIN action chunk must be immutable float32")
        if (
            self.action_is_pad.dtype != np.bool_
            or self.action_is_pad.shape != self.action.shape[:1]
        ):
            raise ContractError("MolmoAct2 CALVIN action padding must align with the horizon")
        if self.action_is_pad.flags.writeable:
            raise ContractError("MolmoAct2 CALVIN action padding must be immutable")
        if (
            isinstance(self.source_global_index, bool | np.bool_)
            or not isinstance(self.source_global_index, Integral)
            or self.source_global_index < 0
            or not self.task_key
        ):
            raise ContractError("MolmoAct2 CALVIN source identity is incomplete")


@dataclass(frozen=True, slots=True)
class CalvinPhysicalSample:
    """Explicitly language-selected host item for one unique physical event."""

    event: CalvinPhysicalEvent
    selected_segment: CalvinLanguageSegment
    incoming_control_span: CalvinRawActionControlSpan
    record: RobotTransitionRecord
    host_sample: CalvinMolmoAct2Sample

    def __post_init__(self) -> None:
        if not isinstance(self.event, CalvinPhysicalEvent):
            raise ContractError("CALVIN physical sample requires a physical event")
        if not isinstance(self.selected_segment, CalvinLanguageSegment):
            raise ContractError("CALVIN physical sample requires an explicit language selection")
        selected = self.event.select_candidate(self.selected_segment.index)
        if selected != self.selected_segment:
            raise ContractError("CALVIN physical sample language selection metadata drifted")
        if not isinstance(self.incoming_control_span, CalvinRawActionControlSpan):
            raise ContractError("CALVIN physical sample requires an incoming control span")
        control_span = self.incoming_control_span
        if (
            control_span.episode != self.event.episode
            or control_span.end_global_index != self.event.global_index
            or control_span.start_global_index
            != (
                self.event.episode.start
                if self.event.previous_event_global_index is None
                else self.event.previous_event_global_index
            )
            or control_span.left_censored_start != (self.event.previous_event_global_index is None)
        ):
            raise ContractError("CALVIN physical sample control span ends at another event")
        if (
            self.record.contract != CALVIN_CONTRACT
            or not self.record.transition_valid
            or self.record.dataset_id != control_span.dataset_id
            or self.record.dataset_revision != control_span.dataset_revision
            or self.record.episode_index != self.event.episode.index
            or self.record.frame_index != self.event.frame_index
            or self.record.global_index != self.event.global_index
            or self.record.task_index != selected.index
            or self.record.task != selected.instruction
        ):
            raise ContractError("CALVIN physical sample record identity or selection drifted")
        if not isinstance(self.host_sample, CalvinMolmoAct2Sample):
            raise ContractError("CALVIN physical sample requires a MolmoAct2 host sample")
        if not isinstance(self.host_sample.observation, _MAPPING_PROXY_TYPE):
            raise ContractError("CALVIN physical host observation mapping must be immutable")
        if (
            self.host_sample.source_global_index != self.event.global_index
            or self.host_sample.task_key != selected.task_key
            or self.host_sample.observation["task"] != selected.instruction
        ):
            raise ContractError("CALVIN physical host sample disagrees with its selection")

        horizon = self.host_sample.action.shape[0]
        available = min(horizon, selected.end - self.event.global_index)
        expected_padding = np.arange(horizon) >= available
        if not np.array_equal(self.host_sample.action_is_pad, expected_padding):
            raise ContractError(
                "CALVIN physical host action horizon was not clipped to the selected segment"
            )
        if (
            not np.isfinite(self.host_sample.action).all()
            or not np.array_equal(self.host_sample.action[0], self.record.action)
            or np.any(self.host_sample.action[available:] != 0.0)
        ):
            raise ContractError("CALVIN physical host action chunk changed source controls")

    @property
    def reset(self) -> bool:
        return self.event.reset

    @property
    def event_key(self) -> str:
        return self.event.event_key

    @property
    def sample_key(self) -> str:
        """Return the task-independent identity used by a frozen stream plan."""

        return self.event.event_key

    @property
    def episode_key(self) -> str:
        """Return the raw source-episode identity, never a language segment."""

        return f"calvin-source-episode-{self.event.episode.index:08d}"

    @property
    def transition_index(self) -> int:
        """Return the labelled-event position within the raw source episode."""

        return self.event.event_index

    @property
    def picf_evidence_frame(self) -> CalvinPICFEvidenceFrame:
        return CalvinPICFEvidenceFrame.from_record(self.record)


@dataclass(frozen=True, slots=True)
class CalvinSampleLocator:
    segment_index: int
    global_index: int

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool | np.bool_) or not isinstance(value, Integral)
            for value in (self.segment_index, self.global_index)
        ):
            raise ContractError("CALVIN sample locator indices must be integers")
        if self.segment_index < 0 or self.global_index < 0:
            raise ContractError("CALVIN sample locator indices must be non-negative")


@dataclass(frozen=True, slots=True)
class CalvinStatefulEpisodeManifest:
    """One language segment as an immutable posterior-stream episode.

    CALVIN language segments may overlap in source-frame coordinates.  The
    segment identity is therefore part of every key; a global frame number alone
    is not a unique training sample and must never drive posterior continuity.
    """

    segment_index: int
    episode_key: str
    sample_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.segment_index, bool | np.bool_)
            or not isinstance(self.segment_index, Integral)
            or self.segment_index < 0
        ):
            raise ContractError("CALVIN stateful segment index must be non-negative")
        if not isinstance(self.episode_key, str) or not self.episode_key:
            raise ContractError("CALVIN stateful episode key cannot be empty")
        if not self.sample_keys or any(
            not isinstance(sample_key, str) or not sample_key for sample_key in self.sample_keys
        ):
            raise ContractError("CALVIN stateful sample keys cannot be empty")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ContractError("CALVIN stateful sample keys must be unique within a segment")


@dataclass(frozen=True, slots=True)
class CalvinPhysicalStreamEpisodeManifest:
    """One raw source episode represented by unique labelled physical events."""

    source_episode_index: int
    episode_key: str
    sample_keys: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_episode_index, bool | np.bool_)
            or not isinstance(self.source_episode_index, Integral)
            or self.source_episode_index < 0
        ):
            raise ContractError("CALVIN physical stream episode index must be non-negative")
        expected_key = f"calvin-source-episode-{self.source_episode_index:08d}"
        if self.episode_key != expected_key:
            raise ContractError("CALVIN physical stream episode key is not canonical")
        if not self.sample_keys or any(
            not isinstance(sample_key, str) or not sample_key for sample_key in self.sample_keys
        ):
            raise ContractError("CALVIN physical stream sample keys cannot be empty")
        if len(set(self.sample_keys)) != len(self.sample_keys):
            raise ContractError("CALVIN physical stream contains a duplicate physical event")


@dataclass(frozen=True, slots=True)
class CalvinStatefulTransitionSample:
    """One current-frame item for detached-posterior episodic training.

    ``previous_executed_action`` is a deploy-visible transition input.  The
    current demonstrator action remains inside ``record``/``host_sample`` as a
    loss target and is absent from :attr:`picf_evidence_frame`.  A segment start
    explicitly resets both posterior state and action history.
    """

    sample_key: str
    episode_key: str
    transition_index: int
    record: RobotTransitionRecord
    previous_executed_action: Float32Vector
    host_sample: CalvinMolmoAct2Sample

    def __post_init__(self) -> None:
        if not isinstance(self.sample_key, str) or not self.sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        if not isinstance(self.episode_key, str) or not self.episode_key:
            raise ContractError("CALVIN stateful episode key cannot be empty")
        if (
            isinstance(self.transition_index, bool | np.bool_)
            or not isinstance(self.transition_index, Integral)
            or self.transition_index < 0
        ):
            raise ContractError("CALVIN stateful transition index must be non-negative")
        if self.record.contract != CALVIN_CONTRACT or not self.record.transition_valid:
            raise ContractError("CALVIN stateful sample requires one valid outgoing transition")
        previous = self.previous_executed_action
        if (
            not isinstance(previous, np.ndarray)
            or previous.shape != (7,)
            or previous.dtype != np.float32
            or previous.flags.writeable
            or not np.isfinite(previous).all()
        ):
            raise ContractError(
                "CALVIN previous executed action must be immutable finite float32 shape-(7,)"
            )
        if self.transition_index == 0 and np.any(previous != 0.0):
            raise ContractError("CALVIN stateful segment start must reset previous action to zero")
        if self.host_sample.source_global_index != self.record.global_index:
            raise ContractError("CALVIN stateful host sample and transition record disagree")
        if (
            self.host_sample.task_key == ""
            or self.host_sample.observation["task"] != self.record.task
        ):
            raise ContractError("CALVIN stateful host task and transition record disagree")

    @property
    def picf_evidence_frame(self) -> CalvinPICFEvidenceFrame:
        """Return the target-free model-facing sensor payload."""

        return CalvinPICFEvidenceFrame.from_record(self.record)


def collate_calvin_molmoact2(
    samples: Sequence[CalvinMolmoAct2Sample],
) -> dict[str, Any]:
    """Build one official LeRobot ``EnvTransition``-shaped NumPy batch.

    Keeping this collation explicit prevents a single sample's ``[T, 7]``
    action chunk from being interpreted as ``T`` independent batch elements by
    the host processor, which expects ``[B, T, D]`` during training.
    """

    if not samples:
        raise ContractError("CALVIN MolmoAct2 batch cannot be empty")
    horizon = samples[0].action.shape[0]
    if any(sample.action.shape != (horizon, 7) for sample in samples):
        raise ContractError("CALVIN MolmoAct2 batch action shapes must agree")
    observation: dict[str, Any] = {
        key: np.stack([np.asarray(sample.observation[key]) for sample in samples])
        for key in (*CALVIN_HOST_IMAGE_KEYS, "observation.state")
    }
    tasks = [str(sample.observation["task"]) for sample in samples]
    action = np.stack([sample.action for sample in samples]).astype(np.float32, copy=False)
    action_is_pad = np.stack([sample.action_is_pad for sample in samples])
    return {
        "observation": observation,
        "action": action,
        "reward": None,
        "done": None,
        "truncated": None,
        "info": None,
        "complementary_data": {
            "task": tasks,
            "action_is_pad": action_is_pad,
            "source_global_index": [sample.source_global_index for sample in samples],
            "task_key": [sample.task_key for sample in samples],
        },
    }


def decode_calvin_frame(
    frame: Mapping[str, NDArray],
    *,
    source_path: Path,
    dataset_id: str,
    dataset_revision: str,
    episode: CalvinEpisode,
    segment: CalvinLanguageSegment,
    global_index: int,
    control_hz: int = CALVIN_CONTROL_HZ,
    verify_relative_action: bool = True,
) -> RobotTransitionRecord:
    """Decode one official CALVIN frame without exposing privileged state."""

    validate_calvin_source_frame(frame, verify_relative_action=verify_relative_action)
    if (
        not isinstance(source_path, Path)
        or not isinstance(dataset_id, str)
        or not dataset_id
        or not isinstance(dataset_revision, str)
        or not dataset_revision
        or isinstance(control_hz, bool | np.bool_)
        or not isinstance(control_hz, Integral)
        or control_hz <= 0
    ):
        raise ContractError("CALVIN dataset identity and control rate must be explicit")
    if isinstance(global_index, bool | np.bool_) or not isinstance(global_index, Integral):
        raise ContractError("CALVIN global index must be an integer")
    if not episode.start <= global_index <= episode.end:
        raise ContractError("CALVIN frame is outside its declared source episode")
    if not segment.start <= global_index <= segment.end or segment.episode_index != episode.index:
        raise ContractError("CALVIN frame is outside its language segment")

    robot_obs = np.asarray(frame["robot_obs"])
    relative_action = np.asarray(frame["rel_actions"])

    frame_index = global_index - episode.start
    timestamp = frame_index / float(control_hz)
    observations: list[ArrayObservation] = []
    for source_key, contract_key, _shape, dtype, units in CALVIN_OBSERVATION_SPECS:
        value = np.asarray(frame[source_key])
        immutable = _readonly_array(value, dtype=dtype, name=source_key)
        observations.append(
            ArrayObservation(
                key=contract_key,
                value=immutable,
                source_path=f"{source_path}#{source_key}",
                timestamp_s=timestamp,
                units=units,
            )
        )

    state = _readonly_float32_vector(robot_obs, width=15, name="robot_obs")
    action = _readonly_float32_vector(relative_action, width=7, name="rel_actions")
    return RobotTransitionRecord(
        contract=CALVIN_CONTRACT,
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        embodiment="franka-calvin-single-arm/v1",
        control_mode="normalized relative end-effector pose",
        control_frame="CALVIN world Cartesian frame; Euler orientation increments",
        state_axes=CALVIN_STATE_AXES,
        state_units=("m", "m", "m", "rad", "rad", "rad", "m", *("rad" for _ in range(7)), "binary"),
        action_axes=CALVIN_ACTION_AXES,
        action_units=("fraction of 0.02 m",) * 3 + ("fraction of 0.05 rad",) * 3 + ("binary",),
        episode_index=episode.index,
        frame_index=frame_index,
        global_index=global_index,
        task_index=segment.index,
        task=segment.instruction,
        timestamp_s=timestamp,
        delta_t_s=1.0 / float(control_hz),
        transition_valid=global_index < segment.end,
        cameras=(),
        state=state,
        state_valid=_readonly_validity(15),
        action=action,
        action_valid=_readonly_validity(7),
        array_observations=tuple(observations),
    )


class CalvinDatasetIndex:
    """Validated random access over official CALVIN NPZ language segments."""

    def __init__(
        self,
        *,
        split_root: Path,
        dataset_id: str,
        dataset_revision: str,
        control_hz: int,
        episodes: tuple[CalvinEpisode, ...],
        segments: tuple[CalvinLanguageSegment, ...],
        dataset_manifest: DatasetFileManifest | None = None,
    ) -> None:
        if (
            not isinstance(dataset_id, str)
            or not dataset_id
            or not isinstance(dataset_revision, str)
            or not dataset_revision
        ):
            raise ContractError("CALVIN dataset identity must be explicit")
        if (
            isinstance(control_hz, bool | np.bool_)
            or not isinstance(control_hz, Integral)
            or control_hz <= 0
        ):
            raise ContractError("CALVIN control_hz must be a positive integer")
        if dataset_manifest is not None:
            if not isinstance(dataset_manifest, DatasetFileManifest):
                raise TypeError("CALVIN dataset manifest must be a DatasetFileManifest")
            manifest_identity = (
                dataset_manifest.dataset_id,
                dataset_manifest.dataset_revision,
                dataset_manifest.split_name,
            )
            if manifest_identity != (dataset_id, dataset_revision, split_root.name):
                raise ContractError("CALVIN dataset manifest identity differs from index")
        if not isinstance(episodes, tuple) or any(
            not isinstance(episode, CalvinEpisode) for episode in episodes
        ):
            raise ContractError("CALVIN source episodes must be an immutable tuple")
        if tuple(episode.index for episode in episodes) != tuple(range(len(episodes))):
            raise ContractError("CALVIN source episode identity or order drifted")
        if any(
            current.start <= previous.end
            for previous, current in zip(episodes, episodes[1:], strict=False)
        ):
            raise ContractError("CALVIN source episode bounds overlap or are unsorted")
        if not isinstance(segments, tuple) or any(
            not isinstance(segment, CalvinLanguageSegment) for segment in segments
        ):
            raise ContractError("CALVIN language segments must be an immutable tuple")
        if tuple(segment.index for segment in segments) != tuple(range(len(segments))):
            raise ContractError("CALVIN language annotation identity or order drifted")
        segments_by_episode: list[list[CalvinLanguageSegment]] = [[] for _episode in episodes]
        for segment in segments:
            if segment.episode_index >= len(episodes):
                raise ContractError("CALVIN language segment references an absent source episode")
            episode = episodes[segment.episode_index]
            if not episode.start <= segment.start < segment.end <= episode.end:
                raise ContractError("CALVIN language segment crossed its source episode")
            segments_by_episode[episode.index].append(segment)
        self.split_root = split_root
        self.dataset_id = dataset_id
        self.dataset_revision = dataset_revision
        self.control_hz = control_hz
        self.episodes = episodes
        self.segments = segments
        self.dataset_manifest = dataset_manifest
        self._physical_segments_by_episode = tuple(
            tuple(episode_segments) for episode_segments in segments_by_episode
        )
        self._episode_starts = tuple(episode.start for episode in episodes)
        self._action_cache: OrderedDict[int, Float32Vector] = OrderedDict()
        self._action_cache_lock = threading.Lock()

    @classmethod
    def load(
        cls,
        split_root: Path,
        *,
        dataset_id: str = CALVIN_DEBUG_DATASET_ID,
        dataset_revision: str = CALVIN_DEBUG_REVISION,
        verify_files: bool = True,
        dataset_manifest: DatasetFileManifest | None = None,
    ) -> CalvinDatasetIndex:
        if not isinstance(verify_files, bool):
            raise ContractError("verify_files must be boolean")
        if dataset_manifest is None:
            raise ContractError(
                "CALVIN loading requires a content-addressed dataset manifest before "
                "decoding object-array metadata"
            )
        split_root = Path(split_root).resolve()
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        expected_identity = (dataset_id, dataset_revision, split_root.name)
        manifest_identity = (
            dataset_manifest.dataset_id,
            dataset_manifest.dataset_revision,
            dataset_manifest.split_name,
        )
        if manifest_identity != expected_identity:
            raise ContractError("CALVIN dataset manifest identity differs from index request")

        def metadata_source(relative: str, *, maximum_bytes: int) -> io.BytesIO:
            return io.BytesIO(
                read_verified_dataset_file(
                    dataset_manifest,
                    split_root,
                    relative,
                    maximum_bytes=maximum_bytes,
                )
            )

        config_source = metadata_source(
            ".hydra/merged_config.yaml",
            maximum_bytes=1024 * 1024,
        )
        try:
            config_text = config_source.getvalue().decode("utf-8")
        except UnicodeDecodeError as error:
            raise ContractError("CALVIN merged config must be UTF-8 text") from error
        matches = re.findall(r"^\s*control_freq:\s*(\d+)\s*$", config_text, re.MULTILINE)
        if len(matches) != 1:
            raise ContractError("CALVIN merged config must declare exactly one control_freq")
        control_hz = int(matches[0])
        if control_hz != CALVIN_CONTROL_HZ:
            raise ContractError(f"CALVIN control frequency drifted from 30 Hz to {control_hz} Hz")

        raw_bounds = np.load(
            metadata_source("ep_start_end_ids.npy", maximum_bytes=64 * 1024 * 1024),
            allow_pickle=False,
        )
        if raw_bounds.ndim != 2 or raw_bounds.shape[1] != 2:
            raise ContractError("CALVIN episode bounds must have shape [episodes, 2]")
        if raw_bounds.dtype == np.bool_ or not np.issubdtype(raw_bounds.dtype, np.integer):
            raise ContractError("CALVIN episode bounds must use an integer dtype")
        episodes = tuple(
            CalvinEpisode(index=index, start=int(bounds[0]), end=int(bounds[1]))
            for index, bounds in enumerate(raw_bounds)
        )
        if not episodes:
            raise ContractError("CALVIN split contains no source episodes")
        for previous, current in zip(episodes, episodes[1:], strict=False):
            if current.start <= previous.end:
                raise ContractError("CALVIN source episode bounds overlap or are unsorted")
        declared_lengths = np.load(
            metadata_source("ep_lens.npy", maximum_bytes=64 * 1024 * 1024),
            allow_pickle=False,
        )
        if declared_lengths.dtype == np.bool_ or not np.issubdtype(
            declared_lengths.dtype, np.integer
        ):
            raise ContractError("CALVIN episode lengths must use an integer dtype")
        actual_lengths = np.array([episode.length for episode in episodes], dtype=np.int64)
        flat_lengths = np.asarray(declared_lengths).reshape(-1)
        if not np.array_equal(flat_lengths, actual_lengths):
            raise ContractError("CALVIN ep_lens disagrees with inclusive episode bounds")

        annotation_source = metadata_source(
            "lang_annotations/auto_lang_ann.npy",
            maximum_bytes=64 * 1024 * 1024,
        )
        # CALVIN publishes this metadata as a pickled object array. The bytes
        # above are retained from the same descriptor used for manifest hashing.
        raw_annotations = np.load(
            annotation_source,
            allow_pickle=True,
        )
        if raw_annotations.shape != ():
            raise ContractError("CALVIN language annotations must be one scalar mapping")
        annotations = raw_annotations.item()
        if not isinstance(annotations, Mapping):
            raise ContractError("CALVIN language annotations must contain one mapping")
        language = annotations.get("language", {})
        info = annotations.get("info", {})
        if not isinstance(language, Mapping) or not isinstance(info, Mapping):
            raise ContractError("CALVIN language annotation sections must be mappings")
        intervals = info.get("indx", ())
        task_keys = language.get("task", ())
        instructions = language.get("ann", ())
        if not (len(intervals) == len(task_keys) == len(instructions)):
            raise ContractError("CALVIN language annotation fields have inconsistent lengths")

        segments: list[CalvinLanguageSegment] = []
        for index, (interval, task_key, instruction) in enumerate(
            zip(intervals, task_keys, instructions, strict=True)
        ):
            interval_array = np.asarray(interval)
            if (
                interval_array.shape != (2,)
                or interval_array.dtype == np.bool_
                or not np.issubdtype(interval_array.dtype, np.integer)
            ):
                raise ContractError("CALVIN language intervals must be integer pairs")
            if not isinstance(task_key, str) or not isinstance(instruction, str):
                raise ContractError("CALVIN task keys and instructions must be strings")
            start, end = (int(interval_array[0]), int(interval_array[1]))
            containing = [
                episode for episode in episodes if episode.start <= start <= end <= episode.end
            ]
            if len(containing) != 1:
                raise ContractError("CALVIN language segment must lie inside exactly one episode")
            segments.append(
                CalvinLanguageSegment(
                    index=index,
                    start=start,
                    end=end,
                    task_key=task_key,
                    instruction=instruction,
                    episode_index=containing[0].index,
                )
            )

        index = cls(
            split_root=split_root,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            control_hz=control_hz,
            episodes=episodes,
            segments=tuple(segments),
            dataset_manifest=dataset_manifest,
        )
        if verify_files:
            index.verify_source_files()
        return index

    def verify_source_files(self) -> None:
        missing = [
            str(self.frame_path(step))
            for episode in self.episodes
            for step in range(episode.start, episode.end + 1)
            if not self.frame_path(step).is_file()
        ]
        if missing:
            preview = missing[:3]
            raise FileNotFoundError(
                f"CALVIN split is missing {len(missing)} frame files: {preview}"
            )

    def frame_path(self, global_index: int) -> Path:
        if isinstance(global_index, bool | np.bool_) or not isinstance(global_index, Integral):
            raise ContractError("CALVIN global index must be an integer")
        return self.split_root / f"episode_{global_index:07d}.npz"

    def _frame_archive_source(self, global_index: int) -> Path | io.BytesIO:
        """Return a path for legacy fixtures or manifest-verified immutable bytes."""

        path = self.frame_path(global_index)
        if self.dataset_manifest is None:
            if not path.is_file():
                raise FileNotFoundError(path)
            return path
        relative = path.relative_to(self.split_root).as_posix()
        record = self.dataset_manifest.record_for(relative)
        payload = read_verified_dataset_file(
            self.dataset_manifest,
            self.split_root,
            relative,
            maximum_bytes=max(record.size_bytes, 1),
        )
        return io.BytesIO(payload)

    def validated_source_frame_arrays(
        self,
        global_index: int,
        *,
        fields: tuple[str, ...] | None = None,
        verify_relative_action: bool = True,
    ) -> Mapping[str, NDArray]:
        """Read immutable offline arrays from one manifest-verified source frame.

        This privileged path is limited to loss-target generation and dataset
        audits. Deploy-visible model inputs continue to use the narrow adapters
        below, which never expose simulator state.
        """

        self._source_episode(global_index)
        if fields is not None and (
            not isinstance(fields, tuple)
            or any(not isinstance(field, str) or not field for field in fields)
            or len(set(fields)) != len(fields)
        ):
            raise ContractError("CALVIN source-frame fields must be unique nonempty strings")
        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            frame = {name: archive[name] for name in archive.files}
        validate_calvin_source_frame(
            frame,
            verify_relative_action=verify_relative_action,
        )
        selected = tuple(frame) if fields is None else fields
        missing = sorted(set(selected).difference(frame))
        if missing:
            raise ContractError(f"CALVIN frame is missing requested fields: {missing}")
        output: dict[str, NDArray] = {}
        for name in selected:
            value = np.asarray(frame[name]).copy()
            value.setflags(write=False)
            output[name] = value
        return MappingProxyType(output)

    def _episode(self, episode_index: int) -> CalvinEpisode:
        if isinstance(episode_index, bool | np.bool_) or not isinstance(episode_index, Integral):
            raise ContractError("CALVIN episode index must be an integer")
        if episode_index < 0 or episode_index >= len(self.episodes):
            raise ContractError(f"unknown CALVIN episode index {episode_index}")
        return self.episodes[episode_index]

    def _source_episode(self, global_index: int) -> CalvinEpisode:
        if isinstance(global_index, bool | np.bool_) or not isinstance(global_index, Integral):
            raise ContractError("CALVIN global index must be an integer")
        position = bisect_right(self._episode_starts, global_index) - 1
        if position < 0 or global_index > self.episodes[position].end:
            raise ContractError(f"CALVIN source frame {global_index} is outside every episode")
        return self.episodes[position]

    def source_episode(self, global_index: int) -> CalvinEpisode:
        """Return the immutable source-episode boundary containing one frame."""

        return self._source_episode(global_index)

    def _segment(self, segment_index: int) -> CalvinLanguageSegment:
        if isinstance(segment_index, bool | np.bool_) or not isinstance(segment_index, Integral):
            raise ContractError("CALVIN segment index must be an integer")
        if segment_index < 0 or segment_index >= len(self.segments):
            raise ContractError(f"unknown CALVIN language segment {segment_index}")
        return self.segments[segment_index]

    def physical_episode_manifest(
        self,
        episode_index: int,
    ) -> CalvinPhysicalEpisodeManifest:
        """Sweep exact unique labelled events for one raw source episode."""

        episode = self._episode(episode_index)
        language_segments = self._physical_segments_by_episode[episode.index]

        starts: dict[int, list[CalvinLanguageSegment]] = {}
        ends: dict[int, list[CalvinLanguageSegment]] = {}
        for segment in language_segments:
            starts.setdefault(segment.start, []).append(segment)
            ends.setdefault(segment.end, []).append(segment)
        active: dict[int, CalvinLanguageSegment] = {}
        events: list[CalvinPhysicalEvent] = []
        if language_segments:
            first = min(segment.start for segment in language_segments)
            stop = max(segment.end for segment in language_segments)
            for global_index in range(first, stop):
                for segment in ends.get(global_index, ()):
                    if active.pop(segment.index, None) != segment:
                        raise ContractError(
                            "CALVIN physical-event sweep lost language-segment order"
                        )
                for segment in starts.get(global_index, ()):
                    if segment.index in active:
                        raise ContractError(
                            "CALVIN physical-event sweep contains a duplicate segment"
                        )
                    active[segment.index] = segment
                if active:
                    events.append(
                        CalvinPhysicalEvent(
                            episode=episode,
                            global_index=global_index,
                            event_index=len(events),
                            previous_event_global_index=(
                                None if not events else events[-1].global_index
                            ),
                            candidate_segments=tuple(active[index] for index in sorted(active)),
                        )
                    )
            for segment in ends.get(stop, ()):
                if active.pop(segment.index, None) != segment:
                    raise ContractError("CALVIN physical-event sweep ended out of order")
            if active:
                raise ContractError("CALVIN physical-event sweep omitted terminal boundaries")
        return CalvinPhysicalEpisodeManifest(
            episode=episode,
            language_segments=language_segments,
            events=tuple(events),
        )

    def iter_physical_events(self) -> Iterator[CalvinPhysicalEvent]:
        """Yield each labelled source transition once in episode/frame order."""

        for episode in self.episodes:
            yield from self.physical_episode_manifest(episode.index).events

    def physical_event(self, global_index: int) -> CalvinPhysicalEvent:
        episode = self._source_episode(global_index)
        language_segments = self._physical_segments_by_episode[episode.index]
        candidates = tuple(
            segment for segment in language_segments if segment.start <= global_index < segment.end
        )
        if not candidates:
            raise ContractError(
                f"CALVIN source frame {global_index} has no labelled physical event"
            )

        merged_intervals: list[tuple[int, int]] = []
        for start, end in sorted(
            ((segment.start, segment.end) for segment in language_segments),
            key=lambda interval: (interval[0], interval[1]),
        ):
            if merged_intervals and start <= merged_intervals[-1][1]:
                previous_start, previous_end = merged_intervals[-1]
                merged_intervals[-1] = (previous_start, max(previous_end, end))
            else:
                merged_intervals.append((start, end))
        event_index = 0
        previous_event_global_index: int | None = None
        found = False
        for start, end in merged_intervals:
            if global_index >= end:
                event_index += end - start
                previous_event_global_index = end - 1
                continue
            if start <= global_index < end:
                event_index += global_index - start
                if global_index > start:
                    previous_event_global_index = global_index - 1
                found = True
                break
        if not found:
            raise RuntimeError("CALVIN physical event disappeared from its interval union")
        return CalvinPhysicalEvent(
            episode=episode,
            global_index=global_index,
            event_index=event_index,
            previous_event_global_index=previous_event_global_index,
            candidate_segments=candidates,
        )

    def physical_control_span(self, global_index: int) -> CalvinRawActionControlSpan:
        """Read the exact incoming raw controls since the prior labelled event."""

        return self._physical_control_span(self.physical_event(global_index))

    def _physical_control_span(
        self,
        event: CalvinPhysicalEvent,
    ) -> CalvinRawActionControlSpan:
        episode = event.episode
        previous_global_index = event.previous_event_global_index
        left_censored_start = previous_global_index is None
        start_global_index = (
            episode.start if previous_global_index is None else previous_global_index
        )
        action_global_indices = tuple(range(start_global_index, event.global_index))
        if action_global_indices:
            raw_actions = np.stack(
                [self.action(index) for index in action_global_indices],
                axis=0,
            ).astype(np.float32, copy=False)
        else:
            raw_actions = np.empty((0, 7), dtype=np.float32)
        return CalvinRawActionControlSpan.from_raw_actions(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            episode=episode,
            start_global_index=start_global_index,
            end_global_index=event.global_index,
            action_global_indices=action_global_indices,
            raw_actions=raw_actions,
            left_censored_start=left_censored_start,
        )

    def physical_sample(
        self,
        global_index: int,
        *,
        selected_segment_index: int,
        action_horizon: int,
    ) -> CalvinPhysicalSample:
        """Materialize one host sample after explicit candidate selection."""

        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be positive")
        event = self.physical_event(global_index)
        selected_segment = event.select_candidate(selected_segment_index)
        if self._segment(selected_segment.index) != selected_segment:
            raise ContractError("CALVIN physical selection differs from source annotation")
        record = self.record(selected_segment.index, event.global_index)
        mutable_host_sample = self._molmoact2_sample_from_record(
            selected_segment,
            record,
            action_horizon=action_horizon,
        )
        host_sample = CalvinMolmoAct2Sample(
            observation=MappingProxyType(dict(mutable_host_sample.observation)),
            action=mutable_host_sample.action,
            action_is_pad=mutable_host_sample.action_is_pad,
            source_global_index=mutable_host_sample.source_global_index,
            task_key=mutable_host_sample.task_key,
        )
        return CalvinPhysicalSample(
            event=event,
            selected_segment=selected_segment,
            incoming_control_span=self._physical_control_span(event),
            record=record,
            host_sample=host_sample,
        )

    def record(self, segment_index: int, global_index: int) -> RobotTransitionRecord:
        segment = self._segment(segment_index)
        episode = self._episode(segment.episode_index)
        path = self.frame_path(global_index)
        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            frame = {name: archive[name] for name in archive.files}
        return decode_calvin_frame(
            frame,
            source_path=path,
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            episode=episode,
            segment=segment,
            global_index=global_index,
            control_hz=self.control_hz,
        )

    def _load_action(self, global_index: int) -> Float32Vector:
        """Read one command without decompressing unrelated sensor arrays."""

        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            action_frame = {
                name: archive[name]
                for name in ("robot_obs", "actions", "rel_actions")
                if name in archive
            }
        return _validated_relative_action(action_frame, verify_relative_action=True)

    def state_and_action(self, global_index: int) -> tuple[Float32Vector, Float32Vector]:
        """Read only the normalized host state and verified outgoing command.

        Offline normalization-stat generation uses this narrow path so it does
        not decompress RGB, depth or tactile arrays for every training sample.
        """

        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            fields = {
                name: archive[name]
                for name in ("robot_obs", "actions", "rel_actions")
                if name in archive
            }
        action = _validated_relative_action(fields, verify_relative_action=True)
        state = _readonly_float32_vector(fields["robot_obs"], width=15, name="robot_obs")
        return state, action

    def molmoact2_source_observation(
        self,
        global_index: int,
    ) -> CalvinMolmoAct2SourceObservation:
        """Read one task-independent source frame for native visual encoding."""

        episode = self._source_episode(global_index)
        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            required = ("robot_obs", "rgb_static", "rgb_gripper")
            if any(name not in archive.files for name in required):
                raise ContractError("CALVIN source frame omitted a MolmoAct2 observation field")
            state = _readonly_float32_vector(
                archive["robot_obs"],
                width=15,
                name="robot_obs",
            )
            images = {
                CALVIN_HOST_IMAGE_KEYS[0]: _readonly_array(
                    archive["rgb_static"],
                    dtype=np.dtype(np.uint8),
                    name="rgb_static",
                ),
                CALVIN_HOST_IMAGE_KEYS[1]: _readonly_array(
                    archive["rgb_gripper"],
                    dtype=np.dtype(np.uint8),
                    name="rgb_gripper",
                ),
            }
        timestamp_s = (global_index - episode.start) / float(self.control_hz)
        return CalvinMolmoAct2SourceObservation(
            images=images,
            state=state,
            state_valid=_readonly_validity(15),
            timestamp_s=timestamp_s,
            delta_t_s=1.0 / float(self.control_hz),
        )

    def source_robot_state(self, global_index: int) -> Float32Vector:
        """Read only the deploy-visible proprioceptive state for one source frame."""

        self._source_episode(global_index)
        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            if "robot_obs" not in archive.files:
                raise ContractError("CALVIN source frame omitted robot_obs")
            return _readonly_float32_vector(
                archive["robot_obs"],
                width=15,
                name="robot_obs",
            )

    def picf_evidence_frame(
        self,
        segment_index: int,
        global_index: int,
    ) -> CalvinPICFEvidenceFrame:
        """Read only deploy-visible PICF sensors for one valid transition.

        This path deliberately does not decode robot state, actions, language,
        masks or simulator state. The segment index only enforces the
        posterior reset boundary and is absent from the model-facing value.
        """

        segment = self._segment(segment_index)
        if isinstance(global_index, bool | np.bool_) or not isinstance(global_index, Integral):
            raise ContractError("CALVIN global index must be an integer")
        if not segment.start <= global_index < segment.end:
            raise ContractError("PICF evidence requires a valid CALVIN language transition")
        if self._source_episode(global_index) != self._episode(segment.episode_index):
            raise ContractError("PICF evidence segment and source episode differ")
        return self.source_picf_evidence_frame(global_index)

    def source_picf_evidence_frame(
        self,
        global_index: int,
    ) -> CalvinPICFEvidenceFrame:
        """Read deploy-visible PICF sensors on the raw physical time axis.

        This is the cache/online-inference boundary for task-independent PICF.
        It decodes no language, action, simulator state, mask or source identity
        into the returned value and therefore remains valid between annotation
        boundaries inside one raw episode.
        """

        episode = self._source_episode(global_index)
        with np.load(self._frame_archive_source(global_index), allow_pickle=False) as archive:
            required = tuple(spec[0] for spec in CALVIN_OBSERVATION_SPECS)
            if any(name not in archive.files for name in required):
                raise ContractError("CALVIN source frame omitted a PICF sensor field")
            timestamp_s = (global_index - episode.start) / float(self.control_hz)
            observations = []
            for source_key, contract_key, shape, dtype, units in CALVIN_OBSERVATION_SPECS:
                value = np.asarray(archive[source_key])
                if value.shape != shape or value.dtype != dtype:
                    raise ContractError(
                        f"CALVIN {source_key} must be {shape} {dtype}, "
                        f"got {value.shape} {value.dtype}"
                    )
                observations.append(
                    CalvinPICFSensorObservation(
                        key=contract_key,
                        value=_readonly_array(value, dtype=dtype, name=source_key),
                        timestamp_s=timestamp_s,
                        units=units,
                    )
                )
        return CalvinPICFEvidenceFrame(
            sensor_observations=tuple(observations),
            timestamp_s=timestamp_s,
            delta_t_s=1.0 / float(self.control_hz),
        )

    def source_picf_evidence_prefix(
        self,
        global_index: int,
        *,
        maximum_source_frames: int,
    ) -> tuple[CalvinPICFEvidenceFrame, ...]:
        """Return the contiguous raw-episode sensor prefix ending at one frame."""

        if (
            not isinstance(maximum_source_frames, int)
            or isinstance(maximum_source_frames, bool)
            or maximum_source_frames <= 0
        ):
            raise ContractError("maximum_source_frames must be a positive integer")
        episode = self._source_episode(global_index)
        first = max(episode.start, global_index - maximum_source_frames + 1)
        return tuple(
            self.source_picf_evidence_frame(source_index)
            for source_index in range(first, global_index + 1)
        )

    def action(self, global_index: int) -> Float32Vector:
        """Return an immutable copy of one verified outgoing command.

        Action horizons overlap heavily in a stateful stream.  The bounded cache
        keeps those seven-dimensional commands hot without retaining images,
        depth or tactile observations in host memory.
        """

        self.frame_path(global_index)
        with self._action_cache_lock:
            cached = self._action_cache.get(global_index)
            if cached is not None:
                self._action_cache.move_to_end(global_index)
        if cached is None:
            loaded = self._load_action(global_index)
            with self._action_cache_lock:
                cached = self._action_cache.setdefault(global_index, loaded)
                self._action_cache.move_to_end(global_index)
                while len(self._action_cache) > CALVIN_ACTION_CACHE_CAPACITY:
                    self._action_cache.popitem(last=False)
        return _readonly_float32_vector(cached, width=7, name="rel_actions")

    def clear_action_cache(self) -> None:
        with self._action_cache_lock:
            self._action_cache.clear()

    def iter_segment(self, segment_index: int) -> Iterator[RobotTransitionRecord]:
        segment = self._segment(segment_index)
        for global_index in range(segment.start, segment.end + 1):
            yield self.record(segment_index, global_index)

    def iter_windows(
        self,
        sequence_length: int,
        *,
        stride: int = 1,
        segment_indices: Sequence[int] | None = None,
    ) -> Iterator[CalvinTrainingWindow]:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in (sequence_length, stride)
        ):
            raise ValueError("sequence_length and stride must be positive")
        indices = range(len(self.segments)) if segment_indices is None else segment_indices
        for segment_index in indices:
            segment = self._segment(segment_index)
            if segment.transition_count < sequence_length:
                continue
            stop = segment.end - sequence_length + 1
            for start in range(segment.start, stop, stride):
                records = tuple(
                    self.record(segment.index, step)
                    for step in range(start, start + sequence_length)
                )
                yield CalvinTrainingWindow(segment=segment, records=records)

    def molmoact2_sample(
        self,
        segment_index: int,
        global_index: int,
        *,
        action_horizon: int,
    ) -> CalvinMolmoAct2Sample:
        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be positive")
        segment = self._segment(segment_index)
        if not segment.start <= global_index < segment.end:
            raise ContractError("MolmoAct2 sample requires a valid CALVIN transition")
        record = self.record(segment_index, global_index)
        return self._molmoact2_sample_from_record(
            segment,
            record,
            action_horizon=action_horizon,
        )

    def _molmoact2_sample_from_record(
        self,
        segment: CalvinLanguageSegment,
        record: RobotTransitionRecord,
        *,
        action_horizon: int,
    ) -> CalvinMolmoAct2Sample:
        if (
            record.contract != CALVIN_CONTRACT
            or record.task_index != segment.index
            or not segment.start <= record.global_index < segment.end
            or not record.transition_valid
        ):
            raise ContractError("MolmoAct2 host record does not belong to the requested segment")
        arrays = {observation.key: observation.value for observation in record.array_observations}
        actions = np.zeros((action_horizon, 7), dtype=np.float32)
        padded = np.ones(action_horizon, dtype=np.bool_)
        available = min(action_horizon, segment.end - record.global_index)
        for offset in range(available):
            actions[offset] = (
                record.action if offset == 0 else self.action(record.global_index + offset)
            )
            padded[offset] = False
        actions = _readonly_array(
            actions,
            dtype=np.dtype(np.float32),
            name="MolmoAct2 actions",
        )
        padded = _readonly_array(
            padded,
            dtype=np.dtype(np.bool_),
            name="MolmoAct2 action padding",
        )
        observation: dict[str, NDArray | str] = {
            CALVIN_HOST_IMAGE_KEYS[0]: arrays["observation.images.rgb_static"],
            CALVIN_HOST_IMAGE_KEYS[1]: arrays["observation.images.rgb_gripper"],
            "observation.state": record.state,
            "task": record.task,
        }
        return CalvinMolmoAct2Sample(
            observation=observation,
            action=actions,
            action_is_pad=padded,
            source_global_index=record.global_index,
            task_key=segment.task_key,
        )

    def stateful_transition_sample(
        self,
        segment_index: int,
        global_index: int,
        *,
        action_horizon: int,
    ) -> CalvinStatefulTransitionSample:
        """Decode one exact stream transition with its causal predecessor action."""

        segment = self._segment(segment_index)
        if not segment.start <= global_index < segment.end:
            raise ContractError("stateful sample requires a valid CALVIN language transition")
        transition_index = global_index - segment.start
        record = self.record(segment_index, global_index)
        if transition_index == 0:
            previous_action = np.zeros(7, dtype=np.float32)
        else:
            previous_action = self.action(global_index - 1)
        previous_action.setflags(write=False)
        episode_key = _calvin_stateful_episode_key(segment.index)
        return CalvinStatefulTransitionSample(
            sample_key=_calvin_stateful_sample_key(segment, global_index),
            episode_key=episode_key,
            transition_index=transition_index,
            record=record,
            previous_executed_action=previous_action,
            host_sample=self._molmoact2_sample_from_record(
                segment,
                record,
                action_horizon=action_horizon,
            ),
        )


class CalvinMolmoAct2Dataset:
    """Random-access language-conditioned CALVIN samples for a host DataLoader."""

    def __init__(self, index: CalvinDatasetIndex, *, action_horizon: int) -> None:
        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be positive")
        self.index = index
        self.action_horizon = action_horizon
        self.locators = tuple(
            CalvinSampleLocator(segment.index, step)
            for segment in index.segments
            for step in range(segment.start, segment.end)
        )

    def __len__(self) -> int:
        return len(self.locators)

    def __getitem__(self, item: int) -> CalvinMolmoAct2Sample:
        locator = self.locators[item]
        return self.index.molmoact2_sample(
            locator.segment_index,
            locator.global_index,
            action_horizon=self.action_horizon,
        )


class CalvinPosteriorWindowDataset:
    """Random-access windows for explicit PICF truncated posterior training."""

    def __init__(self, index: CalvinDatasetIndex, *, sequence_length: int) -> None:
        if (
            not isinstance(sequence_length, int)
            or isinstance(sequence_length, bool)
            or sequence_length <= 0
        ):
            raise ValueError("sequence_length must be positive")
        self.index = index
        self.sequence_length = sequence_length
        self.locators = tuple(
            CalvinSampleLocator(segment.index, start)
            for segment in index.segments
            for start in range(segment.start, segment.end - sequence_length + 1)
        )

    def __len__(self) -> int:
        return len(self.locators)

    def __getitem__(self, item: int) -> CalvinTrainingWindow:
        locator = self.locators[item]
        segment = self.index.segments[locator.segment_index]
        records = tuple(
            self.index.record(locator.segment_index, step)
            for step in range(locator.global_index, locator.global_index + self.sequence_length)
        )
        return CalvinTrainingWindow(segment=segment, records=records)


def _calvin_stateful_episode_key(segment_index: int) -> str:
    return f"calvin-language-segment-{segment_index:08d}"


def _calvin_stateful_sample_key(
    segment: CalvinLanguageSegment,
    global_index: int,
) -> str:
    transition_index = global_index - segment.start
    return (
        f"{_calvin_stateful_episode_key(segment.index)}/"
        f"transition-{transition_index:08d}-frame-{global_index:08d}"
    )


class CalvinStatefulTransitionDataset:
    """Random access and key lookup for the production one-transition stream."""

    def __init__(self, index: CalvinDatasetIndex, *, action_horizon: int) -> None:
        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be positive")
        self.index = index
        self.action_horizon = action_horizon
        self.episode_manifest = tuple(
            CalvinStatefulEpisodeManifest(
                segment_index=segment.index,
                episode_key=_calvin_stateful_episode_key(segment.index),
                sample_keys=tuple(
                    _calvin_stateful_sample_key(segment, global_index)
                    for global_index in range(segment.start, segment.end)
                ),
            )
            for segment in index.segments
        )
        self.locators = tuple(
            CalvinSampleLocator(segment.index, global_index)
            for segment in index.segments
            for global_index in range(segment.start, segment.end)
        )
        flattened_keys = tuple(
            sample_key for episode in self.episode_manifest for sample_key in episode.sample_keys
        )
        if len(flattened_keys) != len(self.locators) or len(set(flattened_keys)) != len(
            flattened_keys
        ):
            raise ContractError("CALVIN stateful manifest is not one-to-one with transitions")
        self._locator_by_key = dict(zip(flattened_keys, self.locators, strict=True))
        self._index_by_key = {sample_key: index for index, sample_key in enumerate(flattened_keys)}
        self._episode_position_by_key = {
            sample_key: (episode, transition_index)
            for episode in self.episode_manifest
            for transition_index, sample_key in enumerate(episode.sample_keys)
        }

    @property
    def sample_keys(self) -> tuple[str, ...]:
        return tuple(self._locator_by_key)

    def __len__(self) -> int:
        return len(self.locators)

    def _materialize(self, locator: CalvinSampleLocator) -> CalvinStatefulTransitionSample:
        return self.index.stateful_transition_sample(
            locator.segment_index,
            locator.global_index,
            action_horizon=self.action_horizon,
        )

    def __getitem__(self, item: int) -> CalvinStatefulTransitionSample:
        return self._materialize(self.locators[item])

    def by_key(self, sample_key: str) -> CalvinStatefulTransitionSample:
        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        try:
            locator = self._locator_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc
        sample = self._materialize(locator)
        if sample.sample_key != sample_key:
            raise RuntimeError("CALVIN stateful key lookup changed after manifest construction")
        return sample

    def locator_by_key(self, sample_key: str) -> CalvinSampleLocator:
        """Resolve immutable source coordinates without decoding observations."""

        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        try:
            return self._locator_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc

    def task_key_by_key(self, sample_key: str) -> str:
        """Resolve frozen language-task identity without decoding observations."""

        locator = self.locator_by_key(sample_key)
        try:
            segment = self.index.segments[locator.segment_index]
        except IndexError as exc:
            raise RuntimeError("CALVIN locator references an absent language segment") from exc
        if segment.index != locator.segment_index:
            raise RuntimeError("CALVIN language segment order differs from its frozen identity")
        return segment.task_key

    def source_global_index_by_key(self, sample_key: str) -> int:
        """Resolve immutable source identity without decoding a frame payload."""

        return self.locator_by_key(sample_key).global_index

    def available_future_transitions_by_key(self, sample_key: str) -> int:
        """Return contiguous future action-bearing samples in the same segment."""

        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        try:
            episode, transition_index = self._episode_position_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc
        return len(episode.sample_keys) - transition_index - 1

    def future_sample_keys(self, sample_key: str, *, count: int) -> tuple[str, ...]:
        """Resolve exactly ``count`` later transitions without crossing a reset."""

        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContractError("CALVIN future transition count must be non-negative")
        try:
            episode, transition_index = self._episode_position_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc
        stop = transition_index + count + 1
        if stop > len(episode.sample_keys):
            raise ContractError("CALVIN future transition request crosses a language reset")
        return episode.sample_keys[transition_index + 1 : stop]

    def history_sample_keys(self, sample_key: str) -> tuple[str, ...]:
        """Return the exact same-segment prefix strictly before ``sample_key``."""

        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        try:
            episode, transition_index = self._episode_position_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc
        return episode.sample_keys[:transition_index]

    def evidence_prefix_by_key(
        self,
        sample_key: str,
        *,
        maximum_source_frames: int,
    ) -> tuple[CalvinPICFEvidenceFrame, ...]:
        """Resolve a target-free causal sensor prefix ending at ``sample_key``.

        Prefixes never cross a language-segment reset, even when two CALVIN
        annotations overlap in source coordinates. No frame is padded or
        repeated to satisfy ``maximum_source_frames``.
        """

        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        if (
            not isinstance(maximum_source_frames, int)
            or isinstance(maximum_source_frames, bool)
            or maximum_source_frames <= 0
        ):
            raise ContractError("maximum_source_frames must be a positive integer")
        try:
            locator = self._locator_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc
        segment = self.index._segment(locator.segment_index)
        first = max(segment.start, locator.global_index - maximum_source_frames + 1)
        return tuple(
            self.index.picf_evidence_frame(locator.segment_index, global_index)
            for global_index in range(first, locator.global_index + 1)
        )

    def index_for_key(self, sample_key: str) -> int:
        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN stateful sample key cannot be empty")
        try:
            return self._index_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN stateful sample key {sample_key!r}") from exc


class CalvinPhysicalTransitionDataset:
    """Unique labelled physical events on each raw CALVIN episode time axis.

    A key identifies only a source event. Language remains an explicit overlay:
    :meth:`by_key` refuses to materialize an item until the caller names one of
    the event's exact candidate annotations.
    """

    def __init__(self, index: CalvinDatasetIndex, *, action_horizon: int) -> None:
        if (
            not isinstance(action_horizon, int)
            or isinstance(action_horizon, bool)
            or action_horizon <= 0
        ):
            raise ValueError("action_horizon must be positive")
        self.index = index
        self.action_horizon = action_horizon

        manifests: list[CalvinPhysicalStreamEpisodeManifest] = []
        events: list[CalvinPhysicalEvent] = []
        for episode in index.episodes:
            physical = index.physical_episode_manifest(episode.index)
            if not physical.events:
                continue
            episode_key = f"calvin-source-episode-{episode.index:08d}"
            manifests.append(
                CalvinPhysicalStreamEpisodeManifest(
                    source_episode_index=episode.index,
                    episode_key=episode_key,
                    sample_keys=tuple(event.event_key for event in physical.events),
                )
            )
            events.extend(physical.events)
        if not manifests:
            raise ContractError("CALVIN physical stream contains no labelled source events")
        keys = tuple(event.event_key for event in events)
        if len(keys) != len(set(keys)):
            raise ContractError("CALVIN physical event keys are not globally unique")
        self.episode_manifest = tuple(manifests)
        self._events = tuple(events)
        self._event_by_key = dict(zip(keys, self._events, strict=True))
        self._index_by_key = {sample_key: position for position, sample_key in enumerate(keys)}
        self._episode_position_by_key = {
            sample_key: (manifest, event_index)
            for manifest in self.episode_manifest
            for event_index, sample_key in enumerate(manifest.sample_keys)
        }

    @property
    def sample_keys(self) -> tuple[str, ...]:
        return tuple(self._event_by_key)

    def __len__(self) -> int:
        return len(self._events)

    def event_by_key(self, sample_key: str) -> CalvinPhysicalEvent:
        if not isinstance(sample_key, str) or not sample_key:
            raise ContractError("CALVIN physical sample key cannot be empty")
        try:
            return self._event_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN physical sample key {sample_key!r}") from exc

    def by_key(
        self,
        sample_key: str,
        *,
        selected_segment_index: int,
    ) -> CalvinPhysicalSample:
        event = self.event_by_key(sample_key)
        sample = self.index.physical_sample(
            event.global_index,
            selected_segment_index=selected_segment_index,
            action_horizon=self.action_horizon,
        )
        if sample.sample_key != sample_key:
            raise RuntimeError("CALVIN physical key lookup changed after manifest construction")
        return sample

    def candidate_segment_indices_by_key(self, sample_key: str) -> tuple[int, ...]:
        return tuple(segment.index for segment in self.event_by_key(sample_key).candidate_segments)

    def source_global_index_by_key(self, sample_key: str) -> int:
        return self.event_by_key(sample_key).global_index

    def timestamp_s_by_key(self, sample_key: str) -> float:
        """Return the physical observation clock without decoding its sensor archive."""

        event = self.event_by_key(sample_key)
        return (event.global_index - event.episode.start) / float(self.index.control_hz)

    def source_episode_index_by_key(self, sample_key: str) -> int:
        return self.event_by_key(sample_key).episode.index

    def evidence_prefix_by_key(
        self,
        sample_key: str,
        *,
        maximum_source_frames: int,
    ) -> tuple[CalvinPICFEvidenceFrame, ...]:
        """Resolve target-free sensors on the same raw-episode axis as the posterior."""

        event = self.event_by_key(sample_key)
        return self.index.source_picf_evidence_prefix(
            event.global_index,
            maximum_source_frames=maximum_source_frames,
        )

    def available_future_transitions_by_key(self, sample_key: str) -> int:
        try:
            episode, transition_index = self._episode_position_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN physical sample key {sample_key!r}") from exc
        return len(episode.sample_keys) - transition_index - 1

    def future_sample_keys(self, sample_key: str, *, count: int) -> tuple[str, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContractError("CALVIN future physical-event count must be non-negative")
        try:
            episode, transition_index = self._episode_position_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN physical sample key {sample_key!r}") from exc
        stop = transition_index + count + 1
        if stop > len(episode.sample_keys):
            raise ContractError("CALVIN future physical-event request crosses a raw episode reset")
        return episode.sample_keys[transition_index + 1 : stop]

    def future_source_global_indices_by_key(
        self,
        sample_key: str,
        *,
        count: int,
    ) -> tuple[int, ...]:
        """Return consecutive raw-frame indices after one labelled event.

        Physical-event keys enumerate the union of language-labelled transitions;
        adjacent event keys are therefore not necessarily adjacent camera frames.
        Source video objectives need the raw episode clock instead.
        """

        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ContractError("CALVIN future source-frame count must be non-negative")
        event = self.event_by_key(sample_key)
        stop = event.global_index + count
        if stop > event.episode.end:
            raise ContractError("CALVIN future source-frame request crosses a raw episode reset")
        return tuple(range(event.global_index + 1, stop + 1))

    def index_for_key(self, sample_key: str) -> int:
        try:
            return self._index_by_key[sample_key]
        except KeyError as exc:
            raise KeyError(f"unknown CALVIN physical sample key {sample_key!r}") from exc

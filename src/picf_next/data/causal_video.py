"""Causal, duplicate-free video clips for read-only motion context.

This module is intentionally encoder agnostic and torch free. Source identity
is a content/timestamp hash used only for cache and intervention auditing; it is
never emitted as model evidence.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from picf_next.contracts import ContractError
from picf_next.data.calvin import CalvinPICFEvidenceFrame


def _strict_positive_int(name: str, value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def causal_video_source_sha256(
    image: NDArray[np.uint8],
    *,
    timestamp_s: float,
    sensor_key: str,
) -> str:
    """Hash one deploy-visible image observation without dataset row IDs."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ContractError("causal video images must be H-by-W-by-3 uint8")
    if not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ContractError("causal video timestamps must be finite and nonnegative")
    if not isinstance(sensor_key, str) or not sensor_key:
        raise ContractError("causal video sensor key must be nonempty")
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(b"picf-next.causal-video-source.v1\0")
    digest.update(sensor_key.encode("utf-8"))
    digest.update(b"\0")
    digest.update(float(timestamp_s).hex().encode("ascii"))
    digest.update(b"\0")
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(b"\0uint8\0")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CausalVideoSourceFrame:
    """One immutable source observation before temporal sampling."""

    image: NDArray[np.uint8]
    timestamp_s: float
    sensor_key: str
    source_sha256: str

    def __post_init__(self) -> None:
        image = np.asarray(self.image)
        if (
            image.ndim != 3
            or image.shape[2] != 3
            or image.dtype != np.uint8
            or image.flags.writeable
        ):
            raise ContractError("causal source image must be immutable H-by-W-by-3 uint8")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ContractError("causal source timestamp is invalid")
        if not isinstance(self.sensor_key, str) or not self.sensor_key:
            raise ContractError("causal source sensor key is invalid")
        expected = causal_video_source_sha256(
            image,
            timestamp_s=self.timestamp_s,
            sensor_key=self.sensor_key,
        )
        if self.source_sha256 != expected:
            raise ContractError("causal source hash differs from its image and timestamp")

    @classmethod
    def from_image(
        cls,
        image: NDArray[np.uint8],
        *,
        timestamp_s: float,
        sensor_key: str,
    ) -> CausalVideoSourceFrame:
        array = np.asarray(image)
        return cls(
            image=array,
            timestamp_s=float(timestamp_s),
            sensor_key=sensor_key,
            source_sha256=causal_video_source_sha256(
                array,
                timestamp_s=float(timestamp_s),
                sensor_key=sensor_key,
            ),
        )


@dataclass(frozen=True, slots=True)
class CausalVideoClip:
    """One unique-frame causal clip ending exactly at the current observation."""

    images: tuple[NDArray[np.uint8], ...]
    frame_timestamps_s: tuple[float, ...]
    source_frame_sha256: tuple[str, ...]
    sensor_key: str
    current_timestamp_s: float
    intervention: str = "identity"

    def __post_init__(self) -> None:
        count = len(self.images)
        if (
            count == 0
            or len(self.frame_timestamps_s) != count
            or len(self.source_frame_sha256) != count
        ):
            raise ContractError("causal clip fields must have one nonempty aligned length")
        if len(set(self.source_frame_sha256)) != count:
            raise ContractError("causal clip cannot contain duplicate source frames")
        timestamps = np.asarray(self.frame_timestamps_s, dtype=np.float64)
        if not np.isfinite(timestamps).all() or (timestamps < 0.0).any():
            raise ContractError("causal clip timestamps are invalid")
        if count > 1 and not (np.diff(timestamps) > 0.0).all():
            raise ContractError("causal clip timestamp slots must be strictly increasing")
        if not math.isclose(
            float(timestamps[-1]),
            self.current_timestamp_s,
            rel_tol=0.0,
            abs_tol=1e-7,
        ):
            raise ContractError("causal clip must end at the current timestamp")
        if not isinstance(self.sensor_key, str) or not self.sensor_key:
            raise ContractError("causal clip sensor key is invalid")
        if not isinstance(self.intervention, str) or not self.intervention:
            raise ContractError("causal clip intervention name is invalid")
        for image in self.images:
            array = np.asarray(image)
            if (
                array.ndim != 3
                or array.shape[2] != 3
                or array.dtype != np.uint8
                or array.flags.writeable
            ):
                raise ContractError("causal clip images must be immutable H-by-W-by-3 uint8")

    def permute_history_content(self, permutation: Sequence[int]) -> CausalVideoClip:
        """Permute historical image content while preserving causal time slots."""

        history_count = len(self.images) - 1
        frozen = tuple(permutation)
        if len(frozen) != history_count or sorted(frozen) != list(range(history_count)):
            raise ContractError("history permutation must be a complete bijection")
        images = tuple(self.images[index] for index in frozen) + (self.images[-1],)
        hashes = tuple(self.source_frame_sha256[index] for index in frozen) + (
            self.source_frame_sha256[-1],
        )
        return CausalVideoClip(
            images=images,
            frame_timestamps_s=self.frame_timestamps_s,
            source_frame_sha256=hashes,
            sensor_key=self.sensor_key,
            current_timestamp_s=self.current_timestamp_s,
            intervention="history-content-permutation.v1",
        )

    def shift_history_timestamps(self, offset_s: float) -> CausalVideoClip:
        """Shift history metadata only, keeping current content/time fixed."""

        if not math.isfinite(offset_s) or offset_s == 0.0:
            raise ContractError("history timestamp shift must be finite and nonzero")
        shifted = tuple(value + offset_s for value in self.frame_timestamps_s[:-1]) + (
            self.frame_timestamps_s[-1],
        )
        return CausalVideoClip(
            images=self.images,
            frame_timestamps_s=shifted,
            source_frame_sha256=self.source_frame_sha256,
            sensor_key=self.sensor_key,
            current_timestamp_s=self.current_timestamp_s,
            intervention="history-timestamp-shift.v1",
        )


@dataclass(frozen=True, slots=True)
class CausalVideoEncoderInput:
    """Fixed-length encoder input with an explicit repeated-source pad prefix."""

    images: tuple[NDArray[np.uint8], ...]
    frame_timestamps_s: tuple[float, ...]
    source_frame_sha256: tuple[str, ...]
    source_valid: tuple[bool, ...]
    sensor_key: str
    current_timestamp_s: float

    def __post_init__(self) -> None:
        count = len(self.images)
        if (
            count == 0
            or len(self.frame_timestamps_s) != count
            or len(self.source_frame_sha256) != count
            or len(self.source_valid) != count
        ):
            raise ContractError("causal encoder-input fields must have one aligned length")
        valid = np.asarray(self.source_valid, dtype=np.bool_)
        if not valid.any():
            raise ContractError("causal encoder input must contain a real source frame")
        first_valid = int(np.flatnonzero(valid)[0])
        if valid[:first_valid].any() or not valid[first_valid:].all():
            raise ContractError("causal encoder padding must be one left prefix")
        timestamps = np.asarray(self.frame_timestamps_s, dtype=np.float64)
        if not np.isfinite(timestamps).all() or (timestamps < 0.0).any():
            raise ContractError("causal encoder-input timestamps are invalid")
        if first_valid and not np.all(timestamps[:first_valid] == timestamps[first_valid]):
            raise ContractError("causal encoder padding must repeat the first real timestamp")
        if count - first_valid > 1 and not (np.diff(timestamps[first_valid:]) > 0.0).all():
            raise ContractError("real causal encoder-input timestamps must be strictly increasing")
        if not math.isclose(
            float(timestamps[-1]), self.current_timestamp_s, rel_tol=0.0, abs_tol=1e-7
        ):
            raise ContractError("causal encoder input must end at the current timestamp")
        hashes = self.source_frame_sha256
        if first_valid and any(value != hashes[first_valid] for value in hashes[:first_valid]):
            raise ContractError("causal encoder padding must repeat the first real source hash")
        if len(set(hashes[first_valid:])) != count - first_valid:
            raise ContractError("real causal encoder-input sources must remain unique")
        if not isinstance(self.sensor_key, str) or not self.sensor_key:
            raise ContractError("causal encoder-input sensor key is invalid")
        first_image = self.images[first_valid]
        for index, image in enumerate(self.images):
            array = np.asarray(image)
            if (
                array.ndim != 3
                or array.shape[2] != 3
                or array.dtype != np.uint8
                or array.flags.writeable
            ):
                raise ContractError("causal encoder images must be immutable H-by-W-by-3 uint8")
            if index < first_valid and image is not first_image:
                raise ContractError("causal encoder padding must reference the first real image")

    @property
    def padding_count(self) -> int:
        return self.source_valid.index(True)


def left_pad_causal_video_clip(
    clip: CausalVideoClip,
    *,
    frame_count: int,
) -> CausalVideoEncoderInput:
    """Match a fixed-frame pretrained encoder without fabricating source evidence."""

    frame_count = _strict_positive_int("frame_count", frame_count)
    if not isinstance(clip, CausalVideoClip):
        raise TypeError("left padding requires one validated CausalVideoClip")
    if len(clip.images) > frame_count:
        raise ContractError("causal clip exceeds the fixed encoder frame count")
    padding_count = frame_count - len(clip.images)
    images = (clip.images[0],) * padding_count + clip.images
    timestamps = (clip.frame_timestamps_s[0],) * padding_count + clip.frame_timestamps_s
    hashes = (clip.source_frame_sha256[0],) * padding_count + clip.source_frame_sha256
    return CausalVideoEncoderInput(
        images=images,
        frame_timestamps_s=timestamps,
        source_frame_sha256=hashes,
        source_valid=(False,) * padding_count + (True,) * len(clip.images),
        sensor_key=clip.sensor_key,
        current_timestamp_s=clip.current_timestamp_s,
    )


def build_causal_video_clip(
    source_frames: Sequence[CausalVideoSourceFrame],
    *,
    current_timestamp_s: float,
    maximum_frames: int,
    tubelet_size: int,
    frame_step: int = 1,
) -> CausalVideoClip | None:
    """Deduplicate, tail-sample and form complete tubelets without padding."""

    maximum_frames = _strict_positive_int("maximum_frames", maximum_frames)
    tubelet_size = _strict_positive_int("tubelet_size", tubelet_size)
    frame_step = _strict_positive_int("frame_step", frame_step)
    if maximum_frames < tubelet_size:
        raise ContractError("maximum_frames must fit at least one complete tubelet")
    if not math.isfinite(current_timestamp_s) or current_timestamp_s < 0.0:
        raise ContractError("current causal-video timestamp is invalid")
    if not source_frames:
        return None

    unique: list[CausalVideoSourceFrame] = []
    seen: set[str] = set()
    previous_timestamp = -math.inf
    timestamp_to_hash: dict[float, str] = {}
    sensor_key = source_frames[0].sensor_key
    for frame in source_frames:
        if not isinstance(frame, CausalVideoSourceFrame):
            raise TypeError("causal clip inputs must be CausalVideoSourceFrame values")
        if frame.sensor_key != sensor_key:
            raise ContractError("causal clip cannot mix sensor streams")
        if frame.timestamp_s < previous_timestamp:
            raise ContractError("causal source frames must arrive in chronological order")
        previous_timestamp = frame.timestamp_s
        existing = timestamp_to_hash.get(frame.timestamp_s)
        if existing is not None and existing != frame.source_sha256:
            raise ContractError("one causal timestamp maps to conflicting source content")
        timestamp_to_hash[frame.timestamp_s] = frame.source_sha256
        if frame.source_sha256 in seen:
            continue
        seen.add(frame.source_sha256)
        unique.append(frame)

    if not unique or not math.isclose(
        unique[-1].timestamp_s,
        current_timestamp_s,
        rel_tol=0.0,
        abs_tol=1e-7,
    ):
        raise ContractError("causal source prefix must end at the current observation")
    reverse_indices = range(len(unique) - 1, -1, -frame_step)
    selected = [unique[index] for index in list(reverse_indices)[:maximum_frames]]
    selected.reverse()
    complete_count = len(selected) - (len(selected) % tubelet_size)
    if complete_count == 0:
        return None
    selected = selected[-complete_count:]
    return CausalVideoClip(
        images=tuple(frame.image for frame in selected),
        frame_timestamps_s=tuple(frame.timestamp_s for frame in selected),
        source_frame_sha256=tuple(frame.source_sha256 for frame in selected),
        sensor_key=sensor_key,
        current_timestamp_s=float(current_timestamp_s),
    )


def build_calvin_causal_video_clip(
    evidence_prefix: Sequence[CalvinPICFEvidenceFrame],
    *,
    sensor_key: str,
    maximum_frames: int,
    tubelet_size: int,
    frame_step: int = 1,
) -> CausalVideoClip | None:
    """Adapt a target-free CALVIN evidence prefix to the generic clip contract."""

    if not evidence_prefix:
        return None
    source_frames = []
    for frame in evidence_prefix:
        matching = tuple(
            observation
            for observation in frame.sensor_observations
            if observation.key == sensor_key
        )
        if len(matching) != 1:
            raise ContractError(f"CALVIN evidence must contain exactly one {sensor_key!r} image")
        source_frames.append(
            CausalVideoSourceFrame.from_image(
                matching[0].value,
                timestamp_s=frame.timestamp_s,
                sensor_key=sensor_key,
            )
        )
    return build_causal_video_clip(
        source_frames,
        current_timestamp_s=evidence_prefix[-1].timestamp_s,
        maximum_frames=maximum_frames,
        tubelet_size=tubelet_size,
        frame_step=frame_step,
    )

"""Deterministic current-plus-future source batches for joint PICF training."""

from __future__ import annotations

import hashlib
import random
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F

from picf_next.content_addressing import canonical_payload_sha256
from picf_next.contracts import ContractError
from picf_next.videomt_exact.calvin_full_dataset import (
    CalvinAllSourcePhysicalSidecar,
    CalvinVerifiedRGBIndex,
    materialize_calvin_videomt_clip,
)
from picf_next.videomt_exact.calvin_targets import (
    VIDEOMT_YTVIS19_CLIP_LENGTH,
    PreparedCalvinVidEoMTClip,
    prepare_calvin_videomt_training_clip,
)
from picf_next.videomt_exact.preprocessing import PreparedVidEoMTFrames, prepare_rgb_frames


class CalvinFutureSourceDataset(Protocol):
    def source_global_index_by_key(self, sample_key: str) -> int: ...

    def future_source_global_indices_by_key(
        self,
        sample_key: str,
        *,
        count: int,
    ) -> tuple[int, ...]: ...


NATIVE_VIDEOMT_SOURCE_ELIGIBILITY_SCHEMA = "picf-next.native-videomt-source-eligibility/v1"


@dataclass(frozen=True, slots=True)
class NativeVidEoMTSourceEligibilityReceipt:
    stream_plan_sha256: str
    required_future_source_frames: int
    episode_count: int
    eligible_sample_count: int
    source_windows_sha256: str

    def __post_init__(self) -> None:
        for name, value in (
            ("stream plan", self.stream_plan_sha256),
            ("source windows", self.source_windows_sha256),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"native VidEoMT {name} digest is not one SHA-256")
        if self.required_future_source_frames != VIDEOMT_YTVIS19_CLIP_LENGTH - 1:
            raise ValueError("native VidEoMT eligibility changed its complete source horizon")
        if self.episode_count <= 0 or self.eligible_sample_count <= 0:
            raise ValueError("native VidEoMT eligibility receipt cannot describe an empty domain")

    @property
    def content(self) -> dict[str, object]:
        return {
            "eligible_sample_count": self.eligible_sample_count,
            "episode_count": self.episode_count,
            "required_future_source_frames": self.required_future_source_frames,
            "schema": NATIVE_VIDEOMT_SOURCE_ELIGIBILITY_SCHEMA,
            "source_windows_sha256": self.source_windows_sha256,
            "stream_plan_sha256": self.stream_plan_sha256,
        }

    @property
    def artifact_sha256(self) -> str:
        return canonical_payload_sha256(
            NATIVE_VIDEOMT_SOURCE_ELIGIBILITY_SCHEMA,
            self.content,
        )

    def to_dict(self) -> dict[str, object]:
        return {**self.content, "artifact_sha256": self.artifact_sha256}


@dataclass(frozen=True, slots=True)
class PreparedNativeVidEoMTCurrentFrame:
    source_global_index: int
    source_rgb: np.ndarray
    source_rgb_sha256: str
    frames: PreparedVidEoMTFrames

    def __post_init__(self) -> None:
        if (
            isinstance(self.source_global_index, bool)
            or not isinstance(self.source_global_index, int)
            or self.source_global_index < 0
        ):
            raise ValueError("native VidEoMT current source index must be non-negative")
        if (
            not isinstance(self.source_rgb, np.ndarray)
            or self.source_rgb.shape != (200, 200, 3)
            or self.source_rgb.dtype != np.uint8
            or self.source_rgb.flags.writeable
        ):
            raise ValueError("native VidEoMT current RGB must be immutable uint8 200x200")
        if hashlib.sha256(self.source_rgb.tobytes(order="C")).hexdigest() != (
            self.source_rgb_sha256
        ):
            raise ValueError("native VidEoMT current RGB digest differs")
        if self.frames.model_input.shape[0] != 1:
            raise ValueError("native VidEoMT current preprocessing must contain one frame")


def prepare_native_videomt_current_frame(
    index: CalvinVerifiedRGBIndex,
    source_global_index: int,
) -> PreparedNativeVidEoMTCurrentFrame:
    """Read and deterministically preprocess current RGB without target access."""

    if (
        isinstance(source_global_index, bool)
        or not isinstance(source_global_index, int)
        or source_global_index < 0
    ):
        raise ValueError("native VidEoMT current source index must be non-negative")
    arrays = index.validated_source_frame_arrays(
        source_global_index,
        fields=("rgb_static",),
        verify_relative_action=False,
    )
    source_rgb = np.array(arrays["rgb_static"], copy=True, order="C")
    if source_rgb.shape != (200, 200, 3) or source_rgb.dtype != np.uint8:
        raise ContractError("native VidEoMT current static RGB contract drifted")
    source_rgb.setflags(write=False)
    return PreparedNativeVidEoMTCurrentFrame(
        source_global_index=source_global_index,
        source_rgb=source_rgb,
        source_rgb_sha256=hashlib.sha256(source_rgb.tobytes(order="C")).hexdigest(),
        frames=prepare_rgb_frames((source_rgb,)),
    )


@contextmanager
def _isolated_augmentation_seed(seed: int):
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("VidEoMT augmentation seed must be a non-negative integer")
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    try:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.manual_seed(seed)
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.set_rng_state(torch_state)


def current_future_source_indices(
    dataset: CalvinFutureSourceDataset,
    sample_key: str,
) -> tuple[int, ...]:
    """Resolve the exact five-frame source clip without crossing a reset."""

    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("joint source sample key must be non-empty")
    current = dataset.source_global_index_by_key(sample_key)
    future = dataset.future_source_global_indices_by_key(
        sample_key,
        count=VIDEOMT_YTVIS19_CLIP_LENGTH - 1,
    )
    indices = (current, *future)
    if len(indices) != VIDEOMT_YTVIS19_CLIP_LENGTH or any(
        right != left + 1 for left, right in zip(indices, indices[1:], strict=False)
    ):
        raise ContractError("joint VidEoMT source clip is not five consecutive raw frames")
    return indices


def audit_native_videomt_source_eligibility(
    dataset: CalvinFutureSourceDataset,
    stream_plan: object,
) -> NativeVidEoMTSourceEligibilityReceipt:
    """Prove every frozen-domain event owns one complete same-episode source clip."""

    plan_sha256 = getattr(stream_plan, "plan_sha256", None)
    episodes = getattr(stream_plan, "episodes", None)
    if not isinstance(plan_sha256, str) or not isinstance(episodes, tuple) or not episodes:
        raise TypeError("native VidEoMT eligibility requires a frozen episode stream plan")
    sample_keys = tuple(
        sample_key for episode in episodes for sample_key in getattr(episode, "sample_keys", ())
    )
    if not sample_keys or len(sample_keys) != len(set(sample_keys)):
        raise ContractError("native VidEoMT stream domain is empty or contains duplicate keys")
    windows = tuple(
        current_future_source_indices(dataset, sample_key) for sample_key in sample_keys
    )
    return NativeVidEoMTSourceEligibilityReceipt(
        stream_plan_sha256=plan_sha256,
        required_future_source_frames=VIDEOMT_YTVIS19_CLIP_LENGTH - 1,
        episode_count=len(episodes),
        eligible_sample_count=len(sample_keys),
        source_windows_sha256=canonical_payload_sha256(
            "picf-next.native-videomt-source-windows/v1",
            windows,
        ),
    )


def _pad_target(
    target: Mapping[str, torch.Tensor],
    *,
    padded_size: tuple[int, int],
    device: torch.device | str,
) -> dict[str, torch.Tensor]:
    if set(target) != {"labels", "ids", "masks", "valid_pixels"}:
        raise ContractError("joint VidEoMT target inventory drifted")
    masks = target["masks"]
    valid_pixels = target["valid_pixels"]
    height, width = masks.shape[-2:]
    target_height, target_width = padded_size
    if height > target_height or width > target_width:
        raise ContractError("joint VidEoMT target exceeds its batch canvas")
    padding = (0, target_width - width, 0, target_height - height)
    return {
        "labels": target["labels"].to(device=device),
        "ids": target["ids"].to(device=device),
        "masks": F.pad(masks, padding, value=0.0).to(device=device),
        "valid_pixels": F.pad(valid_pixels, padding, value=False).to(device=device),
    }


@dataclass(frozen=True, slots=True)
class PreparedNativeVidEoMTSourceBatch:
    normalized_padded_rgb: torch.Tensor
    host_aligned_current_rgb: torch.Tensor
    clip_targets: tuple[Mapping[str, torch.Tensor], ...]
    sample_keys: tuple[str, ...]
    global_indices: tuple[tuple[int, ...], ...]
    identity_keys: tuple[tuple[str, ...], ...]
    augmentation_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        batch = len(self.sample_keys)
        if (
            batch <= 0
            or self.normalized_padded_rgb.ndim != 5
            or self.normalized_padded_rgb.shape[:3] != (batch, VIDEOMT_YTVIS19_CLIP_LENGTH, 3)
            or self.host_aligned_current_rgb.ndim != 5
            or self.host_aligned_current_rgb.shape[:3] != (batch, 1, 3)
            or len(self.clip_targets) != batch
            or len(self.global_indices) != batch
            or len(self.identity_keys) != batch
            or len(self.augmentation_seeds) != batch
        ):
            raise ValueError("joint VidEoMT source batch axes differ")
        if (
            not self.normalized_padded_rgb.is_floating_point()
            or not torch.isfinite(self.normalized_padded_rgb).all()
            or not self.host_aligned_current_rgb.is_floating_point()
            or not torch.isfinite(self.host_aligned_current_rgb).all()
        ):
            raise ValueError("joint VidEoMT source RGB must be finite floating data")
        device = self.normalized_padded_rgb.device
        if self.host_aligned_current_rgb.device != device:
            raise ValueError("source-training and host-aligned RGB must share one device")
        height, width = self.normalized_padded_rgb.shape[-2:]
        for target, indices in zip(self.clip_targets, self.global_indices, strict=True):
            if len(indices) != VIDEOMT_YTVIS19_CLIP_LENGTH or any(
                right != left + 1 for left, right in zip(indices, indices[1:], strict=False)
            ):
                raise ValueError("joint VidEoMT source indices are not one causal clip")
            if (
                target["masks"].shape[-2:] != (height, width)
                or target["valid_pixels"].shape[-2:] != (height, width)
                or any(value.device != device for value in target.values())
            ):
                raise ValueError("joint VidEoMT source targets differ from the batch canvas")

    @property
    def target_count(self) -> int:
        return sum(int(target["labels"].numel()) for target in self.clip_targets)


def prepare_native_videomt_source_batch(
    dataset: CalvinFutureSourceDataset,
    index: CalvinVerifiedRGBIndex,
    sidecar: CalvinAllSourcePhysicalSidecar,
    *,
    sample_keys: Sequence[str],
    augmentation_seeds: Sequence[int],
    device: torch.device | str,
) -> PreparedNativeVidEoMTSourceBatch:
    """Materialize full source supervision while keeping host inputs current-only."""

    keys = tuple(sample_keys)
    seeds = tuple(augmentation_seeds)
    if not keys or len(keys) != len(seeds):
        raise ValueError("joint source keys and augmentation seeds must have equal positive length")
    prepared_clips: list[PreparedCalvinVidEoMTClip] = []
    host_aligned_frames: list[PreparedVidEoMTFrames] = []
    windows: list[tuple[int, ...]] = []
    for sample_key, seed in zip(keys, seeds, strict=True):
        indices = current_future_source_indices(dataset, sample_key)
        source = materialize_calvin_videomt_clip(index, sidecar, indices)
        host_aligned_frames.append(prepare_rgb_frames((source.rgb_static[0],)))
        with _isolated_augmentation_seed(seed):
            prepared = prepare_calvin_videomt_training_clip(
                source.rgb_static,
                source.supervision,
            )
        prepared_clips.append(prepared)
        windows.append(indices)

    padded_height = max(clip.frames.padded_size[0] for clip in prepared_clips)
    padded_width = max(clip.frames.padded_size[1] for clip in prepared_clips)
    padded_size = (padded_height, padded_width)
    frames = []
    targets = []
    for clip in prepared_clips:
        height, width = clip.frames.padded_size
        frames.append(
            F.pad(
                clip.frames.model_input,
                (0, padded_width - width, 0, padded_height - height),
                value=0.0,
            )
        )
        targets.append(_pad_target(clip.target, padded_size=padded_size, device=device))
    host_height = max(frame.padded_size[0] for frame in host_aligned_frames)
    host_width = max(frame.padded_size[1] for frame in host_aligned_frames)
    host_frames = [
        F.pad(
            frame.model_input,
            (0, host_width - frame.padded_size[1], 0, host_height - frame.padded_size[0]),
            value=0.0,
        )
        for frame in host_aligned_frames
    ]
    return PreparedNativeVidEoMTSourceBatch(
        normalized_padded_rgb=torch.stack(frames, dim=0).to(
            device=device,
            dtype=torch.float32,
        ),
        host_aligned_current_rgb=torch.stack(host_frames, dim=0).to(
            device=device,
            dtype=torch.float32,
        ),
        clip_targets=tuple(targets),
        sample_keys=keys,
        global_indices=tuple(windows),
        identity_keys=tuple(clip.identity_keys for clip in prepared_clips),
        augmentation_seeds=seeds,
    )

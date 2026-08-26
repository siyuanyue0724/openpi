"""Fail-closed persistent-storage projection for LingBot-native CALVIN runs.

The projection is deliberately outside the model graph.  It combines exact
frozen-plan cache coverage with measured physical-sidecar shard sizes and
conservative format bounds.  A run may consume the report as a deployment
gate; it must not reinterpret a failure as a training hyperparameter result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from picf_next.contracts import ContractError

LINGBOT_CALVIN_ARTIFACT_CAPACITY_SCHEMA = "picf-next.lingbot-calvin-artifact-capacity/v1"

_CURRENT_PATCH_TOKENS = 256
_DINO_HIDDEN_SIZE = 1024
_FLOAT16_BYTES = 2
_FLOAT32_BYTES = 4
_SHA256_UNICODE_BYTES = 64 * 4
_NPZ_SHARD_ALLOWANCE_BYTES = 64 * 1024
_CURRENT_SHARD_ROWS = 512
_PREDICTIVE_SHARD_ROWS = 2048


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


def _finite_at_least_one(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 1.0
    ):
        raise ContractError(f"{name} must be finite and at least one")
    return float(value)


@dataclass(frozen=True, slots=True)
class PhysicalSidecarStorageSample:
    """One independently generated contiguous sidecar stratum."""

    frame_count: int
    shard_bytes: int
    maximum_shard_bytes_per_frame: float
    scenes: tuple[str, ...]
    maximum_object_count: int
    maximum_identity_key_characters: int

    def __post_init__(self) -> None:
        _positive_integer(self.frame_count, name="physical sample frame count")
        _positive_integer(self.shard_bytes, name="physical sample shard bytes")
        _finite_at_least_one(
            self.maximum_shard_bytes_per_frame,
            name="physical sample maximum shard bytes per frame",
        )
        if self.maximum_shard_bytes_per_frame < self.bytes_per_frame:
            raise ContractError(
                "physical maximum shard bytes/frame cannot be below the stratum mean"
            )
        if (
            not isinstance(self.scenes, tuple)
            or not self.scenes
            or tuple(sorted(set(self.scenes))) != self.scenes
            or any(not isinstance(scene, str) or not scene for scene in self.scenes)
        ):
            raise ContractError("physical sample scenes must be sorted unique text")
        maximum_objects = _positive_integer(
            self.maximum_object_count,
            name="physical sample maximum object count",
        )
        if maximum_objects >= 255:
            raise ContractError("physical sample object count exceeds uint8 owner capacity")
        _positive_integer(
            self.maximum_identity_key_characters,
            name="physical sample maximum identity-key characters",
        )

    @property
    def bytes_per_frame(self) -> float:
        return self.shard_bytes / self.frame_count


@dataclass(frozen=True, slots=True)
class LingBotCalvinArtifactCapacity:
    """Conservative capacity decision for one exact 30k artifact plan."""

    free_bytes: int
    checkpoint_reserve_bytes: int
    minimum_headroom_bytes: int
    physical_total_frames: int
    required_scenes: tuple[str, ...]
    physical_samples: tuple[PhysicalSidecarStorageSample, ...]
    current_grid_record_count: int
    predictive_record_count: int
    physical_safety_factor: float = 1.25

    def __post_init__(self) -> None:
        _positive_integer(self.free_bytes, name="free bytes")
        _nonnegative_integer(
            self.checkpoint_reserve_bytes,
            name="checkpoint reserve bytes",
        )
        _positive_integer(self.minimum_headroom_bytes, name="minimum headroom bytes")
        _positive_integer(self.physical_total_frames, name="physical total frames")
        if (
            not isinstance(self.required_scenes, tuple)
            or not self.required_scenes
            or tuple(sorted(set(self.required_scenes))) != self.required_scenes
            or any(not isinstance(scene, str) or not scene for scene in self.required_scenes)
        ):
            raise ContractError("required physical scenes must be sorted unique text")
        if (
            not isinstance(self.physical_samples, tuple)
            or len(self.physical_samples) < 3
            or any(
                not isinstance(sample, PhysicalSidecarStorageSample)
                for sample in self.physical_samples
            )
        ):
            raise ContractError(
                "physical storage projection requires at least three independent strata"
            )
        if sum(sample.frame_count for sample in self.physical_samples) < 3_000:
            raise ContractError("physical storage projection requires at least 3000 sampled frames")
        sampled_scenes = tuple(
            sorted({scene for sample in self.physical_samples for scene in sample.scenes})
        )
        if sampled_scenes != self.required_scenes:
            raise ContractError("physical storage strata do not cover every dataset scene")
        _positive_integer(
            self.current_grid_record_count,
            name="current-grid record count",
        )
        _positive_integer(
            self.predictive_record_count,
            name="predictive record count",
        )
        _finite_at_least_one(
            self.physical_safety_factor,
            name="physical safety factor",
        )

    @property
    def physical_maximum_sample_bytes_per_frame(self) -> float:
        return max(sample.maximum_shard_bytes_per_frame for sample in self.physical_samples)

    @property
    def projected_physical_bytes(self) -> int:
        return math.ceil(
            self.physical_total_frames
            * self.physical_maximum_sample_bytes_per_frame
            * self.physical_safety_factor
        )

    @property
    def projected_current_grid_bytes(self) -> int:
        feature_bytes = _CURRENT_PATCH_TOKENS * _DINO_HIDDEN_SIZE * _FLOAT16_BYTES
        record_bytes = feature_bytes + _SHA256_UNICODE_BYTES + 8
        shard_count = math.ceil(self.current_grid_record_count / _CURRENT_SHARD_ROWS)
        return (
            self.current_grid_record_count * record_bytes + shard_count * _NPZ_SHARD_ALLOWANCE_BYTES
        )

    @property
    def projected_predictive_bytes(self) -> int:
        maximum_object_count = max(sample.maximum_object_count for sample in self.physical_samples)
        maximum_identity_key_unicode_bytes = (
            max(sample.maximum_identity_key_characters for sample in self.physical_samples) * 4
        )
        object_bytes = (
            _DINO_HIDDEN_SIZE * _FLOAT16_BYTES + _FLOAT32_BYTES + maximum_identity_key_unicode_bytes
        )
        fixed_record_bytes = 3 * 8 + 2 * _SHA256_UNICODE_BYTES + 2 * 8
        record_bytes = fixed_record_bytes + maximum_object_count * object_bytes
        shard_count = math.ceil(self.predictive_record_count / _PREDICTIVE_SHARD_ROWS)
        return (
            self.predictive_record_count * record_bytes + shard_count * _NPZ_SHARD_ALLOWANCE_BYTES
        )

    @property
    def required_bytes(self) -> int:
        return sum(
            (
                self.checkpoint_reserve_bytes,
                self.minimum_headroom_bytes,
                self.projected_physical_bytes,
                self.projected_current_grid_bytes,
                self.projected_predictive_bytes,
            )
        )

    @property
    def status(self) -> str:
        return "PASS" if self.required_bytes <= self.free_bytes else "FAIL"

    @property
    def residual_bytes(self) -> int:
        return self.free_bytes - self.required_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "schema": LINGBOT_CALVIN_ARTIFACT_CAPACITY_SCHEMA,
            "status": self.status,
            "free_bytes": self.free_bytes,
            "required_bytes": self.required_bytes,
            "residual_bytes": self.residual_bytes,
            "checkpoint_reserve_bytes": self.checkpoint_reserve_bytes,
            "minimum_headroom_bytes": self.minimum_headroom_bytes,
            "physical": {
                "total_frames": self.physical_total_frames,
                "required_scenes": list(self.required_scenes),
                "safety_factor": self.physical_safety_factor,
                "maximum_sample_bytes_per_frame": (self.physical_maximum_sample_bytes_per_frame),
                "projected_bytes": self.projected_physical_bytes,
                "samples": [
                    {
                        "frame_count": sample.frame_count,
                        "shard_bytes": sample.shard_bytes,
                        "bytes_per_frame": sample.bytes_per_frame,
                        "maximum_shard_bytes_per_frame": (sample.maximum_shard_bytes_per_frame),
                        "scenes": list(sample.scenes),
                        "maximum_object_count": sample.maximum_object_count,
                        "maximum_identity_key_characters": (sample.maximum_identity_key_characters),
                    }
                    for sample in self.physical_samples
                ],
            },
            "current_grid": {
                "record_count": self.current_grid_record_count,
                "projected_bytes": self.projected_current_grid_bytes,
                "projection": ("exact-float16-feature-payload-plus-conservative-npz-overhead/v1"),
            },
            "predictive": {
                "record_count": self.predictive_record_count,
                "projected_bytes": self.projected_predictive_bytes,
                "maximum_object_count": max(
                    sample.maximum_object_count for sample in self.physical_samples
                ),
                "maximum_identity_key_characters": max(
                    sample.maximum_identity_key_characters for sample in self.physical_samples
                ),
                "projection": ("all-dataset-scenes-maximum-object-inventory/v1"),
            },
        }

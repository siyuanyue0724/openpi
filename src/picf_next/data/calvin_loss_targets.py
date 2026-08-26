"""Host-neutral locators for constructing CALVIN loss-only targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from picf_next.data.calvin_physical_supervision_schema import (
    source_array_sha256,
)
from picf_next.data.calvin_target_request import CALVIN_PHYSICAL_SOURCE_FIELDS
from picf_next.data.robot_record import RobotTransitionRecord


def calvin_physical_source_hashes(
    record: RobotTransitionRecord,
) -> tuple[tuple[str, str], ...]:
    """Fingerprint the materialized CALVIN sensors used by physical labels."""

    if not isinstance(record, RobotTransitionRecord):
        raise TypeError("CALVIN physical hashes require a RobotTransitionRecord")
    values: dict[str, np.ndarray] = {}
    for observation in record.array_observations:
        _archive_path, separator, source_field = observation.source_path.rpartition("#")
        if separator and source_field in CALVIN_PHYSICAL_SOURCE_FIELDS:
            if source_field in values:
                raise ValueError(f"duplicate CALVIN source field {source_field!r}")
            values[source_field] = observation.value
    if set(values) != CALVIN_PHYSICAL_SOURCE_FIELDS:
        missing = sorted(CALVIN_PHYSICAL_SOURCE_FIELDS - set(values))
        extra = sorted(set(values) - CALVIN_PHYSICAL_SOURCE_FIELDS)
        raise ValueError(f"CALVIN physical source fields differ: missing={missing}, extra={extra}")
    return tuple((name, source_array_sha256(name, values[name])) for name in sorted(values))


def _validate_common_request_fields(
    *,
    sample_key: str,
    source_global_index: int,
    augmentation_seed: int,
    source_sensor_sha256: tuple[tuple[str, str], ...],
    request_name: str,
) -> None:
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError(f"{request_name} sample key cannot be empty")
    if (
        not isinstance(source_global_index, int)
        or isinstance(source_global_index, bool)
        or source_global_index < 0
    ):
        raise ValueError(f"{request_name} source index must be non-negative")
    if (
        not isinstance(augmentation_seed, int)
        or isinstance(augmentation_seed, bool)
        or augmentation_seed < 0
    ):
        raise ValueError(f"{request_name} augmentation seed must be non-negative")
    if not isinstance(source_sensor_sha256, tuple) or any(
        not isinstance(item, tuple)
        or len(item) != 2
        or not isinstance(item[0], str)
        or not item[0]
        or not isinstance(item[1], str)
        or len(item[1]) != 64
        or any(character not in "0123456789abcdef" for character in item[1])
        for item in source_sensor_sha256
    ):
        raise ValueError(f"{request_name} source sensor hashes are invalid")
    names = tuple(name for name, _digest in source_sensor_sha256)
    if names != tuple(sorted(names)) or len(set(names)) != len(names):
        raise ValueError(f"{request_name} source sensor hashes must be sorted and unique")


@dataclass(frozen=True, slots=True)
class CalvinStatefulLossTargetRequest:
    """Loss-side locator with no action, task text or deploy-visible payload."""

    sample_key: str
    segment_index: int
    source_global_index: int
    augmentation_seed: int
    source_sensor_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_common_request_fields(
            sample_key=self.sample_key,
            source_global_index=self.source_global_index,
            augmentation_seed=self.augmentation_seed,
            source_sensor_sha256=self.source_sensor_sha256,
            request_name="loss-target",
        )
        if (
            not isinstance(self.segment_index, int)
            or isinstance(self.segment_index, bool)
            or self.segment_index < 0
        ):
            raise ValueError("loss-target segment index must be non-negative")

    @property
    def source_sensor_hash_by_field(self) -> dict[str, str]:
        return dict(self.source_sensor_sha256)


@dataclass(frozen=True, slots=True)
class CalvinSourceFrameLossTargetRequest:
    """Loss-side locator for a task-independent CALVIN source frame."""

    sample_key: str
    source_global_index: int
    augmentation_seed: int
    source_sensor_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        _validate_common_request_fields(
            sample_key=self.sample_key,
            source_global_index=self.source_global_index,
            augmentation_seed=self.augmentation_seed,
            source_sensor_sha256=self.source_sensor_sha256,
            request_name="source-frame loss-target",
        )

    @property
    def source_sensor_hash_by_field(self) -> dict[str, str]:
        return dict(self.source_sensor_sha256)

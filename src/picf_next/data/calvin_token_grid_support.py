"""Measure physical CALVIN identities on the exact LingBot/Qwen token grid.

This is loss-side audit code. It projects the existing depth-verified owner
rasters through the same fixed Qwen geometry used by training. The resulting
measurements are never model inputs and contain no learned selector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real

import numpy as np

from picf_next.contracts import ContractError
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
)
from picf_next.data.lingbot_calvin_projection import (
    validate_lingbot_calvin_projection_payload,
)
from picf_next.data.qwen3vl_raster import project_qwen3vl_segmentation


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ContractError(f"{name} must be a finite non-negative number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ContractError(f"{name} must be a finite non-negative number")
    return result


def _nonnegative_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ContractError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class CalvinTokenGridViewSupport:
    """One physical identity's exact soft support in one Qwen camera view."""

    camera_name: str
    merged_grid_hw: tuple[int, int]
    target_mass: float
    maximum_target_probability: float
    positive_token_count: int
    strict_object_winner_token_count: int
    strict_categorical_winner_token_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.camera_name, str) or not self.camera_name:
            raise ContractError("CALVIN token-grid camera name must be non-empty")
        if (
            not isinstance(self.merged_grid_hw, tuple)
            or len(self.merged_grid_hw) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in self.merged_grid_hw
            )
        ):
            raise ContractError("CALVIN token-grid shape must be two positive integers")
        token_count = math.prod(self.merged_grid_hw)
        target_mass = _finite_nonnegative(
            self.target_mass,
            name="CALVIN projected target mass",
        )
        maximum = _finite_nonnegative(
            self.maximum_target_probability,
            name="CALVIN maximum target probability",
        )
        if target_mass > token_count + 1e-6 or maximum > 1.0 + 1e-6:
            raise ContractError("CALVIN token-grid probability mass exceeds its support")
        counts = (
            _nonnegative_int(
                self.positive_token_count,
                name="CALVIN positive token count",
            ),
            _nonnegative_int(
                self.strict_object_winner_token_count,
                name="CALVIN strict object-winner token count",
            ),
            _nonnegative_int(
                self.strict_categorical_winner_token_count,
                name="CALVIN strict categorical-winner token count",
            ),
        )
        if any(value > token_count for value in counts):
            raise ContractError("CALVIN token-grid count exceeds the camera token count")
        if counts[2] > counts[1] or counts[1] > counts[0]:
            raise ContractError("CALVIN token-grid winner counts are inconsistent")
        if (target_mass > 0) != (maximum > 0) or (target_mass > 0) != (counts[0] > 0):
            raise ContractError("CALVIN token-grid positive support metrics disagree")

    @property
    def measurable(self) -> bool:
        return self.target_mass > 0 and self.positive_token_count > 0

    @property
    def object_row_addressable(self) -> bool:
        return self.strict_object_winner_token_count > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "camera_name": self.camera_name,
            "measurable": self.measurable,
            "merged_grid_hw": list(self.merged_grid_hw),
            "maximum_target_probability": self.maximum_target_probability,
            "positive_token_count": self.positive_token_count,
            "strict_categorical_winner_token_count": (self.strict_categorical_winner_token_count),
            "strict_object_winner_token_count": self.strict_object_winner_token_count,
            "object_row_addressable": self.object_row_addressable,
            "target_mass": self.target_mass,
        }


@dataclass(frozen=True, slots=True)
class CalvinTokenGridIdentitySupport:
    """One physical identity's support across every deploy-visible camera."""

    identity_key: str
    views: tuple[CalvinTokenGridViewSupport, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.identity_key, str) or not self.identity_key:
            raise ContractError("CALVIN token-grid identity must be non-empty")
        if not isinstance(self.views, tuple) or any(
            not isinstance(value, CalvinTokenGridViewSupport) for value in self.views
        ):
            raise ContractError("CALVIN token-grid identity views are malformed")
        camera_names = tuple(value.camera_name for value in self.views)
        if not camera_names or len(set(camera_names)) != len(camera_names):
            raise ContractError("CALVIN token-grid identity requires unique camera views")

    @property
    def target_mass(self) -> float:
        return math.fsum(value.target_mass for value in self.views)

    @property
    def maximum_target_probability(self) -> float:
        return max(value.maximum_target_probability for value in self.views)

    @property
    def positive_token_count(self) -> int:
        return sum(value.positive_token_count for value in self.views)

    @property
    def strict_object_winner_token_count(self) -> int:
        return sum(value.strict_object_winner_token_count for value in self.views)

    @property
    def strict_categorical_winner_token_count(self) -> int:
        return sum(value.strict_categorical_winner_token_count for value in self.views)

    @property
    def measurable(self) -> bool:
        return self.target_mass > 0 and self.positive_token_count > 0

    @property
    def object_row_addressable(self) -> bool:
        return self.strict_object_winner_token_count > 0

    def as_dict(self) -> dict[str, object]:
        return {
            "identity_key": self.identity_key,
            "measurable": self.measurable,
            "maximum_target_probability": self.maximum_target_probability,
            "positive_token_count": self.positive_token_count,
            "strict_categorical_winner_token_count": (self.strict_categorical_winner_token_count),
            "strict_object_winner_token_count": self.strict_object_winner_token_count,
            "object_row_addressable": self.object_row_addressable,
            "target_mass": self.target_mass,
            "views": [value.as_dict() for value in self.views],
        }


def _view_support(
    *,
    camera_name: str,
    owner_id: int,
    projected_instance_ids: tuple[int, ...],
    object_probability: np.ndarray,
    context_probability: np.ndarray,
    observed_fraction: np.ndarray,
    supervised: np.ndarray,
    merged_grid_hw: tuple[int, int],
) -> CalvinTokenGridViewSupport:
    token_count = math.prod(merged_grid_hw)
    if (
        object_probability.ndim != 2
        or object_probability.shape[0] != token_count
        or context_probability.shape != (token_count,)
        or observed_fraction.shape != (token_count,)
        or supervised.shape != (token_count,)
        or supervised.dtype != np.bool_
    ):
        raise ContractError("CALVIN projected token support differs from Qwen geometry")
    if owner_id not in projected_instance_ids:
        target = np.zeros(token_count, dtype=np.float32)
        other_maximum = np.zeros(token_count, dtype=np.float32)
    else:
        column = projected_instance_ids.index(owner_id)
        target = object_probability[:, column]
        other = np.delete(object_probability, column, axis=1)
        other_maximum = (
            other.max(axis=1)
            if other.shape[1]
            else np.zeros(token_count, dtype=object_probability.dtype)
        )
    positive = supervised & (target > 0)
    strict_object_winner = positive & (target > other_maximum)
    strict_categorical_winner = strict_object_winner & (target > context_probability)
    return CalvinTokenGridViewSupport(
        camera_name=camera_name,
        merged_grid_hw=merged_grid_hw,
        target_mass=float(np.sum(observed_fraction * target, dtype=np.float64)),
        maximum_target_probability=(float(target[positive].max()) if positive.any() else 0.0),
        positive_token_count=int(positive.sum()),
        strict_object_winner_token_count=int(strict_object_winner.sum()),
        strict_categorical_winner_token_count=int(strict_categorical_winner.sum()),
    )


def project_calvin_token_grid_identity_support(
    frame: CalvinPhysicalSupervisionFrame,
    *,
    projection: Mapping[str, object],
    minimum_supervised_fraction: float = 0.0,
) -> tuple[CalvinTokenGridIdentitySupport, ...]:
    """Project every physical identity through the frozen training geometry."""

    if not isinstance(frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("CALVIN token-grid projection requires a physical frame")
    if (
        isinstance(minimum_supervised_fraction, bool)
        or not isinstance(minimum_supervised_fraction, Real)
        or not math.isfinite(float(minimum_supervised_fraction))
        or not 0.0 <= float(minimum_supervised_fraction) <= 1.0
    ):
        raise ContractError("CALVIN minimum supervised fraction must lie in [0,1]")
    validated = validate_lingbot_calvin_projection_payload(projection)
    views = validated["views"]
    cameras = {camera.camera_name: camera for camera in frame.cameras}
    if len(cameras) != len(frame.cameras) or set(cameras) != set(views):
        raise ContractError("CALVIN physical cameras differ from the Qwen projection views")
    owner_ids = tuple(range(1, len(frame.identity_keys) + 1))
    projected_by_camera = {}
    for camera_name in sorted(cameras):
        camera = cameras[camera_name]
        view = views[camera_name]
        source_shape = tuple(int(value) for value in view["source_shape"][:2])
        if (
            camera.owner_index.shape != source_shape
            or camera.owner_supervised.shape != source_shape
        ):
            raise ContractError("CALVIN owner raster differs from measured Qwen source geometry")
        projected_by_camera[camera_name] = project_qwen3vl_segmentation(
            camera.owner_index,
            instance_ids=owner_ids,
            image_grid_thw=np.asarray(view["image_grid_thw"], dtype=np.int64),
            patch_size=int(validated["patch_size"]),
            merge_size=int(validated["merge_size"]),
            pixel_supervised=camera.owner_supervised,
            minimum_supervised_fraction=float(minimum_supervised_fraction),
        ).merged

    output = []
    for owner_id, identity_key in zip(owner_ids, frame.identity_keys, strict=True):
        view_support = []
        for camera_name in sorted(cameras):
            projected = projected_by_camera[camera_name]
            view = views[camera_name]
            merged_grid_hw = (
                int(view["merged_grid_hw"][0]),
                int(view["merged_grid_hw"][1]),
            )
            view_support.append(
                _view_support(
                    camera_name=camera_name,
                    owner_id=owner_id,
                    projected_instance_ids=projected.instance_ids,
                    object_probability=projected.object_probability,
                    context_probability=projected.context_probability,
                    observed_fraction=projected.observed_fraction,
                    supervised=projected.supervised,
                    merged_grid_hw=merged_grid_hw,
                )
            )
        output.append(
            CalvinTokenGridIdentitySupport(
                identity_key=identity_key,
                views=tuple(view_support),
            )
        )
    return tuple(output)

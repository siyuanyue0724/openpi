"""Evaluation-only controlled RGB occlusion for CALVIN task targets.

The physical sidecar is used only to construct and audit a counterfactual
observation.  The returned value is a target-free ``CalvinPICFEvidenceFrame``;
no identity, mask, box, simulator state or action target crosses the model
boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from picf_next.data.calvin import (
    CALVIN_OBSERVATION_SPECS,
    CalvinPICFEvidenceFrame,
)
from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_CAMERA_SPECS,
    source_array_sha256,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
)

_RGB_KEY_BY_SOURCE_FIELD = {
    source_field: observation_key
    for source_field, observation_key, _shape, _dtype, _units in CALVIN_OBSERVATION_SPECS
    if source_field.startswith("rgb_")
}


@dataclass(frozen=True, slots=True)
class CalvinControlledOcclusionCamera:
    camera_name: str
    host_image_key: str
    source_rgb_field: str
    source_observation_key: str
    target_pixel_count: int
    supervised_target_pixel_count: int
    target_bbox_xyxy: tuple[int, int, int, int] | None
    occluder_bbox_xyxy: tuple[int, int, int, int] | None
    occluder_pixel_count: int
    occluded_fraction: float
    fill_rgb: tuple[int, int, int] | None
    source_rgb_sha256: str
    occluded_rgb_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_name": self.camera_name,
            "fill_rgb": None if self.fill_rgb is None else list(self.fill_rgb),
            "host_image_key": self.host_image_key,
            "occluded_fraction": self.occluded_fraction,
            "occluded_rgb_sha256": self.occluded_rgb_sha256,
            "occluder_bbox_xyxy": (
                None if self.occluder_bbox_xyxy is None else list(self.occluder_bbox_xyxy)
            ),
            "occluder_pixel_count": self.occluder_pixel_count,
            "source_observation_key": self.source_observation_key,
            "source_rgb_field": self.source_rgb_field,
            "source_rgb_sha256": self.source_rgb_sha256,
            "supervised_target_pixel_count": self.supervised_target_pixel_count,
            "target_bbox_xyxy": (
                None if self.target_bbox_xyxy is None else list(self.target_bbox_xyxy)
            ),
            "target_pixel_count": self.target_pixel_count,
        }


@dataclass(frozen=True, slots=True)
class CalvinControlledOcclusion:
    evidence_frame: CalvinPICFEvidenceFrame
    target_identity_keys: tuple[str, ...]
    cameras: tuple[CalvinControlledOcclusionCamera, ...]
    bbox_expansion_fraction: float
    minimum_margin_pixels: int

    def contract_dict(self) -> dict[str, Any]:
        return {
            "bbox_expansion_fraction": self.bbox_expansion_fraction,
            "cameras": [camera.to_dict() for camera in self.cameras],
            "method": "target-owner-bbox.global-rgb-median-fill.v1",
            "minimum_margin_pixels": self.minimum_margin_pixels,
            "model_input_contains_structural_target": False,
            "target_identity_keys": list(self.target_identity_keys),
        }


def _expanded_bbox(
    mask: np.ndarray,
    *,
    expansion_fraction: float,
    minimum_margin_pixels: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    yy, xx = np.nonzero(mask)
    if yy.size == 0:
        raise ValueError("target bounding box requires at least one target pixel")
    target = (int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1)
    target_width = target[2] - target[0]
    target_height = target[3] - target[1]
    margin_x = max(minimum_margin_pixels, math.ceil(target_width * expansion_fraction))
    margin_y = max(minimum_margin_pixels, math.ceil(target_height * expansion_fraction))
    height, width = mask.shape
    expanded = (
        max(0, target[0] - margin_x),
        max(0, target[1] - margin_y),
        min(width, target[2] + margin_x),
        min(height, target[3] + margin_y),
    )
    return target, expanded


def build_calvin_controlled_rgb_occlusion(
    evidence_frame: CalvinPICFEvidenceFrame,
    physical_frame: CalvinPhysicalSupervisionFrame,
    *,
    target_identity_keys: tuple[str, ...],
    bbox_expansion_fraction: float = 0.25,
    minimum_margin_pixels: int = 2,
) -> CalvinControlledOcclusion:
    """Hide visible target pixels in every CALVIN RGB camera deterministically."""

    if not isinstance(evidence_frame, CalvinPICFEvidenceFrame):
        raise TypeError("controlled occlusion requires a CALVIN PICF evidence frame")
    if not isinstance(physical_frame, CalvinPhysicalSupervisionFrame):
        raise TypeError("controlled occlusion requires a CALVIN physical supervision frame")
    targets = tuple(target_identity_keys)
    if not targets or any(not isinstance(key, str) or not key for key in targets):
        raise ValueError("controlled occlusion target identities must be nonempty strings")
    if len(set(targets)) != len(targets):
        raise ValueError("controlled occlusion target identities must be unique")
    if (
        isinstance(bbox_expansion_fraction, bool)
        or not isinstance(bbox_expansion_fraction, int | float)
        or not math.isfinite(float(bbox_expansion_fraction))
        or not 0.0 <= float(bbox_expansion_fraction) <= 1.0
    ):
        raise ValueError("bbox expansion fraction must be finite and lie in [0, 1]")
    if (
        not isinstance(minimum_margin_pixels, int)
        or isinstance(minimum_margin_pixels, bool)
        or minimum_margin_pixels < 0
    ):
        raise ValueError("minimum occlusion margin must be a nonnegative integer")

    identity_to_owner = {key: index + 1 for index, key in enumerate(physical_frame.identity_keys)}
    missing = sorted(set(targets) - set(identity_to_owner))
    if missing:
        raise ValueError(
            f"controlled occlusion targets are absent from physical inventory: {missing}"
        )
    target_owner_indices = np.asarray(
        [identity_to_owner[key] for key in targets],
        dtype=np.uint8,
    )
    sensor_by_key = {
        observation.key: observation for observation in evidence_frame.sensor_observations
    }
    if len(sensor_by_key) != len(evidence_frame.sensor_observations):
        raise ValueError("controlled occlusion evidence contains duplicate sensor keys")
    camera_by_name = {camera.camera_name: camera for camera in physical_frame.cameras}
    if len(camera_by_name) != len(physical_frame.cameras):
        raise ValueError("controlled occlusion physical frame contains duplicate cameras")

    replacements: dict[str, np.ndarray] = {}
    reports = []
    total_target_pixels = 0
    for spec in CALVIN_CAMERA_SPECS:
        camera_name = str(spec["camera_name"])
        source_field = str(spec["source_rgb_field"])
        observation_key = _RGB_KEY_BY_SOURCE_FIELD.get(source_field)
        camera = camera_by_name.get(camera_name)
        observation = sensor_by_key.get(str(observation_key))
        if observation_key is None or camera is None or observation is None:
            raise ValueError("controlled occlusion camera/evidence contract is incomplete")
        image = observation.value
        expected_shape = (int(spec["height"]), int(spec["width"]), 3)
        if image.shape != expected_shape or image.dtype != np.uint8:
            raise ValueError("controlled occlusion source RGB shape or dtype changed")
        source_digest = source_array_sha256(source_field, image)
        if source_digest != camera.source_rgb_sha256:
            raise ValueError("controlled occlusion source RGB differs from the physical sidecar")
        target_mask = np.isin(camera.owner_index, target_owner_indices)
        supervised_target = target_mask & camera.owner_supervised
        target_pixels = int(target_mask.sum())
        total_target_pixels += target_pixels
        if target_pixels == 0:
            occluded = image
            target_bbox = None
            occluder_bbox = None
            occluder_pixels = 0
            fill_rgb = None
        else:
            target_bbox, occluder_bbox = _expanded_bbox(
                target_mask,
                expansion_fraction=float(bbox_expansion_fraction),
                minimum_margin_pixels=minimum_margin_pixels,
            )
            fill_array = np.rint(np.median(image.reshape(-1, 3), axis=0)).astype(np.uint8)
            fill_rgb = tuple(int(value) for value in fill_array.tolist())
            x0, y0, x1, y1 = occluder_bbox
            occluded = image.copy()
            occluded[y0:y1, x0:x1] = fill_array
            occluded.setflags(write=False)
            occluder_pixels = (x1 - x0) * (y1 - y0)
            replacements[observation_key] = occluded
        occluded_digest = source_array_sha256(source_field, occluded)
        reports.append(
            CalvinControlledOcclusionCamera(
                camera_name=camera_name,
                host_image_key=str(spec["host_image_key"]),
                source_rgb_field=source_field,
                source_observation_key=observation_key,
                target_pixel_count=target_pixels,
                supervised_target_pixel_count=int(supervised_target.sum()),
                target_bbox_xyxy=target_bbox,
                occluder_bbox_xyxy=occluder_bbox,
                occluder_pixel_count=occluder_pixels,
                occluded_fraction=float(occluder_pixels / target_mask.size),
                fill_rgb=fill_rgb,
                source_rgb_sha256=source_digest,
                occluded_rgb_sha256=occluded_digest,
            )
        )
    if total_target_pixels == 0:
        raise ValueError("controlled occlusion target has no visible pixel in either RGB camera")

    observations = tuple(
        replace(observation, value=replacements[observation.key])
        if observation.key in replacements
        else observation
        for observation in evidence_frame.sensor_observations
    )
    return CalvinControlledOcclusion(
        evidence_frame=replace(evidence_frame, sensor_observations=observations),
        target_identity_keys=targets,
        cameras=tuple(reports),
        bbox_expansion_fraction=float(bbox_expansion_fraction),
        minimum_margin_pixels=minimum_margin_pixels,
    )

"""Versioned contracts for calibrated Euclidean object geometry.

PICF uses a diagonal Gaussian only for coordinates with declared physical
semantics.  A tensor width alone cannot distinguish a robot-frame position
from an image coordinate, a learned embedding, or an angle on a manifold.
This contract makes that distinction explicit and gives data, model and loss
boundaries one immutable value to compare.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _text_vector(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of nonempty strings")
    return tuple(_nonempty_text(item, f"{name}[{index}]") for index, item in enumerate(value))


def _finite_vector(value: object, name: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of finite numbers")
    converted = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{name}[{index}] must be a finite number")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite number")
        converted.append(number)
    return tuple(converted)


@dataclass(frozen=True, slots=True)
class PhysicalGeometryContract:
    """One normalized Euclidean chart for an object's physical state.

    Model tensors contain ``(raw - offset) / scale`` for every coordinate.
    ``quantity`` distinguishes, for example, an AABB centre from a centre of
    mass even when both use the same frame and axes.  Periodic angles,
    quaternions and unconstrained learned features are deliberately excluded:
    they require a distribution appropriate to their topology rather than the
    diagonal Gaussian used by the current filter.
    """

    name: str
    quantity: str
    reference_frame: str
    axes: tuple[str, ...]
    units: tuple[str, ...]
    normalization_offset: tuple[float, ...]
    normalization_scale: tuple[float, ...]
    schema: str = "picf.physical-euclidean-geometry.v1"

    def __post_init__(self) -> None:
        text_fields = {
            "name": self.name,
            "quantity": self.quantity,
            "reference_frame": self.reference_frame,
            "schema": self.schema,
        }
        if any(not isinstance(value, str) or not value.strip() for value in text_fields.values()):
            raise ValueError("geometry contract text fields must be nonempty strings")
        if self.schema != "picf.physical-euclidean-geometry.v1":
            raise ValueError("unsupported physical geometry contract schema")
        if not self.axes or any(
            not isinstance(axis, str) or not axis.strip() for axis in self.axes
        ):
            raise ValueError("geometry axes must be nonempty strings")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("geometry axes must be unique")
        width = len(self.axes)
        fields = {
            "units": self.units,
            "normalization_offset": self.normalization_offset,
            "normalization_scale": self.normalization_scale,
        }
        if any(len(value) != width for value in fields.values()):
            raise ValueError("geometry units and normalization must align with axes")
        if any(not isinstance(unit, str) or not unit.strip() for unit in self.units):
            raise ValueError("geometry units must be nonempty strings")
        numeric = (*self.normalization_offset, *self.normalization_scale)
        if any(isinstance(value, bool) or not math.isfinite(value) for value in numeric):
            raise ValueError("geometry normalization values must be finite numbers")
        if any(scale <= 0.0 for scale in self.normalization_scale):
            raise ValueError("geometry normalization scales must be positive")

    @property
    def dimension(self) -> int:
        return len(self.axes)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return the canonical JSON-serializable contract payload."""

        return {
            "axes": list(self.axes),
            "name": self.name,
            "normalization_offset": list(self.normalization_offset),
            "normalization_scale": list(self.normalization_scale),
            "quantity": self.quantity,
            "reference_frame": self.reference_frame,
            "schema": self.schema,
            "units": list(self.units),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PhysicalGeometryContract:
        """Parse a manifest payload without accepting unknown or missing fields."""

        if not isinstance(payload, Mapping):
            raise TypeError("physical geometry contract payload must be a mapping")
        required = {
            "axes",
            "name",
            "normalization_offset",
            "normalization_scale",
            "quantity",
            "reference_frame",
            "schema",
            "units",
        }
        if set(payload) != required:
            raise ValueError("physical geometry contract fields differ from schema v1")
        return cls(
            name=_nonempty_text(payload["name"], "geometry.name"),
            quantity=_nonempty_text(payload["quantity"], "geometry.quantity"),
            reference_frame=_nonempty_text(payload["reference_frame"], "geometry.reference_frame"),
            axes=_text_vector(payload["axes"], "geometry.axes"),
            units=_text_vector(payload["units"], "geometry.units"),
            normalization_offset=_finite_vector(
                payload["normalization_offset"], "geometry.normalization_offset"
            ),
            normalization_scale=_finite_vector(
                payload["normalization_scale"], "geometry.normalization_scale"
            ),
            schema=_nonempty_text(payload["schema"], "geometry.schema"),
        )

    def normalize_values(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Normalize one raw coordinate row without introducing tensor dependencies."""

        if len(values) != self.dimension:
            raise ValueError("raw geometry width differs from its contract")
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("raw geometry values must be finite numbers")
        return tuple(
            (value - offset) / scale
            for value, offset, scale in zip(
                values,
                self.normalization_offset,
                self.normalization_scale,
                strict=True,
            )
        )

    def normalize_variance(self, variance: tuple[float, ...]) -> tuple[float, ...]:
        """Map raw coordinate variance into the normalized model chart."""

        if len(variance) != self.dimension:
            raise ValueError("raw geometry variance width differs from its contract")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0 for value in variance
        ):
            raise ValueError("raw geometry variance must be finite and nonnegative")
        return tuple(
            value / (scale * scale)
            for value, scale in zip(variance, self.normalization_scale, strict=True)
        )

    def denormalize_values(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Map one model-chart coordinate row back to declared physical units."""

        if len(values) != self.dimension:
            raise ValueError("normalized geometry width differs from its contract")
        if any(isinstance(value, bool) or not math.isfinite(value) for value in values):
            raise ValueError("normalized geometry values must be finite numbers")
        return tuple(
            value * scale + offset
            for value, offset, scale in zip(
                values,
                self.normalization_offset,
                self.normalization_scale,
                strict=True,
            )
        )

    def denormalize_variance(self, variance: tuple[float, ...]) -> tuple[float, ...]:
        """Map model-chart coordinate variance back to physical squared units."""

        if len(variance) != self.dimension:
            raise ValueError("normalized geometry variance width differs from its contract")
        if any(
            isinstance(value, bool) or not math.isfinite(value) or value < 0.0 for value in variance
        ):
            raise ValueError("normalized geometry variance must be finite and nonnegative")
        return tuple(
            value * scale * scale
            for value, scale in zip(variance, self.normalization_scale, strict=True)
        )

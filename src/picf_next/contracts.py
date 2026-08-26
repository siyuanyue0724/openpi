"""Host-neutral data contracts for PICF-Next.

These structures deliberately contain no CALVIN labels, masks, host-layer
objects, or mutable recurrent module state. Training targets belong in a
separate loss-side contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.floating]
BoolArray = NDArray[np.bool_]
IntArray = NDArray[np.integer]


class ContractError(ValueError):
    """Raised when a PICF boundary violates a shape or semantic invariant."""


def _require_rank(name: str, value: NDArray, rank: int) -> None:
    if not isinstance(value, np.ndarray):
        raise ContractError(f"{name} must be a NumPy array")
    if value.ndim != rank:
        raise ContractError(f"{name} must have rank {rank}, got shape {value.shape}")


def _require_float(name: str, value: NDArray) -> None:
    if not np.issubdtype(value.dtype, np.floating):
        raise ContractError(f"{name} must be floating point, got {value.dtype}")
    if not np.isfinite(value).all():
        raise ContractError(f"{name} contains NaN or infinity")


def _require_probability(name: str, value: FloatArray) -> None:
    _require_float(name, value)
    if ((value < 0.0) | (value > 1.0)).any():
        raise ContractError(f"{name} must lie in [0, 1]")


def _require_unit_rows(name: str, value: FloatArray, valid: BoolArray) -> None:
    """Require occupied identity addresses to lie on the unit sphere."""

    if value.shape[1] == 0:
        raise ContractError(f"{name} must have positive width")
    if valid.any():
        norms = np.linalg.norm(value[valid], axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
            raise ContractError(f"valid {name} rows must have unit norm")


@dataclass(frozen=True, slots=True)
class DenseEvidence:
    """One sample's complete selected token stream for a single modality.

    `tokens` are the native encoder outputs at a versioned boundary. PICF may
    add metadata and object ownership, but it may not delete or overwrite these
    values. A missing modality has `available=False` and exactly zero tokens.
    """

    modality: str
    encoder_contract: str
    tokens: FloatArray
    available: bool
    timestamps: FloatArray
    confidence: FloatArray
    geometry: FloatArray | None = None
    group_ids: IntArray | None = None
    current_measurement_valid: BoolArray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.modality, str) or not self.modality.strip():
            raise ContractError("modality must be non-empty")
        if not isinstance(self.encoder_contract, str) or not self.encoder_contract.strip():
            raise ContractError("encoder_contract must be non-empty and versioned")
        if not isinstance(self.available, bool | np.bool_):
            raise ContractError("available must be boolean")

        _require_rank("tokens", self.tokens, 2)
        _require_float("tokens", self.tokens)
        token_count = self.tokens.shape[0]

        _require_rank("timestamps", self.timestamps, 1)
        _require_float("timestamps", self.timestamps)
        _require_rank("confidence", self.confidence, 1)
        _require_probability("confidence", self.confidence)
        if self.timestamps.shape != (token_count,):
            raise ContractError("timestamps must contain one value per token")
        if (self.timestamps < 0.0).any():
            raise ContractError("timestamps must be non-negative")
        if self.confidence.shape != (token_count,):
            raise ContractError("confidence must contain one value per token")

        if not self.available and token_count != 0:
            raise ContractError("a missing modality must not emit learned-looking tokens")

        if self.geometry is not None:
            _require_rank("geometry", self.geometry, 2)
            _require_float("geometry", self.geometry)
            if self.geometry.shape[0] != token_count:
                raise ContractError("geometry must contain one row per token")

        if self.group_ids is not None:
            _require_rank("group_ids", self.group_ids, 1)
            if not np.issubdtype(self.group_ids.dtype, np.integer):
                raise ContractError("group_ids must be integer")
            if self.group_ids.shape != (token_count,):
                raise ContractError("group_ids must contain one value per token")
            if (self.group_ids < -1).any():
                raise ContractError("group_ids must be -1 or non-negative")

        if self.current_measurement_valid is not None:
            _require_rank("current_measurement_valid", self.current_measurement_valid, 1)
            if not np.issubdtype(self.current_measurement_valid.dtype, np.bool_):
                raise ContractError("current_measurement_valid must be boolean")
            if self.current_measurement_valid.shape != (token_count,):
                raise ContractError("current_measurement_valid must contain one value per token")
            if self.current_measurement_valid.any():
                newest = float(self.timestamps.max())
                if not np.allclose(
                    self.timestamps[self.current_measurement_valid],
                    newest,
                    rtol=0.0,
                    atol=1e-7,
                ):
                    raise ContractError(
                        "current measurements must come from the newest evidence timestamp"
                    )
        elif token_count > 1 and not np.allclose(
            self.timestamps,
            self.timestamps[0],
            rtol=0.0,
            atol=1e-7,
        ):
            raise ContractError(
                "multi-timestamp evidence requires an explicit current_measurement_valid role"
            )

    @property
    def token_count(self) -> int:
        return self.tokens.shape[0]

    @property
    def effective_current_measurement_valid(self) -> BoolArray:
        """Return tokens allowed to define the current observation likelihood."""

        if self.current_measurement_valid is not None:
            return self.current_measurement_valid
        return np.ones(self.token_count, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class ObjectBeliefSet:
    """A bounded unordered posterior set for one sample and one time step.

    Row order has no physical meaning. `valid` distinguishes occupied posterior
    rows from unused capacity. `address` carries persistent identity semantics;
    `content` and `geometry` carry time-varying state.
    """

    address: FloatArray
    content: FloatArray
    geometry: FloatArray
    geometry_covariance_diag: FloatArray
    existence: FloatArray
    visibility: FloatArray
    measurement_age_s: FloatArray
    valid: BoolArray
    age: IntArray

    def __post_init__(self) -> None:
        state_arrays = {
            "address": self.address,
            "content": self.content,
            "geometry": self.geometry,
        }
        arrays_2d = {**state_arrays, "geometry_covariance_diag": self.geometry_covariance_diag}
        for name, value in arrays_2d.items():
            _require_rank(name, value, 2)
            _require_float(name, value)

        capacity = self.address.shape[0]
        for name, value in arrays_2d.items():
            if value.shape[0] != capacity:
                raise ContractError(f"{name} capacity differs from address capacity")

        geometry_width = self.geometry.shape[1]
        if self.geometry_covariance_diag.shape[1] != geometry_width:
            raise ContractError("geometry_covariance_diag width must equal geometry width")

        if (self.geometry_covariance_diag < 0.0).any():
            raise ContractError("geometry_covariance_diag must be non-negative")

        for name, value in {
            "existence": self.existence,
            "visibility": self.visibility,
        }.items():
            _require_rank(name, value, 1)
            _require_probability(name, value)
            if value.shape != (capacity,):
                raise ContractError(f"{name} must contain one value per posterior row")

        _require_rank("measurement_age_s", self.measurement_age_s, 1)
        _require_float("measurement_age_s", self.measurement_age_s)
        if self.measurement_age_s.shape != (capacity,):
            raise ContractError("measurement_age_s must contain one value per posterior row")
        if (self.measurement_age_s < 0.0).any():
            raise ContractError("measurement_age_s must be non-negative")

        _require_rank("valid", self.valid, 1)
        if self.valid.dtype != np.bool_ or self.valid.shape != (capacity,):
            raise ContractError("valid must be a bool vector with one value per posterior row")
        _require_unit_rows("address", self.address, self.valid)

        _require_rank("age", self.age, 1)
        if not np.issubdtype(self.age.dtype, np.integer) or self.age.shape != (capacity,):
            raise ContractError("age must be an integer vector with one value per posterior row")
        if (self.age < 0).any():
            raise ContractError("age must be non-negative")

        if (self.visibility > self.existence + 1e-7).any():
            raise ContractError("visibility probability cannot exceed existence probability")
        if (self.existence[~self.valid] != 0.0).any() or (
            self.visibility[~self.valid] != 0.0
        ).any():
            raise ContractError("unused capacity rows must have zero existence and visibility")
        if (self.measurement_age_s[~self.valid] != 0.0).any():
            raise ContractError("unused capacity rows must have zero measurement age")
        for name, value in arrays_2d.items():
            if (value[~self.valid] != 0.0).any():
                raise ContractError(f"unused capacity rows must have zero {name}")
        if (self.age[~self.valid] != 0).any():
            raise ContractError("unused capacity rows must have zero age")

    @property
    def capacity(self) -> int:
        return self.address.shape[0]

    @property
    def object_count(self) -> int:
        return int(self.valid.sum())

    @property
    def state_width(self) -> int:
        return self.address.shape[1] + self.content.shape[1] + self.geometry.shape[1]

    @property
    def dynamic_width(self) -> int:
        return self.content.shape[1] + self.geometry.shape[1]


@dataclass(frozen=True, slots=True)
class ObjectObservationSet:
    """Task-independent current-frame object observations.

    Observation rows are unordered and may include unused capacity. They are
    not persistent identities: `address` is evidence used for association, not
    a row index promoted to an object identifier.
    """

    address: FloatArray
    content: FloatArray
    geometry: FloatArray
    geometry_covariance_diag: FloatArray
    existence: FloatArray
    valid: BoolArray

    def __post_init__(self) -> None:
        state_arrays = {
            "address": self.address,
            "content": self.content,
            "geometry": self.geometry,
        }
        arrays_2d = {**state_arrays, "geometry_covariance_diag": self.geometry_covariance_diag}
        for name, value in arrays_2d.items():
            _require_rank(name, value, 2)
            _require_float(name, value)

        capacity = self.address.shape[0]
        for name, value in arrays_2d.items():
            if value.shape[0] != capacity:
                raise ContractError(f"{name} capacity differs from address capacity")

        geometry_width = self.geometry.shape[1]
        if self.geometry_covariance_diag.shape[1] != geometry_width:
            raise ContractError("geometry_covariance_diag width must equal geometry width")
        if (self.geometry_covariance_diag < 0.0).any():
            raise ContractError("geometry_covariance_diag must be non-negative")

        _require_rank("existence", self.existence, 1)
        _require_probability("existence", self.existence)
        if self.existence.shape != (capacity,):
            raise ContractError("existence must contain one value per observation row")

        _require_rank("valid", self.valid, 1)
        if self.valid.dtype != np.bool_ or self.valid.shape != (capacity,):
            raise ContractError("valid must be a bool vector with one value per observation row")
        _require_unit_rows("address", self.address, self.valid)
        if (self.existence[~self.valid] != 0.0).any():
            raise ContractError("unused observation rows must have zero existence")
        for name, value in arrays_2d.items():
            if (value[~self.valid] != 0.0).any():
                raise ContractError(f"unused observation rows must have zero {name}")

    @property
    def state_width(self) -> int:
        return self.address.shape[1] + self.content.shape[1] + self.geometry.shape[1]

    @property
    def dynamic_width(self) -> int:
        return self.content.shape[1] + self.geometry.shape[1]


@dataclass(frozen=True, slots=True)
class PICFContext:
    """The complete host-neutral forward context exposed to a VLA adapter."""

    evidence: tuple[DenseEvidence, ...]
    posterior: ObjectBeliefSet
    innovation: FloatArray
    ownership: tuple[FloatArray, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, tuple) or not all(
            isinstance(item, DenseEvidence) for item in self.evidence
        ):
            raise ContractError("evidence must be a tuple of DenseEvidence streams")
        if not isinstance(self.posterior, ObjectBeliefSet):
            raise ContractError("posterior must be an ObjectBeliefSet")
        if not isinstance(self.ownership, tuple):
            raise ContractError("ownership must be a tuple of arrays")
        modalities = [item.modality for item in self.evidence]
        if len(modalities) != len(set(modalities)):
            raise ContractError("PICFContext may contain at most one stream per modality")

        _require_rank("innovation", self.innovation, 2)
        _require_float("innovation", self.innovation)
        expected_innovation = (self.posterior.capacity, self.posterior.dynamic_width)
        if self.innovation.shape != expected_innovation:
            raise ContractError(
                f"innovation must have posterior dynamic shape {expected_innovation}"
            )
        if (self.innovation[~self.posterior.valid] != 0.0).any():
            raise ContractError("unused posterior rows must have zero innovation")

        if len(self.ownership) != len(self.evidence):
            raise ContractError("ownership must align one-to-one with evidence streams")
        for stream, assignment in zip(self.evidence, self.ownership, strict=True):
            _require_rank(f"ownership[{stream.modality}]", assignment, 2)
            _require_probability(f"ownership[{stream.modality}]", assignment)
            expected = (stream.token_count, self.posterior.capacity + 1)
            if assignment.shape != expected:
                actual = assignment.shape
                raise ContractError(
                    f"ownership[{stream.modality}] must have shape {expected}, got {actual}"
                )
            if assignment.shape[0] and not np.allclose(assignment.sum(axis=1), 1.0, atol=1e-6):
                raise ContractError("each token ownership row must sum to one including context")
            if (assignment[:, : self.posterior.capacity][:, ~self.posterior.valid] != 0.0).any():
                raise ContractError("ownership cannot assign mass to unused posterior rows")
            if stream.group_ids is not None:
                for group_id in np.unique(stream.group_ids):
                    if group_id < 0:
                        continue
                    group = assignment[stream.group_ids == group_id]
                    if group.shape[0] > 1 and not np.allclose(group, group[0], atol=1e-6):
                        raise ContractError(
                            f"ownership[{stream.modality}] must be shared within token group "
                            f"{int(group_id)}"
                        )

    def evidence_for(self, modality: str) -> DenseEvidence | None:
        return next((item for item in self.evidence if item.modality == modality), None)

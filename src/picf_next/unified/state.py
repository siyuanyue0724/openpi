"""Typed, fixed-capacity persistent state for the unified PICF graph."""

from __future__ import annotations

import struct
from collections.abc import Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import Any

import numpy as np
import torch

LIFECYCLE_MODES = 3
CONTINUE = 0
BIRTH = 1
EMPTY = 2
_SERIAL_MAGIC = b"PICFUB01"
_SERIAL_VERSION = 1
_SERIAL_HEADER = struct.Struct("<8sB6I")


@dataclass(frozen=True, slots=True)
class GeometrySchema:
    """Static meaning of the explicit geometry coordinates.

    A state carries only numeric sufficient statistics.  Names, units and frame
    are configuration, so serialized state size is independent of episode length
    and the set of modalities available on a particular step.
    """

    names: tuple[str, ...]
    units: tuple[str, ...]
    frame: str

    def __post_init__(self) -> None:
        if not isinstance(self.names, tuple) or not isinstance(self.units, tuple):
            raise TypeError("geometry names and units must be immutable tuples")
        if any(
            not isinstance(value, str) for value in (*self.names, *self.units)
        ) or not isinstance(self.frame, str):
            raise TypeError("geometry names, units and frame must be strings")
        if not self.names:
            raise ValueError("geometry schema must contain at least one coordinate")
        if len(self.names) != len(self.units):
            raise ValueError("geometry names and units must have equal length")
        if len(set(self.names)) != len(self.names):
            raise ValueError("geometry coordinate names must be unique")
        if any(not value for value in (*self.names, *self.units, self.frame)):
            raise ValueError("geometry names, units and frame must be non-empty")

    @property
    def width(self) -> int:
        return len(self.names)

    def canonical_dict(self) -> dict[str, Any]:
        """Return the stable payload bound into runtime state schemas."""

        return {
            "names": list(self.names),
            "units": list(self.units),
            "frame": self.frame,
        }


@dataclass(frozen=True, slots=True)
class UnifiedBeliefState:
    """One bounded posterior set carried across control steps.

    Every numeric sufficient statistic is stored in float32, independently of
    the host model's activation dtype.  Lifecycle is represented by normalized
    log probabilities in the fixed order
    ``continue, birth, empty``.  There is deliberately no active bit, confidence
    threshold, row class or learned persistent row identity.
    """

    content: torch.Tensor
    lifecycle_log_probs: torch.Tensor
    geometry_mean: torch.Tensor
    geometry_information: torch.Tensor
    geometry_valid: torch.Tensor
    content_log_variance: torch.Tensor
    expected_age: torch.Tensor
    evidence_age: torch.Tensor

    def __post_init__(self) -> None:
        self.validate()

    @property
    def batch_size(self) -> int:
        return self.content.shape[0]

    @property
    def capacity(self) -> int:
        return self.content.shape[1]

    @property
    def content_dim(self) -> int:
        return self.content.shape[2]

    @property
    def geometry_dim(self) -> int:
        return self.geometry_mean.shape[2]

    @property
    def uncertainty_dim(self) -> int:
        return self.content_log_variance.shape[2]

    @property
    def lifecycle_probs(self) -> torch.Tensor:
        return self.lifecycle_log_probs.exp()

    @property
    def nonempty_probability(self) -> torch.Tensor:
        return self.lifecycle_probs[..., CONTINUE] + self.lifecycle_probs[..., BIRTH]

    @classmethod
    def canonical_width(
        cls,
        *,
        content_dim: int,
        geometry_dim: int,
        uncertainty_dim: int,
    ) -> int:
        dimensions = (content_dim, geometry_dim, uncertainty_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("belief dimensions must be integers")
        if min(dimensions) <= 0:
            raise ValueError("all belief dimensions must be positive")
        information_width = geometry_dim * (geometry_dim + 1) // 2
        return (
            content_dim + 2 + geometry_dim + information_width + geometry_dim + uncertainty_dim + 2
        )

    def validate(self, *, check_psd: bool = True, tolerance: float = 1e-5) -> None:
        if not isinstance(check_psd, bool):
            raise TypeError("check_psd must be boolean")
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)):
            raise TypeError("belief validation tolerance must be real-valued")
        if not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("belief validation tolerance must be finite and non-negative")
        expected_ranks = {
            "content": 3,
            "lifecycle_log_probs": 3,
            "geometry_mean": 3,
            "geometry_information": 4,
            "geometry_valid": 3,
            "content_log_variance": 3,
            "expected_age": 2,
            "evidence_age": 2,
        }
        for name, rank in expected_ranks.items():
            if getattr(self, name).ndim != rank:
                raise ValueError(f"{name} must have rank {rank}")
        batch, capacity, _ = self.content.shape
        if (
            batch <= 0
            or capacity <= 0
            or self.content_dim <= 0
            or self.geometry_dim <= 0
            or self.uncertainty_dim <= 0
        ):
            raise ValueError("belief batch, capacity and feature widths must be positive")
        geometry_dim = self.geometry_mean.shape[-1]
        expected = {
            "lifecycle_log_probs": (batch, capacity, LIFECYCLE_MODES),
            "geometry_mean": (batch, capacity, geometry_dim),
            "geometry_information": (batch, capacity, geometry_dim, geometry_dim),
            "geometry_valid": (batch, capacity, geometry_dim),
            "content_log_variance": (batch, capacity, self.uncertainty_dim),
            "expected_age": (batch, capacity),
            "evidence_age": (batch, capacity),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape:
                raise ValueError(f"{name} must have shape {shape}, got {tuple(value.shape)}")
        float_fields = (
            self.content,
            self.lifecycle_log_probs,
            self.geometry_mean,
            self.geometry_information,
            self.content_log_variance,
            self.expected_age,
            self.evidence_age,
        )
        if any(value.dtype != torch.float32 for value in float_fields):
            raise TypeError("persistent belief numeric fields must use torch.float32")
        if self.geometry_valid.dtype != torch.bool:
            raise TypeError("geometry_valid must be boolean")
        device = self.content.device
        if any(value.device != device for value in (*float_fields, self.geometry_valid)):
            raise ValueError("all belief fields must be on the same device")
        if any(not torch.isfinite(value).all() for value in float_fields):
            raise ValueError("belief fields must be finite")
        normalization = torch.logsumexp(self.lifecycle_log_probs.float(), dim=-1)
        if not torch.allclose(normalization, torch.zeros_like(normalization), atol=tolerance):
            raise ValueError("lifecycle_log_probs must be normalized")
        if (self.expected_age < 0).any() or (self.evidence_age < 0).any():
            raise ValueError("belief ages must be non-negative")
        information = self.geometry_information.float()
        if not torch.allclose(information, information.transpose(-1, -2), atol=tolerance):
            raise ValueError("geometry_information must be symmetric")
        valid_pair = self.geometry_valid.unsqueeze(-1) & self.geometry_valid.unsqueeze(-2)
        if (information.masked_fill(valid_pair, 0).abs() > tolerance).any():
            raise ValueError("invalid geometry coordinates must carry zero information")
        if check_psd and geometry_dim:
            eigenvalues = torch.linalg.eigvalsh(information)
            if (eigenvalues < -tolerance).any():
                raise ValueError("geometry_information must be positive semidefinite")

    def canonical(self) -> torch.Tensor:
        """Pack each row into the single canonical sufficient-statistic vector."""

        lifecycle_odds = self.lifecycle_log_probs[..., :2] - self.lifecycle_log_probs[..., 2:3]
        rows, cols = torch.triu_indices(
            self.geometry_dim,
            self.geometry_dim,
            device=self.content.device,
        )
        information = self.geometry_information[..., rows, cols]
        return torch.cat(
            (
                self.content,
                lifecycle_odds,
                self.geometry_mean,
                information,
                self.geometry_valid.to(torch.float32),
                self.content_log_variance,
                self.expected_age.unsqueeze(-1),
                self.evidence_age.unsqueeze(-1),
            ),
            dim=-1,
        )

    @classmethod
    def from_canonical(
        cls,
        packed: torch.Tensor,
        *,
        content_dim: int,
        geometry_dim: int,
        uncertainty_dim: int,
    ) -> UnifiedBeliefState:
        dimensions = (content_dim, geometry_dim, uncertainty_dim)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
            raise TypeError("belief dimensions must be integers")
        if packed.ndim != 3:
            raise ValueError("canonical belief must have shape [batch, capacity, width]")
        if not packed.is_floating_point():
            raise TypeError("canonical belief must be floating point")
        expected_width = cls.canonical_width(
            content_dim=content_dim,
            geometry_dim=geometry_dim,
            uncertainty_dim=uncertainty_dim,
        )
        if packed.shape[-1] != expected_width:
            raise ValueError(
                f"canonical belief width must be {expected_width}, got {packed.shape[-1]}"
            )
        cursor = 0

        def take(width: int) -> torch.Tensor:
            nonlocal cursor
            value = packed[..., cursor : cursor + width]
            cursor += width
            return value

        packed = packed.float()
        content = take(content_dim)
        lifecycle_odds = take(2)
        lifecycle_logits = torch.cat(
            (lifecycle_odds, torch.zeros_like(lifecycle_odds[..., :1])), dim=-1
        )
        lifecycle_log_probs = torch.log_softmax(lifecycle_logits, dim=-1)
        geometry_mean = take(geometry_dim)
        information_width = geometry_dim * (geometry_dim + 1) // 2
        information_upper = take(information_width)
        rows, cols = torch.triu_indices(geometry_dim, geometry_dim, device=packed.device)
        information = packed.new_zeros((*packed.shape[:2], geometry_dim, geometry_dim))
        information[..., rows, cols] = information_upper
        information[..., cols, rows] = information_upper
        geometry_valid = take(geometry_dim) > 0.5
        valid_pair = geometry_valid.unsqueeze(-1) & geometry_valid.unsqueeze(-2)
        information = information.masked_fill(~valid_pair, 0)
        content_log_variance = take(uncertainty_dim)
        expected_age = take(1).squeeze(-1).clamp_min(0)
        evidence_age = take(1).squeeze(-1).clamp_min(0)
        if cursor != expected_width:
            raise AssertionError("canonical belief parser did not consume its declared width")
        return cls(
            content=content,
            lifecycle_log_probs=lifecycle_log_probs,
            geometry_mean=geometry_mean,
            geometry_information=information,
            geometry_valid=geometry_valid,
            content_log_variance=content_log_variance,
            expected_age=expected_age,
            evidence_age=evidence_age,
        )

    def detached(self) -> UnifiedBeliefState:
        values = {
            field: getattr(self, field).detach().clone() for field in self.__dataclass_fields__
        }
        return replace(self, **values)

    def permute_rows(self, permutation: torch.Tensor) -> UnifiedBeliefState:
        if permutation.ndim != 1 or permutation.numel() != self.capacity:
            raise ValueError("permutation must contain one index per belief row")
        if permutation.dtype != torch.long:
            raise TypeError("permutation must use torch.long indices")
        if not torch.equal(torch.sort(permutation.cpu()).values, torch.arange(self.capacity)):
            raise ValueError("permutation must contain every row exactly once")
        values = {
            field: getattr(self, field).index_select(1, permutation.to(self.content.device))
            for field in self.__dataclass_fields__
        }
        return replace(self, **values)

    def serialize(self) -> bytes:
        """Serialize a state to a deterministic float32 wire representation."""

        packed = self.canonical().detach().to(device="cpu", dtype=torch.float32).contiguous()
        header = _SERIAL_HEADER.pack(
            _SERIAL_MAGIC,
            _SERIAL_VERSION,
            self.batch_size,
            self.capacity,
            self.content_dim,
            self.geometry_dim,
            self.uncertainty_dim,
            packed.shape[-1],
        )
        return header + packed.numpy().astype("<f4", copy=False).tobytes(order="C")

    @classmethod
    def deserialize(cls, payload: bytes) -> UnifiedBeliefState:
        if len(payload) < _SERIAL_HEADER.size:
            raise ValueError("serialized belief payload is truncated")
        magic, version, batch, capacity, content_dim, geometry_dim, uncertainty_dim, width = (
            _SERIAL_HEADER.unpack_from(payload)
        )
        if magic != _SERIAL_MAGIC or version != _SERIAL_VERSION:
            raise ValueError("serialized belief schema is unsupported")
        expected_width = cls.canonical_width(
            content_dim=content_dim,
            geometry_dim=geometry_dim,
            uncertainty_dim=uncertainty_dim,
        )
        if width != expected_width:
            raise ValueError("serialized belief width disagrees with its dimensions")
        count = batch * capacity * width
        expected_bytes = _SERIAL_HEADER.size + count * np.dtype("<f4").itemsize
        if len(payload) != expected_bytes:
            raise ValueError(
                f"serialized belief has {len(payload)} bytes, expected {expected_bytes}"
            )
        array = np.frombuffer(payload, dtype="<f4", offset=_SERIAL_HEADER.size, count=count)
        packed = torch.from_numpy(array.copy()).reshape(batch, capacity, width)
        return cls.from_canonical(
            packed,
            content_dim=content_dim,
            geometry_dim=geometry_dim,
            uncertainty_dim=uncertainty_dim,
        )


def empty_belief_state(
    *,
    batch_size: int,
    capacity: int,
    content_dim: int,
    geometry_dim: int,
    uncertainty_dim: int,
    birth_hazard: float = 0.01,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> UnifiedBeliefState:
    """Construct an FP32 empty-set prior without an active-row decision.

    ``dtype`` records the surrounding activation dtype for API compatibility;
    persistent sufficient statistics are deliberately promoted to float32.
    """

    dimensions = (batch_size, capacity, content_dim, geometry_dim, uncertainty_dim)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in dimensions):
        raise TypeError("empty belief dimensions must be integers")
    if isinstance(birth_hazard, bool) or not isinstance(birth_hazard, (int, float)):
        raise TypeError("birth_hazard must be real-valued")
    if not np.isfinite(birth_hazard) or not 0 < birth_hazard < 1:
        raise ValueError("birth_hazard must be strictly between zero and one")
    if min(dimensions) <= 0:
        raise ValueError("all empty belief dimensions must be positive")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("dtype must be floating point")
    shape = (batch_size, capacity)
    persistent_dtype = torch.float32
    lifecycle = torch.tensor(
        [torch.finfo(persistent_dtype).tiny, birth_hazard, 1.0 - birth_hazard],
        device=device,
        dtype=persistent_dtype,
    )
    lifecycle_log_probs = lifecycle.log().expand(*shape, LIFECYCLE_MODES).clone()
    return UnifiedBeliefState(
        content=torch.zeros((*shape, content_dim), device=device, dtype=persistent_dtype),
        lifecycle_log_probs=lifecycle_log_probs,
        geometry_mean=torch.zeros((*shape, geometry_dim), device=device, dtype=persistent_dtype),
        geometry_information=torch.zeros(
            (*shape, geometry_dim, geometry_dim), device=device, dtype=persistent_dtype
        ),
        geometry_valid=torch.zeros((*shape, geometry_dim), device=device, dtype=torch.bool),
        content_log_variance=torch.zeros(
            (*shape, uncertainty_dim), device=device, dtype=persistent_dtype
        ),
        expected_age=torch.zeros(shape, device=device, dtype=persistent_dtype),
        evidence_age=torch.zeros(shape, device=device, dtype=persistent_dtype),
    )


def deterministic_birth_noise(
    *,
    episode_keys: Sequence[str],
    frame_indices: Sequence[int],
    capacity: int,
    content_dim: int,
    base_seed: int = 0,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create replayable exchangeable birth noise without persistent row IDs."""

    if len(episode_keys) != len(frame_indices) or not episode_keys:
        raise ValueError("episode_keys and frame_indices must be equal non-empty batches")
    controls = (capacity, content_dim, base_seed)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in controls):
        raise TypeError("birth-noise dimensions and base_seed must be integers")
    if capacity <= 0 or content_dim <= 0 or base_seed < 0:
        raise ValueError("birth-noise dimensions and base_seed must be valid")
    rows = []
    for episode_key, frame_index in zip(episode_keys, frame_indices, strict=True):
        if not isinstance(episode_key, str) or not episode_key:
            raise ValueError("birth-noise episode keys must be non-empty strings")
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
            raise ValueError("birth-noise episode keys and frame indices must be valid")
        digest = sha256(f"{base_seed}:{episode_key}:{frame_index}".encode()).digest()
        seed = int.from_bytes(digest[:8], "little") % (2**63 - 1)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        rows.append(torch.randn((capacity, content_dim), generator=generator))
    return torch.stack(rows).to(device=device, dtype=torch.float32)


def stack_belief_states(
    states: Sequence[UnifiedBeliefState],
    *,
    device: torch.device | str | None = None,
) -> UnifiedBeliefState:
    """Stack one-state lane records into a batch without changing semantics."""

    if not states:
        raise ValueError("at least one belief state is required")
    reference = states[0]
    reference_shape = (
        reference.capacity,
        reference.content_dim,
        reference.geometry_dim,
        reference.uncertainty_dim,
    )
    for state in states:
        if state.batch_size != 1:
            raise ValueError("each lane belief must contain exactly one batch item")
        shape = (
            state.capacity,
            state.content_dim,
            state.geometry_dim,
            state.uncertainty_dim,
        )
        if shape != reference_shape:
            raise ValueError("lane beliefs must share one state schema")
    target_device = reference.content.device if device is None else torch.device(device)
    values = {
        field: torch.cat(
            [getattr(state, field).to(device=target_device) for state in states],
            dim=0,
        )
        for field in reference.__dataclass_fields__
    }
    return replace(reference, **values)


def unbind_belief_state(state: UnifiedBeliefState) -> tuple[UnifiedBeliefState, ...]:
    """Split a batched state into detached one-item lane records."""

    detached = state.detached()
    return tuple(
        replace(
            detached,
            **{
                field: getattr(detached, field)[index : index + 1]
                for field in detached.__dataclass_fields__
            },
        )
        for index in range(detached.batch_size)
    )

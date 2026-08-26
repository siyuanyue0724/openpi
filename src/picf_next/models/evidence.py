"""Host-neutral typed evidence and comparison-space projection.

The shared binding projection exists only for object discovery and association.
Every native token bank is returned unchanged for the direct host action path;
there is no pooling, top-k selection or common action pre-bottleneck.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True, slots=True)
class ModalityProjectionSpec:
    name: str
    token_dim: int
    geometry_dim: int = 0
    require_single_active_group: bool = False

    def __post_init__(self) -> None:
        if not self.name or "." in self.name:
            raise ValueError("modality names must be nonempty and cannot contain dots")
        if (
            not isinstance(self.token_dim, int)
            or isinstance(self.token_dim, bool)
            or self.token_dim <= 0
        ):
            raise ValueError("token_dim must be positive")
        if (
            not isinstance(self.geometry_dim, int)
            or isinstance(self.geometry_dim, bool)
            or self.geometry_dim < 0
        ):
            raise ValueError("geometry_dim cannot be negative")
        if not isinstance(self.require_single_active_group, bool):
            raise ValueError("require_single_active_group must be boolean")


@dataclass(frozen=True, slots=True)
class NativeTokenBank:
    modality: str
    tokens: torch.Tensor
    valid: torch.Tensor
    geometry: torch.Tensor | None = None
    group_id: torch.Tensor | None = None
    timestamps: torch.Tensor | None = None
    confidence: torch.Tensor | None = None
    encoder_contract: str | None = None
    current_measurement_valid: torch.Tensor | None = None

    @property
    def effective_current_measurement_valid(self) -> torch.Tensor:
        """Return the valid-token subset that may update the current posterior."""

        if self.current_measurement_valid is not None:
            return self.current_measurement_valid
        return self.valid


@dataclass(frozen=True, slots=True)
class ModalityTokenSpan:
    modality: str
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class BindingProjectionOutput:
    native_banks: tuple[NativeTokenBank, ...]
    binding_features: torch.Tensor
    token_valid: torch.Tensor
    current_measurement_valid: torch.Tensor
    token_group_id: torch.Tensor
    modality_index: torch.Tensor
    spans: tuple[ModalityTokenSpan, ...]

    @property
    def total_tokens(self) -> int:
        return self.binding_features.shape[1]

    def current_discovery_inputs(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return the only evidence allowed to define the current likelihood."""

        current_features = self.binding_features * self.current_measurement_valid.unsqueeze(-1)
        current_group_id = torch.where(
            self.current_measurement_valid,
            self.token_group_id,
            -1,
        )
        return current_features, self.current_measurement_valid, current_group_id


class MultimodalBindingProjector(nn.Module):
    """Project heterogeneous native tokens only into an object-binding space."""

    def __init__(
        self,
        specs: tuple[ModalityProjectionSpec, ...],
        *,
        binding_dim: int,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        if not specs:
            raise ValueError("at least one modality projection spec is required")
        if not isinstance(binding_dim, int) or isinstance(binding_dim, bool) or binding_dim <= 0:
            raise ValueError("binding_dim must be positive")
        names = [spec.name for spec in specs]
        if len(set(names)) != len(names):
            raise ValueError("modality projection specs must be unique")
        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        self.specs = {spec.name: spec for spec in specs}
        self.modality_to_index = {name: index for index, name in enumerate(names)}
        self.binding_dim = binding_dim
        self.validate_tensor_values = validate_tensor_values
        self.content_projection = nn.ModuleDict(
            {spec.name: nn.Linear(spec.token_dim, binding_dim, bias=False) for spec in specs}
        )
        self.geometry_projection = nn.ModuleDict(
            {
                spec.name: nn.Linear(spec.geometry_dim, binding_dim, bias=False)
                for spec in specs
                if spec.geometry_dim
            }
        )
        self.output_norm = nn.ModuleDict({spec.name: nn.LayerNorm(binding_dim) for spec in specs})
        self.modality_embedding = nn.Parameter(torch.empty(len(specs), binding_dim))
        nn.init.normal_(self.modality_embedding, std=binding_dim**-0.5)

    def _validate_bank(self, bank: NativeTokenBank) -> None:
        if bank.modality not in self.specs:
            raise ValueError(f"modality {bank.modality} is not configured")
        spec = self.specs[bank.modality]
        if bank.tokens.ndim != 3 or bank.tokens.shape[-1] != spec.token_dim:
            raise ValueError(f"{bank.modality} token shape differs from its projection spec")
        if bank.valid.dtype != torch.bool or bank.valid.shape != bank.tokens.shape[:2]:
            raise ValueError(f"{bank.modality} validity must be bool batch-by-token")
        if bank.valid.device != bank.tokens.device:
            raise ValueError(f"{bank.modality} validity and tokens must share a device")
        if not torch.is_floating_point(bank.tokens):
            raise ValueError(f"{bank.modality} tokens must be floating tensors")
        if self.validate_tensor_values:
            if not torch.isfinite(bank.tokens).all():
                raise ValueError(f"{bank.modality} tokens must be finite floating tensors")
            if (bank.tokens[~bank.valid] != 0.0).any():
                raise ValueError(f"{bank.modality} invalid token padding must be exactly zero")

        current_measurement_valid = bank.current_measurement_valid
        if current_measurement_valid is not None:
            if (
                current_measurement_valid.dtype != torch.bool
                or current_measurement_valid.shape != bank.valid.shape
            ):
                raise ValueError(
                    f"{bank.modality} current measurement role must be bool batch-by-token"
                )
            if current_measurement_valid.device != bank.tokens.device:
                raise ValueError(
                    f"{bank.modality} current measurement role must match token device"
                )
            if self.validate_tensor_values:
                if (current_measurement_valid & ~bank.valid).any():
                    raise ValueError(
                        f"{bank.modality} current measurements must be a subset of valid tokens"
                    )
                if bank.timestamps is None and not torch.equal(
                    current_measurement_valid,
                    bank.valid,
                ):
                    raise ValueError(
                        f"{bank.modality} non-current evidence requires auditable timestamps"
                    )

        if spec.geometry_dim:
            if bank.geometry is None:
                raise ValueError(f"{bank.modality} requires geometry")
            if bank.geometry.shape != (*bank.tokens.shape[:2], spec.geometry_dim):
                raise ValueError(f"{bank.modality} geometry shape differs from its spec")
            if (
                bank.geometry.device != bank.tokens.device
                or bank.geometry.dtype != bank.tokens.dtype
            ):
                raise ValueError(f"{bank.modality} geometry must match token dtype/device")
            if self.validate_tensor_values:
                if not torch.isfinite(bank.geometry).all():
                    raise ValueError(f"{bank.modality} geometry contains NaN or infinity")
                if (bank.geometry[~bank.valid] != 0.0).any():
                    raise ValueError(
                        f"{bank.modality} invalid geometry padding must be exactly zero"
                    )
        elif bank.geometry is not None:
            raise ValueError(f"{bank.modality} geometry was supplied but not configured")

        if bank.group_id is not None:
            if not spec.require_single_active_group:
                raise ValueError(
                    f"{bank.modality} does not permit runtime group IDs; "
                    "object labels belong only in loss targets"
                )
            if bank.group_id.dtype != torch.long or bank.group_id.shape != bank.valid.shape:
                raise ValueError(f"{bank.modality} group IDs must be long batch-by-token")
            if bank.group_id.device != bank.tokens.device:
                raise ValueError(f"{bank.modality} group IDs must match token device")
            if self.validate_tensor_values and (
                (bank.group_id[~bank.valid] != -1).any() or (bank.group_id < -1).any()
            ):
                raise ValueError(f"{bank.modality} invalid group IDs")
        if spec.require_single_active_group:
            if bank.group_id is None:
                raise ValueError(f"{bank.modality} requires explicit contact group IDs")
            if self.validate_tensor_values:
                for valid, groups in zip(bank.valid, bank.group_id, strict=True):
                    active_groups = torch.unique(groups[valid])
                    if valid.any() and (len(active_groups) != 1 or active_groups[0] < 0):
                        raise ValueError(
                            f"active {bank.modality} tokens must share one nonnegative group ID"
                        )

        if bank.timestamps is not None:
            if (
                bank.timestamps.shape != bank.valid.shape
                or bank.timestamps.device != bank.tokens.device
                or bank.timestamps.dtype not in {torch.float32, torch.float64}
            ):
                raise ValueError(
                    f"{bank.modality} timestamps must be float32/float64 batch-by-token metadata "
                    "on the token device"
                )
            if self.validate_tensor_values:
                if not torch.isfinite(bank.timestamps).all() or (bank.timestamps < 0.0).any():
                    raise ValueError(f"{bank.modality} timestamps must be finite and nonnegative")
                if (bank.timestamps[~bank.valid] != 0.0).any():
                    raise ValueError(f"{bank.modality} invalid timestamp padding must be zero")
                if bank.current_measurement_valid is None:
                    for valid, timestamps in zip(bank.valid, bank.timestamps, strict=True):
                        if valid.any() and not torch.allclose(
                            timestamps[valid],
                            timestamps[valid][0].expand_as(timestamps[valid]),
                            rtol=0.0,
                            atol=1e-7,
                        ):
                            raise ValueError(
                                f"{bank.modality} multi-timestamp evidence requires an explicit "
                                "current measurement role"
                            )
                else:
                    for valid, current, timestamps in zip(
                        bank.valid,
                        bank.current_measurement_valid,
                        bank.timestamps,
                        strict=True,
                    ):
                        if current.any():
                            newest = timestamps[valid].max()
                            if not torch.allclose(
                                timestamps[current],
                                newest.expand_as(timestamps[current]),
                                rtol=0.0,
                                atol=1e-7,
                            ):
                                raise ValueError(
                                    f"{bank.modality} current measurements must use the newest "
                                    "evidence timestamp"
                                )
        if bank.confidence is not None:
            if (
                bank.confidence.shape != bank.valid.shape
                or bank.confidence.device != bank.tokens.device
                or bank.confidence.dtype != bank.tokens.dtype
            ):
                raise ValueError(
                    f"{bank.modality} confidence must match token batch, dtype and device"
                )
            if self.validate_tensor_values:
                if (
                    not torch.isfinite(bank.confidence).all()
                    or (bank.confidence < 0.0).any()
                    or (bank.confidence > 1.0).any()
                ):
                    raise ValueError(f"{bank.modality} confidence must lie in [0, 1]")
                if (bank.confidence[~bank.valid] != 0.0).any():
                    raise ValueError(f"{bank.modality} invalid confidence padding must be zero")
        if bank.encoder_contract is not None and (
            not isinstance(bank.encoder_contract, str) or not bank.encoder_contract.strip()
        ):
            raise ValueError(
                f"{bank.modality} encoder contract must be a nonempty string when supplied"
            )

    def forward(self, banks: tuple[NativeTokenBank, ...]) -> BindingProjectionOutput:
        if not banks:
            raise ValueError("at least one native token bank is required")
        seen: set[str] = set()
        binding_parts = []
        valid_parts = []
        current_measurement_parts = []
        group_parts = []
        modality_parts = []
        spans = []
        batch_size = None
        cursor = 0
        for bank in banks:
            if bank.modality in seen:
                raise ValueError(f"modality {bank.modality} appears more than once")
            seen.add(bank.modality)
            self._validate_bank(bank)
            if batch_size is None:
                batch_size = bank.tokens.shape[0]
            elif bank.tokens.shape[0] != batch_size:
                raise ValueError("all modality banks must share a batch size")
            modality_index = self.modality_to_index[bank.modality]
            projected = self.content_projection[bank.modality](bank.tokens)
            if bank.geometry is not None:
                projected = projected + self.geometry_projection[bank.modality](bank.geometry)
            projected = projected + self.modality_embedding[modality_index]
            projected = self.output_norm[bank.modality](projected)
            projected = projected * bank.valid.unsqueeze(-1)
            binding_parts.append(projected)
            valid_parts.append(bank.valid)
            current_measurement_parts.append(bank.effective_current_measurement_valid)
            if bank.group_id is None:
                groups = torch.full_like(bank.valid, -1, dtype=torch.long)
            else:
                # Group IDs are local to an encoder stream. Namespace them before
                # concatenation so, for example, touch group 0 cannot tie an audio
                # group 0 to the same discovery assignment.
                groups = torch.where(
                    bank.valid,
                    bank.group_id * len(self.specs) + modality_index,
                    -1,
                )
            group_parts.append(groups)
            modality_parts.append(torch.full_like(groups, modality_index))
            stop = cursor + bank.tokens.shape[1]
            spans.append(ModalityTokenSpan(bank.modality, cursor, stop))
            cursor = stop

        return BindingProjectionOutput(
            native_banks=banks,
            binding_features=torch.cat(binding_parts, dim=1),
            token_valid=torch.cat(valid_parts, dim=1),
            current_measurement_valid=torch.cat(current_measurement_parts, dim=1),
            token_group_id=torch.cat(group_parts, dim=1),
            modality_index=torch.cat(modality_parts, dim=1),
            spans=tuple(spans),
        )

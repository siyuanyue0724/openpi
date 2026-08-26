"""Host-neutral learned PICF core composition.

This module contains no task language, training labels, masks, VLA host objects
or mutable hidden state. It composes complete native-token projection, current
set discovery and the explicit persistent probabilistic filter.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .discovery import (
    ObjectDiscoveryConfig,
    ObjectDiscoveryOutput,
    TaskIndependentObjectDiscovery,
)
from .evidence import (
    BindingProjectionOutput,
    ModalityProjectionSpec,
    MultimodalBindingProjector,
    NativeTokenBank,
)
from .filter import (
    ObjectActionBank,
    PersistentFilterOutput,
    PersistentObjectFilter,
    build_object_action_bank,
)
from .temporal import (
    ObjectBeliefBatch,
    TemporalFilterConfig,
    empty_object_belief,
)


@dataclass(frozen=True, slots=True)
class PICFCoreConfig:
    """Single source of truth for one host-neutral PICF core instance.

    ``full`` runtime validation scans tensor values and is mandatory for
    preflight, probes and checkpoint boundaries. ``metadata`` retains shape,
    dtype and device contracts without synchronizing CUDA for repeated finite,
    range or padding scans; it is allowed only after the same data path passes
    full validation and must be frozen in the matched run contract.
    """

    modality_specs: tuple[ModalityProjectionSpec, ...]
    binding_dim: int
    discovery: ObjectDiscoveryConfig
    temporal: TemporalFilterConfig
    posterior_capacity: int
    runtime_validation: str = "full"

    def __post_init__(self) -> None:
        if not self.modality_specs:
            raise ValueError("PICF core requires at least one modality")
        dimensions = (self.binding_dim, self.posterior_capacity)
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in dimensions
        ):
            raise ValueError("binding_dim and posterior_capacity must be positive")
        if self.discovery.input_dim != self.binding_dim:
            raise ValueError("discovery input width must equal the binding width")
        if not isinstance(self.runtime_validation, str) or self.runtime_validation not in {
            "full",
            "metadata",
        }:
            raise ValueError("runtime_validation must be 'full' or 'metadata'")
        if (
            self.discovery.address_dim,
            self.discovery.content_dim,
            self.discovery.geometry_dim,
        ) != (
            self.temporal.address_dim,
            self.temporal.content_dim,
            self.temporal.geometry_dim,
        ):
            raise ValueError("discovery and temporal state dimensions must agree")
        if self.discovery.geometry_contract != self.temporal.geometry_contract:
            raise ValueError("discovery and temporal geometry contracts must agree exactly")

    @property
    def dense_token_dims(self) -> dict[str, int]:
        return {spec.name: spec.token_dim for spec in self.modality_specs}

    @property
    def object_address_dim(self) -> int:
        return self.temporal.address_dim

    @property
    def object_value_dim(self) -> int:
        # predicted dynamic + posterior dynamic + log geometry covariance
        # + dynamic innovation
        # + predicted/posterior existence/visibility lifecycle. Integer step age is diagnostic
        # metadata, not a physical or control-rate-invariant action feature.
        return 3 * self.temporal.dynamic_dim + self.temporal.geometry_dim + 4

    def build(self) -> PICFCore:
        return PICFCore(
            MultimodalBindingProjector(
                self.modality_specs,
                binding_dim=self.binding_dim,
                validate_tensor_values=self.runtime_validation == "full",
            ),
            TaskIndependentObjectDiscovery(
                self.discovery,
                validate_tensor_values=self.runtime_validation == "full",
            ),
            PersistentObjectFilter(
                self.temporal,
                validate_tensor_values=self.runtime_validation == "full",
            ),
        )

    def build_current_frame(self) -> PICFCurrentFrameModel:
        """Build only the deployable current-frame observation model.

        M2 must establish object discovery before temporal association or action
        adoption can receive credit.  The module names intentionally match the
        corresponding ``PICFCore`` submodules so an accepted M2 checkpoint can
        initialize the full core without a translation layer.
        """

        return PICFCurrentFrameModel(
            MultimodalBindingProjector(
                self.modality_specs,
                binding_dim=self.binding_dim,
                validate_tensor_values=self.runtime_validation == "full",
            ),
            TaskIndependentObjectDiscovery(
                self.discovery,
                validate_tensor_values=self.runtime_validation == "full",
            ),
        )

    def empty_belief(
        self,
        *,
        batch_size: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> ObjectBeliefBatch:
        return empty_object_belief(
            self.temporal,
            batch_size=batch_size,
            capacity=self.posterior_capacity,
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True, slots=True)
class PICFCoreOutput:
    projection: BindingProjectionOutput
    discovery: ObjectDiscoveryOutput
    posterior: PersistentFilterOutput
    action_bank: ObjectActionBank

    @property
    def dense_ownership(self) -> tuple[torch.Tensor, ...]:
        """Split persistent object/context ownership back into native banks."""

        return tuple(
            self.posterior.ownership[:, span.start : span.stop] for span in self.projection.spans
        )


@dataclass(frozen=True, slots=True)
class PICFCurrentFrameOutput:
    projection: BindingProjectionOutput
    discovery: ObjectDiscoveryOutput


class PICFCurrentFrameModel(nn.Module):
    """Complete-token projection followed by task-independent set discovery."""

    def __init__(
        self,
        projector: MultimodalBindingProjector,
        discovery: TaskIndependentObjectDiscovery,
    ) -> None:
        super().__init__()
        if projector.binding_dim != discovery.config.input_dim:
            raise ValueError("projector binding width must equal discovery input width")
        if projector.validate_tensor_values != discovery.validate_tensor_values:
            raise ValueError("current-frame components must share one validation policy")
        self.projector = projector
        self.discovery = discovery
        self.validate_tensor_values = projector.validate_tensor_values

    def forward(
        self,
        native_banks: tuple[NativeTokenBank, ...],
    ) -> PICFCurrentFrameOutput:
        projection = self.projector(native_banks)
        discovery = self.discovery(*projection.current_discovery_inputs())
        return PICFCurrentFrameOutput(
            projection=projection,
            discovery=discovery,
        )


class PICFCore(nn.Module):
    """One deployable task-independent multimodal object-posterior model."""

    def __init__(
        self,
        projector: MultimodalBindingProjector,
        discovery: TaskIndependentObjectDiscovery,
        posterior_filter: PersistentObjectFilter,
    ) -> None:
        super().__init__()
        if projector.binding_dim != discovery.config.input_dim:
            raise ValueError("projector binding width must equal discovery input width")
        temporal = posterior_filter.config
        discovered = discovery.config
        if (
            temporal.address_dim,
            temporal.content_dim,
            temporal.geometry_dim,
        ) != (
            discovered.address_dim,
            discovered.content_dim,
            discovered.geometry_dim,
        ):
            raise ValueError("discovery state dimensions must equal temporal filter dimensions")
        if discovered.geometry_contract != temporal.geometry_contract:
            raise ValueError("discovery and temporal geometry contracts must agree exactly")
        validation_modes = {
            projector.validate_tensor_values,
            discovery.validate_tensor_values,
            posterior_filter.validate_tensor_values,
        }
        if len(validation_modes) != 1:
            raise ValueError("PICF core components must share one runtime validation policy")
        self.projector = projector
        self.discovery = discovery
        self.posterior_filter = posterior_filter
        self.validate_tensor_values = validation_modes.pop()

    def forward(
        self,
        native_banks: tuple[NativeTokenBank, ...],
        prior: ObjectBeliefBatch,
        previous_executed_action: torch.Tensor,
        delta_t_s: torch.Tensor,
    ) -> PICFCoreOutput:
        projection = self.projector(native_banks)
        discovery = self.discovery(*projection.current_discovery_inputs())
        posterior = self.posterior_filter(
            prior,
            discovery,
            previous_executed_action,
            delta_t_s,
        )
        return PICFCoreOutput(
            projection=projection,
            discovery=discovery,
            posterior=posterior,
            action_bank=build_object_action_bank(posterior),
        )

"""Deep LingBot integration for the strict ADR-74 production graph."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import overload

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.lingbot_native.action_posterior_receipt import (
    LingBotActionAttentionLayout,
    build_lingbot_action_attention_layout,
)
from picf_next.lingbot_native.addresses import (
    EpisodeAddressState,
    fixed_orthogonal_address_codebook,
)
from picf_next.lingbot_native.controls import ExecutedControlBatch
from picf_next.lingbot_native.graph import (
    NativeRole,
    NativeTokenLayout,
    expand_square_mask,
    native_attention_mask,
    native_layerwise_history_mask,
    native_layerwise_prior_history_mask,
    native_layerwise_prior_trace_mask,
    native_role_permissions,
    posterior_adoption_action_key_visibility,
    posterior_adoption_attention_mask,
)
from picf_next.lingbot_native.modalities import (
    TOKEN_LAYER_NORM,
    NativeModalityBatch,
    NativeModalitySpec,
    NativeObjectQuerySpatialSpec,
    NativeRelationSurfaceSpec,
    initialize_column_isometry,
    modality_bridge_input,
    validate_modality_specs,
    validate_object_query_spatial_specs,
    validate_relation_surface_specs,
)
from picf_next.lingbot_native.physical_relations import (
    NATIVE_OBJECT_QUERY_POSTERIOR_INTERFACE,
    TASK_INDEPENDENT_PHYSICAL_INTERFACE,
    ContextualObjectQuerySpatialInput,
    NativeObjectQueryPosteriorOutput,
    PhysicalEntityReadout,
    PhysicalRelationOutput,
    PhysicalRelationSurfaceInput,
)
from picf_next.lingbot_native.prediction import (
    NativePredictionRequest,
    PredictionEvidence,
    PredictionSource,
)
from picf_next.lingbot_native.predictive_objective import NativePredictiveReadout
from picf_next.lingbot_native.relations import (
    HOST_NATIVE_MATCH_INTERFACE,
    RelationOutput,
    SharedRelationReadout,
)
from picf_next.lingbot_native.state import (
    AddressedLayerwisePosteriorState,
    AddressedLayerwisePriorTrace,
    NativeLayerwisePosteriorState,
    NativeLayerwisePriorTrace,
    NativePosteriorState,
)
from picf_next.lingbot_native.task_address_graph import (
    TaskAddressActionInformationSet,
    TaskAddressRole,
    TaskAddressStateSlices,
    TaskAddressTokenLayout,
    normalize_task_address_action_information_sets,
    task_address_action_key_visibility,
    task_address_attention_mask,
    task_address_layerwise_state_mask,
)
from picf_next.lingbot_native.task_address_receipt import TaskAddressAttentionLayout

CONTENT_ADDRESSED_SET_TRANSITION = "content_addressed_set_v1"
LEGACY_TASK_MATCH_ARCHITECTURE = "content_addressed_task_match_v1"
TASK_INDEPENDENT_ENTITY_POSTERIOR = "task_independent_entity_posterior_v1"
LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR = "task_independent_entity_posterior_v2"
UNIFIED_LAYERWISE_PREDICT_CORRECT = "task_independent_predict_correct_v3"
LINGBOT_TASK_QUERY_OBJECT_VALUE_READ = "lingbot_task_query_object_value_read_v1"
NATIVE_VIDEOMT_QUERY_POSTERIOR = "native_videomt_query_posterior_v1"
NATIVE_VIDEOMT_QUERY_COUNT = 200
EXACT_NATIVE_MODALITY_BRIDGE = "exact_tokens_v1"
LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE = "lingbot_task_token_resampler_v1"
NATIVE_MODALITY_BRIDGES = (
    EXACT_NATIVE_MODALITY_BRIDGE,
    LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE,
)
NATIVE_GRAPH_ARCHITECTURES = (
    LEGACY_TASK_MATCH_ARCHITECTURE,
    TASK_INDEPENDENT_ENTITY_POSTERIOR,
    LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
    UNIFIED_LAYERWISE_PREDICT_CORRECT,
    LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
)


class ObjectReadActionIntervention(str, Enum):
    """Evaluation-only intervention on the sole PICF-to-action Value edge."""

    FACTUAL = "factual"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class LingBotNativeGraphConfig:
    capacity: int
    host_width: int
    executed_action_dim: int
    num_layers: int = 36
    maximum_control_tokens: int = 8
    initializer_range: float = 0.02
    relation_temperature: float = 0.07
    prediction_route_count: int = 1
    task_query_count: int = 0
    prediction_address_width: int = 0
    predictive_target_widths: tuple[tuple[str, int], ...] = ()
    modality_specs: tuple[NativeModalitySpec, ...] = ()
    modality_bridge_identity: str = EXACT_NATIVE_MODALITY_BRIDGE
    modality_bridge_query_count: int = 0
    resampled_modality_names: tuple[str, ...] = ()
    direct_action_modality_names: tuple[str, ...] = ()
    relation_surface_specs: tuple[NativeRelationSurfaceSpec, ...] = ()
    object_query_spatial_specs: tuple[NativeObjectQuerySpatialSpec, ...] = ()
    relation_supervision_layers: tuple[int, ...] = ()
    object_transition: str = CONTENT_ADDRESSED_SET_TRANSITION
    architecture_identity: str = LEGACY_TASK_MATCH_ARCHITECTURE

    def __post_init__(self) -> None:
        integer_values = (
            self.capacity,
            self.host_width,
            self.executed_action_dim,
            self.num_layers,
            self.maximum_control_tokens,
            self.prediction_route_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_values):
            raise TypeError("LingBot-native graph dimensions must be integers")
        if min(integer_values) <= 0:
            raise ValueError("LingBot-native graph dimensions must be positive")
        if self.num_layers < 2:
            raise ValueError("LingBot-native PICF requires at least two shared layers")
        if self.object_transition != CONTENT_ADDRESSED_SET_TRANSITION:
            raise ValueError("LingBot-native object transition is unsupported")
        if self.architecture_identity not in NATIVE_GRAPH_ARCHITECTURES:
            raise ValueError("LingBot-native architecture identity is unsupported")
        if (
            isinstance(self.task_query_count, bool)
            or not isinstance(self.task_query_count, int)
            or self.task_query_count < 0
        ):
            raise ValueError("task query count must be a non-negative integer")
        if self.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ:
            if self.task_query_count <= 0:
                raise ValueError("task-addressed architecture requires task query rows")
        elif self.task_query_count:
            raise ValueError("historical architectures cannot declare task query rows")
        if not isinstance(self.relation_supervision_layers, tuple):
            raise TypeError("relation supervision layers must be an immutable tuple")
        if any(
            isinstance(layer, bool) or not isinstance(layer, int)
            for layer in self.relation_supervision_layers
        ):
            raise TypeError("relation supervision layers must contain integers")
        if tuple(sorted(set(self.relation_supervision_layers))) != self.relation_supervision_layers:
            raise ValueError("relation supervision layers must be sorted and unique")
        if any(
            layer < 0 or layer >= self.num_layers - 1 for layer in self.relation_supervision_layers
        ):
            raise ValueError("relation supervision layers must be non-final host layer indices")
        if (
            isinstance(self.prediction_address_width, bool)
            or not isinstance(self.prediction_address_width, int)
            or self.prediction_address_width < 0
        ):
            raise ValueError("prediction address width must be a non-negative integer")
        if not isinstance(self.predictive_target_widths, tuple):
            raise TypeError("predictive target widths must be an immutable tuple")
        names: list[str] = []
        for value in self.predictive_target_widths:
            if not isinstance(value, tuple) or len(value) != 2:
                raise TypeError("each predictive target width must be one (name, width) pair")
            name, width = value
            if (
                not isinstance(name, str)
                or not name
                or name != name.lower()
                or any(
                    character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in name
                )
            ):
                raise ValueError(
                    "predictive target names must be lowercase module-safe identifiers"
                )
            if isinstance(width, bool) or not isinstance(width, int) or width < 2:
                raise ValueError("predictive target widths must be integers of at least two")
            names.append(name)
        if names != sorted(names) or len(set(names)) != len(names):
            raise ValueError("predictive target widths must be sorted with unique names")
        validate_modality_specs(self.modality_specs)
        validate_relation_surface_specs(
            self.relation_surface_specs,
            modality_specs=self.modality_specs,
        )
        if self.modality_bridge_identity not in NATIVE_MODALITY_BRIDGES:
            raise ValueError("native modality bridge identity is unsupported")
        if (
            isinstance(self.modality_bridge_query_count, bool)
            or not isinstance(self.modality_bridge_query_count, int)
            or self.modality_bridge_query_count < 0
        ):
            raise ValueError("native modality bridge query count must be a non-negative integer")
        if (
            not isinstance(self.resampled_modality_names, tuple)
            or tuple(sorted(set(self.resampled_modality_names))) != self.resampled_modality_names
        ):
            raise ValueError("resampled modality names must be a sorted unique tuple")
        declared_modalities = {spec.name for spec in self.modality_specs}
        if not set(self.resampled_modality_names) <= declared_modalities:
            raise ValueError("resampled modality names must be declared native modalities")
        validate_object_query_spatial_specs(
            self.object_query_spatial_specs,
            modality_specs=self.modality_specs,
            resampled_modality_names=self.resampled_modality_names,
        )
        if self.architecture_identity == NATIVE_VIDEOMT_QUERY_POSTERIOR:
            if self.capacity != NATIVE_VIDEOMT_QUERY_COUNT:
                raise ValueError("native VidEoMT posterior requires all 200 source queries")
            if self.modality_bridge_identity != EXACT_NATIVE_MODALITY_BRIDGE:
                raise ValueError("native VidEoMT posterior forbids query resampling")
            if len(self.object_query_spatial_specs) != 1:
                raise ValueError("native VidEoMT posterior requires one complete spatial relation")
            source_name = self.object_query_spatial_specs[0].query_modality
            source_specs = tuple(spec for spec in self.modality_specs if spec.name == source_name)
            if (
                len(source_specs) != 1
                or source_specs[0].maximum_tokens != NATIVE_VIDEOMT_QUERY_COUNT
            ):
                raise ValueError("native VidEoMT posterior requires one exact 200-query stream")
            if self.relation_supervision_layers:
                raise ValueError(
                    "native VidEoMT posterior uses complete source auxiliary supervision, "
                    "not a second host relation head"
                )
        if not set(self.direct_action_modality_names) <= declared_modalities:
            raise ValueError("direct action modality names must be declared native modalities")
        if set(self.direct_action_modality_names) & set(self.resampled_modality_names):
            raise ValueError("direct action modalities cannot enter a mixed resampling bridge")
        if self.modality_bridge_identity == EXACT_NATIVE_MODALITY_BRIDGE:
            if self.modality_bridge_query_count or self.resampled_modality_names:
                raise ValueError("exact-token modality bridge cannot declare resampled rows")
        elif self.modality_bridge_query_count <= 0 or not self.resampled_modality_names:
            raise ValueError("LingBot query resampling requires queries and source modalities")
        for spec in self.modality_specs:
            if spec.input_width > self.host_width:
                raise ValueError(
                    f"modality {spec.name!r} would compress before the shared LingBot host"
                )
            if spec.metadata_width > self.host_width:
                raise ValueError(
                    f"modality {spec.name!r} metadata would compress before the shared host"
                )
        real_values = (self.initializer_range, self.relation_temperature)
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
            for value in real_values
        ):
            raise ValueError("LingBot-native initialization controls must be finite and positive")

    @classmethod
    def from_policy(
        cls,
        policy: nn.Module,
        *,
        capacity: int,
        maximum_control_tokens: int = 8,
        prediction_route_count: int = 1,
        task_query_count: int = 0,
        prediction_address_width: int = 0,
        predictive_target_widths: tuple[tuple[str, int], ...] = (),
        modality_specs: tuple[NativeModalitySpec, ...] = (),
        modality_bridge_identity: str = EXACT_NATIVE_MODALITY_BRIDGE,
        modality_bridge_query_count: int = 0,
        resampled_modality_names: tuple[str, ...] = (),
        direct_action_modality_names: tuple[str, ...] = (),
        relation_surface_specs: tuple[NativeRelationSurfaceSpec, ...] = (),
        object_query_spatial_specs: tuple[NativeObjectQuerySpatialSpec, ...] = (),
        relation_supervision_layers: tuple[int, ...] = (),
        architecture_identity: str = LEGACY_TASK_MATCH_ARCHITECTURE,
    ) -> LingBotNativeGraphConfig:
        model = getattr(policy, "model", None)
        host = getattr(model, "qwenvl_with_expert", None)
        if model is None or host is None:
            raise TypeError("policy does not expose the official LingBot VLA2 joint host")
        try:
            language_model = host.qwenvl.model.language_model
            action_model = host.qwen_expert.model
            host_width = int(language_model.layers[0].hidden_size)
            num_layers = len(language_model.layers)
            action_layers = len(action_model.layers)
            action_dim = int(model.config.max_action_dim)
            initializer_range = float(language_model.config.initializer_range)
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise TypeError("policy has an incomplete LingBot VLA2 host contract") from error
        if num_layers != action_layers:
            raise ValueError("LingBot language and action streams have different depths")
        return cls(
            capacity=capacity,
            host_width=host_width,
            executed_action_dim=action_dim,
            num_layers=num_layers,
            maximum_control_tokens=maximum_control_tokens,
            initializer_range=initializer_range,
            prediction_route_count=prediction_route_count,
            task_query_count=task_query_count,
            prediction_address_width=prediction_address_width,
            predictive_target_widths=predictive_target_widths,
            modality_specs=modality_specs,
            modality_bridge_identity=modality_bridge_identity,
            modality_bridge_query_count=modality_bridge_query_count,
            resampled_modality_names=resampled_modality_names,
            direct_action_modality_names=direct_action_modality_names,
            relation_surface_specs=relation_surface_specs,
            object_query_spatial_specs=object_query_spatial_specs,
            relation_supervision_layers=relation_supervision_layers,
            architecture_identity=architecture_identity,
        )


@dataclass(slots=True)
class LingBotNativeContext:
    """Deploy-visible inputs and write-once outputs for one observation event."""

    controls: ExecutedControlBatch
    previous_state: NativePosteriorState | None = None
    previous_state_valid: torch.Tensor | None = None
    previous_memory: NativeLayerwisePosteriorState | None = None
    previous_memory_valid: torch.Tensor | None = None
    prior_trace: NativeLayerwisePriorTrace | None = None
    prediction_request: NativePredictionRequest | None = None
    modalities: NativeModalityBatch | None = None
    posterior_adoption_route: torch.Tensor | None = None
    # Diagnostic-only cache intervention; training and deployment leave this unset.
    posterior_action_row_visible: torch.Tensor | None = None
    supervise_intermediate_relations: bool = False
    object_read_action_intervention: ObjectReadActionIntervention = (
        ObjectReadActionIntervention.FACTUAL
    )
    action_information_sets: tuple[TaskAddressActionInformationSet, ...] = ()
    object_read_source_row_visible: torch.Tensor | None = None
    native_roles: torch.Tensor | None = None
    native_valid: torch.Tensor | None = None
    native_sensor_boundary: torch.Tensor | None = None
    native_host_current: torch.Tensor | None = None
    native_host_future: torch.Tensor | None = None
    instruction_last_index: torch.Tensor | None = None
    episode_address_state: EpisodeAddressState | None = None
    prior_state: NativePosteriorState | None = field(init=False, default=None)
    posterior_state: NativePosteriorState | None = field(init=False, default=None)
    posterior_memory: NativeLayerwisePosteriorState | None = field(init=False, default=None)
    relation_output: (
        RelationOutput | PhysicalRelationOutput | NativeObjectQueryPosteriorOutput | None
    ) = field(
        init=False,
        default=None,
    )
    intermediate_relation_outputs: Mapping[
        int, RelationOutput | PhysicalRelationOutput | NativeObjectQueryPosteriorOutput
    ] = field(
        init=False,
        default_factory=lambda: MappingProxyType({}),
    )
    prediction_hidden: torch.Tensor | None = field(init=False, default=None)
    prediction_outputs: Mapping[str, torch.Tensor] = field(
        init=False,
        default_factory=lambda: MappingProxyType({}),
    )
    expanded_cache_valid: torch.Tensor | None = field(init=False, default=None)
    expanded_cache_position_ids: torch.Tensor | None = field(init=False, default=None)
    expanded_action_cache_visible: torch.Tensor | None = field(init=False, default=None)
    expanded_posterior_indices: torch.Tensor | None = field(init=False, default=None)
    expanded_posterior_valid: torch.Tensor | None = field(init=False, default=None)
    task_address_attention_layout: TaskAddressAttentionLayout | None = field(
        init=False,
        default=None,
    )
    _finalized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.supervise_intermediate_relations, bool):
            raise TypeError("intermediate relation supervision flag must be boolean")
        if not isinstance(
            self.object_read_action_intervention,
            ObjectReadActionIntervention,
        ):
            raise TypeError("object-read action intervention must use its typed enum")
        self.action_information_sets = normalize_task_address_action_information_sets(
            self.action_information_sets or None,
            batch_size=self.controls.batch_size,
        )
        if self.previous_state is not None and not isinstance(
            self.previous_state, NativePosteriorState
        ):
            raise TypeError("previous_state must use the final-row posterior schema")
        if self.previous_memory is not None and not isinstance(
            self.previous_memory, NativeLayerwisePosteriorState
        ):
            raise TypeError("previous_memory must use the persistent layerwise posterior schema")
        if self.prior_trace is not None and not isinstance(
            self.prior_trace, NativeLayerwisePriorTrace
        ):
            raise TypeError("prior_trace must use the transient v3 prior schema")
        metadata = (self.native_roles, self.native_valid, self.instruction_last_index)
        if any(value is None for value in metadata):
            if not all(value is None for value in metadata):
                raise ValueError("native prefix metadata must be entirely bound or unbound")
            batch = self.controls.batch_size
        else:
            self._validate_bound_prefix()
            if self.native_roles is None:
                raise RuntimeError("bound native prefix omitted role metadata")
            batch = self.native_roles.shape[0]
        if self.controls.batch_size != batch:
            raise ValueError("native prefix and executed controls have different batch sizes")
        source_row_visible = self.object_read_source_row_visible
        if source_row_visible is not None:
            if not isinstance(source_row_visible, torch.Tensor):
                raise TypeError("object-read source-row visibility must be a tensor")
            if source_row_visible.ndim != 2 or source_row_visible.shape[0] != batch:
                raise ValueError(
                    "object-read source-row visibility must have shape [batch, capacity]"
                )
            if source_row_visible.dtype != torch.bool:
                raise TypeError("object-read source-row visibility must be boolean")
            if source_row_visible.device != self.controls.values.device:
                raise ValueError(
                    "object-read source-row visibility and controls must share one device"
                )
        if self.episode_address_state is not None:
            if not isinstance(self.episode_address_state, EpisodeAddressState):
                raise TypeError("native context address state must use EpisodeAddressState")
            if self.episode_address_state.batch_size != batch:
                raise ValueError("native context and address state have different batches")
            if self.episode_address_state.device != self.controls.values.device:
                raise ValueError("native context address state and controls must share one device")
        if self.modalities is not None:
            if not isinstance(self.modalities, NativeModalityBatch):
                raise TypeError("native context modalities must use the typed batch contract")
            if self.modalities.batch_size != batch:
                raise ValueError("native modalities and executed controls have different batches")
            if self.modalities.device != self.controls.values.device:
                raise ValueError("native modalities and controls must share one device")
        if self.posterior_adoption_route is not None:
            if (
                self.posterior_adoption_route.shape != (batch,)
                or self.posterior_adoption_route.dtype != torch.bool
                or self.posterior_adoption_route.device != self.controls.values.device
            ):
                raise ValueError("posterior-adoption route must be boolean [batch]")
            if self.posterior_adoption_route.any() and self.prior_trace is None:
                raise ValueError("posterior-adoption routing requires a completed v3 prior trace")
        posterior_action_row_visible = self.posterior_action_row_visible
        if posterior_action_row_visible is not None:
            if (
                posterior_action_row_visible.ndim != 2
                or posterior_action_row_visible.shape[0] != batch
                or posterior_action_row_visible.dtype != torch.bool
                or posterior_action_row_visible.device != self.controls.values.device
            ):
                raise ValueError(
                    "posterior action-row visibility must be boolean [batch, capacity]"
                )
            if self.posterior_adoption_route is None or not bool(
                self.posterior_adoption_route.all()
            ):
                raise ValueError(
                    "posterior action-row intervention requires every sample on the direct route"
                )
        if self.previous_state_valid is None:
            state_device = (
                self.controls.values.device
                if self.previous_state is None
                else self.previous_state.rows.device
            )
            self.previous_state_valid = torch.full(
                (batch,),
                self.previous_state is not None,
                dtype=torch.bool,
                device=state_device,
            )
        if (
            self.previous_state_valid.shape != (batch,)
            or self.previous_state_valid.dtype != torch.bool
            or self.previous_state_valid.device != self.controls.values.device
        ):
            raise ValueError("previous_state_valid must be boolean with one value per sample")
        if self.previous_state is None and self.previous_state_valid.any():
            raise ValueError("previous_state_valid cannot be true without previous rows")
        recurrent_inputs = (self.previous_state, self.previous_memory, self.prior_trace)
        if sum(value is not None for value in recurrent_inputs) > 1:
            raise ValueError(
                "legacy rows, historical layerwise memory and a produced prior trace are "
                "mutually exclusive"
            )
        if self.previous_memory_valid is None:
            memory_device = (
                self.controls.values.device
                if self.previous_memory is None
                else self.previous_memory.layer_rows.device
            )
            self.previous_memory_valid = torch.full(
                (batch,),
                self.previous_memory is not None,
                dtype=torch.bool,
                device=memory_device,
            )
        if (
            self.previous_memory_valid.shape != (batch,)
            or self.previous_memory_valid.dtype != torch.bool
            or self.previous_memory_valid.device != self.controls.values.device
        ):
            raise ValueError("previous_memory_valid must be boolean with one value per sample")
        if self.previous_memory is None and self.previous_memory_valid.any():
            raise ValueError("previous_memory_valid cannot be true without layerwise memory")
        if self.prior_trace is not None:
            if self.prior_trace.batch_size != batch:
                raise ValueError("prior trace and executed controls have different batches")
            if self.prior_trace.layer_rows.device != self.controls.values.device:
                raise ValueError("prior trace and executed controls must share one device")
        if self.prediction_request is not None:
            request = self.prediction_request
            if request.evidence is PredictionEvidence.FUTURE:
                raise ValueError("future prediction requires the row-only prior context")
            if request.batch_size != batch:
                raise ValueError("prediction request and native prefix batches differ")
            if request.route_ids.device != self.controls.values.device:
                raise ValueError("prediction request and controls must share one device")

    def _validate_bound_prefix(self) -> None:
        if (
            self.native_roles is None
            or self.native_valid is None
            or self.instruction_last_index is None
        ):
            raise ValueError("native prefix metadata is not fully bound")
        if self.native_roles.ndim != 2 or self.native_valid.shape != self.native_roles.shape:
            raise ValueError("native role metadata must have shape [batch, prefix_tokens]")
        if self.native_roles.dtype != torch.long or self.native_valid.dtype != torch.bool:
            raise TypeError("native roles must be long and validity must be boolean")
        if self.native_roles.device != self.native_valid.device:
            raise ValueError("native roles and validity must share one device")
        for name, host_mask in (
            ("sensor-boundary", self.native_sensor_boundary),
            ("current", self.native_host_current),
            ("future", self.native_host_future),
        ):
            if host_mask is None:
                continue
            if (
                host_mask.shape != self.native_roles.shape
                or host_mask.dtype != torch.bool
                or host_mask.device != self.native_roles.device
            ):
                raise ValueError(f"native host-{name} mask must match prefix metadata")
            if (host_mask & (self.native_roles != int(NativeRole.HOST_AUX))).any():
                raise ValueError(f"native host-{name} mask selects a non-aux token")
        classification_masks = tuple(
            mask
            for mask in (
                self.native_sensor_boundary,
                self.native_host_current,
                self.native_host_future,
            )
            if mask is not None
        )
        for index, first in enumerate(classification_masks):
            for second in classification_masks[index + 1 :]:
                if (first & second).any():
                    raise ValueError("native auxiliary classification masks overlap")
        allowed_native = torch.tensor(
            [int(NativeRole.SENSOR), int(NativeRole.LANGUAGE), int(NativeRole.HOST_AUX)],
            device=self.native_roles.device,
        )
        if not torch.isin(self.native_roles, allowed_native).all():
            raise ValueError("native prefix metadata contains a non-prefix role")
        batch = self.native_roles.shape[0]
        if (
            self.instruction_last_index.shape != (batch,)
            or self.instruction_last_index.dtype != torch.long
        ):
            raise ValueError("instruction_last_index must contain one long index per sample")
        if self.instruction_last_index.device != self.native_roles.device:
            raise ValueError("instruction indexes and native roles must share one device")
        if (self.instruction_last_index < 0).any() or (
            self.instruction_last_index >= self.native_roles.shape[1]
        ).any():
            raise ValueError("instruction_last_index is outside the native prefix")
        batch_index = torch.arange(batch, device=self.native_roles.device)
        if not (
            self.native_valid[batch_index, self.instruction_last_index]
            & (
                self.native_roles[batch_index, self.instruction_last_index]
                == int(NativeRole.LANGUAGE)
            )
        ).all():
            raise ValueError("instruction_last_index must select a valid language token")

    def bind_native_prefix(
        self,
        *,
        native_valid: torch.Tensor,
        visual_sensor_mask: torch.Tensor,
        visual_boundary_mask: torch.Tensor | None = None,
        language_start: int,
        language_valid: torch.Tensor,
        host_current_mask: torch.Tensor | None = None,
        host_future_mask: torch.Tensor | None = None,
    ) -> None:
        """Bind roles once, after the official host has built its exact prefix."""

        if any(
            value is not None
            for value in (self.native_roles, self.native_valid, self.instruction_last_index)
        ):
            raise RuntimeError("native prefix metadata may be bound only once")
        if native_valid.ndim != 2 or native_valid.dtype != torch.bool:
            raise ValueError("native prefix validity must be boolean [batch,prefix]")
        if visual_sensor_mask.shape != native_valid.shape or visual_sensor_mask.dtype != torch.bool:
            raise ValueError("visual sensor mask must be boolean and match the native prefix")
        if language_valid.ndim != 2 or language_valid.dtype != torch.bool:
            raise ValueError("language validity must be boolean [batch,language_tokens]")
        if language_valid.shape[0] != native_valid.shape[0]:
            raise ValueError("language and native prefix batches differ")
        language_stop = language_start + language_valid.shape[1]
        if not 0 <= language_start < language_stop <= native_valid.shape[1]:
            raise ValueError("language span lies outside the official native prefix")
        if visual_sensor_mask[:, language_start:language_stop].any():
            raise ValueError("official visual and language prefix spans overlap")
        if not language_valid.any(dim=1).all():
            raise ValueError("every sample requires at least one valid instruction token")
        if not torch.equal(
            native_valid[:, language_start:language_stop] & language_valid,
            language_valid,
        ):
            raise ValueError("valid language tokens must also be valid native prefix tokens")
        roles = torch.full_like(native_valid, int(NativeRole.HOST_AUX), dtype=torch.long)
        roles[visual_sensor_mask] = int(NativeRole.SENSOR)
        roles[:, language_start:language_stop] = int(NativeRole.LANGUAGE)
        reverse = language_valid.flip(dims=(1,)).to(torch.int64).argmax(dim=1)
        instruction_last = language_start + language_valid.shape[1] - 1 - reverse
        self.native_roles = roles
        self.native_valid = native_valid
        self.native_sensor_boundary = visual_boundary_mask
        self.native_host_current = host_current_mask
        self.native_host_future = host_future_mask
        self.instruction_last_index = instruction_last
        self._validate_bound_prefix()

    def root_output_tensors(self) -> tuple[torch.Tensor, ...]:
        """Expose every floating PICF output consumed outside the policy root."""

        if (
            not self._finalized
            or self.prior_state is None
            or self.posterior_state is None
            or self.relation_output is None
        ):
            raise RuntimeError("native context outputs are incomplete or not finalized")
        relation = self.relation_output
        if isinstance(relation, NativeObjectQueryPosteriorOutput):
            if relation.interface != NATIVE_OBJECT_QUERY_POSTERIOR_INTERFACE:
                raise RuntimeError("native policy root received an unknown query interface")
            class_logits = relation.relation.class_logits
            if class_logits is None:
                raise RuntimeError("native query posterior lost complete source class logits")
            outputs = (
                self.prior_state.rows,
                self.posterior_state.rows,
                relation.posterior_rows,
                relation.relation.object_logits,
                class_logits,
                relation.relation.mask_logits,
            )
            if self.posterior_memory is not None:
                outputs += (self.posterior_memory.layer_rows,)
            if self.intermediate_relation_outputs:
                raise RuntimeError("native query posterior forbids a second intermediate readout")
            if self.prediction_hidden is not None:
                outputs += (self.prediction_hidden,)
            outputs += tuple(
                self.prediction_outputs[name] for name in sorted(self.prediction_outputs)
            )
            if any(not value.is_floating_point() for value in outputs):
                raise TypeError("native query posterior root outputs must be floating tensors")
            return outputs
        if isinstance(relation, PhysicalRelationOutput):
            if relation.interface != TASK_INDEPENDENT_PHYSICAL_INTERFACE:
                raise RuntimeError("native policy root received an unknown physical interface")
            outputs = (
                self.prior_state.rows,
                self.posterior_state.rows,
                relation.support_logits,
                relation.visible_support,
                relation.ownership,
                relation.ownership_log_probability,
                relation.existence,
                relation.existence_logits,
                relation.row_embeddings,
                relation.relation_temperature,
            )
            for surface in relation.relation_surfaces:
                outputs += (
                    surface.support_logits,
                    surface.ownership,
                    surface.ownership_log_probability,
                )
                if surface.donor_query_probability is not None:
                    if (
                        surface.donor_context_probability is None
                        or surface.contextual_query_ownership is None
                    ):
                        raise RuntimeError("native object-query decomposition is incomplete")
                    outputs += (
                        surface.donor_query_probability,
                        surface.donor_context_probability,
                        surface.contextual_query_ownership,
                    )
            if self.posterior_memory is not None:
                outputs += (self.posterior_memory.layer_rows,)
            for layer in sorted(self.intermediate_relation_outputs):
                intermediate = self.intermediate_relation_outputs[layer]
                if not isinstance(intermediate, PhysicalRelationOutput):
                    raise RuntimeError("task-independent graph exposed a legacy intermediate")
                outputs += (
                    intermediate.ownership,
                    intermediate.ownership_log_probability,
                )
                for surface in intermediate.relation_surfaces:
                    outputs += (
                        surface.support_logits,
                        surface.ownership,
                        surface.ownership_log_probability,
                    )
                    if surface.donor_query_probability is not None:
                        if (
                            surface.donor_context_probability is None
                            or surface.contextual_query_ownership is None
                        ):
                            raise RuntimeError(
                                "intermediate object-query decomposition is incomplete"
                            )
                        outputs += (
                            surface.donor_query_probability,
                            surface.donor_context_probability,
                            surface.contextual_query_ownership,
                        )
            if self.prediction_hidden is not None:
                outputs += (self.prediction_hidden,)
            outputs += tuple(
                self.prediction_outputs[name] for name in sorted(self.prediction_outputs)
            )
            if any(not value.is_floating_point() for value in outputs):
                raise TypeError("native policy root outputs must be floating tensors")
            return outputs
        if (
            not isinstance(relation, RelationOutput)
            or relation.task_interface != HOST_NATIVE_MATCH_INTERFACE
            or relation.match_embeddings is None
            or relation.task_embedding is not None
            or relation.task_object_log_probability is None
            or relation.task_object_probability is None
            or relation.task_event_distribution is None
            or relation.task_row_probability is None
            or relation.ownership_log_probability is None
        ):
            raise RuntimeError("native policy root received a non-match task interface")
        outputs = (
            self.prior_state.rows,
            self.posterior_state.rows,
            relation.support_logits,
            relation.visible_support,
            relation.ownership,
            relation.task_relevance,
            relation.task_relevance_logits,
            relation.match_embeddings,
            relation.row_embeddings,
            relation.relation_temperature,
            relation.dense_task_grounding,
            relation.dense_task_grounding_logits,
            relation.existence,
            relation.existence_logits,
            relation.task_object_log_probability,
            relation.task_object_probability,
            relation.task_event_distribution,
            relation.task_row_probability,
            relation.ownership_log_probability,
        )
        for layer in sorted(self.intermediate_relation_outputs):
            intermediate = self.intermediate_relation_outputs[layer]
            if intermediate.ownership_log_probability is None:
                raise RuntimeError("intermediate relation lacks attached ownership log probability")
            outputs += (intermediate.ownership, intermediate.ownership_log_probability)
        if self.prediction_hidden is not None:
            outputs += (self.prediction_hidden,)
        outputs += tuple(self.prediction_outputs[name] for name in sorted(self.prediction_outputs))
        if any(not value.is_floating_point() for value in outputs):
            raise TypeError("native policy root outputs must be floating tensors")
        return outputs


@dataclass(frozen=True, slots=True)
class LingBotActionCache:
    """Exact released prefix cache plus explicitly authorized PICF Value rows."""

    past_key_values: dict[int, dict[str, torch.Tensor]]
    valid: torch.Tensor
    position_ids: torch.Tensor
    position_valid: torch.Tensor
    selected_inserted_indices: torch.Tensor
    action_attention_layout: LingBotActionAttentionLayout | None = None


def compact_lingbot_action_cache(
    *,
    native_past_key_values: dict[int, dict[str, torch.Tensor]],
    expanded_past_key_values: dict[int, dict[str, torch.Tensor]],
    native_valid: torch.Tensor,
    native_position_ids: torch.Tensor,
    context: LingBotNativeContext,
    suffix_count: int | None = None,
    action_query_count: int | None = None,
) -> LingBotActionCache:
    """Keep native K/V bit-exact and expose only action-authorized cache rows.

    The native cache is produced by the released prefix shape.  PICF rows are
    produced by a separate pass through the same host.  This function never
    copies an expanded native-prefix K/V into the action cache, so disabling
    the sole PICF-to-action edge is exactly the released action interface.
    """

    if not isinstance(context, LingBotNativeContext):
        raise TypeError("action-cache compaction requires a LingBotNativeContext")
    if native_valid.ndim != 2 or native_valid.dtype != torch.bool:
        raise ValueError("native action-cache validity must be boolean [batch,tokens]")
    if native_position_ids.shape != (3, native_valid.shape[0], native_valid.shape[1]):
        raise ValueError("native action-cache MRoPE positions have the wrong shape")
    if context.native_valid is None or not torch.equal(context.native_valid, native_valid):
        raise ValueError("context and released action cache disagree on native validity")
    expanded_valid = context.expanded_cache_valid
    expanded_positions = context.expanded_cache_position_ids
    action_visible = context.expanded_action_cache_visible
    if any(value is None for value in (expanded_valid, expanded_positions, action_visible)):
        raise RuntimeError("strict PICF did not publish expanded action-cache metadata")
    if expanded_valid is None or expanded_positions is None or action_visible is None:
        raise AssertionError("unreachable optional action-cache metadata")
    native_count = native_valid.shape[1]
    if (
        expanded_valid.ndim != 2
        or expanded_valid.dtype != torch.bool
        or action_visible.shape != expanded_valid.shape
        or action_visible.dtype != torch.bool
        or expanded_positions.shape != (3, expanded_valid.shape[0], expanded_valid.shape[1])
        or expanded_valid.shape[0] != native_valid.shape[0]
        or expanded_valid.shape[1] < native_count
        or expanded_valid.device != native_valid.device
        or action_visible.device != native_valid.device
        or expanded_positions.device != native_position_ids.device
    ):
        raise ValueError("expanded action-cache metadata has an invalid tensor contract")
    if not torch.equal(expanded_valid[:, :native_count], native_valid) or not torch.equal(
        expanded_positions[:, :, :native_count],
        native_position_ids,
    ):
        raise ValueError("expanded PICF prefix changed the released native metadata")

    native_action_valid = action_visible[:, :native_count]
    if torch.equal(native_action_valid, native_valid):
        native_action_valid = native_valid
    inserted_visible = action_visible[:, native_count:]
    selected_relative = torch.nonzero(inserted_visible.any(dim=0), as_tuple=False).flatten()
    selected_absolute = selected_relative + native_count
    if suffix_count is None and action_query_count is not None:
        raise ValueError("an action query count requires a full suffix count")
    action_attention_layout = None
    if suffix_count is not None:
        expanded_posterior = context.expanded_posterior_indices
        posterior_valid = context.expanded_posterior_valid
        if expanded_posterior is None or posterior_valid is None:
            raise RuntimeError("action-attention layout requires typed expanded posterior rows")
        action_attention_layout = build_lingbot_action_attention_layout(
            batch_size=native_valid.shape[0],
            native_prefix_count=native_count,
            suffix_count=suffix_count,
            action_query_count=action_query_count,
            selected_inserted_indices=selected_absolute,
            expanded_posterior_indices=expanded_posterior,
            posterior_key_valid=posterior_valid,
        )
    if context.object_read_action_intervention is ObjectReadActionIntervention.BLOCKED:
        attention_layout = context.task_address_attention_layout
        if attention_layout is None:
            if selected_relative.numel():
                raise RuntimeError(
                    "blocked object-read intervention cannot verify inserted action Values"
                )
        else:
            object_read = torch.zeros_like(action_visible)
            object_read[:, attention_layout.object_read_slice] = True
            if (action_visible & object_read).any():
                raise RuntimeError("blocked object-read intervention retained OBJECT_READ")

    if set(native_past_key_values) != set(expanded_past_key_values):
        raise ValueError("native and PICF action caches contain different layer sets")
    if not native_past_key_values:
        raise ValueError("LingBot action cache contains no transformer layers")
    for cache_name, cache in (
        ("native", native_past_key_values),
        ("expanded", expanded_past_key_values),
    ):
        expected_tokens = native_count if cache_name == "native" else expanded_valid.shape[1]
        for layer_idx, layer in cache.items():
            if set(layer) != {"key_states", "value_states"}:
                raise ValueError(f"LingBot {cache_name} cache layer {layer_idx} has unknown fields")
            key = layer["key_states"]
            value = layer["value_states"]
            if (
                key.ndim < 3
                or value.shape != key.shape
                or key.shape[0] != native_valid.shape[0]
                or key.shape[1] != expected_tokens
                or key.device != native_valid.device
                or not key.is_floating_point()
                or not value.is_floating_point()
            ):
                raise ValueError(
                    f"LingBot {cache_name} cache layer {layer_idx} has an invalid tensor shape"
                )

    if not selected_relative.numel():
        return LingBotActionCache(
            past_key_values=native_past_key_values,
            valid=native_action_valid,
            position_ids=native_position_ids,
            position_valid=native_action_valid,
            selected_inserted_indices=selected_absolute,
            action_attention_layout=action_attention_layout,
        )

    compact_cache: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx, native_layer in native_past_key_values.items():
        expanded_layer = expanded_past_key_values[layer_idx]
        compact_cache[layer_idx] = {
            name: torch.cat(
                (
                    native_layer[name],
                    expanded_layer[name].index_select(1, selected_absolute),
                ),
                dim=1,
            )
            for name in ("key_states", "value_states")
        }
    selected_valid = inserted_visible.index_select(1, selected_relative)
    return LingBotActionCache(
        past_key_values=compact_cache,
        valid=torch.cat((native_action_valid, selected_valid), dim=1),
        position_ids=torch.cat(
            (
                native_position_ids,
                expanded_positions.index_select(2, selected_absolute),
            ),
            dim=2,
        ),
        position_valid=torch.cat(
            (
                native_action_valid,
                torch.zeros_like(selected_valid),
            ),
            dim=1,
        ),
        selected_inserted_indices=selected_absolute,
        action_attention_layout=action_attention_layout,
    )


def native_context_from_persistent_state(
    *,
    controls: ExecutedControlBatch,
    persistent_state: NativePosteriorState | NativeLayerwisePosteriorState | None,
    persistent_state_valid: torch.Tensor | None = None,
    prediction_request: NativePredictionRequest | None = None,
    modalities: NativeModalityBatch | None = None,
    supervise_intermediate_relations: bool = False,
    object_read_action_intervention: ObjectReadActionIntervention = (
        ObjectReadActionIntervention.FACTUAL
    ),
    action_information_sets: tuple[TaskAddressActionInformationSet, ...] = (),
    object_read_source_row_visible: torch.Tensor | None = None,
) -> LingBotNativeContext:
    """Route one typed persistent-state schema into its non-overlapping host ABI."""

    common = {
        "controls": controls,
        "prediction_request": prediction_request,
        "modalities": modalities,
        "supervise_intermediate_relations": supervise_intermediate_relations,
        "object_read_action_intervention": object_read_action_intervention,
        "action_information_sets": action_information_sets,
        "object_read_source_row_visible": object_read_source_row_visible,
    }
    if isinstance(persistent_state, NativeLayerwisePosteriorState):
        return LingBotNativeContext(
            previous_memory=persistent_state,
            previous_memory_valid=persistent_state_valid,
            episode_address_state=(
                persistent_state.episode_address_state
                if isinstance(persistent_state, AddressedLayerwisePosteriorState)
                else None
            ),
            **common,
        )
    if persistent_state is None or isinstance(persistent_state, NativePosteriorState):
        return LingBotNativeContext(
            previous_state=persistent_state,
            previous_state_valid=persistent_state_valid,
            **common,
        )
    raise TypeError("native context received an unknown persistent-state schema")


def native_context_from_prior_trace(
    *,
    controls: ExecutedControlBatch,
    prior_trace: NativeLayerwisePriorTrace,
    prediction_request: NativePredictionRequest | None = None,
    modalities: NativeModalityBatch | None = None,
    posterior_adoption_route: torch.Tensor | None = None,
    posterior_action_row_visible: torch.Tensor | None = None,
    supervise_intermediate_relations: bool = False,
    episode_address_state: EpisodeAddressState | None = None,
    object_read_action_intervention: ObjectReadActionIntervention = (
        ObjectReadActionIntervention.FACTUAL
    ),
    action_information_sets: tuple[TaskAddressActionInformationSet, ...] = (),
    object_read_source_row_visible: torch.Tensor | None = None,
) -> LingBotNativeContext:
    """Build the ADR149 correction/action context from a completed prior pass."""

    if not isinstance(prior_trace, NativeLayerwisePriorTrace):
        raise TypeError("unified correction requires a typed layerwise prior trace")
    if isinstance(prior_trace, AddressedLayerwisePriorTrace):
        if episode_address_state is None:
            episode_address_state = prior_trace.episode_address_state
        elif not episode_address_state.same_assignment(prior_trace.episode_address_state):
            raise ValueError("correction address state differs from the prior trace receipt")
    return LingBotNativeContext(
        controls=controls,
        prior_trace=prior_trace,
        prediction_request=prediction_request,
        modalities=modalities,
        posterior_adoption_route=posterior_adoption_route,
        posterior_action_row_visible=posterior_action_row_visible,
        supervise_intermediate_relations=supervise_intermediate_relations,
        episode_address_state=episode_address_state,
        object_read_action_intervention=object_read_action_intervention,
        action_information_sets=action_information_sets,
        object_read_source_row_visible=object_read_source_row_visible,
    )


@dataclass(slots=True)
class LingBotPriorRolloutContext:
    """Observation-free prior pass through the same LingBot host weights."""

    controls: ExecutedControlBatch
    previous_state: NativePosteriorState | None = None
    previous_state_valid: torch.Tensor | None = None
    previous_memory: NativeLayerwisePosteriorState | None = None
    previous_memory_valid: torch.Tensor | None = None
    source_prior_trace: NativeLayerwisePriorTrace | None = None
    source_prior_trace_valid: torch.Tensor | None = None
    prediction_request: NativePredictionRequest | None = None
    episode_address_state: EpisodeAddressState | None = None
    episode_ids: torch.Tensor | None = None
    prior_state: NativePosteriorState | None = field(init=False, default=None)
    prior_trace: NativeLayerwisePriorTrace | None = field(init=False, default=None)
    prediction_hidden: torch.Tensor | None = field(init=False, default=None)
    prediction_outputs: Mapping[str, torch.Tensor] = field(
        init=False,
        default_factory=lambda: MappingProxyType({}),
    )
    _finalized: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        batch = self.controls.batch_size
        if self.episode_address_state is not None:
            if not isinstance(self.episode_address_state, EpisodeAddressState):
                raise TypeError("prior context address state must use EpisodeAddressState")
            if self.episode_address_state.batch_size != batch:
                raise ValueError("prior context and address state have different batches")
            if self.episode_address_state.device != self.controls.values.device:
                raise ValueError("prior address state and controls must share one device")
        if self.episode_ids is not None:
            if self.episode_ids.shape != (batch,) or self.episode_ids.dtype != torch.long:
                raise ValueError("episode IDs must be long [batch]")
            if self.episode_ids.device != self.controls.values.device:
                raise ValueError("episode IDs and controls must share one device")
        if self.previous_state is not None and not isinstance(
            self.previous_state, NativePosteriorState
        ):
            raise TypeError("row-only previous state must use the final-row posterior schema")
        if self.previous_memory is not None and not isinstance(
            self.previous_memory, NativeLayerwisePosteriorState
        ):
            raise TypeError("prior-only memory must use the persistent layerwise posterior schema")
        if self.source_prior_trace is not None and not isinstance(
            self.source_prior_trace,
            NativeLayerwisePriorTrace,
        ):
            raise TypeError("chained prior source must use the transient v3 prior schema")
        if (
            sum(
                value is not None
                for value in (
                    self.previous_state,
                    self.previous_memory,
                    self.source_prior_trace,
                )
            )
            > 1
        ):
            raise ValueError(
                "final rows, persistent posterior memory, and transient prior source are "
                "mutually exclusive"
            )
        if self.previous_state is not None:
            if self.previous_state.batch_size != batch:
                raise ValueError("row-only controls and posterior state have different batches")
            if self.previous_state.rows.device != self.controls.values.device:
                raise ValueError("row-only controls and posterior state must share one device")
        if self.previous_state_valid is None:
            self.previous_state_valid = torch.full(
                (batch,),
                self.previous_state is not None,
                dtype=torch.bool,
                device=self.controls.values.device,
            )
        if (
            self.previous_state_valid.shape != (batch,)
            or self.previous_state_valid.dtype != torch.bool
            or self.previous_state_valid.device != self.controls.values.device
        ):
            raise ValueError("row-only state validity must be boolean [batch]")
        if self.previous_state is None and self.previous_state_valid.any():
            raise ValueError("row-only state validity cannot select absent final rows")
        if self.previous_memory is not None:
            if self.previous_memory.batch_size != batch:
                raise ValueError("row-only controls and layerwise posterior have different batches")
            if self.previous_memory.layer_rows.device != self.controls.values.device:
                raise ValueError("row-only controls and layerwise posterior must share one device")
        if self.previous_memory_valid is None:
            self.previous_memory_valid = torch.full(
                (batch,),
                self.previous_memory is not None,
                dtype=torch.bool,
                device=self.controls.values.device,
            )
        if (
            self.previous_memory_valid.shape != (batch,)
            or self.previous_memory_valid.dtype != torch.bool
            or self.previous_memory_valid.device != self.controls.values.device
        ):
            raise ValueError("row-only memory validity must be boolean [batch]")
        if self.previous_memory is None and self.previous_memory_valid.any():
            raise ValueError("row-only memory validity cannot select an absent posterior trace")
        if self.source_prior_trace is not None:
            if self.source_prior_trace.batch_size != batch:
                raise ValueError("row-only controls and chained prior have different batches")
            if self.source_prior_trace.layer_rows.device != self.controls.values.device:
                raise ValueError("row-only controls and chained prior must share one device")
        if self.source_prior_trace_valid is None:
            self.source_prior_trace_valid = torch.full(
                (batch,),
                self.source_prior_trace is not None,
                dtype=torch.bool,
                device=self.controls.values.device,
            )
        if (
            self.source_prior_trace_valid.shape != (batch,)
            or self.source_prior_trace_valid.dtype != torch.bool
            or self.source_prior_trace_valid.device != self.controls.values.device
        ):
            raise ValueError("chained prior validity must be boolean [batch]")
        if self.source_prior_trace is None and self.source_prior_trace_valid.any():
            raise ValueError("chained prior validity cannot select an absent prior trace")
        if self.prediction_request is not None:
            request = self.prediction_request
            if request.source is not PredictionSource.PRIOR or request.evidence not in {
                PredictionEvidence.FUTURE,
                PredictionEvidence.CURRENT_PRIOR,
            }:
                raise ValueError(
                    "prior-only prediction requires PRIOR source with FUTURE or "
                    "CURRENT_PRIOR evidence"
                )
            if request.batch_size != batch:
                raise ValueError("row-only prediction request and state batches differ")
            if request.route_ids.device != self.controls.values.device:
                raise ValueError("row-only prediction request and controls must share one device")


@dataclass(frozen=True, slots=True)
class _NativeRuntime:
    context: LingBotNativeContext
    original_prefix_count: int
    modality_slice: slice
    modality_valid: torch.Tensor
    modality_slices: tuple[tuple[str, slice], ...]
    relation_surfaces: tuple[PhysicalRelationSurfaceInput, ...]
    control_slice: slice
    prior_slice: slice
    posterior_slice: slice
    match_slice: slice | None
    prediction_slice: slice | None
    prediction_query_count: int
    layout: NativeTokenLayout
    episode_addresses: torch.Tensor | None = None
    layerwise_outputs: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _TaskAddressRuntime:
    context: LingBotNativeContext
    original_prefix_count: int
    language_slice: slice
    task_text_slice: slice
    modality_slice: slice
    modality_valid: torch.Tensor
    modality_slices: tuple[tuple[str, slice], ...]
    relation_surfaces: tuple[PhysicalRelationSurfaceInput, ...]
    control_slice: slice
    prior_slice: slice
    posterior_slice: slice
    task_query_slice: slice
    object_read_slice: slice
    prediction_slice: slice | None
    prediction_query_count: int
    layout: TaskAddressTokenLayout
    episode_addresses: torch.Tensor
    match_slice: None = None
    layerwise_outputs: list[torch.Tensor] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PriorRolloutRuntime:
    context: LingBotPriorRolloutContext
    prior_slice: slice
    prediction_slice: slice | None
    prediction_query_count: int
    layout: NativeTokenLayout
    episode_addresses: torch.Tensor | None = None
    layerwise_outputs: list[torch.Tensor] = field(default_factory=list)


_Runtime = _NativeRuntime | _TaskAddressRuntime | _PriorRolloutRuntime


class _LingBotReleasedTaskTokenBridge(nn.Module):
    """Bound dense evidence with LingBot's released pretrained query resampler.

    The injected projector is the checkpointed ``TaskTokenResampler`` from the
    LingBot current-video alignment head.  This wrapper adds only a
    column-isometric width expansion.  It neither selects objects nor receives
    task or lifecycle labels; all semantic decisions remain in the shared host.
    """

    def __init__(
        self,
        *,
        projector: nn.Module,
        queries: torch.Tensor,
        host_width: int,
        query_count: int,
        device: torch.device | str | None,
        dtype: torch.dtype | None,
    ) -> None:
        super().__init__()
        if not isinstance(projector, nn.Module) or not callable(
            getattr(projector, "forward", None)
        ):
            raise TypeError("LingBot modality bridge requires its released projector module")
        if (
            not torch.is_tensor(queries)
            or queries.ndim != 2
            or tuple(queries.shape) != (query_count, host_width)
            or not queries.is_floating_point()
            or not torch.isfinite(queries).all()
        ):
            raise ValueError("LingBot modality bridge queries differ from the released contract")
        proj_in1 = getattr(projector, "proj_in1", None)
        proj_in2 = getattr(projector, "proj_in2", None)
        proj_out = getattr(projector, "proj_out", None)
        norm_out = getattr(projector, "norm_out", None)
        layers = getattr(projector, "layers", None)
        if (
            not isinstance(proj_in1, nn.Linear)
            or not isinstance(proj_in2, nn.Linear)
            or not isinstance(proj_out, nn.Linear)
            or not isinstance(norm_out, nn.LayerNorm)
            or not isinstance(layers, nn.ModuleList)
            or len(layers) != 1
            or getattr(projector, "num_queries", None) != query_count
            or proj_in1.in_features != host_width
            or proj_in2.in_features != host_width
            or proj_in1.out_features != host_width
            or proj_in2.out_features != host_width
            or proj_out.in_features != host_width
            or tuple(norm_out.normalized_shape) != (proj_out.out_features,)
        ):
            raise ValueError("LingBot modality bridge projector topology is not the released head")
        factory = {"device": device, "dtype": dtype}
        self.projector = projector.to(**factory)
        self.queries = nn.Parameter(queries.detach().clone().to(**factory))
        self.host_projection = nn.Linear(
            proj_out.out_features,
            host_width,
            bias=False,
            **factory,
        )
        with torch.no_grad():
            initialize_column_isometry(self.host_projection)

    def forward(
        self,
        tokens: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            tokens.ndim != 3
            or tokens.shape[-1] != self.queries.shape[-1]
            or valid.shape != tokens.shape[:2]
            or valid.dtype != torch.bool
            or valid.device != tokens.device
            or tokens.device != self.queries.device
            or tokens.dtype != self.queries.dtype
        ):
            raise ValueError("LingBot modality bridge received an invalid typed evidence set")
        outputs: list[torch.Tensor] = []
        output_valid: list[torch.Tensor] = []
        for sample_index in range(tokens.shape[0]):
            sample_valid = valid[sample_index]
            observed = sample_valid.any()
            if bool(observed.item()):
                sample = tokens[sample_index : sample_index + 1, sample_valid]
            else:
                # Every rank executes the same released module even when its
                # local optional evidence is absent. The resulting transport
                # rows are multiplied by zero and masked out before the host.
                sample = tokens.new_zeros(1, 1, tokens.shape[-1])
            queries = self.queries.unsqueeze(0)
            value = self.host_projection(self.projector(sample, queries))
            value = value * observed.to(dtype=value.dtype)
            # FSDP2 requires identical backward participation across ranks.
            # Preserve a zero-valued path through locally absent source adapters.
            value = (
                value
                + tokens[sample_index : sample_index + 1].sum(
                    dim=(1, 2),
                    keepdim=True,
                )
                * 0.0
            )
            outputs.append(value)
            output_valid.append(observed.expand(self.queries.shape[0]))
        return torch.cat(outputs, dim=0), torch.stack(output_valid, dim=0)


class LingBotNativeGraph(nn.Module):
    """Typed rows inside LingBot; all nonlinear semantics remain in the host."""

    def __init__(
        self,
        config: LingBotNativeGraphConfig,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        modality_bridge_projector: nn.Module | None = None,
        modality_bridge_queries: torch.Tensor | None = None,
        source_mask_head: nn.Module | None = None,
        source_mask_refiner: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if config.architecture_identity == NATIVE_VIDEOMT_QUERY_POSTERIOR and (
            source_mask_head is not None or source_mask_refiner is not None
        ):
            raise ValueError(
                "native VidEoMT posterior uses the complete source masks directly and "
                "forbids a second host-side decoder"
            )
        if (source_mask_head is not None or source_mask_refiner is not None) and (
            not self.task_independent or not config.object_query_spatial_specs
        ):
            raise ValueError(
                "a source spatial module requires a task-independent object-query spatial graph"
            )
        factory = {"device": device, "dtype": dtype}
        if self.task_query_object_value_read:
            self.register_buffer(
                "episode_address_codebook",
                fixed_orthogonal_address_codebook(
                    config.capacity,
                    config.host_width,
                    device=device,
                    dtype=torch.get_default_dtype() if dtype is None else dtype,
                ),
            )
            self.register_parameter("object_addresses", None)
            self.register_parameter("object_queries", None)
            self.task_query_embeddings = nn.Parameter(
                torch.empty(config.task_query_count, config.host_width, **factory)
            )
        elif self.layerwise_recurrence:
            self.register_buffer("episode_address_codebook", None)
            self.object_addresses = nn.Parameter(
                torch.empty(config.capacity, config.host_width, **factory)
            )
            self.register_parameter("object_queries", None)
            self.register_parameter("task_query_embeddings", None)
        else:
            self.register_buffer("episode_address_codebook", None)
            self.object_queries = nn.Parameter(
                torch.empty(config.capacity, config.host_width, **factory)
            )
            self.register_parameter("object_addresses", None)
            self.register_parameter("task_query_embeddings", None)
        role_count = 3 if self.task_independent else 4
        self.role_embeddings = nn.Parameter(torch.empty(role_count, config.host_width, **factory))
        self.prediction_role = nn.Parameter(torch.empty(config.host_width, **factory))
        self.prediction_route_embeddings = nn.Parameter(
            torch.empty(config.prediction_route_count, config.host_width, **factory)
        )
        self.prediction_horizon_projection = nn.Linear(2, config.host_width, bias=False, **factory)
        self.prediction_address_projection = (
            nn.Linear(
                config.prediction_address_width,
                config.host_width,
                bias=False,
                **factory,
            )
            if config.prediction_address_width
            else None
        )
        control_width = 2 * config.executed_action_dim + 3
        self.control_projection = nn.Linear(control_width, config.host_width, bias=False, **factory)
        if self.native_videomt_query_posterior:
            self.relation_readout = None
        else:
            self.relation_readout = (
                PhysicalEntityReadout(
                    config.host_width,
                    source_mask_head=source_mask_head,
                    source_mask_refiner=source_mask_refiner,
                    temperature_init=config.relation_temperature,
                )
                if self.task_independent
                else SharedRelationReadout(
                    config.host_width,
                    temperature_init=config.relation_temperature,
                )
            ).to(**factory)
        self.predictive_readouts = nn.ModuleDict(
            {
                name: NativePredictiveReadout(
                    config.host_width,
                    target_width,
                    config.prediction_route_count,
                    **factory,
                )
                for name, target_width in config.predictive_target_widths
            }
        )
        self.modality_projections = nn.ModuleDict(
            {
                spec.name: nn.Linear(spec.input_width, config.host_width, bias=False, **factory)
                for spec in config.modality_specs
            }
        )
        self.modality_metadata_projections = nn.ModuleDict(
            {
                spec.name: nn.Linear(
                    spec.metadata_width,
                    config.host_width,
                    bias=False,
                    **factory,
                )
                for spec in config.modality_specs
                if spec.metadata_width
            }
        )
        if config.modality_specs:
            self.modality_embeddings = nn.Parameter(
                torch.empty(len(config.modality_specs), config.host_width, **factory)
            )
        else:
            self.register_parameter("modality_embeddings", None)
        if config.modality_bridge_identity == EXACT_NATIVE_MODALITY_BRIDGE:
            if modality_bridge_projector is not None or modality_bridge_queries is not None:
                raise ValueError("exact-token graph received an undeclared modality bridge")
            self.modality_bridge = None
        else:
            if modality_bridge_projector is None or modality_bridge_queries is None:
                raise ValueError("LingBot resampled graph omitted its released bridge weights")
            self.modality_bridge = _LingBotReleasedTaskTokenBridge(
                projector=modality_bridge_projector,
                queries=modality_bridge_queries,
                host_width=config.host_width,
                query_count=config.modality_bridge_query_count,
                device=device,
                dtype=dtype,
            )
        with torch.no_grad():
            if self.task_query_object_value_read:
                if self.task_query_embeddings is None:
                    raise RuntimeError("task-addressed graph omitted its native query rows")
                self.task_query_embeddings.normal_(std=config.initializer_range)
            else:
                self.object_parameter.normal_(std=config.initializer_range)
            self.role_embeddings.normal_(std=config.initializer_range)
            self.prediction_role.normal_(std=config.initializer_range)
            self.prediction_route_embeddings.normal_(std=config.initializer_range)
            self.control_projection.weight.normal_(std=config.initializer_range)
            self.prediction_horizon_projection.weight.normal_(std=config.initializer_range)
            if self.prediction_address_projection is not None:
                self.prediction_address_projection.weight.normal_(std=config.initializer_range)
            for projection in self.modality_projections.values():
                if not isinstance(projection, nn.Linear):
                    raise RuntimeError("native modality projection changed implementation type")
                initialize_column_isometry(projection)
            for projection in self.modality_metadata_projections.values():
                if not isinstance(projection, nn.Linear):
                    raise RuntimeError("native modality metadata projection changed type")
                initialize_column_isometry(projection)
            if self.modality_embeddings is not None:
                self.modality_embeddings.normal_(std=config.initializer_range)

    @property
    def task_independent(self) -> bool:
        return self.config.architecture_identity in {
            TASK_INDEPENDENT_ENTITY_POSTERIOR,
            LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
            UNIFIED_LAYERWISE_PREDICT_CORRECT,
            LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
            NATIVE_VIDEOMT_QUERY_POSTERIOR,
        }

    @property
    def layerwise_recurrence(self) -> bool:
        return self.config.architecture_identity in {
            LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
            UNIFIED_LAYERWISE_PREDICT_CORRECT,
            LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
            NATIVE_VIDEOMT_QUERY_POSTERIOR,
        }

    @property
    def unified_predict_correct(self) -> bool:
        return self.config.architecture_identity in {
            UNIFIED_LAYERWISE_PREDICT_CORRECT,
            LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
            NATIVE_VIDEOMT_QUERY_POSTERIOR,
        }

    @property
    def task_query_object_value_read(self) -> bool:
        return self.config.architecture_identity == LINGBOT_TASK_QUERY_OBJECT_VALUE_READ

    @property
    def native_videomt_query_posterior(self) -> bool:
        return self.config.architecture_identity == NATIVE_VIDEOMT_QUERY_POSTERIOR

    @property
    def object_parameter(self) -> nn.Parameter:
        value = self.object_addresses if self.layerwise_recurrence else self.object_queries
        if not isinstance(value, nn.Parameter):
            raise RuntimeError("native graph object parameter is not initialized")
        return value

    @property
    def _parameter_reference(self) -> nn.Parameter:
        if self.task_query_object_value_read:
            value = self.task_query_embeddings
            if not isinstance(value, nn.Parameter):
                raise RuntimeError("task-addressed graph query parameter is not initialized")
            return value
        return self.object_parameter

    def _materialize_episode_addresses(
        self,
        state: EpisodeAddressState,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.task_query_object_value_read:
            raise RuntimeError("episode address states belong only to the task-addressed identity")
        codebook = self.episode_address_codebook
        if not isinstance(codebook, torch.Tensor):
            raise RuntimeError("task-addressed graph omitted its fixed address codebook")
        if state.batch_size != batch or state.capacity != self.config.capacity:
            raise ValueError("episode address state differs from the task-address host contract")
        if state.device != device or codebook.device != device:
            raise ValueError("episode addresses, fixed codebook and LingBot host must align")
        addresses = state.materialize(codebook).to(dtype=dtype)
        expected = (batch, self.config.capacity, self.config.host_width)
        if addresses.shape != expected or addresses.dtype != dtype:
            raise RuntimeError("materialized episode addresses differ from the host contract")
        return addresses

    def _resolve_prior_episode_addresses(
        self,
        context: LingBotPriorRolloutContext,
        *,
        source_valid: torch.Tensor,
        source_prior_trace: NativeLayerwisePriorTrace | None,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[EpisodeAddressState, torch.Tensor, torch.Tensor]:
        """Resolve one explicit gauge and mask reset lanes out of old memory."""

        if source_valid.shape != (context.controls.batch_size,) or (
            source_valid.dtype != torch.bool or source_valid.device != device
        ):
            raise ValueError("prior source validity must be boolean [batch] on the host device")
        source_address_state: EpisodeAddressState | None = None
        if context.previous_memory is not None:
            if not isinstance(context.previous_memory, AddressedLayerwisePosteriorState):
                raise ValueError(
                    "task-addressed posterior memory must carry its episode address receipt"
                )
            if context.previous_memory.architecture_identity != self.config.architecture_identity:
                raise ValueError("posterior memory belongs to another architecture identity")
            source_address_state = context.previous_memory.episode_address_state
        if source_prior_trace is not None:
            if not isinstance(source_prior_trace, AddressedLayerwisePriorTrace):
                raise ValueError(
                    "task-addressed chained prior must carry its episode address receipt"
                )
            if source_prior_trace.architecture_identity != self.config.architecture_identity:
                raise ValueError("chained prior trace belongs to another architecture identity")
            prior_address_state = source_prior_trace.episode_address_state
            if source_address_state is not None and not source_address_state.same_assignment(
                prior_address_state
            ):
                raise ValueError("prior sources carry different episode address receipts")
            source_address_state = prior_address_state

        codebook = self.episode_address_codebook
        if not isinstance(codebook, torch.Tensor):
            raise RuntimeError("task-addressed graph omitted its fixed address codebook")
        generated = (
            None
            if context.episode_ids is None
            else EpisodeAddressState.from_episode_ids(
                codebook=codebook,
                episode_ids=context.episode_ids,
            )
        )
        explicit = context.episode_address_state
        if (
            explicit is not None
            and generated is not None
            and not explicit.same_assignment(generated)
        ):
            raise ValueError("explicit address assignment disagrees with deterministic episode IDs")

        reset = context.controls.reset.any(dim=1)
        continuation = source_valid & ~reset
        fresh = ~continuation
        if explicit is None:
            if source_address_state is not None:
                if fresh.any():
                    if generated is None:
                        raise ValueError(
                            "reset or invalid prior lanes require explicit addresses or episode IDs"
                        )
                    permutation = source_address_state.permutation.clone()
                    permutation[fresh] = generated.permutation[fresh]
                    explicit = EpisodeAddressState(
                        permutation=permutation,
                        codebook_sha256=source_address_state.codebook_sha256,
                    )
                else:
                    explicit = source_address_state
            elif generated is not None:
                explicit = generated
            else:
                raise ValueError(
                    "task-addressed prior requires explicit addresses or deterministic episode IDs"
                )

        if source_address_state is not None and continuation.any():
            continued = torch.nonzero(continuation, as_tuple=False).flatten()
            if not explicit.index_select(continued).same_assignment(
                source_address_state.index_select(continued)
            ):
                raise ValueError("continuing prior lanes changed their episode address receipt")

        addresses = self._materialize_episode_addresses(
            explicit,
            batch=context.controls.batch_size,
            device=device,
            dtype=dtype,
        )
        context.episode_address_state = explicit
        return explicit, continuation, addresses

    def _task_address_correction_addresses(
        self,
        context: LingBotNativeContext,
        *,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        trace = context.prior_trace
        if not isinstance(trace, AddressedLayerwisePriorTrace):
            raise ValueError("task-addressed correction requires an addressed prior trace")
        if trace.architecture_identity != self.config.architecture_identity:
            raise ValueError("task-addressed prior trace belongs to another architecture identity")
        state = context.episode_address_state
        if state is None:
            raise ValueError("task-addressed correction requires explicit EpisodeAddressState")
        if not state.same_assignment(trace.episode_address_state):
            raise ValueError("correction address state differs from the prior trace receipt")
        return self._materialize_episode_addresses(
            state,
            batch=prefix.shape[0],
            device=prefix.device,
            dtype=prefix.dtype,
        )

    def predictive_readout(self, target_name: str) -> NativePredictiveReadout:
        """Return one config-owned linear target adapter before/after FSDP wrapping."""

        if not isinstance(target_name, str) or not target_name:
            raise ValueError("predictive readout name must be non-empty")
        try:
            value = self.predictive_readouts[target_name]
        except KeyError as error:
            raise KeyError(f"undeclared predictive target space {target_name!r}") from error
        if not isinstance(value, NativePredictiveReadout):
            raise RuntimeError("native graph predictive readout changed implementation type")
        return value

    def _project_prediction_outputs(
        self,
        hidden: torch.Tensor,
        request: NativePredictionRequest,
    ) -> Mapping[str, torch.Tensor]:
        """Project shared-host queries before the enclosing FSDP root can reshard."""

        outputs = {
            name: self.predictive_readout(name)(hidden, request.route_ids)
            for name, _width in self.config.predictive_target_widths
        }
        return MappingProxyType(outputs)

    def _validate_context(
        self,
        context: LingBotNativeContext,
        *,
        prefix: torch.Tensor,
    ) -> None:
        config = self.config
        previous_state_valid = context.previous_state_valid
        previous_memory_valid = context.previous_memory_valid
        if previous_state_valid is None or previous_memory_valid is None:
            raise RuntimeError("validated native context omitted recurrent validity metadata")
        if any(
            value is None
            for value in (
                context.native_roles,
                context.native_valid,
                context.instruction_last_index,
            )
        ):
            raise ValueError("LingBot-native context reached the graph before prefix binding")
        context._validate_bound_prefix()
        if context.native_roles is None or context.native_valid is None:
            raise RuntimeError("validated native context lost prefix metadata")
        context.controls.validate_bound(config.maximum_control_tokens)
        if context.controls.action_dim != config.executed_action_dim:
            raise ValueError("executed-control width differs from the frozen LingBot contract")
        if context.native_roles.shape != prefix.shape[:2]:
            raise ValueError("native role metadata does not match prefix embeddings")
        if (
            context.native_roles.device != prefix.device
            or context.controls.values.device != prefix.device
        ):
            raise ValueError("context tensors and LingBot prefix must share one device")
        if prefix.shape[-1] != config.host_width:
            raise ValueError("LingBot prefix width differs from the frozen PICF contract")
        source_row_visible = context.object_read_source_row_visible
        if source_row_visible is not None:
            if not self.task_query_object_value_read:
                raise ValueError(
                    "object-read source-row visibility requires the task-addressed architecture"
                )
            if source_row_visible.shape != (prefix.shape[0], config.capacity):
                raise ValueError(
                    "object-read source-row visibility must have shape [batch, capacity]"
                )
            if source_row_visible.dtype != torch.bool or source_row_visible.device != prefix.device:
                raise ValueError(
                    "object-read source-row visibility must be boolean on the LingBot device"
                )
        if (
            prefix.device != self._parameter_reference.device
            or prefix.dtype != self._parameter_reference.dtype
        ):
            raise ValueError("LingBot prefix and PICF parameters must share device and dtype")
        if not config.modality_specs:
            if context.modalities is not None:
                raise ValueError("runtime modalities were supplied to an undeclared graph")
        else:
            if context.modalities is None:
                raise ValueError("declared graph modalities require one typed runtime batch")
            context.modalities.validate_against(config.modality_specs)
            context.modalities.validate_object_query_spatial_relations(
                config.object_query_spatial_specs
            )
            if (
                context.modalities.device != prefix.device
                or context.modalities.dtype != prefix.dtype
            ):
                raise ValueError("runtime modality tokens must share LingBot device and dtype")
        if context.previous_state is not None:
            previous = context.previous_state
            if (
                previous.batch_size != prefix.shape[0]
                or previous.capacity != config.capacity
                or previous.host_width != config.host_width
            ):
                raise ValueError("previous posterior rows differ from the frozen state contract")
            if previous.rows.device != prefix.device or previous.rows.dtype != prefix.dtype:
                raise ValueError("previous posterior rows must share LingBot device and dtype")
        if self.unified_predict_correct:
            if (
                context.previous_state is not None
                or previous_state_valid.any()
                or context.previous_memory is not None
                or previous_memory_valid.any()
            ):
                raise ValueError(
                    "unified correction cannot read a previous posterior directly; "
                    "run the prior-only pass first"
                )
            trace = context.prior_trace
            if not isinstance(trace, NativeLayerwisePriorTrace):
                raise ValueError("unified correction requires a completed layerwise prior trace")
            if (
                trace.batch_size != prefix.shape[0]
                or trace.num_layers != config.num_layers
                or trace.capacity != config.capacity
                or trace.host_width != config.host_width
            ):
                raise ValueError("unified prior trace differs from the v3 host contract")
            if trace.layer_rows.device != prefix.device or trace.layer_rows.dtype != prefix.dtype:
                raise ValueError("unified prior trace must share LingBot device and dtype")
        elif self.layerwise_recurrence:
            if context.prior_trace is not None:
                raise ValueError("layerwise PICF v2 cannot consume a v3 prior trace")
            if context.previous_state is not None or previous_state_valid.any():
                raise ValueError("layerwise PICF rejects the retired final-row recurrent ABI")
            if context.previous_memory is not None:
                memory = context.previous_memory
                if (
                    memory.batch_size != prefix.shape[0]
                    or memory.num_layers != config.num_layers
                    or memory.capacity != config.capacity
                    or memory.host_width != config.host_width
                ):
                    raise ValueError("previous layerwise memory differs from the v2 state contract")
                if (
                    memory.layer_rows.device != prefix.device
                    or memory.layer_rows.dtype != prefix.dtype
                ):
                    raise ValueError(
                        "previous layerwise memory must share LingBot device and dtype"
                    )
        elif (
            context.previous_memory is not None
            or previous_memory_valid.any()
            or context.prior_trace is not None
        ):
            raise ValueError("legacy PICF cannot consume a layerwise recurrent ABI")
        if context.prediction_request is not None:
            request = context.prediction_request
            if not self.training:
                raise ValueError("prediction queries are training-only")
            if (
                request.evidence is PredictionEvidence.CURRENT_CORRECTION
                and self.unified_predict_correct
            ):
                raise ValueError("CURRENT_CORRECTION evidence is reserved for v2 replay")
            elif request.evidence in {
                PredictionEvidence.CURRENT_PRIOR,
                PredictionEvidence.CURRENT_POSTERIOR,
            }:
                if not self.unified_predict_correct:
                    raise ValueError(
                        "current prior/posterior evidence requires the v3 architecture"
                    )
                if request.evidence is PredictionEvidence.CURRENT_PRIOR:
                    raise ValueError("CURRENT_PRIOR evidence requires the v3 prior-only pass")
            if (request.route_ids >= config.prediction_route_count).any():
                raise ValueError("prediction request uses an undeclared route")
            if request.address_width != config.prediction_address_width:
                raise ValueError("prediction request address width differs from the graph contract")
            if request.addresses.dtype != prefix.dtype:
                raise ValueError("prediction addresses and LingBot prefix must share one dtype")

    def _project_modalities(
        self,
        context: LingBotNativeContext,
        *,
        prefix: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        tuple[PhysicalRelationSurfaceInput, ...],
    ]:
        batch = prefix.shape[0]
        if not self.config.modality_specs:
            return (
                prefix.new_empty(batch, 0, self.config.host_width),
                torch.empty(batch, 0, dtype=torch.bool, device=prefix.device),
                torch.empty(batch, 0, dtype=torch.bool, device=prefix.device),
                (),
            )
        if context.modalities is None or self.modality_embeddings is None:
            raise RuntimeError("declared native modality adapters are unbound")
        exact_projected: list[torch.Tensor] = []
        exact_valid: list[torch.Tensor] = []
        exact_direct_action_visible: list[torch.Tensor] = []
        resampled_projected: list[torch.Tensor] = []
        resampled_valid: list[torch.Tensor] = []
        relation_surfaces: list[PhysicalRelationSurfaceInput] = []
        resampled_names = set(self.config.resampled_modality_names)
        direct_action_names = set(self.config.direct_action_modality_names)
        relation_spec_by_name = {spec.name: spec for spec in self.config.relation_surface_specs}
        object_query_sources = {
            spec.query_modality for spec in self.config.object_query_spatial_specs
        }
        native_anchor_source = (
            self.config.object_query_spatial_specs[0].query_modality
            if self.native_videomt_query_posterior
            else None
        )
        for index, stream in enumerate(context.modalities.streams):
            spec = self.config.modality_specs[index]
            if stream.name != spec.name:
                raise RuntimeError("runtime modality order differs from the validated graph ABI")
            if (
                stream.name in resampled_names
                or stream.name in relation_spec_by_name
                or stream.name in object_query_sources
            ):
                stream = stream.canonicalized()
            if stream.name == native_anchor_source:
                # The complete source-query bank enters once, as the posterior
                # state itself. Its mask/class tensors remain in the typed
                # spatial relation and are never reconstructed by the host.
                continue
            value = self.modality_projections[stream.name](modality_bridge_input(stream, spec))
            if stream.metadata is not None:
                metadata = stream.metadata
                if spec.metadata_normalization == TOKEN_LAYER_NORM:
                    metadata = torch.nn.functional.layer_norm(
                        metadata,
                        (stream.metadata_width,),
                    )
                value = value + self.modality_metadata_projections[stream.name](
                    metadata
                )
            value = value + self.modality_embeddings[index]
            value = value * stream.valid.unsqueeze(-1).to(value.dtype)
            relation_spec = relation_spec_by_name.get(stream.name)
            if relation_spec is not None:
                relation_surfaces.append(
                    PhysicalRelationSurfaceInput(
                        name=relation_spec.name,
                        geometry_kind=relation_spec.geometry_kind,
                        target_kind=relation_spec.target_kind,
                        layout=relation_spec.layout,
                        sensor_hidden=value,
                        sensor_valid=stream.valid,
                        canonical_token_ids=stream.canonical_token_ids,
                    )
                )
            if stream.name in resampled_names:
                resampled_projected.append(value)
                resampled_valid.append(stream.valid)
            else:
                exact_projected.append(value)
                exact_valid.append(stream.valid)
                exact_direct_action_visible.append(
                    stream.valid
                    if stream.name in direct_action_names
                    else torch.zeros_like(stream.valid)
                )
        if self.modality_bridge is not None:
            if not resampled_projected:
                raise RuntimeError("declared LingBot modality bridge received no source streams")
            bridge_tokens, bridge_valid = self.modality_bridge(
                torch.cat(resampled_projected, dim=1),
                torch.cat(resampled_valid, dim=1),
            )
            exact_projected.insert(0, bridge_tokens)
            exact_valid.insert(0, bridge_valid)
            exact_direct_action_visible.insert(0, torch.zeros_like(bridge_valid))
        elif resampled_projected:
            raise RuntimeError("resampled modality streams reached an exact-token graph")
        if exact_projected:
            projected = torch.cat(exact_projected, dim=1)
            projected_valid = torch.cat(exact_valid, dim=1)
            direct_action_visible = torch.cat(exact_direct_action_visible, dim=1)
        else:
            projected = prefix.new_empty(batch, 0, self.config.host_width)
            projected_valid = torch.empty(batch, 0, dtype=torch.bool, device=prefix.device)
            direct_action_visible = torch.empty(
                batch, 0, dtype=torch.bool, device=prefix.device
            )
        return projected, projected_valid, direct_action_visible, tuple(relation_surfaces)

    def _contextual_modality_slices(
        self,
        context: LingBotNativeContext,
    ) -> tuple[tuple[str, slice], ...]:
        """Map exact source streams onto their shared-host insertion rows."""

        if context.modalities is None:
            return ()
        resampled = set(self.config.resampled_modality_names)
        native_anchor_source = (
            self.config.object_query_spatial_specs[0].query_modality
            if self.native_videomt_query_posterior
            else None
        )
        offset = self.config.modality_bridge_query_count if resampled else 0
        result: list[tuple[str, slice]] = []
        for stream in context.modalities.streams:
            if stream.name in resampled or stream.name == native_anchor_source:
                continue
            stop = offset + stream.token_count
            result.append((stream.name, slice(offset, stop)))
            offset = stop
        return tuple(result)

    def _native_videomt_posterior_seed(
        self,
        context: LingBotNativeContext,
        *,
        prefix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project every canonical source query once into its paired host row."""

        if not self.native_videomt_query_posterior:
            raise RuntimeError("native source-query seeding requires its architecture profile")
        if context.modalities is None:
            raise RuntimeError("native source-query seeding omitted runtime modalities")
        source_name = self.config.object_query_spatial_specs[0].query_modality
        indexed = tuple(
            (index, stream)
            for index, stream in enumerate(context.modalities.streams)
            if stream.name == source_name
        )
        if len(indexed) != 1:
            raise RuntimeError("native source-query seeding requires exactly one source stream")
        index, stream = indexed[0]
        stream = stream.canonicalized()
        spec = self.config.modality_specs[index]
        if (
            stream.token_count != self.config.capacity
            or not stream.valid.all()
            or stream.canonical_token_ids is None
        ):
            raise ValueError("native posterior requires all canonical source-query rows")
        expected_ids = torch.arange(
            self.config.capacity,
            dtype=torch.long,
            device=stream.valid.device,
        ).expand(stream.batch_size, -1)
        if not torch.equal(stream.canonical_token_ids, expected_ids):
            raise ValueError("native posterior source-query indices are not canonical")
        seed = self.modality_projections[source_name](modality_bridge_input(stream, spec))
        seed = seed + self.modality_embeddings[index]
        seed = seed * stream.valid.unsqueeze(-1).to(seed.dtype)
        expected = (prefix.shape[0], self.config.capacity, self.config.host_width)
        if seed.shape != expected or seed.device != prefix.device or seed.dtype != prefix.dtype:
            raise RuntimeError("native posterior source-query projection changed its host ABI")
        return seed, stream.valid

    def _object_content_seed(
        self,
        context: LingBotNativeContext,
        *,
        prefix: torch.Tensor,
    ) -> torch.Tensor:
        if self.layerwise_recurrence:
            raise RuntimeError("layerwise PICF separates object address from mutable content")
        batch = prefix.shape[0]
        if self.object_queries is None:
            raise RuntimeError("legacy PICF object queries are unavailable")
        discovery = self.object_queries.unsqueeze(0).expand(batch, -1, -1)
        if context.previous_state is None:
            return discovery
        previous_state_valid = context.previous_state_valid
        if previous_state_valid is None:
            raise RuntimeError("native previous rows lost their validity mask")
        previous = F.layer_norm(
            context.previous_state.rows,
            (self.config.host_width,),
        )
        return torch.where(previous_state_valid[:, None, None], previous, discovery)

    def _initial_rows(self, content: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        prior = content + self.role_embeddings[0]
        posterior = content + self.role_embeddings[1]
        return prior, posterior

    def _layerwise_initial_rows(
        self,
        *,
        batch: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Create content-neutral rows; stable identity enters only the Q/K path."""

        if not self.layerwise_recurrence:
            raise RuntimeError("content-neutral rows belong only to the layerwise architecture")
        neutral = self.role_embeddings.new_zeros(
            batch,
            self.config.capacity,
            self.config.host_width,
        )
        return self._initial_rows(neutral)

    def _match_queries(self, content: torch.Tensor) -> torch.Tensor:
        if self.task_independent:
            raise RuntimeError("task-independent graph has no match-query surface")
        return content + self.role_embeddings[3]

    def _row_only_prior(self, context: LingBotPriorRolloutContext) -> torch.Tensor:
        previous = context.previous_state
        config = self.config
        if previous is None:
            raise ValueError("legacy row-only prior requires final-row posterior state")
        if previous.capacity != config.capacity or previous.host_width != config.host_width:
            raise ValueError("row-only posterior differs from the frozen state contract")
        if (
            previous.rows.device != self._parameter_reference.device
            or previous.rows.dtype != self._parameter_reference.dtype
        ):
            raise ValueError("row-only posterior and graph must share device and dtype")
        normalized = F.layer_norm(previous.rows, (config.host_width,))
        previous_state_valid = context.previous_state_valid
        if previous_state_valid is None:
            raise RuntimeError("row-only previous rows lost their validity mask")
        if self.object_queries is None:
            raise RuntimeError("row-only rollout is unsupported by layerwise PICF v2")
        discovery = self.object_queries.unsqueeze(0).expand(previous.batch_size, -1, -1)
        content = torch.where(previous_state_valid[:, None, None], normalized, discovery)
        return content + self.role_embeddings[0]

    def _prediction_queries(
        self,
        request: NativePredictionRequest,
    ) -> torch.Tensor:
        route = self.prediction_route_embeddings[request.route_ids]
        horizon = request.horizons.to(route.dtype)
        horizon_features = torch.stack((horizon, torch.log1p(horizon)), dim=-1)
        query = route + self.prediction_horizon_projection(horizon_features)
        if self.prediction_address_projection is not None:
            query = query + self.prediction_address_projection(request.addresses)
        query = query + self.prediction_role
        return query.unsqueeze(1).expand(-1, self.config.capacity, -1, -1).flatten(1, 2)

    @staticmethod
    def _restrict_prediction_queries(
        mask: torch.Tensor,
        *,
        request: NativePredictionRequest,
        source_slice: slice,
        prediction_slice: slice,
        capacity: int,
    ) -> torch.Tensor:
        query_count = request.query_count
        mask[:, prediction_slice, :] = False
        for row_index in range(capacity):
            query_start = prediction_slice.start + row_index * query_count
            query_indices = torch.arange(
                query_start,
                query_start + query_count,
                device=mask.device,
            )
            query_valid = request.valid
            mask[:, query_indices, query_indices] = query_valid
            mask[:, query_indices, source_slice.start + row_index] = query_valid
        return mask

    @staticmethod
    def _restrict_match_set(
        mask: torch.Tensor,
        *,
        layout: NativeTokenLayout,
        match_slice: slice,
        capacity: int,
    ) -> torch.Tensor:
        """Expose the complete physical set and prompt to every ephemeral match row."""

        if match_slice.stop - match_slice.start != capacity:
            raise ValueError("match slice differs from the configured capacity")
        source_role = (
            (layout.roles == int(NativeRole.SENSOR))
            | (layout.roles == int(NativeRole.LANGUAGE))
            | (layout.roles == int(NativeRole.POSTERIOR))
            | (layout.roles == int(NativeRole.MATCH))
        )
        permitted = mask[:, match_slice].clone()
        mask[:, match_slice] = permitted & source_role[:, None, :]
        return mask

    @staticmethod
    def _insert_positions(
        position_ids: torch.Tensor,
        *,
        insertion_index: int,
        inserted_count: int,
    ) -> torch.Tensor:
        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("LingBot position IDs must have shape [3, batch, tokens]")
        identity = torch.zeros(
            (3, position_ids.shape[1], inserted_count),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        return torch.cat(
            (
                position_ids[:, :, :insertion_index],
                identity,
                position_ids[:, :, insertion_index:],
            ),
            dim=-1,
        )

    @staticmethod
    def _instruction_span(native_roles: torch.Tensor) -> slice:
        """Return the shared contiguous official LANGUAGE span."""

        language = native_roles == int(NativeRole.LANGUAGE)
        if not torch.equal(language, language[:1].expand_as(language)):
            raise ValueError("official language span differs across the batch")
        indices = torch.nonzero(language[0], as_tuple=False).flatten()
        if not indices.numel():
            raise ValueError("task-addressed host requires the official language span")
        start = int(indices[0].item())
        stop = int(indices[-1].item()) + 1
        if not language[:, start:stop].all() or language.sum().item() != (
            native_roles.shape[0] * (stop - start)
        ):
            raise ValueError("official language span must be contiguous")
        return slice(start, stop)

    @staticmethod
    def _task_address_native_roles(
        context: LingBotNativeContext,
    ) -> torch.Tensor:
        """Translate the released prefix roles without guessing future queries."""

        native_roles = context.native_roles
        native_valid = context.native_valid
        if native_roles is None or native_valid is None:
            raise RuntimeError("task-addressed context lost its native prefix metadata")
        roles = torch.full_like(
            native_roles,
            int(TaskAddressRole.HOST_CURRENT),
            dtype=torch.long,
        )
        roles[native_roles == int(NativeRole.SENSOR)] = int(TaskAddressRole.SENSOR)
        roles[native_roles == int(NativeRole.LANGUAGE)] = int(TaskAddressRole.LANGUAGE)

        aux_valid = (native_roles == int(NativeRole.HOST_AUX)) & native_valid
        sensor_boundary = context.native_sensor_boundary
        current = context.native_host_current
        future = context.native_host_future
        if sensor_boundary is None:
            sensor_boundary = torch.zeros_like(aux_valid)
        if current is None or future is None:
            if aux_valid.any():
                raise RuntimeError(
                    "task-addressed host requires explicit current/future aux classification"
                )
            return roles
        classified = (sensor_boundary | current | future) & native_valid
        if not torch.equal(classified, aux_valid):
            raise RuntimeError(
                "task-addressed current/future masks must classify every valid aux token"
            )
        roles[sensor_boundary] = int(TaskAddressRole.SENSOR_BOUNDARY)
        roles[current] = int(TaskAddressRole.HOST_CURRENT)
        roles[future] = int(TaskAddressRole.HOST_FUTURE)
        return roles

    @staticmethod
    def _insert_task_address_positions(
        position_ids: torch.Tensor,
        *,
        insertion_index: int,
        inserted_count: int,
        language_slice: slice,
        task_text_slice: slice,
    ) -> torch.Tensor:
        """Insert neutral rows while exactly copying task-token three-axis MRoPE."""

        if position_ids.ndim != 3 or position_ids.shape[0] != 3:
            raise ValueError("LingBot position IDs must have shape [3,batch,tokens]")
        relative_task = slice(
            task_text_slice.start - insertion_index,
            task_text_slice.stop - insertion_index,
        )
        inserted = torch.zeros(
            (3, position_ids.shape[1], inserted_count),
            dtype=position_ids.dtype,
            device=position_ids.device,
        )
        inserted[:, :, relative_task] = position_ids[:, :, language_slice]
        return torch.cat(
            (
                position_ids[:, :, :insertion_index],
                inserted,
                position_ids[:, :, insertion_index:],
            ),
            dim=-1,
        )

    def _prepare_task_address_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotNativeContext,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _TaskAddressRuntime,
    ]:
        """Build the addressed correction/action surface inside the shared host."""

        if len(inputs_embeds) != 2 or inputs_embeds[0] is None:
            raise ValueError("task-addressed PICF requires the official prefix stream")
        if context.posterior_adoption_route is not None:
            raise ValueError("task-addressed PICF does not use posterior-adoption routing")
        if context.posterior_action_row_visible is not None:
            raise ValueError("task-addressed PICF cannot apply direct posterior-row intervention")
        prefix = inputs_embeds[0]
        action = inputs_embeds[1]
        self._validate_context(context, prefix=prefix)
        episode_addresses = self._task_address_correction_addresses(
            context,
            prefix=prefix,
        )
        batch, original_prefix_count, _ = prefix.shape
        native_roles = context.native_roles
        native_valid = context.native_valid
        if native_roles is None or native_valid is None:
            raise RuntimeError("validated task-addressed context lost prefix metadata")
        action_count = 0 if action is None else action.shape[1]
        old_total = original_prefix_count + action_count
        if attention_mask.shape != (batch, old_total, old_total) or (
            attention_mask.dtype != torch.bool
        ):
            raise ValueError("LingBot host mask must be boolean and match prefix plus action")
        if position_ids.shape != (3, batch, old_total):
            raise ValueError("LingBot MRoPE position IDs have an unexpected shape")

        language_slice = self._instruction_span(native_roles)
        task_text = prefix[:, language_slice].clone()
        task_text_valid = native_valid[:, language_slice].clone()
        task_queries = self.task_query_embeddings
        if task_queries is None:
            raise RuntimeError("task-addressed graph omitted its native task queries")
        task_queries = task_queries.unsqueeze(0).expand(batch, -1, -1)
        object_reads = prefix.new_zeros(
            batch,
            self.config.task_query_count,
            self.config.host_width,
        )
        control_features = context.controls.canonical_features().to(
            device=prefix.device,
            dtype=prefix.dtype,
        )
        (
            modalities,
            modality_valid,
            modality_direct_action_visible,
            relation_surfaces,
        ) = self._project_modalities(context, prefix=prefix)
        modality_slices = self._contextual_modality_slices(context)
        if modalities.shape[1]:
            prefix = prefix + modalities.sum(dim=(1, 2), keepdim=True) * 0.0
        controls = self.control_projection(control_features) + self.role_embeddings[2]
        prior, posterior = self._layerwise_initial_rows(batch=batch)
        prediction_request = context.prediction_request
        prediction = (
            None if prediction_request is None else self._prediction_queries(prediction_request)
        )
        inserted_parts = [
            modalities,
            task_text,
            controls,
            prior,
            posterior,
            task_queries,
            object_reads,
        ]
        if prediction is not None:
            inserted_parts.append(prediction)
        inserted = torch.cat(inserted_parts, dim=1)
        expanded_prefix = torch.cat((prefix, inserted), dim=1)

        modality_start = original_prefix_count
        task_text_start = modality_start + modalities.shape[1]
        control_start = task_text_start + task_text.shape[1]
        prior_start = control_start + context.controls.token_count
        posterior_start = prior_start + self.config.capacity
        task_query_start = posterior_start + self.config.capacity
        object_read_start = task_query_start + self.config.task_query_count
        prediction_start = object_read_start + self.config.task_query_count
        modality_slice = slice(modality_start, task_text_start)
        task_text_slice = slice(task_text_start, control_start)
        control_slice = slice(control_start, prior_start)
        prior_slice = slice(prior_start, posterior_start)
        posterior_slice = slice(posterior_start, task_query_start)
        task_query_slice = slice(task_query_start, object_read_start)
        object_read_slice = slice(object_read_start, prediction_start)
        prediction_query_count = 0 if prediction_request is None else prediction_request.query_count
        prediction_slice = (
            None
            if prediction is None
            else slice(prediction_start, prediction_start + prediction.shape[1])
        )

        native_task_roles = self._task_address_native_roles(context)
        action_roles = torch.full(
            (batch, action_count),
            int(TaskAddressRole.ACTION),
            dtype=torch.long,
            device=prefix.device,
        )
        inserted_roles = torch.tensor(
            [
                *([int(TaskAddressRole.SENSOR)] * modalities.shape[1]),
                *([int(TaskAddressRole.TASK_TEXT)] * task_text.shape[1]),
                *([int(TaskAddressRole.CONTROL)] * context.controls.token_count),
                *([int(TaskAddressRole.PRIOR)] * self.config.capacity),
                *([int(TaskAddressRole.POSTERIOR)] * self.config.capacity),
                *([int(TaskAddressRole.TASK_QUERY)] * self.config.task_query_count),
                *([int(TaskAddressRole.OBJECT_READ)] * self.config.task_query_count),
                *(
                    [int(TaskAddressRole.PREDICT)]
                    * (0 if prediction is None else prediction.shape[1])
                ),
            ],
            dtype=torch.long,
            device=prefix.device,
        ).expand(batch, -1)
        roles = torch.cat((native_task_roles, inserted_roles, action_roles), dim=1)
        inserted_valid = torch.cat(
            (
                modality_valid,
                task_text_valid,
                context.controls.token_valid,
                torch.ones(
                    (
                        batch,
                        2 * self.config.capacity + 2 * self.config.task_query_count,
                    ),
                    dtype=torch.bool,
                    device=prefix.device,
                ),
                *(
                    ()
                    if prediction_request is None
                    else (
                        prediction_request.valid.unsqueeze(1)
                        .expand(-1, self.config.capacity, -1)
                        .flatten(1, 2),
                    )
                ),
            ),
            dim=1,
        )
        action_valid = attention_mask.diagonal(dim1=-2, dim2=-1)[:, original_prefix_count:]
        valid = torch.cat((native_valid, inserted_valid, action_valid), dim=1)
        layout = TaskAddressTokenLayout(roles=roles, valid=valid)
        expanded_host_mask = expand_square_mask(
            attention_mask,
            insertion_index=original_prefix_count,
            inserted_count=inserted.shape[1],
        )
        unified_mask = task_address_attention_mask(
            layout,
            host_mask=expanded_host_mask,
            control_slice=control_slice,
            state_slices=TaskAddressStateSlices(
                prior=prior_slice,
                posterior=posterior_slice,
                capacity=self.config.capacity,
            ),
            action_information_sets=context.action_information_sets,
        )
        source_row_visible = context.object_read_source_row_visible
        if source_row_visible is not None:
            object_read_queries = roles == int(TaskAddressRole.OBJECT_READ)
            object_read_source_visible = ~object_read_queries.unsqueeze(
                -1
            ) | source_row_visible.unsqueeze(1)
            for source_slice in (prior_slice, posterior_slice):
                source_mask = unified_mask[:, :, source_slice].clone()
                source_mask &= object_read_source_visible
                unified_mask[:, :, source_slice] = source_mask
        if context.object_read_action_intervention is ObjectReadActionIntervention.BLOCKED:
            action_queries = roles == int(TaskAddressRole.ACTION)
            action_to_object_read = unified_mask[:, :, object_read_slice].clone()
            action_to_object_read &= ~action_queries.unsqueeze(-1)
            unified_mask[:, :, object_read_slice] = action_to_object_read
        if modality_slice.stop > modality_slice.start:
            action_queries = roles == int(TaskAddressRole.ACTION)
            action_to_modality = unified_mask[:, :, modality_slice].clone()
            action_to_modality &= ~(
                action_queries.unsqueeze(-1) & ~modality_direct_action_visible.unsqueeze(1)
            )
            unified_mask[:, :, modality_slice] = action_to_modality
        if prediction_slice is not None:
            if prediction_request is None:
                raise RuntimeError("prediction tokens were inserted without a request")
            source_slice = (
                prior_slice
                if prediction_request.source == PredictionSource.PRIOR
                else posterior_slice
            )
            unified_mask = self._restrict_prediction_queries(
                unified_mask,
                request=prediction_request,
                source_slice=source_slice,
                prediction_slice=prediction_slice,
                capacity=self.config.capacity,
            )
        expanded_positions = self._insert_task_address_positions(
            position_ids,
            insertion_index=original_prefix_count,
            inserted_count=inserted.shape[1],
            language_slice=language_slice,
            task_text_slice=task_text_slice,
        )
        if visual_pos_masks is not None:
            if visual_pos_masks.shape != (batch, original_prefix_count):
                raise ValueError("visual_pos_masks does not match the native prefix")
            visual_pos_masks = torch.cat(
                (
                    visual_pos_masks,
                    torch.zeros(
                        (batch, inserted.shape[1]),
                        dtype=torch.bool,
                        device=prefix.device,
                    ),
                ),
                dim=1,
            )

        context.expanded_cache_valid = valid
        context.expanded_cache_position_ids = expanded_positions
        context.expanded_posterior_indices = torch.arange(
            posterior_slice.start,
            posterior_slice.stop,
            dtype=torch.long,
            device=prefix.device,
        )
        context.expanded_posterior_valid = valid[:, posterior_slice]
        if context.task_address_attention_layout is not None:
            raise RuntimeError("task-address attention layout may be bound only once")
        context.task_address_attention_layout = TaskAddressAttentionLayout(
            batch_size=batch,
            query_count=layout.token_count,
            capacity=self.config.capacity,
            object_read_slice=object_read_slice,
            prior_slice=prior_slice,
            posterior_slice=posterior_slice,
        )
        action_visible = task_address_action_key_visibility(
            layout,
            action_information_sets=context.action_information_sets,
        )
        if modality_slice.stop > modality_slice.start:
            action_visible[:, modality_slice] &= modality_direct_action_visible
        if context.object_read_action_intervention is ObjectReadActionIntervention.BLOCKED:
            action_visible[:, object_read_slice] = False
        context.expanded_action_cache_visible = action_visible
        runtime = _TaskAddressRuntime(
            context=context,
            original_prefix_count=original_prefix_count,
            language_slice=language_slice,
            task_text_slice=task_text_slice,
            modality_slice=modality_slice,
            modality_valid=modality_valid,
            modality_slices=modality_slices,
            relation_surfaces=relation_surfaces,
            control_slice=control_slice,
            prior_slice=prior_slice,
            posterior_slice=posterior_slice,
            task_query_slice=task_query_slice,
            object_read_slice=object_read_slice,
            prediction_slice=prediction_slice,
            prediction_query_count=prediction_query_count,
            layout=layout,
            episode_addresses=episode_addresses,
        )
        return (
            [expanded_prefix, action],
            unified_mask,
            expanded_positions,
            visual_pos_masks,
            runtime,
        )

    @overload
    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotNativeContext,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _NativeRuntime,
    ]: ...

    @overload
    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotPriorRolloutContext,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _PriorRolloutRuntime,
    ]: ...

    @overload
    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: None,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        None,
    ]: ...

    @overload
    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotNativeContext | LingBotPriorRolloutContext | None,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _Runtime | None,
    ]: ...

    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotNativeContext | LingBotPriorRolloutContext | None,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _Runtime | None,
    ]:
        if context is None:
            return inputs_embeds, attention_mask, position_ids, visual_pos_masks, None
        if isinstance(context, LingBotPriorRolloutContext):
            return self._prepare_prior_rollout_inputs(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                context=context,
            )
        if self.task_query_object_value_read and context.posterior_adoption_route is None:
            return self._prepare_task_address_joint_inputs(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                context=context,
            )
        if len(inputs_embeds) != 2:
            raise ValueError("LingBot-native PICF requires the official prefix stream")
        prefix = inputs_embeds[0]
        action = inputs_embeds[1]
        if prefix is None:
            raise ValueError("LingBot-native PICF requires the official prefix stream")
        self._validate_context(context, prefix=prefix)
        batch, original_prefix_count, _ = prefix.shape
        episode_addresses = (
            self._task_address_correction_addresses(context, prefix=prefix)
            if self.task_query_object_value_read
            else None
        )
        native_roles = context.native_roles
        native_valid = context.native_valid
        if native_roles is None or native_valid is None:
            raise RuntimeError("validated native context lost prefix metadata")
        action_count = 0 if action is None else action.shape[1]
        old_total = original_prefix_count + action_count
        if (
            attention_mask.shape != (batch, old_total, old_total)
            or attention_mask.dtype != torch.bool
        ):
            raise ValueError("LingBot host mask must be boolean and match prefix plus action")
        if position_ids.shape != (3, batch, old_total):
            raise ValueError("LingBot MRoPE position IDs have an unexpected shape")

        control_features = context.controls.canonical_features().to(
            device=prefix.device,
            dtype=prefix.dtype,
        )
        (
            modalities,
            modality_valid,
            modality_direct_action_visible,
            relation_surfaces,
        ) = self._project_modalities(context, prefix=prefix)
        modality_slices = self._contextual_modality_slices(context)
        if modalities.shape[1]:
            # Invalid optional rows must remain numerically absent while their
            # adapters retain an exact-zero action-loss path on every rank.
            prefix = prefix + modalities.sum(dim=(1, 2), keepdim=True) * 0.0
        controls = self.control_projection(control_features) + self.role_embeddings[2]
        if self.native_videomt_query_posterior:
            source_seed, source_valid = self._native_videomt_posterior_seed(
                context,
                prefix=prefix,
            )
            prior, _neutral_posterior = self._layerwise_initial_rows(batch=batch)
            posterior = source_seed + self.role_embeddings[1]
            if not source_valid.all():
                raise RuntimeError("native VidEoMT posterior unexpectedly lost a source query")
            object_content = None
        elif self.layerwise_recurrence:
            prior, posterior = self._layerwise_initial_rows(batch=batch)
            object_content = None
        else:
            object_content = self._object_content_seed(context, prefix=prefix)
            prior, posterior = self._initial_rows(object_content)
        if self.task_independent:
            match = None
        else:
            if object_content is None:
                raise RuntimeError("legacy task-match graph omitted its object-content seed")
            match = self._match_queries(object_content)
        prediction_request = context.prediction_request
        prediction = (
            None if prediction_request is None else self._prediction_queries(prediction_request)
        )
        inserted_parts = [modalities, controls, prior, posterior]
        if match is not None:
            inserted_parts.append(match)
        if prediction is not None:
            inserted_parts.append(prediction)
        inserted = torch.cat(inserted_parts, dim=1)
        expanded_prefix = torch.cat((prefix, inserted), dim=1)
        modality_start = original_prefix_count
        control_start = modality_start + modalities.shape[1]
        prior_start = control_start + context.controls.token_count
        posterior_start = prior_start + self.config.capacity
        posterior_stop = posterior_start + self.config.capacity
        match_start = posterior_stop
        match_stop = match_start + (0 if match is None else self.config.capacity)
        control_slice = slice(control_start, prior_start)
        prior_slice = slice(prior_start, posterior_start)
        posterior_slice = slice(posterior_start, posterior_stop)
        match_slice = None if match is None else slice(match_start, match_stop)
        prediction_query_count = 0 if prediction_request is None else prediction_request.query_count
        prediction_slice = (
            None if prediction is None else slice(match_stop, match_stop + prediction.shape[1])
        )

        action_roles = torch.full(
            (batch, action_count),
            int(NativeRole.ACTION),
            dtype=torch.long,
            device=prefix.device,
        )
        action_valid = attention_mask.diagonal(dim1=-2, dim2=-1)[:, original_prefix_count:]
        inserted_roles = torch.tensor(
            [
                *([int(NativeRole.SENSOR)] * modalities.shape[1]),
                *([int(NativeRole.CONTROL)] * context.controls.token_count),
                *([int(NativeRole.PRIOR)] * self.config.capacity),
                *([int(NativeRole.POSTERIOR)] * self.config.capacity),
                *([int(NativeRole.MATCH)] * (0 if match is None else self.config.capacity)),
                *([int(NativeRole.PREDICT)] * (0 if prediction is None else prediction.shape[1])),
            ],
            dtype=torch.long,
            device=prefix.device,
        ).expand(batch, -1)
        roles = torch.cat((native_roles, inserted_roles, action_roles), dim=1)
        inserted_valid = torch.cat(
            (
                modality_valid,
                context.controls.token_valid,
                torch.ones(
                    (batch, (2 if match is None else 3) * self.config.capacity),
                    dtype=torch.bool,
                    device=prefix.device,
                ),
                *(
                    ()
                    if prediction_request is None
                    else (
                        prediction_request.valid.unsqueeze(1)
                        .expand(-1, self.config.capacity, -1)
                        .flatten(1, 2),
                    )
                ),
            ),
            dim=1,
        )
        valid = torch.cat((native_valid, inserted_valid, action_valid), dim=1)
        layout = NativeTokenLayout(roles=roles, valid=valid)
        direct_action_visible = torch.zeros_like(valid)
        direct_action_visible[:, modality_start:control_start] = modality_direct_action_visible
        expanded_host_mask = expand_square_mask(
            attention_mask,
            insertion_index=original_prefix_count,
            inserted_count=inserted.shape[1],
        )
        unified_mask = native_attention_mask(
            layout,
            host_mask=expanded_host_mask,
            control_slice=control_slice,
        )
        if match_slice is not None:
            unified_mask = self._restrict_match_set(
                unified_mask,
                layout=layout,
                match_slice=match_slice,
                capacity=self.config.capacity,
            )
        if prediction_slice is not None:
            if prediction_request is None:
                raise RuntimeError("prediction tokens were inserted without a request")
            source_slice = (
                prior_slice
                if prediction_request.source == PredictionSource.PRIOR
                else posterior_slice
            )
            unified_mask = self._restrict_prediction_queries(
                unified_mask,
                request=prediction_request,
                source_slice=source_slice,
                prediction_slice=prediction_slice,
                capacity=self.config.capacity,
            )
        posterior_adoption_route = context.posterior_adoption_route
        if posterior_adoption_route is not None:
            unified_mask = posterior_adoption_attention_mask(
                unified_mask,
                layout=layout,
                enabled=posterior_adoption_route,
                direct_action_visible=direct_action_visible,
            )
        expanded_positions = self._insert_positions(
            position_ids,
            insertion_index=original_prefix_count,
            inserted_count=inserted.shape[1],
        )
        if visual_pos_masks is not None:
            if visual_pos_masks.shape != (batch, original_prefix_count):
                raise ValueError("visual_pos_masks does not match the native prefix")
            visual_pos_masks = torch.cat(
                (
                    visual_pos_masks,
                    torch.zeros(
                        (batch, inserted.shape[1]),
                        dtype=torch.bool,
                        device=prefix.device,
                    ),
                ),
                dim=1,
            )

        context.expanded_cache_valid = valid
        context.expanded_cache_position_ids = expanded_positions
        context.expanded_posterior_indices = torch.arange(
            posterior_slice.start,
            posterior_slice.stop,
            dtype=torch.long,
            device=prefix.device,
        )
        context.expanded_posterior_valid = valid[:, posterior_slice]
        if posterior_adoption_route is None:
            action_permissions = native_role_permissions(device=prefix.device)[
                int(NativeRole.ACTION)
            ]
            context.expanded_action_cache_visible = action_permissions[roles] & valid
        else:
            context.expanded_action_cache_visible = posterior_adoption_action_key_visibility(
                layout,
                enabled=posterior_adoption_route,
                direct_action_visible=direct_action_visible,
            )
        posterior_action_row_visible = context.posterior_action_row_visible
        if posterior_action_row_visible is not None:
            if posterior_action_row_visible.shape != (batch, self.config.capacity):
                raise ValueError(
                    "posterior action-row visibility differs from the native graph capacity"
                )
            posterior_visibility = context.expanded_action_cache_visible[:, posterior_slice].clone()
            posterior_visibility &= posterior_action_row_visible
            context.expanded_action_cache_visible[:, posterior_slice] = posterior_visibility
        runtime = _NativeRuntime(
            context=context,
            original_prefix_count=original_prefix_count,
            modality_slice=slice(modality_start, control_start),
            modality_valid=modality_valid,
            modality_slices=modality_slices,
            relation_surfaces=relation_surfaces,
            control_slice=control_slice,
            prior_slice=prior_slice,
            posterior_slice=posterior_slice,
            match_slice=match_slice,
            prediction_slice=prediction_slice,
            prediction_query_count=prediction_query_count,
            layout=layout,
            episode_addresses=episode_addresses,
        )
        return (
            [expanded_prefix, action],
            unified_mask,
            expanded_positions,
            visual_pos_masks,
            runtime,
        )

    def _prepare_prior_rollout_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotPriorRolloutContext,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _PriorRolloutRuntime,
    ]:
        if len(inputs_embeds) != 2 or inputs_embeds[0] is None or inputs_embeds[1] is not None:
            raise ValueError("row-only rollout requires an empty official prefix and no action")
        prefix = inputs_embeds[0]
        batch = context.controls.batch_size
        if prefix.shape != (batch, 0, self.config.host_width):
            raise ValueError("row-only rollout prefix must be empty [batch,0,host_width]")
        if attention_mask.shape != (batch, 0, 0) or attention_mask.dtype != torch.bool:
            raise ValueError("row-only rollout host mask must be empty boolean")
        if position_ids.shape != (3, batch, 0):
            raise ValueError("row-only rollout position IDs must be empty MRoPE")
        if visual_pos_masks is not None and visual_pos_masks.shape != (batch, 0):
            raise ValueError("row-only rollout visual mask must be absent or empty")
        context.controls.validate_bound(self.config.maximum_control_tokens)
        if context.controls.action_dim != self.config.executed_action_dim:
            raise ValueError("row-only executed-control width differs from the graph contract")
        if (
            context.controls.values.device != prefix.device
            or context.controls.values.dtype != prefix.dtype
            or prefix.device != self._parameter_reference.device
            or prefix.dtype != self._parameter_reference.dtype
        ):
            raise ValueError(
                "row-only controls, graph prefix and host parameters must align: "
                f"controls={context.controls.values.device}/{context.controls.values.dtype}, "
                f"prefix={prefix.device}/{prefix.dtype}, "
                f"parameters={self._parameter_reference.device}/"
                f"{self._parameter_reference.dtype}"
            )
        previous_state_valid = context.previous_state_valid
        previous_memory_valid = context.previous_memory_valid
        source_prior_trace_valid = context.source_prior_trace_valid
        if (
            previous_state_valid is None
            or previous_memory_valid is None
            or source_prior_trace_valid is None
        ):
            raise RuntimeError("row-only context lost recurrent validity metadata")
        episode_addresses: torch.Tensor | None = None
        if self.unified_predict_correct:
            if context.previous_state is not None or previous_state_valid.any():
                raise ValueError("unified prior-only pass requires a layerwise posterior trace")
            if context.previous_memory is not None and context.source_prior_trace is not None:
                raise ValueError("unified prior pass cannot mix posterior and prior sources")
            memory = (
                context.previous_memory
                if context.previous_memory is not None
                else context.source_prior_trace
            )
            if memory is not None:
                if not isinstance(
                    memory,
                    NativeLayerwisePosteriorState | NativeLayerwisePriorTrace,
                ):
                    raise TypeError(
                        "unified prior-only pass requires typed layerwise posterior/prior memory"
                    )
                if (
                    memory.batch_size != batch
                    or memory.num_layers != self.config.num_layers
                    or memory.capacity != self.config.capacity
                    or memory.host_width != self.config.host_width
                ):
                    raise ValueError("prior-only posterior trace differs from the v3 host contract")
                if (
                    memory.layer_rows.device != prefix.device
                    or memory.layer_rows.dtype != prefix.dtype
                ):
                    raise ValueError("prior-only source trace must share LingBot device and dtype")
            source_valid = (
                previous_memory_valid
                if context.previous_memory is not None
                else source_prior_trace_valid
            )
            if memory is None and source_valid.any():
                raise ValueError("unified prior validity selected an absent source trace")
            if self.task_query_object_value_read:
                _address_state, source_valid, episode_addresses = (
                    self._resolve_prior_episode_addresses(
                        context,
                        source_valid=source_valid,
                        source_prior_trace=context.source_prior_trace,
                        device=prefix.device,
                        dtype=prefix.dtype,
                    )
                )
                if context.previous_memory is not None:
                    context.previous_memory_valid = source_valid
                else:
                    context.source_prior_trace_valid = source_valid
            prior, _posterior = self._layerwise_initial_rows(batch=batch)
        else:
            if context.previous_memory is not None or previous_memory_valid.any():
                raise ValueError("non-v3 row-only pass cannot consume a layerwise posterior trace")
            prior = self._row_only_prior(context)
        control_features = context.controls.canonical_features().to(
            device=prefix.device,
            dtype=prefix.dtype,
        )
        controls = self.control_projection(control_features) + self.role_embeddings[2]
        prediction_request = context.prediction_request
        if prediction_request is not None:
            if self.unified_predict_correct:
                if prediction_request.evidence not in {
                    PredictionEvidence.FUTURE,
                    PredictionEvidence.CURRENT_PRIOR,
                }:
                    raise ValueError(
                        "v3 prior-only prediction requires FUTURE or CURRENT_PRIOR evidence"
                    )
            elif prediction_request.evidence is not PredictionEvidence.FUTURE:
                raise ValueError("legacy prior-only prediction requires FUTURE evidence")
        prediction = (
            None if prediction_request is None else self._prediction_queries(prediction_request)
        )
        inserted = torch.cat(
            (controls, prior) if prediction is None else (controls, prior, prediction),
            dim=1,
        )
        control_count = context.controls.token_count
        control_slice = slice(0, control_count)
        prior_slice = slice(control_count, control_count + self.config.capacity)
        prediction_query_count = 0 if prediction_request is None else prediction_request.query_count
        prediction_slice = (
            None
            if prediction is None
            else slice(prior_slice.stop, prior_slice.stop + prediction.shape[1])
        )
        roles = torch.tensor(
            [
                *([int(NativeRole.CONTROL)] * control_count),
                *([int(NativeRole.PRIOR)] * self.config.capacity),
                *([int(NativeRole.PREDICT)] * (0 if prediction is None else prediction.shape[1])),
            ],
            dtype=torch.long,
            device=prefix.device,
        ).expand(batch, -1)
        valid = torch.cat(
            (
                context.controls.token_valid,
                torch.ones(
                    batch,
                    self.config.capacity,
                    dtype=torch.bool,
                    device=prefix.device,
                ),
                *(
                    ()
                    if prediction_request is None
                    else (
                        prediction_request.valid.unsqueeze(1)
                        .expand(-1, self.config.capacity, -1)
                        .flatten(1, 2),
                    )
                ),
            ),
            dim=1,
        )
        layout = NativeTokenLayout(roles=roles, valid=valid)
        host_mask = torch.ones(
            batch,
            inserted.shape[1],
            inserted.shape[1],
            dtype=torch.bool,
            device=prefix.device,
        )
        unified_mask = native_attention_mask(
            layout,
            host_mask=host_mask,
            control_slice=control_slice,
        )
        if self.task_query_object_value_read:
            row = torch.arange(self.config.capacity, device=prefix.device)
            prior_query = prior_slice.start + row
            same_row = unified_mask[:, prior_query, prior_slice.start + row].clone()
            unified_mask[:, prior_query, prior_slice] = False
            unified_mask[:, prior_query, prior_slice.start + row] = same_row
        if prediction_slice is not None:
            if prediction_request is None:
                raise RuntimeError("row-only prediction tokens were inserted without a request")
            if (prediction_request.route_ids >= self.config.prediction_route_count).any():
                raise ValueError("row-only prediction request uses an undeclared route")
            if prediction_request.address_width != self.config.prediction_address_width:
                raise ValueError("row-only prediction address width differs from graph contract")
            if prediction_request.addresses.dtype != prefix.dtype:
                raise ValueError("row-only prediction addresses and graph must share one dtype")
            unified_mask = self._restrict_prediction_queries(
                unified_mask,
                request=prediction_request,
                source_slice=prior_slice,
                prediction_slice=prediction_slice,
                capacity=self.config.capacity,
            )
        positions = torch.zeros(
            3,
            batch,
            inserted.shape[1],
            dtype=position_ids.dtype,
            device=prefix.device,
        )
        visual = (
            None
            if visual_pos_masks is None
            else torch.zeros(
                batch,
                inserted.shape[1],
                dtype=torch.bool,
                device=prefix.device,
            )
        )
        runtime = _PriorRolloutRuntime(
            context=context,
            prior_slice=prior_slice,
            prediction_slice=prediction_slice,
            prediction_query_count=prediction_query_count,
            layout=layout,
            episode_addresses=episode_addresses,
        )
        return [inserted, None], unified_mask, positions, visual, runtime

    def finalize_joint_outputs(
        self,
        *,
        outputs_embeds: list[torch.Tensor | None],
        runtime: _Runtime | None,
    ) -> list[torch.Tensor | None]:
        if runtime is None:
            return outputs_embeds
        if len(outputs_embeds) != 2 or outputs_embeds[0] is None:
            raise RuntimeError("LingBot did not return its normalized prefix stream")
        if isinstance(runtime, _PriorRolloutRuntime):
            context = runtime.context
            if context._finalized:
                raise RuntimeError("a row-only rollout context may be finalized only once")
            context.prior_state = NativePosteriorState(outputs_embeds[0][:, runtime.prior_slice])
            if self.unified_predict_correct:
                if len(runtime.layerwise_outputs) != self.config.num_layers:
                    raise RuntimeError("unified prior pass did not capture every host layer")
                layer_rows = torch.stack(runtime.layerwise_outputs, dim=1)
                if self.task_query_object_value_read:
                    if context.episode_address_state is None:
                        raise RuntimeError("task-addressed prior lost its episode address state")
                    context.prior_trace = AddressedLayerwisePriorTrace(
                        layer_rows=layer_rows,
                        episode_address_state=context.episode_address_state,
                        architecture_identity=self.config.architecture_identity,
                    )
                else:
                    context.prior_trace = NativeLayerwisePriorTrace(layer_rows)
            elif runtime.layerwise_outputs:
                raise RuntimeError("legacy row-only prior unexpectedly produced a layerwise trace")
            if runtime.prediction_slice is not None:
                if context.prediction_request is None:
                    raise RuntimeError("row-only prediction tokens lack their typed request")
                prediction = outputs_embeds[0][:, runtime.prediction_slice]
                context.prediction_hidden = prediction.reshape(
                    prediction.shape[0],
                    self.config.capacity,
                    runtime.prediction_query_count,
                    self.config.host_width,
                )
                context.prediction_outputs = self._project_prediction_outputs(
                    context.prediction_hidden,
                    context.prediction_request,
                )
            context._finalized = True
            return outputs_embeds
        context = runtime.context
        if context._finalized:
            raise RuntimeError("a LingBot-native context may be finalized only once")
        prefix = outputs_embeds[0]
        prior = (
            context.prior_trace.layer(self.config.num_layers - 1)
            if self.unified_predict_correct and context.prior_trace is not None
            else prefix[:, runtime.prior_slice]
        )
        posterior = prefix[:, runtime.posterior_slice]
        context.prior_state = NativePosteriorState(prior)
        context.posterior_state = NativePosteriorState(posterior)
        if self.layerwise_recurrence:
            if len(runtime.layerwise_outputs) != self.config.num_layers:
                raise RuntimeError("layerwise PICF did not capture exactly one posterior per layer")
            layer_rows = torch.stack(runtime.layerwise_outputs, dim=1)
            if self.task_query_object_value_read:
                if context.episode_address_state is None:
                    raise RuntimeError("task-addressed correction lost its episode address state")
                context.posterior_memory = AddressedLayerwisePosteriorState(
                    layer_rows=layer_rows,
                    episode_address_state=context.episode_address_state,
                    architecture_identity=self.config.architecture_identity,
                )
            else:
                context.posterior_memory = NativeLayerwisePosteriorState(layer_rows)
        if runtime.prediction_slice is not None:
            if context.prediction_request is None:
                raise RuntimeError("native prediction tokens lack their typed request")
            prediction = prefix[:, runtime.prediction_slice]
            context.prediction_hidden = prediction.reshape(
                prediction.shape[0],
                self.config.capacity,
                runtime.prediction_query_count,
                self.config.host_width,
            )
            context.prediction_outputs = self._project_prediction_outputs(
                context.prediction_hidden,
                context.prediction_request,
            )
        expected_intermediate = (
            set(self.config.relation_supervision_layers)
            if context.supervise_intermediate_relations
            else set()
        )
        if set(context.intermediate_relation_outputs) != expected_intermediate:
            raise RuntimeError(
                "intermediate relation outputs differ from the configured host depths"
            )
        context.relation_output = self._read_relation(prefix=prefix, runtime=runtime)
        context._finalized = True
        return outputs_embeds

    def layerwise_qk_address_bias(
        self,
        *,
        prefix_hidden: torch.Tensor,
        runtime: _Runtime | None,
    ) -> torch.Tensor | None:
        """Return DETR-style object addresses for Q/K only; Value remains content-only."""

        if not self.layerwise_recurrence or not isinstance(
            runtime, (_NativeRuntime, _TaskAddressRuntime, _PriorRolloutRuntime)
        ):
            return None
        if isinstance(runtime, _PriorRolloutRuntime):
            prefix_count = (
                runtime.prediction_slice.stop
                if runtime.prediction_slice is not None
                else runtime.prior_slice.stop
            )
        elif isinstance(runtime, _TaskAddressRuntime):
            prefix_count = (
                runtime.prediction_slice.stop
                if runtime.prediction_slice is not None
                else runtime.object_read_slice.stop
            )
        else:
            prefix_count = (
                runtime.prediction_slice.stop
                if runtime.prediction_slice is not None
                else (
                    runtime.match_slice.stop
                    if runtime.match_slice is not None
                    else runtime.posterior_slice.stop
                )
            )
        if (
            prefix_hidden.ndim != 3
            or prefix_hidden.shape[:2] != (runtime.layout.batch_size, prefix_count)
            or prefix_hidden.shape[-1] != self.config.host_width
        ):
            raise ValueError("layerwise Q/K address input differs from the native token layout")
        bias = torch.zeros_like(prefix_hidden)
        if isinstance(runtime, _TaskAddressRuntime):
            query = prefix_hidden[:, runtime.task_query_slice]
            expected = (
                prefix_hidden.shape[0],
                self.config.task_query_count,
                self.config.host_width,
            )
            if query.shape != expected or (
                runtime.object_read_slice.stop - runtime.object_read_slice.start
                != self.config.task_query_count
            ):
                raise RuntimeError("task-query/object-read pairing differs from the config")
            if runtime.episode_addresses.shape != (
                prefix_hidden.shape[0],
                self.config.capacity,
                self.config.host_width,
            ):
                raise RuntimeError("task-addressed runtime lost its episode row addresses")
            bias[:, runtime.prior_slice] = runtime.episode_addresses
            bias[:, runtime.posterior_slice] = runtime.episode_addresses
            bias[:, runtime.object_read_slice] = query
            return bias
        if self.task_query_object_value_read and isinstance(
            runtime,
            (_NativeRuntime, _PriorRolloutRuntime),
        ):
            address = runtime.episode_addresses
            if address is None:
                raise RuntimeError("task-addressed native runtime lost its episode addresses")
            bias[:, runtime.prior_slice] = address
            if isinstance(runtime, _NativeRuntime):
                bias[:, runtime.posterior_slice] = address
            return bias
        if self.object_addresses is None:
            raise RuntimeError("layerwise object addresses are unavailable")
        address = self.object_addresses.unsqueeze(0).expand(prefix_hidden.shape[0], -1, -1)
        bias[:, runtime.prior_slice] = address
        if isinstance(runtime, _NativeRuntime):
            bias[:, runtime.posterior_slice] = address
        return bias

    def layerwise_memory_inputs(
        self,
        *,
        layer_index: int,
        runtime: _Runtime | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Return same-layer hidden memory, its Q/K address, and paired visibility."""

        if not self.layerwise_recurrence or not isinstance(
            runtime, (_NativeRuntime, _TaskAddressRuntime, _PriorRolloutRuntime)
        ):
            return None
        if (
            isinstance(layer_index, bool)
            or not isinstance(layer_index, int)
            or not 0 <= layer_index < self.config.num_layers
        ):
            raise IndexError("layerwise memory request is outside the LingBot host depth")
        if self.unified_predict_correct:
            if isinstance(runtime, _PriorRolloutRuntime):
                memory = (
                    runtime.context.previous_memory
                    if runtime.context.previous_memory is not None
                    else runtime.context.source_prior_trace
                )
                if memory is None:
                    return None
                valid = (
                    runtime.context.previous_memory_valid
                    if runtime.context.previous_memory is not None
                    else runtime.context.source_prior_trace_valid
                )
                if valid is None:
                    raise RuntimeError("unified prior pass lost source-trace validity")
                visibility = native_layerwise_prior_history_mask(
                    runtime.layout,
                    prior_slice=runtime.prior_slice,
                    capacity=self.config.capacity,
                    previous_memory_valid=valid,
                )
            elif isinstance(runtime, _TaskAddressRuntime):
                memory = runtime.context.prior_trace
                if memory is None:
                    raise RuntimeError("task-addressed correction lost its prior trace")
                visibility = task_address_layerwise_state_mask(
                    runtime.layout,
                    prior_slice=runtime.prior_slice,
                    posterior_slice=runtime.posterior_slice,
                    capacity=self.config.capacity,
                    state_valid=torch.ones(
                        runtime.layout.batch_size,
                        dtype=torch.bool,
                        device=runtime.layout.roles.device,
                    ),
                )
            else:
                memory = runtime.context.prior_trace
                if memory is None:
                    raise RuntimeError("unified correction lost its completed prior trace")
                visibility = native_layerwise_prior_trace_mask(
                    runtime.layout,
                    prior_slice=runtime.prior_slice,
                    posterior_slice=runtime.posterior_slice,
                    capacity=self.config.capacity,
                )
        else:
            if not isinstance(runtime, _NativeRuntime):
                return None
            memory = runtime.context.previous_memory
            if memory is None:
                return None
            valid = runtime.context.previous_memory_valid
            if valid is None:
                raise RuntimeError("layerwise memory lost its batch validity")
            visibility = native_layerwise_history_mask(
                runtime.layout,
                prior_slice=runtime.prior_slice,
                posterior_slice=runtime.posterior_slice,
                capacity=self.config.capacity,
                previous_memory_valid=valid,
            )
        hidden = memory.layer(layer_index)
        if self.task_query_object_value_read:
            address = runtime.episode_addresses
            if address is None or address.shape != hidden.shape:
                raise RuntimeError("task-addressed memory lost its episode Q/K addresses")
        else:
            if self.object_addresses is None:
                raise RuntimeError("layerwise object addresses are unavailable")
            address = self.object_addresses.unsqueeze(0).expand(hidden.shape[0], -1, -1)
        return hidden, address, visibility

    def record_layerwise_posterior(
        self,
        *,
        prefix_hidden: torch.Tensor,
        runtime: _Runtime | None,
        layer_index: int,
    ) -> None:
        """Capture one layer output through the patch's shared trace callback.

        The callback name is retained for the v2 patch ABI. Under v3 it records
        PRIOR rows in the prior-only pass and attached POSTERIOR rows in the
        correction/action pass.
        """

        if not self.layerwise_recurrence or not isinstance(
            runtime, (_NativeRuntime, _TaskAddressRuntime, _PriorRolloutRuntime)
        ):
            return
        if isinstance(runtime, _PriorRolloutRuntime) and not self.unified_predict_correct:
            return
        if layer_index != len(runtime.layerwise_outputs):
            raise RuntimeError("layerwise trace callbacks must follow exact host depth order")
        if layer_index >= self.config.num_layers:
            raise RuntimeError("layerwise trace callback exceeds the configured host depth")
        row_slice = (
            runtime.prior_slice
            if isinstance(runtime, _PriorRolloutRuntime)
            else runtime.posterior_slice
        )
        selected = prefix_hidden[:, row_slice]
        rows = selected.clone() if self.unified_predict_correct else selected.detach().clone()
        expected = (
            prefix_hidden.shape[0],
            self.config.capacity,
            self.config.host_width,
        )
        if rows.shape != expected:
            raise RuntimeError("layerwise trace callback selected an invalid row surface")
        runtime.layerwise_outputs.append(rows)

    def requires_intermediate_relation(
        self,
        *,
        layer_index: int,
        runtime: _Runtime | None,
    ) -> bool:
        """Return whether this native forward requires one attached depth read."""

        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("relation supervision layer index must be an integer")
        return (
            isinstance(runtime, (_NativeRuntime, _TaskAddressRuntime))
            and runtime.context.supervise_intermediate_relations
            and layer_index in self.config.relation_supervision_layers
        )

    def record_intermediate_relation(
        self,
        *,
        normalized_prefix: torch.Tensor,
        runtime: _Runtime | None,
        layer_index: int,
    ) -> None:
        """Record a training-only relation from one native shared-host depth."""

        if not self.requires_intermediate_relation(
            layer_index=layer_index,
            runtime=runtime,
        ):
            raise RuntimeError("unexpected intermediate relation callback")
        if not isinstance(runtime, (_NativeRuntime, _TaskAddressRuntime)):
            raise RuntimeError("intermediate relation callback lacks a native runtime")
        context = runtime.context
        if context._finalized:
            raise RuntimeError("cannot record an intermediate relation after finalization")
        if layer_index in context.intermediate_relation_outputs:
            raise RuntimeError("an intermediate relation depth may be recorded only once")
        relation = self._read_relation(prefix=normalized_prefix, runtime=runtime)
        values = dict(context.intermediate_relation_outputs)
        values[layer_index] = relation
        context.intermediate_relation_outputs = MappingProxyType(values)

    def _read_relation(
        self,
        *,
        prefix: torch.Tensor,
        runtime: _NativeRuntime | _TaskAddressRuntime,
    ) -> RelationOutput | PhysicalRelationOutput | NativeObjectQueryPosteriorOutput:
        """Apply the one shared relation definition to one normalized host surface."""

        required_stop = (
            runtime.posterior_slice.stop
            if runtime.match_slice is None
            else runtime.match_slice.stop
        )
        if (
            prefix.ndim != 3
            or prefix.shape[1] < required_stop
            or prefix.shape[-1] != self.config.host_width
        ):
            raise ValueError("normalized relation prefix has an invalid shape")
        context = runtime.context
        original = prefix[:, : runtime.original_prefix_count]
        modality_hidden = prefix[:, runtime.modality_slice]
        posterior = prefix[:, runtime.posterior_slice]
        if self.native_videomt_query_posterior:
            if context.modalities is None:
                raise RuntimeError("native query posterior lost its source modalities")
            relations = context.modalities.object_query_spatial_relations
            if len(relations) != 1:
                raise RuntimeError("native query posterior requires one source spatial relation")
            relation = relations[0]
            if relation.query_count != self.config.capacity:
                raise RuntimeError("native source relation differs from posterior capacity")
            return NativeObjectQueryPosteriorOutput(
                posterior_rows=posterior,
                relation=relation,
            )
        native_valid = context.native_valid
        native_roles = context.native_roles
        if native_valid is None or native_roles is None or context.instruction_last_index is None:
            raise RuntimeError("native relation context lost its bound prefix metadata")
        native_sensor_valid = native_valid & (native_roles == int(NativeRole.SENSOR))
        sensor_hidden = torch.cat((original, modality_hidden), dim=1)
        sensor_valid = torch.cat((native_sensor_valid, runtime.modality_valid), dim=1)
        structural_sensor_valid = torch.cat(
            (native_sensor_valid, torch.zeros_like(runtime.modality_valid)),
            dim=1,
        )
        if runtime.match_slice is None:
            if not isinstance(self.relation_readout, PhysicalEntityReadout):
                raise RuntimeError("task-independent runtime has a legacy relation readout")
            modality_slice_by_name = dict(runtime.modality_slices)
            object_query_inputs: list[ContextualObjectQuerySpatialInput] = []
            if context.modalities is not None:
                for relation in context.modalities.object_query_spatial_relations:
                    local_slice = modality_slice_by_name.get(relation.query_modality)
                    if local_slice is None:
                        raise RuntimeError(
                            "object-query relation source is absent from contextual modalities"
                        )
                    hidden = modality_hidden[:, local_slice]
                    if hidden.shape[1] != relation.query_count:
                        raise RuntimeError(
                            "contextual object-query count differs from its dense relation"
                        )
                    query_projection_weight = None
                    if relation.dense_mask_features is not None:
                        if relation.query_modality not in self.modality_projections:
                            raise RuntimeError(
                                "direct row-mask source omitted its semantic-query projection"
                            )
                        query_projection_weight = self.modality_projections[
                            relation.query_modality
                        ].weight
                    object_query_inputs.append(
                        ContextualObjectQuerySpatialInput(
                            relation=relation,
                            query_hidden=hidden,
                            query_projection_weight=query_projection_weight,
                        )
                    )
            return self.relation_readout(
                posterior_rows=posterior,
                sensor_hidden=sensor_hidden,
                sensor_valid=sensor_valid,
                structural_sensor_valid=structural_sensor_valid,
                relation_surfaces=runtime.relation_surfaces,
                object_query_spatial_inputs=tuple(
                    sorted(object_query_inputs, key=lambda value: value.relation.name)
                ),
            )
        if not isinstance(self.relation_readout, SharedRelationReadout):
            raise RuntimeError("legacy match runtime has a physical-only relation readout")
        match = prefix[:, runtime.match_slice]
        return self.relation_readout(
            posterior_rows=posterior,
            sensor_hidden=sensor_hidden,
            sensor_valid=sensor_valid,
            match_hidden=match,
            structural_sensor_valid=structural_sensor_valid,
        )


def install_lingbot_native_graph(policy: nn.Module, graph: LingBotNativeGraph) -> None:
    """Attach strict PICF only through the audited official LingBot host hook."""

    model = getattr(policy, "model", None)
    host = getattr(model, "qwenvl_with_expert", None)
    setter = getattr(host, "set_picf_native_graph", None)
    if not callable(setter):
        raise TypeError("LingBot policy lacks the audited set_picf_native_graph hook")
    expected = LingBotNativeGraphConfig.from_policy(
        policy,
        capacity=graph.config.capacity,
        maximum_control_tokens=graph.config.maximum_control_tokens,
        prediction_route_count=graph.config.prediction_route_count,
        task_query_count=graph.config.task_query_count,
        prediction_address_width=graph.config.prediction_address_width,
        predictive_target_widths=graph.config.predictive_target_widths,
        modality_specs=graph.config.modality_specs,
        modality_bridge_identity=graph.config.modality_bridge_identity,
        modality_bridge_query_count=graph.config.modality_bridge_query_count,
        resampled_modality_names=graph.config.resampled_modality_names,
        direct_action_modality_names=graph.config.direct_action_modality_names,
        relation_surface_specs=graph.config.relation_surface_specs,
        object_query_spatial_specs=graph.config.object_query_spatial_specs,
        relation_supervision_layers=graph.config.relation_supervision_layers,
        architecture_identity=graph.config.architecture_identity,
    )
    for field_name in ("host_width", "executed_action_dim", "num_layers"):
        if getattr(expected, field_name) != getattr(graph.config, field_name):
            raise ValueError(f"LingBot host differs from graph config field {field_name}")
    setter(graph)


class LingBotNativePriorStepper:
    """Parameter-free prior-only caller through the installed policy root."""

    def __init__(self, policy: nn.Module, graph: LingBotNativeGraph) -> None:
        host = getattr(getattr(policy, "model", None), "qwenvl_with_expert", None)
        if getattr(host, "picf_native_graph", None) is not graph:
            raise ValueError("prior stepper requires the exact graph installed in the host")
        prior_forward = getattr(policy, "picf_native_prior_forward", None)
        if not callable(prior_forward):
            raise TypeError("prior stepper policy must expose the registered root forward method")
        self.policy = policy
        self.graph = graph

    def _validated_prior_forward(self):
        prior_forward = getattr(self.policy, "picf_native_prior_forward", None)
        if not callable(prior_forward):
            raise RuntimeError("registered LingBot row-only root method became unavailable")
        return prior_forward

    def _build_context(
        self,
        previous_state: (
            NativePosteriorState | NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None
        ),
        controls: ExecutedControlBatch,
        *,
        previous_memory_valid: torch.Tensor | None,
        prediction_request: NativePredictionRequest | None = None,
        episode_address_state: EpisodeAddressState | None = None,
        episode_ids: torch.Tensor | None = None,
    ) -> tuple[LingBotPriorRolloutContext, torch.Tensor]:
        if self.graph.unified_predict_correct:
            if previous_state is not None and not isinstance(
                previous_state,
                NativeLayerwisePosteriorState | NativeLayerwisePriorTrace,
            ):
                raise TypeError("unified prior step requires a typed layerwise source or None")
            common = {
                "controls": controls,
                "prediction_request": prediction_request,
            }
            if self.graph.task_query_object_value_read:
                inherited_address_state: EpisodeAddressState | None = None
                if isinstance(
                    previous_state,
                    AddressedLayerwisePosteriorState | AddressedLayerwisePriorTrace,
                ):
                    if (
                        previous_state.architecture_identity
                        != self.graph.config.architecture_identity
                    ):
                        raise ValueError("prior source belongs to another architecture identity")
                    inherited_address_state = previous_state.episode_address_state
                elif previous_state is not None:
                    raise ValueError(
                        "task-addressed prior source must carry its episode address receipt"
                    )
                source_valid = (
                    torch.full(
                        (controls.batch_size,),
                        previous_state is not None,
                        dtype=torch.bool,
                        device=controls.values.device,
                    )
                    if previous_memory_valid is None
                    else previous_memory_valid
                )
                continuation = source_valid & ~controls.reset.any(dim=1)
                if episode_address_state is None and continuation.all():
                    episode_address_state = inherited_address_state
                elif (
                    inherited_address_state is not None
                    and episode_address_state is not None
                    and continuation.any()
                    and not episode_address_state.same_assignment(inherited_address_state)
                ):
                    continued = torch.nonzero(continuation, as_tuple=False).flatten()
                    if not episode_address_state.index_select(continued).same_assignment(
                        inherited_address_state.index_select(continued)
                    ):
                        raise ValueError(
                            "explicit address state differs from the prior source receipt"
                        )
                common.update(
                    episode_address_state=episode_address_state,
                    episode_ids=episode_ids,
                )
            elif episode_address_state is not None or episode_ids is not None:
                raise ValueError("episode addresses belong only to the task-addressed identity")
            if isinstance(previous_state, NativeLayerwisePriorTrace):
                context = LingBotPriorRolloutContext(
                    source_prior_trace=previous_state,
                    source_prior_trace_valid=previous_memory_valid,
                    **common,
                )
            else:
                context = LingBotPriorRolloutContext(
                    previous_memory=previous_state,
                    previous_memory_valid=previous_memory_valid,
                    **common,
                )
            # Before a cold FSDP2 root call, graph parameters still use their
            # master dtype. Executed controls already carry the runtime compute
            # dtype and device required by the registered row-only forward.
            reference = controls.values if previous_state is None else previous_state.layer_rows
            return context, reference
        if previous_memory_valid is not None:
            raise ValueError("explicit layerwise memory validity requires the v3 architecture")
        if not isinstance(previous_state, NativePosteriorState):
            raise TypeError("legacy prior step requires final-row posterior state")
        return (
            LingBotPriorRolloutContext(
                controls=controls,
                previous_state=previous_state,
                prediction_request=prediction_request,
            ),
            previous_state.rows,
        )

    def __call__(
        self,
        previous_state: (
            NativePosteriorState | NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None
        ),
        controls: ExecutedControlBatch,
        *,
        previous_memory_valid: torch.Tensor | None = None,
        episode_address_state: EpisodeAddressState | None = None,
        episode_ids: torch.Tensor | None = None,
    ) -> NativePosteriorState | NativeLayerwisePriorTrace:
        config = self.graph.config
        context, reference = self._build_context(
            previous_state,
            controls,
            previous_memory_valid=previous_memory_valid,
            episode_address_state=episode_address_state,
            episode_ids=episode_ids,
        )
        batch = controls.batch_size
        empty = reference.new_empty(batch, 0, config.host_width)
        self._validated_prior_forward()(
            attention_mask=torch.empty(
                batch,
                0,
                0,
                dtype=torch.bool,
                device=reference.device,
            ),
            position_ids=torch.empty(
                3,
                batch,
                0,
                dtype=torch.long,
                device=reference.device,
            ),
            inputs_embeds=[empty, None],
            visual_pos_masks=torch.empty(
                batch,
                0,
                dtype=torch.bool,
                device=reference.device,
            ),
            picf_native_context=context,
        )
        if self.graph.unified_predict_correct:
            if context.prior_trace is None:
                raise RuntimeError(
                    "official LingBot host did not finalize its layerwise prior trace"
                )
            return context.prior_trace
        if context.prior_state is None:
            raise RuntimeError("official LingBot host did not finalize its row-only prior")
        return context.prior_state

    def step_with_prediction(
        self,
        previous_state: (
            NativePosteriorState | NativeLayerwisePosteriorState | NativeLayerwisePriorTrace | None
        ),
        controls: ExecutedControlBatch,
        prediction_request: NativePredictionRequest,
        *,
        target_name: str,
        previous_memory_valid: torch.Tensor | None = None,
        episode_address_state: EpisodeAddressState | None = None,
        episode_ids: torch.Tensor | None = None,
    ) -> tuple[NativePosteriorState | NativeLayerwisePriorTrace, torch.Tensor]:
        """Advance the shared prior and return one projection from that root forward."""

        config = self.graph.config
        context, reference = self._build_context(
            previous_state,
            controls,
            previous_memory_valid=previous_memory_valid,
            prediction_request=prediction_request,
            episode_address_state=episode_address_state,
            episode_ids=episode_ids,
        )
        batch = controls.batch_size
        empty = reference.new_empty(batch, 0, config.host_width)
        self._validated_prior_forward()(
            attention_mask=torch.empty(
                batch,
                0,
                0,
                dtype=torch.bool,
                device=reference.device,
            ),
            position_ids=torch.empty(
                3,
                batch,
                0,
                dtype=torch.long,
                device=reference.device,
            ),
            inputs_embeds=[empty, None],
            visual_pos_masks=torch.empty(
                batch,
                0,
                dtype=torch.bool,
                device=reference.device,
            ),
            picf_native_context=context,
        )
        if context.prior_state is None or context.prediction_hidden is None:
            raise RuntimeError("official LingBot host omitted row-only predictive outputs")
        try:
            prediction = context.prediction_outputs[target_name]
        except KeyError as error:
            raise KeyError(f"row-only forward omitted predictive target {target_name!r}") from error
        prior: NativePosteriorState | NativeLayerwisePriorTrace
        if self.graph.unified_predict_correct:
            if context.prior_trace is None:
                raise RuntimeError("official LingBot host omitted its layerwise prior trace")
            prior = context.prior_trace
        else:
            prior = context.prior_state
        return prior, prediction

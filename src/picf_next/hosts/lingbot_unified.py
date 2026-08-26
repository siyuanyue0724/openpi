"""Native per-layer LingBot VLA2 integration for the unified PICF graph.

This is not the historical read-only action sidecar.  The pinned host patch
calls this module inside LingBot's official per-layer
``Q/K/V -> concatenate -> MRoPE -> attention -> split`` loop.  Belief rows use
the Qwen3-VL stream and action remains the released Qwen2 flow expert.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace

import torch
from torch import nn

from picf_next.unified.codec import (
    BeliefCodecConfig,
    PairedBeliefTokens,
    UnifiedBeliefCodec,
)
from picf_next.unified.coreference import (
    CoreferenceOutput,
    GroupedRelationEvidence,
    grouped_relation_evidence,
    repeat_kv_heads,
    responsibility_weighted_message,
    shared_qk_coreference,
)
from picf_next.unified.graph import (
    TokenLayout,
    TokenRole,
    block_causal_attention_mask,
    expand_host_mask_for_inserted_tokens,
    insert_layout_block,
    role_graph_contract_digest,
    role_permission_matrix,
)
from picf_next.unified.lifecycle import (
    deterministic_logdet_ci_weights,
    generalized_covariance_intersection,
    logarithmic_lifecycle_pool,
    posterior_expected_age,
    reliability_simplex,
)
from picf_next.unified.predictive import (
    ROW_SUMMARY_TARGET,
    PredictionQueryRequest,
)
from picf_next.unified.state import GeometrySchema, UnifiedBeliefState

_ALLOWED_NATIVE_ROLES = (
    int(TokenRole.SENSOR),
    int(TokenRole.LANGUAGE),
    int(TokenRole.MEASUREMENT_QUERY),
    int(TokenRole.HOST_FUTURE_QUERY),
)


@dataclass(frozen=True, slots=True)
class LingBotHostContract:
    """Dimensions read from one instantiated official LingBot VLA2 policy."""

    prefix_width: int
    attention_value_width: int
    num_layers: int
    executed_action_dim: int
    native_measurement_query_tokens: int
    native_prediction_query_tokens: int

    @property
    def native_training_query_tokens(self) -> int:
        return self.native_measurement_query_tokens + self.native_prediction_query_tokens

    @classmethod
    def from_policy(cls, policy: nn.Module) -> LingBotHostContract:
        model = getattr(policy, "model", None)
        host = getattr(model, "qwenvl_with_expert", None)
        if model is None or host is None:
            raise TypeError("policy does not expose the official LingBot joint host")
        try:
            text_model = host.qwenvl.model.language_model
            action_model = host.qwen_expert.model
            text_layer = text_model.layers[0]
            action_layer = action_model.layers[0]
            prefix_width = int(text_layer.hidden_size)
            text_q_width = int(text_layer.self_attn.q_proj.out_features)
            action_q_width = int(action_layer.self_attn.q_proj.out_features)
            text_head_dim = int(text_layer.self_attn.head_dim)
            action_head_dim = int(action_layer.self_attn.head_dim)
            num_layers = len(text_model.layers)
            action_num_layers = len(action_model.layers)
            executed_action_dim = _required_positive_int(model.config, "max_action_dim")
        except (AttributeError, IndexError, TypeError, ValueError) as error:
            raise TypeError("policy has an incomplete LingBot VLA2 shape contract") from error
        if prefix_width <= 0 or text_q_width <= 0 or executed_action_dim <= 0:
            raise ValueError("LingBot host dimensions must be positive")
        if text_q_width != action_q_width or text_head_dim != action_head_dim:
            raise ValueError("LingBot prefix and action streams have incompatible attention spaces")
        if text_q_width % text_head_dim:
            raise ValueError("LingBot attention width is not divisible by its head dimension")
        if num_layers != action_num_layers:
            raise ValueError("LingBot prefix and action streams have different depths")
        (
            native_measurement_query_tokens,
            native_prediction_query_tokens,
        ) = _lingbot_query_token_counts(model)
        return cls(
            prefix_width=prefix_width,
            attention_value_width=text_q_width,
            num_layers=num_layers,
            executed_action_dim=executed_action_dim,
            native_measurement_query_tokens=native_measurement_query_tokens,
            native_prediction_query_tokens=native_prediction_query_tokens,
        )


def _lingbot_query_token_counts(model: nn.Module) -> tuple[int, int]:
    if (
        not getattr(model, "use_depth_align", False)
        or getattr(model, "align_type", None) != "query"
    ):
        return 0, 0
    measurement_count = _required_positive_int(model, "num_task_tokens")
    prediction_count = 0
    if getattr(model, "use_future_video", False):
        if getattr(model, "use_future_video_cls", False):
            prediction_count += 1
        if getattr(model, "use_future_video_patch", True) and not getattr(
            model,
            "future_video_share_future_depth_query",
            False,
        ):
            prediction_count += measurement_count
    if getattr(model, "use_future_depth", False):
        prediction_count += measurement_count
    return measurement_count, prediction_count


def _required_positive_int(owner: object, name: str) -> int:
    value = getattr(owner, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"LingBot {name} must be a positive integer")
    return value


@dataclass(frozen=True, slots=True)
class LingBotUnifiedGraphConfig:
    codec: BeliefCodecConfig
    geometry_schema: GeometrySchema
    attention_value_width: int
    num_layers: int = 36
    retrieval_tokens: int = 1
    executed_action_dim: int = 55
    native_measurement_query_tokens: int = 0
    native_prediction_query_tokens: int = 0
    modality_names: tuple[str, ...] = ("vision",)
    grouped_assignment_modalities: tuple[str, ...] = ()
    modality_reliability: tuple[float, ...] = (1.0,)
    prior_reliability: float = 1.0
    robust_clip: float = 8.0
    relation_adoption_init: float = 0.0
    ci_iterations: int = 8
    ci_step_size: float = 0.25
    birth_scale_init: float = 0.02

    def __post_init__(self) -> None:
        integer_controls = (
            self.attention_value_width,
            self.num_layers,
            self.retrieval_tokens,
            self.executed_action_dim,
            self.native_measurement_query_tokens,
            self.native_prediction_query_tokens,
            self.ci_iterations,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integer_controls):
            raise TypeError("unified graph counts and dimensions must be integers")
        real_controls = (
            *self.modality_reliability,
            self.prior_reliability,
            self.robust_clip,
            self.relation_adoption_init,
            self.ci_step_size,
            self.birth_scale_init,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, float))
            for value in real_controls
        ):
            raise TypeError("unified graph reliability and solver controls must be real-valued")
        if not all(math.isfinite(value) for value in real_controls):
            raise ValueError("unified graph reliability and solver controls must be finite")
        if self.geometry_schema.width != self.codec.geometry_dim:
            raise ValueError("geometry schema width must match the belief codec")
        if self.codec.host_width < 2 * self.codec.canonical_width:
            raise ValueError("unified action pairing requires host_width >= 2 * canonical_width")
        if self.attention_value_width <= 0:
            raise ValueError("attention_value_width must be positive")
        if self.num_layers < 2:
            raise ValueError("unified LingBot graph requires at least two layers")
        if self.retrieval_tokens <= 0:
            raise ValueError("retrieval_tokens must be positive")
        if self.executed_action_dim <= 0:
            raise ValueError("executed_action_dim must be positive")
        if min(self.native_measurement_query_tokens, self.native_prediction_query_tokens) < 0:
            raise ValueError("native query token counts must be non-negative")
        if not self.modality_names or any(not name for name in self.modality_names):
            raise ValueError("at least one named physical modality is required")
        if len(set(self.modality_names)) != len(self.modality_names):
            raise ValueError("physical modality names must be unique")
        if len(set(self.grouped_assignment_modalities)) != len(
            self.grouped_assignment_modalities
        ) or not set(self.grouped_assignment_modalities).issubset(self.modality_names):
            raise ValueError("grouped assignment modalities must be unique declared modalities")
        if len(self.modality_reliability) != len(self.modality_names):
            raise ValueError("every modality requires one calibrated reliability")
        if self.prior_reliability <= 0 or any(value < 0 for value in self.modality_reliability):
            raise ValueError("prior/modality reliabilities must be non-negative")
        if self.robust_clip <= 0:
            raise ValueError("robust_clip must be positive")
        if self.ci_iterations <= 0 or self.ci_step_size <= 0:
            raise ValueError("CI solver controls must be positive")
        if self.birth_scale_init <= 0:
            raise ValueError("birth_scale_init must be positive")

    @property
    def penultimate_layer(self) -> int:
        return self.num_layers - 2

    @property
    def modality_count(self) -> int:
        return len(self.modality_names)

    @property
    def native_training_query_tokens(self) -> int:
        return self.native_measurement_query_tokens + self.native_prediction_query_tokens

    @property
    def contract_digest(self) -> str:
        """Content-address every static choice that changes state semantics."""

        payload = {
            "attention_value_width": self.attention_value_width,
            "birth_scale_init": float(self.birth_scale_init).hex(),
            "ci_iterations": self.ci_iterations,
            "ci_step_size": float(self.ci_step_size).hex(),
            "codec": {
                "content_dim": self.codec.content_dim,
                "geometry_dim": self.codec.geometry_dim,
                "host_width": self.codec.host_width,
                "information_floor": float(self.codec.information_floor).hex(),
                "uncertainty_dim": self.codec.uncertainty_dim,
            },
            "executed_action_dim": self.executed_action_dim,
            "geometry_schema": self.geometry_schema.canonical_dict(),
            "grouped_assignment_modalities": self.grouped_assignment_modalities,
            "modality_names": self.modality_names,
            "modality_reliability": tuple(
                float(value).hex() for value in self.modality_reliability
            ),
            "native_allowed_roles": _ALLOWED_NATIVE_ROLES,
            "native_measurement_query_tokens": self.native_measurement_query_tokens,
            "native_prediction_query_tokens": self.native_prediction_query_tokens,
            "num_layers": self.num_layers,
            "prior_reliability": float(self.prior_reliability).hex(),
            "relation_adoption_init": float(self.relation_adoption_init).hex(),
            "retrieval_tokens": self.retrieval_tokens,
            "role_graph_contract_digest": role_graph_contract_digest(),
            "robust_clip": float(self.robust_clip).hex(),
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_policy(
        cls,
        policy: nn.Module,
        *,
        codec: BeliefCodecConfig,
        **kwargs,
    ) -> LingBotUnifiedGraphConfig:
        contract = LingBotHostContract.from_policy(policy)
        if codec.host_width != contract.prefix_width:
            raise ValueError("belief codec host width differs from the loaded LingBot policy")
        return cls(
            codec=codec,
            attention_value_width=contract.attention_value_width,
            num_layers=contract.num_layers,
            executed_action_dim=contract.executed_action_dim,
            native_measurement_query_tokens=contract.native_measurement_query_tokens,
            native_prediction_query_tokens=contract.native_prediction_query_tokens,
            **kwargs,
        )


@dataclass(slots=True)
class LingBotUnifiedContext:
    """Per-control-step inputs and auditable outputs of the physical graph."""

    previous_posterior: UnifiedBeliefState
    modality_geometry_valid: torch.Tensor
    elapsed_time: torch.Tensor
    previous_executed_action: torch.Tensor
    previous_action_valid: torch.Tensor
    birth_proposal_noise: torch.Tensor
    native_roles: torch.Tensor | None = None
    native_valid: torch.Tensor | None = None
    native_footprint: torch.Tensor | None = None
    native_modality_ids: torch.Tensor | None = None
    native_group_ids: torch.Tensor | None = None
    prediction_request: PredictionQueryRequest | None = None
    predictive_prior: UnifiedBeliefState | None = None
    posterior: UnifiedBeliefState | None = None
    last_coreference: CoreferenceOutput | None = None
    last_grouped_evidence: GroupedRelationEvidence | None = None
    final_action_pair: PairedBeliefTokens | None = None
    row_prediction_hidden: torch.Tensor | None = None
    expanded_prefix_valid: torch.Tensor | None = None
    expanded_prefix_position_ids: torch.Tensor | None = None
    expanded_cache_valid: torch.Tensor | None = None
    expanded_cache_position_ids: torch.Tensor | None = None
    expanded_action_cache_visible: torch.Tensor | None = None

    def validate(
        self,
        *,
        host_width: int,
        modality_names: tuple[str, ...],
        grouped_assignment_modalities: tuple[str, ...] = (),
    ) -> None:
        modality_count = len(modality_names)
        metadata = (
            self.native_roles,
            self.native_valid,
            self.native_footprint,
            self.native_modality_ids,
        )
        if any(value is None for value in metadata):
            raise ValueError("native token metadata has not been materialized")
        assert self.native_roles is not None
        assert self.native_valid is not None
        assert self.native_footprint is not None
        assert self.native_modality_ids is not None
        if self.native_group_ids is None:
            raise ValueError("native token groups have not been materialized")
        prefix_shape = self.native_roles.shape
        if self.native_roles.ndim != 2:
            raise ValueError("native_roles must have shape [batch, prefix_tokens]")
        if self.native_roles.dtype != torch.long:
            raise TypeError("native_roles must use torch.long")
        if self.native_valid.shape != prefix_shape or self.native_valid.dtype != torch.bool:
            raise ValueError("native_valid must be boolean and match native_roles")
        if (
            self.native_footprint.shape != prefix_shape
            or not self.native_footprint.is_floating_point()
        ):
            raise ValueError("native_footprint must match native_roles")
        if (self.native_footprint < 0).any() or not torch.isfinite(self.native_footprint).all():
            raise ValueError("native_footprint must be finite and non-negative")
        if (
            self.native_modality_ids.shape != prefix_shape
            or self.native_modality_ids.dtype != torch.long
        ):
            raise ValueError("native_modality_ids must be long and match native_roles")
        if ((self.native_modality_ids < -1) | (self.native_modality_ids >= modality_count)).any():
            raise ValueError("native_modality_ids contain an undeclared modality")
        if self.native_group_ids.shape != prefix_shape or self.native_group_ids.dtype != torch.long:
            raise ValueError("native_group_ids must be long and match native_roles")
        if (self.native_group_ids < -1).any():
            raise ValueError("native_group_ids must use -1 for independent tokens")
        if self.previous_posterior.batch_size != prefix_shape[0]:
            raise ValueError("belief and native prefix batches must match")
        expected_geometry = (
            self.previous_posterior.batch_size,
            modality_count,
            self.previous_posterior.capacity,
            self.previous_posterior.geometry_dim,
        )
        if (
            self.modality_geometry_valid.shape != expected_geometry
            or self.modality_geometry_valid.dtype != torch.bool
        ):
            raise ValueError("modality_geometry_valid must match modality/belief geometry")
        if self.elapsed_time.shape != (self.previous_posterior.batch_size,):
            raise ValueError("elapsed_time must contain one scalar per batch item")
        if not self.elapsed_time.is_floating_point() or not torch.isfinite(self.elapsed_time).all():
            raise ValueError("elapsed_time must be finite floating point")
        if (self.elapsed_time < 0).any():
            raise ValueError("elapsed_time must be non-negative")
        if self.previous_executed_action.ndim != 2 or self.previous_executed_action.shape[0] != (
            self.previous_posterior.batch_size
        ):
            raise ValueError("previous_executed_action must have shape [batch, action_dim]")
        if (
            not self.previous_executed_action.is_floating_point()
            or not torch.isfinite(self.previous_executed_action).all()
        ):
            raise ValueError("previous_executed_action must be finite floating point")
        if (
            self.previous_action_valid.shape != (self.previous_posterior.batch_size,)
            or self.previous_action_valid.dtype != torch.bool
        ):
            raise ValueError("previous_action_valid must be boolean with one value per batch")
        expected_birth_noise = (
            self.previous_posterior.batch_size,
            self.previous_posterior.capacity,
            self.previous_posterior.content_dim,
        )
        if (
            self.birth_proposal_noise.shape != expected_birth_noise
            or self.birth_proposal_noise.dtype != torch.float32
            or not torch.isfinite(self.birth_proposal_noise).all()
        ):
            raise ValueError("birth_proposal_noise must be finite FP32 [batch, capacity, content]")
        allowed_native = torch.tensor(_ALLOWED_NATIVE_ROLES, device=self.native_roles.device)
        if not torch.isin(self.native_roles, allowed_native).all():
            raise ValueError("native prefix contains a role owned by the unified graph")
        nonsensor = self.native_roles != int(TokenRole.SENSOR)
        if (self.native_footprint.masked_select(nonsensor) != 0).any():
            raise ValueError("only sensor tokens may carry physical footprint")
        if (self.native_footprint.masked_select(~self.native_valid) != 0).any():
            raise ValueError("invalid native tokens cannot carry physical footprint")
        if (self.native_group_ids.masked_select(nonsensor | ~self.native_valid) != -1).any():
            raise ValueError("non-sensor or invalid tokens cannot carry a physical group ID")
        valid_sensor = (~nonsensor) & self.native_valid
        if (self.native_modality_ids.masked_select(~valid_sensor) != -1).any():
            raise ValueError("only valid sensor tokens may carry a physical modality ID")
        if (self.native_modality_ids.masked_select(valid_sensor) < 0).any():
            raise ValueError("valid sensor tokens require a physical modality ID")
        token_modalities = torch.nn.functional.one_hot(
            self.native_modality_ids.clamp_min(0),
            modality_count,
        ).to(self.native_footprint)
        token_modalities = token_modalities * valid_sensor.unsqueeze(-1)
        modality_mass = torch.einsum(
            "bn,bnm->bm",
            self.native_footprint,
            token_modalities,
        )
        modality_available = token_modalities.bool().any(dim=1)
        if not torch.allclose(
            modality_mass,
            modality_available.to(modality_mass.dtype),
            atol=1e-5,
            rtol=0,
        ):
            raise ValueError("every available physical modality must have unit footprint mass")
        for modality in grouped_assignment_modalities:
            modality_id = modality_names.index(modality)
            active = valid_sensor & (self.native_modality_ids == modality_id)
            if (self.native_group_ids.masked_select(active) < 0).any():
                raise ValueError(f"valid {modality} tokens require a physical token group")
        grouped_members = (self.native_group_ids >= 0).nonzero(as_tuple=False)
        if grouped_members.numel():
            group_keys = torch.stack(
                (
                    grouped_members[:, 0],
                    self.native_group_ids[self.native_group_ids >= 0],
                ),
                dim=-1,
            )
            unique_groups, group_inverse = torch.unique(group_keys, dim=0, return_inverse=True)
            group_count = unique_groups.shape[0]
            grouped_modalities = self.native_modality_ids[self.native_group_ids >= 0]
            modality_membership = torch.zeros(
                group_count,
                modality_count,
                dtype=torch.long,
                device=self.native_roles.device,
            )
            modality_membership.index_add_(
                0,
                group_inverse,
                torch.nn.functional.one_hot(grouped_modalities, modality_count),
            )
            if ((modality_membership > 0).sum(dim=-1) != 1).any():
                raise ValueError("one physical token group cannot mix modalities")
            group_mass = torch.zeros(
                group_count,
                dtype=torch.float32,
                device=self.native_roles.device,
            )
            group_mass.index_add_(
                0,
                group_inverse,
                self.native_footprint[self.native_group_ids >= 0].float(),
            )
            if (group_mass <= 0).any():
                raise ValueError("every physical token group requires positive footprint mass")
        if host_width < self.previous_posterior.canonical().shape[-1]:
            raise ValueError("host width cannot hold the canonical belief state")
        if self.prediction_request is not None:
            if self.prediction_request.source_batch_size != self.previous_posterior.batch_size:
                raise ValueError("prediction request source batch size differs from context")
            if self.prediction_request.target_kind != ROW_SUMMARY_TARGET:
                raise ValueError("unified row prediction only accepts row-summary requests")
            if self.prediction_request.modality not in modality_names:
                raise ValueError("prediction request names an undeclared physical modality")
            if self.prediction_request.horizon == 0:
                if modality_count < 2:
                    raise ValueError("cross-modal prediction requires a second physical modality")
                target_modality = modality_names.index(self.prediction_request.modality)
                valid_sensor = self.native_valid & (self.native_roles == int(TokenRole.SENSOR))
                target_visible = valid_sensor & (self.native_modality_ids == target_modality)
                if target_visible.any():
                    raise ValueError("cross-modal target modality was not withheld from input")
                source_visible = valid_sensor & (self.native_modality_ids != target_modality)
                if not source_visible.any(dim=-1).all():
                    raise ValueError(
                        "cross-modal prediction requires a valid non-target physical sensor "
                        "for every batch item"
                    )
        device = self.previous_posterior.content.device
        tensors = (
            self.native_roles,
            self.native_valid,
            self.native_footprint,
            self.native_modality_ids,
            self.native_group_ids,
            self.modality_geometry_valid,
            self.elapsed_time,
            self.previous_executed_action,
            self.previous_action_valid,
            self.birth_proposal_noise,
        )
        if any(value.device != device for value in tensors):
            raise ValueError("unified LingBot context tensors must share one device")


@dataclass(slots=True)
class _JointRuntime:
    context: LingBotUnifiedContext
    assignment_group_ids: torch.Tensor | None
    original_prefix_count: int
    prefix_count: int
    action_count: int
    transition_index: int
    prior_slice: slice
    posterior_slice: slice
    context_index: int
    retrieval_slice: slice
    prediction_slice: slice | None
    layout: TokenLayout
    relation_message: torch.Tensor | None = None
    grouped_evidence: GroupedRelationEvidence | None = None


class LingBotUnifiedBeliefGraph(nn.Module):
    """Small typed operators embedded in LingBot's existing 36-layer graph."""

    def __init__(self, config: LingBotUnifiedGraphConfig) -> None:
        super().__init__()
        self.config = config
        self.codec = UnifiedBeliefCodec(config.codec)
        host_width = config.codec.host_width
        self.context_seed = nn.Parameter(torch.zeros(1, 1, host_width))
        self.retrieval_seed = nn.Parameter(torch.zeros(1, config.retrieval_tokens, host_width))
        self.transition_projection = nn.Linear(config.executed_action_dim + 2, host_width)
        tail_width = host_width - config.codec.canonical_width
        self.role_tail = nn.Parameter(torch.zeros(len(TokenRole), tail_width))
        self.predictive_modality_embedding = nn.Parameter(
            torch.zeros(config.modality_count, host_width)
        )
        self.predictive_horizon_projection = nn.Linear(2, host_width, bias=False)
        self.relation_adoption = nn.Parameter(
            torch.full((config.num_layers - 1,), config.relation_adoption_init)
        )
        self.birth_mean = nn.Parameter(torch.zeros(config.codec.content_dim))
        self.birth_log_scale = nn.Parameter(
            torch.full(
                (config.codec.content_dim,),
                torch.tensor(config.birth_scale_init).log().item(),
            )
        )
        geometry = config.codec.geometry_dim
        measurement_width = 2 + geometry + geometry * (geometry + 1) // 2
        self.measurement_projection = nn.Linear(
            config.attention_value_width + 2,
            measurement_width,
        )
        nn.init.zeros_(self.transition_projection.weight)
        nn.init.zeros_(self.transition_projection.bias)
        nn.init.zeros_(self.predictive_horizon_projection.weight)
        nn.init.normal_(self.measurement_projection.weight, std=1e-3)
        nn.init.zeros_(self.measurement_projection.bias)

    def prepare_joint_inputs(
        self,
        *,
        inputs_embeds: list[torch.Tensor | None],
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        context: LingBotUnifiedContext | None,
    ) -> tuple[
        list[torch.Tensor | None],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        _JointRuntime | None,
    ]:
        """Insert belief roles before the action stream, or return exact identity."""

        if context is None:
            return inputs_embeds, attention_mask, position_ids, visual_pos_masks, None
        if context.prediction_request is not None and not self.training:
            raise ValueError("row prediction queries are training-only")
        if len(inputs_embeds) != 2 or inputs_embeds[0] is None:
            raise ValueError("unified graph preparation requires the native prefix stream")
        prefix = inputs_embeds[0]
        action = inputs_embeds[1]
        if prefix.shape[-1] != self.config.codec.host_width:
            raise ValueError("LingBot prefix width does not match the belief codec")
        if context.previous_executed_action.shape[-1] != self.config.executed_action_dim:
            raise ValueError("previous executed action width differs from the host contract")
        batch, original_prefix_count, _ = prefix.shape
        action_count = 0 if action is None else action.shape[1]
        old_total = original_prefix_count + action_count
        if attention_mask.shape != (batch, old_total, old_total):
            raise ValueError("LingBot host mask has an unexpected shape")
        if position_ids.shape != (3, batch, old_total):
            raise ValueError("LingBot MRoPE position_ids have an unexpected shape")
        self._materialize_native_metadata(
            context,
            attention_mask=attention_mask,
            visual_pos_masks=visual_pos_masks,
            original_prefix_count=original_prefix_count,
        )
        self._validate_native_query_layout(
            context,
            original_prefix_count=original_prefix_count,
        )
        context.validate(
            host_width=self.config.codec.host_width,
            modality_names=self.config.modality_names,
            grouped_assignment_modalities=self.config.grouped_assignment_modalities,
        )
        assert context.native_roles is not None
        assert context.native_valid is not None
        assert context.native_footprint is not None
        assert context.native_modality_ids is not None
        if prefix.shape[:2] != context.native_roles.shape:
            raise ValueError("native role layout does not match LingBot prefix embeddings")

        initial_state = self._with_birth_proposals(
            context.previous_posterior,
            context.birth_proposal_noise,
        )
        prior = self._typed_belief_tokens(initial_state, TokenRole.PRIOR)
        posterior = self._typed_belief_tokens(initial_state, TokenRole.POSTERIOR)
        transition_features = torch.cat(
            (
                context.previous_executed_action
                * context.previous_action_valid.unsqueeze(-1).to(
                    context.previous_executed_action.dtype
                ),
                context.previous_action_valid.unsqueeze(-1).to(
                    context.previous_executed_action.dtype
                ),
                context.elapsed_time.unsqueeze(-1).to(context.previous_executed_action),
            ),
            dim=-1,
        ).to(self.transition_projection.weight)
        transition = self._add_role_tail(
            self.transition_projection(transition_features).unsqueeze(1),
            TokenRole.HISTORY,
        )
        context_token = self.context_seed.to(prefix).expand(batch, -1, -1)
        retrieval = self.retrieval_seed.to(prefix).expand(batch, -1, -1)
        prediction = self._row_prediction_queries(
            prefix,
            context=context,
            capacity=context.previous_posterior.capacity,
        )
        extra_parts = [transition, prior, posterior, context_token, retrieval]
        if prediction is not None:
            extra_parts.append(prediction)
        extras = torch.cat(extra_parts, dim=1)
        prefix_with_belief = torch.cat((prefix, extras), dim=1)
        capacity = context.previous_posterior.capacity
        extra_count = extras.shape[1]
        prefix_count = original_prefix_count + extra_count
        transition_index = original_prefix_count
        prior_start = transition_index + 1
        posterior_start = prior_start + capacity
        context_index = posterior_start + capacity
        retrieval_start = context_index + 1
        retrieval_stop = retrieval_start + self.config.retrieval_tokens
        prediction_slice = (
            slice(retrieval_stop, retrieval_stop + capacity) if prediction is not None else None
        )

        base_roles = context.native_roles
        base_valid = context.native_valid
        if action_count:
            action_roles = torch.full(
                (batch, action_count),
                int(TokenRole.ACTION),
                dtype=torch.long,
                device=prefix.device,
            )
            # LingBot's native action-expert suffix is exactly
            # ``[current state, noisy action chunk]``.  The current state is a
            # deploy-visible correction input; only the remaining suffix is a
            # training/action role that physical beliefs must not read.
            action_roles[:, 0] = int(TokenRole.CURRENT_STATE)
            action_valid = attention_mask.diagonal(dim1=-2, dim2=-1)[:, original_prefix_count:]
            base_roles = torch.cat((base_roles, action_roles), dim=1)
            base_valid = torch.cat((base_valid, action_valid), dim=1)
        base_layout = TokenLayout(roles=base_roles, valid=base_valid)
        inserted_roles = torch.tensor(
            [
                int(TokenRole.HISTORY),
                *([int(TokenRole.PRIOR)] * capacity),
                *([int(TokenRole.POSTERIOR)] * capacity),
                int(TokenRole.CONTEXT),
                *([int(TokenRole.RETRIEVAL)] * self.config.retrieval_tokens),
                *([int(TokenRole.PREDICT_QUERY)] * capacity if prediction is not None else []),
            ],
            dtype=torch.long,
            device=prefix.device,
        ).expand(batch, -1)
        inserted_layout = TokenLayout(
            roles=inserted_roles,
            valid=torch.ones_like(inserted_roles, dtype=torch.bool),
        )
        layout = insert_layout_block(
            base_layout,
            inserted_layout,
            insertion_index=original_prefix_count,
        )
        expanded_host_mask = expand_host_mask_for_inserted_tokens(
            attention_mask,
            insertion_index=original_prefix_count,
            inserted_count=extra_count,
        )
        unified_mask = block_causal_attention_mask(layout, base_mask=expanded_host_mask)
        if prediction_slice is not None:
            unified_mask = self._restrict_row_prediction_mask(
                unified_mask,
                layout=layout,
                transition_index=transition_index,
                prior_slice=slice(prior_start, posterior_start),
                posterior_slice=slice(posterior_start, context_index),
                prediction_slice=prediction_slice,
            )
        expanded_positions = self._insert_position_ids(
            position_ids,
            native_valid=context.native_valid,
            original_prefix_count=original_prefix_count,
            capacity=capacity,
            action_count=action_count,
            prediction_count=0 if prediction is None else capacity,
        )
        if visual_pos_masks is not None:
            if visual_pos_masks.shape != (batch, original_prefix_count):
                raise ValueError("visual_pos_masks does not match the native prefix")
            visual_pos_masks = torch.cat(
                (
                    visual_pos_masks,
                    torch.zeros((batch, extra_count), dtype=torch.bool, device=prefix.device),
                ),
                dim=1,
            )

        context.expanded_prefix_valid = layout.valid[:, :prefix_count]
        context.expanded_prefix_position_ids = expanded_positions[:, :, :prefix_count]
        # In training this is the complete prefix+suffix layout.  During
        # unified inference the one suffix token is the current robot state,
        # so these tensors describe the exact prefix+belief+state KV cache that
        # subsequent denoising action queries must address.
        context.expanded_cache_valid = layout.valid
        context.expanded_cache_position_ids = expanded_positions
        action_permissions = role_permission_matrix(device=layout.roles.device)[
            int(TokenRole.ACTION)
        ]
        context.expanded_action_cache_visible = action_permissions[layout.roles] & layout.valid
        assert context.native_group_ids is not None
        assignment_group_ids = (
            context.native_group_ids if bool((context.native_group_ids >= 0).any().item()) else None
        )
        runtime = _JointRuntime(
            context=context,
            assignment_group_ids=assignment_group_ids,
            original_prefix_count=original_prefix_count,
            prefix_count=prefix_count,
            action_count=action_count,
            transition_index=transition_index,
            prior_slice=slice(prior_start, posterior_start),
            posterior_slice=slice(posterior_start, context_index),
            context_index=context_index,
            retrieval_slice=slice(retrieval_start, retrieval_stop),
            prediction_slice=prediction_slice,
            layout=layout,
        )
        return (
            [prefix_with_belief, action],
            unified_mask,
            expanded_positions,
            visual_pos_masks,
            runtime,
        )

    def observe_joint_qkv(
        self,
        *,
        layer_index: int,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        runtime: _JointRuntime,
    ) -> _JointRuntime:
        self._validate_layer(layer_index)
        if layer_index == self.config.num_layers - 1:
            # The penultimate layer is the only state-write boundary. The final
            # layer reads the resulting pairs for action/prediction and must not
            # overwrite the co-reference diagnostic that produced that state.
            runtime.relation_message = None
            return runtime
        if query_states.ndim != 4 or key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("joint Q/K/V must have shape [batch, tokens, heads, head_dim]")
        if query_states.shape[:2] != key_states.shape[:2] or key_states.shape != value_states.shape:
            raise ValueError("joint Q/K/V batch, token, and K/V shapes must agree")
        if query_states.shape[-1] != key_states.shape[-1]:
            raise ValueError("joint query and key/value head dimensions must agree")
        if query_states.shape[-2] * query_states.shape[-1] != self.config.attention_value_width:
            raise ValueError("joint Q heads do not match the declared attention value width")
        if query_states.shape[1] < runtime.prefix_count + runtime.action_count:
            raise ValueError("joint Q/K/V sequence is shorter than the unified token layout")
        native_roles = runtime.context.native_roles
        native_valid = runtime.context.native_valid
        native_footprint = runtime.context.native_footprint
        native_modality_ids = runtime.context.native_modality_ids
        if any(
            value is None
            for value in (native_roles, native_valid, native_footprint, native_modality_ids)
        ):
            raise RuntimeError("unified runtime lost materialized native token metadata")
        assert native_roles is not None
        assert native_valid is not None
        assert native_footprint is not None
        assert native_modality_ids is not None
        sensor_valid = native_valid & (native_roles == int(TokenRole.SENSOR))
        queries = torch.cat(
            (
                query_states[:, runtime.posterior_slice],
                query_states[:, runtime.context_index : runtime.context_index + 1],
            ),
            dim=1,
        )
        keys = key_states[:, : runtime.original_prefix_count]
        output = shared_qk_coreference(
            queries,
            keys,
            native_footprint,
            sensor_valid,
            group_ids=runtime.assignment_group_ids,
            robust_clip=self.config.robust_clip,
        )
        values = repeat_kv_heads(
            value_states[:, : runtime.original_prefix_count],
            query_states.shape[-2],
        )
        normalized_relation = responsibility_weighted_message(
            output.evidence.responsibilities,
            values,
            native_footprint,
            sensor_valid,
        )
        # The normalized conditional value identifies what a row would read,
        # while support records how much physical footprint actually backs that
        # relation.  Keeping both factors prevents an arbitrarily small
        # responsibility from becoming a full-amplitude residual.
        runtime.relation_message = (
            normalized_relation * output.evidence.support.unsqueeze(-1).unsqueeze(-1)
        ).flatten(-2)
        runtime.grouped_evidence = grouped_relation_evidence(
            output.logits,
            output.evidence.responsibilities,
            values,
            native_footprint,
            sensor_valid,
            native_modality_ids,
            modality_count=self.config.modality_count,
            robust_clip=self.config.robust_clip,
        )
        runtime.context.last_coreference = output
        runtime.context.last_grouped_evidence = runtime.grouped_evidence
        return runtime

    def apply_relation_message(
        self,
        *,
        layer_index: int,
        attention_output: torch.Tensor,
        runtime: _JointRuntime,
    ) -> torch.Tensor:
        self._validate_layer(layer_index)
        if layer_index == self.config.num_layers - 1:
            return attention_output
        if runtime.relation_message is None:
            return attention_output
        if attention_output.ndim != 3:
            raise ValueError(
                "official LingBot attention output must have shape [batch, tokens, H*D]"
            )
        if attention_output.shape[-1] != self.config.attention_value_width:
            raise ValueError("LingBot attention output width differs from the graph contract")
        if attention_output.shape[1] < runtime.posterior_slice.stop:
            raise ValueError("LingBot attention output is shorter than the posterior rows")
        adoption = torch.tanh(self.relation_adoption[layer_index]).to(attention_output)
        updated = attention_output.clone()
        updated[:, runtime.posterior_slice] = updated[
            :, runtime.posterior_slice
        ] + adoption * runtime.relation_message.to(attention_output)
        return updated

    def after_layer(
        self,
        *,
        layer_index: int,
        outputs_embeds: list[torch.Tensor | None],
        runtime: _JointRuntime,
    ) -> tuple[list[torch.Tensor | None], _JointRuntime]:
        self._validate_layer(layer_index)
        if layer_index == self.config.num_layers - 1:
            if runtime.prediction_slice is not None:
                prefix = outputs_embeds[0]
                if prefix is None:
                    raise ValueError("final unified graph output lost its prefix stream")
                runtime.context.row_prediction_hidden = prefix[:, runtime.prediction_slice]
            return outputs_embeds, runtime
        if layer_index != self.config.penultimate_layer:
            return outputs_embeds, runtime
        prefix = outputs_embeds[0]
        if prefix is None:
            raise ValueError("penultimate unified graph output lost its prefix stream")
        prior = self.codec.decode_prediction(
            prefix[:, runtime.prior_slice],
            geometry_valid=runtime.context.previous_posterior.geometry_valid,
        )
        # Age is deterministic posterior metadata, not a free transition head.
        # The predictive prior advances time exactly once even when every
        # physical modality is absent. Observation evidence may subsequently
        # revise the continue/birth mixture, but it cannot stop the clock.
        prior = replace(
            prior,
            expected_age=posterior_expected_age(
                runtime.context.previous_posterior.expected_age,
                prior.lifecycle_log_probs,
                runtime.context.elapsed_time[:, None],
            ),
            evidence_age=(
                runtime.context.previous_posterior.evidence_age
                + runtime.context.elapsed_time[:, None]
            ),
        )
        geometry_valid = runtime.context.previous_posterior.geometry_valid | (
            runtime.context.modality_geometry_valid.any(dim=1)
        )
        raw_posterior = self.codec.decode_prediction(
            prefix[:, runtime.posterior_slice],
            geometry_valid=geometry_valid,
        )
        if runtime.grouped_evidence is None:
            raise RuntimeError("state write requires the penultimate shared-Q/K/V evidence")
        posterior = self._fuse_modality_observations(
            prior,
            raw_posterior,
            runtime.grouped_evidence,
            runtime.context,
        )
        pair = self.codec.paired_action_tokens(prior, posterior)
        paired_tokens = torch.cat(
            (
                self._add_role_tail(pair.prior_tokens, TokenRole.PRIOR),
                self._add_role_tail(
                    pair.pair_tokens,
                    TokenRole.POSTERIOR,
                    protected_width=2 * self.config.codec.canonical_width,
                ),
            ),
            dim=1,
        )
        updated_prefix = prefix.clone()
        updated_prefix[:, runtime.prior_slice.start : runtime.posterior_slice.stop] = paired_tokens
        updated = list(outputs_embeds)
        updated[0] = updated_prefix
        runtime.context.predictive_prior = prior
        runtime.context.posterior = posterior
        runtime.context.final_action_pair = PairedBeliefTokens(
            tokens=paired_tokens,
            prior_canonical=pair.prior_canonical,
            posterior_canonical=pair.posterior_canonical,
            capacity=pair.capacity,
        )
        return updated, runtime

    def _typed_belief_tokens(
        self,
        state: UnifiedBeliefState,
        role: TokenRole,
    ) -> torch.Tensor:
        return self._add_role_tail(self.codec.encode(state), role)

    def _materialize_native_metadata(
        self,
        context: LingBotUnifiedContext,
        *,
        attention_mask: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        original_prefix_count: int,
    ) -> None:
        metadata = (
            context.native_roles,
            context.native_valid,
            context.native_footprint,
            context.native_modality_ids,
        )
        if all(value is not None for value in metadata):
            if context.native_group_ids is None:
                assert context.native_roles is not None
                context.native_group_ids = torch.full_like(context.native_roles, -1)
            return
        if any(value is not None for value in metadata):
            raise ValueError("native token metadata must be supplied completely or inferred")
        if visual_pos_masks is None or visual_pos_masks.shape != (
            context.previous_posterior.batch_size,
            original_prefix_count,
        ):
            raise ValueError("vision metadata inference requires the exact host visual mask")
        if "vision" not in self.config.modality_names:
            raise ValueError("vision metadata inference requires a declared vision modality")
        measurement_count = self.config.native_measurement_query_tokens
        prediction_count = self.config.native_prediction_query_tokens
        training_count = measurement_count + prediction_count
        if training_count > original_prefix_count:
            raise ValueError("training-query tail is longer than the native prefix")
        valid = attention_mask.diagonal(dim1=-2, dim2=-1)[:, :original_prefix_count].clone()
        sensor = visual_pos_masks & valid
        if training_count and sensor[:, -training_count:].any():
            raise ValueError("training-query tail overlaps host visual tokens")
        roles = torch.full_like(valid, int(TokenRole.LANGUAGE), dtype=torch.long)
        roles[sensor] = int(TokenRole.SENSOR)
        if measurement_count:
            query_start = original_prefix_count - training_count
            roles[:, query_start : query_start + measurement_count] = int(
                TokenRole.MEASUREMENT_QUERY
            )
        if prediction_count:
            roles[:, -prediction_count:] = int(TokenRole.HOST_FUTURE_QUERY)
        sensor_count = sensor.sum(dim=-1, keepdim=True)
        footprint = sensor.to(torch.float32) / sensor_count.clamp_min(1)
        modality_ids = torch.full_like(roles, -1)
        vision_id = self.config.modality_names.index("vision")
        modality_ids[sensor] = vision_id
        context.native_roles = roles
        context.native_valid = valid
        context.native_footprint = footprint
        context.native_modality_ids = modality_ids
        context.native_group_ids = torch.full_like(roles, -1)

    def _validate_native_query_layout(
        self,
        context: LingBotUnifiedContext,
        *,
        original_prefix_count: int,
    ) -> None:
        """Bind trusted query roles to the immutable released-host tail spans."""

        assert context.native_roles is not None
        measurement_count = self.config.native_measurement_query_tokens
        prediction_count = self.config.native_prediction_query_tokens
        query_count = measurement_count + prediction_count
        if query_count > original_prefix_count:
            raise ValueError("native query contract is longer than the host prefix")
        expected_measurement = torch.zeros_like(context.native_roles, dtype=torch.bool)
        expected_prediction = torch.zeros_like(context.native_roles, dtype=torch.bool)
        if measurement_count:
            start = original_prefix_count - query_count
            expected_measurement[:, start : start + measurement_count] = True
        if prediction_count:
            expected_prediction[:, -prediction_count:] = True
        actual_measurement = context.native_roles == int(TokenRole.MEASUREMENT_QUERY)
        actual_prediction = context.native_roles == int(TokenRole.HOST_FUTURE_QUERY)
        if not torch.equal(actual_measurement, expected_measurement):
            raise ValueError("native measurement-query roles differ from the host tail contract")
        if not torch.equal(actual_prediction, expected_prediction):
            raise ValueError("native prediction-query roles differ from the host tail contract")

    def _with_birth_proposals(
        self,
        state: UnifiedBeliefState,
        noise: torch.Tensor,
    ) -> UnifiedBeliefState:
        proposal = self.birth_mean.float() + self.birth_log_scale.float().exp() * noise
        retained = state.nonempty_probability.unsqueeze(-1)
        content = retained * state.content + (1.0 - retained) * proposal
        return replace(state, content=content)

    def _add_role_tail(
        self,
        tokens: torch.Tensor,
        role: TokenRole,
        *,
        protected_width: int | None = None,
    ) -> torch.Tensor:
        canonical_width = self.config.codec.canonical_width
        protected_width = canonical_width if protected_width is None else protected_width
        if not canonical_width <= protected_width <= self.config.codec.host_width:
            raise ValueError("role-tail protected width is outside the host token")
        if protected_width == self.config.codec.host_width:
            return tokens
        updated = tokens.clone()
        role_offset = protected_width - canonical_width
        updated[..., protected_width:] += self.role_tail[int(role), role_offset:].to(tokens)
        return updated

    def _row_prediction_queries(
        self,
        prefix: torch.Tensor,
        *,
        context: LingBotUnifiedContext,
        capacity: int,
    ) -> torch.Tensor | None:
        request = context.prediction_request
        if request is None:
            return None
        if self.config.native_prediction_query_tokens <= 0:
            raise ValueError("row prediction requires released host future-query seeds")
        assert context.native_roles is not None
        assert context.native_valid is not None
        source = (context.native_roles == int(TokenRole.HOST_FUTURE_QUERY)) & context.native_valid
        counts = source.sum(dim=-1)
        if not torch.equal(
            counts,
            torch.full_like(counts, self.config.native_prediction_query_tokens),
        ):
            raise ValueError("native future-query seed count differs from the host contract")
        seed = (prefix * source.unsqueeze(-1).to(prefix)).sum(dim=1, keepdim=True)
        seed = seed / counts.clamp_min(1).to(prefix).reshape(-1, 1, 1)
        modality_index = self.config.modality_names.index(request.modality)
        horizon = float(request.horizon)
        horizon_features = prefix.new_tensor([math.log1p(horizon), 1.0 / (1.0 + horizon)]).reshape(
            1, 2
        )
        typed = self.predictive_modality_embedding[modality_index].to(prefix)
        typed = typed + self.predictive_horizon_projection(
            horizon_features.to(self.predictive_horizon_projection.weight)
        ).reshape(-1).to(prefix)
        return (seed + typed.reshape(1, 1, -1)).expand(-1, capacity, -1)

    @staticmethod
    def _restrict_row_prediction_mask(
        mask: torch.Tensor,
        *,
        layout: TokenLayout,
        transition_index: int,
        prior_slice: slice,
        posterior_slice: slice,
        prediction_slice: slice,
    ) -> torch.Tensor:
        """Make row query ``k`` read only physical row ``k`` and causal metadata."""

        capacity = prediction_slice.stop - prediction_slice.start
        if capacity != prior_slice.stop - prior_slice.start or capacity != (
            posterior_slice.stop - posterior_slice.start
        ):
            raise ValueError("row prediction slices must have equal capacity")
        if (
            not 0 <= transition_index < mask.shape[-1]
            or (layout.roles[:, transition_index] != int(TokenRole.HISTORY)).any()
        ):
            raise ValueError("row prediction requires the inserted transition token")
        updated = mask.clone()
        queries = torch.arange(
            prediction_slice.start,
            prediction_slice.stop,
            device=mask.device,
        )
        # Start from the strict source-side state boundary.  In particular, an
        # arbitrary native token marked HISTORY must not bypass the posterior-
        # sufficiency probe; only the graph-owned transition token is restored.
        current_state = layout.roles == int(TokenRole.CURRENT_STATE)
        updated[:, prediction_slice, :] &= current_state.unsqueeze(1)
        rows = torch.arange(capacity, device=mask.device)
        updated[:, queries, transition_index] = True
        updated[:, queries, prior_slice.start + rows] = True
        updated[:, queries, posterior_slice.start + rows] = True
        updated[:, queries, queries] = True
        return updated

    def _insert_position_ids(
        self,
        position_ids: torch.Tensor,
        *,
        native_valid: torch.Tensor,
        original_prefix_count: int,
        capacity: int,
        action_count: int,
        prediction_count: int,
    ) -> torch.Tensor:
        prefix_positions = position_ids[:, :, :original_prefix_count]
        masked = prefix_positions.masked_fill(~native_valid.unsqueeze(0), 0)
        maximum = masked.amax(dim=-1, keepdim=True)
        transition = maximum + 1
        prior = (maximum + 2).expand(-1, -1, capacity)
        posterior = (maximum + 3).expand(-1, -1, capacity)
        context = maximum + 3
        retrieval = (maximum + 4).expand(-1, -1, self.config.retrieval_tokens)
        pieces = [prefix_positions, transition, prior, posterior, context, retrieval]
        position_shift = 4
        if prediction_count:
            prediction = (maximum + 5).expand(-1, -1, prediction_count)
            pieces.append(prediction)
            position_shift = 5
        if action_count:
            pieces.append(position_ids[:, :, original_prefix_count:] + position_shift)
        return torch.cat(pieces, dim=-1)

    def _fuse_modality_observations(
        self,
        prior: UnifiedBeliefState,
        amortized_posterior: UnifiedBeliefState,
        evidence: GroupedRelationEvidence,
        context: LingBotUnifiedContext,
    ) -> UnifiedBeliefState:
        batch, modalities, capacity = evidence.support.shape
        if (batch, modalities, capacity) != (
            prior.batch_size,
            self.config.modality_count,
            prior.capacity,
        ):
            raise ValueError("grouped evidence does not match the configured belief state")
        grouped_message = evidence.message.permute(0, 2, 1, 3, 4).flatten(-2)
        if grouped_message.shape[-1] != self.config.attention_value_width:
            raise ValueError("shared value heads do not reconstruct the attention value width")
        support = evidence.support.permute(0, 2, 1)
        robust = evidence.robust_log_likelihood_ratio.permute(0, 2, 1)
        measurement_input = torch.cat(
            (grouped_message, support.unsqueeze(-1), robust.unsqueeze(-1)),
            dim=-1,
        ).to(self.measurement_projection.weight)
        measurement = self.measurement_projection(measurement_input).float()
        cursor = 0

        def take(width: int) -> torch.Tensor:
            nonlocal cursor
            value = measurement[..., cursor : cursor + width]
            cursor += width
            return value

        lifecycle_odds = take(2)
        incremental = torch.cat(
            (lifecycle_odds, torch.zeros_like(lifecycle_odds[..., :1])),
            dim=-1,
        )
        # The projection may use robust local log odds to retain a small strong
        # region, but its correction magnitude must still vanish continuously
        # with physical ownership. This is a likelihood strength, not an
        # identity-changing confidence threshold.
        incremental = incremental * support.unsqueeze(-1)
        available = evidence.available[:, None, :].expand(-1, capacity, -1)
        reliability = reliability_simplex(
            prior.content.new_tensor(self.config.prior_reliability),
            prior.content.new_tensor(self.config.modality_reliability),
            available,
        )
        lifecycle = logarithmic_lifecycle_pool(
            prior.lifecycle_log_probs,
            incremental,
            reliability[..., 1:],
            available=available,
        )

        geometry = self.config.codec.geometry_dim
        observation_means = take(geometry)
        triangle = geometry * (geometry + 1) // 2
        cholesky_values = take(triangle)
        if cursor != measurement.shape[-1]:
            raise AssertionError("measurement parser did not consume its declared width")
        rows, cols = torch.tril_indices(geometry, geometry, device=measurement.device)
        cholesky = measurement.new_zeros((*measurement.shape[:3], geometry, geometry))
        cholesky[..., rows, cols] = cholesky_values
        # A normalized relation message identifies the measurement mean but no
        # longer carries how much physical evidence supported that relation.
        # Precision is therefore proportional to the tokenizer-invariant soft
        # ownership mass.  This makes zero support an exact zero likelihood
        # factor and prevents a tiny responsibility from producing a full-
        # strength geometry update.
        information_increments = support.unsqueeze(-1).unsqueeze(-1) * (
            cholesky @ cholesky.transpose(-1, -2)
        )
        modality_geometry_valid = context.modality_geometry_valid.permute(0, 2, 1, 3)
        valid_pair = modality_geometry_valid.unsqueeze(-1) & modality_geometry_valid.unsqueeze(-2)
        information_increments = information_increments.masked_fill(~valid_pair, 0)
        observation_means = observation_means * modality_geometry_valid.to(measurement.dtype)
        geometry_available = available & modality_geometry_valid.any(dim=-1)
        ci_weights = deterministic_logdet_ci_weights(
            prior.geometry_information,
            information_increments,
            geometry_available,
            iterations=self.config.ci_iterations,
            step_size=self.config.ci_step_size,
            information_floor=self.config.codec.information_floor,
        )
        geometry_fusion = generalized_covariance_intersection(
            prior.geometry_mean,
            prior.geometry_information,
            observation_means,
            information_increments,
            ci_weights,
        )
        final_geometry_valid = prior.geometry_valid | modality_geometry_valid.any(dim=-2)
        geometry_mean = geometry_fusion.mean * final_geometry_valid.to(geometry_fusion.mean.dtype)
        geometry_information = geometry_fusion.information.masked_fill(
            ~(final_geometry_valid.unsqueeze(-1) & final_geometry_valid.unsqueeze(-2)),
            0,
        )

        evidence_strength = 1.0 - torch.prod(
            1.0 - support.clamp(0, 1) * available.to(support.dtype), dim=-1
        )
        expected_age = posterior_expected_age(
            context.previous_posterior.expected_age,
            lifecycle,
            context.elapsed_time[:, None],
        )
        evidence_age = (1.0 - evidence_strength) * prior.evidence_age
        content_strength = evidence_strength.unsqueeze(-1)

        fused = UnifiedBeliefState(
            content=prior.content
            + content_strength * (amortized_posterior.content - prior.content),
            lifecycle_log_probs=lifecycle,
            geometry_mean=geometry_mean,
            geometry_information=geometry_information,
            geometry_valid=final_geometry_valid,
            content_log_variance=prior.content_log_variance
            + content_strength
            * (amortized_posterior.content_log_variance - prior.content_log_variance),
            expected_age=expected_age,
            evidence_age=evidence_age,
        )
        batch_available = evidence.available.any(dim=-1)

        def select(current: torch.Tensor, fallback: torch.Tensor) -> torch.Tensor:
            shape = (batch_available.shape[0],) + (1,) * (current.ndim - 1)
            return torch.where(batch_available.reshape(shape), current, fallback)

        return UnifiedBeliefState(
            content=select(fused.content, prior.content),
            lifecycle_log_probs=select(fused.lifecycle_log_probs, prior.lifecycle_log_probs),
            geometry_mean=select(fused.geometry_mean, prior.geometry_mean),
            geometry_information=select(fused.geometry_information, prior.geometry_information),
            geometry_valid=select(fused.geometry_valid, prior.geometry_valid),
            content_log_variance=select(
                fused.content_log_variance,
                prior.content_log_variance,
            ),
            expected_age=select(fused.expected_age, prior.expected_age),
            evidence_age=select(fused.evidence_age, prior.evidence_age),
        )

    def _validate_layer(self, layer_index: int) -> None:
        if type(layer_index) is not int:
            raise TypeError("layer_index must be a Python int")
        if not 0 <= layer_index < self.config.num_layers:
            raise ValueError("layer_index is outside the configured LingBot depth")


def install_lingbot_unified_belief_graph(
    policy: nn.Module,
    graph: LingBotUnifiedBeliefGraph,
) -> None:
    """Register the unified graph on an official patched LingBot VLA2 policy."""

    contract = LingBotHostContract.from_policy(policy)
    expected = (
        graph.config.codec.host_width,
        graph.config.attention_value_width,
        graph.config.num_layers,
        graph.config.executed_action_dim,
        graph.config.native_measurement_query_tokens,
        graph.config.native_prediction_query_tokens,
    )
    actual = (
        contract.prefix_width,
        contract.attention_value_width,
        contract.num_layers,
        contract.executed_action_dim,
        contract.native_measurement_query_tokens,
        contract.native_prediction_query_tokens,
    )
    if expected != actual:
        raise ValueError(f"unified graph/host contract mismatch: expected {expected}, got {actual}")
    model = getattr(policy, "model", None)
    host = getattr(model, "qwenvl_with_expert", None)
    setter = getattr(host, "set_unified_belief_graph", None)
    if model is None or host is None or setter is None:
        raise TypeError("policy does not expose the pinned unified LingBot graph hook")
    reference = host.qwenvl.model.language_model.layers[0].self_attn.q_proj.weight
    graph.to(device=reference.device, dtype=reference.dtype)
    graph.train(getattr(policy, "training", True))
    setter(graph)

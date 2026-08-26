from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final

import torch
import yaml
from torch import nn

from picf_next.wla_upstream import WLASourceReceipt, load_wla_action_symbols


WLA_METAQUERY_COUNT: Final = 64
WLA_ADDED_TOKEN_COUNT: Final = WLA_METAQUERY_COUNT + 2
WLA_META_TOKEN_COUNT: Final = WLA_METAQUERY_COUNT + 4
WLA_ACTION_LAYER_COUNT: Final = 28
LINGBOT_HOST_LAYER_COUNT: Final = 36


@dataclass(frozen=True, slots=True)
class WLAMetaQueryLayout:
    base_count: int
    newline_index: int
    begin_index: int
    query_slice: slice
    end_index: int
    im_end_index: int

    @property
    def total_count(self) -> int:
        return self.im_end_index + 1


@dataclass(frozen=True, slots=True)
class LingBotWLAHostOutput:
    layerwise_query_states: tuple[torch.Tensor, ...]
    normalized_host: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    layout: WLAMetaQueryLayout
    picf_runtime: Any


@dataclass(frozen=True, slots=True)
class LingBotWLAActionOutput:
    loss: torch.Tensor
    host: LingBotWLAHostOutput


@dataclass(frozen=True, slots=True)
class LingBotWLAInferenceOutput:
    actions: torch.Tensor
    host: LingBotWLAHostOutput


@dataclass(frozen=True, slots=True)
class LingBotWLACalvinOutput:
    action: LingBotWLAActionOutput
    current_visual_embeddings: torch.Tensor
    current_visual_valid: torch.Tensor
    native_root_outputs: tuple[torch.Tensor, ...]


def wla_calvin_action_mask(model_inputs: dict[str, Any]) -> torch.Tensor:
    """Return the exact CALVIN joint-valid and non-padding action surface."""

    actions = model_inputs["actions"]
    joint_mask = model_inputs["joint_mask"]
    action_is_pad = model_inputs["action_is_pad"]
    if (
        not isinstance(actions, torch.Tensor)
        or not isinstance(joint_mask, torch.Tensor)
        or not isinstance(action_is_pad, torch.Tensor)
        or joint_mask.dtype != torch.bool
        or joint_mask.shape != actions.shape
        or action_is_pad.dtype != torch.bool
        or action_is_pad.shape != actions.shape[:2]
    ):
        raise ValueError("CALVIN action validity requires bool joint and time-padding masks")
    return joint_mask & ~action_is_pad.unsqueeze(-1)


def append_wla_metaquery_surface(
    *,
    base_hidden: torch.Tensor,
    base_attention_mask: torch.Tensor,
    base_position_ids: torch.Tensor,
    visual_pos_masks: torch.Tensor | None,
    meta_token_embeddings: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor | None,
    WLAMetaQueryLayout,
]:
    """Append WLA's BOI, 64 metaqueries, EOI, and IM_END token surface.

    WLA inserts ``\n<begin_of_img><img0>...<img63><end_of_img><|im_end|>``
    after the multimodal prompt. The newline and ``im_end`` embeddings come from
    the host vocabulary; BOI, EOI, and the 64 image-query embeddings are the
    upstream-added rows. Its Qwen3-VL backbone is autoregressive, so prompt rows
    cannot read the appended tokens, while each appended row can read the
    complete valid prompt and its causal token prefix. LingBot keeps its released
    prompt-to-prompt mask; only the appended WLA suffix receives this source
    causal contract.
    """

    if base_hidden.ndim != 3:
        raise ValueError("LingBot base hidden states must have shape [batch,tokens,width]")
    batch, base_count, width = base_hidden.shape
    if (
        base_attention_mask.shape != (batch, base_count, base_count)
        or base_attention_mask.dtype != torch.bool
        or base_attention_mask.device != base_hidden.device
    ):
        raise ValueError("LingBot base attention mask must be boolean [batch,tokens,tokens]")
    if (
        base_position_ids.shape != (3, batch, base_count)
        or base_position_ids.device != base_hidden.device
    ):
        raise ValueError("LingBot base MRoPE IDs must have shape [3,batch,tokens]")
    if (
        meta_token_embeddings.shape != (WLA_META_TOKEN_COUNT, width)
        or meta_token_embeddings.device != base_hidden.device
        or meta_token_embeddings.dtype != base_hidden.dtype
    ):
        raise ValueError("WLA meta-token embeddings differ from the LingBot host surface")
    if visual_pos_masks is not None and (
        visual_pos_masks.shape != (batch, base_count)
        or visual_pos_masks.dtype != torch.bool
        or visual_pos_masks.device != base_hidden.device
    ):
        raise ValueError("LingBot visual-position mask differs from its base token surface")

    base_valid = base_attention_mask.diagonal(dim1=-2, dim2=-1)
    if not base_valid.any(dim=1).all():
        raise ValueError("every LingBot sample must contain at least one valid base token")
    suffix = meta_token_embeddings.unsqueeze(0).expand(batch, -1, -1)
    hidden = torch.cat((base_hidden, suffix), dim=1)

    total_count = base_count + WLA_META_TOKEN_COUNT
    attention = torch.zeros(
        batch,
        total_count,
        total_count,
        dtype=torch.bool,
        device=base_hidden.device,
    )
    attention[:, :base_count, :base_count] = base_attention_mask
    attention[:, base_count:, :base_count] = base_valid.unsqueeze(1)
    suffix_causal = torch.ones(
        WLA_META_TOKEN_COUNT,
        WLA_META_TOKEN_COUNT,
        dtype=torch.bool,
        device=base_hidden.device,
    ).tril()
    attention[:, base_count:, base_count:] = suffix_causal

    valid_positions = base_position_ids.masked_fill(~base_valid.unsqueeze(0), 0)
    offsets = valid_positions.amax(dim=(0, 2)) + 1
    suffix_1d = offsets[:, None] + torch.arange(
        WLA_META_TOKEN_COUNT,
        dtype=base_position_ids.dtype,
        device=base_hidden.device,
    ).unsqueeze(0)
    suffix_positions = suffix_1d.unsqueeze(0).expand(3, -1, -1)
    positions = torch.cat((base_position_ids, suffix_positions), dim=-1)

    if visual_pos_masks is not None:
        visual_pos_masks = torch.cat(
            (
                visual_pos_masks,
                torch.zeros(
                    batch,
                    WLA_META_TOKEN_COUNT,
                    dtype=torch.bool,
                    device=base_hidden.device,
                ),
            ),
            dim=1,
        )

    layout = WLAMetaQueryLayout(
        base_count=base_count,
        newline_index=base_count,
        begin_index=base_count + 1,
        query_slice=slice(base_count + 2, base_count + 2 + WLA_METAQUERY_COUNT),
        end_index=base_count + 2 + WLA_METAQUERY_COUNT,
        im_end_index=base_count + 3 + WLA_METAQUERY_COUNT,
    )
    if layout.total_count != total_count:
        raise RuntimeError("WLA metaquery layout arithmetic is inconsistent")
    return hidden, attention, positions, visual_pos_masks, layout


class LingBotWLASharedInterface(nn.Module):
    """WLA's complete layerwise action interface on the LingBot/PICF host.

    This module owns no object selector or lifecycle predictor. PICF evidence is
    first processed by the complete LingBot graph. WLA's learned suffix tokens
    then read that shared state through all 36 LingBot layers, and the untouched
    upstream 28-layer Action Expert consumes the final 28 layerwise query sets.
    """

    def __init__(
        self,
        *,
        action_head: nn.Module,
        source: WLASourceReceipt,
        host_width: int = 2048,
        repeated_diffusion_steps: int = 16,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if host_width <= 0:
            raise ValueError("LingBot host width must be positive")
        if repeated_diffusion_steps != 16:
            raise ValueError("pinned WLA LIBERO source uses exactly 16 repeated diffusion steps")
        blocks = getattr(getattr(action_head, "model", None), "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != WLA_ACTION_LAYER_COUNT:
            raise ValueError("WLA action head differs from its exact 28-layer source topology")
        action_condition_width = int(
            getattr(getattr(action_head, "model", None).config, "cross_attention_dim", -1)
        )
        if action_condition_width != host_width:
            raise ValueError(
                "WLA action cross-attention width must equal the native LingBot host width"
            )
        self.action_head = action_head
        self.source = source
        self.host_width = host_width
        self.repeated_diffusion_steps = repeated_diffusion_steps
        self.added_meta_token_embeddings = nn.Parameter(
            torch.empty(WLA_ADDED_TOKEN_COUNT, host_width, device=device, dtype=dtype)
        )
        self.register_buffer(
            "meta_tokens_initialized",
            torch.tensor(False, dtype=torch.bool, device=device),
            persistent=True,
        )
        self.register_buffer(
            "newline_token_id",
            torch.tensor(-1, dtype=torch.long, device=device),
            persistent=True,
        )
        self.register_buffer(
            "im_end_token_id",
            torch.tensor(-1, dtype=torch.long, device=device),
            persistent=True,
        )

    @classmethod
    def from_pinned_source(
        cls,
        source_root: Path | str,
        *,
        host_width: int = 2048,
        max_action_dim: int | None = None,
        max_state_dim: int | None = None,
        chunk_size: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LingBotWLASharedInterface":
        symbols = load_wla_action_symbols(source_root)
        source_config = yaml.safe_load(
            (symbols.source.root / "configs/libero_all_image_action.yaml").read_text()
        )
        required = (
            "diffusion_model_cfg",
            "max_action_dim",
            "max_state_dim",
            "chunk_size",
            "num_inference_timesteps",
            "add_pos_embed",
            "max_seq_len",
            "noise_beta_alpha",
            "noise_beta_beta",
            "num_timestep_buckets",
            "noise_s",
            "repeated_diffusion_steps",
        )
        missing = [name for name in required if name not in source_config]
        if missing:
            raise ValueError(f"pinned WLA LIBERO config is missing {missing}")
        selected = {name: source_config[name] for name in required}
        selected["diffusion_model_cfg"] = dict(selected["diffusion_model_cfg"])
        selected["diffusion_model_cfg"]["cross_attention_dim"] = host_width
        if max_action_dim is not None:
            selected["max_action_dim"] = max_action_dim
        if max_state_dim is not None:
            selected["max_state_dim"] = max_state_dim
        if chunk_size is not None:
            selected["chunk_size"] = chunk_size
        config = SimpleNamespace(**selected)
        action_head = symbols.action_head(config).to(device=device, dtype=dtype)
        return cls(
            action_head=action_head,
            source=symbols.source,
            repeated_diffusion_steps=config.repeated_diffusion_steps,
            host_width=host_width,
            device=device,
            dtype=dtype,
        )

    @torch.no_grad()
    def initialize_meta_tokens_from_lingbot(
        self,
        joint: nn.Module,
        *,
        newline_token_id: int,
        im_end_token_id: int,
    ) -> None:
        """Use the same HF mean-resizing primitive called by upstream WLA."""

        if bool(self.meta_tokens_initialized.item()):
            raise RuntimeError("WLA meta-token embeddings may be initialized only once")
        try:
            old_embeddings = joint.qwenvl.model.language_model.embed_tokens
            initializer = joint.qwenvl._init_added_embeddings_weights_with_mean
        except AttributeError as error:
            raise TypeError("LingBot host lacks the Qwen3-VL mean-resizing primitive") from error
        if (
            not isinstance(old_embeddings, nn.Embedding)
            or old_embeddings.embedding_dim != self.host_width
        ):
            raise ValueError("LingBot token table differs from the WLA query parameter surface")
        for token_id in (newline_token_id, im_end_token_id):
            if token_id < 0 or token_id >= old_embeddings.num_embeddings:
                raise ValueError("WLA suffix references a token outside the LingBot vocabulary")
        temporary = nn.Embedding(
            WLA_ADDED_TOKEN_COUNT,
            self.host_width,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
        )
        initializer(
            old_embeddings,
            temporary,
            old_embeddings.embedding_dim,
            old_embeddings.num_embeddings,
            WLA_ADDED_TOKEN_COUNT,
        )
        # LingBot's released loader materializes the host in FP32 before FSDP2
        # establishes WLA's published BF16 training surface.  Run Qwen's exact
        # mean-resize initializer in the host table's native precision, then
        # perform the same one-time cast that FSDP mixed precision applies to
        # the trainable WLA parameter.  Requiring equal pre-FSDP dtypes would
        # reject the released LingBot loading order without preserving any
        # upstream WLA behavior.
        self.added_meta_token_embeddings.copy_(
            temporary.weight.to(
                device=self.added_meta_token_embeddings.device,
                dtype=self.added_meta_token_embeddings.dtype,
            )
        )
        self.newline_token_id.fill_(newline_token_id)
        self.im_end_token_id.fill_(im_end_token_id)
        self.meta_tokens_initialized.fill_(True)

    def _meta_token_surface(self, joint: nn.Module) -> torch.Tensor:
        """Assemble WLA's exact token order while sharing existing host rows."""

        embedding = joint.qwenvl.model.language_model.embed_tokens
        existing_ids = torch.stack((self.newline_token_id, self.im_end_token_id))
        existing = embedding(existing_ids)
        newline = existing[0:1]
        im_end = existing[1:2]
        # Upstream adds IDs in [BOI, EOI, IMG0..IMG63] order, then emits them
        # in [BOI, IMG0..IMG63, EOI] order in the prompt suffix.
        added = self.added_meta_token_embeddings
        return torch.cat((newline, added[0:1], added[2:], added[1:2], im_end), dim=0)

    def _validate_joint(self, joint: nn.Module) -> None:
        try:
            host_model = joint.qwenvl.model.language_model
            layers = host_model.layers
        except AttributeError as error:
            raise TypeError("LingBot joint host lacks its released language layers") from error
        if len(layers) != LINGBOT_HOST_LAYER_COUNT:
            raise ValueError("ADR224 requires LingBot's complete 36-layer host")
        if int(getattr(joint.qwenvl.config.text_config, "hidden_size", -1)) != self.host_width:
            raise ValueError("LingBot host width differs from WLA's published interface")
        if getattr(joint.config, "attention_implementation", None) not in {
            "eager",
            "flex",
            "flex_cached",
        }:
            raise ValueError("LingBot attention implementation is unsupported")
        if not bool(self.meta_tokens_initialized.item()):
            raise RuntimeError("WLA meta-token embeddings were not source-faithfully initialized")
        if int(self.newline_token_id.item()) < 0 or int(self.im_end_token_id.item()) < 0:
            raise RuntimeError("WLA existing suffix token IDs were not initialized")

    def encode_host(
        self,
        joint: nn.Module,
        *,
        prefix_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
        picf_native_context: Any = None,
    ) -> LingBotWLAHostOutput:
        self._validate_joint(joint)
        if prefix_embeds.dtype != self.added_meta_token_embeddings.dtype or (
            prefix_embeds.device != self.added_meta_token_embeddings.device
        ):
            raise ValueError("LingBot prefix and WLA interface must share device and dtype")

        graph = getattr(joint, "picf_native_graph", None)
        picf_runtime = None
        if graph is not None:
            (
                prepared_embeds,
                attention_mask,
                position_ids,
                visual_pos_masks,
                picf_runtime,
            ) = graph.prepare_joint_inputs(
                inputs_embeds=[prefix_embeds, None],
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                context=picf_native_context,
            )
            prefix_embeds = prepared_embeds[0]
            if prefix_embeds is None or prepared_embeds[1] is not None:
                raise RuntimeError("PICF preparation changed the host-only WLA contract")

        (
            host_hidden,
            attention_mask,
            position_ids,
            visual_pos_masks,
            layout,
        ) = append_wla_metaquery_surface(
            base_hidden=prefix_embeds,
            base_attention_mask=attention_mask,
            base_position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            meta_token_embeddings=self._meta_token_surface(joint),
        )
        host_model = joint.qwenvl.model.language_model
        layerwise_queries: list[torch.Tensor] = []

        for layer_index in range(LINGBOT_HOST_LAYER_COUNT):
            prefix_qk_bias = None
            if graph is not None:
                base_bias = graph.layerwise_qk_address_bias(
                    prefix_hidden=host_hidden[:, : layout.base_count],
                    runtime=picf_runtime,
                )
                if base_bias is not None:
                    prefix_qk_bias = torch.cat(
                        (
                            base_bias,
                            torch.zeros_like(host_hidden[:, layout.base_count :]),
                        ),
                        dim=1,
                    )
            query, key, value = host_model.layers[layer_index](
                host_hidden,
                compute_kqv=True,
                qk_input_bias=prefix_qk_bias,
            )
            query, key = joint.apply_mrope(query.float(), key.float(), position_ids)
            value = value.float()
            layer_mask = attention_mask

            memory_inputs = None
            if graph is not None:
                memory_inputs = graph.layerwise_memory_inputs(
                    layer_index=layer_index,
                    runtime=picf_runtime,
                )
            if memory_inputs is not None:
                memory_hidden, memory_qk_bias, history_visibility = memory_inputs
                memory_query, memory_key, memory_value = host_model.layers[layer_index](
                    memory_hidden,
                    compute_kqv=True,
                    qk_input_bias=memory_qk_bias,
                )
                memory_positions = position_ids.new_zeros(
                    (3, memory_hidden.shape[0], memory_hidden.shape[1])
                )
                _, memory_key = joint.apply_mrope(
                    memory_query.float(),
                    memory_key.float(),
                    memory_positions,
                )
                query_history_visibility = torch.zeros(
                    history_visibility.shape[0],
                    WLA_META_TOKEN_COUNT,
                    history_visibility.shape[2],
                    dtype=torch.bool,
                    device=history_visibility.device,
                )
                history_visibility = torch.cat(
                    (history_visibility, query_history_visibility),
                    dim=1,
                )
                key = torch.cat((memory_key, key), dim=1)
                value = torch.cat((memory_value.float(), value), dim=1)
                layer_mask = torch.cat((history_visibility, layer_mask), dim=-1)

            attention_output = joint.attention_interface(query, key, value, layer_mask)
            host_hidden = host_model.layers[layer_index](
                host_hidden,
                attention_output,
                0,
                host_hidden.shape[1],
                output_atten=True,
            )
            host_hidden = joint._apply_deepstack(
                host_hidden,
                layer_index,
                visual_pos_masks,
                deepstack_visual_embeds,
            )
            if graph is not None:
                graph.record_layerwise_posterior(
                    prefix_hidden=host_hidden[:, : layout.base_count],
                    runtime=picf_runtime,
                    layer_index=layer_index,
                )
                if graph.requires_intermediate_relation(
                    layer_index=layer_index,
                    runtime=picf_runtime,
                ):
                    graph.record_intermediate_relation(
                        normalized_prefix=host_model.norm(
                            host_hidden[:, : layout.base_count]
                        ),
                        runtime=picf_runtime,
                        layer_index=layer_index,
                    )
            layerwise_queries.append(host_hidden[:, layout.query_slice])

        normalized = host_model.norm(host_hidden)
        # Transformers' Qwen3-VL hidden-state recorder replaces the final
        # unnormalized layer output with the final normalized hidden state.
        # WLA then slices the last 28 entries. Mirror that exact behavior.
        layerwise_queries[-1] = normalized[:, layout.query_slice]
        selected_queries = tuple(layerwise_queries[-WLA_ACTION_LAYER_COUNT:])
        if len(selected_queries) != WLA_ACTION_LAYER_COUNT or any(
            query_state.shape[1:] != (WLA_METAQUERY_COUNT, self.host_width)
            for query_state in selected_queries
        ):
            raise RuntimeError("LingBot did not expose WLA's exact layerwise query surface")

        normalized_host = normalized[:, : layout.base_count]
        if graph is not None:
            finalized = graph.finalize_joint_outputs(
                outputs_embeds=[normalized_host, None],
                runtime=picf_runtime,
            )
            normalized_host = finalized[0]
            if normalized_host is None or finalized[1] is not None:
                raise RuntimeError("PICF finalization changed the WLA host-only contract")

        return LingBotWLAHostOutput(
            layerwise_query_states=selected_queries,
            normalized_host=normalized_host,
            attention_mask=attention_mask,
            position_ids=position_ids,
            layout=layout,
            picf_runtime=picf_runtime,
        )

    def run_prior_rollout(
        self,
        joint: nn.Module,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: list[torch.Tensor | None],
        visual_pos_masks: torch.Tensor | None,
        picf_native_context: Any,
    ) -> tuple[list[torch.Tensor | None], None, list[torch.Tensor]]:
        """Run PICF's recurrent prior through the same complete LingBot host.

        The released LingBot action expert is replaced by WLA and therefore
        cannot remain the owner of PICF state propagation.  The row-only input
        is instead processed by the exact host path used by WLA's factual
        action call.  WLA suffix rows cannot affect the base rows because the
        suffix attention surface is block lower triangular, so the finalized
        prior is mathematically independent of those suffix values.
        """

        if (
            not isinstance(inputs_embeds, list)
            or len(inputs_embeds) != 2
            or inputs_embeds[0] is None
            or inputs_embeds[1] is not None
        ):
            raise ValueError("WLA prior rollout requires LingBot's row-only input ABI")
        if picf_native_context is None:
            raise ValueError("WLA prior rollout requires a typed PICF context")
        host = self.encode_host(
            joint,
            prefix_embeds=inputs_embeds[0],
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=None,
            picf_native_context=picf_native_context,
        )
        if host.picf_runtime is None:
            raise RuntimeError("WLA prior rollout did not execute the PICF graph")
        return [host.normalized_host, None], None, []

    def forward(
        self,
        joint: nn.Module,
        *,
        prefix_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        state: torch.Tensor,
        picf_native_context: Any = None,
    ) -> LingBotWLAActionOutput:
        host = self.encode_host(
            joint,
            prefix_embeds=prefix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            picf_native_context=picf_native_context,
        )
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or state.shape[1] != 1:
            raise ValueError("WLA state must have shape [batch,state_dim] or [batch,1,state_dim]")
        repeats = self.repeated_diffusion_steps
        action_parameter = next(self.action_head.parameters())
        action_device = action_parameter.device
        action_dtype = action_parameter.dtype
        repeated_queries = [
            value.to(device=action_device, dtype=action_dtype).repeat(repeats, 1, 1)
            for value in host.layerwise_query_states
        ]
        action_values = actions.to(device=action_device, dtype=action_dtype)
        action_valid = action_mask.to(device=action_device, dtype=action_dtype)
        state_values = state.to(device=action_device, dtype=action_dtype)
        loss = self.action_head(
            repeated_queries,
            action_values.repeat(repeats, 1, 1),
            action_valid.repeat(repeats, 1, 1),
            state_values.repeat(repeats, 1, 1),
        )
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("pinned WLA action source returned a non-finite scalar")
        return LingBotWLAActionOutput(loss=loss, host=host)

    @torch.no_grad()
    def predict_action(
        self,
        joint: nn.Module,
        *,
        prefix_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
        state: torch.Tensor,
        picf_native_context: Any = None,
    ) -> LingBotWLAInferenceOutput:
        """Run the pinned WLA ActionHead ``predict_action`` without rewriting it."""

        host = self.encode_host(
            joint,
            prefix_embeds=prefix_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            picf_native_context=picf_native_context,
        )
        if state.ndim == 2:
            state = state.unsqueeze(1)
        if state.ndim != 3 or state.shape[1] != 1:
            raise ValueError("WLA state must have shape [batch,state_dim] or [batch,1,state_dim]")
        action_parameter = next(self.action_head.parameters())
        action_device = action_parameter.device
        action_dtype = action_parameter.dtype
        layerwise_queries = [
            value.to(device=action_device, dtype=action_dtype)
            for value in host.layerwise_query_states
        ]
        state_values = state.to(device=action_device, dtype=action_dtype)
        actions = self.action_head.predict_action(layerwise_queries, state_values)
        expected = (
            prefix_embeds.shape[0],
            int(getattr(self.action_head, "action_horizon", -1)),
            int(getattr(self.action_head, "action_dim", -1)),
        )
        if (
            not isinstance(actions, torch.Tensor)
            or actions.shape != expected
            or not actions.is_floating_point()
            or not torch.isfinite(actions).all()
        ):
            raise RuntimeError("pinned WLA predict_action returned an invalid action tensor")
        return LingBotWLAInferenceOutput(actions=actions, host=host)


def _prepare_lingbot_wla_calvin_prefix(
    policy: nn.Module,
    *,
    model_inputs: dict[str, Any],
    picf_native_context: Any,
) -> tuple[
    nn.Module,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    list[torch.Tensor] | None,
]:
    """Delegate CALVIN prefix packing, MRoPE and masks to released LingBot code."""

    required = (
        "images",
        "img_masks",
        "lang_tokens",
        "lang_masks",
        "image_grid_thw",
        "state",
    )
    missing = tuple(name for name in required if name not in model_inputs)
    if missing:
        raise KeyError(f"CALVIN WLA prefix is missing released LingBot inputs: {missing}")
    flow = getattr(policy, "model", None)
    if flow is None or not callable(getattr(flow, "embed_prefix", None)):
        raise TypeError("CALVIN WLA prefix requires the released LingBot flow model")
    joint = getattr(flow, "qwenvl_with_expert", None)
    if joint is None:
        raise TypeError("CALVIN WLA prefix cannot find LingBot's shared host")
    (
        prefix_embeds,
        prefix_pad_masks,
        prefix_attention_starts,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = flow.embed_prefix(
        model_inputs["images"],
        model_inputs["img_masks"],
        model_inputs["lang_tokens"],
        model_inputs["lang_masks"],
        image_grid_thw=model_inputs["image_grid_thw"],
    )
    flow._bind_picf_native_prefix(
        picf_native_context,
        prefix_pad_masks=prefix_pad_masks,
        visual_pos_masks=visual_pos_masks,
        lang_masks=model_inputs["lang_masks"],
    )
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks

    attention_mask = make_att_2d_masks(prefix_pad_masks, prefix_attention_starts)
    return (
        joint,
        prefix_embeds,
        attention_mask,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    )


def run_lingbot_wla_calvin_forward(
    policy: nn.Module,
    interface: LingBotWLASharedInterface,
    *,
    model_inputs: dict[str, Any],
    picf_native_context: Any,
) -> LingBotWLACalvinOutput:
    """Execute the released LingBot CALVIN prefix and the WLA action source.

    This function intentionally delegates prefix construction and the 2-D mask
    to LingBot's released implementation. It does not duplicate or simplify
    image processing, DeepStack, MRoPE, language packing, or PICF binding.
    """

    required = ("actions", "action_is_pad", "joint_mask")
    missing = tuple(name for name in required if name not in model_inputs)
    if missing:
        raise KeyError(f"CALVIN WLA forward is missing released LingBot inputs: {missing}")
    (
        joint,
        prefix_embeds,
        attention_mask,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = _prepare_lingbot_wla_calvin_prefix(
        policy,
        model_inputs=model_inputs,
        picf_native_context=picf_native_context,
    )

    image_grid = model_inputs["image_grid_thw"]
    if image_grid.ndim == 2:
        image_grid = image_grid.unsqueeze(0)
    if image_grid.ndim != 3 or image_grid.shape[1] < 1:
        raise ValueError("LingBot image grid must expose the primary CALVIN camera")
    merge_size = int(joint.qwenvl.config.vision_config.spatial_merge_size)
    primary_counts = image_grid[:, 0].prod(dim=-1) // (merge_size**2)
    if not torch.equal(primary_counts, primary_counts[:1].expand_as(primary_counts)):
        raise ValueError("WLA world conditioning requires one padded primary-camera width")
    primary_count = int(primary_counts[0].item())
    current_visual_embeddings = prefix_embeds[:, 1 : 1 + primary_count]
    current_visual_valid = visual_pos_masks[:, 1 : 1 + primary_count]
    if (
        current_visual_embeddings.shape[1] != primary_count
        or current_visual_valid.shape != current_visual_embeddings.shape[:2]
        or not current_visual_valid.all()
    ):
        raise RuntimeError("LingBot primary-camera embedding span is inconsistent")

    action_mask = wla_calvin_action_mask(model_inputs)
    action = interface(
        joint,
        prefix_embeds=prefix_embeds,
        attention_mask=attention_mask,
        position_ids=prefix_position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        actions=model_inputs["actions"],
        action_mask=action_mask,
        state=model_inputs["state"],
        picf_native_context=picf_native_context,
    )
    root_outputs_fn = getattr(picf_native_context, "root_output_tensors", None)
    if not callable(root_outputs_fn):
        raise TypeError("CALVIN WLA forward requires the complete PICF native context")
    native_root_outputs = root_outputs_fn()
    if not isinstance(native_root_outputs, tuple) or not native_root_outputs or any(
        not isinstance(value, torch.Tensor) for value in native_root_outputs
    ):
        raise RuntimeError("CALVIN WLA forward did not expose finalized PICF root tensors")
    return LingBotWLACalvinOutput(
        action=action,
        current_visual_embeddings=current_visual_embeddings,
        current_visual_valid=current_visual_valid,
        native_root_outputs=native_root_outputs,
    )


@torch.no_grad()
def predict_lingbot_wla_calvin_actions(
    policy: nn.Module,
    interface: LingBotWLASharedInterface,
    *,
    model_inputs: dict[str, Any],
    picf_native_context: Any,
) -> torch.Tensor:
    """Run released LingBot packing and WLA's untouched 32-step action sampler."""

    (
        joint,
        prefix_embeds,
        attention_mask,
        prefix_position_ids,
        visual_pos_masks,
        deepstack_visual_embeds,
    ) = _prepare_lingbot_wla_calvin_prefix(
        policy,
        model_inputs=model_inputs,
        picf_native_context=picf_native_context,
    )
    result = interface.predict_action(
        joint,
        prefix_embeds=prefix_embeds,
        attention_mask=attention_mask,
        position_ids=prefix_position_ids,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        state=model_inputs["state"],
        picf_native_context=picf_native_context,
    )
    return result.actions

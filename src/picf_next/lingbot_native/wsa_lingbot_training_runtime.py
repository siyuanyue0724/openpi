from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import nn

from picf_next.lingbot_native.action_posterior_receipt import (
    build_lingbot_joint_action_attention_layout,
)
from picf_next.lingbot_native.wsa_full_depth_adaptation import repeat_lingbot_kv_heads
from picf_next.lingbot_native.wsa_future_expert_runtime import (
    WSAFutureExpertRuntime,
    WSAFutureRuntime,
)
from picf_next.lingbot_native.wsa_joint_surface import (
    WSAJointTokenLayout,
    block_wsa_future_to_action_information_edge,
    build_wsa_joint_attention_mask_with_layout,
    concatenate_wsa_joint_qkv,
    insert_future_history_queries,
    isolate_wsa_future_from_all_action_queries,
)


class WSALingBotAttentionIntervention(str, Enum):
    """Measurement-only edits to the released WSA attention graph."""

    BLOCK_FUTURE_TO_ACTION = "block_future_to_action"


class WSALingBotActionCoupling(str, Enum):
    """Production information contract between Future3D and LingBot action."""

    DIRECT_FUTURE_KEYS = "direct_future_keys"
    AUXILIARY_WORLD_DECODER = "auxiliary_world_decoder"


@dataclass
class WSALingBotJointOutput:
    native_outputs: list[torch.Tensor | None]
    past_key_values: Any
    router_logits: list[torch.Tensor]
    future_runtime: WSAFutureRuntime
    future_projections: tuple[torch.Tensor, ...]
    layout: WSAJointTokenLayout


class WSALingBotTrainingRuntime(nn.Module):
    """Full WSA third-expert execution for LingBot's non-cached training path."""

    def __init__(
        self,
        future: WSAFutureExpertRuntime,
        *,
        action_coupling: WSALingBotActionCoupling = (
            WSALingBotActionCoupling.DIRECT_FUTURE_KEYS
        ),
    ):
        super().__init__()
        if not isinstance(action_coupling, WSALingBotActionCoupling):
            raise TypeError("WSA action coupling must be typed")
        self.future = future
        self.action_coupling = action_coupling
        self._latest_output: WSALingBotJointOutput | None = None
        self._attention_intervention_scope_open = False
        self._attention_intervention: WSALingBotAttentionIntervention | None = None

    @contextmanager
    def attention_intervention_scope(
        self,
        intervention: WSALingBotAttentionIntervention | None,
    ) -> Iterator[None]:
        """Apply a typed intervention for exactly one outer policy forward."""

        if intervention is not None and not isinstance(
            intervention,
            WSALingBotAttentionIntervention,
        ):
            raise TypeError("WSA attention intervention must be typed")
        if self._attention_intervention_scope_open:
            raise RuntimeError("WSA attention intervention scopes cannot be nested")
        self._attention_intervention_scope_open = True
        self._attention_intervention = intervention
        try:
            yield
        finally:
            self._attention_intervention = None
            self._attention_intervention_scope_open = False

    def take_latest_output(self) -> WSALingBotJointOutput:
        output = self._latest_output
        self._latest_output = None
        if output is None:
            raise RuntimeError("ADR218 LingBot joint runtime did not execute")
        return output

    def assert_output_consumed(self) -> None:
        """Reject a new policy transaction while an older WSA result is pending."""

        if self._latest_output is not None:
            raise RuntimeError("ADR218 LingBot joint runtime has an unconsumed output")

    def discard_latest_output(self) -> None:
        """Clear a partial policy transaction after its outer forward failed."""

        self._latest_output = None

    def _publish_output(self, output: WSALingBotJointOutput) -> None:
        if self._latest_output is not None:
            raise RuntimeError("ADR218 LingBot joint runtime executed more than once")
        self._latest_output = output

    def _validate_path(
        self,
        joint: nn.Module,
        *,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool | None,
        fill_kv_cache: bool | None,
    ) -> None:
        if len(inputs_embeds) != 2 or any(tensor is None for tensor in inputs_embeds):
            raise RuntimeError("ADR218 training requires simultaneous host and action streams")
        if use_cache:
            raise RuntimeError("ADR218 cached inference is not yet authorized")
        if joint.config.attention_implementation not in ("eager", "flex", "flex_cached"):
            raise RuntimeError(
                "ADR218 received an unsupported LingBot attention implementation: "
                f"{joint.config.attention_implementation!r}"
            )

    def forward(
        self,
        joint: nn.Module,
        *,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values: Any,
        inputs_embeds: list[torch.Tensor | None],
        use_cache: bool | None,
        fill_kv_cache: bool | None,
        ada_cond: torch.Tensor | None,
        visual_pos_masks: torch.Tensor | None,
        deepstack_visual_embeds: list[torch.Tensor] | None,
        picf_native_context: Any,
        picf_action_attention_callback: Any = None,
    ) -> WSALingBotJointOutput:
        self._validate_path(
            joint,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            fill_kv_cache=fill_kv_cache,
        )
        picf_runtime = None
        if joint.picf_native_graph is not None:
            (
                inputs_embeds,
                attention_mask,
                position_ids,
                visual_pos_masks,
                picf_runtime,
            ) = joint.picf_native_graph.prepare_joint_inputs(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                visual_pos_masks=visual_pos_masks,
                context=picf_native_context,
            )
        host_hidden, action_hidden = inputs_embeds
        if host_hidden is None or action_hidden is None:
            raise RuntimeError("ADR218 joint preparation removed a required stream")
        if position_ids.shape[-1] != host_hidden.shape[1] + action_hidden.shape[1]:
            raise RuntimeError("ADR218 position IDs differ from native host/action tokens")

        host_model = joint.qwenvl.model.language_model
        action_model = joint.qwen_expert.model
        layer_count = joint.qwenvl.config.text_config.num_hidden_layers
        if layer_count != 36 or len(action_model.layers) != layer_count:
            raise RuntimeError("ADR218 requires the released 36-layer LingBot topology")
        joint_mask, layout = build_wsa_joint_attention_mask_with_layout(
            attention_mask,
            host_count=host_hidden.shape[1],
            future_count=self.future.expert.num_query_tokens,
        )
        if (
            self.action_coupling
            is WSALingBotActionCoupling.AUXILIARY_WORLD_DECODER
        ):
            if self._attention_intervention is not None:
                raise RuntimeError(
                    "an auxiliary WSA world decoder cannot also run a direct-edge intervention"
                )
            joint_mask = isolate_wsa_future_from_all_action_queries(
                joint_mask,
                layout=layout,
            )
        if (
            self._attention_intervention
            is WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION
        ):
            joint_mask = block_wsa_future_to_action_information_edge(
                joint_mask,
                layout=layout,
            )
        host_position_ids = position_ids[:, :, : layout.host_count]
        action_position_ids = position_ids[:, :, layout.host_count :]
        future_runtime = self.future.prepare(
            batch_size=host_hidden.shape[0],
            device=host_hidden.device,
            dtype=action_hidden.dtype,
        )
        router_logits: list[torch.Tensor] = []
        cached_block_mask = None
        cached_attention_shape: tuple[int, int, int] | None = None

        for layer_index in range(layer_count):
            prefix_qk_bias = None
            if joint.picf_native_graph is not None:
                prefix_qk_bias = joint.picf_native_graph.layerwise_qk_address_bias(
                    prefix_hidden=host_hidden,
                    runtime=picf_runtime,
                )
            host_q, host_k, host_v = host_model.layers[layer_index](
                host_hidden,
                compute_kqv=True,
                qk_input_bias=prefix_qk_bias,
            )
            action_q, action_k, action_v = action_model.layers[layer_index](
                action_hidden,
                compute_kqv=True,
                ada_cond=ada_cond,
            )
            host_q, host_k = joint.apply_mrope(
                host_q.float(),
                host_k.float(),
                host_position_ids,
            )
            action_q, action_k = joint.apply_mrope(
                action_q.float(),
                action_k.float(),
                action_position_ids,
            )
            future_qkv = self.future.compute_layer_qkv(
                future_runtime,
                layer_index=layer_index,
            )
            query, key, value = concatenate_wsa_joint_qkv(
                host_qkv=(host_q, host_k, host_v.float()),
                future_qkv=future_qkv,
                action_qkv=(action_q, action_k, action_v.float()),
            )

            layer_mask = joint_mask
            memory_inputs = None
            if joint.picf_native_graph is not None:
                memory_inputs = joint.picf_native_graph.layerwise_memory_inputs(
                    layer_index=layer_index,
                    runtime=picf_runtime,
                )
            if memory_inputs is not None:
                memory_hidden, memory_qk_bias, history_visibility = memory_inputs
                memory_q, memory_k, memory_v = host_model.layers[layer_index](
                    memory_hidden,
                    compute_kqv=True,
                    qk_input_bias=memory_qk_bias,
                )
                memory_position_ids = position_ids.new_zeros(
                    (3, memory_hidden.shape[0], memory_hidden.shape[1])
                )
                _, memory_k = joint.apply_mrope(
                    memory_q.float(),
                    memory_k.float(),
                    memory_position_ids,
                )
                memory_k = repeat_lingbot_kv_heads(memory_k, target_heads=32)
                memory_v = repeat_lingbot_kv_heads(memory_v.float(), target_heads=32)
                history_visibility = insert_future_history_queries(
                    history_visibility,
                    layout=layout,
                )
                key = torch.cat((memory_k, key), dim=1)
                value = torch.cat((memory_v, value), dim=1)
                layer_mask = torch.cat((history_visibility, layer_mask), dim=-1)

            if picf_action_attention_callback is not None:
                if picf_native_context is None:
                    raise RuntimeError(
                        "WSA action-posterior receipt requires a typed PICF context"
                    )
                posterior_indices = picf_native_context.expanded_posterior_indices
                posterior_valid = picf_native_context.expanded_posterior_valid
                if posterior_indices is None or posterior_valid is None:
                    raise RuntimeError(
                        "WSA action-posterior receipt lost the prepared posterior layout"
                    )
                memory_count = int(key.shape[1] - layout.total_count)
                if memory_count < 0:
                    raise RuntimeError("WSA joint key axis is shorter than its token layout")
                receipt_layout = build_lingbot_joint_action_attention_layout(
                    batch_size=int(query.shape[0]),
                    query_count=int(query.shape[1]),
                    key_count=int(key.shape[1]),
                    action_query_slice=slice(layout.action.start + 1, layout.action.stop),
                    posterior_key_indices=posterior_indices + memory_count,
                    posterior_key_valid=posterior_valid,
                )
                picf_action_attention_callback(
                    query_states=query,
                    key_states=key,
                    attention_mask=layer_mask,
                    layout=receipt_layout,
                    layer_index=layer_index,
                    layer_count=layer_count,
                )
            if joint.config.attention_implementation == "flex_cached":
                from lingbotvla.models.vla.lingbot_vla.flex_attention import (
                    build_block_mask,
                    flex_attention_with_block_mask,
                )

                attention_shape = (
                    int(query.shape[1]),
                    int(key.shape[1]),
                    int(query.shape[2]),
                )
                if cached_block_mask is None:
                    cached_block_mask = build_block_mask(
                        layer_mask,
                        attention_shape[2],
                        attention_shape[0],
                        attention_shape[1],
                    )
                    cached_attention_shape = attention_shape
                elif cached_attention_shape != attention_shape:
                    raise RuntimeError(
                        "ADR218 flex-cached joint attention shape changed across layers"
                    )
                attention_output = flex_attention_with_block_mask(
                    query,
                    key,
                    value,
                    cached_block_mask,
                    attention_shape[0],
                )
            else:
                attention_output = joint.attention_interface(query, key, value, layer_mask)
            host_hidden = host_model.layers[layer_index](
                host_hidden,
                attention_output,
                layout.host.start,
                layout.host.stop,
                output_atten=True,
            )
            action_hidden, layer_router_logits = action_model.layers[layer_index](
                action_hidden,
                attention_output,
                layout.action.start,
                layout.action.stop,
                output_atten=True,
                ada_cond=ada_cond,
            )
            if layer_router_logits is not None:
                router_logits.append(layer_router_logits)
            self.future.apply_layer_attention(
                future_runtime,
                layer_index=layer_index,
                attention_output=attention_output[:, layout.future],
            )
            host_hidden = joint._apply_deepstack(
                host_hidden,
                layer_index,
                visual_pos_masks,
                deepstack_visual_embeds,
            )
            if joint.picf_native_graph is not None:
                joint.picf_native_graph.record_layerwise_posterior(
                    prefix_hidden=host_hidden,
                    runtime=picf_runtime,
                    layer_index=layer_index,
                )
                if joint.picf_native_graph.requires_intermediate_relation(
                    layer_index=layer_index,
                    runtime=picf_runtime,
                ):
                    joint.picf_native_graph.record_intermediate_relation(
                        normalized_prefix=host_model.norm(host_hidden),
                        runtime=picf_runtime,
                        layer_index=layer_index,
                    )

        host_hidden = host_model.norm(host_hidden)
        if joint.config.final_norm_adanorm:
            action_hidden, _ = action_model.norm(action_hidden, ada_cond)
        else:
            action_hidden = action_model.norm(action_hidden)
        native_outputs: list[torch.Tensor | None] = [host_hidden, action_hidden]
        if joint.picf_native_graph is not None:
            native_outputs = joint.picf_native_graph.finalize_joint_outputs(
                outputs_embeds=native_outputs,
                runtime=picf_runtime,
            )
        output = WSALingBotJointOutput(
            native_outputs=native_outputs,
            past_key_values=past_key_values,
            router_logits=router_logits,
            future_runtime=future_runtime,
            future_projections=self.future.project_targets(future_runtime),
            layout=layout,
        )
        self._publish_output(output)
        return output

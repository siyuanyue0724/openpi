"""Real LingBot readout for the diagnostic-only frozen-posterior ABI.

This module deliberately bypasses the production runtime.  It never calls the
released image prefix, factual correction, session state, or policy
``sample_actions`` path.  Instead it constructs a new whitelist-only KV cache
from deploy-visible language, executed controls, and a detached layerwise
posterior snapshot, then runs LingBot's released action expert with fixed
inference noise.

No trainable module is defined here.  Semantic processing remains in the
released LingBot language layers and action expert; PICF contributes only its
trained control projection and stable object-address Q/K bias.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass

import torch
from torch import nn

from picf_next.lingbot_native.frozen_posterior_diagnostic import (
    FrozenPosteriorActionRequest,
    FrozenPosteriorVisibility,
    audit_frozen_posterior_visibility,
)
from picf_next.lingbot_native.host import LingBotNativeGraph


@dataclass(frozen=True, slots=True)
class FrozenPosteriorActionInformationContract:
    """Concrete source visibility for one isolated LingBot action read."""

    visibility: FrozenPosteriorVisibility
    language_reads_posterior: bool
    action_cache_contains_posterior: bool
    allowed_action_sources: tuple[str, ...]
    forbidden_action_sources: tuple[str, ...]


def frozen_posterior_action_information_contract(
    visibility: FrozenPosteriorVisibility,
) -> FrozenPosteriorActionInformationContract:
    """Return the exact whitelist used by the real diagnostic adapter."""

    audit = audit_frozen_posterior_visibility(visibility)
    allowed = ["language", "executed-control", "proprioception"]
    if audit.direct_posterior_path or audit.language_mediated_posterior_path:
        allowed.append("frozen-posterior")
    return FrozenPosteriorActionInformationContract(
        visibility=visibility,
        language_reads_posterior=audit.language_mediated_posterior_path,
        action_cache_contains_posterior=audit.direct_posterior_path,
        allowed_action_sources=tuple(allowed),
        forbidden_action_sources=(
            "current-rgb",
            "current-dense-modality",
            "history",
            "prior",
            "external-trace",
            "host-aux",
            "match",
            "ground-truth-action",
            "target-row",
            "sidecar-label",
        ),
    )


@dataclass(frozen=True, slots=True)
class _IsolatedPrefixCache:
    past_key_values: dict[int, dict[str, torch.Tensor]]
    valid: torch.Tensor
    position_ids: torch.Tensor


def _make_attention_mask(pad_mask: torch.Tensor, block_start: torch.Tensor) -> torch.Tensor:
    """Exact local copy of LingBot's released ``make_att_2d_masks`` primitive."""

    if pad_mask.ndim != 2 or block_start.shape != pad_mask.shape:
        raise ValueError("attention mask inputs must share shape [batch,tokens]")
    if pad_mask.dtype != torch.bool or block_start.dtype != torch.bool:
        raise TypeError("attention mask inputs must be boolean")
    cumulative = torch.cumsum(block_start, dim=1)
    causal = cumulative[:, None, :] <= cumulative[:, :, None]
    valid = pad_mask[:, None, :] & pad_mask[:, :, None]
    return causal & valid


def _is_dtensor(value: object) -> bool:
    value_type = type(value)
    return value_type.__name__ == "DTensor" and value_type.__module__.startswith(
        "torch.distributed"
    )


class LingBotFrozenPosteriorActionAdapter:
    """Run a real, label-free LingBot action read from one frozen posterior.

    The adapter supports an unsharded evaluation model whose official PICF
    graph is already installed.  It fails closed for training mode, DTensor
    parameters, incomplete patched LingBot APIs, or any shape/device/dtype
    mismatch.  It does not mutate the model or register modules.
    """

    def __init__(
        self,
        policy: nn.Module,
        *,
        _registered_fsdp_root: bool = False,
    ) -> None:
        if not isinstance(policy, nn.Module):
            raise TypeError("frozen-posterior LingBot adapter requires an nn.Module policy")
        flow = getattr(policy, "model", None)
        if flow is None or not isinstance(flow, nn.Module):
            raise TypeError("policy does not expose the released LingBot flow model")
        joint = getattr(flow, "qwenvl_with_expert", None)
        if joint is None or not isinstance(joint, nn.Module):
            raise TypeError("policy does not expose the released LingBot joint host")
        graph = getattr(joint, "picf_native_graph", None)
        if not isinstance(graph, LingBotNativeGraph):
            raise TypeError("released LingBot joint host does not contain a PICF native graph")

        self._policy = policy
        self._flow = flow
        self._joint = joint
        self._graph = graph
        self._registered_fsdp_root = _registered_fsdp_root
        self._validate_static_topology()

    def __call__(self, request: FrozenPosteriorActionRequest, /) -> torch.Tensor:
        if not isinstance(request, FrozenPosteriorActionRequest):
            raise TypeError("adapter requires a FrozenPosteriorActionRequest")
        self._validate_eval_mode()
        self._validate_request(request)
        with torch.inference_mode():
            cache = self._build_isolated_prefix_cache(request)
            return self._denoise_action(request, cache)

    def _validate_static_topology(self) -> None:
        graph = self._graph
        flow = self._flow
        joint = self._joint
        if not graph.layerwise_recurrence:
            raise ValueError("frozen-posterior action readout requires layerwise PICF recurrence")
        if graph.object_addresses is None:
            raise RuntimeError("layerwise PICF object addresses are unavailable")
        if _is_dtensor(graph.object_addresses) and not self._registered_fsdp_root:
            raise RuntimeError(
                "diagnostic adapter cannot safely read sharded DTensor object addresses; "
                "run it on an unsharded evaluation model"
            )
        if (
            any(_is_dtensor(parameter) for parameter in self._policy.parameters())
            and not self._registered_fsdp_root
        ):
            raise RuntimeError(
                "diagnostic adapter cannot safely bypass an FSDP/DTensor root forward; "
                "run it on an unsharded evaluation model"
            )
        if self._registered_fsdp_root:
            try:
                from torch.distributed.fsdp import FSDPModule
            except ImportError as error:  # pragma: no cover - pinned runtime owns this path.
                raise RuntimeError("registered frozen-posterior read requires FSDP2") from error
            if not isinstance(self._policy, FSDPModule):
                raise RuntimeError(
                    "registered frozen-posterior read must execute through the FSDP2 policy root"
                )
        if any(parameter.is_meta for parameter in self._policy.parameters()):
            raise RuntimeError("diagnostic adapter requires materialized LingBot weights")
        try:
            language_model = joint.qwenvl.model.language_model
            action_model = joint.qwen_expert.model
            language_layers = language_model.layers
            action_layers = action_model.layers
        except AttributeError as error:
            raise TypeError(
                "released LingBot joint host is missing language/action layers"
            ) from error
        if len(language_layers) != graph.config.num_layers or len(action_layers) != len(
            language_layers
        ):
            raise ValueError("LingBot, action expert, and PICF graph depths differ")
        required_joint = (
            "embed_language_tokens",
            "build_prefix_position_ids",
            "apply_mrope",
            "forward",
        )
        required_flow = ("embed_suffix", "_build_full_position_ids", "action_out_proj")
        if any(not hasattr(joint, name) for name in required_joint):
            raise TypeError("released LingBot joint host lacks the patched diagnostic primitives")
        if any(not hasattr(flow, name) for name in required_flow):
            raise TypeError("released LingBot flow model lacks the action readout primitives")
        forward_parameters = inspect.signature(joint.forward).parameters.values()
        if not any(
            parameter.name == "picf_native_context"
            or parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in forward_parameters
        ):
            raise TypeError("released LingBot joint forward lacks the PICF context patch")
        if bool(getattr(flow.config, "action_fp32", False)) and not callable(
            getattr(flow, "_fp32_linear", None)
        ):
            raise TypeError("LingBot action_fp32 mode lacks its released FP32 linear primitive")
        attention_implementation = getattr(joint.config, "attention_implementation", None)
        if attention_implementation not in {"eager", "flex", "flex_cached"}:
            raise ValueError("unsupported LingBot attention implementation for isolated readout")
        self._validate_eval_mode()

    def _validate_eval_mode(self) -> None:
        modules = (self._policy, self._flow, self._joint, self._graph)
        if any(module.training for module in modules):
            raise RuntimeError("frozen-posterior action diagnostic requires eval mode")

    def _validate_request(self, request: FrozenPosteriorActionRequest) -> None:
        graph = self._graph
        posterior = request.posterior
        addresses = graph.object_addresses
        if addresses is None:
            raise RuntimeError("layerwise PICF object addresses are unavailable")
        expected = (
            graph.config.num_layers,
            graph.config.capacity,
            graph.config.host_width,
        )
        if posterior.layer_rows.shape[1:] != expected:
            raise ValueError("frozen posterior shape differs from the installed PICF graph")
        if posterior.layer_rows.device != addresses.device:
            raise ValueError("frozen posterior and PICF graph must share one device")
        if posterior.layer_rows.dtype != addresses.dtype:
            raise TypeError("frozen posterior and PICF graph must share one dtype")
        if request.controls.action_dim != graph.config.executed_action_dim:
            raise ValueError("executed controls differ from the installed action width")
        request.controls.validate_bound(graph.config.maximum_control_tokens)
        if request.controls.values.dtype != addresses.dtype:
            raise TypeError("executed controls and PICF graph must share one dtype")
        if request.proprioception.ndim != 2:
            raise ValueError("LingBot proprioception must have shape [batch,state_dim]")
        if request.proprioception.dtype != request.inference_noise.dtype:
            raise TypeError("proprioception and fixed inference noise must share one dtype")
        if request.proprioception.dtype != addresses.dtype:
            raise TypeError("proprioception and LingBot host must share one dtype")
        config = self._flow.config
        expected_noise = (
            posterior.batch_size,
            int(config.n_action_steps),
            int(config.max_action_dim),
        )
        if request.inference_noise.shape != expected_noise:
            raise ValueError("fixed inference noise differs from the released action surface")
        if not isinstance(config.num_steps, int) or config.num_steps <= 0:
            raise ValueError("LingBot flow num_steps must be a positive integer")

    def _build_isolated_prefix_cache(
        self,
        request: FrozenPosteriorActionRequest,
    ) -> _IsolatedPrefixCache:
        contract = frozen_posterior_action_information_contract(request.visibility)
        joint = self._joint
        graph = self._graph
        addresses = graph.object_addresses
        if addresses is None:
            raise RuntimeError("layerwise PICF object addresses are unavailable")

        language = joint.embed_language_tokens(request.language.token_ids)
        if language.ndim != 3 or language.shape[-1] != graph.config.host_width:
            raise RuntimeError("LingBot language embedding differs from the PICF host width")
        if language.device != addresses.device:
            raise RuntimeError("LingBot language embedding and PICF graph use different devices")
        language = language.to(dtype=addresses.dtype)
        control_features = request.controls.canonical_features().to(dtype=addresses.dtype)
        controls = graph.control_projection(control_features) + graph.role_embeddings[2]
        prefix_hidden = torch.cat((language, controls), dim=1)
        prefix_valid = torch.cat(
            (request.language.token_valid, request.controls.token_valid),
            dim=1,
        )
        language_count = language.shape[1]
        control_count = controls.shape[1]
        prefix_positions = self._prefix_position_ids(
            request=request,
            language_count=language_count,
            control_count=control_count,
        )
        prefix_attention = self._language_control_attention_mask(
            prefix_valid=prefix_valid,
            language_count=language_count,
        )
        posterior_positions = prefix_positions.new_zeros(
            (3, request.posterior.batch_size, request.posterior.capacity)
        )
        address_bias = addresses.unsqueeze(0).expand(request.posterior.batch_size, -1, -1)
        cache: dict[int, dict[str, torch.Tensor]] = {}
        language_layers = joint.qwenvl.model.language_model.layers

        for layer_index, layer in enumerate(language_layers):
            query, key, value = self._layer_qkv(layer, prefix_hidden, qk_input_bias=None)
            query, key = joint.apply_mrope(query.float(), key.float(), prefix_positions)
            value = value.float()

            posterior_hidden = request.posterior.layer(layer_index)
            posterior_query, posterior_key, posterior_value = self._layer_qkv(
                layer,
                posterior_hidden,
                qk_input_bias=address_bias,
            )
            _, posterior_key = joint.apply_mrope(
                posterior_query.float(),
                posterior_key.float(),
                posterior_positions,
            )
            posterior_value = posterior_value.float()

            layer_key = key
            layer_value = value
            layer_mask = prefix_attention
            if contract.language_reads_posterior:
                memory_visibility = torch.zeros(
                    (
                        request.posterior.batch_size,
                        language_count + control_count,
                        request.posterior.capacity,
                    ),
                    dtype=torch.bool,
                    device=prefix_hidden.device,
                )
                memory_visibility[:, :language_count] = request.posterior_row_visible[:, None]
                layer_key = torch.cat((posterior_key, key), dim=1)
                layer_value = torch.cat((posterior_value, value), dim=1)
                layer_mask = torch.cat((memory_visibility, prefix_attention), dim=-1)

            attention_output = self._attention(
                query=query,
                key=layer_key,
                value=layer_value,
                mask=layer_mask,
            )
            prefix_hidden = self._layer_output(layer, prefix_hidden, attention_output)

            cache_key = key
            cache_value = value
            if contract.action_cache_contains_posterior:
                cache_key = torch.cat((key, posterior_key), dim=1)
                cache_value = torch.cat((value, posterior_value), dim=1)
            cache[layer_index] = {
                "key_states": cache_key,
                "value_states": cache_value,
            }

        cache_valid = prefix_valid
        cache_positions = prefix_positions
        if contract.action_cache_contains_posterior:
            cache_valid = torch.cat((prefix_valid, request.posterior_row_visible), dim=1)
            cache_positions = torch.cat((prefix_positions, posterior_positions), dim=-1)
        self._validate_cache(cache, valid=cache_valid, position_ids=cache_positions)
        return _IsolatedPrefixCache(
            past_key_values=cache,
            valid=cache_valid,
            position_ids=cache_positions,
        )

    def _prefix_position_ids(
        self,
        *,
        request: FrozenPosteriorActionRequest,
        language_count: int,
        control_count: int,
    ) -> torch.Tensor:
        positions = self._joint.build_prefix_position_ids(
            request.language.token_ids,
            request.language.token_valid.long(),
            image_grid_thw=None,
            video_grid_thw=None,
        )
        expected = (3, request.posterior.batch_size, language_count)
        if positions.shape != expected or positions.dtype not in {torch.int32, torch.int64}:
            raise RuntimeError("LingBot returned invalid text-only MRoPE position IDs")
        if positions.device != request.language.token_ids.device:
            raise RuntimeError("LingBot text-only MRoPE positions changed device")
        control_positions = positions.new_zeros((3, request.posterior.batch_size, control_count))
        return torch.cat((positions, control_positions), dim=-1)

    def _language_control_attention_mask(
        self,
        *,
        prefix_valid: torch.Tensor,
        language_count: int,
    ) -> torch.Tensor:
        batch, token_count = prefix_valid.shape
        mask = torch.zeros(
            (batch, token_count, token_count),
            dtype=torch.bool,
            device=prefix_valid.device,
        )
        language_valid = prefix_valid[:, :language_count]
        if bool(getattr(self._flow.config, "vlm_causal", False)):
            causal = torch.ones(
                (language_count, language_count),
                dtype=torch.bool,
                device=prefix_valid.device,
            ).tril()
            mask[:, :language_count, :language_count] = (
                causal[None] & language_valid[:, :, None] & language_valid[:, None, :]
            )
        else:
            mask[:, :language_count, :language_count] = (
                language_valid[:, :, None] & language_valid[:, None, :]
            )
        control_valid = prefix_valid[:, language_count:]
        mask[:, language_count:, language_count:] = (
            control_valid[:, :, None] & control_valid[:, None, :]
        )
        return mask

    @staticmethod
    def _layer_qkv(
        layer: nn.Module,
        hidden: torch.Tensor,
        *,
        qk_input_bias: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        try:
            if qk_input_bias is None:
                output = layer(hidden, compute_kqv=True)
            else:
                output = layer(
                    hidden,
                    compute_kqv=True,
                    qk_input_bias=qk_input_bias,
                )
        except TypeError as error:
            raise RuntimeError(
                "installed LingBot layer lacks the PICF Q/K address-bias patch"
            ) from error
        if not isinstance(output, tuple) or len(output) != 3:
            raise RuntimeError("LingBot layer did not return a Q/K/V triple")
        query, key, value = output
        if any(not isinstance(item, torch.Tensor) or item.ndim != 4 for item in output):
            raise RuntimeError("LingBot layer returned invalid Q/K/V tensors")
        return query, key, value

    @staticmethod
    def _layer_output(
        layer: nn.Module,
        hidden: torch.Tensor,
        attention_output: torch.Tensor,
    ) -> torch.Tensor:
        output = layer(
            hidden,
            attention_output,
            0,
            hidden.shape[1],
            output_atten=True,
        )
        if isinstance(output, tuple):
            output = output[0]
        if not isinstance(output, torch.Tensor) or output.shape != hidden.shape:
            raise RuntimeError("LingBot language layer returned an invalid hidden state")
        return output

    def _attention(
        self,
        *,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        expected = (query.shape[0], query.shape[1], key.shape[1])
        if mask.shape != expected or mask.dtype != torch.bool:
            raise RuntimeError("isolated prefix mask differs from the LingBot attention surface")
        implementation = self._joint.config.attention_implementation
        if implementation == "flex_cached":
            module = importlib.import_module(type(self._joint).__module__)
            build_block_mask = getattr(module, "build_block_mask", None)
            flex_attention = getattr(module, "flex_attention_with_block_mask", None)
            if not callable(build_block_mask) or not callable(flex_attention):
                raise RuntimeError("LingBot flex-cached attention helpers are unavailable")
            heads = int(self._joint.qwenvl.config.text_config.num_attention_heads)
            block_mask = build_block_mask(mask, heads, query.shape[1], key.shape[1])
            return flex_attention(query, key, value, block_mask, query.shape[1])
        return self._joint.attention_interface(query, key, value, mask)

    def _validate_cache(
        self,
        cache: dict[int, dict[str, torch.Tensor]],
        *,
        valid: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> None:
        if set(cache) != set(range(self._graph.config.num_layers)):
            raise RuntimeError("isolated action cache does not cover every LingBot layer")
        if valid.ndim != 2 or valid.dtype != torch.bool:
            raise RuntimeError("isolated action cache validity must be boolean [batch,tokens]")
        if position_ids.shape != (3, valid.shape[0], valid.shape[1]):
            raise RuntimeError("isolated action cache positions differ from cache validity")
        if position_ids.device != valid.device:
            raise RuntimeError("isolated action cache positions changed device")
        for layer_index, values in cache.items():
            if set(values) != {"key_states", "value_states"}:
                raise RuntimeError(f"isolated layer {layer_index} cache has an unknown schema")
            key = values["key_states"]
            value = values["value_states"]
            if key.ndim != 4 or value.shape != key.shape:
                raise RuntimeError(f"isolated layer {layer_index} cache has invalid K/V shape")
            if key.shape[:2] != valid.shape:
                raise RuntimeError(f"isolated layer {layer_index} cache length is inconsistent")
            if key.device != valid.device or value.device != valid.device:
                raise RuntimeError(f"isolated layer {layer_index} cache changed device")
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise RuntimeError(f"isolated layer {layer_index} cache is non-finite")

    def _denoise_action(
        self,
        request: FrozenPosteriorActionRequest,
        cache: _IsolatedPrefixCache,
    ) -> torch.Tensor:
        flow = self._flow
        x_t = request.inference_noise.detach().clone()
        dtype = x_t.dtype
        device = x_t.device
        batch = x_t.shape[0]
        delta = torch.tensor(-1.0 / flow.config.num_steps, dtype=dtype, device=device)
        time = torch.tensor(1.0, dtype=dtype, device=device)
        for _ in range(flow.config.num_steps + 1):
            if bool((time < -delta / 2).item()):
                break
            velocity = self._predict_velocity(
                proprioception=request.proprioception,
                x_t=x_t,
                timestep=time.expand(batch),
                cache=cache,
            )
            x_t = x_t + delta * velocity
            time = time + delta
        else:
            raise RuntimeError("LingBot denoising loop exceeded its declared step count")
        return x_t

    def _predict_velocity(
        self,
        *,
        proprioception: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        cache: _IsolatedPrefixCache,
    ) -> torch.Tensor:
        flow = self._flow
        time_embeddings, suffix, suffix_valid, suffix_blocks = flow.embed_suffix(
            proprioception,
            x_t,
            timestep,
        )
        if suffix_valid.dtype != torch.bool or suffix_blocks.dtype != torch.bool:
            raise RuntimeError("LingBot suffix masks must be boolean")
        suffix_attention = _make_attention_mask(suffix_valid, suffix_blocks)
        prefix_attention = cache.valid[:, None, :].expand(
            cache.valid.shape[0],
            suffix.shape[1],
            cache.valid.shape[1],
        )
        attention = torch.cat((prefix_attention, suffix_attention), dim=-1)
        full_positions = flow._build_full_position_ids(
            cache.position_ids,
            cache.valid,
            suffix_valid,
        )
        suffix_positions = full_positions[:, :, -suffix.shape[1] :]
        outputs, _, _ = self._joint.forward(
            attention_mask=attention,
            position_ids=suffix_positions,
            past_key_values=cache.past_key_values,
            inputs_embeds=[None, suffix],
            use_cache=True,
            fill_kv_cache=False,
            ada_cond=(
                time_embeddings if bool(getattr(flow.config, "adanorm_time", False)) else None
            ),
            picf_native_context=None,
        )
        if not isinstance(outputs, list) or len(outputs) != 2 or outputs[1] is None:
            raise RuntimeError("LingBot action expert did not return its suffix stream")
        suffix_output = outputs[1][:, -flow.config.n_action_steps :]
        if bool(getattr(flow.config, "action_fp32", False)):
            velocity = flow._fp32_linear(flow.action_out_proj, suffix_output)
        else:
            weight = flow.action_out_proj.weight
            if suffix_output.dtype != weight.dtype:
                suffix_output = suffix_output.to(weight.dtype)
            velocity = flow.action_out_proj(suffix_output)
        if velocity.shape != x_t.shape or not torch.isfinite(velocity).all():
            raise RuntimeError("LingBot action expert returned an invalid velocity field")
        return velocity.to(dtype=x_t.dtype)


def run_registered_lingbot_frozen_posterior_action(
    policy: nn.Module,
    request: FrozenPosteriorActionRequest,
) -> torch.Tensor:
    """Execute the isolated action read inside a registered FSDP2 root method.

    FSDP2 parameters remain DTensors outside a registered root call.  The
    released LingBot policy method calls this helper only after its custom
    forward pre-hook has materialized the root and its nested sharded layers.
    No observation tensor, target label, sidecar, or trainable adapter enters
    this path.
    """

    return LingBotFrozenPosteriorActionAdapter(
        policy,
        _registered_fsdp_root=True,
    )(request)


@torch.no_grad()
def run_native_frozen_posterior_action_forward(
    policy: nn.Module,
    *,
    request: FrozenPosteriorActionRequest,
) -> torch.Tensor:
    """Call the audited action-only method through the policy root.

    This is the only public FSDP-capable entrypoint.  The request schema has no
    image, dense-modality, target, row-identity, sidecar, or action-label field.
    """

    if not isinstance(policy, nn.Module) or policy.training:
        raise ValueError("frozen-posterior root forward requires a policy in eval mode")
    if not isinstance(request, FrozenPosteriorActionRequest):
        raise TypeError("frozen-posterior root forward requires its typed request")
    root_forward = getattr(policy, "picf_native_frozen_posterior_action_forward", None)
    if not callable(root_forward):
        raise TypeError("LingBot policy lacks the registered frozen-posterior root method")
    action = root_forward(request=request)
    if (
        not isinstance(action, torch.Tensor)
        or action.shape != request.inference_noise.shape
        or action.device != request.inference_noise.device
        or not action.is_floating_point()
        or not torch.isfinite(action).all()
        or action.requires_grad
    ):
        raise RuntimeError("frozen-posterior root method returned an invalid action tensor")
    return action

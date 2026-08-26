"""Read-only deep PICF context for the pinned LingBot-VLA 2.0 action expert.

The insertion point follows
``Robbyant/lingbot-vla-v2@69729b4ef24c63ec25e750915491635f4753be1d``
(Apache-2.0), specifically the released 6B checkpoint path
``QwenvlWithExpertV2Model.forward`` and ``Qwen2DecoderLayer.forward``. The
official model jointly advances VLM-prefix and action-suffix streams at every
layer. PICF adds two independent read-only residual cross-attention branches to
the action stream after the official layer output and before the next layer.

PICF memory is deliberately not appended to the official joint attention:
doing so would change its softmax denominator even with zero-valued context and
would violate exact vanilla parity. This adapter owns no LingBot parameters and
can be called from a small upstream hook without changing posterior semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from picf_next.hosts.context import PICFActionEvidence
from picf_next.models.evidence import NativeTokenBank


@dataclass(frozen=True, slots=True)
class LingBotPICFContext:
    """Projected read-only memory reused by all action layers/flow steps."""

    dense_key: torch.Tensor | None
    dense_value: torch.Tensor | None
    dense_valid: torch.Tensor | None
    object_key: torch.Tensor | None
    object_value: torch.Tensor | None
    object_valid: torch.Tensor | None
    object_log_prior: torch.Tensor | None


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        dtype = value.dtype
        normalized = value.float()
        normalized = normalized * torch.rsqrt(normalized.square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


def _initialize_linear(linear: nn.Linear, *, scale: float = 1.0) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if scale != 1.0:
        with torch.no_grad():
            linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class _ReadOnlyCrossAttention(nn.Module):
    """GQA-compatible unordered memory read with an exact-zero residual gate."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        residual_scale: float,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.query_norm = _RMSNorm(hidden_size)
        self.query_projection = nn.Linear(hidden_size, num_attention_heads * head_dim, bias=False)
        self.output_projection = nn.Linear(num_attention_heads * head_dim, hidden_size, bias=False)
        self.gate = nn.Parameter(torch.zeros(()))
        _initialize_linear(self.query_projection)
        _initialize_linear(self.output_projection, scale=residual_scale)

    def forward(
        self,
        hidden_states: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        valid: torch.Tensor,
        log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor:
        active = valid.any(dim=1)
        if key.shape[1] == 0:
            return torch.zeros_like(hidden_states)
        safe_valid = valid.clone()
        safe_valid[:, 0] |= ~active
        batch, query_count = hidden_states.shape[:2]
        query = self.query_projection(self.query_norm(hidden_states)).view(
            batch, query_count, self.num_attention_heads, self.head_dim
        )
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        groups = self.num_attention_heads // self.num_key_value_heads
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
        attention_mask: torch.Tensor
        if log_prior is None:
            attention_mask = safe_valid[:, None, None, :]
        else:
            if (
                log_prior.shape != valid.shape
                or log_prior.device != valid.device
                or log_prior.dtype != hidden_states.dtype
            ):
                raise ValueError("attention log prior must align with object validity")
            attention_mask = torch.where(
                safe_valid,
                log_prior,
                torch.full_like(log_prior, torch.finfo(log_prior.dtype).min),
            )[:, None, None, :]
        branch = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        branch = branch.transpose(1, 2).reshape(batch, query_count, -1)
        branch = self.output_projection(branch)
        return self.gate * branch * active[:, None, None]


class LingBotVLA2PICFAdapter(nn.Module):
    """Project one PICF evidence contract into every LingBot action layer."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        dense_token_dims: Mapping[str, int],
        object_address_dim: int,
        object_value_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        positive = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "object_address_dim": object_address_dim,
            "object_value_dim": object_value_dim,
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in positive.values()
        ):
            raise ValueError("LingBot adapter dimensions must be positive")
        if num_attention_heads % num_key_value_heads:
            raise ValueError("attention heads must be divisible by key/value heads")
        if not dense_token_dims:
            raise ValueError("at least one dense modality must be configured")
        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")
        normalized_dims = dict(dense_token_dims)
        if any(
            not name
            or "." in name
            or not isinstance(width, int)
            or isinstance(width, bool)
            or width <= 0
            for name, width in normalized_dims.items()
        ):
            raise ValueError("dense modality names and widths must be valid")

        factory_kwargs = {"device": device, "dtype": dtype}
        kv_width = num_key_value_heads * head_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.validate_tensor_values = validate_tensor_values
        self.dense_key_projection = nn.ModuleDict(
            {
                name: nn.Linear(width, kv_width, bias=False, **factory_kwargs)
                for name, width in normalized_dims.items()
            }
        )
        self.dense_value_projection = nn.ModuleDict(
            {
                name: nn.Linear(width, kv_width, bias=False, **factory_kwargs)
                for name, width in normalized_dims.items()
            }
        )
        self.object_key_projection = nn.Linear(
            object_address_dim, kv_width, bias=False, **factory_kwargs
        )
        self.object_value_projection = nn.Linear(
            object_value_dim, kv_width, bias=False, **factory_kwargs
        )
        self.dense_owner_value_projection = nn.Linear(
            object_address_dim, kv_width, bias=False, **factory_kwargs
        )
        self.dense_key_norm = _RMSNorm(kv_width).to(**factory_kwargs)
        self.dense_value_norm = _RMSNorm(kv_width).to(**factory_kwargs)
        self.object_key_norm = _RMSNorm(kv_width).to(**factory_kwargs)
        self.object_value_norm = _RMSNorm(kv_width).to(**factory_kwargs)
        residual_scale = (2 * num_layers) ** -0.5
        branch_kwargs = {
            "hidden_size": hidden_size,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "residual_scale": residual_scale,
        }
        self.dense_branches = nn.ModuleList(
            [
                _ReadOnlyCrossAttention(**branch_kwargs).to(**factory_kwargs)
                for _ in range(num_layers)
            ]
        )
        self.object_branches = nn.ModuleList(
            [
                _ReadOnlyCrossAttention(**branch_kwargs).to(**factory_kwargs)
                for _ in range(num_layers)
            ]
        )
        for projection in (
            *self.dense_key_projection.values(),
            *self.dense_value_projection.values(),
            self.object_key_projection,
            self.object_value_projection,
            self.dense_owner_value_projection,
        ):
            _initialize_linear(projection)

    @property
    def dense_gates(self) -> torch.Tensor:
        return torch.stack([branch.gate for branch in self.dense_branches])

    @property
    def object_gates(self) -> torch.Tensor:
        return torch.stack([branch.gate for branch in self.object_branches])

    def _validate_bank(
        self,
        name: str,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        valid: torch.Tensor | None,
        *,
        key_width: int,
        value_width: int,
    ) -> None:
        if key is None or value is None or valid is None:
            if not (key is None and value is None and valid is None):
                raise ValueError(f"{name} key, value and validity must be all present or absent")
            return
        parameter = self.object_key_projection.weight
        if key.ndim != 3 or value.ndim != 3 or key.shape[:2] != value.shape[:2]:
            raise ValueError(f"{name} key and value must align batch-by-token")
        if key.shape[-1] != key_width or value.shape[-1] != value_width:
            raise ValueError(f"{name} feature width differs from adapter configuration")
        if valid.dtype != torch.bool or valid.shape != key.shape[:2]:
            raise ValueError(f"{name} validity must be bool batch-by-token")
        for tensor in (key, value, valid):
            if tensor.device != parameter.device:
                raise ValueError(f"{name} tensors must share the adapter device")
        if key.dtype != parameter.dtype or value.dtype != parameter.dtype:
            raise ValueError(f"{name} key and value must share the adapter dtype")
        if self.validate_tensor_values:
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinity")
            if (key[~valid] != 0.0).any() or (value[~valid] != 0.0).any():
                raise ValueError(f"{name} invalid padding must be exactly zero")

    def _project_dense(
        self,
        banks: tuple[NativeTokenBank, ...],
        owner_addresses: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if not banks:
            return None, None, None
        seen: set[str] = set()
        key_parts = []
        value_parts = []
        valid_parts = []
        batch_size = None
        if len(owner_addresses) != len(banks):
            raise ValueError("dense owner-address metadata must align with dense banks")
        for bank, owner_address in zip(banks, owner_addresses, strict=True):
            if bank.modality in seen:
                raise ValueError(f"dense modality {bank.modality} appears more than once")
            seen.add(bank.modality)
            if bank.modality not in self.dense_key_projection:
                raise ValueError(f"dense modality {bank.modality} is not configured")
            width = self.dense_key_projection[bank.modality].in_features
            self._validate_bank(
                f"dense/{bank.modality}",
                bank.tokens,
                bank.tokens,
                bank.valid,
                key_width=width,
                value_width=width,
            )
            if batch_size is None:
                batch_size = bank.tokens.shape[0]
            elif bank.tokens.shape[0] != batch_size:
                raise ValueError("all dense banks must share a batch size")
            if bank.tokens.shape[1] == 0:
                continue
            projected_key = self.dense_key_projection[bank.modality](bank.tokens)
            projected_value = self.dense_value_projection[bank.modality](bank.tokens)
            if owner_address is not None:
                projected_key = projected_key + self.object_key_projection(owner_address)
                projected_value = projected_value + self.dense_owner_value_projection(owner_address)
            key_parts.append(projected_key)
            value_parts.append(projected_value)
            valid_parts.append(bank.valid)
        if not key_parts:
            return None, None, None
        key = self.dense_key_norm(torch.cat(key_parts, dim=1))
        value = self.dense_value_norm(torch.cat(value_parts, dim=1))
        valid = torch.cat(valid_parts, dim=1)
        shape = (*key.shape[:2], self.num_key_value_heads, self.head_dim)
        return key.view(shape), value.view(shape), valid

    def prepare_picf_context(self, evidence: PICFActionEvidence) -> LingBotPICFContext:
        evidence.batch_size()
        owner_addresses = evidence.ownership_weighted_addresses(
            validate_tensor_values=self.validate_tensor_values
        )
        dense_key, dense_value, dense_valid = self._project_dense(
            evidence.dense_banks, owner_addresses
        )
        self._validate_bank(
            "object",
            evidence.object_address,
            evidence.object_value,
            evidence.object_valid,
            key_width=self.object_key_projection.in_features,
            value_width=self.object_value_projection.in_features,
        )
        object_key = None
        object_value = None
        if evidence.object_address is not None and evidence.object_address.shape[1]:
            projected_key = self.object_key_norm(
                self.object_key_projection(evidence.object_address)
            )
            projected_value = self.object_value_norm(
                self.object_value_projection(evidence.object_value)
            )
            shape = (*projected_key.shape[:2], self.num_key_value_heads, self.head_dim)
            object_key = projected_key.view(shape)
            object_value = projected_value.view(shape)
        return LingBotPICFContext(
            dense_key=dense_key,
            dense_value=dense_value,
            dense_valid=dense_valid,
            object_key=object_key,
            object_value=object_value,
            object_valid=evidence.object_valid,
            object_log_prior=evidence.object_log_prior,
        )

    def apply_layer(
        self,
        action_hidden_states: torch.Tensor,
        *,
        layer_index: int,
        context: LingBotPICFContext | None,
    ) -> torch.Tensor:
        """Apply the hook after LingBot's official action-layer output."""

        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError("layer_index is outside the LingBot adapter")
        if action_hidden_states.ndim != 3 or action_hidden_states.shape[-1] != self.hidden_size:
            raise ValueError("action hidden state shape differs from the LingBot adapter")
        parameter = self.object_key_projection.weight
        if (
            action_hidden_states.device != parameter.device
            or action_hidden_states.dtype != parameter.dtype
        ):
            raise ValueError("action hidden states must share adapter dtype and device")
        if context is None:
            return action_hidden_states
        context_tensors = (
            context.dense_key,
            context.dense_value,
            context.dense_valid,
            context.object_key,
            context.object_value,
            context.object_valid,
            context.object_log_prior,
        )
        if any(
            tensor is not None and tensor.shape[0] != action_hidden_states.shape[0]
            for tensor in context_tensors
        ):
            raise ValueError("LingBot action and PICF context batch sizes must match")
        output = action_hidden_states
        if context.dense_key is not None:
            output = output + self.dense_branches[layer_index](
                output, context.dense_key, context.dense_value, context.dense_valid
            )
        if context.object_key is not None:
            output = output + self.object_branches[layer_index](
                output,
                context.object_key,
                context.object_value,
                context.object_valid,
                context.object_log_prior,
            )
        return output

    def forward(
        self,
        action_hidden_states: torch.Tensor,
        *,
        layer_index: int,
        context: LingBotPICFContext | None,
    ) -> torch.Tensor:
        """Match the pinned LingBot V2 action-layer adapter protocol."""

        return self.apply_layer(
            action_hidden_states,
            layer_index=layer_index,
            context=context,
        )


def install_lingbot_vla2_picf_adapter(
    policy: nn.Module,
    adapter: LingBotVLA2PICFAdapter,
) -> None:
    """Install PICF before optimizer/FSDP construction on a patched V2 policy."""

    flow_model = getattr(policy, "model", None)
    host = getattr(flow_model, "qwenvl_with_expert", None)
    setter = getattr(host, "set_action_layer_adapter", None)
    if host is None or not callable(setter):
        raise TypeError(
            "policy is not the pinned patched LingBot V2 host; "
            "apply references/patches/lingbot_vla2_action_layer_adapter.patch"
        )

    config = getattr(host, "config", None)
    expert_config = getattr(config, "qwen_expert_config", None)
    expected = {
        "hidden_size": getattr(expert_config, "hidden_size", None),
        "num_layers": getattr(expert_config, "num_hidden_layers", None),
        "num_attention_heads": getattr(expert_config, "num_attention_heads", None),
        "num_key_value_heads": getattr(expert_config, "num_key_value_heads", None),
        "head_dim": getattr(expert_config, "head_dim", None),
    }
    actual = {
        "hidden_size": adapter.hidden_size,
        "num_layers": adapter.num_layers,
        "num_attention_heads": adapter.num_attention_heads,
        "num_key_value_heads": adapter.num_key_value_heads,
        "head_dim": adapter.head_dim,
    }
    mismatch = {
        name: (expected[name], actual[name]) for name in expected if expected[name] != actual[name]
    }
    if mismatch:
        raise ValueError(f"LingBot V2 host/adapter dimensions differ: {mismatch}")

    installed = getattr(host, "action_layer_adapter", None)
    if installed is not None and installed is not adapter:
        raise RuntimeError("LingBot V2 already has a different action-layer adapter")
    setter(adapter)
    if getattr(host, "action_layer_adapter", None) is not adapter:
        raise RuntimeError("LingBot V2 did not register the PICF adapter")

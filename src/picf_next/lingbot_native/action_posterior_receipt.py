"""Exact loss-side receipt for native action attention to posterior rows.

This module has no parameter and no deploy-time branch. It replays the released
LingBot eager attention equation from the actual post-MRoPE action queries,
compact-cache keys, and executed boolean mask exposed by the host integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

ACTION_POSTERIOR_LAYOUT_SCHEMA = "picf-next.lingbot-action-posterior-layout.v1"


def _concrete_slice(name: str, value: slice, *, length: int) -> tuple[int, int]:
    if value.step not in (None, 1) or value.start is None or value.stop is None:
        raise ValueError(f"{name} must be one contiguous concrete slice")
    if value.start < 0 or value.stop > length or value.stop <= value.start:
        raise ValueError(f"{name} is outside its declared token axis")
    return value.start, value.stop


@dataclass(frozen=True, slots=True)
class LingBotActionAttentionLayout:
    """Typed mapping from the released suffix/cache axes to posterior rows."""

    batch_size: int
    query_count: int
    key_count: int
    native_prefix_count: int
    compact_prefix_count: int
    state_query_slice: slice
    action_query_slice: slice
    posterior_key_indices: torch.Tensor
    posterior_key_valid: torch.Tensor
    expanded_posterior_indices: torch.Tensor
    selected_inserted_indices: torch.Tensor
    schema: str = ACTION_POSTERIOR_LAYOUT_SCHEMA

    def __post_init__(self) -> None:
        integers = (
            self.batch_size,
            self.query_count,
            self.key_count,
            self.native_prefix_count,
            self.compact_prefix_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("action-posterior layout counts must be integers")
        if self.batch_size <= 0 or self.query_count <= 1 or self.key_count <= 0:
            raise ValueError("action-posterior layout counts must be positive")
        if not 0 <= self.native_prefix_count <= self.compact_prefix_count <= self.key_count:
            raise ValueError("native/compact prefix counts are outside the key axis")
        state_start, state_stop = _concrete_slice(
            "state_query_slice",
            self.state_query_slice,
            length=self.query_count,
        )
        action_start, _action_stop = _concrete_slice(
            "action_query_slice",
            self.action_query_slice,
            length=self.query_count,
        )
        if state_stop - state_start != 1 or state_start != 0 or action_start != state_stop:
            raise ValueError("released LingBot suffix must be state token 0 followed by action")
        indices = self.posterior_key_indices
        valid = self.posterior_key_valid
        expanded_posterior = self.expanded_posterior_indices
        selected = self.selected_inserted_indices
        if indices.ndim != 1 or indices.dtype != torch.long or not indices.numel():
            raise ValueError("posterior key indices must be a non-empty long vector")
        if valid.shape != (self.batch_size, indices.numel()) or valid.dtype != torch.bool:
            raise ValueError("posterior key validity must be boolean [batch,capacity]")
        if expanded_posterior.shape != indices.shape or expanded_posterior.dtype != torch.long:
            raise ValueError("expanded posterior indices must match the posterior capacity")
        if selected.ndim != 1 or selected.dtype != torch.long:
            raise ValueError("selected inserted indices must be a long vector")
        if (
            indices.device != valid.device
            or expanded_posterior.device != indices.device
            or selected.device != indices.device
        ):
            raise ValueError("action-posterior layout tensors must share one device")
        if (indices < 0).any() or (indices >= self.compact_prefix_count).any():
            raise ValueError("posterior key index is outside the compact key axis")
        if torch.unique(indices).numel() != indices.numel() or not torch.equal(
            indices,
            indices.sort().values,
        ):
            raise ValueError("posterior key indices must be unique and ordered")
        if (indices < self.native_prefix_count).any():
            raise ValueError("posterior rows cannot alias the released native prefix")
        if selected.numel() != self.compact_prefix_count - self.native_prefix_count:
            raise ValueError("selected inserted rows do not span the compact prefix")
        if torch.unique(selected).numel() != selected.numel() or not torch.equal(
            selected,
            selected.sort().values,
        ):
            raise ValueError("selected inserted indices must be unique and ordered")
        if torch.unique(expanded_posterior).numel() != expanded_posterior.numel():
            raise ValueError("expanded posterior indices must be unique")
        matches = selected[:, None] == expanded_posterior[None, :]
        if not matches.any(dim=0).all() or (matches.sum(dim=0) != 1).any():
            raise ValueError("every posterior row must be selected exactly once")
        expected_compact = self.native_prefix_count + matches.float().argmax(dim=0)
        if not torch.equal(indices, expected_compact):
            raise ValueError("posterior compact indices do not match expanded-cache selection")


@dataclass(frozen=True, slots=True)
class LingBotJointActionAttentionLayout:
    """Typed action/posterior indices on a simultaneous joint-attention axis.

    Unlike :class:`LingBotActionAttentionLayout`, this layout does not describe
    a compact prefix cache. It indexes the actual synchronous
    ``[history, host, Future3D, action]`` Q/K surface without moving or
    renormalizing any model token.
    """

    batch_size: int
    query_count: int
    key_count: int
    action_query_slice: slice
    posterior_key_indices: torch.Tensor
    posterior_key_valid: torch.Tensor

    def __post_init__(self) -> None:
        integers = (self.batch_size, self.query_count, self.key_count)
        if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
            raise TypeError("joint action-posterior layout counts must be integers")
        if self.batch_size <= 0 or self.query_count <= 0 or self.key_count <= 0:
            raise ValueError("joint action-posterior layout counts must be positive")
        _concrete_slice(
            "action_query_slice",
            self.action_query_slice,
            length=self.query_count,
        )
        indices = self.posterior_key_indices
        valid = self.posterior_key_valid
        if indices.ndim != 1 or indices.dtype != torch.long or not indices.numel():
            raise ValueError("joint posterior key indices must be a non-empty long vector")
        if valid.shape != (self.batch_size, indices.numel()) or valid.dtype != torch.bool:
            raise ValueError("joint posterior key validity must be boolean [batch,capacity]")
        if indices.device != valid.device:
            raise ValueError("joint posterior key indices and validity must share one device")
        if (indices < 0).any() or (indices >= self.key_count).any():
            raise ValueError("joint posterior key index is outside the key axis")
        if torch.unique(indices).numel() != indices.numel() or not torch.equal(
            indices,
            indices.sort().values,
        ):
            raise ValueError("joint posterior key indices must be unique and ordered")


ActionPosteriorAttentionLayout = (
    LingBotActionAttentionLayout | LingBotJointActionAttentionLayout
)


def build_lingbot_action_attention_layout(
    *,
    batch_size: int,
    native_prefix_count: int,
    suffix_count: int,
    action_query_count: int | None = None,
    selected_inserted_indices: torch.Tensor,
    expanded_posterior_indices: torch.Tensor,
    posterior_key_valid: torch.Tensor,
) -> LingBotActionAttentionLayout:
    """Build the exact expanded-prefix to compact-action key mapping."""

    if isinstance(suffix_count, bool) or not isinstance(suffix_count, int) or suffix_count <= 1:
        raise ValueError("released LingBot suffix requires one state and at least one action token")
    resolved_action_query_count = (
        suffix_count if action_query_count is None else action_query_count
    )
    if (
        isinstance(resolved_action_query_count, bool)
        or not isinstance(resolved_action_query_count, int)
        or not 1 < resolved_action_query_count <= suffix_count
    ):
        raise ValueError("action query span must follow state and fit inside the full suffix")
    selected = selected_inserted_indices
    posterior = expanded_posterior_indices
    if selected.ndim != 1 or selected.dtype != torch.long:
        raise ValueError("selected inserted indices must be a long vector")
    if posterior.ndim != 1 or posterior.dtype != torch.long or not posterior.numel():
        raise ValueError("expanded posterior indices must be a non-empty long vector")
    if selected.device != posterior.device or posterior_key_valid.device != posterior.device:
        raise ValueError("cache layout inputs must share one device")
    matches = selected[:, None] == posterior[None, :]
    if not matches.any(dim=0).all() or (matches.sum(dim=0) != 1).any():
        raise ValueError("every expanded posterior row must occur once in the compact selection")
    compact_prefix_count = native_prefix_count + selected.numel()
    posterior_key_indices = native_prefix_count + matches.float().argmax(dim=0)
    return LingBotActionAttentionLayout(
        batch_size=batch_size,
        query_count=suffix_count,
        key_count=compact_prefix_count + suffix_count,
        native_prefix_count=native_prefix_count,
        compact_prefix_count=compact_prefix_count,
        state_query_slice=slice(0, 1),
        action_query_slice=slice(1, resolved_action_query_count),
        posterior_key_indices=posterior_key_indices,
        posterior_key_valid=posterior_key_valid,
        expanded_posterior_indices=posterior,
        selected_inserted_indices=selected,
    )


def build_lingbot_joint_action_attention_layout(
    *,
    batch_size: int,
    query_count: int,
    key_count: int,
    action_query_slice: slice,
    posterior_key_indices: torch.Tensor,
    posterior_key_valid: torch.Tensor,
) -> LingBotJointActionAttentionLayout:
    """Bind posterior rows to their exact positions on a joint WSA surface."""

    return LingBotJointActionAttentionLayout(
        batch_size=batch_size,
        query_count=query_count,
        key_count=key_count,
        action_query_slice=action_query_slice,
        posterior_key_indices=posterior_key_indices,
        posterior_key_valid=posterior_key_valid,
    )


@dataclass(frozen=True, slots=True)
class ActionPosteriorAttentionReceipt:
    """Executed action attention restricted to typed posterior key positions."""

    layer_index: int
    layer_count: int
    posterior_attention: torch.Tensor
    total_posterior_mass: torch.Tensor

    def __post_init__(self) -> None:
        if (
            isinstance(self.layer_index, bool)
            or isinstance(self.layer_count, bool)
            or not isinstance(self.layer_index, int)
            or not isinstance(self.layer_count, int)
            or self.layer_count <= 0
            or not 0 <= self.layer_index < self.layer_count
        ):
            raise ValueError("action-posterior receipt has an invalid explicit layer identity")
        attention = self.posterior_attention
        total = self.total_posterior_mass
        if attention.ndim != 4 or total.shape != attention.shape[:3]:
            raise ValueError("action-posterior receipt must be [batch,head,action,row]")
        if not attention.is_floating_point() or not total.is_floating_point():
            raise TypeError("action-posterior receipt tensors must be floating point")
        if not torch.isfinite(attention).all() or not torch.isfinite(total).all():
            raise ValueError("action-posterior receipt contains NaN or infinity")
        if (attention < 0).any() or (total < 0).any() or (total > 1.000001).any():
            raise ValueError("action-posterior receipt contains invalid probability mass")


def action_posterior_attention_receipt(
    *,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    attention_mask: torch.Tensor,
    layout: ActionPosteriorAttentionLayout,
    layer_index: int,
    layer_count: int,
) -> ActionPosteriorAttentionReceipt:
    """Replay the released eager attention equation on action/posterior slices."""

    if query_states.ndim != 4 or key_states.ndim != 4:
        raise ValueError("action-posterior Q/K tensors must be [batch,token,head,dim]")
    batch, query_count, query_heads, head_dim = query_states.shape
    if (
        batch != layout.batch_size
        or query_count != layout.query_count
        or key_states.shape[0] != batch
        or key_states.shape[1] != layout.key_count
        or key_states.shape[3] != head_dim
        or attention_mask.shape != (batch, query_count, layout.key_count)
        or attention_mask.dtype != torch.bool
    ):
        raise ValueError("action-posterior Q/K/mask surface differs from its layout")
    if (
        query_states.device != key_states.device
        or attention_mask.device != query_states.device
        or layout.posterior_key_indices.device != query_states.device
    ):
        raise ValueError("action-posterior Q/K/mask/layout must share one device")
    key_heads = key_states.shape[2]
    if key_heads <= 0 or query_heads % key_heads:
        raise ValueError("action query heads must evenly group compact-cache key heads")

    action_queries = query_states[:, layout.action_query_slice].float()
    action_mask = attention_mask[:, layout.action_query_slice]
    if not action_mask.any(dim=-1).all():
        raise ValueError("every action query must expose at least one executed key")
    invalid_posterior = ~layout.posterior_key_valid
    posterior_visibility = action_mask.index_select(-1, layout.posterior_key_indices)
    if (posterior_visibility & invalid_posterior[:, None, :]).any():
        raise ValueError("an invalid posterior row is visible to action")

    repeated_keys = key_states.float().repeat_interleave(query_heads // key_heads, dim=2)
    logits = torch.einsum("bahd,bkhd->bhak", action_queries, repeated_keys)
    logits.mul_(head_dim**-0.5)
    logits = torch.where(
        action_mask[:, None],
        logits,
        torch.tensor(-2.3819763e38, dtype=logits.dtype, device=logits.device),
    )
    probabilities = torch.softmax(logits, dim=-1)
    posterior = probabilities.index_select(-1, layout.posterior_key_indices)
    total = posterior.sum(dim=-1)
    return ActionPosteriorAttentionReceipt(
        layer_index=layer_index,
        layer_count=layer_count,
        posterior_attention=posterior,
        total_posterior_mass=total,
    )

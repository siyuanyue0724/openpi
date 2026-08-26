from __future__ import annotations

from dataclasses import dataclass

import torch

from picf_next.lingbot_native.wsa_full_depth_adaptation import repeat_lingbot_kv_heads


@dataclass(frozen=True)
class WSAJointTokenLayout:
    host_count: int
    future_count: int
    action_count: int

    def __post_init__(self) -> None:
        if self.host_count <= 0 or self.future_count <= 0 or self.action_count <= 0:
            raise ValueError("host, future, and action streams must all be non-empty")

    @property
    def host(self) -> slice:
        return slice(0, self.host_count)

    @property
    def future(self) -> slice:
        return slice(self.host_count, self.host_count + self.future_count)

    @property
    def action(self) -> slice:
        return slice(self.future.stop, self.future.stop + self.action_count)

    @property
    def total_count(self) -> int:
        return self.host_count + self.future_count + self.action_count


def build_wsa_joint_attention_mask_with_layout(
    lingbot_mask: torch.Tensor,
    *,
    host_count: int,
    future_count: int,
) -> tuple[torch.Tensor, WSAJointTokenLayout]:
    if lingbot_mask.ndim != 3 or lingbot_mask.shape[1] != lingbot_mask.shape[2]:
        raise ValueError("LingBot mask must have shape [B, tokens, tokens]")
    if lingbot_mask.dtype is not torch.bool:
        raise TypeError("LingBot mask must be boolean")
    native_count = lingbot_mask.shape[1]
    if host_count <= 0 or host_count >= native_count:
        raise ValueError("host_count must split non-empty LingBot host and action streams")
    layout = WSAJointTokenLayout(
        host_count=host_count,
        future_count=future_count,
        action_count=native_count - host_count,
    )
    batch = lingbot_mask.shape[0]
    result = torch.zeros(
        (batch, layout.total_count, layout.total_count),
        dtype=torch.bool,
        device=lingbot_mask.device,
    )
    native_host = slice(0, host_count)
    native_action = slice(host_count, native_count)
    result[:, layout.host, layout.host] = lingbot_mask[:, native_host, native_host]
    result[:, layout.host, layout.action] = lingbot_mask[:, native_host, native_action]
    result[:, layout.action, layout.host] = lingbot_mask[:, native_action, native_host]
    result[:, layout.action, layout.action] = lingbot_mask[:, native_action, native_action]

    native_valid = lingbot_mask.diagonal(dim1=-2, dim2=-1)
    host_valid = native_valid[:, native_host]
    action_valid = native_valid[:, native_action]
    future_valid = torch.ones(
        (batch, future_count),
        dtype=torch.bool,
        device=lingbot_mask.device,
    )
    result[:, layout.future, layout.host] = future_valid.unsqueeze(-1) & host_valid.unsqueeze(1)
    result[:, layout.future, layout.future] = (
        future_valid.unsqueeze(-1) & future_valid.unsqueeze(1)
    )
    result[:, layout.future, layout.action] = (
        future_valid.unsqueeze(-1) & action_valid.unsqueeze(1)
    )
    result[:, layout.action, layout.future] = (
        action_valid.unsqueeze(-1) & future_valid.unsqueeze(1)
    )
    return result, layout


def block_wsa_future_to_action_information_edge(
    joint_mask: torch.Tensor,
    *,
    layout: WSAJointTokenLayout,
) -> torch.Tensor:
    """Block only action-output queries from reading Future3D keys.

    Attention masks are indexed as ``[query, key]``.  The first native action
    token is LingBot's state token, so the intervention excludes it and changes
    only the rows whose final hidden states parameterize the action objective.
    """

    if joint_mask.ndim != 3 or joint_mask.shape[1:] != (
        layout.total_count,
        layout.total_count,
    ):
        raise ValueError("WSA joint mask does not match its token layout")
    if joint_mask.dtype is not torch.bool:
        raise TypeError("WSA joint mask must be boolean")
    if layout.action_count <= 1:
        raise ValueError("WSA intervention requires an action output after the state token")
    intervened = joint_mask.clone()
    action_outputs = slice(layout.action.start + 1, layout.action.stop)
    intervened[:, action_outputs, layout.future] = False
    return intervened


def isolate_wsa_future_from_all_action_queries(
    joint_mask: torch.Tensor,
    *,
    layout: WSAJointTokenLayout,
) -> torch.Tensor:
    """Keep Future3D as an auxiliary world decoder, never an action key.

    Unlike the measurement-only intervention above, this production invariant
    includes LingBot's state query.  Blocking the complete action slice is
    required to prevent the state token from relaying Future3D information to
    action-output queries in a later shared layer.
    """

    if joint_mask.ndim != 3 or joint_mask.shape[1:] != (
        layout.total_count,
        layout.total_count,
    ):
        raise ValueError("WSA joint mask does not match its token layout")
    if joint_mask.dtype is not torch.bool:
        raise TypeError("WSA joint mask must be boolean")
    isolated = joint_mask.clone()
    isolated[:, layout.action, layout.future] = False
    return isolated


def insert_future_history_queries(
    history_visibility: torch.Tensor,
    *,
    layout: WSAJointTokenLayout,
) -> torch.Tensor:
    """Keep candidate future queries from reading deploy-persistent external memory."""
    if history_visibility.ndim != 3:
        raise ValueError("History visibility must have shape [B, native queries, memory]")
    if history_visibility.shape[1] != layout.host_count + layout.action_count:
        raise ValueError("History visibility differs from the native LingBot query count")
    future_hidden = torch.zeros(
        (history_visibility.shape[0], layout.future_count, history_visibility.shape[2]),
        dtype=torch.bool,
        device=history_visibility.device,
    )
    return torch.cat(
        (
            history_visibility[:, : layout.host_count],
            future_hidden,
            history_visibility[:, layout.host_count :],
        ),
        dim=1,
    )


def concatenate_wsa_joint_qkv(
    *,
    host_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    future_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    action_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construct the 32-head mixed surface without learned bridge parameters."""
    host_q, host_k, host_v = host_qkv
    future_q, future_k, future_v = future_qkv
    action_q, action_k, action_v = action_qkv
    for name, tensor in (
        ("host query", host_q),
        ("future query", future_q),
        ("action query", action_q),
    ):
        if tensor.ndim != 4 or tensor.shape[2:] != (32, 128):
            raise ValueError(f"{name} must have shape [B,S,32,128]")
    host_k = repeat_lingbot_kv_heads(host_k, target_heads=32)
    host_v = repeat_lingbot_kv_heads(host_v, target_heads=32)
    action_k = repeat_lingbot_kv_heads(action_k, target_heads=32)
    action_v = repeat_lingbot_kv_heads(action_v, target_heads=32)
    for name, tensor in (
        ("future key", future_k),
        ("future value", future_v),
        ("expanded host key", host_k),
        ("expanded host value", host_v),
        ("expanded action key", action_k),
        ("expanded action value", action_v),
    ):
        if tensor.ndim != 4 or tensor.shape[2:] != (32, 128):
            raise ValueError(f"{name} must have shape [B,S,32,128]")
    return (
        torch.cat((host_q, future_q, action_q), dim=1),
        torch.cat((host_k, future_k, action_k), dim=1),
        torch.cat((host_v, future_v, action_v), dim=1),
    )


def split_wsa_joint_attention(
    attention_output: torch.Tensor,
    *,
    layout: WSAJointTokenLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if attention_output.ndim != 3 or attention_output.shape[1:] != (
        layout.total_count,
        4096,
    ):
        raise ValueError("Mixed attention output differs from the registered joint layout")
    return (
        attention_output[:, layout.host],
        attention_output[:, layout.future],
        attention_output[:, layout.action],
    )

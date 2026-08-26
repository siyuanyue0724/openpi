"""One block-causal information graph for physical belief and action."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import IntEnum

import torch


class TokenRole(IntEnum):
    """Runtime role, not object identity or semantic class."""

    NATIVE_BASELINE = 0
    HISTORY = 1
    SENSOR = 2
    CURRENT_STATE = 3
    PRIOR = 4
    POSTERIOR = 5
    LANGUAGE = 6
    RETRIEVAL = 7
    ACTION = 8
    MEASUREMENT_QUERY = 9
    HOST_FUTURE_QUERY = 10
    PREDICT_QUERY = 11
    CONTEXT = 12


_ROLE_COUNT = len(TokenRole)


def role_permission_matrix(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Return ``allowed[query_role, key_role]`` for ADR 65's factorization."""

    allowed = torch.zeros((_ROLE_COUNT, _ROLE_COUNT), dtype=torch.bool, device=device)

    def permit(query: TokenRole, *keys: TokenRole) -> None:
        allowed[int(query), torch.tensor([int(key) for key in keys], device=allowed.device)] = True

    permit(TokenRole.NATIVE_BASELINE, TokenRole.NATIVE_BASELINE)
    permit(TokenRole.HISTORY, TokenRole.HISTORY)
    permit(TokenRole.PRIOR, TokenRole.HISTORY, TokenRole.PRIOR)
    permit(TokenRole.SENSOR, TokenRole.SENSOR)
    permit(TokenRole.CURRENT_STATE, TokenRole.SENSOR, TokenRole.CURRENT_STATE)
    permit(
        TokenRole.POSTERIOR,
        TokenRole.PRIOR,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.POSTERIOR,
    )
    permit(
        TokenRole.LANGUAGE,
        TokenRole.PRIOR,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.POSTERIOR,
        TokenRole.LANGUAGE,
    )
    permit(
        TokenRole.RETRIEVAL,
        TokenRole.PRIOR,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.POSTERIOR,
        TokenRole.LANGUAGE,
        TokenRole.RETRIEVAL,
    )
    # A released-host measurement query (for example, LingBot's current-depth
    # query) has no target token as input and remains available at deployment.
    # Keep it on the native observation side of the graph.
    permit(
        TokenRole.MEASUREMENT_QUERY,
        TokenRole.SENSOR,
        TokenRole.LANGUAGE,
        TokenRole.MEASUREMENT_QUERY,
    )
    # Released-host future queries are auxiliary prediction placeholders. They
    # preserve the host's native observation context, but action cannot read
    # them and their supervision is never inserted as a key.
    permit(
        TokenRole.HOST_FUTURE_QUERY,
        TokenRole.SENSOR,
        TokenRole.LANGUAGE,
        TokenRole.MEASUREMENT_QUERY,
        TokenRole.HOST_FUTURE_QUERY,
    )
    permit(
        TokenRole.ACTION,
        TokenRole.PRIOR,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.POSTERIOR,
        TokenRole.LANGUAGE,
        TokenRole.RETRIEVAL,
        TokenRole.MEASUREMENT_QUERY,
        TokenRole.ACTION,
    )
    # Graph-owned prediction queries are loss-side probes of belief
    # sufficiency. They may use only source-time physical state and causal
    # transition metadata, unlike the released host's native future queries.
    permit(
        TokenRole.PREDICT_QUERY,
        TokenRole.HISTORY,
        TokenRole.CURRENT_STATE,
        TokenRole.PRIOR,
        TokenRole.POSTERIOR,
        TokenRole.PREDICT_QUERY,
    )
    # The assignment null/background query is a transient sink. It may form a
    # current-step competitor from physical evidence, but no persistent or
    # action role may read its hidden state as an unassigned-evidence shortcut.
    permit(
        TokenRole.CONTEXT,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.PRIOR,
        TokenRole.CONTEXT,
    )
    return allowed


def role_graph_contract_digest() -> str:
    """Content-address role identities and every legal information-flow edge."""

    payload = {
        "roles": [(role.name, int(role)) for role in TokenRole],
        "permissions": role_permission_matrix().to(torch.uint8).tolist(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TokenLayout:
    roles: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.roles.ndim != 2 or self.valid.shape != self.roles.shape:
            raise ValueError("roles and valid must have shape [batch, tokens]")
        if self.roles.dtype != torch.long:
            raise TypeError("roles must use torch.long")
        if self.valid.dtype != torch.bool:
            raise TypeError("valid must be boolean")
        if self.roles.device != self.valid.device:
            raise ValueError("roles and valid must be on the same device")
        if ((self.roles < 0) | (self.roles >= _ROLE_COUNT)).any():
            raise ValueError("roles contain an unknown token role")

    @property
    def batch_size(self) -> int:
        return self.roles.shape[0]

    @property
    def token_count(self) -> int:
        return self.roles.shape[1]

    def validate_unified(self) -> None:
        """Reject ambiguous baseline roles in a non-baseline graph."""

        native = self.roles == int(TokenRole.NATIVE_BASELINE)
        if native.any() and not native.all():
            raise ValueError("NATIVE_BASELINE cannot be mixed with typed unified roles")


def block_causal_attention_mask(
    layout: TokenLayout,
    *,
    base_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a boolean visibility mask and optionally intersect a host mask."""

    layout.validate_unified()
    if (layout.roles == int(TokenRole.NATIVE_BASELINE)).all() and base_mask is not None:
        if (
            base_mask.shape
            != (
                layout.batch_size,
                layout.token_count,
                layout.token_count,
            )
            or base_mask.dtype != torch.bool
        ):
            raise ValueError("base_mask must be boolean with shape [batch, tokens, tokens]")
        return base_mask
    permissions = role_permission_matrix(device=layout.roles.device)
    queries = layout.roles.unsqueeze(-1).expand(-1, -1, layout.token_count)
    keys = layout.roles.unsqueeze(1).expand(-1, layout.token_count, -1)
    mask = permissions[queries, keys]
    mask = mask & layout.valid.unsqueeze(-1) & layout.valid.unsqueeze(1)
    if base_mask is not None:
        if base_mask.shape != mask.shape or base_mask.dtype != torch.bool:
            raise ValueError("base_mask must be boolean with shape [batch, tokens, tokens]")
        mask = mask & base_mask
    return mask


def expand_host_mask_for_inserted_tokens(
    base_mask: torch.Tensor,
    *,
    insertion_index: int,
    inserted_count: int,
) -> torch.Tensor:
    """Insert unconstrained rows/columns while preserving every old host edge."""

    if base_mask.ndim != 3 or base_mask.shape[-1] != base_mask.shape[-2]:
        raise ValueError("base_mask must have shape [batch, tokens, tokens]")
    if base_mask.dtype != torch.bool:
        raise TypeError("base_mask must be boolean")
    old_count = base_mask.shape[-1]
    if not 0 <= insertion_index <= old_count or inserted_count < 0:
        raise ValueError("invalid insertion_index or inserted_count")
    if inserted_count == 0:
        return base_mask
    new_count = old_count + inserted_count
    expanded = torch.ones(
        (base_mask.shape[0], new_count, new_count),
        dtype=torch.bool,
        device=base_mask.device,
    )
    old_indices = torch.cat(
        (
            torch.arange(insertion_index, device=base_mask.device),
            torch.arange(insertion_index + inserted_count, new_count, device=base_mask.device),
        )
    )
    expanded[:, old_indices[:, None], old_indices[None, :]] = base_mask
    return expanded


def insert_layout_block(
    base: TokenLayout,
    inserted: TokenLayout,
    *,
    insertion_index: int,
) -> TokenLayout:
    if base.batch_size != inserted.batch_size:
        raise ValueError("base and inserted layouts must have equal batch size")
    if not 0 <= insertion_index <= base.token_count:
        raise ValueError("insertion_index is outside the base layout")
    roles = torch.cat(
        (
            base.roles[:, :insertion_index],
            inserted.roles,
            base.roles[:, insertion_index:],
        ),
        dim=1,
    )
    valid = torch.cat(
        (
            base.valid[:, :insertion_index],
            inserted.valid,
            base.valid[:, insertion_index:],
        ),
        dim=1,
    )
    return TokenLayout(roles=roles, valid=valid)

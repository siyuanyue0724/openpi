"""Layerwise causal information graph for strict ADR-74 PICF."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

import torch


class NativeRole(IntEnum):
    SENSOR = 0
    LANGUAGE = 1
    HOST_AUX = 2
    CONTROL = 3
    PRIOR = 4
    POSTERIOR = 5
    ACTION = 6
    PREDICT = 7
    MATCH = 8


_ROLE_COUNT = len(NativeRole)


def native_role_permissions(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Return ``allowed[query, key]`` for one LingBot layer."""

    allowed = torch.zeros((_ROLE_COUNT, _ROLE_COUNT), dtype=torch.bool, device=device)

    def permit(query: NativeRole, *keys: NativeRole) -> None:
        indices = torch.tensor([int(key) for key in keys], device=allowed.device)
        allowed[int(query), indices] = True

    permit(NativeRole.SENSOR, NativeRole.SENSOR)
    permit(
        NativeRole.LANGUAGE,
        NativeRole.SENSOR,
        NativeRole.LANGUAGE,
        NativeRole.PRIOR,
        NativeRole.POSTERIOR,
    )
    permit(NativeRole.HOST_AUX, NativeRole.SENSOR, NativeRole.LANGUAGE, NativeRole.HOST_AUX)
    permit(NativeRole.CONTROL, NativeRole.CONTROL)
    permit(NativeRole.PRIOR, NativeRole.CONTROL, NativeRole.PRIOR)
    permit(
        NativeRole.POSTERIOR,
        NativeRole.CONTROL,
        NativeRole.PRIOR,
        NativeRole.SENSOR,
        NativeRole.POSTERIOR,
    )
    permit(
        NativeRole.ACTION,
        NativeRole.SENSOR,
        NativeRole.LANGUAGE,
        NativeRole.HOST_AUX,
        NativeRole.CONTROL,
        NativeRole.PRIOR,
        NativeRole.POSTERIOR,
        NativeRole.MATCH,
        NativeRole.ACTION,
    )
    permit(NativeRole.PREDICT, NativeRole.PREDICT)
    permit(
        NativeRole.MATCH,
        NativeRole.SENSOR,
        NativeRole.LANGUAGE,
        NativeRole.POSTERIOR,
        NativeRole.MATCH,
    )
    return allowed


def transitive_information_paths() -> torch.Tensor:
    """Return repeated-layer reachability as ``reachable[source, sink]``."""

    direct = native_role_permissions().T.clone()
    # Host preparation narrows each prediction query to one paired source row.
    # Reachability records the union of those legal pairwise edges.
    direct[int(NativeRole.PRIOR), int(NativeRole.PREDICT)] = True
    direct[int(NativeRole.POSTERIOR), int(NativeRole.PREDICT)] = True
    closure = direct.clone()
    for intermediate in range(_ROLE_COUNT):
        closure = closure | (
            closure[:, intermediate].unsqueeze(1) & closure[intermediate].unsqueeze(0)
        )
    return closure


def validate_native_causality() -> None:
    closure = transitive_information_paths()
    forbidden_prior = (
        NativeRole.SENSOR,
        NativeRole.LANGUAGE,
        NativeRole.HOST_AUX,
        NativeRole.POSTERIOR,
        NativeRole.ACTION,
        NativeRole.PREDICT,
        NativeRole.MATCH,
    )
    forbidden_posterior = (
        NativeRole.LANGUAGE,
        NativeRole.HOST_AUX,
        NativeRole.ACTION,
        NativeRole.PREDICT,
        NativeRole.MATCH,
    )
    if any(closure[int(source), int(NativeRole.PRIOR)] for source in forbidden_prior):
        raise RuntimeError("a forbidden repeated-layer path reaches prior rows")
    if any(closure[int(source), int(NativeRole.POSTERIOR)] for source in forbidden_posterior):
        raise RuntimeError("a forbidden repeated-layer path reaches posterior rows")


@dataclass(frozen=True, slots=True)
class NativeTokenLayout:
    roles: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.roles.ndim != 2 or self.valid.shape != self.roles.shape:
            raise ValueError("native roles and validity must have shape [batch, tokens]")
        if self.roles.dtype != torch.long or self.valid.dtype != torch.bool:
            raise TypeError("native roles must be long and validity must be boolean")
        if self.roles.device != self.valid.device:
            raise ValueError("native roles and validity must share one device")
        if ((self.roles < 0) | (self.roles >= _ROLE_COUNT)).any():
            raise ValueError("native layout contains an unknown role")

    @property
    def batch_size(self) -> int:
        return self.roles.shape[0]

    @property
    def token_count(self) -> int:
        return self.roles.shape[1]


def native_attention_mask(
    layout: NativeTokenLayout,
    *,
    host_mask: torch.Tensor,
    control_slice: slice | None = None,
) -> torch.Tensor:
    if (
        host_mask.shape
        != (
            layout.batch_size,
            layout.token_count,
            layout.token_count,
        )
        or host_mask.dtype != torch.bool
    ):
        raise ValueError("host mask must be boolean with shape [batch, tokens, tokens]")
    permissions = native_role_permissions(device=layout.roles.device)
    query_roles = layout.roles.unsqueeze(-1).expand(-1, -1, layout.token_count)
    key_roles = layout.roles.unsqueeze(1).expand(-1, layout.token_count, -1)
    result = permissions[query_roles, key_roles]
    result &= host_mask
    result &= layout.valid.unsqueeze(-1) & layout.valid.unsqueeze(1)
    if control_slice is not None:
        count = control_slice.stop - control_slice.start
        causal = torch.ones((count, count), dtype=torch.bool, device=result.device).tril()
        result[:, control_slice, control_slice] &= causal
    return result


def posterior_adoption_action_key_visibility(
    layout: NativeTokenLayout,
    *,
    enabled: torch.Tensor,
    direct_action_visible: torch.Tensor,
) -> torch.Tensor:
    """Restrict current-scene keys while preserving typed direct state evidence.

    This is an experimental information-route intervention, not a second
    semantic model. Scene evidence must reach action through posterior rows;
    explicitly declared direct state rows such as proprioception remain visible.
    """

    if (
        enabled.shape != (layout.batch_size,)
        or enabled.dtype != torch.bool
        or enabled.device != layout.roles.device
    ):
        raise ValueError("posterior-adoption enablement must be boolean [batch]")
    if (
        direct_action_visible.shape != layout.roles.shape
        or direct_action_visible.dtype != torch.bool
        or direct_action_visible.device != layout.roles.device
    ):
        raise ValueError("direct action visibility must be boolean [batch,tokens]")
    sensor = layout.roles == int(NativeRole.SENSOR)
    if (direct_action_visible & ~sensor).any():
        raise ValueError("only SENSOR rows may be declared direct action evidence")
    if (direct_action_visible & ~layout.valid).any():
        raise ValueError("invalid rows cannot be declared direct action evidence")
    permissions = native_role_permissions(device=layout.roles.device)[int(NativeRole.ACTION)]
    default = permissions[layout.roles] & layout.valid
    blocked_scene = (
        (sensor & ~direct_action_visible)
        | (layout.roles == int(NativeRole.HOST_AUX))
        | (layout.roles == int(NativeRole.PRIOR))
        | (layout.roles == int(NativeRole.MATCH))
    )
    restricted = default & ~blocked_scene
    return torch.where(enabled[:, None], restricted, default)


def posterior_adoption_attention_mask(
    mask: torch.Tensor,
    *,
    layout: NativeTokenLayout,
    enabled: torch.Tensor,
    direct_action_visible: torch.Tensor,
) -> torch.Tensor:
    """Close direct and repeated-layer scene bypasses for selected samples."""

    if (
        mask.shape != (layout.batch_size, layout.token_count, layout.token_count)
        or mask.dtype != torch.bool
        or mask.device != layout.roles.device
    ):
        raise ValueError("posterior-adoption mask must match the native layout")
    action_visible = posterior_adoption_action_key_visibility(
        layout,
        enabled=enabled,
        direct_action_visible=direct_action_visible,
    )
    result = mask.clone()
    action_queries = layout.roles == int(NativeRole.ACTION)
    result &= ~(enabled[:, None, None] & action_queries[:, :, None] & ~action_visible[:, None, :])

    # Otherwise current scene information can enter LANGUAGE in an early layer
    # and reach ACTION through the still-valid language route in a later layer.
    language_queries = layout.roles == int(NativeRole.LANGUAGE)
    sensor_keys = layout.roles == int(NativeRole.SENSOR)
    result &= ~(
        enabled[:, None, None]
        & language_queries[:, :, None]
        & sensor_keys[:, None, :]
    )

    # A direct state row is itself typed SENSOR. Without this restriction it
    # can absorb dense scene SENSOR rows in one shared layer and relay them to
    # ACTION in the next, reopening the bypass that this treatment is meant to
    # close. Direct state rows may exchange only explicitly direct evidence.
    direct_sensor_queries = sensor_keys & direct_action_visible
    non_direct_sensor_keys = sensor_keys & ~direct_action_visible
    result &= ~(
        enabled[:, None, None]
        & direct_sensor_queries[:, :, None]
        & non_direct_sensor_keys[:, None, :]
    )
    return result


def native_layerwise_history_mask(
    layout: NativeTokenLayout,
    *,
    prior_slice: slice,
    posterior_slice: slice,
    capacity: int,
    previous_memory_valid: torch.Tensor,
) -> torch.Tensor:
    """Expose memory ``k`` only to its paired prior/posterior queries.

    Both edges are required. Transformer queries in one layer are evaluated in
    parallel, so a posterior query cannot consume the updated prior output from
    that same layer. Direct access to the same-layer previous posterior closes
    the recurrent self-loop without exposing memory to sensors, language,
    actions, or another object row.
    """

    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        or prior_slice.start is None
        or prior_slice.stop is None
        or prior_slice.stop - prior_slice.start != capacity
        or posterior_slice.start is None
        or posterior_slice.stop is None
        or posterior_slice.stop - posterior_slice.start != capacity
    ):
        raise ValueError(
            "history attention requires one current prior and posterior per memory row"
        )
    if (
        previous_memory_valid.shape != (layout.batch_size,)
        or previous_memory_valid.dtype != torch.bool
        or previous_memory_valid.device != layout.roles.device
    ):
        raise ValueError("history validity must be boolean with one value per batch lane")
    mask = torch.zeros(
        (layout.batch_size, layout.token_count, capacity),
        dtype=torch.bool,
        device=layout.roles.device,
    )
    row = torch.arange(capacity, device=layout.roles.device)
    mask[:, prior_slice.start + row, row] = previous_memory_valid[:, None]
    mask[:, posterior_slice.start + row, row] = previous_memory_valid[:, None]
    mask &= layout.valid.unsqueeze(-1)
    return mask


def native_layerwise_prior_history_mask(
    layout: NativeTokenLayout,
    *,
    prior_slice: slice,
    capacity: int,
    previous_memory_valid: torch.Tensor,
) -> torch.Tensor:
    """Expose historical posterior row ``k`` only to prior query ``k``.

    ADR149 uses this mask in the prior-only pass. The correction/action pass is
    never given the historical posterior tensor, so it cannot bypass the prior
    trace produced by that pass.
    """

    if (
        isinstance(capacity, bool)
        or not isinstance(capacity, int)
        or capacity <= 0
        or prior_slice.start is None
        or prior_slice.stop is None
        or prior_slice.stop - prior_slice.start != capacity
    ):
        raise ValueError("prior history attention requires one prior query per memory row")
    if (
        previous_memory_valid.shape != (layout.batch_size,)
        or previous_memory_valid.dtype != torch.bool
        or previous_memory_valid.device != layout.roles.device
    ):
        raise ValueError("history validity must be boolean with one value per batch lane")
    mask = torch.zeros(
        (layout.batch_size, layout.token_count, capacity),
        dtype=torch.bool,
        device=layout.roles.device,
    )
    row = torch.arange(capacity, device=layout.roles.device)
    mask[:, prior_slice.start + row, row] = previous_memory_valid[:, None]
    mask &= layout.valid.unsqueeze(-1)
    return mask


def native_layerwise_prior_trace_mask(
    layout: NativeTokenLayout,
    *,
    prior_slice: slice,
    posterior_slice: slice,
    capacity: int,
) -> torch.Tensor:
    """Expose a produced prior trace to correction rows and official action queries.

    Object rows retain paired identity: prior/posterior query ``k`` sees trace row
    ``k``. Official action queries may read the complete prior object set. This
    mask is valid only for the transient v3 trace, never the previous posterior.
    """

    trace_valid = torch.ones(
        layout.batch_size,
        dtype=torch.bool,
        device=layout.roles.device,
    )
    mask = native_layerwise_history_mask(
        layout,
        prior_slice=prior_slice,
        posterior_slice=posterior_slice,
        capacity=capacity,
        previous_memory_valid=trace_valid,
    )
    action_queries = layout.valid & (layout.roles == int(NativeRole.ACTION))
    mask |= action_queries.unsqueeze(-1)
    return mask


def expand_square_mask(
    host_mask: torch.Tensor,
    *,
    insertion_index: int,
    inserted_count: int,
) -> torch.Tensor:
    if host_mask.ndim != 3 or host_mask.shape[1] != host_mask.shape[2]:
        raise ValueError("host attention mask must be square [batch, tokens, tokens]")
    if host_mask.dtype != torch.bool:
        raise TypeError("host attention mask must be boolean")
    old_count = host_mask.shape[1]
    if not 0 <= insertion_index <= old_count or inserted_count < 0:
        raise ValueError("mask insertion range is invalid")
    if inserted_count == 0:
        return host_mask
    new_count = old_count + inserted_count
    expanded = torch.ones(
        (host_mask.shape[0], new_count, new_count),
        dtype=torch.bool,
        device=host_mask.device,
    )
    old_indices = torch.cat(
        (
            torch.arange(insertion_index, device=host_mask.device),
            torch.arange(insertion_index + inserted_count, new_count, device=host_mask.device),
        )
    )
    expanded[:, old_indices[:, None], old_indices[None, :]] = host_mask
    return expanded


validate_native_causality()

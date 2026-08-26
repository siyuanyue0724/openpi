"""Causal role graph for the ADR-158/159 task-addressed object posterior.

The graph keeps the released dense LingBot path while separating task encoding
from object-value transport. ``TASK_QUERY`` affects ``OBJECT_READ`` through the
host Q/K path only; ``OBJECT_READ`` is the sole PICF-specific value mediator to
``ACTION``. Serialized state writers are row-local across every shared layer.
This versioned contract does not mutate the historical ADR-74 graph.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, IntEnum

import torch


class TaskAddressRole(IntEnum):
    """Runtime role, never an object class or persistent semantic identity."""

    SENSOR = 0
    LANGUAGE = 1
    TASK_TEXT = 2
    TASK_QUERY = 3
    HOST_CURRENT = 4
    HOST_FUTURE = 5
    CONTROL = 6
    OBJECT_MEMORY = 7
    PRIOR = 8
    POSTERIOR = 9
    OBJECT_READ = 10
    ACTION = 11
    PREDICT = 12
    SENSOR_BOUNDARY = 13


class TaskAddressActionInformationSet(str, Enum):
    """Per-sample action information set, separate from evaluation interventions."""

    FACTUAL = "factual"
    MEDIATOR_REQUIRED = "mediator-required"


MEDIATOR_REQUIRED_RAW_ACTION_KEY_ROLES = (
    TaskAddressRole.SENSOR,
    TaskAddressRole.SENSOR_BOUNDARY,
    TaskAddressRole.HOST_CURRENT,
    TaskAddressRole.LANGUAGE,
)


_ROLE_COUNT = len(TaskAddressRole)


def normalize_task_address_action_information_sets(
    values: Sequence[TaskAddressActionInformationSet] | None,
    *,
    batch_size: int,
) -> tuple[TaskAddressActionInformationSet, ...]:
    """Return one typed information-set assignment per batch sample."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("task-address information sets require a positive batch size")
    if values is None:
        return (TaskAddressActionInformationSet.FACTUAL,) * batch_size
    normalized = tuple(values)
    if len(normalized) != batch_size:
        raise ValueError("task-address information sets must match the batch size")
    if any(not isinstance(value, TaskAddressActionInformationSet) for value in normalized):
        raise TypeError("task-address information sets must use their typed enum")
    return normalized


def task_address_role_permissions(
    *,
    action_information_set: TaskAddressActionInformationSet = (
        TaskAddressActionInformationSet.FACTUAL
    ),
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return ``allowed[query, key]`` for one shared LingBot layer."""

    if not isinstance(action_information_set, TaskAddressActionInformationSet):
        raise TypeError("action information set must use its typed enum")

    allowed = torch.zeros((_ROLE_COUNT, _ROLE_COUNT), dtype=torch.bool, device=device)

    def permit(query: TaskAddressRole, *keys: TaskAddressRole) -> None:
        indices = torch.tensor([int(key) for key in keys], device=allowed.device)
        allowed[int(query), indices] = True

    # The released dense path remains available without reading PICF state.
    permit(
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
    )
    permit(
        TaskAddressRole.SENSOR_BOUNDARY,
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
    )
    permit(
        TaskAddressRole.LANGUAGE,
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
        TaskAddressRole.LANGUAGE,
    )
    # TASK_TEXT duplicates the complete input instruction embeddings, but it
    # cannot absorb scene state. TASK_QUERY encodes that text inside the same
    # host and is never directly visible to ACTION.
    permit(TaskAddressRole.TASK_TEXT, TaskAddressRole.TASK_TEXT)
    permit(
        TaskAddressRole.TASK_QUERY,
        TaskAddressRole.TASK_TEXT,
        TaskAddressRole.TASK_QUERY,
    )
    permit(
        TaskAddressRole.HOST_CURRENT,
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
        TaskAddressRole.LANGUAGE,
        TaskAddressRole.HOST_CURRENT,
    )
    permit(
        TaskAddressRole.HOST_FUTURE,
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
        TaskAddressRole.LANGUAGE,
        TaskAddressRole.HOST_CURRENT,
        TaskAddressRole.HOST_FUTURE,
    )
    permit(TaskAddressRole.CONTROL, TaskAddressRole.CONTROL)

    # Physical belief is task-free. Historical memory is an immutable input
    # within one control step; strict row-pair restrictions are applied by the
    # only mask builder below.
    permit(TaskAddressRole.OBJECT_MEMORY, TaskAddressRole.OBJECT_MEMORY)
    permit(
        TaskAddressRole.PRIOR,
        TaskAddressRole.OBJECT_MEMORY,
        TaskAddressRole.CONTROL,
        TaskAddressRole.PRIOR,
    )
    permit(
        TaskAddressRole.POSTERIOR,
        TaskAddressRole.SENSOR,
        TaskAddressRole.CONTROL,
        TaskAddressRole.PRIOR,
        TaskAddressRole.POSTERIOR,
    )
    # OBJECT_READ receives only physical Values. TASK_QUERY conditions its Q/K
    # path separately, so task semantics cannot be relayed as an unmediated
    # residual Value. Dense context remains in SENSOR; null is readout-side.
    permit(
        TaskAddressRole.OBJECT_READ,
        TaskAddressRole.PRIOR,
        TaskAddressRole.POSTERIOR,
    )
    action_keys = [
        TaskAddressRole.TASK_TEXT,
        TaskAddressRole.OBJECT_READ,
        TaskAddressRole.ACTION,
    ]
    if action_information_set is TaskAddressActionInformationSet.FACTUAL:
        action_keys = [*MEDIATOR_REQUIRED_RAW_ACTION_KEY_ROLES, *action_keys]
    permit(TaskAddressRole.ACTION, *action_keys)

    # Prediction queries receive only causal physical state. Targets are
    # loss-side tensors and never token keys.
    permit(
        TaskAddressRole.PREDICT,
        TaskAddressRole.CONTROL,
        TaskAddressRole.PRIOR,
        TaskAddressRole.POSTERIOR,
        TaskAddressRole.PREDICT,
    )
    return allowed


def task_address_qk_conditioning(*, device: torch.device | str | None = None) -> torch.Tensor:
    """Return non-Value information edges as ``conditioned[source, sink]``.

    The current contract has exactly one such role edge: TASK_QUERY changes the
    Q/K address of OBJECT_READ. It is deliberately absent from the attention
    Value permissions above.
    """

    conditioned = torch.zeros((_ROLE_COUNT, _ROLE_COUNT), dtype=torch.bool, device=device)
    conditioned[int(TaskAddressRole.TASK_QUERY), int(TaskAddressRole.OBJECT_READ)] = True
    return conditioned


def task_address_information_paths(
    *,
    action_information_set: TaskAddressActionInformationSet = (
        TaskAddressActionInformationSet.FACTUAL
    ),
) -> torch.Tensor:
    """Return repeated-layer reachability as ``reachable[source, sink]``."""

    direct = task_address_role_permissions(
        action_information_set=action_information_set,
    ).T.clone()
    direct |= task_address_qk_conditioning()
    closure = direct.clone()
    for intermediate in range(_ROLE_COUNT):
        closure |= closure[:, intermediate].unsqueeze(1) & closure[intermediate].unsqueeze(0)
    return closure


def task_address_paths_without_mediator(
    *,
    action_information_set: TaskAddressActionInformationSet = (
        TaskAddressActionInformationSet.FACTUAL
    ),
) -> torch.Tensor:
    """Return closure after removing OBJECT_READ as a source and sink."""

    direct = task_address_role_permissions(
        action_information_set=action_information_set,
    ).T.clone()
    direct |= task_address_qk_conditioning()
    mediator = int(TaskAddressRole.OBJECT_READ)
    direct[mediator] = False
    direct[:, mediator] = False
    closure = direct.clone()
    for intermediate in range(_ROLE_COUNT):
        closure |= closure[:, intermediate].unsqueeze(1) & closure[intermediate].unsqueeze(0)
    return closure


def validate_task_address_causality() -> None:
    """Reject future leakage, language-written belief, or raw-row action bypass."""

    paths = task_address_information_paths()
    without_mediator = task_address_paths_without_mediator()
    required_paths = task_address_information_paths(
        action_information_set=TaskAddressActionInformationSet.MEDIATOR_REQUIRED,
    )
    required_without_mediator = task_address_paths_without_mediator(
        action_information_set=TaskAddressActionInformationSet.MEDIATOR_REQUIRED,
    )

    for source in (
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
        TaskAddressRole.LANGUAGE,
        TaskAddressRole.TASK_TEXT,
        TaskAddressRole.HOST_CURRENT,
        TaskAddressRole.HOST_FUTURE,
        TaskAddressRole.POSTERIOR,
        TaskAddressRole.TASK_QUERY,
        TaskAddressRole.OBJECT_READ,
        TaskAddressRole.ACTION,
        TaskAddressRole.PREDICT,
    ):
        if paths[int(source), int(TaskAddressRole.PRIOR)]:
            raise RuntimeError(f"forbidden repeated-layer path {source.name}->PRIOR")

    for source in (
        TaskAddressRole.LANGUAGE,
        TaskAddressRole.TASK_TEXT,
        TaskAddressRole.TASK_QUERY,
        TaskAddressRole.HOST_CURRENT,
        TaskAddressRole.HOST_FUTURE,
        TaskAddressRole.OBJECT_READ,
        TaskAddressRole.ACTION,
        TaskAddressRole.PREDICT,
    ):
        if paths[int(source), int(TaskAddressRole.POSTERIOR)]:
            raise RuntimeError(f"forbidden repeated-layer path {source.name}->POSTERIOR")

    if paths[int(TaskAddressRole.HOST_FUTURE), int(TaskAddressRole.ACTION)]:
        raise RuntimeError("future host queries reach action")
    if required_paths[int(TaskAddressRole.HOST_FUTURE), int(TaskAddressRole.ACTION)]:
        raise RuntimeError("future host queries reach mediator-required action")

    for source in (
        TaskAddressRole.CONTROL,
        TaskAddressRole.OBJECT_MEMORY,
        TaskAddressRole.PRIOR,
        TaskAddressRole.POSTERIOR,
    ):
        if without_mediator[int(source), int(TaskAddressRole.ACTION)]:
            raise RuntimeError(f"raw PICF state bypasses OBJECT_READ from {source.name}")

    value_permissions = task_address_role_permissions()
    if value_permissions[int(TaskAddressRole.OBJECT_READ), int(TaskAddressRole.TASK_QUERY)]:
        raise RuntimeError("task-query Value can bypass object selection")
    if value_permissions[int(TaskAddressRole.ACTION), int(TaskAddressRole.TASK_QUERY)]:
        raise RuntimeError("task-query Value can reach action")
    if not paths[int(TaskAddressRole.TASK_QUERY), int(TaskAddressRole.OBJECT_READ)]:
        raise RuntimeError("task query cannot condition object reads")
    if not paths[int(TaskAddressRole.POSTERIOR), int(TaskAddressRole.OBJECT_READ)]:
        raise RuntimeError("posterior cannot reach object reads")
    if not paths[int(TaskAddressRole.OBJECT_READ), int(TaskAddressRole.ACTION)]:
        raise RuntimeError("object reads cannot reach action")
    if not required_paths[int(TaskAddressRole.TASK_TEXT), int(TaskAddressRole.ACTION)]:
        raise RuntimeError("pure task text cannot reach mediator-required action")
    if not required_paths[int(TaskAddressRole.OBJECT_READ), int(TaskAddressRole.ACTION)]:
        raise RuntimeError("object reads cannot reach mediator-required action")
    for source in MEDIATOR_REQUIRED_RAW_ACTION_KEY_ROLES:
        if required_without_mediator[int(source), int(TaskAddressRole.ACTION)]:
            raise RuntimeError(
                f"raw object evidence bypasses OBJECT_READ from {source.name}"
            )


@dataclass(frozen=True, slots=True)
class TaskAddressTokenLayout:
    roles: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self) -> None:
        if self.roles.ndim != 2 or self.valid.shape != self.roles.shape:
            raise ValueError("roles and validity must have shape [batch, tokens]")
        if self.roles.dtype != torch.long or self.valid.dtype != torch.bool:
            raise TypeError("roles must be long and validity must be boolean")
        if self.roles.device != self.valid.device:
            raise ValueError("roles and validity must share one device")
        if ((self.roles < 0) | (self.roles >= _ROLE_COUNT)).any():
            raise ValueError("layout contains an unknown task-address role")

    @property
    def batch_size(self) -> int:
        return self.roles.shape[0]

    @property
    def token_count(self) -> int:
        return self.roles.shape[1]


def task_address_action_key_visibility(
    layout: TaskAddressTokenLayout,
    *,
    action_information_sets: Sequence[TaskAddressActionInformationSet] | None = None,
) -> torch.Tensor:
    """Return action-authorized keys for each sample under its information set."""

    information_sets = normalize_task_address_action_information_sets(
        action_information_sets,
        batch_size=layout.batch_size,
    )
    visibility = torch.zeros_like(layout.valid)
    for batch_index, information_set in enumerate(information_sets):
        action_keys = task_address_role_permissions(
            action_information_set=information_set,
            device=layout.roles.device,
        )[int(TaskAddressRole.ACTION)]
        visibility[batch_index] = action_keys[layout.roles[batch_index]]
    return visibility & layout.valid


@dataclass(frozen=True, slots=True)
class TaskAddressStateSlices:
    """Contiguous row slices whose serialized writers must remain row-local."""

    prior: slice
    posterior: slice
    capacity: int
    memory: slice | None = None

    def __post_init__(self) -> None:
        if isinstance(self.capacity, bool) or not isinstance(self.capacity, int):
            raise TypeError("capacity must be an integer")
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        for name, token_slice in (
            ("prior", self.prior),
            ("posterior", self.posterior),
            ("memory", self.memory),
        ):
            if token_slice is None:
                continue
            if (
                token_slice.start is None
                or token_slice.stop is None
                or token_slice.stop - token_slice.start != self.capacity
            ):
                raise ValueError(f"{name} slice must contain exactly capacity rows")


def _validate_state_slice_roles(
    layout: TaskAddressTokenLayout,
    state: TaskAddressStateSlices,
) -> None:
    expected = [
        (state.prior, TaskAddressRole.PRIOR),
        (state.posterior, TaskAddressRole.POSTERIOR),
    ]
    if state.memory is not None:
        expected.append((state.memory, TaskAddressRole.OBJECT_MEMORY))
    for token_slice, role in expected:
        if token_slice.stop > layout.token_count:
            raise ValueError("state slice lies outside the task-address layout")
        if not (layout.roles[:, token_slice] == int(role)).all():
            raise ValueError(f"{role.name} slice contains another role")


def _restrict_serialized_state_writers(
    mask: torch.Tensor,
    *,
    layout: TaskAddressTokenLayout,
    state: TaskAddressStateSlices,
) -> None:
    """Apply row-local Value access to every serialized state writer in place."""

    _validate_state_slice_roles(layout, state)
    row = torch.arange(state.capacity, device=mask.device)

    prior_queries = state.prior.start + row
    posterior_queries = state.posterior.start + row

    prior_self = mask[:, prior_queries, state.prior.start + row].clone()
    mask[:, prior_queries, state.prior] = False
    mask[:, prior_queries, state.prior.start + row] = prior_self

    posterior_prior = mask[:, posterior_queries, state.prior.start + row].clone()
    posterior_self = mask[:, posterior_queries, state.posterior.start + row].clone()
    mask[:, posterior_queries, state.prior] = False
    mask[:, posterior_queries, state.posterior] = False
    mask[:, posterior_queries, state.prior.start + row] = posterior_prior
    mask[:, posterior_queries, state.posterior.start + row] = posterior_self

    if state.memory is not None:
        memory_queries = state.memory.start + row
        memory_self = mask[:, memory_queries, state.memory.start + row].clone()
        prior_memory = mask[:, prior_queries, state.memory.start + row].clone()
        mask[:, memory_queries, :] = False
        mask[:, memory_queries, state.memory.start + row] = memory_self
        mask[:, prior_queries, state.memory] = False
        mask[:, prior_queries, state.memory.start + row] = prior_memory


def task_address_attention_mask(
    layout: TaskAddressTokenLayout,
    *,
    host_mask: torch.Tensor,
    control_slice: slice | None = None,
    state_slices: TaskAddressStateSlices | None = None,
    action_information_sets: Sequence[TaskAddressActionInformationSet] | None = None,
) -> torch.Tensor:
    """Intersect the official host mask with the ADR-158 role contract."""

    expected = (layout.batch_size, layout.token_count, layout.token_count)
    if host_mask.shape != expected or host_mask.dtype != torch.bool:
        raise ValueError("host mask must be boolean [batch,tokens,tokens]")
    if host_mask.device != layout.roles.device:
        raise ValueError("host mask and layout must share one device")

    information_sets = normalize_task_address_action_information_sets(
        action_information_sets,
        batch_size=layout.batch_size,
    )
    query_roles = layout.roles.unsqueeze(-1).expand(-1, -1, layout.token_count)
    key_roles = layout.roles.unsqueeze(1).expand(-1, layout.token_count, -1)
    result = torch.stack(
        tuple(
            task_address_role_permissions(
                action_information_set=information_set,
                device=layout.roles.device,
            )[query_roles[batch_index], key_roles[batch_index]]
            for batch_index, information_set in enumerate(information_sets)
        ),
        dim=0,
    )
    result &= host_mask
    result &= layout.valid.unsqueeze(-1) & layout.valid.unsqueeze(1)

    if control_slice is not None:
        if control_slice.start is None or control_slice.stop is None:
            raise ValueError("control slice must be bounded")
        count = control_slice.stop - control_slice.start
        if count < 0:
            raise ValueError("control slice is invalid")
        causal = torch.ones((count, count), dtype=torch.bool, device=result.device).tril()
        result[:, control_slice, control_slice] &= causal
    state_roles = (
        (layout.roles == int(TaskAddressRole.OBJECT_MEMORY))
        | (layout.roles == int(TaskAddressRole.PRIOR))
        | (layout.roles == int(TaskAddressRole.POSTERIOR))
    )
    if state_roles.any() and state_slices is None:
        raise ValueError("serialized state roles require explicit row-local slices")
    if state_slices is not None:
        _restrict_serialized_state_writers(
            result,
            layout=layout,
            state=state_slices,
        )
    return result


def task_address_layerwise_state_mask(
    layout: TaskAddressTokenLayout,
    *,
    prior_slice: slice,
    posterior_slice: slice,
    capacity: int,
    state_valid: torch.Tensor,
) -> torch.Tensor:
    """Expose external state row ``k`` only to current row writers ``k``.

    The returned mask is ``[batch, current_queries, external_rows]``.  It is
    intentionally separate from the square in-host Value mask because the
    released LingBot patch prepends persistent Values after the official cache
    capture.  No action, task, sensor, or different-row query can read this
    external state surface.
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
        raise ValueError("layerwise state requires paired prior/posterior slices")
    if (
        state_valid.shape != (layout.batch_size,)
        or state_valid.dtype != torch.bool
        or state_valid.device != layout.roles.device
    ):
        raise ValueError("layerwise state validity must be boolean [batch]")
    _validate_state_slice_roles(
        layout,
        TaskAddressStateSlices(
            prior=prior_slice,
            posterior=posterior_slice,
            capacity=capacity,
        ),
    )

    mask = torch.zeros(
        (layout.batch_size, layout.token_count, capacity),
        dtype=torch.bool,
        device=layout.roles.device,
    )
    row = torch.arange(capacity, device=layout.roles.device)
    enabled = state_valid[:, None]
    mask[:, prior_slice.start + row, row] = enabled
    mask[:, posterior_slice.start + row, row] = enabled
    mask &= layout.valid.unsqueeze(-1)
    return mask


def paired_task_query_object_read_conditioning(
    layout: TaskAddressTokenLayout,
    *,
    task_query_slice: slice,
    object_read_slice: slice,
    query_count: int,
) -> torch.Tensor:
    """Return token-level Q/K conditioning edges for paired task/object reads."""

    if isinstance(query_count, bool) or not isinstance(query_count, int) or query_count <= 0:
        raise ValueError("query_count must be a positive integer")
    for name, token_slice, role in (
        ("task query", task_query_slice, TaskAddressRole.TASK_QUERY),
        ("object read", object_read_slice, TaskAddressRole.OBJECT_READ),
    ):
        if (
            token_slice.start is None
            or token_slice.stop is None
            or token_slice.stop - token_slice.start != query_count
        ):
            raise ValueError(f"{name} slice must contain exactly query_count rows")
        if (
            token_slice.stop > layout.token_count
            or not (layout.roles[:, token_slice] == int(role)).all()
        ):
            raise ValueError(f"{name} slice contains another role or lies outside the layout")

    mask = torch.zeros(
        (layout.batch_size, layout.token_count, layout.token_count),
        dtype=torch.bool,
        device=layout.roles.device,
    )
    row = torch.arange(query_count, device=layout.roles.device)
    source = task_query_slice.start + row
    sink = object_read_slice.start + row
    # This matrix is oriented [source, sink], unlike an attention query/key mask.
    mask[:, source, sink] = layout.valid[:, source] & layout.valid[:, sink]
    return mask


def token_information_paths(
    attention_mask: torch.Tensor,
    *,
    qk_conditioning: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return token-level repeated-layer closure as ``[batch, source, sink]``."""

    if (
        attention_mask.ndim != 3
        or attention_mask.shape[1] != attention_mask.shape[2]
        or attention_mask.dtype != torch.bool
    ):
        raise ValueError("attention mask must be boolean [batch,tokens,tokens]")
    direct = attention_mask.transpose(1, 2).clone()
    if qk_conditioning is not None:
        if qk_conditioning.shape != direct.shape or qk_conditioning.dtype != torch.bool:
            raise ValueError("Q/K conditioning must be boolean and match the token graph")
        direct |= qk_conditioning
    closure = direct.clone()
    for intermediate in range(attention_mask.shape[1]):
        closure |= closure[:, :, intermediate].unsqueeze(2) & closure[:, intermediate, :].unsqueeze(
            1
        )
    return closure


validate_task_address_causality()

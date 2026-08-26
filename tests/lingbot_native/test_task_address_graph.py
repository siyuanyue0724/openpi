from __future__ import annotations

import pytest
import torch

from picf_next.lingbot_native.task_address_graph import (
    TaskAddressActionInformationSet,
    TaskAddressRole,
    TaskAddressStateSlices,
    TaskAddressTokenLayout,
    paired_task_query_object_read_conditioning,
    task_address_attention_mask,
    task_address_information_paths,
    task_address_paths_without_mediator,
    task_address_qk_conditioning,
    task_address_role_permissions,
    token_information_paths,
    validate_task_address_causality,
)


def test_task_address_role_permissions_separate_value_and_qk_flow() -> None:
    expected = {
        TaskAddressRole.SENSOR: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.SENSOR_BOUNDARY,
        },
        TaskAddressRole.SENSOR_BOUNDARY: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.SENSOR_BOUNDARY,
        },
        TaskAddressRole.LANGUAGE: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.SENSOR_BOUNDARY,
            TaskAddressRole.LANGUAGE,
        },
        TaskAddressRole.TASK_TEXT: {TaskAddressRole.TASK_TEXT},
        TaskAddressRole.TASK_QUERY: {
            TaskAddressRole.TASK_TEXT,
            TaskAddressRole.TASK_QUERY,
        },
        TaskAddressRole.HOST_CURRENT: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.SENSOR_BOUNDARY,
            TaskAddressRole.LANGUAGE,
            TaskAddressRole.HOST_CURRENT,
        },
        TaskAddressRole.HOST_FUTURE: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.SENSOR_BOUNDARY,
            TaskAddressRole.LANGUAGE,
            TaskAddressRole.HOST_CURRENT,
            TaskAddressRole.HOST_FUTURE,
        },
        TaskAddressRole.CONTROL: {TaskAddressRole.CONTROL},
        TaskAddressRole.OBJECT_MEMORY: {TaskAddressRole.OBJECT_MEMORY},
        TaskAddressRole.PRIOR: {
            TaskAddressRole.OBJECT_MEMORY,
            TaskAddressRole.CONTROL,
            TaskAddressRole.PRIOR,
        },
        TaskAddressRole.POSTERIOR: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.CONTROL,
            TaskAddressRole.PRIOR,
            TaskAddressRole.POSTERIOR,
        },
        TaskAddressRole.OBJECT_READ: {
            TaskAddressRole.PRIOR,
            TaskAddressRole.POSTERIOR,
        },
        TaskAddressRole.ACTION: {
            TaskAddressRole.SENSOR,
            TaskAddressRole.SENSOR_BOUNDARY,
            TaskAddressRole.LANGUAGE,
            TaskAddressRole.HOST_CURRENT,
            TaskAddressRole.TASK_TEXT,
            TaskAddressRole.OBJECT_READ,
            TaskAddressRole.ACTION,
        },
        TaskAddressRole.PREDICT: {
            TaskAddressRole.CONTROL,
            TaskAddressRole.PRIOR,
            TaskAddressRole.POSTERIOR,
            TaskAddressRole.PREDICT,
        },
    }
    permissions = task_address_role_permissions()
    for query in TaskAddressRole:
        observed = {
            key
            for key in TaskAddressRole
            if bool(permissions[int(query), int(key)].item())
        }
        assert observed == expected[query]

    qk = task_address_qk_conditioning()
    assert qk.sum().item() == 1
    assert qk[int(TaskAddressRole.TASK_QUERY), int(TaskAddressRole.OBJECT_READ)]
    assert not permissions[int(TaskAddressRole.OBJECT_READ), int(TaskAddressRole.TASK_QUERY)]


def test_sensor_boundary_preserves_current_prefix_without_becoming_physical_evidence() -> None:
    roles = torch.tensor(
        [[
            int(TaskAddressRole.SENSOR_BOUNDARY),
            int(TaskAddressRole.SENSOR),
            int(TaskAddressRole.SENSOR_BOUNDARY),
            int(TaskAddressRole.LANGUAGE),
        ]]
    )
    layout = TaskAddressTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    host_mask = torch.ones(1, 4, 4, dtype=torch.bool).tril()
    observed = task_address_attention_mask(layout, host_mask=host_mask)
    assert torch.equal(observed, host_mask)
    assert not task_address_role_permissions()[
        int(TaskAddressRole.POSTERIOR),
        int(TaskAddressRole.SENSOR_BOUNDARY),
    ]


def test_repeated_layers_keep_physical_belief_task_free() -> None:
    validate_task_address_causality()
    paths = task_address_information_paths()
    for source in (
        TaskAddressRole.SENSOR_BOUNDARY,
        TaskAddressRole.LANGUAGE,
        TaskAddressRole.TASK_TEXT,
        TaskAddressRole.TASK_QUERY,
        TaskAddressRole.HOST_CURRENT,
        TaskAddressRole.HOST_FUTURE,
        TaskAddressRole.OBJECT_READ,
        TaskAddressRole.ACTION,
        TaskAddressRole.PREDICT,
    ):
        assert not paths[int(source), int(TaskAddressRole.PRIOR)]
        if source is not TaskAddressRole.SENSOR_BOUNDARY:
            assert not paths[int(source), int(TaskAddressRole.POSTERIOR)]


def test_object_read_is_the_only_picf_state_path_to_action() -> None:
    paths = task_address_information_paths()
    without_mediator = task_address_paths_without_mediator()
    for source in (
        TaskAddressRole.CONTROL,
        TaskAddressRole.OBJECT_MEMORY,
        TaskAddressRole.PRIOR,
        TaskAddressRole.POSTERIOR,
    ):
        assert paths[int(source), int(TaskAddressRole.ACTION)]
        assert not without_mediator[int(source), int(TaskAddressRole.ACTION)]
    assert paths[int(TaskAddressRole.TASK_QUERY), int(TaskAddressRole.OBJECT_READ)]
    assert paths[int(TaskAddressRole.POSTERIOR), int(TaskAddressRole.OBJECT_READ)]
    assert paths[int(TaskAddressRole.OBJECT_READ), int(TaskAddressRole.ACTION)]


def test_future_host_queries_cannot_reach_action() -> None:
    paths = task_address_information_paths()
    assert not paths[int(TaskAddressRole.HOST_FUTURE), int(TaskAddressRole.ACTION)]


def _complete_layout() -> tuple[TaskAddressTokenLayout, TaskAddressStateSlices]:
    roles = torch.tensor(
        [[
            int(TaskAddressRole.SENSOR),
            int(TaskAddressRole.LANGUAGE),
            int(TaskAddressRole.TASK_TEXT),
            int(TaskAddressRole.TASK_QUERY),
            int(TaskAddressRole.HOST_CURRENT),
            int(TaskAddressRole.HOST_FUTURE),
            int(TaskAddressRole.CONTROL),
            int(TaskAddressRole.OBJECT_MEMORY),
            int(TaskAddressRole.OBJECT_MEMORY),
            int(TaskAddressRole.PRIOR),
            int(TaskAddressRole.PRIOR),
            int(TaskAddressRole.POSTERIOR),
            int(TaskAddressRole.POSTERIOR),
            int(TaskAddressRole.OBJECT_READ),
            int(TaskAddressRole.ACTION),
        ]]
    )
    layout = TaskAddressTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    state = TaskAddressStateSlices(
        memory=slice(7, 9),
        prior=slice(9, 11),
        posterior=slice(11, 13),
        capacity=2,
    )
    return layout, state


def test_attention_mask_preserves_dense_baseline_and_blocks_raw_rows() -> None:
    layout, state = _complete_layout()
    mask = task_address_attention_mask(
        layout,
        host_mask=torch.ones(
            1, layout.token_count, layout.token_count, dtype=torch.bool
        ),
        state_slices=state,
    )[0]
    action = 14
    assert mask[action, 0]
    assert mask[action, 1]
    assert mask[action, 2]
    assert not mask[action, 3]
    assert mask[action, 4]
    assert not mask[action, 5]
    assert not mask[action, 6]
    assert not mask[action, 7]
    assert not mask[action, 9]
    assert not mask[action, 11]
    assert mask[action, 13]


def test_task_query_is_not_an_object_read_value_source() -> None:
    layout, state = _complete_layout()
    mask = task_address_attention_mask(
        layout,
        host_mask=torch.ones(
            1, layout.token_count, layout.token_count, dtype=torch.bool
        ),
        state_slices=state,
    )[0]
    task_query = 3
    object_read = 13
    assert mask[task_query, 2]
    assert mask[task_query, task_query]
    assert not mask[task_query, 0]
    assert not mask[task_query, 11]
    assert not mask[object_read, task_query]
    assert mask[object_read, 9]
    assert mask[object_read, 10]
    assert mask[object_read, 11]
    assert mask[object_read, 12]


def test_serialized_state_is_row_local_under_multilayer_closure() -> None:
    layout, state = _complete_layout()
    attention = task_address_attention_mask(
        layout,
        host_mask=torch.ones(
            1, layout.token_count, layout.token_count, dtype=torch.bool
        ),
        state_slices=state,
    )
    paths = token_information_paths(attention)[0]

    memory0, memory1 = 7, 8
    prior0, prior1 = 9, 10
    posterior0, posterior1 = 11, 12
    assert paths[memory0, prior0]
    assert paths[memory0, posterior0]
    assert not paths[memory0, prior1]
    assert not paths[memory0, posterior1]
    assert paths[memory1, prior1]
    assert paths[memory1, posterior1]
    assert not paths[memory1, prior0]
    assert not paths[memory1, posterior0]
    assert not paths[prior0, prior1]
    assert not paths[prior0, posterior1]
    assert not paths[posterior0, posterior1]


def test_task_text_has_a_pure_semantic_action_path_and_conditions_object_read() -> None:
    layout, state = _complete_layout()
    attention = task_address_attention_mask(
        layout,
        host_mask=torch.ones(
            1, layout.token_count, layout.token_count, dtype=torch.bool
        ),
        state_slices=state,
    )
    value_paths = token_information_paths(attention)[0]
    task_text, task_query, object_read, action = 2, 3, 13, 14
    assert value_paths[task_text, task_query]
    assert not value_paths[task_text, object_read]
    assert value_paths[task_text, action]

    conditioning = paired_task_query_object_read_conditioning(
        layout,
        task_query_slice=slice(3, 4),
        object_read_slice=slice(13, 14),
        query_count=1,
    )
    conditioned_paths = token_information_paths(
        attention,
        qk_conditioning=conditioning,
    )[0]
    assert conditioned_paths[task_query, object_read]
    assert conditioned_paths[task_text, object_read]
    assert conditioned_paths[task_text, action]


def test_mediator_required_multilayer_cut_set_blocks_every_raw_object_role() -> None:
    paths = task_address_information_paths(
        action_information_set=TaskAddressActionInformationSet.MEDIATOR_REQUIRED
    )
    without_mediator = task_address_paths_without_mediator(
        action_information_set=TaskAddressActionInformationSet.MEDIATOR_REQUIRED
    )
    for source in (
        TaskAddressRole.SENSOR,
        TaskAddressRole.SENSOR_BOUNDARY,
    ):
        assert paths[int(source), int(TaskAddressRole.ACTION)]
        assert not without_mediator[int(source), int(TaskAddressRole.ACTION)]
    for source in (
        TaskAddressRole.HOST_CURRENT,
        TaskAddressRole.LANGUAGE,
    ):
        assert not paths[int(source), int(TaskAddressRole.ACTION)]
        assert not without_mediator[int(source), int(TaskAddressRole.ACTION)]
    assert paths[int(TaskAddressRole.TASK_TEXT), int(TaskAddressRole.ACTION)]
    assert paths[int(TaskAddressRole.OBJECT_READ), int(TaskAddressRole.ACTION)]

    layout, state = _complete_layout()
    attention = task_address_attention_mask(
        layout,
        host_mask=torch.ones(1, layout.token_count, layout.token_count, dtype=torch.bool),
        state_slices=state,
        action_information_sets=(
            TaskAddressActionInformationSet.MEDIATOR_REQUIRED,
        ),
    )
    action = 14
    for raw_source in (0, 1, 4):
        assert not attention[0, action, raw_source]
    assert attention[0, action, 2]
    assert attention[0, action, 13]

    without_object_read = attention.clone()
    without_object_read[:, 13, :] = False
    without_object_read[:, :, 13] = False
    token_paths = token_information_paths(without_object_read)[0]
    for raw_source in (0, 1, 4):
        assert not token_paths[raw_source, action]


def test_state_roles_require_the_unique_row_local_mask_builder() -> None:
    layout, _state = _complete_layout()
    with pytest.raises(ValueError, match="row-local"):
        task_address_attention_mask(
            layout,
            host_mask=torch.ones(
                1, layout.token_count, layout.token_count, dtype=torch.bool
            ),
        )


def test_invalid_memory_row_cannot_enter_its_paired_prior() -> None:
    layout, state = _complete_layout()
    valid = layout.valid.clone()
    valid[:, 8] = False
    invalid_layout = TaskAddressTokenLayout(roles=layout.roles, valid=valid)
    mask = task_address_attention_mask(
        invalid_layout,
        host_mask=torch.ones(
            1, layout.token_count, layout.token_count, dtype=torch.bool
        ),
        state_slices=state,
    )[0]
    assert mask[9, 7]
    assert not mask[10, 8]


def test_row_locality_never_reopens_an_edge_denied_by_the_host() -> None:
    layout, state = _complete_layout()
    host_mask = torch.ones(
        1, layout.token_count, layout.token_count, dtype=torch.bool
    )
    host_mask[:, 9, 7] = False
    host_mask[:, 11, 9] = False
    host_mask[:, 11, 11] = False
    mask = task_address_attention_mask(
        layout,
        host_mask=host_mask,
        state_slices=state,
    )[0]
    assert not mask[9, 7]
    assert not mask[11, 9]
    assert not mask[11, 11]


def test_control_block_is_causal_without_serialized_state() -> None:
    roles = torch.tensor(
        [[
            int(TaskAddressRole.CONTROL),
            int(TaskAddressRole.CONTROL),
            int(TaskAddressRole.ACTION),
        ]]
    )
    layout = TaskAddressTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    mask = task_address_attention_mask(
        layout,
        host_mask=torch.ones(1, 3, 3, dtype=torch.bool),
        control_slice=slice(0, 2),
    )[0]
    assert mask[0, 0]
    assert not mask[0, 1]
    assert mask[1, 0]
    assert mask[1, 1]

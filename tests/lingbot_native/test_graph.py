from __future__ import annotations

import torch

from picf_next.lingbot_native.graph import (
    NativeRole,
    NativeTokenLayout,
    native_attention_mask,
    native_layerwise_history_mask,
    native_layerwise_prior_history_mask,
    native_layerwise_prior_trace_mask,
    native_role_permissions,
    posterior_adoption_action_key_visibility,
    posterior_adoption_attention_mask,
    transitive_information_paths,
    validate_native_causality,
)


def test_repeated_layer_closure_blocks_task_action_and_sensor_from_prior() -> None:
    validate_native_causality()
    paths = transitive_information_paths()
    for source in (
        NativeRole.SENSOR,
        NativeRole.LANGUAGE,
        NativeRole.HOST_AUX,
        NativeRole.POSTERIOR,
        NativeRole.ACTION,
        NativeRole.PREDICT,
        NativeRole.MATCH,
    ):
        assert not paths[int(source), int(NativeRole.PRIOR)]
    for source in (
        NativeRole.LANGUAGE,
        NativeRole.HOST_AUX,
        NativeRole.ACTION,
        NativeRole.PREDICT,
        NativeRole.MATCH,
    ):
        assert not paths[int(source), int(NativeRole.POSTERIOR)]
    assert paths[int(NativeRole.PRIOR), int(NativeRole.PREDICT)]
    assert paths[int(NativeRole.POSTERIOR), int(NativeRole.PREDICT)]
    assert paths[int(NativeRole.MATCH), int(NativeRole.ACTION)]
    assert not paths[int(NativeRole.MATCH), int(NativeRole.POSTERIOR)]


def test_role_permission_matrix_matches_the_frozen_information_graph_exactly() -> None:
    """Catch both newly opened leak paths and silently removed required paths."""

    expected = {
        NativeRole.SENSOR: {NativeRole.SENSOR},
        NativeRole.LANGUAGE: {
            NativeRole.SENSOR,
            NativeRole.LANGUAGE,
            NativeRole.PRIOR,
            NativeRole.POSTERIOR,
        },
        NativeRole.HOST_AUX: {
            NativeRole.SENSOR,
            NativeRole.LANGUAGE,
            NativeRole.HOST_AUX,
        },
        NativeRole.CONTROL: {NativeRole.CONTROL},
        NativeRole.PRIOR: {NativeRole.CONTROL, NativeRole.PRIOR},
        NativeRole.POSTERIOR: {
            NativeRole.SENSOR,
            NativeRole.CONTROL,
            NativeRole.PRIOR,
            NativeRole.POSTERIOR,
        },
        NativeRole.ACTION: {
            NativeRole.SENSOR,
            NativeRole.LANGUAGE,
            NativeRole.HOST_AUX,
            NativeRole.CONTROL,
            NativeRole.PRIOR,
            NativeRole.POSTERIOR,
            NativeRole.ACTION,
            NativeRole.MATCH,
        },
        NativeRole.PREDICT: {NativeRole.PREDICT},
        NativeRole.MATCH: {
            NativeRole.SENSOR,
            NativeRole.LANGUAGE,
            NativeRole.POSTERIOR,
            NativeRole.MATCH,
        },
    }
    permissions = native_role_permissions()
    assert permissions.shape == (len(NativeRole), len(NativeRole))
    for query in NativeRole:
        observed = {key for key in NativeRole if bool(permissions[int(query), int(key)].item())}
        assert observed == expected[query]


def test_layer_mask_enforces_triangular_physical_graph_and_causal_controls() -> None:
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.CONTROL),
                int(NativeRole.CONTROL),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.ACTION),
            ]
        ]
    )
    layout = NativeTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    mask = native_attention_mask(
        layout,
        host_mask=torch.ones(1, 7, 7, dtype=torch.bool),
        control_slice=slice(2, 4),
    )[0]
    assert not mask[0, 1]
    assert not mask[4, 0]
    assert not mask[4, 1]
    assert not mask[4, 5]
    assert mask[5, 0]
    assert not mask[5, 1]
    assert mask[5, 4]
    assert not mask[2, 3]
    assert mask[3, 2]
    assert mask[6, 0]
    assert mask[6, 1]
    assert mask[6, 4]
    assert mask[6, 5]


def test_posterior_adoption_route_closes_direct_and_language_scene_bypasses() -> None:
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.HOST_AUX),
                int(NativeRole.CONTROL),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.MATCH),
                int(NativeRole.ACTION),
            ],
            [
                int(NativeRole.SENSOR),
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.HOST_AUX),
                int(NativeRole.CONTROL),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.MATCH),
                int(NativeRole.ACTION),
            ],
        ]
    )
    layout = NativeTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    base = native_attention_mask(
        layout,
        host_mask=torch.ones(2, 9, 9, dtype=torch.bool),
    )
    direct = torch.zeros_like(roles, dtype=torch.bool)
    direct[:, 1] = True
    routed = posterior_adoption_attention_mask(
        base,
        layout=layout,
        enabled=torch.tensor([True, False]),
        direct_action_visible=direct,
    )

    action = 8
    language = 2
    assert not routed[0, action, 0]
    assert routed[0, action, 1]
    assert routed[0, action, language]
    assert not routed[0, action, 3]
    assert routed[0, action, 4]
    assert not routed[0, action, 5]
    assert routed[0, action, 6]
    assert not routed[0, action, 7]
    assert not routed[0, language, 0]
    assert not routed[0, language, 1]
    assert not routed[0, 1, 0]
    assert routed[0, 1, 1]
    assert torch.equal(routed[1], base[1])


def test_posterior_adoption_route_closes_repeated_layer_scene_bypass() -> None:
    roles = torch.tensor(
        [[
            int(NativeRole.SENSOR),
            int(NativeRole.SENSOR),
            int(NativeRole.LANGUAGE),
            int(NativeRole.HOST_AUX),
            int(NativeRole.CONTROL),
            int(NativeRole.PRIOR),
            int(NativeRole.POSTERIOR),
            int(NativeRole.MATCH),
            int(NativeRole.ACTION),
        ]]
    )
    layout = NativeTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    direct = torch.zeros_like(roles, dtype=torch.bool)
    direct[:, 1] = True
    routed = posterior_adoption_attention_mask(
        native_attention_mask(
            layout,
            host_mask=torch.ones(1, roles.shape[1], roles.shape[1], dtype=torch.bool),
        ),
        layout=layout,
        enabled=torch.tensor([True]),
        direct_action_visible=direct,
    )[0]

    # Convert query<-key attention into source->sink information flow and
    # remove POSTERIOR. No non-direct scene token may then reach ACTION through
    # any number of shared layers; direct proprioception remains available.
    flow = routed.T.clone()
    posterior = 6
    flow[posterior] = False
    flow[:, posterior] = False
    closure = flow.clone()
    for intermediate in range(closure.shape[0]):
        closure |= (
            closure[:, intermediate].unsqueeze(1)
            & closure[intermediate].unsqueeze(0)
        )
    assert not closure[0, 8]
    assert closure[1, 8]


def test_posterior_adoption_action_cache_visibility_matches_training_route() -> None:
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.SENSOR),
                int(NativeRole.LANGUAGE),
                int(NativeRole.HOST_AUX),
                int(NativeRole.CONTROL),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.MATCH),
            ]
        ]
    )
    valid = torch.ones_like(roles, dtype=torch.bool)
    layout = NativeTokenLayout(roles=roles, valid=valid)
    direct = torch.zeros_like(valid)
    direct[:, 1] = True
    visible = posterior_adoption_action_key_visibility(
        layout,
        enabled=torch.tensor([True]),
        direct_action_visible=direct,
    )
    assert torch.equal(
        visible,
        torch.tensor([[False, True, True, False, True, False, True, False]]),
    )


def test_posterior_adoption_route_rejects_non_sensor_direct_evidence() -> None:
    roles = torch.tensor([[int(NativeRole.SENSOR), int(NativeRole.LANGUAGE)]])
    layout = NativeTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    direct = torch.tensor([[False, True]])
    try:
        posterior_adoption_action_key_visibility(
            layout,
            enabled=torch.tensor([True]),
            direct_action_visible=direct,
        )
    except ValueError as error:
        assert "only SENSOR" in str(error)
    else:
        raise AssertionError("non-sensor direct evidence was accepted")


def test_layerwise_history_exposes_only_paired_prior_and_posterior_to_memory() -> None:
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.PRIOR),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.ACTION),
            ],
            [
                int(NativeRole.SENSOR),
                int(NativeRole.PRIOR),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.ACTION),
            ],
        ]
    )
    valid = torch.ones_like(roles, dtype=torch.bool)
    valid[0, 2] = False
    layout = NativeTokenLayout(roles=roles, valid=valid)
    mask = native_layerwise_history_mask(
        layout,
        prior_slice=slice(1, 3),
        posterior_slice=slice(3, 5),
        capacity=2,
        previous_memory_valid=torch.tensor([True, False]),
    )
    expected = torch.zeros(2, 6, 2, dtype=torch.bool)
    expected[0, 1, 0] = True
    expected[0, 3, 0] = True
    expected[0, 4, 1] = True
    assert torch.equal(mask, expected)


def test_unified_prior_history_exposes_previous_posterior_only_to_paired_prior() -> None:
    roles = torch.tensor(
        [
            [
                int(NativeRole.CONTROL),
                int(NativeRole.PRIOR),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.ACTION),
            ],
            [
                int(NativeRole.CONTROL),
                int(NativeRole.PRIOR),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.ACTION),
            ],
        ]
    )
    layout = NativeTokenLayout(roles=roles, valid=torch.ones_like(roles, dtype=torch.bool))
    mask = native_layerwise_prior_history_mask(
        layout,
        prior_slice=slice(1, 3),
        capacity=2,
        previous_memory_valid=torch.tensor([True, False]),
    )

    expected = torch.zeros(2, 6, 2, dtype=torch.bool)
    expected[0, 1, 0] = True
    expected[0, 2, 1] = True
    assert torch.equal(mask, expected)
    assert not mask[:, 3:].any()


def test_unified_correction_trace_is_paired_to_rows_and_visible_to_action() -> None:
    roles = torch.tensor(
        [
            [
                int(NativeRole.SENSOR),
                int(NativeRole.PRIOR),
                int(NativeRole.PRIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.POSTERIOR),
                int(NativeRole.ACTION),
                int(NativeRole.ACTION),
            ]
        ]
    )
    valid = torch.tensor([[True, True, True, True, True, True, False]])
    layout = NativeTokenLayout(roles=roles, valid=valid)
    mask = native_layerwise_prior_trace_mask(
        layout,
        prior_slice=slice(1, 3),
        posterior_slice=slice(3, 5),
        capacity=2,
    )

    expected = torch.zeros(1, 7, 2, dtype=torch.bool)
    expected[0, 1, 0] = True
    expected[0, 2, 1] = True
    expected[0, 3, 0] = True
    expected[0, 4, 1] = True
    expected[0, 5, :] = True
    assert torch.equal(mask, expected)

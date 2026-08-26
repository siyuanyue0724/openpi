from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from picf_next.hosts.lingbot_unified import (  # noqa: E402
    LingBotHostContract,
    LingBotUnifiedBeliefGraph,
    LingBotUnifiedContext,
    LingBotUnifiedGraphConfig,
    install_lingbot_unified_belief_graph,
)
from picf_next.hosts.lingbot_unified_training import (  # noqa: E402
    lingbot_row_prediction_term,
)
from picf_next.unified.codec import BeliefCodecConfig  # noqa: E402
from picf_next.unified.coreference import GroupedRelationEvidence  # noqa: E402
from picf_next.unified.graph import TokenRole  # noqa: E402
from picf_next.unified.lifecycle import posterior_expected_age  # noqa: E402
from picf_next.unified.predictive import (  # noqa: E402
    ROW_SUMMARY_TARGET,
    PredictionQueryRequest,
    PredictiveTargetProvenance,
    make_predictive_target,
    predictive_source_batch_digest,
)
from picf_next.unified.state import (  # noqa: E402
    GeometrySchema,
    UnifiedBeliefState,
    empty_belief_state,
)

GEOMETRY_SCHEMA = GeometrySchema(
    names=("x", "y"),
    units=("normalized", "normalized"),
    frame="camera",
)
ROW_QUERY_SCHEMA_DIGEST = "c" * 64
SOURCE_BATCH_DIGEST = predictive_source_batch_digest(("episode-a",), (4,))


def _belief(*, offset: float = 0.0) -> UnifiedBeliefState:
    return UnifiedBeliefState(
        content=torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]) + offset,
        lifecycle_log_probs=torch.log_softmax(
            torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, -0.5]]]), dim=-1
        ),
        geometry_mean=torch.tensor([[[0.0, 1.0], [2.0, 3.0]]]),
        geometry_information=torch.eye(2).expand(1, 2, 2, 2).clone(),
        geometry_valid=torch.ones(1, 2, 2, dtype=torch.bool),
        content_log_variance=torch.zeros(1, 2, 1),
        expected_age=torch.ones(1, 2),
        evidence_age=torch.ones(1, 2),
    )


def _graph() -> LingBotUnifiedBeliefGraph:
    return LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(3, 2, 1, 32),
            geometry_schema=GEOMETRY_SCHEMA,
            attention_value_width=32,
            num_layers=3,
            retrieval_tokens=1,
            executed_action_dim=2,
        )
    )


def _predictive_graph() -> LingBotUnifiedBeliefGraph:
    return LingBotUnifiedBeliefGraph(
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(3, 2, 1, 32),
            geometry_schema=GEOMETRY_SCHEMA,
            attention_value_width=32,
            num_layers=3,
            retrieval_tokens=1,
            executed_action_dim=2,
            native_measurement_query_tokens=1,
            native_prediction_query_tokens=2,
            modality_names=("vision", "touch"),
            grouped_assignment_modalities=("touch",),
            modality_reliability=(1.0, 1.0),
        )
    )


def test_graph_rejects_geometry_semantics_that_do_not_match_the_codec() -> None:
    with pytest.raises(ValueError, match="schema width"):
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(3, 2, 1, 32),
            geometry_schema=GeometrySchema(
                names=("x",),
                units=("normalized",),
                frame="camera",
            ),
            attention_value_width=32,
            num_layers=3,
        )


def test_graph_rejects_a_host_that_cannot_hold_exact_action_pairs() -> None:
    with pytest.raises(ValueError, match=r"2 \* canonical_width"):
        LingBotUnifiedGraphConfig(
            codec=BeliefCodecConfig(3, 2, 1, 16),
            geometry_schema=GEOMETRY_SCHEMA,
            attention_value_width=16,
            num_layers=3,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("num_layers", 3.5, "counts and dimensions"),
        ("modality_reliability", (float("nan"),), "must be finite"),
        ("relation_adoption_init", float("inf"), "must be finite"),
        ("ci_step_size", float("nan"), "must be finite"),
    ),
)
def test_graph_rejects_nonintegral_or_nonfinite_static_controls(
    field: str,
    value: object,
    message: str,
) -> None:
    kwargs = {
        "codec": BeliefCodecConfig(3, 2, 1, 32),
        "geometry_schema": GEOMETRY_SCHEMA,
        "attention_value_width": 32,
        "num_layers": 3,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, ValueError), match=message):
        LingBotUnifiedGraphConfig(**kwargs)


def _context(*, sensors: bool = True) -> LingBotUnifiedContext:
    roles = torch.tensor([[int(TokenRole.SENSOR), int(TokenRole.SENSOR), int(TokenRole.LANGUAGE)]])
    valid = torch.ones(1, 3, dtype=torch.bool)
    footprint = torch.tensor([[0.4, 0.6, 0.0]]) if sensors else torch.zeros(1, 3)
    if not sensors:
        roles = torch.full_like(roles, int(TokenRole.LANGUAGE))
    return LingBotUnifiedContext(
        previous_posterior=_belief(),
        native_roles=roles,
        native_valid=valid,
        native_footprint=footprint,
        native_modality_ids=torch.where(
            roles == int(TokenRole.SENSOR),
            torch.zeros_like(roles),
            torch.full_like(roles, -1),
        ),
        modality_geometry_valid=torch.ones(1, 1, 2, 2, dtype=torch.bool),
        elapsed_time=torch.ones(1),
        previous_executed_action=torch.tensor([[0.25, -0.5]]),
        previous_action_valid=torch.ones(1, dtype=torch.bool),
        birth_proposal_noise=torch.tensor([[[1.0, 0.0, -1.0], [-1.0, 0.5, 1.0]]]),
    )


def _prepare(
    graph: LingBotUnifiedBeliefGraph,
    context: LingBotUnifiedContext | None,
):
    torch.manual_seed(3)
    prefix = torch.randn(1, 3, 32)
    action = torch.randn(1, 2, 16)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone()
    visual = torch.tensor([[True, True, False]])
    result = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )
    return (prefix, action, mask, positions, visual), result


def _predictive_context() -> LingBotUnifiedContext:
    request = PredictionQueryRequest(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        horizon=0,
        query_schema_digest=ROW_QUERY_SCHEMA_DIGEST,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        source_batch_size=1,
    )
    roles = torch.tensor(
        [
            [
                int(TokenRole.SENSOR),
                int(TokenRole.LANGUAGE),
                int(TokenRole.MEASUREMENT_QUERY),
                int(TokenRole.HOST_FUTURE_QUERY),
                int(TokenRole.HOST_FUTURE_QUERY),
            ]
        ]
    )
    return replace(
        _context(),
        native_roles=roles,
        native_valid=torch.ones(1, 5, dtype=torch.bool),
        native_footprint=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
        native_modality_ids=torch.tensor([[0, -1, -1, -1, -1]]),
        modality_geometry_valid=torch.ones(1, 2, 2, 2, dtype=torch.bool),
        prediction_request=request,
    )


def _prepare_predictive(graph: LingBotUnifiedBeliefGraph, context: LingBotUnifiedContext):
    torch.manual_seed(31)
    prefix = torch.randn(1, 5, 32)
    action = torch.randn(1, 2, 16)
    mask = torch.ones(1, 7, 7, dtype=torch.bool)
    positions = torch.arange(7).reshape(1, 1, 7).expand(3, 1, 7).clone()
    visual = torch.tensor([[True, False, False, False, False]])
    return graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=visual,
        context=context,
    )


def test_context_requires_unit_footprint_per_available_modality() -> None:
    graph = _graph()
    context = _context()
    context.native_footprint = torch.tensor([[0.4, 0.5, 0.0]])
    with pytest.raises(ValueError, match="unit footprint mass"):
        _prepare(graph, context)

    context = _context()
    context.native_valid = torch.tensor([[True, False, True]])
    with pytest.raises(ValueError, match="invalid native tokens"):
        _prepare(graph, context)


def _observe_for_state_write(
    graph: LingBotUnifiedBeliefGraph,
    inputs: list[torch.Tensor | None],
    runtime,
):
    total = inputs[0].shape[1] + (0 if inputs[1] is None else inputs[1].shape[1])
    torch.manual_seed(18)
    return graph.observe_joint_qkv(
        layer_index=graph.config.penultimate_layer,
        query_states=torch.randn(1, total, 4, 8),
        key_states=torch.randn(1, total, 2, 8),
        value_states=torch.randn(1, total, 2, 8),
        runtime=runtime,
    )


def test_no_context_is_exact_object_identity_for_baseline_parity() -> None:
    graph = _graph()
    original, prepared = _prepare(graph, None)
    inputs, mask, positions, visual, runtime = prepared
    assert inputs[0] is original[0]
    assert inputs[1] is original[1]
    assert mask is original[2]
    assert positions is original[3]
    assert visual is original[4]
    assert runtime is None


def test_prepare_inserts_roles_before_action_and_extends_all_host_contracts() -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, mask, positions, visual, runtime = prepared
    assert runtime is not None
    # native 3 + transition 1 + prior 2 + post 2 + null 1 + retrieval 1
    assert inputs[0].shape == (1, 10, 32)
    assert inputs[1].shape == (1, 2, 16)
    assert mask.shape == (1, 12, 12)
    assert positions.shape == (3, 1, 12)
    assert visual.shape == (1, 10)
    assert visual.sum() == 2
    assert context.expanded_prefix_valid.shape == (1, 10)
    assert context.expanded_prefix_position_ids.shape == (3, 1, 10)
    assert context.expanded_cache_valid.shape == (1, 12)
    assert context.expanded_cache_position_ids.shape == (3, 1, 12)
    assert context.expanded_action_cache_visible.shape == (1, 12)
    assert not context.expanded_action_cache_visible[:, runtime.transition_index].any()
    assert context.expanded_action_cache_visible[:, runtime.posterior_slice].all()
    assert not context.expanded_action_cache_visible[:, runtime.context_index].any()
    assert runtime.layout.roles[0, runtime.prefix_count] == int(TokenRole.CURRENT_STATE)
    assert runtime.layout.roles[0, runtime.prefix_count + 1] == int(TokenRole.ACTION)
    # Every exchangeable row within a role has the same MRoPE coordinates.
    torch.testing.assert_close(
        positions[:, :, runtime.prior_slice.start],
        positions[:, :, runtime.prior_slice.start + 1],
    )
    torch.testing.assert_close(
        positions[:, :, runtime.posterior_slice.start],
        positions[:, :, runtime.posterior_slice.start + 1],
    )
    # Prior cannot read current sensors; posterior can. Physical rows cannot read language.
    assert not mask[0, runtime.prior_slice, :2].any()
    assert mask[0, runtime.prior_slice, runtime.transition_index].all()
    assert mask[0, runtime.posterior_slice, :2].all()
    assert mask[0, runtime.posterior_slice, runtime.prefix_count].all()
    assert not mask[0, runtime.posterior_slice, runtime.prefix_count + 1].any()
    assert not mask[0, runtime.posterior_slice, runtime.transition_index].any()
    assert not mask[0, runtime.posterior_slice, runtime.context_index].any()
    assert mask[0, runtime.context_index, :2].all()
    assert mask[0, runtime.context_index, runtime.prior_slice].all()
    assert mask[0, runtime.context_index, runtime.prefix_count]
    assert not mask[0, runtime.prior_slice.start : runtime.context_index + 1, 2].any()
    assert mask[0, -1, runtime.posterior_slice].all()


def test_row_prediction_query_is_exchangeable_and_cannot_read_target_inputs() -> None:
    graph = _predictive_graph()
    context = _predictive_context()
    inputs, mask, positions, _, runtime = _prepare_predictive(graph, context)
    assert runtime is not None and runtime.prediction_slice is not None
    assert inputs[0].shape == (1, 14, 32)
    assert inputs[1].shape == (1, 2, 16)
    query_slice = runtime.prediction_slice
    torch.testing.assert_close(
        inputs[0][:, query_slice.start],
        inputs[0][:, query_slice.start + 1],
    )
    torch.testing.assert_close(
        positions[:, :, query_slice.start],
        positions[:, :, query_slice.start + 1],
    )
    expected_seed = inputs[0][:, 3:5].mean(dim=1, keepdim=True).expand(-1, 2, -1)
    torch.testing.assert_close(inputs[0][:, query_slice], expected_seed)
    # Current measurement is deploy-visible; future and PICF loss queries are not.
    action_query = runtime.prefix_count + 1
    assert mask[0, action_query, 2]
    assert not mask[0, action_query, 3:5].any()
    assert not mask[0, action_query, query_slice].any()
    assert not context.expanded_action_cache_visible[:, 3:5].any()
    assert not context.expanded_action_cache_visible[:, query_slice].any()
    assert mask[0, 3, 0]
    assert mask[0, 3, 1]
    assert mask[0, 3, 2]

    for row in range(2):
        query = query_slice.start + row
        assert mask[0, query, runtime.prior_slice.start + row]
        assert mask[0, query, runtime.posterior_slice.start + row]
        assert not mask[0, query, runtime.prior_slice.start + (1 - row)]
        assert not mask[0, query, runtime.posterior_slice.start + (1 - row)]
        assert mask[0, query, runtime.transition_index]
        assert mask[0, query, runtime.prefix_count]  # current source-time robot state
        assert mask[0, query, query]
        assert not mask[0, query, 0]  # withheld/current physical sensor
        assert not mask[0, query, 1]  # language
        assert not mask[0, query, 2]  # current-measurement query
        assert not mask[0, query, 3:5].any()  # native future queries
        assert not mask[0, query, runtime.context_index]
        assert not mask[0, query, runtime.retrieval_slice].any()
        assert not mask[0, query, query_slice.start + (1 - row)]


def test_native_history_is_rejected_as_a_graph_owned_temporal_role() -> None:
    graph = _predictive_graph()
    context = _predictive_context()
    roles = context.native_roles.clone()
    roles[0, 0] = int(TokenRole.HISTORY)
    context = replace(
        context,
        prediction_request=PredictionQueryRequest(
            modality="touch",
            target_kind=ROW_SUMMARY_TARGET,
            horizon=1,
            query_schema_digest=ROW_QUERY_SCHEMA_DIGEST,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            source_batch_size=1,
        ),
        native_roles=roles,
        native_footprint=torch.zeros(1, 5),
        native_modality_ids=torch.full((1, 5), -1, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="role owned by the unified graph"):
        _prepare_predictive(graph, context)


def test_no_prediction_request_adds_no_query_tokens_or_persistent_state() -> None:
    graph = _predictive_graph()
    context = replace(_predictive_context(), prediction_request=None)
    before = context.previous_posterior.serialize()
    inputs, _, _, _, runtime = _prepare_predictive(graph, context)
    assert runtime is not None and runtime.prediction_slice is None
    # native 5 + transition/prior/posterior/context/retrieval = 12
    assert inputs[0].shape == (1, 12, 32)
    assert context.row_prediction_hidden is None
    assert context.previous_posterior.serialize() == before


def test_final_row_prediction_executes_a_provenance_bound_objective() -> None:
    graph = _predictive_graph()
    context = _predictive_context()
    inputs, _, _, _, runtime = _prepare_predictive(graph, context)
    assert runtime is not None and runtime.prediction_slice is not None
    assert inputs[0] is not None
    inputs[0].retain_grad()
    graph.after_layer(
        layer_index=graph.config.num_layers - 1,
        outputs_embeds=inputs,
        runtime=runtime,
    )
    assert context.row_prediction_hidden is not None
    provenance = PredictiveTargetProvenance(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        target_data_digest="a" * 64,
        target_model_digest="d" * 64,
        assignment_schema_digest="b" * 64,
        query_schema_digest=ROW_QUERY_SCHEMA_DIGEST,
        validity_semantics="positive detached target support",
        optimizer_step=4,
    )
    target = make_predictive_target(
        "touch",
        torch.randn_like(context.row_prediction_hidden),
        torch.ones(1, 2, dtype=torch.bool),
        horizon=0,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        target_data_digest=provenance.target_data_digest,
        encoder_digest=provenance.target_model_digest,
        target_kind=ROW_SUMMARY_TARGET,
        assignment_digest=provenance.assignment_schema_digest,
        query_schema_digest=ROW_QUERY_SCHEMA_DIGEST,
        validity_semantics="positive detached target support",
        provenance_digest=provenance.digest,
    )
    assert context.prediction_request is not None
    term = lingbot_row_prediction_term(
        context,
        target,
        provenance,
        weight=0.25,
    )
    with pytest.raises(ValueError, match="provenance digest"):
        lingbot_row_prediction_term(
            context,
            target,
            replace(provenance, optimizer_step=5),
            weight=0.25,
        )
    with pytest.raises(ValueError, match="assignment schema"):
        lingbot_row_prediction_term(
            context,
            replace(target, assignment_digest="e" * 64),
            provenance,
            weight=0.25,
        )
    assert term.name == "xmod/touch"
    term.normalized().backward()
    assert inputs[0].grad is not None
    assert inputs[0].grad[:, runtime.prediction_slice].abs().sum() > 0
    assert graph.predictive_modality_embedding.grad is not None
    assert graph.predictive_horizon_projection.weight.grad is not None


def test_row_prediction_query_and_output_are_simultaneously_permutation_equivariant() -> None:
    graph = _predictive_graph()
    context = _predictive_context()
    inputs, _, _, _, runtime = _prepare_predictive(graph, context)
    assert runtime is not None and runtime.prediction_slice is not None
    permutation = torch.tensor([1, 0])
    permuted_context = replace(
        _predictive_context(),
        previous_posterior=context.previous_posterior.permute_rows(permutation),
        birth_proposal_noise=context.birth_proposal_noise.index_select(1, permutation),
        modality_geometry_valid=context.modality_geometry_valid.index_select(2, permutation),
    )
    permuted_inputs, _, _, _, permuted_runtime = _prepare_predictive(graph, permuted_context)
    assert permuted_runtime is not None and permuted_runtime.prediction_slice is not None
    torch.testing.assert_close(
        permuted_inputs[0][:, permuted_runtime.prior_slice],
        inputs[0][:, runtime.prior_slice].index_select(1, permutation),
    )
    torch.testing.assert_close(
        permuted_inputs[0][:, permuted_runtime.posterior_slice],
        inputs[0][:, runtime.posterior_slice].index_select(1, permutation),
    )
    # Query seeds are exchangeable; the corresponding mask edges carry row identity.
    torch.testing.assert_close(
        permuted_inputs[0][:, permuted_runtime.prediction_slice],
        inputs[0][:, runtime.prediction_slice].index_select(1, permutation),
    )


def test_native_vision_metadata_is_inferred_without_a_second_image_forward() -> None:
    graph = _graph()
    context = replace(
        _context(),
        native_roles=None,
        native_valid=None,
        native_footprint=None,
        native_modality_ids=None,
    )
    _prepare(graph, context)
    assert context.native_roles is not None
    assert context.native_valid is not None
    assert context.native_footprint is not None
    assert context.native_modality_ids is not None
    assert context.native_roles.tolist() == [
        [int(TokenRole.SENSOR), int(TokenRole.SENSOR), int(TokenRole.LANGUAGE)]
    ]
    torch.testing.assert_close(context.native_footprint, torch.tensor([[0.5, 0.5, 0.0]]))
    assert context.native_modality_ids.tolist() == [[0, 0, -1]]


def test_native_query_tail_is_split_into_measurement_and_prediction_roles() -> None:
    graph = _predictive_graph()
    context = replace(
        _predictive_context(),
        native_roles=None,
        native_valid=None,
        native_footprint=None,
        native_modality_ids=None,
    )
    _prepare_predictive(graph, context)
    assert context.native_roles is not None
    assert context.native_roles.tolist() == [
        [
            int(TokenRole.SENSOR),
            int(TokenRole.LANGUAGE),
            int(TokenRole.MEASUREMENT_QUERY),
            int(TokenRole.HOST_FUTURE_QUERY),
            int(TokenRole.HOST_FUTURE_QUERY),
        ]
    ]


def test_explicit_metadata_cannot_spoof_a_sensor_as_a_prediction_query() -> None:
    graph = _predictive_graph()
    context = _predictive_context()
    spoofed_roles = context.native_roles.clone()
    spoofed_roles[0, 0] = int(TokenRole.HOST_FUTURE_QUERY)
    spoofed_roles[0, 3] = int(TokenRole.SENSOR)
    spoofed = replace(
        context,
        native_roles=spoofed_roles,
        native_footprint=torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0]]),
        native_modality_ids=torch.tensor([[-1, -1, -1, 0, -1]]),
    )
    with pytest.raises(ValueError, match="tail contract"):
        _prepare_predictive(graph, spoofed)


def test_cross_modal_query_rejects_an_input_that_still_contains_target_modality() -> None:
    graph = _predictive_graph()
    context = _predictive_context()
    visible_target = replace(
        context,
        native_modality_ids=torch.tensor([[1, -1, -1, -1, -1]]),
        native_group_ids=torch.tensor([[3, -1, -1, -1, -1]]),
    )
    with pytest.raises(ValueError, match="not withheld"):
        _prepare_predictive(graph, visible_target)


def test_cross_modal_query_requires_a_valid_source_modality_per_sample() -> None:
    graph = _predictive_graph()
    context = replace(
        _predictive_context(),
        native_valid=torch.tensor([[False, True, True, True, True]]),
        native_footprint=torch.zeros(1, 5),
        native_modality_ids=torch.full((1, 5), -1, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="valid non-target physical sensor"):
        _prepare_predictive(graph, context)


def test_touch_group_is_required_and_shares_one_assignment_distribution() -> None:
    graph = _predictive_graph()
    future_request = PredictionQueryRequest(
        modality="touch",
        target_kind=ROW_SUMMARY_TARGET,
        horizon=1,
        query_schema_digest=ROW_QUERY_SCHEMA_DIGEST,
        source_batch_digest=SOURCE_BATCH_DIGEST,
        source_batch_size=1,
    )
    roles = torch.tensor(
        [
            [
                int(TokenRole.SENSOR),
                int(TokenRole.SENSOR),
                int(TokenRole.MEASUREMENT_QUERY),
                int(TokenRole.HOST_FUTURE_QUERY),
                int(TokenRole.HOST_FUTURE_QUERY),
            ]
        ]
    )
    ungrouped = replace(
        _predictive_context(),
        prediction_request=future_request,
        native_roles=roles,
        native_footprint=torch.tensor([[0.4, 0.6, 0.0, 0.0, 0.0]]),
        native_modality_ids=torch.tensor([[1, 1, -1, -1, -1]]),
    )
    with pytest.raises(ValueError, match="valid touch tokens require"):
        _prepare_predictive(graph, ungrouped)

    grouped = replace(ungrouped, native_group_ids=torch.tensor([[9, 9, -1, -1, -1]]))
    inputs, _, _, _, runtime = _prepare_predictive(graph, grouped)
    assert runtime is not None and inputs[0] is not None and inputs[1] is not None
    assert runtime.assignment_group_ids is grouped.native_group_ids
    total = inputs[0].shape[1] + inputs[1].shape[1]
    graph.observe_joint_qkv(
        layer_index=0,
        query_states=torch.randn(1, total, 4, 8),
        key_states=torch.randn(1, total, 2, 8),
        value_states=torch.randn(1, total, 2, 8),
        runtime=runtime,
    )
    assert grouped.last_coreference is not None
    torch.testing.assert_close(
        grouped.last_coreference.evidence.responsibilities[:, 0],
        grouped.last_coreference.evidence.responsibilities[:, 1],
    )


def test_one_physical_token_group_cannot_mix_modalities() -> None:
    graph = _predictive_graph()
    context = replace(
        _predictive_context(),
        prediction_request=PredictionQueryRequest(
            modality="touch",
            target_kind=ROW_SUMMARY_TARGET,
            horizon=1,
            query_schema_digest=ROW_QUERY_SCHEMA_DIGEST,
            source_batch_digest=SOURCE_BATCH_DIGEST,
            source_batch_size=1,
        ),
        native_roles=torch.tensor(
            [
                [
                    int(TokenRole.SENSOR),
                    int(TokenRole.SENSOR),
                    int(TokenRole.MEASUREMENT_QUERY),
                    int(TokenRole.HOST_FUTURE_QUERY),
                    int(TokenRole.HOST_FUTURE_QUERY),
                ]
            ]
        ),
        native_footprint=torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0]]),
        native_modality_ids=torch.tensor([[0, 1, -1, -1, -1]]),
        native_group_ids=torch.tensor([[4, 4, -1, -1, -1]]),
    )
    with pytest.raises(ValueError, match="cannot mix modalities"):
        _prepare_predictive(graph, context)


def test_prediction_query_is_rejected_by_the_evaluation_graph() -> None:
    graph = _predictive_graph().eval()
    with pytest.raises(ValueError, match="training-only"):
        _prepare_predictive(graph, _predictive_context())


def test_executed_action_enters_only_the_causal_transition_token() -> None:
    graph = _graph()
    with torch.no_grad():
        graph.transition_projection.weight.zero_()
        graph.transition_projection.bias.zero_()
        graph.transition_projection.weight[0, 0] = 1.0
    left_context = _context()
    right_context = replace(
        _context(),
        previous_executed_action=torch.tensor([[9.0, -0.5]]),
    )
    _, left = _prepare(graph, left_context)
    _, right = _prepare(graph, right_context)
    left_inputs, left_mask, _, _, left_runtime = left
    right_inputs, _, _, _, right_runtime = right
    assert left_runtime is not None and right_runtime is not None
    difference = left_inputs[0] != right_inputs[0]
    changed_tokens = difference.any(dim=-1).nonzero(as_tuple=False)[:, 1].tolist()
    assert changed_tokens == [left_runtime.transition_index]
    assert left_mask[0, left_runtime.prior_slice, left_runtime.transition_index].all()
    assert not left_mask[0, left_runtime.posterior_slice, left_runtime.transition_index].any()
    assert not left_mask[0, -1, left_runtime.transition_index]


def test_empty_rows_receive_exchangeable_ephemeral_birth_proposals() -> None:
    graph = _graph()
    empty = empty_belief_state(
        batch_size=1,
        capacity=2,
        content_dim=3,
        geometry_dim=2,
        uncertainty_dim=1,
    )
    context = replace(_context(), previous_posterior=empty)
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    assert runtime.assignment_group_ids is None
    prior_content = inputs[0][..., :3][:, runtime.prior_slice]
    assert not torch.equal(prior_content[:, 0], prior_content[:, 1])

    permutation = torch.tensor([1, 0])
    permuted_context = replace(
        context,
        previous_posterior=empty.permute_rows(permutation),
        birth_proposal_noise=context.birth_proposal_noise.index_select(1, permutation),
    )
    _, permuted = _prepare(graph, permuted_context)
    permuted_inputs, _, _, _, permuted_runtime = permuted
    assert permuted_runtime is not None
    torch.testing.assert_close(
        permuted_inputs[0][..., :3][:, permuted_runtime.prior_slice],
        prior_content.index_select(1, permutation),
    )


def test_context_rejects_modality_and_device_contract_violations() -> None:
    graph = _graph()
    materialized_groups = torch.full((1, 3), -1, dtype=torch.long)
    invalid_ids = replace(
        _context(),
        native_modality_ids=torch.tensor([[0, 0, 0]]),
        native_group_ids=materialized_groups,
    )
    with pytest.raises(ValueError, match="only valid sensor"):
        invalid_ids.validate(host_width=32, modality_names=("vision",))
    bad_footprint = replace(
        _context(),
        native_footprint=torch.tensor([[float("nan"), 0.0, 0.0]]),
        native_group_ids=materialized_groups,
    )
    with pytest.raises(ValueError, match="finite"):
        bad_footprint.validate(host_width=32, modality_names=("vision",))
    assert graph.config.modality_names == ("vision",)


def test_shared_qk_relation_is_gqa_compatible_and_zero_gate_is_exact_parity() -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    total = inputs[0].shape[1] + inputs[1].shape[1]
    torch.manual_seed(8)
    query = torch.randn(1, total, 4, 8)
    key = torch.randn(1, total, 2, 8)
    value = torch.randn(1, total, 2, 8)
    runtime = graph.observe_joint_qkv(
        layer_index=0,
        query_states=query,
        key_states=key,
        value_states=value,
        runtime=runtime,
    )
    assert context.last_coreference is not None
    assert context.last_coreference.evidence.responsibilities.shape == (1, 3, 2)
    attention_output = torch.randn(1, total, 32)
    unchanged = graph.apply_relation_message(
        layer_index=0,
        attention_output=attention_output,
        runtime=runtime,
    )
    assert torch.equal(unchanged, attention_output)

    with torch.no_grad():
        graph.relation_adoption[0] = 1
    changed = graph.apply_relation_message(
        layer_index=0,
        attention_output=attention_output,
        runtime=runtime,
    )
    assert not torch.equal(
        changed[:, runtime.posterior_slice], attention_output[:, runtime.posterior_slice]
    )
    torch.testing.assert_close(
        changed[:, : runtime.posterior_slice.start],
        attention_output[:, : runtime.posterior_slice.start],
    )


def test_final_layer_is_read_only_and_preserves_state_write_coreference() -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    runtime = _observe_for_state_write(graph, inputs, runtime)
    state_write_coreference = context.last_coreference
    assert state_write_coreference is not None
    assert graph.relation_adoption.shape == (graph.config.num_layers - 1,)

    total = inputs[0].shape[1] + inputs[1].shape[1]
    runtime = graph.observe_joint_qkv(
        layer_index=graph.config.num_layers - 1,
        query_states=torch.randn(1, total, 4, 8),
        key_states=torch.randn(1, total, 2, 8),
        value_states=torch.randn(1, total, 2, 8),
        runtime=runtime,
    )
    assert context.last_coreference is state_write_coreference
    assert runtime.relation_message is None
    final_attention = torch.randn(1, total, 32)
    assert (
        graph.apply_relation_message(
            layer_index=graph.config.num_layers - 1,
            attention_output=final_attention,
            runtime=runtime,
        )
        is final_attention
    )


@pytest.mark.parametrize("layer_index", [True, False, 1.0, "1"])
def test_graph_layer_boundary_rejects_non_integer_indices(layer_index: object) -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    total = inputs[0].shape[1] + inputs[1].shape[1]

    with pytest.raises(TypeError, match="Python int"):
        graph.observe_joint_qkv(
            layer_index=layer_index,  # type: ignore[arg-type]
            query_states=torch.randn(1, total, 4, 8),
            key_states=torch.randn(1, total, 2, 8),
            value_states=torch.randn(1, total, 2, 8),
            runtime=runtime,
        )


def test_relation_residual_retains_absolute_physical_support() -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    total = inputs[0].shape[1] + inputs[1].shape[1]
    query = torch.zeros(1, total, 4, 8)
    key = torch.zeros(1, total, 2, 8)
    value = torch.ones(1, total, 2, 8)
    runtime = graph.observe_joint_qkv(
        layer_index=0,
        query_states=query,
        key_states=key,
        value_states=value,
        runtime=runtime,
    )
    assert runtime.relation_message is not None
    assert context.last_coreference is not None
    support = context.last_coreference.evidence.support
    expected = support.unsqueeze(-1).expand_as(runtime.relation_message)
    torch.testing.assert_close(runtime.relation_message, expected)


def test_penultimate_write_reembeds_prior_and_exact_rowwise_action_pairs() -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    runtime = _observe_for_state_write(graph, inputs, runtime)
    outputs, runtime = graph.after_layer(
        layer_index=graph.config.penultimate_layer,
        outputs_embeds=inputs,
        runtime=runtime,
    )
    assert context.predictive_prior is not None
    assert context.posterior is not None
    assert context.final_action_pair is not None
    pair = context.final_action_pair
    torch.testing.assert_close(pair.canonical[:, :2], context.predictive_prior.canonical())
    torch.testing.assert_close(pair.canonical[:, 2:], context.posterior.canonical())
    pair_width = 2 * graph.config.codec.canonical_width
    torch.testing.assert_close(
        outputs[0][:, runtime.prior_slice, : graph.config.codec.canonical_width],
        context.predictive_prior.canonical(),
    )
    torch.testing.assert_close(
        outputs[0][:, runtime.posterior_slice, :pair_width],
        pair.paired_canonical,
    )


def test_penultimate_write_applies_elapsed_time_once_to_deterministic_age() -> None:
    graph = _graph()
    context = replace(_context(), elapsed_time=torch.tensor([2.0]))
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    modified = list(inputs)
    modified[0] = modified[0].clone()
    # Corrupt both age coordinates in the raw prior hidden rows.  They must not
    # become a second learned clock.
    modified[0][:, runtime.prior_slice, graph.config.codec.canonical_width - 2 :] = 99
    runtime = _observe_for_state_write(graph, modified, runtime)
    graph.after_layer(
        layer_index=graph.config.penultimate_layer,
        outputs_embeds=modified,
        runtime=runtime,
    )
    assert context.predictive_prior is not None
    assert context.posterior is not None
    torch.testing.assert_close(
        context.predictive_prior.expected_age,
        posterior_expected_age(
            context.previous_posterior.expected_age,
            context.predictive_prior.lifecycle_log_probs,
            context.elapsed_time[:, None],
        ),
    )
    torch.testing.assert_close(
        context.predictive_prior.evidence_age,
        context.previous_posterior.evidence_age + context.elapsed_time[:, None],
    )
    torch.testing.assert_close(
        context.posterior.expected_age,
        posterior_expected_age(
            context.previous_posterior.expected_age,
            context.posterior.lifecycle_log_probs,
            context.elapsed_time[:, None],
        ),
    )
    assert (context.posterior.evidence_age <= context.predictive_prior.evidence_age).all()


def test_bfloat16_host_boundary_runs_with_fp32_persistent_statistics() -> None:
    graph = _graph().to(torch.bfloat16)
    context = _context()
    prefix = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    action = torch.randn(1, 2, 16, dtype=torch.bfloat16)
    mask = torch.ones(1, 5, 5, dtype=torch.bool)
    positions = torch.arange(5).reshape(1, 1, 5).expand(3, 1, 5).clone()
    inputs, _, _, _, runtime = graph.prepare_joint_inputs(
        inputs_embeds=[prefix, action],
        attention_mask=mask,
        position_ids=positions,
        visual_pos_masks=torch.tensor([[True, True, False]]),
        context=context,
    )
    assert runtime is not None
    total = inputs[0].shape[1] + inputs[1].shape[1]
    runtime = graph.observe_joint_qkv(
        layer_index=graph.config.penultimate_layer,
        query_states=torch.randn(1, total, 4, 8),
        key_states=torch.randn(1, total, 2, 8),
        value_states=torch.randn(1, total, 2, 8),
        runtime=runtime,
    )
    outputs, _ = graph.after_layer(
        layer_index=graph.config.penultimate_layer,
        outputs_embeds=inputs,
        runtime=runtime,
    )
    assert outputs[0].dtype == torch.bfloat16
    assert context.posterior is not None
    for name in context.posterior.__dataclass_fields__:
        value = getattr(context.posterior, name)
        if value.dtype != torch.bool:
            assert value.dtype == torch.float32


def test_absent_physical_observation_is_exactly_an_absent_likelihood_factor() -> None:
    graph = _graph()
    context = _context(sensors=False)
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    modified = list(inputs)
    modified[0] = modified[0].clone()
    modified[0][:, runtime.posterior_slice] += torch.randn_like(
        modified[0][:, runtime.posterior_slice]
    )
    runtime = _observe_for_state_write(graph, modified, runtime)
    graph.after_layer(
        layer_index=graph.config.penultimate_layer,
        outputs_embeds=modified,
        runtime=runtime,
    )
    assert context.predictive_prior is not None
    assert context.posterior is not None
    torch.testing.assert_close(
        context.predictive_prior.expected_age,
        posterior_expected_age(
            context.previous_posterior.expected_age,
            context.predictive_prior.lifecycle_log_probs,
            context.elapsed_time[:, None],
        ),
    )
    torch.testing.assert_close(
        context.predictive_prior.evidence_age,
        context.previous_posterior.evidence_age + context.elapsed_time[:, None],
    )
    for name in context.predictive_prior.__dataclass_fields__:
        torch.testing.assert_close(
            getattr(context.posterior, name),
            getattr(context.predictive_prior, name),
        )


def test_geometry_information_strength_is_weighted_by_soft_object_support() -> None:
    graph = _graph()
    with torch.no_grad():
        graph.measurement_projection.weight.zero_()
        graph.measurement_projection.bias.zero_()
        # [lifecycle odds 2, mean 2, lower-triangle information factor 3]
        graph.measurement_projection.bias[:2] = torch.tensor([3.0, -2.0])
        graph.measurement_projection.bias[2:4] = torch.tensor([8.0, -4.0])
        graph.measurement_projection.bias[4:] = torch.tensor([1.0, 0.0, 1.0])
    prior = _belief()
    raw = replace(
        _belief(offset=5.0),
        content_log_variance=torch.full_like(prior.content_log_variance, 5.0),
    )
    context = _context()

    def fuse(support_value: float) -> UnifiedBeliefState:
        support = torch.full((1, 1, 2), support_value)
        evidence = GroupedRelationEvidence(
            message=torch.zeros(1, 1, 2, 4, 8),
            support=support,
            robust_log_likelihood_ratio=torch.zeros_like(support),
            available=torch.ones(1, 1, dtype=torch.bool),
            valid_footprint_mass=torch.ones(1, 1),
        )
        return graph._fuse_modality_observations(prior, raw, evidence, context)

    unsupported = fuse(0.0)
    weak = fuse(0.1)
    supported = fuse(1.0)
    torch.testing.assert_close(unsupported.content, prior.content)
    torch.testing.assert_close(unsupported.lifecycle_log_probs, prior.lifecycle_log_probs)
    torch.testing.assert_close(unsupported.content_log_variance, prior.content_log_variance)
    torch.testing.assert_close(unsupported.geometry_mean, prior.geometry_mean)
    torch.testing.assert_close(unsupported.geometry_information, prior.geometry_information)
    torch.testing.assert_close(weak.content, prior.content + 0.1 * (raw.content - prior.content))
    torch.testing.assert_close(supported.content, raw.content)
    torch.testing.assert_close(supported.content_log_variance, raw.content_log_variance)
    assert torch.all(
        supported.geometry_information.diagonal(dim1=-2, dim2=-1)
        > prior.geometry_information.diagonal(dim1=-2, dim2=-1)
    )


def test_joint_coreference_geometry_and_posterior_write_have_finite_gradients() -> None:
    graph = _graph()
    context = _context()
    _, prepared = _prepare(graph, context)
    inputs, _, _, _, runtime = prepared
    assert runtime is not None
    assert inputs[0] is not None
    inputs[0].requires_grad_()
    inputs[0].retain_grad()
    total = inputs[0].shape[1] + inputs[1].shape[1]
    torch.manual_seed(29)
    query = torch.randn(1, total, 4, 8, requires_grad=True)
    key = torch.randn(1, total, 2, 8, requires_grad=True)
    value = torch.randn(1, total, 2, 8, requires_grad=True)
    runtime = graph.observe_joint_qkv(
        layer_index=graph.config.penultimate_layer,
        query_states=query,
        key_states=key,
        value_states=value,
        runtime=runtime,
    )
    outputs, _ = graph.after_layer(
        layer_index=graph.config.penultimate_layer,
        outputs_embeds=inputs,
        runtime=runtime,
    )
    assert outputs[0] is not None
    assert context.posterior is not None
    assert context.final_action_pair is not None
    posterior = context.posterior
    loss = (
        posterior.content.square().mean()
        + posterior.lifecycle_log_probs.square().mean()
        + posterior.geometry_mean.square().mean()
        + posterior.geometry_information.square().mean()
        + posterior.content_log_variance.square().mean()
        + context.final_action_pair.tokens.square().mean()
    )
    loss.backward()

    for tensor in (inputs[0], query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    parameter_gradients = [
        parameter.grad
        for parameter in graph.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert parameter_gradients
    assert all(torch.isfinite(gradient).all() for gradient in parameter_gradients)


def test_posterior_intervention_changes_a_declared_action_read() -> None:
    graph = _graph()
    prior = _belief()
    posterior_a = _belief(offset=0)
    posterior_b = _belief(offset=20)
    pair_a = graph.codec.paired_action_tokens(prior, posterior_a).tokens
    pair_b = graph.codec.paired_action_tokens(prior, posterior_b).tokens
    action_query = torch.ones(1, 1, 32)
    read_a = torch.softmax(action_query @ pair_a.transpose(-1, -2), -1) @ pair_a
    read_b = torch.softmax(action_query @ pair_b.transpose(-1, -2), -1) @ pair_b
    assert not torch.allclose(read_a, read_b)


def test_install_requires_the_pinned_native_hook() -> None:
    graph = _graph()

    class Host:
        def __init__(self) -> None:
            self.installed = None
            text_layer = SimpleNamespace(
                hidden_size=32,
                self_attn=SimpleNamespace(
                    q_proj=torch.nn.Linear(32, 32, bias=False),
                    head_dim=8,
                ),
            )
            action_layer = SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=torch.nn.Linear(32, 32, bias=False),
                    head_dim=8,
                ),
            )
            self.qwenvl = SimpleNamespace(
                model=SimpleNamespace(
                    language_model=SimpleNamespace(layers=[text_layer] * 3),
                ),
            )
            self.qwen_expert = SimpleNamespace(
                model=SimpleNamespace(layers=[action_layer] * 3),
            )

        def set_unified_belief_graph(self, value) -> None:
            self.installed = value

    class Model:
        def __init__(self) -> None:
            self.qwenvl_with_expert = Host()
            self.config = SimpleNamespace(max_action_dim=2)

    class Policy:
        def __init__(self) -> None:
            self.model = Model()

    policy = Policy()
    assert LingBotHostContract.from_policy(policy) == LingBotHostContract(
        prefix_width=32,
        attention_value_width=32,
        num_layers=3,
        executed_action_dim=2,
        native_measurement_query_tokens=0,
        native_prediction_query_tokens=0,
    )
    derived = LingBotUnifiedGraphConfig.from_policy(
        policy,
        codec=BeliefCodecConfig(3, 2, 1, 32),
        geometry_schema=GEOMETRY_SCHEMA,
    )
    assert derived.attention_value_width == 32
    assert derived.native_training_query_tokens == 0
    install_lingbot_unified_belief_graph(policy, graph)
    assert policy.model.qwenvl_with_expert.installed is graph
    with pytest.raises(TypeError, match="official"):
        install_lingbot_unified_belief_graph(object(), graph)


def test_host_contract_separates_current_measurement_and_future_query_counts() -> None:
    text_layer = SimpleNamespace(
        hidden_size=32,
        self_attn=SimpleNamespace(
            q_proj=torch.nn.Linear(32, 32, bias=False),
            head_dim=8,
        ),
    )
    action_layer = SimpleNamespace(
        self_attn=SimpleNamespace(
            q_proj=torch.nn.Linear(32, 32, bias=False),
            head_dim=8,
        ),
    )
    host = SimpleNamespace(
        qwenvl=SimpleNamespace(
            model=SimpleNamespace(language_model=SimpleNamespace(layers=[text_layer] * 3))
        ),
        qwen_expert=SimpleNamespace(model=SimpleNamespace(layers=[action_layer] * 3)),
    )
    model = SimpleNamespace(
        qwenvl_with_expert=host,
        config=SimpleNamespace(max_action_dim=2),
        use_depth_align=True,
        align_type="query",
        num_task_tokens=2,
        use_future_video=True,
        use_future_video_cls=True,
        use_future_video_patch=True,
        future_video_share_future_depth_query=False,
        use_future_depth=True,
    )
    contract = LingBotHostContract.from_policy(SimpleNamespace(model=model))
    assert contract.native_measurement_query_tokens == 2
    assert contract.native_prediction_query_tokens == 5
    assert contract.native_training_query_tokens == 7


def test_install_rejects_a_toy_contract_that_cannot_match_the_real_host() -> None:
    graph = _graph()

    class Host:
        def __init__(self) -> None:
            text_layer = SimpleNamespace(
                hidden_size=32,
                self_attn=SimpleNamespace(
                    q_proj=torch.nn.Linear(32, 64, bias=False),
                    head_dim=8,
                ),
            )
            action_layer = SimpleNamespace(
                self_attn=SimpleNamespace(
                    q_proj=torch.nn.Linear(32, 64, bias=False),
                    head_dim=8,
                ),
            )
            self.qwenvl = SimpleNamespace(
                model=SimpleNamespace(
                    language_model=SimpleNamespace(layers=[text_layer] * 3),
                ),
            )
            self.qwen_expert = SimpleNamespace(
                model=SimpleNamespace(layers=[action_layer] * 3),
            )

        def set_unified_belief_graph(self, value) -> None:
            raise AssertionError("mismatched graph must not be installed")

    policy = SimpleNamespace(
        model=SimpleNamespace(
            qwenvl_with_expert=Host(),
            config=SimpleNamespace(max_action_dim=2),
        )
    )
    with pytest.raises(ValueError, match="contract mismatch"):
        install_lingbot_unified_belief_graph(policy, graph)

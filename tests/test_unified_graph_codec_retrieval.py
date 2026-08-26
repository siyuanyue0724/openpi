from __future__ import annotations

from dataclasses import replace

import pytest

torch = pytest.importorskip("torch")

from picf_next.unified.codec import BeliefCodecConfig, UnifiedBeliefCodec  # noqa: E402
from picf_next.unified.coreference import tie_assignment_logits_by_group  # noqa: E402
from picf_next.unified.graph import (  # noqa: E402
    TokenLayout,
    TokenRole,
    block_causal_attention_mask,
    expand_host_mask_for_inserted_tokens,
    insert_layout_block,
    role_graph_contract_digest,
)
from picf_next.unified.retrieval import retrieve_task_context  # noqa: E402
from picf_next.unified.state import UnifiedBeliefState  # noqa: E402


def _layout(roles: list[TokenRole]) -> TokenLayout:
    role_tensor = torch.tensor([[int(role) for role in roles]], dtype=torch.long)
    return TokenLayout(roles=role_tensor, valid=torch.ones_like(role_tensor, dtype=torch.bool))


def test_belief_codec_config_rejects_ambiguous_or_nonfinite_values() -> None:
    with pytest.raises(TypeError, match="dimensions must be integers"):
        BeliefCodecConfig(3, 2, 1, 32.0)
    with pytest.raises(ValueError, match="finite and positive"):
        BeliefCodecConfig(3, 2, 1, 32, information_floor=float("nan"))


def test_role_graph_contract_digest_pins_every_causal_edge() -> None:
    assert role_graph_contract_digest() == (
        "27ee9605b65d1edf70d5ef5e5073572a412054ae70042f2273c126ebb8890510"
    )


def _masked_attention(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    scores = hidden @ hidden.transpose(-1, -2) / hidden.shape[-1] ** 0.5
    weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1)
    return weights @ hidden


def _belief(*, offset: float = 0.0) -> UnifiedBeliefState:
    batch, capacity = 1, 2
    valid = torch.ones(batch, capacity, 2, dtype=torch.bool)
    return UnifiedBeliefState(
        content=torch.arange(6, dtype=torch.float32).reshape(batch, capacity, 3) + offset,
        lifecycle_log_probs=torch.log_softmax(
            torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, -0.5]]]), dim=-1
        ),
        geometry_mean=torch.zeros(batch, capacity, 2),
        geometry_information=torch.eye(2).expand(batch, capacity, 2, 2).clone(),
        geometry_valid=valid,
        content_log_variance=torch.zeros(batch, capacity, 1),
        expected_age=torch.ones(batch, capacity),
        evidence_age=torch.ones(batch, capacity),
    )


def test_source_known_token_group_shares_one_refinement_invariant_assignment() -> None:
    logits = torch.tensor(
        [[[2.0, 0.0, -1.0], [-2.0, 4.0, 1.0], [0.5, 0.0, 2.0]]],
        requires_grad=True,
    )
    tied = tie_assignment_logits_by_group(
        logits,
        torch.tensor([[0.25, 0.75, 1.0]]),
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[7, 7, -1]]),
    )
    expected = 0.25 * logits[:, 0] + 0.75 * logits[:, 1]
    torch.testing.assert_close(tied[:, 0], expected)
    torch.testing.assert_close(tied[:, 1], expected)
    torch.testing.assert_close(tied[:, 2], logits[:, 2])
    tied.square().sum().backward()
    assert logits.grad is not None
    assert (logits.grad[:, :2].abs().sum(dim=-1) > 0).all()

    refined = tie_assignment_logits_by_group(
        torch.cat((logits.detach()[:, :1], logits.detach()[:, :1], logits.detach()[:, 1:]), dim=1),
        torch.tensor([[0.125, 0.125, 0.75, 1.0]]),
        torch.ones(1, 4, dtype=torch.bool),
        torch.tensor([[7, 7, 7, -1]]),
    )
    torch.testing.assert_close(refined[:, :3], expected[:, None].expand(-1, 3, -1))

    with pytest.raises(ValueError, match="positive footprint"):
        tie_assignment_logits_by_group(
            logits.detach(),
            torch.tensor([[0.0, 0.0, 1.0]]),
            torch.ones(1, 3, dtype=torch.bool),
            torch.tensor([[7, 7, -1]]),
        )


def test_grouped_assignment_uses_fp32_accumulation_for_bfloat16_logits() -> None:
    logits = torch.tensor(
        [[[4096.0, -4096.0, 1.0], [4080.0, -4080.0, 3.0], [0.0, 1.0, 2.0]]],
        dtype=torch.bfloat16,
    )
    footprint = torch.tensor([[0.001, 0.999, 1.0]], dtype=torch.float32)
    tied = tie_assignment_logits_by_group(
        logits,
        footprint,
        torch.ones(1, 3, dtype=torch.bool),
        torch.tensor([[2, 2, -1]]),
    )
    expected = (
        logits[:, 0].float() * footprint[:, :1] + logits[:, 1].float() * footprint[:, 1:2]
    ).to(torch.bfloat16)
    torch.testing.assert_close(tied[:, 0], expected)
    torch.testing.assert_close(tied[:, 1], expected)
    torch.testing.assert_close(tied[:, 2], logits[:, 2])


@pytest.mark.parametrize(
    ("logits", "footprint", "message"),
    [
        (torch.ones(1, 2, 3, dtype=torch.long), torch.ones(1, 2), "floating point"),
        (torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.long), "floating point"),
        (
            torch.tensor([[[float("nan"), 0.0, 1.0], [0.0, 1.0, 2.0]]]),
            torch.ones(1, 2),
            "finite",
        ),
    ],
)
def test_grouped_assignment_rejects_invalid_numeric_inputs(
    logits: torch.Tensor,
    footprint: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        tie_assignment_logits_by_group(
            logits,
            footprint,
            torch.ones(1, 2, dtype=torch.bool),
            torch.tensor([[0, 0]]),
        )


def test_native_baseline_mask_is_bitwise_unchanged() -> None:
    layout = _layout([TokenRole.NATIVE_BASELINE] * 5)
    base = torch.randint(0, 2, (1, 5, 5), dtype=torch.bool)
    result = block_causal_attention_mask(layout, base_mask=base)
    assert result is base


def test_query_roles_preserve_host_causality_and_isolate_loss_side_probe() -> None:
    roles = [
        TokenRole.HISTORY,
        TokenRole.PRIOR,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.POSTERIOR,
        TokenRole.LANGUAGE,
        TokenRole.RETRIEVAL,
        TokenRole.ACTION,
        TokenRole.MEASUREMENT_QUERY,
        TokenRole.HOST_FUTURE_QUERY,
        TokenRole.PREDICT_QUERY,
        TokenRole.CONTEXT,
    ]
    mask = block_causal_attention_mask(_layout(roles))
    physical_rows = torch.tensor([0, 1, 2, 3, 4])
    language_column = 5
    measurement_column = 8
    host_future_column = 9
    predict_column = 10
    context_column = 11
    assert not mask[0, physical_rows, language_column].any()
    assert not mask[0, physical_rows, measurement_column:].any()
    assert mask[0, measurement_column, 2]  # native sensor context
    assert mask[0, measurement_column, language_column]
    assert mask[0, measurement_column, measurement_column]
    assert not mask[0, measurement_column, :2].any()
    assert not mask[0, measurement_column, 3:5].any()
    assert mask[0, host_future_column, 2]
    assert mask[0, host_future_column, language_column]
    assert mask[0, host_future_column, measurement_column]
    assert mask[0, host_future_column, host_future_column]
    assert mask[0, 7, measurement_column]  # deploy-visible measurement reaches action
    assert not mask[0, 7, host_future_column]  # future supervision cannot reach action
    assert not mask[0, 7, predict_column]  # PICF loss-side probe cannot reach action
    assert not mask[0, :predict_column, predict_column].any()
    assert mask[0, predict_column, 0]  # source-time history is legal
    assert mask[0, predict_column, 1]  # prior is legal
    assert mask[0, predict_column, 3]  # current state is legal
    assert mask[0, predict_column, 4]  # posterior is legal
    assert not mask[0, predict_column, 2]  # current target sensor is forbidden
    assert not mask[0, predict_column, 5]  # language is forbidden
    assert not mask[0, predict_column, 6]  # retrieval is forbidden
    assert not mask[0, predict_column, measurement_column]
    assert not mask[0, predict_column, host_future_column]
    assert not mask[0, 4, 0]  # posterior cannot bypass the predictive prior
    assert mask[0, 4, 1:5].all()  # posterior reads prior, sensor, state and posterior
    assert mask[0, 6, 5]  # retrieval reads language
    assert mask[0, 7, 4]  # action reads posterior
    assert mask[0, context_column, 1]  # null query may compare against the prior inventory
    assert mask[0, context_column, 2]  # and summarize current physical evidence
    assert mask[0, context_column, 3]  # and current deploy-visible robot state
    assert mask[0, context_column, context_column]
    assert not mask[0, context_column, 4:7].any()
    assert not mask[0, 4, context_column]  # context cannot leak unowned evidence into posterior
    assert not mask[0, 6:8, context_column].any()


def test_only_predictive_prior_can_read_history() -> None:
    roles = [
        TokenRole.HISTORY,
        TokenRole.SENSOR,
        TokenRole.CURRENT_STATE,
        TokenRole.PRIOR,
        TokenRole.POSTERIOR,
        TokenRole.LANGUAGE,
        TokenRole.RETRIEVAL,
        TokenRole.ACTION,
        TokenRole.MEASUREMENT_QUERY,
    ]
    mask = block_causal_attention_mask(_layout(roles))
    assert mask[0, 3, 0]
    assert not mask[0, 1, 0]
    assert not mask[0, 2, 0]
    assert not mask[0, 4:, 0].any()
    assert not mask[0, 3, 2]  # predictive prior cannot read current state
    assert mask[0, 4, 2]  # posterior can correct from current state
    assert mask[0, 4, 3]
    assert mask[0, 7, 3]
    assert not mask[0, 8, 7]


def test_prompt_perturbation_is_invisible_to_physical_rows_across_layers() -> None:
    roles = [
        TokenRole.PRIOR,
        TokenRole.SENSOR,
        TokenRole.POSTERIOR,
        TokenRole.LANGUAGE,
        TokenRole.RETRIEVAL,
        TokenRole.ACTION,
    ]
    mask = block_causal_attention_mask(_layout(roles))
    torch.manual_seed(2)
    left = torch.randn(1, len(roles), 8)
    right = left.clone()
    right[:, 3] += 1000
    for _ in range(3):
        left = left + _masked_attention(left, mask)
        right = right + _masked_attention(right, mask)
    torch.testing.assert_close(left[:, :3], right[:, :3])
    assert not torch.allclose(left[:, 4:], right[:, 4:])


def test_inserted_token_block_preserves_every_old_host_edge() -> None:
    base = torch.tensor([[[True, False, False], [True, True, False], [True, True, True]]])
    expanded = expand_host_mask_for_inserted_tokens(
        base,
        insertion_index=2,
        inserted_count=2,
    )
    old_indices = torch.tensor([0, 1, 4])
    assert torch.equal(expanded[:, old_indices[:, None], old_indices[None, :]], base)
    assert expanded[:, 2:4].all()
    assert expanded[:, :, 2:4].all()

    base_layout = _layout([TokenRole.SENSOR, TokenRole.LANGUAGE, TokenRole.ACTION])
    inserted = _layout([TokenRole.PRIOR, TokenRole.POSTERIOR])
    combined = insert_layout_block(base_layout, inserted, insertion_index=2)
    assert combined.roles.tolist() == [
        [
            int(TokenRole.SENSOR),
            int(TokenRole.LANGUAGE),
            int(TokenRole.PRIOR),
            int(TokenRole.POSTERIOR),
            int(TokenRole.ACTION),
        ]
    ]


def test_unified_layout_rejects_mixed_ambiguous_baseline_roles() -> None:
    layout = _layout([TokenRole.NATIVE_BASELINE, TokenRole.SENSOR])
    with pytest.raises(ValueError, match="cannot be mixed"):
        layout.validate_unified()
    with pytest.raises(ValueError, match="cannot be mixed"):
        block_causal_attention_mask(layout)


def test_codec_preserves_exact_canonical_state_in_host_tokens() -> None:
    state = _belief()
    config = BeliefCodecConfig(
        content_dim=3,
        geometry_dim=2,
        uncertainty_dim=1,
        host_width=32,
    )
    codec = UnifiedBeliefCodec(config)
    encoded = codec.encode(state)
    torch.testing.assert_close(encoded[..., : config.canonical_width], state.canonical())
    torch.testing.assert_close(
        encoded[..., config.canonical_width :],
        torch.zeros_like(encoded[..., config.canonical_width :]),
    )


def test_codec_exposes_exact_prior_and_rowwise_prior_posterior_pairs() -> None:
    prior = _belief()
    posterior = _belief(offset=5.0)
    config = BeliefCodecConfig(3, 2, 1, 32)
    codec = UnifiedBeliefCodec(config)
    pair = codec.paired_action_tokens(prior, posterior)
    torch.testing.assert_close(pair.canonical[:, :2], prior.canonical())
    torch.testing.assert_close(pair.canonical[:, 2:], posterior.canonical())
    torch.testing.assert_close(
        pair.paired_canonical,
        torch.cat((prior.canonical(), posterior.canonical()), dim=-1),
    )
    torch.testing.assert_close(pair.prior_tokens, codec.encode(prior))
    torch.testing.assert_close(
        pair.pair_tokens[..., : 2 * config.canonical_width],
        pair.paired_canonical,
    )
    torch.testing.assert_close(
        pair.pair_tokens[..., 2 * config.canonical_width :],
        torch.zeros_like(pair.pair_tokens[..., 2 * config.canonical_width :]),
    )


def test_action_pair_changes_under_independent_posterior_permutation() -> None:
    prior = _belief()
    posterior = _belief(offset=5.0)
    codec = UnifiedBeliefCodec(BeliefCodecConfig(3, 2, 1, 32))

    original = codec.paired_action_tokens(prior, posterior).pair_tokens
    independently_permuted = codec.paired_action_tokens(
        prior,
        posterior.permute_rows(torch.tensor([1, 0])),
    ).pair_tokens

    width = codec.config.canonical_width
    original_cross_moment = (original[..., :width] * original[..., width : 2 * width]).sum(dim=1)
    permuted_cross_moment = (
        independently_permuted[..., :width] * independently_permuted[..., width : 2 * width]
    ).sum(dim=1)
    assert not torch.equal(original_cross_moment, permuted_cross_moment)


def test_action_pair_is_equivariant_to_simultaneous_row_permutation() -> None:
    prior = _belief()
    posterior = _belief(offset=5.0)
    codec = UnifiedBeliefCodec(BeliefCodecConfig(3, 2, 1, 32))
    permutation = torch.tensor([1, 0])

    original = codec.paired_action_tokens(prior, posterior)
    permuted = codec.paired_action_tokens(
        prior.permute_rows(permutation),
        posterior.permute_rows(permutation),
    )
    torch.testing.assert_close(
        permuted.prior_tokens,
        original.prior_tokens.index_select(1, permutation),
    )
    torch.testing.assert_close(
        permuted.pair_tokens,
        original.pair_tokens.index_select(1, permutation),
    )


def test_action_pair_requires_room_for_both_canonical_rows() -> None:
    codec = UnifiedBeliefCodec(BeliefCodecConfig(3, 2, 1, 16))
    with pytest.raises(ValueError, match="exact action pair"):
        codec.paired_action_tokens(_belief(), _belief(offset=1.0))


def test_action_pair_backpropagates_to_both_prior_and_posterior_rows() -> None:
    prior_content = _belief().content.detach().clone().requires_grad_()
    posterior_content = (_belief(offset=5.0).content.detach().clone()).requires_grad_()
    prior = replace(_belief(), content=prior_content)
    posterior = replace(_belief(offset=5.0), content=posterior_content)
    codec = UnifiedBeliefCodec(BeliefCodecConfig(3, 2, 1, 32))

    pair = codec.paired_action_tokens(prior, posterior)
    pair.pair_tokens.square().mean().backward()

    assert prior_content.grad is not None and prior_content.grad.abs().sum() > 0
    assert posterior_content.grad is not None and posterior_content.grad.abs().sum() > 0


def test_prediction_decoder_enforces_normalized_lifecycle_and_psd_geometry() -> None:
    config = BeliefCodecConfig(3, 2, 1, 32)
    codec = UnifiedBeliefCodec(config)
    hidden = torch.randn(2, 4, 32, requires_grad=True)
    valid = torch.tensor([True, False]).expand(2, 4, 2).clone()
    state = codec.decode_prediction(hidden, geometry_valid=valid)
    torch.testing.assert_close(state.lifecycle_probs.sum(-1), torch.ones(2, 4))
    eigenvalues = torch.linalg.eigvalsh(state.geometry_information)
    assert (eigenvalues >= -1e-7).all()
    assert (state.geometry_information[..., 1, :].abs() == 0).all()
    state.content.sum().backward()
    assert hidden.grad is not None


def test_psd_projection_has_finite_geometry_gradients_at_repeated_eigenvalues() -> None:
    config = BeliefCodecConfig(3, 6, 1, 128)
    codec = UnifiedBeliefCodec(config)
    hidden = torch.zeros(2, 4, 128, requires_grad=True)
    state = codec.decode_prediction(
        hidden,
        geometry_valid=torch.ones(2, 4, 6, dtype=torch.bool),
    )
    state.geometry_information.sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in codec.parameters()
    )


def test_codec_prediction_boundary_is_identity_initialized() -> None:
    original = _belief()
    config = BeliefCodecConfig(3, 2, 1, 32)
    codec = UnifiedBeliefCodec(config)
    restored = codec.decode_prediction(
        codec.encode(original),
        geometry_valid=original.geometry_valid,
    )
    for name in original.__dataclass_fields__:
        torch.testing.assert_close(getattr(restored, name), getattr(original, name))


def test_bfloat16_codec_boundary_preserves_fp32_probability_state() -> None:
    config = BeliefCodecConfig(3, 2, 1, 32)
    codec = UnifiedBeliefCodec(config).to(torch.bfloat16)
    hidden = torch.randn(1, 2, 32, dtype=torch.bfloat16, requires_grad=True)
    valid = torch.ones(1, 2, 2, dtype=torch.bool)
    state = codec.decode_prediction(hidden, geometry_valid=valid)
    for name in state.__dataclass_fields__:
        value = getattr(state, name)
        if value.dtype != torch.bool:
            assert value.dtype == torch.float32
    encoded = codec.encode(state)
    assert encoded.dtype == torch.bfloat16
    assert torch.isfinite(encoded).all()
    state.content.sum().backward()
    assert hidden.grad is not None


def test_task_retrieval_changes_selection_without_mutating_physical_inputs() -> None:
    belief_keys = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    belief_values = torch.tensor([[[10.0, 0.0], [0.0, 20.0]]])
    context_keys = torch.tensor([[[-1.0, 0.0]]])
    context_values = torch.tensor([[[-10.0, 0.0]]])
    nonempty = torch.tensor([[0.9, 0.9]])
    valid = torch.ones(1, 1, dtype=torch.bool)
    snapshots = tuple(value.clone() for value in (belief_keys, belief_values, nonempty))
    blue = retrieve_task_context(
        torch.tensor([[[8.0, 0.0]]]),
        belief_keys,
        belief_values,
        nonempty,
        context_keys,
        context_values,
        valid,
    )
    other = retrieve_task_context(
        torch.tensor([[[0.0, 8.0]]]),
        belief_keys,
        belief_values,
        nonempty,
        context_keys,
        context_values,
        valid,
    )
    assert blue.belief_weights.argmax(-1).item() == 0
    assert other.belief_weights.argmax(-1).item() == 1
    for value, snapshot in zip((belief_keys, belief_values, nonempty), snapshots, strict=True):
        assert torch.equal(value, snapshot)

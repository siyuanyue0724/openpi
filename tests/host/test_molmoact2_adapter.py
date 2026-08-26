from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
configuration = pytest.importorskip("olmo.hf_model.configuration_molmoact2")
modeling = pytest.importorskip("olmo.hf_model.modeling_molmoact2")
image_processing = pytest.importorskip("olmo.hf_model.image_processing_molmoact2")
adapter = pytest.importorskip("picf_next.hosts.molmoact2")
smoke = pytest.importorskip("tools.smoke_molmoact2_full_weight")

MolmoAct2ActionExpertConfig = configuration.MolmoAct2ActionExpertConfig
MolmoAct2AdapterConfig = configuration.MolmoAct2AdapterConfig
MolmoAct2Config = configuration.MolmoAct2Config
MolmoAct2TextConfig = configuration.MolmoAct2TextConfig
MolmoAct2VitConfig = configuration.MolmoAct2VitConfig
ActionExpert = modeling.ActionExpert
MolmoAct2ForConditionalGeneration = modeling.MolmoAct2ForConditionalGeneration
MolmoAct2VisionBackbone = modeling.MolmoAct2VisionBackbone
MolmoAct2ImageProcessor = image_processing.MolmoAct2ImageProcessor
PICFDenseEvidence = adapter.PICFDenseEvidence
MolmoAct2PICFActionExpert = adapter.MolmoAct2PICFActionExpert
MolmoAct2PICFForConditionalGeneration = adapter.MolmoAct2PICFForConditionalGeneration
MolmoAct2HostCheckpointIdentity = adapter.MolmoAct2HostCheckpointIdentity
PICFActionEvidence = adapter.PICFActionEvidence
_dense_patch_partition = adapter._dense_patch_partition
_encode_and_pool_vision_once = adapter._encode_and_pool_vision_once


def _expert() -> ActionExpert:
    config = MolmoAct2ActionExpertConfig(
        max_action_horizon=3,
        max_action_dim=5,
        hidden_size=16,
        num_layers=2,
        num_heads=4,
        mlp_ratio=2.0,
        ffn_multiple_of=8,
        timestep_embed_dim=8,
        dropout=0.0,
        attn_dropout=0.0,
        context_layer_norm=True,
        qk_norm=True,
        qk_norm_eps=1e-6,
        rope=True,
        causal_attn=False,
    )
    return ActionExpert(config, llm_dim=32, llm_kv_dim=12, llm_num_layers=2)


def _make_readout_nontrivial(expert: ActionExpert) -> ActionExpert:
    """Activate official zero-initialized modulation/readout for causal tests."""

    torch.manual_seed(31)
    with torch.no_grad():
        for block in expert.blocks:
            torch.nn.init.xavier_uniform_(block.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.modulation.linear.weight)
        torch.nn.init.xavier_uniform_(expert.final_layer.linear.weight)
    return expert


def _host() -> MolmoAct2ForConditionalGeneration:
    config = MolmoAct2Config(
        vit_config=MolmoAct2VitConfig(
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            image_default_input_size=(14, 14),
            image_patch_size=14,
            image_num_pos=1,
        ),
        adapter_config=MolmoAct2AdapterConfig(
            vit_layers=(0,),
            pooling_attention_mask=True,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=8,
            text_hidden_size=16,
        ),
        text_config=MolmoAct2TextConfig(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            vocab_size=128,
            additional_vocab_size=None,
            num_hidden_layers=1,
            intermediate_size=16,
        ),
        action_expert_config=MolmoAct2ActionExpertConfig(
            max_action_horizon=3,
            max_action_dim=4,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            mlp_ratio=2.0,
            ffn_multiple_of=8,
            timestep_embed_dim=8,
        ),
        add_action_expert=True,
        image_end_token_id=2,
        image_patch_id=3,
        max_action_dim=4,
        max_action_horizon=3,
        n_obs_steps=1,
        action_mode="both",
        state_format="discrete",
        state_token_start_id=10,
        num_state_tokens=8,
        action_start_token_id=20,
        action_end_token_id=21,
        action_token_start_id=22,
        num_action_tokens=16,
        enable_depth_reasoning=False,
    )
    host = MolmoAct2ForConditionalGeneration(config)
    _make_readout_nontrivial(host.model.action_expert)
    return host


def _visual_inputs() -> dict[str, torch.Tensor]:
    torch.manual_seed(17)
    return {
        "input_ids": torch.tensor([[3, 2, 7, 8]], dtype=torch.long),
        "pixel_values": torch.randint(0, 256, (1, 1, 14 * 14 * 3), dtype=torch.uint8),
        "image_token_pooling": torch.tensor([[0, -1, -1, -1]], dtype=torch.long),
        "image_grids": torch.tensor([[1, 1, 0, 0]], dtype=torch.long),
        "image_num_crops": torch.tensor([1], dtype=torch.long),
        "attention_mask": torch.ones(1, 4, dtype=torch.long),
    }


def _host_checkpoint_identity() -> MolmoAct2HostCheckpointIdentity:
    return MolmoAct2HostCheckpointIdentity(
        checkpoint_id="allenai/MolmoAct2-7B-D",
        revision="test-released-revision",
        manifest_sha256="a" * 64,
    )


def _inputs():
    torch.manual_seed(13)
    actions = torch.randn(2, 3, 5)
    timesteps = torch.tensor([0.1, 0.7])
    native_kv = [(torch.randn(2, 4, 12), torch.randn(2, 4, 12)) for _ in range(2)]
    native_mask = torch.ones(2, 4, dtype=torch.bool)
    dense_valid = torch.tensor(
        [
            [True, True, True, True, True, True, True],
            [True, True, True, False, False, False, False],
        ]
    )
    object_valid = torch.tensor([[True, True, False], [True, False, False]])
    vision_tokens = torch.randn(2, 4, 6) * dense_valid[:, :4].unsqueeze(-1)
    touch_tokens = torch.randn(2, 3, 5) * dense_valid[:, 4:].unsqueeze(-1)
    object_address = torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1)
    object_address = object_address * object_valid.unsqueeze(-1)
    object_value = torch.randn(2, 3, 9) * object_valid.unsqueeze(-1)
    evidence = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence("vision", vision_tokens, dense_valid[:, :4]),
            PICFDenseEvidence("touch", touch_tokens, dense_valid[:, 4:]),
        ),
        object_address=object_address,
        object_value=object_value,
        object_valid=object_valid,
        object_log_prior=torch.tensor([[-0.1, -0.4, 0.0], [-0.2, 0.0, 0.0]]),
    )
    return actions, timesteps, native_kv, native_mask, evidence


def test_zero_gates_exactly_reproduce_vanilla_output() -> None:
    vanilla = _make_readout_nontrivial(_expert()).eval()
    wrapped = MolmoAct2PICFActionExpert(
        vanilla,
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    actions, timesteps, native_kv, native_mask, evidence = _inputs()

    with torch.no_grad():
        expected = vanilla(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
        )
        without_context = wrapped(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
        )
        with_zero_context = wrapped(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
            evidence=evidence,
        )

    assert torch.equal(without_context, expected)
    assert torch.equal(with_zero_context, expected)
    assert torch.equal(wrapped.dense_gates, torch.zeros(2))
    assert torch.equal(wrapped.object_gates, torch.zeros(2))


def test_zero_gate_bootstrap_updates_gates_before_context_projections() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _make_readout_nontrivial(_expert()),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    actions, timesteps, native_kv, native_mask, evidence = _inputs()

    output = wrapped(
        actions,
        timesteps,
        encoder_kv_states=native_kv,
        encoder_attention_mask=native_mask,
        evidence=evidence,
    )
    output.square().mean().backward()

    gates = tuple(wrapped.dense_branches) + tuple(wrapped.object_branches)
    gate_gradients = torch.stack([branch.gate.grad for branch in gates])
    assert torch.isfinite(gate_gradients).all()
    assert torch.count_nonzero(gate_gradients) > 0

    context_projections = (
        wrapped.dense_k_proj["vision"].weight,
        wrapped.dense_k_proj["touch"].weight,
        wrapped.object_k_proj.weight,
    )
    assert all(parameter.grad is not None for parameter in context_projections)
    assert all(torch.count_nonzero(parameter.grad) == 0 for parameter in context_projections)


def test_bfloat16_host_keeps_scratch_adapter_parameters_and_optimizer_state_float32() -> None:
    vanilla = _make_readout_nontrivial(_expert()).to(torch.bfloat16)
    wrapped = MolmoAct2PICFActionExpert(
        vanilla,
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    actions, timesteps, native_kv, native_mask, evidence = _inputs()
    typed_evidence = PICFActionEvidence(
        dense_banks=tuple(
            PICFDenseEvidence(bank.modality, bank.tokens.to(torch.bfloat16), bank.valid)
            for bank in evidence.dense_banks
        ),
        object_address=evidence.object_address.to(torch.bfloat16),
        object_value=evidence.object_value.to(torch.bfloat16),
        object_valid=evidence.object_valid,
        object_log_prior=evidence.object_log_prior.to(torch.bfloat16),
    )

    with torch.autocast("cpu", dtype=torch.bfloat16):
        context = wrapped.prepare_picf_context(typed_evidence)
        output = wrapped(
            actions.to(torch.bfloat16),
            timesteps.to(torch.bfloat16),
            encoder_kv_states=[
                (key.to(torch.bfloat16), value.to(torch.bfloat16)) for key, value in native_kv
            ],
            encoder_attention_mask=native_mask,
            evidence=typed_evidence,
        )

    assert all(parameter.dtype == torch.float32 for parameter in wrapped.parameters())
    assert context.dense_kv_contexts is not None
    assert context.object_kv_contexts is not None
    assert context.dense_kv_contexts[0][0].dtype == torch.bfloat16
    assert context.object_kv_contexts[0][0].dtype == torch.bfloat16
    assert output.dtype == torch.bfloat16

    weight = wrapped.object_k_proj.weight
    before = weight.detach().clone()
    weight.grad = torch.ones_like(weight)
    optimizer = torch.optim.AdamW([weight], lr=5e-5, weight_decay=0.0)
    optimizer.step()

    assert not torch.equal(weight, before)
    assert optimizer.state[weight]["exp_avg"].dtype == torch.float32
    torch.testing.assert_close(
        before - weight,
        torch.full_like(weight, 5e-5),
        atol=1e-7,
        rtol=0.0,
    )


def test_typed_branches_are_exactly_zero_for_per_sample_missing_evidence() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    branch = wrapped.dense_branches[0]
    with torch.no_grad():
        branch.gate.fill_(1.0)
        branch.attention.out_proj.bias.fill_(0.75)

    hidden = torch.randn(2, 3, 16)
    shift = torch.zeros(2, 16)
    scale = torch.zeros(2, 16)
    key = torch.randn(2, 4, 4, 4)
    value = torch.randn(2, 4, 4, 4)
    valid = torch.tensor([[True, False, False, False], [False, False, False, False]])
    mask = wrapped._attention_mask(valid, hidden.dtype)
    assert mask is not None

    output = branch(hidden, shift, scale, kv=(key, value), attention_mask=mask)

    assert not torch.equal(output[0], torch.zeros_like(output[0]))
    assert torch.equal(output[1], torch.zeros_like(output[1]))


def test_dense_and_object_domains_do_not_change_native_token_count() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    _, _, _, _, evidence = _inputs()
    dense_before = tuple(bank.tokens.clone() for bank in evidence.dense_banks)
    address_before = evidence.object_address.clone()
    value_before = evidence.object_value.clone()
    context = wrapped.prepare_picf_context(evidence)

    assert context.dense_kv_contexts is not None
    assert context.object_kv_contexts is not None
    assert all(k.shape[1] == 7 and v.shape[1] == 7 for k, v in context.dense_kv_contexts)
    assert all(k.shape[1] == 3 and v.shape[1] == 3 for k, v in context.object_kv_contexts)
    assert sum(bank.tokens.shape[1] for bank in evidence.dense_banks) == 7
    assert all(
        torch.equal(bank.tokens, before)
        for bank, before in zip(evidence.dense_banks, dense_before, strict=True)
    )
    assert wrapped.dense_k_proj["vision"].in_features == 6
    assert wrapped.dense_k_proj["touch"].in_features == 5
    assert torch.equal(evidence.object_address, address_before)
    assert torch.equal(evidence.object_value, value_before)


def test_object_attention_adds_posterior_existence_log_prior() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _make_readout_nontrivial(_expert()),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    wrapped.object_branches[0].gate.data.fill_(0.4)
    actions, timesteps, native_kv, native_mask, evidence = _inputs()
    context = wrapped.prepare_picf_context(evidence)
    assert context.object_mask is not None
    torch.testing.assert_close(
        context.object_mask[:, 0, 0][evidence.object_valid],
        evidence.object_log_prior[evidence.object_valid],
    )
    assert (
        context.object_mask[:, 0, 0][~evidence.object_valid]
        == torch.finfo(context.object_mask.dtype).min
    ).all()

    changed_prior = torch.tensor(
        [[-4.0, -0.01, 0.0], [-0.01, 0.0, 0.0]],
        dtype=evidence.object_log_prior.dtype,
    )
    changed = PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=evidence.object_address,
        object_value=evidence.object_value,
        object_valid=evidence.object_valid,
        object_log_prior=changed_prior,
    )
    with torch.no_grad():
        expected = wrapped(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
            evidence=evidence,
        )
        actual = wrapped(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
            evidence=changed,
        )
    assert not torch.equal(actual, expected)


def test_training_layer_protocol_rejects_coerced_index_and_causal_flag() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    )
    hidden = torch.zeros(1, 3, 16)
    conditioning = torch.zeros(1, 16)
    cross_kv = (torch.zeros(1, 2, 12), torch.zeros(1, 2, 12))

    with pytest.raises(TypeError, match="layer_index must be an integer"):
        wrapped.apply_training_layer(
            hidden,
            conditioning,
            layer_index=0.0,
            cross_kv=cross_kv,
            self_attn_mask=None,
            attn_mask=None,
            is_causal=False,
            modulation=None,
            rope_cache=None,
            context=None,
        )

    with pytest.raises(TypeError, match="is_causal must be a boolean"):
        wrapped.apply_training_layer(
            hidden,
            conditioning,
            layer_index=0,
            cross_kv=cross_kv,
            self_attn_mask=None,
            attn_mask=None,
            is_causal=0,
            modulation=None,
            rope_cache=None,
            context=None,
        )


def test_training_layer_expands_only_the_selected_picf_layer(monkeypatch) -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    _, _, _, _, evidence = _inputs()
    context = wrapped.prepare_picf_context(evidence)
    calls: list[tuple[int, int]] = []
    original = MolmoAct2PICFActionExpert._expand_context_tensor

    def traced_expand(tensor, *, target_batch_size):
        if tensor is not None:
            calls.append((tensor.shape[0], target_batch_size))
        return original(tensor, target_batch_size=target_batch_size)

    monkeypatch.setattr(
        MolmoAct2PICFActionExpert,
        "_expand_context_tensor",
        staticmethod(traced_expand),
    )
    hidden = torch.zeros(4, 3, 16)
    conditioning = torch.zeros(4, 16)
    cross_kv = (torch.zeros(4, 4, 16), torch.zeros(4, 4, 16))

    output = wrapped.apply_training_layer(
        hidden,
        conditioning,
        layer_index=1,
        cross_kv=cross_kv,
        self_attn_mask=None,
        attn_mask=None,
        is_causal=False,
        modulation=None,
        rope_cache=None,
        context=context,
    )

    assert output.shape == hidden.shape
    # Dense K/V/mask and object K/V/mask are each expanded once. Expanding all
    # layers here would make the training bridge quadratic in expert depth.
    assert calls == [(2, 4)] * 6


def test_object_key_depends_only_on_address_and_value_only_on_state() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    _, _, _, _, evidence = _inputs()
    base = wrapped.prepare_picf_context(evidence)
    changed_value = wrapped.prepare_picf_context(
        PICFActionEvidence(
            dense_banks=evidence.dense_banks,
            object_address=evidence.object_address,
            object_value=evidence.object_value + evidence.object_valid.unsqueeze(-1),
            object_valid=evidence.object_valid,
            object_log_prior=evidence.object_log_prior,
        )
    )
    changed_address = wrapped.prepare_picf_context(
        PICFActionEvidence(
            dense_banks=evidence.dense_banks,
            object_address=torch.roll(evidence.object_address, shifts=1, dims=-1),
            object_value=evidence.object_value,
            object_valid=evidence.object_valid,
            object_log_prior=evidence.object_log_prior,
        )
    )
    assert base.object_kv_contexts is not None
    assert changed_value.object_kv_contexts is not None
    assert changed_address.object_kv_contexts is not None
    for (base_key, base_value), (value_key, changed_state), (changed_key, address_value) in zip(
        base.object_kv_contexts,
        changed_value.object_kv_contexts,
        changed_address.object_kv_contexts,
        strict=True,
    ):
        torch.testing.assert_close(base_key, value_key)
        assert not torch.equal(base_value, changed_state)
        assert not torch.equal(base_key, changed_key)
        torch.testing.assert_close(base_value, address_value)


def test_persistent_ownership_adds_shared_address_without_removing_dense_tokens() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    _, _, _, _, evidence = _inputs()
    ownership = []
    for bank in evidence.dense_banks:
        assignment = bank.tokens.new_zeros(*bank.tokens.shape[:2], 4)
        assignment[..., -1] = 1.0
        assignment[..., 0] = bank.valid
        assignment[..., -1] = ~bank.valid
        ownership.append(assignment)
    structured = PICFActionEvidence(
        dense_banks=evidence.dense_banks,
        object_address=evidence.object_address,
        object_value=evidence.object_value,
        object_valid=evidence.object_valid,
        object_log_prior=evidence.object_log_prior,
        dense_ownership=tuple(ownership),
    )

    plain_context = wrapped.prepare_picf_context(evidence)
    structured_context = wrapped.prepare_picf_context(structured)
    changed_value_context = wrapped.prepare_picf_context(
        PICFActionEvidence(
            dense_banks=evidence.dense_banks,
            object_address=evidence.object_address,
            object_value=evidence.object_value + evidence.object_valid.unsqueeze(-1),
            object_valid=evidence.object_valid,
            object_log_prior=evidence.object_log_prior,
            dense_ownership=tuple(ownership),
        )
    )

    assert structured_context.dense_kv_contexts is not None
    assert plain_context.dense_kv_contexts is not None
    assert changed_value_context.dense_kv_contexts is not None
    for plain, structured_layer, changed_value in zip(
        plain_context.dense_kv_contexts,
        structured_context.dense_kv_contexts,
        changed_value_context.dense_kv_contexts,
        strict=True,
    ):
        assert not torch.equal(plain[0], structured_layer[0])
        assert not torch.equal(plain[1], structured_layer[1])
        torch.testing.assert_close(structured_layer[0], changed_value[0])
        torch.testing.assert_close(structured_layer[1], changed_value[1])
        assert structured_layer[0].shape[1] == 7


def test_nonzero_domain_gates_change_output_and_propagate_gradients() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _make_readout_nontrivial(_expert()),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    actions, timesteps, native_kv, native_mask, evidence = _inputs()
    for branch in wrapped.dense_branches:
        branch.gate.data.fill_(0.2)
    for branch in wrapped.object_branches:
        branch.gate.data.fill_(0.3)

    output = wrapped(
        actions,
        timesteps,
        encoder_kv_states=native_kv,
        encoder_attention_mask=native_mask,
        evidence=evidence,
    )
    vanilla_output = wrapped.vanilla(
        actions,
        timesteps,
        encoder_kv_states=native_kv,
        encoder_attention_mask=native_mask,
    )
    assert not torch.equal(output, vanilla_output)

    output.square().mean().backward()
    assert wrapped.dense_k_proj["vision"].weight.grad is not None
    assert wrapped.dense_k_proj["touch"].weight.grad is not None
    assert wrapped.object_k_proj.weight.grad is not None
    assert all(branch.gate.grad is not None for branch in wrapped.dense_branches)
    assert all(branch.gate.grad is not None for branch in wrapped.object_branches)


def test_rejects_partial_bank_or_nonzero_masked_padding() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    with pytest.raises(ValueError, match="all present or absent"):
        wrapped.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(),
                object_address=torch.zeros(1, 2, 4),
                object_value=None,
                object_valid=None,
            )
        )

    with pytest.raises(ValueError, match="share one batch size"):
        wrapped.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(
                    PICFDenseEvidence(
                        "vision",
                        torch.zeros(2, 1, 6),
                        torch.ones(2, 1, dtype=torch.bool),
                    ),
                ),
                object_address=torch.zeros(1, 1, 4),
                object_value=torch.zeros(1, 1, 9),
                object_valid=torch.ones(1, 1, dtype=torch.bool),
                object_log_prior=torch.zeros(1, 1),
            )
        )
    with pytest.raises(ValueError, match="padding"):
        wrapped.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(
                    PICFDenseEvidence(
                        "vision",
                        torch.ones(1, 2, 6),
                        torch.zeros(1, 2, dtype=torch.bool),
                    ),
                ),
                object_address=None,
                object_value=None,
                object_valid=None,
            )
        )


def test_missing_extra_modalities_are_zero_residual_per_sample() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _make_readout_nontrivial(_expert()),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    actions, timesteps, native_kv, native_mask, _ = _inputs()
    for branch in wrapped.dense_branches:
        branch.gate.data.fill_(0.4)
    dense_tokens = torch.zeros(2, 3, 6)
    dense_tokens[0] = torch.randn(3, 6)
    evidence = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence(
                "vision",
                dense_tokens,
                torch.tensor([[True, True, True], [False, False, False]]),
            ),
        ),
        object_address=None,
        object_value=None,
        object_valid=None,
    )
    with torch.no_grad():
        actual = wrapped(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
            evidence=evidence,
        )
        vanilla = wrapped.vanilla(
            actions,
            timesteps,
            encoder_kv_states=native_kv,
            encoder_attention_mask=native_mask,
        )

    assert not torch.equal(actual[0], vanilla[0])
    torch.testing.assert_close(actual[1], vanilla[1], atol=0.0, rtol=0.0)


def test_zero_length_dense_banks_do_not_create_empty_attention_context() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    evidence = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence(
                "vision",
                torch.zeros(2, 0, 6),
                torch.zeros(2, 0, dtype=torch.bool),
            ),
            PICFDenseEvidence(
                "touch",
                torch.zeros(2, 0, 5),
                torch.zeros(2, 0, dtype=torch.bool),
            ),
        ),
        object_address=None,
        object_value=None,
        object_valid=None,
    )

    context = wrapped.prepare_picf_context(evidence)

    assert context.dense_kv_contexts is None
    assert context.dense_mask is None
    assert context.object_kv_contexts is None
    assert context.object_mask is None


def test_zero_length_modality_is_a_graph_connected_concatenation_identity() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    evidence = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence(
                "vision",
                torch.randn(2, 3, 6),
                torch.ones(2, 3, dtype=torch.bool),
            ),
            PICFDenseEvidence(
                "touch",
                torch.zeros(2, 0, 5),
                torch.zeros(2, 0, dtype=torch.bool),
            ),
        ),
        object_address=None,
        object_value=None,
        object_valid=None,
    )

    context = wrapped.prepare_picf_context(evidence)

    assert context.dense_kv_contexts is not None
    assert all(key.shape[1] == 3 for key, _value in context.dense_kv_contexts)
    sum(
        key.square().mean() + value.square().mean() for key, value in context.dense_kv_contexts
    ).backward()
    for projection in (wrapped.dense_k_proj["touch"], wrapped.dense_v_proj["touch"]):
        assert projection.weight.grad is not None
        assert torch.count_nonzero(projection.weight.grad) == 0


def test_disabled_posterior_action_route_freezes_only_its_parameters() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    )

    wrapped.set_posterior_action_context_trainable(False)

    object_modules = (
        wrapped.object_k_proj,
        wrapped.object_v_proj,
        wrapped.dense_owner_v_proj,
        wrapped.object_context_norm,
        wrapped.object_branches,
    )
    assert all(
        not parameter.requires_grad
        for module in object_modules
        for parameter in module.parameters()
    )
    dense_modules = (
        wrapped.dense_k_proj,
        wrapped.dense_v_proj,
        wrapped.dense_context_norm,
        wrapped.dense_branches,
    )
    assert all(
        parameter.requires_grad for module in dense_modules for parameter in module.parameters()
    )

    with pytest.raises(TypeError, match="must be boolean"):
        wrapped.set_posterior_action_context_trainable(1)  # type: ignore[arg-type]


def test_rejects_implicit_typed_bank_dtype_conversion() -> None:
    wrapped = MolmoAct2PICFActionExpert(
        _expert(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    )
    with pytest.raises(ValueError, match="action-expert dtype"):
        wrapped.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(
                    PICFDenseEvidence(
                        "vision",
                        torch.zeros(1, 2, 6, dtype=torch.float64),
                        torch.ones(1, 2, dtype=torch.bool),
                    ),
                ),
                object_address=None,
                object_value=None,
                object_valid=None,
            )
        )


def test_host_level_fixed_noise_zero_gate_matches_official_generation() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    kv_dim = host.config.text_config.num_key_value_heads * host.config.text_config.head_dim
    torch.manual_seed(47)
    encoder_kv_states = [(torch.randn(2, 5, kv_dim), torch.randn(2, 5, kv_dim))]
    encoder_mask = torch.ones(2, 5, dtype=torch.bool)
    evidence = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence(
                "vision",
                torch.randn(2, 4, 6),
                torch.ones(2, 4, dtype=torch.bool),
            ),
            PICFDenseEvidence(
                "touch",
                torch.randn(2, 3, 5),
                torch.ones(2, 3, dtype=torch.bool),
            ),
        ),
        object_address=torch.nn.functional.normalize(torch.randn(2, 3, 4), dim=-1),
        object_value=torch.randn(2, 3, 9),
        object_valid=torch.ones(2, 3, dtype=torch.bool),
        object_log_prior=torch.zeros(2, 3),
    )
    common = {
        "input_ids": torch.ones(2, 5, dtype=torch.long),
        "encoder_kv_states": encoder_kv_states,
        "encoder_attention_mask": encoder_mask,
        "num_steps": 3,
    }

    expected = host.model.generate_actions_from_inputs(
        **common,
        generator=torch.Generator().manual_seed(53),
    )
    actual = wrapped.generate_actions_from_inputs(
        **common,
        evidence=evidence,
        generator=torch.Generator().manual_seed(53),
    )

    assert torch.equal(actual, expected)


def test_generation_rejects_explicit_zero_flow_steps() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    kv_dim = host.config.text_config.num_key_value_heads * host.config.text_config.head_dim
    encoder_kv_states = [(torch.randn(1, 2, kv_dim), torch.randn(1, 2, kv_dim))]

    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        wrapped.generate_actions_from_inputs(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            evidence=None,
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            num_steps=0,
        )


def test_external_encoder_context_rejects_ignored_visual_inputs_and_batch_drift() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    kv_dim = host.config.text_config.num_key_value_heads * host.config.text_config.head_dim
    encoder_kv_states = [(torch.randn(1, 2, kv_dim), torch.randn(1, 2, kv_dim))]

    with pytest.raises(ValueError, match="would be ignored"):
        wrapped.generate_actions_from_inputs(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            evidence=None,
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            pixel_values=torch.zeros(1, 1, 3),
        )

    mismatched = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence(
                "vision",
                torch.zeros(2, 1, 6),
                torch.ones(2, 1, dtype=torch.bool),
            ),
        ),
        object_address=None,
        object_value=None,
        object_valid=None,
    )
    with pytest.raises(ValueError, match="evidence and encoder KV"):
        wrapped.generate_actions_from_inputs(
            input_ids=torch.ones(1, 2, dtype=torch.long),
            evidence=mismatched,
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=torch.ones(1, 2, dtype=torch.bool),
        )


def test_external_encoder_context_fails_closed_on_every_structural_field() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    kv_dim = host.config.text_config.num_key_value_heads * host.config.text_config.head_dim
    valid_kv = [(torch.randn(1, 2, kv_dim), torch.randn(1, 2, kv_dim))]
    common = {
        "input_ids": torch.ones(1, 2, dtype=torch.long),
        "evidence": None,
        "encoder_attention_mask": torch.ones(1, 2, dtype=torch.bool),
        "num_steps": 1,
    }

    with pytest.raises(ValueError, match="must contain 1 layers"):
        wrapped.generate_actions_from_inputs(**common, encoder_kv_states=[])
    with pytest.raises(ValueError, match="key/value shape"):
        wrapped.generate_actions_from_inputs(
            **common,
            encoder_kv_states=[(torch.randn(1, 2, kv_dim + 1), torch.randn(1, 2, kv_dim + 1))],
        )
    corrupt = valid_kv[0][0].clone()
    corrupt[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        wrapped.generate_actions_from_inputs(
            **common,
            encoder_kv_states=[(corrupt, valid_kv[0][1])],
        )
    with pytest.raises(ValueError, match="attention mask must align"):
        wrapped.generate_actions_from_inputs(
            **{**common, "encoder_attention_mask": torch.ones(1, 1, dtype=torch.bool)},
            encoder_kv_states=valid_kv,
        )
    with pytest.raises(ValueError, match="action_horizon must be an integer"):
        wrapped.generate_actions_from_inputs(
            **common,
            encoder_kv_states=valid_kv,
            action_horizon=1.5,
        )
    with pytest.raises(ValueError, match="action_dim_is_pad must be a boolean"):
        wrapped.generate_actions_from_inputs(
            **common,
            encoder_kv_states=valid_kv,
            action_dim_is_pad=torch.zeros(1, 4),
        )


def test_target_only_fields_cannot_change_final_fixed_noise_action() -> None:
    host = _host().eval()
    clean = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "attention_mask": torch.ones(1, 5, dtype=torch.long),
    }
    contaminated = {
        **clean,
        "labels": torch.tensor([[1, 2, 3, 4, 5]]),
        "object_mask_target": torch.rand(1, 5),
        "simulator_instance_id": torch.tensor([91]),
        "task_owner_target": torch.tensor([3]),
    }
    clean_runtime = smoke._move_inputs(clean, torch.device("cpu"))
    contaminated_runtime = smoke._move_inputs(contaminated, torch.device("cpu"))

    assert set(clean_runtime) == set(contaminated_runtime)
    for key in clean_runtime:
        assert torch.equal(clean_runtime[key], contaminated_runtime[key])
    with torch.no_grad():
        expected = host.model.generate_actions_from_inputs(
            **clean_runtime,
            num_steps=2,
            generator=torch.Generator().manual_seed(59),
        )
        actual = host.model.generate_actions_from_inputs(
            **contaminated_runtime,
            num_steps=2,
            generator=torch.Generator().manual_seed(59),
        )
    assert torch.equal(actual, expected)


def test_one_pass_dense_vision_bundle_exactly_matches_official_encoder_kv() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"molmo_vision_patch": 8},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    inputs = _visual_inputs()

    with torch.no_grad():
        expected_outputs = host.model(**inputs, use_cache=True)
        expected_kv = tuple(host.model._extract_kv_states(expected_outputs.past_key_values))
        actual = wrapped.encode_inputs_for_picf(**inputs)

    assert len(actual.encoder_kv_states) == len(expected_kv)
    for (actual_key, actual_value), (expected_key, expected_value) in zip(
        actual.encoder_kv_states,
        expected_kv,
        strict=True,
    ):
        assert torch.equal(actual_key, expected_key)
        assert torch.equal(actual_value, expected_value)
    assert torch.equal(
        actual.encoder_attention_mask,
        host.model._get_encoder_attention_mask(inputs["input_ids"], inputs["attention_mask"]),
    )
    assert actual.vision_patch_bank is not None
    assert actual.vision_patch_bank.modality == "molmo_vision_patch"
    assert actual.vision_patch_bank.tokens.shape == (1, 1, 8)
    assert torch.equal(actual.vision_patch_bank.valid, torch.ones(1, 1, dtype=torch.bool))


def test_molmo_dense_width_is_the_concatenated_prepooling_vit_width() -> None:
    with pytest.raises(ValueError, match="concatenated pre-pooling ViT width"):
        MolmoAct2PICFForConditionalGeneration(
            _host(),
            dense_token_dims={"molmo_vision_patch": 16},
            object_address_dim=4,
            object_value_dim=9,
        )


def test_one_pass_dense_vision_bundle_calls_vit_encoder_once(monkeypatch) -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"molmo_vision_patch": 8},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    original = host.model.vision_backbone.encode_image
    calls = 0

    def counted(images):
        nonlocal calls
        calls += 1
        return original(images)

    monkeypatch.setattr(host.model.vision_backbone, "encode_image", counted)
    with torch.no_grad():
        wrapped.encode_inputs_for_picf(**_visual_inputs())
    assert calls == 1


def test_one_pass_dense_and_native_paths_both_reach_the_vision_encoder() -> None:
    host = _host().train()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"molmo_vision_patch": 8},
        object_address_dim=4,
        object_value_dim=9,
    ).train()
    bundle = wrapped.encode_inputs_for_picf(**_visual_inputs())
    assert bundle.vision_patch_bank is not None
    dense_loss = bundle.vision_patch_bank.tokens.square().mean()
    native_loss = bundle.encoder_kv_states[0][0].square().mean()
    (dense_loss + native_loss).backward()

    backbone = host.model.vision_backbone
    assert backbone.image_vit.patch_embedding.weight.grad is not None
    assert any(parameter.grad is not None for parameter in backbone.image_pooling_2d.parameters())
    assert any(parameter.grad is not None for parameter in backbone.image_projector.parameters())


def test_visual_bundle_zero_gate_exactly_matches_official_fixed_noise_action() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"molmo_vision_patch": 8},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    inputs = _visual_inputs()
    with torch.no_grad():
        expected = host.model.generate_actions_from_inputs(
            **inputs,
            num_steps=2,
            generator=torch.Generator().manual_seed(71),
        )
        bundle = wrapped.encode_inputs_for_picf(**inputs)
        assert bundle.vision_patch_bank is not None
        actual = wrapped.generate_actions_from_inputs(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            evidence=PICFActionEvidence(
                dense_banks=(bundle.vision_patch_bank,),
                object_address=None,
                object_value=None,
                object_valid=None,
            ),
            encoder_kv_states=bundle.encoder_kv_states,
            encoder_attention_mask=bundle.encoder_attention_mask,
            num_steps=2,
            generator=torch.Generator().manual_seed(71),
        )

    assert torch.equal(actual, expected)


def test_dense_patch_partition_rejects_duplicate_missing_and_partial_crop() -> None:
    valid = _dense_patch_partition(
        torch.tensor([[[0, 1, 2, 3], [4, 5, 6, 7]]]),
        num_crops=1,
        patches_per_crop=8,
    )
    assert torch.equal(valid, torch.ones(1, 8, dtype=torch.bool))

    with pytest.raises(ValueError, match="exactly once"):
        _dense_patch_partition(
            torch.tensor([[[0, 1, 2, 2], [4, 5, 6, 7]]]),
            num_crops=1,
            patches_per_crop=8,
        )
    with pytest.raises(ValueError, match="whole contiguous crops"):
        _dense_patch_partition(
            torch.tensor([[[0, 1, 2, 3]]]),
            num_crops=1,
            patches_per_crop=8,
        )
    with pytest.raises(ValueError, match="outside the dense patch bank"):
        _dense_patch_partition(
            torch.tensor([[[-2, 0, 1, 2]]]),
            num_crops=1,
            patches_per_crop=4,
        )
    with pytest.raises(ValueError, match="integer tensors"):
        _dense_patch_partition(
            torch.tensor([[[0.0, 1.0, 2.0, 3.0]]]),
            num_crops=1,
            patches_per_crop=4,
        )


def test_one_pass_vision_rejects_non_processor_image_dtype_before_encoding() -> None:
    backbone = _host().model.vision_backbone
    with pytest.raises(ValueError, match="uint8 or floating"):
        _encode_and_pool_vision_once(
            backbone,
            torch.zeros((1, 1, 1, 14 * 14 * 3), dtype=torch.int16),
            torch.tensor([[[0]]], dtype=torch.long),
        )


def test_released_resize_geometry_returns_all_729_dense_patches_and_196_native_tokens() -> None:
    processor = MolmoAct2ImageProcessor(crop_mode="resize")
    metadata = processor(
        images=[np.zeros((200, 240, 3), dtype=np.uint8)],
        return_tensors="np",
    )
    backbone = MolmoAct2VisionBackbone(
        MolmoAct2VitConfig(
            hidden_size=8,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            image_default_input_size=(378, 378),
            image_patch_size=14,
            image_num_pos=729,
        ),
        MolmoAct2AdapterConfig(
            vit_layers=(0,),
            pooling_attention_mask=True,
            hidden_size=8,
            num_attention_heads=2,
            num_key_value_heads=2,
            head_dim=4,
            intermediate_size=8,
            text_hidden_size=16,
        ),
    ).eval()
    images = torch.from_numpy(metadata["pixel_values"]).unsqueeze(0)
    pooling = torch.from_numpy(metadata["image_token_pooling"]).unsqueeze(0)

    with torch.no_grad():
        expected_pooled = backbone(images, pooling)
        actual_pooled, dense, dense_valid = _encode_and_pool_vision_once(
            backbone,
            images,
            pooling,
        )

    assert torch.equal(actual_pooled, expected_pooled)
    assert actual_pooled.shape == (196, 16)
    assert dense.shape == (1, 729, 8)
    assert torch.equal(dense_valid, torch.ones(1, 729, dtype=torch.bool))


def test_host_level_nonzero_typed_context_changes_fixed_noise_action() -> None:
    host = _host().eval()
    wrapped = MolmoAct2PICFForConditionalGeneration(
        host,
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
    ).eval()
    for branch in wrapped.action_adapter.dense_branches:
        branch.gate.data.fill_(0.25)
    for branch in wrapped.action_adapter.object_branches:
        branch.gate.data.fill_(0.35)
    kv_dim = host.config.text_config.num_key_value_heads * host.config.text_config.head_dim
    torch.manual_seed(59)
    encoder_kv_states = [(torch.randn(1, 5, kv_dim), torch.randn(1, 5, kv_dim))]
    common = {
        "input_ids": torch.ones(1, 5, dtype=torch.long),
        "encoder_kv_states": encoder_kv_states,
        "encoder_attention_mask": torch.ones(1, 5, dtype=torch.bool),
        "num_steps": 2,
    }
    evidence = PICFActionEvidence(
        dense_banks=(
            PICFDenseEvidence(
                "vision",
                torch.randn(1, 4, 6),
                torch.ones(1, 4, dtype=torch.bool),
            ),
            PICFDenseEvidence(
                "touch",
                torch.randn(1, 3, 5),
                torch.ones(1, 3, dtype=torch.bool),
            ),
        ),
        object_address=torch.nn.functional.normalize(torch.randn(1, 3, 4), dim=-1),
        object_value=torch.randn(1, 3, 9),
        object_valid=torch.ones(1, 3, dtype=torch.bool),
        object_log_prior=torch.zeros(1, 3),
    )

    expected = host.model.generate_actions_from_inputs(
        **common,
        generator=torch.Generator().manual_seed(61),
    )
    actual = wrapped.generate_actions_from_inputs(
        **common,
        evidence=evidence,
        generator=torch.Generator().manual_seed(61),
    )

    assert not torch.equal(actual, expected)


def test_adapter_checkpoint_excludes_host_and_round_trips(tmp_path) -> None:
    source = MolmoAct2PICFForConditionalGeneration(
        _host(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
        host_checkpoint_identity=_host_checkpoint_identity(),
    )
    for branch in source.action_adapter.dense_branches:
        branch.gate.data.fill_(0.17)
    torch.nn.init.normal_(source.action_adapter.object_k_proj.weight)
    source.save_adapter_pretrained(tmp_path)

    saved_names = set(source.action_adapter.state_dict())
    assert saved_names
    assert not any(name.startswith("vanilla.") for name in saved_names)

    target = MolmoAct2PICFForConditionalGeneration(
        _host(),
        dense_token_dims={"vision": 6, "touch": 5},
        object_address_dim=4,
        object_value_dim=9,
        host_checkpoint_identity=_host_checkpoint_identity(),
    )
    target.load_adapter_pretrained(tmp_path)
    target_state = target.action_adapter.state_dict()
    for name, expected in source.action_adapter.state_dict().items():
        torch.testing.assert_close(target_state[name], expected)

    payload_path = tmp_path / "picf_adapter_config.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    assert len(payload["weights_sha256"]) == 64
    assert payload["host_checkpoint"] == _host_checkpoint_identity().payload
    assert not list(tmp_path.glob(".*.tmp"))


def test_adapter_checkpoint_rejects_metadata_and_weight_corruption(tmp_path) -> None:
    source = MolmoAct2PICFForConditionalGeneration(
        _host(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
        host_checkpoint_identity=_host_checkpoint_identity(),
    )
    source.save_adapter_pretrained(tmp_path)
    target = MolmoAct2PICFForConditionalGeneration(
        _host(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
        host_checkpoint_identity=_host_checkpoint_identity(),
    )
    config_path = tmp_path / "picf_adapter_config.json"
    original = config_path.read_text(encoding="utf-8")
    payload = json.loads(original)

    payload["host_checkpoint"]["revision"] = "wrong-revision"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="host checkpoint differs"):
        target.load_adapter_pretrained(tmp_path)

    payload = json.loads(original)
    payload["dense_token_dims"]["vision"] = "6"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="positive JSON integers"):
        target.load_adapter_pretrained(tmp_path)

    payload = json.loads(original)
    payload["host_architecture"] = "WrongHost"
    config_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="host architecture differs"):
        target.load_adapter_pretrained(tmp_path)

    config_path.write_text(original, encoding="utf-8")
    weights_path = tmp_path / "picf_adapter.safetensors"
    weights = bytearray(weights_path.read_bytes())
    weights[-1] ^= 1
    weights_path.write_bytes(weights)
    with pytest.raises(ValueError, match="weight hash differs"):
        target.load_adapter_pretrained(tmp_path)


def test_standalone_adapter_checkpoint_requires_verified_host_identity(tmp_path) -> None:
    model = MolmoAct2PICFForConditionalGeneration(
        _host(),
        dense_token_dims={"vision": 6},
        object_address_dim=4,
        object_value_dim=9,
    )

    with pytest.raises(RuntimeError, match="requires an externally verified host"):
        model.save_adapter_pretrained(tmp_path)
    with pytest.raises(RuntimeError, match="requires an externally verified host"):
        model.load_adapter_pretrained(tmp_path)

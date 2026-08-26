from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
adapter_module = pytest.importorskip("picf_next.hosts.lingbot_vla2")
context_module = pytest.importorskip("picf_next.hosts.context")
evidence_module = pytest.importorskip("picf_next.models.evidence")
molmo_module = pytest.importorskip("picf_next.hosts.molmoact2")

LingBotVLA2PICFAdapter = adapter_module.LingBotVLA2PICFAdapter
install_lingbot_vla2_picf_adapter = adapter_module.install_lingbot_vla2_picf_adapter
PICFActionEvidence = context_module.PICFActionEvidence
NativeTokenBank = evidence_module.NativeTokenBank


def _adapter() -> LingBotVLA2PICFAdapter:
    torch.manual_seed(701)
    return LingBotVLA2PICFAdapter(
        hidden_size=16,
        num_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        dense_token_dims={"vjepa": 6, "anytouch": 5},
        object_address_dim=3,
        object_value_dim=9,
    )


def _evidence() -> PICFActionEvidence:
    torch.manual_seed(709)
    vision_valid = torch.tensor([[True, True, True, True], [True, True, False, False]])
    touch_valid = torch.tensor([[True, True, True], [False, False, False]])
    object_valid = torch.tensor([[True, True, False], [True, False, False]])
    return PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "vjepa",
                torch.randn(2, 4, 6) * vision_valid.unsqueeze(-1),
                vision_valid,
            ),
            NativeTokenBank(
                "anytouch",
                torch.randn(2, 3, 5) * touch_valid.unsqueeze(-1),
                touch_valid,
                group_id=torch.tensor([[5, 5, 5], [-1, -1, -1]], dtype=torch.long),
            ),
        ),
        object_address=(
            torch.nn.functional.normalize(torch.randn(2, 3, 3), dim=-1) * object_valid.unsqueeze(-1)
        ),
        object_value=torch.randn(2, 3, 9) * object_valid.unsqueeze(-1),
        object_valid=object_valid,
        object_log_prior=torch.tensor([[-0.1, -0.4, 0.0], [-0.2, 0.0, 0.0]]),
    )


def test_shared_action_evidence_contract_is_reexported_by_molmo() -> None:
    assert molmo_module.PICFActionEvidence is PICFActionEvidence


def test_zero_gates_exactly_preserve_lingbot_action_hidden_states() -> None:
    adapter = _adapter().eval()
    evidence = _evidence()
    context = adapter.prepare_picf_context(evidence)
    torch.manual_seed(719)
    hidden = torch.randn(2, 5, 16)

    with torch.no_grad():
        without_context = adapter.apply_layer(hidden, layer_index=1, context=None)
        with_context = adapter.apply_layer(hidden, layer_index=1, context=context)

    assert without_context is hidden
    assert torch.equal(with_context, hidden)
    assert torch.equal(adapter.dense_gates, torch.zeros(3))
    assert torch.equal(adapter.object_gates, torch.zeros(3))


class _FakePatchedLingBotHost(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            qwen_expert_config=SimpleNamespace(
                hidden_size=16,
                num_hidden_layers=3,
                num_attention_heads=4,
                num_key_value_heads=2,
                head_dim=4,
            )
        )
        self.action_layer_adapter = None

    def set_action_layer_adapter(self, adapter) -> None:
        self.action_layer_adapter = adapter


class _FakePatchedLingBotPolicy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.qwenvl_with_expert = _FakePatchedLingBotHost()


def test_install_registers_adapter_before_optimizer_or_fsdp() -> None:
    policy = _FakePatchedLingBotPolicy()
    adapter = _adapter()

    install_lingbot_vla2_picf_adapter(policy, adapter)

    host = policy.model.qwenvl_with_expert
    assert host.action_layer_adapter is adapter
    names = dict(policy.named_parameters())
    assert any("action_layer_adapter.dense_branches" in name for name in names)
    install_lingbot_vla2_picf_adapter(policy, adapter)


def test_install_rejects_dimension_mismatch_and_unpatched_policy() -> None:
    policy = _FakePatchedLingBotPolicy()
    adapter = _adapter()
    adapter.hidden_size = 15
    with pytest.raises(ValueError, match="dimensions differ"):
        install_lingbot_vla2_picf_adapter(policy, adapter)
    with pytest.raises(TypeError, match="pinned patched LingBot V2"):
        install_lingbot_vla2_picf_adapter(torch.nn.Linear(2, 2), _adapter())


def test_context_retains_every_dense_token_and_does_not_mutate_inputs() -> None:
    adapter = _adapter().eval()
    evidence = _evidence()
    dense_before = tuple(bank.tokens.clone() for bank in evidence.dense_banks)
    address_before = evidence.object_address.clone()
    value_before = evidence.object_value.clone()
    context = adapter.prepare_picf_context(evidence)

    assert context.dense_key is not None
    assert context.dense_value is not None
    assert context.dense_valid is not None
    assert context.dense_key.shape == (2, 7, 2, 4)
    assert context.dense_value.shape == (2, 7, 2, 4)
    assert torch.equal(
        context.dense_valid,
        torch.cat(tuple(bank.valid for bank in evidence.dense_banks), dim=1),
    )
    assert context.object_key is not None
    assert context.object_value is not None
    assert context.object_key.shape == (2, 3, 2, 4)
    assert context.object_value.shape == (2, 3, 2, 4)
    assert all(
        torch.equal(bank.tokens, before)
        for bank, before in zip(evidence.dense_banks, dense_before, strict=True)
    )
    assert torch.equal(evidence.object_address, address_before)
    assert torch.equal(evidence.object_value, value_before)


def test_object_key_is_address_only_and_value_is_dynamic_only() -> None:
    adapter = _adapter().eval()
    evidence = _evidence()
    base = adapter.prepare_picf_context(evidence)
    changed_value = adapter.prepare_picf_context(
        PICFActionEvidence(
            dense_banks=evidence.dense_banks,
            object_address=evidence.object_address,
            object_value=(evidence.object_value + evidence.object_valid.unsqueeze(-1)),
            object_valid=evidence.object_valid,
            object_log_prior=evidence.object_log_prior,
        )
    )
    changed_address = adapter.prepare_picf_context(
        PICFActionEvidence(
            dense_banks=evidence.dense_banks,
            object_address=torch.roll(evidence.object_address, shifts=1, dims=-1),
            object_value=evidence.object_value,
            object_valid=evidence.object_valid,
            object_log_prior=evidence.object_log_prior,
        )
    )

    torch.testing.assert_close(base.object_key, changed_value.object_key)
    assert not torch.equal(base.object_value, changed_value.object_value)
    assert not torch.equal(base.object_key, changed_address.object_key)
    torch.testing.assert_close(base.object_value, changed_address.object_value)


def test_lingbot_object_attention_uses_posterior_existence_log_prior() -> None:
    adapter = _adapter().eval()
    adapter.object_branches[0].gate.data.fill_(0.5)
    evidence = _evidence()
    base_context = adapter.prepare_picf_context(evidence)
    assert torch.equal(base_context.object_log_prior, evidence.object_log_prior)
    changed_prior = torch.tensor(
        [[-4.0, -0.01, 0.0], [-0.01, 0.0, 0.0]],
        dtype=evidence.object_log_prior.dtype,
    )
    changed_context = adapter.prepare_picf_context(
        PICFActionEvidence(
            dense_banks=evidence.dense_banks,
            object_address=evidence.object_address,
            object_value=evidence.object_value,
            object_valid=evidence.object_valid,
            object_log_prior=changed_prior,
        )
    )
    hidden = torch.randn(2, 5, 16)
    with torch.no_grad():
        expected = adapter.apply_layer(hidden, layer_index=0, context=base_context)
        actual = adapter.apply_layer(hidden, layer_index=0, context=changed_context)
    assert not torch.equal(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1], atol=0.0, rtol=0.0)


def test_lingbot_dense_context_consumes_persistent_ownership_address() -> None:
    adapter = _adapter().eval()
    evidence = _evidence()
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

    plain = adapter.prepare_picf_context(evidence)
    actual = adapter.prepare_picf_context(structured)

    assert plain.dense_key is not None and actual.dense_key is not None
    assert plain.dense_value is not None and actual.dense_value is not None
    assert not torch.equal(plain.dense_key, actual.dense_key)
    assert not torch.equal(plain.dense_value, actual.dense_value)
    assert actual.dense_key.shape[1] == 7


def test_nonzero_gates_change_output_and_propagate_gradients() -> None:
    adapter = _adapter().train()
    context = adapter.prepare_picf_context(_evidence())
    adapter.dense_branches[0].gate.data.fill_(0.2)
    adapter.object_branches[0].gate.data.fill_(0.3)
    torch.manual_seed(727)
    hidden = torch.randn(2, 5, 16, requires_grad=True)

    output = adapter.apply_layer(hidden, layer_index=0, context=context)

    assert not torch.equal(output, hidden)
    output.square().mean().backward()
    assert adapter.dense_key_projection["vjepa"].weight.grad is not None
    assert adapter.dense_value_projection["anytouch"].weight.grad is not None
    assert adapter.object_key_projection.weight.grad is not None
    assert adapter.object_value_projection.weight.grad is not None
    assert adapter.dense_branches[0].gate.grad is not None
    assert adapter.object_branches[0].gate.grad is not None


def test_missing_tactile_is_a_per_sample_zero_residual_without_fake_tokens() -> None:
    adapter = _adapter().eval()
    adapter.dense_branches[0].gate.data.fill_(0.4)
    tokens = torch.zeros(2, 4, 6)
    tokens[0] = torch.randn(4, 6)
    evidence = PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "vjepa",
                tokens,
                torch.tensor([[True, True, True, True], [False, False, False, False]]),
            ),
        ),
        object_address=None,
        object_value=None,
        object_valid=None,
    )
    context = adapter.prepare_picf_context(evidence)
    hidden = torch.randn(2, 5, 16)

    with torch.no_grad():
        output = adapter.apply_layer(hidden, layer_index=0, context=context)

    assert not torch.equal(output[0], hidden[0])
    torch.testing.assert_close(output[1], hidden[1], atol=0.0, rtol=0.0)
    assert context.dense_valid is not None
    assert context.dense_valid.shape[1] == 4


def test_zero_length_dense_banks_do_not_create_empty_attention_context() -> None:
    adapter = _adapter().eval()
    evidence = PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "vjepa",
                torch.zeros(2, 0, 6),
                torch.zeros(2, 0, dtype=torch.bool),
            ),
            NativeTokenBank(
                "anytouch",
                torch.zeros(2, 0, 5),
                torch.zeros(2, 0, dtype=torch.bool),
            ),
        ),
        object_address=None,
        object_value=None,
        object_valid=None,
    )

    context = adapter.prepare_picf_context(evidence)

    assert context.dense_key is None
    assert context.dense_value is None
    assert context.dense_valid is None
    assert context.object_key is None
    assert context.object_value is None
    assert context.object_valid is None


def test_rejects_partial_object_bank_nonzero_padding_and_invalid_layer() -> None:
    adapter = _adapter()
    with pytest.raises(ValueError, match="all present or absent"):
        adapter.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(),
                object_address=torch.zeros(1, 2, 3),
                object_value=None,
                object_valid=None,
            )
        )
    with pytest.raises(ValueError, match="padding"):
        adapter.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(
                    NativeTokenBank(
                        "vjepa",
                        torch.ones(1, 2, 6),
                        torch.zeros(1, 2, dtype=torch.bool),
                    ),
                ),
                object_address=None,
                object_value=None,
                object_valid=None,
            )
        )
    with pytest.raises(IndexError, match="layer_index"):
        adapter.apply_layer(torch.zeros(1, 2, 16), layer_index=3, context=None)
    with pytest.raises(TypeError, match="layer_index"):
        adapter.apply_layer(torch.zeros(1, 2, 16), layer_index=True, context=None)

    with pytest.raises(ValueError, match="share one batch size"):
        adapter.prepare_picf_context(
            PICFActionEvidence(
                dense_banks=(
                    NativeTokenBank(
                        "vjepa",
                        torch.zeros(2, 1, 6),
                        torch.ones(2, 1, dtype=torch.bool),
                    ),
                ),
                object_address=torch.zeros(1, 1, 3),
                object_value=torch.zeros(1, 1, 9),
                object_valid=torch.ones(1, 1, dtype=torch.bool),
                object_log_prior=torch.zeros(1, 1),
            )
        )

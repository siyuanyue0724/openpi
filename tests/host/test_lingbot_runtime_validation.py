from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
adapter_module = pytest.importorskip("picf_next.hosts.lingbot_vla2")
context_module = pytest.importorskip("picf_next.hosts.context")
evidence_module = pytest.importorskip("picf_next.models.evidence")

LingBotVLA2PICFAdapter = adapter_module.LingBotVLA2PICFAdapter
PICFActionEvidence = context_module.PICFActionEvidence
NativeTokenBank = evidence_module.NativeTokenBank


def _adapter(*, validate_tensor_values: bool) -> LingBotVLA2PICFAdapter:
    return LingBotVLA2PICFAdapter(
        hidden_size=16,
        num_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        dense_token_dims={"vision": 6},
        object_address_dim=3,
        object_value_dim=5,
        validate_tensor_values=validate_tensor_values,
    )


def _evidence() -> PICFActionEvidence:
    valid = torch.tensor([[True, True, False], [False, False, False]])
    object_valid = torch.tensor([[True, False], [False, False]])
    return PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "vision",
                torch.randn(2, 3, 6) * valid.unsqueeze(-1),
                valid,
            ),
        ),
        object_address=(
            torch.nn.functional.normalize(torch.randn(2, 2, 3), dim=-1) * object_valid.unsqueeze(-1)
        ),
        object_value=torch.randn(2, 2, 5) * object_valid.unsqueeze(-1),
        object_valid=object_valid,
        object_log_prior=torch.tensor([[-0.2, 0.0], [0.0, 0.0]]),
    )


def test_lingbot_runtime_validation_modes_are_math_identical() -> None:
    torch.manual_seed(1031)
    full = _adapter(validate_tensor_values=True).eval()
    metadata = _adapter(validate_tensor_values=False).eval()
    metadata.load_state_dict(full.state_dict())
    with torch.no_grad():
        full.dense_branches[0].gate.fill_(1.0)
        full.object_branches[0].gate.fill_(1.0)
        metadata.load_state_dict(full.state_dict())
    evidence = _evidence()
    full_context = full.prepare_picf_context(evidence)
    metadata_context = metadata.prepare_picf_context(evidence)
    hidden = torch.randn(2, 4, 16)

    full_output = full.apply_layer(hidden, layer_index=0, context=full_context)
    metadata_output = metadata.apply_layer(hidden, layer_index=0, context=metadata_context)

    torch.testing.assert_close(metadata_output, full_output, atol=0.0, rtol=0.0)
    torch.testing.assert_close(metadata_output[1], hidden[1], atol=0.0, rtol=0.0)


def test_lingbot_metadata_mode_keeps_structural_validation() -> None:
    adapter = _adapter(validate_tensor_values=False)
    evidence = _evidence()
    malformed = PICFActionEvidence(
        dense_banks=(
            NativeTokenBank(
                "vision",
                torch.zeros(2, 3, 5),
                evidence.dense_banks[0].valid,
            ),
        ),
        object_address=evidence.object_address,
        object_value=evidence.object_value,
        object_valid=evidence.object_valid,
        object_log_prior=evidence.object_log_prior,
    )

    with pytest.raises(ValueError, match="feature width"):
        adapter.prepare_picf_context(malformed)


def test_lingbot_runtime_validation_flag_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="boolean"):
        LingBotVLA2PICFAdapter(
            hidden_size=16,
            num_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            dense_token_dims={"vision": 6},
            object_address_dim=3,
            object_value_dim=5,
            validate_tensor_values=1,
        )

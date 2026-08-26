from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")
smoke = pytest.importorskip("tools.smoke_molmoact2_full_weight")
official_smoke = pytest.importorskip("tools.smoke_molmoact2_lerobot_full_weight")
adapter_module = pytest.importorskip("picf_next.hosts.molmoact2")


def test_full_weight_smoke_input_boundary_drops_every_target_field() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.zeros(1, 1, 3),
        "attention_mask": torch.ones(1, 2),
        "labels": torch.tensor([[7, 8]]),
        "object_mask_target": torch.ones(1, 2),
        "simulator_instance_id": torch.tensor([19]),
    }

    moved = smoke._move_inputs(inputs, torch.device("cpu"))

    assert set(moved) == {"input_ids", "pixel_values", "attention_mask"}
    assert "labels" not in moved
    assert "object_mask_target" not in moved
    assert "simulator_instance_id" not in moved


def test_full_weight_smoke_requires_processor_input_ids() -> None:
    with pytest.raises(RuntimeError, match="no input_ids"):
        smoke._move_inputs({"pixel_values": torch.zeros(1, 1, 3)}, torch.device("cpu"))


@pytest.mark.parametrize("revision", [None, "main", "A" * 40, "a" * 39, "g" * 40])
def test_full_weight_smoke_requires_immutable_revision(revision: object) -> None:
    with pytest.raises(ValueError, match="exact lowercase 40-character commit SHA"):
        smoke._strict_revision(revision)


def test_official_policy_m0_input_boundary_drops_every_target_field() -> None:
    inputs = {
        "input_ids": torch.tensor([[1, 2]]),
        "pixel_values": torch.zeros(1, 1, 3),
        "attention_mask": torch.ones(1, 2),
        "labels": torch.tensor([[7, 8]]),
        "object_mask_target": torch.ones(1, 2),
        "simulator_instance_id": torch.tensor([19]),
    }

    moved = official_smoke._move_processor_inputs(
        inputs,
        torch.device("cpu"),
        torch,
    )

    assert set(moved) == {"input_ids", "pixel_values", "attention_mask"}


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_official_policy_m0_uses_valid_native_evidence_without_replacement(
    dtype: torch.dtype,
) -> None:
    tokens = torch.randn(1, 7, 6, dtype=dtype)
    bank = adapter_module.NativeTokenBank(
        "molmo_vision_patch",
        tokens,
        torch.ones(1, 7, dtype=torch.bool),
    )
    evidence, tensors = official_smoke._make_m0_evidence(
        torch=torch,
        picf_action_evidence=adapter_module.PICFActionEvidence,
        dense_bank=bank,
        device=torch.device("cpu"),
        dtype=dtype,
        seed=17,
        object_count=3,
        address_width=4,
        value_width=5,
    )

    evidence.validate_object_identity()
    assert evidence.dense_banks[0] is bank
    assert tensors["dense_tokens"].data_ptr() == tokens.data_ptr()
    assert tensors["dense_valid"].sum().item() == 7
    assert tensors["dense_ownership"].shape == (1, 7, 4)
    assert torch.equal(tensors["object_log_prior"], torch.zeros(1, 3, dtype=dtype))

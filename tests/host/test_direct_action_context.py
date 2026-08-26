from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
training = pytest.importorskip("picf_next.hosts.molmoact2_training")
evidence_module = pytest.importorskip("picf_next.models.evidence")

action_evidence_from_core = training.action_evidence_from_core
NativeTokenBank = evidence_module.NativeTokenBank


def _output() -> SimpleNamespace:
    measurement = NativeTokenBank(
        modality="current",
        tokens=torch.randn(2, 3, 5),
        valid=torch.ones(2, 3, dtype=torch.bool),
        current_measurement_valid=torch.ones(2, 3, dtype=torch.bool),
    )
    measurement_ownership = torch.zeros(2, 3, 3)
    measurement_ownership[..., 0] = 1.0
    address = torch.nn.functional.normalize(torch.randn(2, 2, 4), dim=-1)
    return SimpleNamespace(
        projection=SimpleNamespace(native_banks=(measurement,)),
        action_bank=SimpleNamespace(
            address=address,
            value=torch.randn(2, 2, 7),
            valid=torch.ones(2, 2, dtype=torch.bool),
            log_prior=torch.zeros(2, 2),
        ),
        dense_ownership=(measurement_ownership,),
    )


def _context(*, current: bool = False) -> NativeTokenBank:
    valid = torch.tensor([[True, True, False], [True, False, False]])
    timestamps = torch.tensor([[0.0, 0.1, 0.0], [0.2, 0.0, 0.0]], dtype=torch.float32)
    return NativeTokenBank(
        modality="video_context",
        tokens=torch.randn(2, 3, 6) * valid.unsqueeze(-1),
        valid=valid,
        timestamps=timestamps,
        current_measurement_valid=torch.full_like(valid, current),
    )


def test_direct_video_context_is_retained_but_receives_only_dustbin_ownership() -> None:
    evidence = action_evidence_from_core(
        _output(),
        direct_context_banks=(_context(),),
    )

    assert tuple(bank.modality for bank in evidence.dense_banks) == (
        "current",
        "video_context",
    )
    assert evidence.dense_ownership is not None
    context_ownership = evidence.dense_ownership[1]
    torch.testing.assert_close(
        context_ownership[..., :-1],
        torch.zeros_like(context_ownership[..., :-1]),
    )
    torch.testing.assert_close(
        context_ownership[..., -1],
        torch.ones_like(context_ownership[..., -1]),
    )
    owner_addresses = evidence.ownership_weighted_addresses()
    assert owner_addresses[1] is not None
    torch.testing.assert_close(owner_addresses[1], torch.zeros_like(owner_addresses[1]))


def test_direct_context_cannot_smuggle_a_current_measurement() -> None:
    with pytest.raises(ValueError, match="cannot contain current"):
        action_evidence_from_core(
            _output(),
            direct_context_banks=(_context(current=True),),
        )


def test_no_posterior_arm_retains_every_dense_token_without_object_addressing() -> None:
    output = _output()
    context = _context()

    evidence = action_evidence_from_core(
        output,
        direct_context_banks=(context,),
        include_posterior=False,
    )

    assert evidence.dense_banks == (*output.projection.native_banks, context)
    assert evidence.dense_ownership is None
    assert evidence.object_address is None
    assert evidence.object_value is None
    assert evidence.object_valid is None
    assert evidence.object_log_prior is None
    assert evidence.ownership_weighted_addresses() == (None, None)

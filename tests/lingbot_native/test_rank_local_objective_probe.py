from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tools.probe_fsdp2_rank_local_objective import (
    _ConditionalObjectiveProbe,
    _objective,
    _require_environment,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/probe_fsdp2_rank_local_objective.py"


def test_rank_local_objective_probe_preserves_required_contract() -> None:
    source = TOOL.read_text(encoding="utf-8")
    for fragment in (
        '"later-detached",',
        '"later-zero",',
        "for call_index in range(unroll_calls)",
        "fully_shard(model.optional_head, **options)",
        "checkpoint(self.encoder, value, use_reentrant=False)",
        'set_checkpoint_early_stop(False)',
        "return required + optional * 0",
        "loss.backward()",
        'dist.init_process_group(\n        "nccl",',
        '"schema": "picf-next.fsdp2-rank-local-objective-probe.v1"',
    ):
        assert fragment in source


def test_rank_local_objective_probe_requires_two_cuda_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(RuntimeError, match="exactly two local CUDA ranks"):
        _require_environment()


def test_rank_zero_keeps_exact_value_and_autograd_connectivity() -> None:
    required = torch.tensor(2.0, requires_grad=True)
    optional = torch.tensor(3.0, requires_grad=True)
    loss = _objective(
        required=required,
        optional=optional,
        connectivity="rank-zero",
        local_rank=1,
        call_index=0,
    )
    assert loss.item() == 2.0
    loss.backward()
    torch.testing.assert_close(required.grad, torch.ones(()))
    torch.testing.assert_close(optional.grad, torch.zeros(()))


def test_rank_detached_omits_optional_autograd_connectivity() -> None:
    required = torch.tensor(2.0, requires_grad=True)
    optional = torch.tensor(3.0, requires_grad=True)
    loss = _objective(
        required=required,
        optional=optional,
        connectivity="rank-detached",
        local_rank=1,
        call_index=0,
    )
    loss.backward()
    torch.testing.assert_close(required.grad, torch.ones(()))
    assert optional.grad is None


def test_later_zero_preserves_only_later_optional_connectivity() -> None:
    required = torch.tensor(2.0, requires_grad=True)
    optional = torch.tensor(3.0, requires_grad=True)
    first = _objective(
        required=required,
        optional=optional,
        connectivity="later-zero",
        local_rank=0,
        call_index=0,
    )
    later = _objective(
        required=required,
        optional=optional,
        connectivity="later-zero",
        local_rank=0,
        call_index=1,
    )
    assert first.item() == 5.0
    assert later.item() == 2.0
    later.backward()
    torch.testing.assert_close(required.grad, torch.ones(()))
    torch.testing.assert_close(optional.grad, torch.zeros(()))


def test_rank_local_objective_probe_model_returns_two_scalar_losses() -> None:
    model = _ConditionalObjectiveProbe(width=8, hidden_width=16)
    required, optional = model(torch.ones(1, 1, 8))
    assert required.shape == optional.shape == ()
    assert torch.isfinite(required) and torch.isfinite(optional)


def test_rank_local_objective_probe_checkpointed_model_returns_two_scalar_losses() -> None:
    model = _ConditionalObjectiveProbe(
        width=8,
        hidden_width=16,
        activation_checkpointing=True,
    )
    required, optional = model(torch.ones(1, 1, 8, requires_grad=True))
    (required + optional).backward()
    assert torch.isfinite(required) and torch.isfinite(optional)
    assert any(parameter.grad is not None for parameter in model.encoder.parameters())

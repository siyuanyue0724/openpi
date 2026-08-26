from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from picf_next.lingbot_native.gradient_audit_runtime import (
    cpu_pair_moments,
    distributed_pair_rows,
    snapshot_local_gradients,
)


def test_cpu_pair_moments_use_exact_float64_accumulation() -> None:
    first = torch.tensor([1.0, 2.0], dtype=torch.float32)
    second = torch.tensor([3.0, 4.0], dtype=torch.float32)
    assert cpu_pair_moments(first, second, torch_module=torch) == (11.0, 5.0, 25.0, 2.0)


def test_snapshot_and_distributed_rows_cover_the_same_trainable_scope() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    model.weight.data.copy_(torch.tensor([[1.0, -1.0]]))
    model(torch.tensor([[1.0, 2.0]])).sum().backward()
    first = snapshot_local_gradients(model, torch_module=torch)

    model.zero_grad(set_to_none=True)
    model(torch.tensor([[3.0, 4.0]])).sum().backward()

    class _Dist:
        ReduceOp = SimpleNamespace(SUM="sum")

        @staticmethod
        def all_reduce(tensor, *, op) -> None:
            assert op == "sum"
            assert tensor.dtype == torch.float64

    names, rows = distributed_pair_rows(
        model,
        first_gradients=first,
        device=torch.device("cpu"),
        dist=_Dist,
        torch_module=torch,
    )
    assert names == ("weight",)
    assert rows == [[11.0, 5.0, 25.0, 2.0]]

    model.bias = torch.nn.Parameter(torch.zeros(1))
    with pytest.raises(RuntimeError, match="scope changed"):
        distributed_pair_rows(
            model,
            first_gradients=first,
            device=torch.device("cpu"),
            dist=_Dist,
            torch_module=torch,
        )


def test_snapshot_rejects_frozen_parameter_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False)
    parameter.grad = torch.tensor([1.0])
    model = torch.nn.ParameterList([parameter])
    with pytest.raises(RuntimeError, match="frozen-parameter"):
        snapshot_local_gradients(model, torch_module=torch)

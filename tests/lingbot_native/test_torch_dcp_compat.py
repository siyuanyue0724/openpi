from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from picf_next.lingbot_native.torch_dcp_compat import (  # noqa: E402
    LINGBOT_DCP_OPTIMIZER_STATE_PREFIX,
    UPSTREAM_COMMIT,
    install_torch_2_8_sparse_optimizer_state_backport,
    prune_synthetic_optimizer_state_from_dcp_template,
)


@pytest.fixture()
def vulnerable_torch_2_8(monkeypatch: pytest.MonkeyPatch):
    state_dict_module = importlib.import_module("torch.distributed.checkpoint.state_dict")
    original = state_dict_module._split_optim_state_dict
    monkeypatch.delattr(
        state_dict_module,
        "_picf_next_sparse_optimizer_state_backport",
        raising=False,
    )
    monkeypatch.setattr(state_dict_module, "_split_optim_state_dict", original)
    return state_dict_module


def _partially_initialized_adamw():
    model = torch.nn.Linear(4, 2, bias=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model.weight.mean().backward()
    optimizer.step()
    assert model.weight in optimizer.state
    assert model.bias not in optimizer.state
    return model, optimizer


class _OptimizerState:
    def __init__(
        self,
        model: Any,
        optimizer: Any,
        *,
        checkpoint_metadata_keys: frozenset[str] | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.checkpoint_metadata_keys = checkpoint_metadata_keys

    def state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict

        return {"optim": get_optimizer_state_dict(self.model, self.optimizer)}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import (
            StateDictOptions,
            set_optimizer_state_dict,
        )

        optim_state_dict = state_dict["optim"]
        if self.checkpoint_metadata_keys is not None:
            prune_synthetic_optimizer_state_from_dcp_template(
                optim_state_dict,
                checkpoint_metadata_keys=self.checkpoint_metadata_keys,
            )
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            optim_state_dict,
            options=StateDictOptions(strict=False),
        )


def test_upstream_backport_loads_unused_parameter_as_lazy_state(
    vulnerable_torch_2_8,
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    source_model, source_optimizer = _partially_initialized_adamw()
    saved = get_optimizer_state_dict(source_model, source_optimizer)

    target_model = torch.nn.Linear(4, 2, bias=True)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
    report = install_torch_2_8_sparse_optimizer_state_backport(torch)
    set_optimizer_state_dict(
        target_model,
        target_optimizer,
        saved,
        options=StateDictOptions(strict=False),
    )

    assert report["upstream_commit"] == UPSTREAM_COMMIT
    assert target_model.weight in target_optimizer.state
    assert target_model.bias not in target_optimizer.state
    for field in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            target_optimizer.state[target_model.weight][field],
            source_optimizer.state[source_model.weight][field],
        )


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled.*:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated.*:UserWarning")
def test_upstream_backport_closes_actual_dcp_cold_restore(
    vulnerable_torch_2_8,
    tmp_path: Path,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    source_model, source_optimizer = _partially_initialized_adamw()
    checkpoint = tmp_path / "optimizer"
    dcp.save(
        {"state": _OptimizerState(source_model, source_optimizer)},
        storage_writer=FileSystemWriter(checkpoint),
    )
    metadata_keys = frozenset(FileSystemReader(checkpoint).read_metadata().state_dict_metadata)
    assert any(key.startswith(LINGBOT_DCP_OPTIMIZER_STATE_PREFIX) for key in metadata_keys)

    target_model = torch.nn.Linear(4, 2, bias=True)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
    install_torch_2_8_sparse_optimizer_state_backport(torch)
    dcp.load(
        {
            "state": _OptimizerState(
                target_model,
                target_optimizer,
                checkpoint_metadata_keys=metadata_keys,
            )
        },
        storage_reader=FileSystemReader(checkpoint),
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    assert target_model.weight in target_optimizer.state
    assert target_model.bias not in target_optimizer.state
    for field in ("step", "exp_avg", "exp_avg_sq"):
        torch.testing.assert_close(
            target_optimizer.state[target_model.weight][field],
            source_optimizer.state[source_model.weight][field],
        )


@pytest.mark.filterwarnings("ignore:torch.distributed is disabled.*:UserWarning")
@pytest.mark.filterwarnings("ignore:TypedStorage is deprecated.*:UserWarning")
def test_backport_restores_combined_optimizer_sequence_without_synthetic_state(
    vulnerable_torch_2_8,
    tmp_path: Path,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    source_model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=True),
        torch.nn.Linear(3, 2, bias=True),
    )
    source_optimizers = [
        torch.optim.AdamW(source_model[0].parameters(), lr=0.01),
        torch.optim.AdamW(source_model[1].parameters(), lr=0.02),
    ]
    source_model[0].weight.mean().backward()
    source_model[1].weight.mean().backward()
    for optimizer in source_optimizers:
        optimizer.step()

    checkpoint = tmp_path / "combined-optimizer"
    dcp.save(
        {"state": _OptimizerState(source_model, source_optimizers)},
        storage_writer=FileSystemWriter(checkpoint),
    )
    reader = FileSystemReader(checkpoint)
    metadata_keys = frozenset(reader.read_metadata().state_dict_metadata)

    target_model = torch.nn.Sequential(
        torch.nn.Linear(4, 3, bias=True),
        torch.nn.Linear(3, 2, bias=True),
    )
    target_optimizers = [
        torch.optim.AdamW(target_model[0].parameters(), lr=0.01),
        torch.optim.AdamW(target_model[1].parameters(), lr=0.02),
    ]
    install_torch_2_8_sparse_optimizer_state_backport(torch)
    dcp.load(
        {
            "state": _OptimizerState(
                target_model,
                target_optimizers,
                checkpoint_metadata_keys=metadata_keys,
            )
        },
        storage_reader=reader,
        planner=DefaultLoadPlanner(allow_partial_load=True),
    )

    for source_layer, target_layer, source_optimizer, target_optimizer in zip(
        source_model,
        target_model,
        source_optimizers,
        target_optimizers,
        strict=True,
    ):
        assert target_layer.weight in target_optimizer.state
        assert target_layer.bias not in target_optimizer.state
        for field in ("step", "exp_avg", "exp_avg_sq"):
            torch.testing.assert_close(
                target_optimizer.state[target_layer.weight][field],
                source_optimizer.state[source_layer.weight][field],
            )


def test_upstream_backport_preserves_strict_missing_state_failure(
    vulnerable_torch_2_8,
) -> None:
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_optimizer_state_dict,
        set_optimizer_state_dict,
    )

    source_model, source_optimizer = _partially_initialized_adamw()
    saved = get_optimizer_state_dict(source_model, source_optimizer)
    target_model = torch.nn.Linear(4, 2, bias=True)
    target_optimizer = torch.optim.AdamW(target_model.parameters(), lr=0.01)
    install_torch_2_8_sparse_optimizer_state_backport(torch)

    with pytest.raises(RuntimeError, match="has no saved optimizer state"):
        set_optimizer_state_dict(
            target_model,
            target_optimizer,
            saved,
            options=StateDictOptions(strict=True),
        )


def test_metadata_pruning_rejects_partially_saved_parameter_state() -> None:
    state = {
        "state": {
            "layer.weight": {
                "step": torch.tensor(1.0),
                "exp_avg": torch.zeros(2),
                "exp_avg_sq": torch.zeros(2),
            }
        }
    }
    metadata = {
        "state.optim.state.layer.weight.step",
        "state.optim.state.layer.weight.exp_avg",
    }
    with pytest.raises(RuntimeError, match="is incomplete"):
        prune_synthetic_optimizer_state_from_dcp_template(
            state,
            checkpoint_metadata_keys=metadata,
        )


def test_upstream_backport_rejects_unreviewed_torch_2_8_source(
    vulnerable_torch_2_8,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        vulnerable_torch_2_8,
        "_split_optim_state_dict",
        lambda model, optim, state_dict, info: state_dict,
    )
    with pytest.raises(RuntimeError, match="differs from the audited source"):
        install_torch_2_8_sparse_optimizer_state_backport(torch)


def test_upstream_backport_is_idempotent(vulnerable_torch_2_8) -> None:
    first = install_torch_2_8_sparse_optimizer_state_backport(torch)
    second = install_torch_2_8_sparse_optimizer_state_backport(torch)
    assert first["status"] == "installed"
    assert second == {
        "schema": "picf-next.torch-dcp-sparse-optimizer-state.v1",
        "status": "already-installed",
        "torch_version": torch.__version__,
        "upstream_commit": UPSTREAM_COMMIT,
    }

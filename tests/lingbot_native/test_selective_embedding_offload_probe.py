from __future__ import annotations

import ast
from pathlib import Path

import pytest
import torch

from tools.probe_fsdp2_selective_embedding_offload import (
    _EmbeddingProbe,
    _require_environment,
    _token_ids,
)

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/probe_fsdp2_selective_embedding_offload.py"


def test_selective_embedding_probe_preserves_required_contract() -> None:
    source = TOOL.read_text(encoding="utf-8")
    ast.parse(source)
    for fragment in (
        "CPUOffloadPolicy(pin_memory=False)",
        'if embedding_placement != "root":',
        "fully_shard(model.embedding, **embedding_options)",
        "fully_shard(\n        model.body,",
        'dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)',
        "get_state_dict(model, optimizer)",
        "set_state_dict(",
        "torch.testing.assert_close(",
        '"--unroll-calls"',
        "for call_index in range(unroll_calls)",
        '"schema": "picf-next.fsdp2-selective-embedding-offload-probe.v1"',
    ):
        assert fragment in source


def test_selective_embedding_probe_requires_two_cuda_ranks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("LOCAL_RANK", "0")
    with pytest.raises(RuntimeError, match="exactly two"):
        _require_environment(tmp_path / "checkpoint")


def test_selective_embedding_probe_requires_absolute_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("torch.cuda.device_count", lambda: 2)
    with pytest.raises(RuntimeError, match="must be absolute"):
        _require_environment(Path("checkpoint"))


def test_selective_embedding_probe_tokens_are_deterministic() -> None:
    first = _token_ids(
        vocab_size=17,
        token_count=9,
        step=3,
        device=torch.device("cpu"),
    )
    second = _token_ids(
        vocab_size=17,
        token_count=9,
        step=3,
        device=torch.device("cpu"),
    )
    assert torch.equal(first, second)
    assert int(first.min()) >= 0
    assert int(first.max()) < 17


def test_selective_embedding_probe_model_produces_scalar_loss() -> None:
    torch.manual_seed(7)
    model = _EmbeddingProbe(vocab_size=17, width=8)
    loss = model(torch.tensor([1, 2, 3]))
    assert loss.shape == ()
    assert torch.isfinite(loss)

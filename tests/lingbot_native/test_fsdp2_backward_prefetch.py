from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_BACKWARD_PREFETCH_DEFAULT,
    FSDP2_BACKWARD_PREFETCH_DISABLED,
    configure_fsdp2_backward_prefetch,
    fsdp2_parameter_layout_manifest,
    merge_fsdp2_factual_gradients_from_cpu,
    spill_fsdp2_factual_gradients_to_cpu,
)

ROOT = Path(__file__).resolve().parents[2]


class _FakeChild:
    def __init__(self, *, sharded: bool) -> None:
        self.calls: list[list[object]] = []
        if sharded:
            self.set_modules_to_backward_prefetch = self._set_modules  # type: ignore[attr-defined]

    def _set_modules(self, modules: list[object]) -> None:
        self.calls.append(modules)


class _FakePolicy:
    def __init__(self, children: list[_FakeChild]) -> None:
        self.children = children

    def modules(self) -> list[_FakeChild]:
        return self.children


def test_default_backward_prefetch_preserves_upstream_schedule() -> None:
    child = _FakeChild(sharded=True)
    report = configure_fsdp2_backward_prefetch(
        _FakePolicy([child]),
        mode=FSDP2_BACKWARD_PREFETCH_DEFAULT,
    )

    assert report == {"mode": "default", "configured_module_count": 0}
    assert child.calls == []


def test_disabled_backward_prefetch_sets_self_only_explicit_schedule() -> None:
    sharded = _FakeChild(sharded=True)
    plain = _FakeChild(sharded=False)
    report = configure_fsdp2_backward_prefetch(
        _FakePolicy([sharded, plain]),
        mode=FSDP2_BACKWARD_PREFETCH_DISABLED,
    )

    assert report == {"mode": "disabled", "configured_module_count": 1}
    assert sharded.calls == [[sharded]]
    assert plain.calls == []


def test_disabled_backward_prefetch_fails_without_fsdp2_modules() -> None:
    with pytest.raises(RuntimeError, match="found no sharded modules"):
        configure_fsdp2_backward_prefetch(
            _FakePolicy([_FakeChild(sharded=False)]),
            mode=FSDP2_BACKWARD_PREFETCH_DISABLED,
        )


def test_adr176_scientific_launcher_forwards_backward_prefetch_contract() -> None:
    script = ROOT / "adr176/run_full_modal_2gpu_prefix1500.sh"
    subprocess.run(("bash", "-n", str(script)), check=True)
    text = script.read_text(encoding="utf-8")
    assert "BACKWARD_PREFETCH=${PICF_FSDP2_BACKWARD_PREFETCH:-disabled}" in text
    assert "FACTUAL_GRADIENT_STORAGE=${PICF_SEQUENTIAL_FACTUAL_GRADIENT_STORAGE:-cpu}" in text
    assert "STALL_DIAGNOSTICS=${PICF_DISTRIBUTED_STALL_DIAGNOSTICS:-enabled}" in text
    assert "STALL_TIMEOUT_SECONDS=${PICF_DISTRIBUTED_STALL_TIMEOUT_SECONDS:-90}" in text
    assert "CUDA_ALLOCATOR=${PICF_CUDA_ALLOCATOR:-native}" in text
    assert "STOP_AFTER_STEP=${PICF_STOP_AFTER_STEP:-1500}" in text
    assert (
        "FORCED_CAUSAL_DIAGNOSTIC_STEP=${PICF_ENGINEERING_FORCE_CAUSAL_DIAGNOSTIC_STEP:-0}" in text
    )
    assert "native|expandable-segments" in text
    assert 'export PICF_DISTRIBUTED_STALL_TARGET="$REPO/$TRAINING_ENTRYPOINT"' in text
    assert "TRAINING_ENTRYPOINT=$DIAGNOSTIC_ENTRYPOINT" in text
    assert '"$TRAINING_ENTRYPOINT" \\' in text
    assert '--fsdp2-backward-prefetch "$BACKWARD_PREFETCH"' in text
    assert '--sequential-factual-gradient-storage "$FACTUAL_GRADIENT_STORAGE"' in text
    assert '--cuda-allocator "$CUDA_ALLOCATOR"' in text
    assert '--stop-after-step "$STOP_AFTER_STEP"' in text
    assert '--engineering-force-causal-diagnostic-step "$FORCED_CAUSAL_DIAGNOSTIC_STEP"' in text


def test_adr176_omitted_diagnostic_can_check_following_ordinary_steps() -> None:
    script = ROOT / "adr176/run_full_modal_2gpu_force_omitted_diagnostic.sh"
    subprocess.run(("bash", "-n", str(script)), check=True)
    text = script.read_text(encoding="utf-8")
    assert "STOP_AFTER_STEP=${PICF_DIAGNOSTIC_STOP_AFTER_STEP:-$FORCED_STEP}" in text
    assert "(( STOP_AFTER_STEP >= FORCED_STEP ))" in text
    assert '--stop-after-step "$STOP_AFTER_STEP"' in text
    assert '--engineering-force-omitted-static-step "$FORCED_STEP"' in text
    assert "STALL_DIAGNOSTICS=${PICF_DISTRIBUTED_STALL_DIAGNOSTICS:-enabled}" in text
    assert '"$TRAINING_ENTRYPOINT" \\' in text


class _GradientPair(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        self.second = nn.Parameter(torch.arange(4, dtype=torch.float32))
        self.omitted_only = nn.Parameter(torch.ones(3, dtype=torch.float32))


def test_factual_gradient_cpu_spill_restores_union_before_one_optimizer_step() -> None:
    module = _GradientPair()
    factual_first = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    factual_second = torch.tensor([7.0, 8.0, 9.0, 10.0])
    module.first.grad = factual_first.clone()
    module.second.grad = factual_second.clone()

    spill = spill_fsdp2_factual_gradients_to_cpu(module)

    assert module.first.grad is None
    assert module.second.grad is None
    assert spill.total_bytes == 40
    assert spill.cuda_source_bytes == 0
    assert spill.distributed_shard_count == 0

    omitted_first = torch.full_like(module.first, 0.5)
    omitted_only = torch.full_like(module.omitted_only, 2.0)
    module.first.grad = omitted_first.clone()
    module.omitted_only.grad = omitted_only.clone()
    report = merge_fsdp2_factual_gradients_from_cpu(module, spill, chunk_bytes=4)

    assert torch.equal(module.first.grad, factual_first + omitted_first)
    assert torch.equal(module.second.grad, factual_second)
    assert torch.equal(module.omitted_only.grad, omitted_only)
    assert report == {
        "shard_count": 2,
        "restored_gradient_count": 1,
        "accumulated_gradient_count": 1,
        "total_bytes": 40,
        "chunk_bytes": 4,
    }


def test_factual_gradient_cpu_spill_requires_one_local_gradient() -> None:
    with pytest.raises(RuntimeError, match="found no local gradients"):
        spill_fsdp2_factual_gradients_to_cpu(_GradientPair())


def test_parameter_layout_manifest_is_stable_and_counts_trainable_parameters() -> None:
    module = _GradientPair()

    first = fsdp2_parameter_layout_manifest(module)
    second = fsdp2_parameter_layout_manifest(module)

    assert first == second
    assert first["parameter_count"] == 3
    assert first["trainable_parameter_count"] == 3
    assert len(str(first["manifest_sha256"])) == 64


def test_factual_gradient_merge_rejects_changed_layout() -> None:
    module = _GradientPair()
    module.first.grad = torch.ones_like(module.first)
    spill = spill_fsdp2_factual_gradients_to_cpu(module)
    shard = spill.shards[0]
    changed_shard = replace(
        shard,
        layout=replace(shard.layout, local_stride=(99, 1)),
    )
    changed_spill = replace(spill, shards=(changed_shard,))

    with pytest.raises(RuntimeError, match="layout changed"):
        merge_fsdp2_factual_gradients_from_cpu(module, changed_spill)

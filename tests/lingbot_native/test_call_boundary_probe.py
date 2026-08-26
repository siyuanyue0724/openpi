from __future__ import annotations

from pathlib import Path

import pytest

from tools.probe_fsdp2_call_boundary_groups import _require_environment

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/probe_fsdp2_call_boundary_groups.py"


def test_call_boundary_probe_preserves_required_execution_contract() -> None:
    source = TOOL.read_text()
    for fragment in (
        "compute_kqv=True",
        "output_attention=True",
        "use_reentrant=False",
        'choices=("external", "layer", "wrapper")',
        'choices=("enabled", "disabled")',
        'choices=("nested", "block")',
        '"--unroll-frames"',
        "for frame_index in range(unroll_frames)",
        "checkpoint_wrapper(text)",
        "official checkpoint wrapper probe requires block FSDP topology",
        "checkpointed_call = partial(super().__call__, **kwargs)",
        "set_checkpoint_early_stop(False)",
        'units["action.mlp"]',
        "fully_shard(model.text, **common)",
        "fully_shard(model.action, **common)",
        "cast_forward_inputs=False",
        'choices=("parent", "projection")',
        'default="parent"',
        "self.input_layernorm.weight",
        "self.q_proj.weight",
        "loss.backward()",
        "_gradient_statistics(model)",
        "call-boundary unit {path} produced no finite nonzero gradients",
        "call-boundary probe loss differs across ranks",
        'dist.init_process_group("nccl", device_id=device)',
    ):
        assert fragment in source


def test_call_boundary_probe_requires_two_cuda_ranks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    with pytest.raises(RuntimeError, match="exactly two local CUDA ranks"):
        _require_environment()

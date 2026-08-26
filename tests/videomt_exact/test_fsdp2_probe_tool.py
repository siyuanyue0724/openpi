from __future__ import annotations

import importlib.util
from pathlib import Path

import torch


_TOOL = Path(__file__).resolve().parents[2] / "tools" / "probe_videomt_exact_fsdp2.py"
_SPEC = importlib.util.spec_from_file_location("probe_videomt_exact_fsdp2", _TOOL)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_synthetic_frame_is_deterministic_and_source_normalized() -> None:
    first = probe._synthetic_frame(32, 207)
    second = probe._synthetic_frame(32, 207)

    assert first.shape == (1, 3, 32, 32)
    assert first.dtype == torch.float32
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


def test_local_gradient_receipt_accepts_finite_nonzero_gradients() -> None:
    module = torch.nn.Linear(4, 2)
    module(torch.ones(1, 4)).sum().backward()

    receipt = probe._local_gradient_receipt(module)

    assert receipt == {
        "gradient_tensors": 2,
        "nonzero_gradient_tensors": 2,
        "nonfinite_gradient_tensors": 0,
        "passed": True,
    }

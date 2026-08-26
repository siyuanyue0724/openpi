from __future__ import annotations

from copy import deepcopy

import pytest

from tools.audit_lingbot_attention_backend_parity import (
    ATTENTION_BACKEND_PARITY_SCHEMA,
    _case_passes,
    validate_attention_backend_parity_report,
)
from tools.bootstrap_lingbot_vla2_native import LINGBOT_NATIVE_SOURCE_COMMIT


def _case(dtype: str, mask: str) -> dict[str, object]:
    return {
        "dtype": dtype,
        "finite": True,
        "gradient_max_absolute_error": 0.0,
        "gradient_mean_absolute_error": 0.0,
        "length": 73,
        "mask": mask,
        "output_max_absolute_error": 0.0,
        "output_mean_absolute_error": 0.0,
        "passed": True,
    }


def _report() -> dict[str, object]:
    return {
        "benchmark": {
            "block_mask_build_ms": 2.0,
            "eager_forward_median_ms": 4.0,
            "flex_cached_forward_median_ms": 2.0,
            "length": 512,
            "repeats": 5,
            "speedup": 2.0,
        },
        "cases": [
            _case(dtype, mask)
            for dtype in ("float32", "bfloat16")
            for mask in ("full", "causal", "structured")
        ],
        "cuda_version": "12.8",
        "device": "NVIDIA A100",
        "implementation_sha256": "b" * 64,
        "patch_sha256": "a" * 64,
        "patched_source_sha256": {"source.py": "c" * 64},
        "schema": ATTENTION_BACKEND_PARITY_SCHEMA,
        "seed": 20260725,
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "status": "PASS",
        "torch_version": "2.8.0",
    }


def test_attention_backend_parity_report_recomputes_complete_pass() -> None:
    report = _report()
    assert validate_attention_backend_parity_report(report) == report


def test_attention_backend_parity_report_rejects_forged_case_decision() -> None:
    report = _report()
    report["cases"][0]["output_max_absolute_error"] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError, match="decision was not recomputed"):
        validate_attention_backend_parity_report(report)


def test_attention_backend_parity_report_rejects_forged_status() -> None:
    report = _report()
    report["status"] = "FAIL"
    with pytest.raises(ValueError, match="status was not recomputed"):
        validate_attention_backend_parity_report(report)


def test_attention_backend_parity_report_rejects_forged_speedup() -> None:
    report = _report()
    report["benchmark"]["speedup"] = 3.0  # type: ignore[index]
    with pytest.raises(ValueError, match="speedup was not recomputed"):
        validate_attention_backend_parity_report(report)


def test_attention_backend_parity_report_requires_every_case_once() -> None:
    report = _report()
    duplicate = deepcopy(report["cases"][0])  # type: ignore[index]
    report["cases"][-1] = duplicate  # type: ignore[index]
    with pytest.raises(ValueError, match="repeats a case"):
        validate_attention_backend_parity_report(report)


def test_attention_backend_case_rejects_nonfinite_measurement() -> None:
    case = _case("float32", "full")
    case["output_mean_absolute_error"] = float("nan")
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _case_passes(case)

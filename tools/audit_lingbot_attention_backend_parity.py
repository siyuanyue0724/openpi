#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Audit eager versus flex-cached LingBot joint attention on one CUDA device."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

from picf_next.artifact_io import write_text_durable_exclusive

try:
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        detect_native_patch_state,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import _git_output
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        detect_native_patch_state,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import _git_output  # type: ignore[no-redef]


ATTENTION_BACKEND_PARITY_SCHEMA = "picf-next.lingbot-attention-backend-parity.v1"
_CASE_IDENTITIES = frozenset(
    (dtype, mask) for dtype in ("float32", "bfloat16") for mask in ("full", "causal", "structured")
)
_CASE_THRESHOLDS = {
    "float32": {
        "output_max_absolute_error": 2.0e-6,
        "gradient_max_absolute_error": 3.0e-6,
        "output_mean_absolute_error": 2.0e-7,
        "gradient_mean_absolute_error": 2.0e-7,
    },
    "bfloat16": {
        "output_max_absolute_error": 3.2e-2,
        "gradient_max_absolute_error": 6.3e-2,
        "output_mean_absolute_error": 2.0e-3,
        "gradient_mean_absolute_error": 4.0e-3,
    },
}
_CASE_FIELDS = frozenset(
    {
        "dtype",
        "finite",
        "gradient_max_absolute_error",
        "gradient_mean_absolute_error",
        "length",
        "mask",
        "output_max_absolute_error",
        "output_mean_absolute_error",
        "passed",
    }
)
_REPORT_FIELDS = frozenset(
    {
        "benchmark",
        "cases",
        "cuda_version",
        "device",
        "implementation_sha256",
        "patch_sha256",
        "patched_source_sha256",
        "schema",
        "seed",
        "source_commit",
        "status",
        "torch_version",
    }
)
_IMPLEMENTATION_PATHS = (
    "src/picf_next/artifact_io.py",
    "tools/audit_lingbot_attention_backend_parity.py",
    "tools/bootstrap_lingbot_vla2_native.py",
    "tools/lingbot_vla2_runtime_helpers.py",
)


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _implementation_sha256(root: Path) -> str:
    manifest = {relative: _sha256(root / relative) for relative in _IMPLEMENTATION_PATHS}
    return hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _sha256_text(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _case_passes(case: dict[str, object]) -> bool:
    if set(case) != _CASE_FIELDS:
        raise ValueError("attention parity case has the wrong fields")
    dtype = case["dtype"]
    mask = case["mask"]
    if (dtype, mask) not in _CASE_IDENTITIES:
        raise ValueError("attention parity case has an unknown identity")
    if not isinstance(case["length"], int) or isinstance(case["length"], bool):
        raise TypeError("attention parity case length must be an integer")
    thresholds = _CASE_THRESHOLDS[str(dtype)]
    errors = {
        name: _finite_nonnegative(case[name], name=f"attention parity {name}")
        for name in thresholds
    }
    return (
        case["length"] > 0
        and case["finite"] is True
        and all(errors[name] <= threshold for name, threshold in thresholds.items())
    )


def validate_attention_backend_parity_report(value: object) -> dict[str, object]:
    """Recompute the CUDA attention-backend parity decision."""

    if not isinstance(value, dict) or set(value) != _REPORT_FIELDS:
        raise ValueError("attention parity report has the wrong fields")
    report = dict(value)
    if (
        report["schema"] != ATTENTION_BACKEND_PARITY_SCHEMA
        or report["source_commit"] != LINGBOT_NATIVE_SOURCE_COMMIT
        or not isinstance(report["torch_version"], str)
        or not report["torch_version"]
        or not isinstance(report["cuda_version"], str)
        or not report["cuda_version"]
        or not isinstance(report["device"], str)
        or not report["device"]
        or not isinstance(report["seed"], int)
        or isinstance(report["seed"], bool)
    ):
        raise ValueError("attention parity report identity is malformed")
    _sha256_text(report["patch_sha256"], name="attention parity patch")
    _sha256_text(report["implementation_sha256"], name="attention parity implementation")
    patched_sources = report["patched_source_sha256"]
    if not isinstance(patched_sources, dict) or not patched_sources:
        raise ValueError("attention parity report has no patched-source manifest")
    for relative, digest in patched_sources.items():
        path = PurePosixPath(relative) if isinstance(relative, str) else None
        if (
            path is None
            or path.is_absolute()
            or not path.parts
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise ValueError("attention parity patched-source path is invalid")
        _sha256_text(digest, name=f"attention parity source {relative}")
    cases = report["cases"]
    if not isinstance(cases, list) or len(cases) != len(_CASE_IDENTITIES):
        raise ValueError("attention parity report has incomplete cases")
    identities: set[tuple[str, str]] = set()
    all_passed = True
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise TypeError("attention parity case must be a mapping")
        case = dict(raw_case)
        identity = (str(case.get("dtype")), str(case.get("mask")))
        if identity in identities:
            raise ValueError("attention parity report repeats a case")
        identities.add(identity)
        recomputed = _case_passes(case)
        if case["passed"] is not recomputed:
            raise ValueError("attention parity case decision was not recomputed")
        all_passed &= recomputed
    if identities != _CASE_IDENTITIES:
        raise ValueError("attention parity report omits a required case")
    benchmark = report["benchmark"]
    if not isinstance(benchmark, dict) or set(benchmark) != {
        "block_mask_build_ms",
        "eager_forward_median_ms",
        "flex_cached_forward_median_ms",
        "length",
        "repeats",
        "speedup",
    }:
        raise ValueError("attention parity benchmark has the wrong fields")
    if (
        not isinstance(benchmark["length"], int)
        or isinstance(benchmark["length"], bool)
        or benchmark["length"] <= 0
        or not isinstance(benchmark["repeats"], int)
        or isinstance(benchmark["repeats"], bool)
        or benchmark["repeats"] <= 0
    ):
        raise ValueError("attention parity benchmark dimensions are invalid")
    eager_ms = _finite_nonnegative(
        benchmark["eager_forward_median_ms"],
        name="eager benchmark time",
    )
    flex_ms = _finite_nonnegative(
        benchmark["flex_cached_forward_median_ms"],
        name="flex benchmark time",
    )
    _finite_nonnegative(benchmark["block_mask_build_ms"], name="block-mask build time")
    speedup = _finite_nonnegative(benchmark["speedup"], name="attention speedup")
    if (
        eager_ms <= 0
        or flex_ms <= 0
        or not math.isclose(
            speedup,
            eager_ms / flex_ms,
            rel_tol=1.0e-9,
            abs_tol=1.0e-12,
        )
    ):
        raise ValueError("attention parity benchmark speedup was not recomputed")
    expected_status = "PASS" if all_passed else "FAIL"
    if report["status"] != expected_status:
        raise ValueError("attention parity report status was not recomputed")
    return report


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = Path(
        os.environ.get(
            "PICF_LINGBOT_NATIVE_SOURCE",
            root / CHECKOUT_RELATIVE_PATH,
        )
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--benchmark-length", type=int, default=512)
    parser.add_argument("--benchmark-repeats", type=int, default=5)
    return parser.parse_args()


def _mask(torch_module: Any, *, kind: str, batch: int, length: int, device: Any) -> Any:
    if kind == "full":
        return torch_module.ones(
            batch,
            length,
            length,
            dtype=torch_module.bool,
            device=device,
        )
    if kind == "causal":
        return (
            torch_module.ones(
                length,
                length,
                dtype=torch_module.bool,
                device=device,
            )
            .tril()
            .unsqueeze(0)
            .expand(batch, -1, -1)
            .contiguous()
        )
    if kind != "structured":
        raise ValueError(f"unknown attention parity mask: {kind}")
    query_block = torch_module.arange(length, device=device)[None, :, None] // 16
    key_block = torch_module.arange(length, device=device)[None, None, :] // 16
    self_mask = torch_module.eye(
        length,
        dtype=torch_module.bool,
        device=device,
    ).unsqueeze(0)
    return ((query_block >= key_block - 1).expand(batch, -1, -1) | self_mask).contiguous()


def _absolute_errors(left: Any, right: Any) -> tuple[float, float]:
    difference = (left.detach().float() - right.detach().float()).abs()
    return float(difference.max().item()), float(difference.mean().item())


def _run_case(
    *,
    torch_module: Any,
    eager_attention: Any,
    build_block_mask: Any,
    flex_attention: Any,
    dtype_name: str,
    mask_name: str,
    length: int,
    device: Any,
) -> dict[str, object]:
    dtype = getattr(torch_module, dtype_name)
    batch = 2
    query_heads = 8
    key_value_heads = 2
    head_dim = 32
    mask = _mask(
        torch_module,
        kind=mask_name,
        batch=batch,
        length=length,
        device=device,
    )
    query = torch_module.randn(
        batch,
        length,
        query_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    key = torch_module.randn(
        batch,
        length,
        key_value_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    value = torch_module.randn_like(key)
    eager_values = tuple(item.detach().clone().requires_grad_(True) for item in (query, key, value))
    flex_values = tuple(item.detach().clone().requires_grad_(True) for item in (query, key, value))
    block_mask = build_block_mask(mask, query_heads, length, length)
    eager_output = eager_attention(*eager_values, mask)
    flex_output = flex_attention(*flex_values, block_mask, length)
    probe = torch_module.randn_like(eager_output)
    (eager_output.float() * probe.float()).sum().backward()
    (flex_output.float() * probe.float()).sum().backward()
    output_max, output_mean = _absolute_errors(eager_output, flex_output)
    gradient_errors = tuple(
        _absolute_errors(eager_value.grad, flex_value.grad)
        for eager_value, flex_value in zip(eager_values, flex_values, strict=True)
    )
    tensors = (
        eager_output,
        flex_output,
        *(value.grad for value in eager_values),
        *(value.grad for value in flex_values),
    )
    result: dict[str, object] = {
        "dtype": dtype_name,
        "finite": all(bool(torch_module.isfinite(value).all().item()) for value in tensors),
        "gradient_max_absolute_error": max(value[0] for value in gradient_errors),
        "gradient_mean_absolute_error": max(value[1] for value in gradient_errors),
        "length": length,
        "mask": mask_name,
        "output_max_absolute_error": output_max,
        "output_mean_absolute_error": output_mean,
        "passed": False,
    }
    result["passed"] = _case_passes(result)
    return result


def _timed_cuda_call(torch_module: Any, function: Any, *, device: Any) -> float:
    start = torch_module.cuda.Event(enable_timing=True)
    end = torch_module.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    torch_module.cuda.synchronize(device)
    return float(start.elapsed_time(end))


def _timed_wall_call(torch_module: Any, function: Any, *, device: Any) -> float:
    torch_module.cuda.synchronize(device)
    started = time.perf_counter()
    function()
    torch_module.cuda.synchronize(device)
    return (time.perf_counter() - started) * 1000.0


def _benchmark(
    *,
    torch_module: Any,
    eager_attention: Any,
    build_block_mask: Any,
    flex_attention: Any,
    length: int,
    repeats: int,
    device: Any,
) -> dict[str, object]:
    batch = 1
    query_heads = 32
    key_value_heads = 8
    head_dim = 128
    mask = _mask(
        torch_module,
        kind="structured",
        batch=batch,
        length=length,
        device=device,
    )
    query = torch_module.randn(
        batch,
        length,
        query_heads,
        head_dim,
        device=device,
        dtype=torch_module.float32,
    )
    key = torch_module.randn(
        batch,
        length,
        key_value_heads,
        head_dim,
        device=device,
        dtype=torch_module.float32,
    )
    value = torch_module.randn_like(key)
    holder: list[Any] = []
    block_build_ms = _timed_wall_call(
        torch_module,
        lambda: holder.append(build_block_mask(mask, query_heads, length, length)),
        device=device,
    )
    block_mask = holder[0]

    def eager_call() -> Any:
        return eager_attention(query, key, value, mask)

    def flex_call() -> Any:
        return flex_attention(query, key, value, block_mask, length)

    for _ in range(2):
        eager_call()
        flex_call()
    torch_module.cuda.synchronize(device)
    eager_times = tuple(
        _timed_cuda_call(torch_module, eager_call, device=device) for _ in range(repeats)
    )
    flex_times = tuple(
        _timed_cuda_call(torch_module, flex_call, device=device) for _ in range(repeats)
    )
    eager_median = float(statistics.median(eager_times))
    flex_median = float(statistics.median(flex_times))
    return {
        "block_mask_build_ms": block_build_ms,
        "eager_forward_median_ms": eager_median,
        "flex_cached_forward_median_ms": flex_median,
        "length": length,
        "repeats": repeats,
        "speedup": eager_median / flex_median,
    }


def main() -> None:
    args = _parse_args()
    if (
        isinstance(args.seed, bool)
        or not isinstance(args.seed, int)
        or isinstance(args.benchmark_length, bool)
        or not isinstance(args.benchmark_length, int)
        or args.benchmark_length <= 0
        or isinstance(args.benchmark_repeats, bool)
        or not isinstance(args.benchmark_repeats, int)
        or args.benchmark_repeats <= 0
    ):
        raise ValueError("attention parity dimensions and seed must be valid integers")
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    if _git_output(args.source_checkout, "rev-parse", "HEAD") != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("attention parity source differs from the pinned LingBot commit")
    if detect_native_patch_state(args.source_checkout, args.patch) != "applied":
        raise RuntimeError("attention parity source patch is not exactly applied")
    patch_sha256 = patch_report.get("patch_sha256")
    if not isinstance(patch_sha256, str) or len(patch_sha256) != 64:
        raise RuntimeError("attention parity patch verifier returned no SHA-256")
    patched_source_sha256 = patch_report.get("patched_source_sha256")
    if not isinstance(patched_source_sha256, dict) or not patched_source_sha256:
        raise RuntimeError("attention parity patch verifier returned no source hashes")

    sys.path.insert(0, str(args.source_checkout.resolve()))
    import torch
    from lingbotvla.models.vla.lingbot_vla.flex_attention import (
        build_block_mask,
        flex_attention_with_block_mask,
    )
    from lingbotvla.models.vla.lingbot_vla.utils import our_eager_attention_forward

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("attention backend parity requires CUDA")
    torch.cuda.set_device(device)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    lengths = {"full": 73, "causal": 129, "structured": 173}
    cases = [
        _run_case(
            torch_module=torch,
            eager_attention=our_eager_attention_forward,
            build_block_mask=build_block_mask,
            flex_attention=flex_attention_with_block_mask,
            dtype_name=dtype,
            mask_name=mask,
            length=lengths[mask],
            device=device,
        )
        for dtype, mask in sorted(_CASE_IDENTITIES)
    ]
    benchmark = _benchmark(
        torch_module=torch,
        eager_attention=our_eager_attention_forward,
        build_block_mask=build_block_mask,
        flex_attention=flex_attention_with_block_mask,
        length=args.benchmark_length,
        repeats=args.benchmark_repeats,
        device=device,
    )
    torch_version = str(torch.__version__)
    cuda_version = str(getattr(getattr(torch, "version", None), "cuda", None))
    if not torch_version or not cuda_version or cuda_version == "None":
        raise RuntimeError("attention backend parity cannot identify the Torch CUDA runtime")
    report: dict[str, object] = {
        "benchmark": benchmark,
        "cases": cases,
        "cuda_version": cuda_version,
        "device": torch.cuda.get_device_name(device),
        "implementation_sha256": _implementation_sha256(root),
        "patch_sha256": patch_sha256,
        "patched_source_sha256": dict(sorted(patched_source_sha256.items())),
        "schema": ATTENTION_BACKEND_PARITY_SCHEMA,
        "seed": args.seed,
        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
        "status": "PASS" if all(bool(case["passed"]) for case in cases) else "FAIL",
        "torch_version": torch_version,
    }
    validate_attention_backend_parity_report(report)
    write_text_durable_exclusive(
        args.output.resolve(),
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    if report["status"] != "PASS":
        raise RuntimeError("attention backend parity failed")


if __name__ == "__main__":
    main()

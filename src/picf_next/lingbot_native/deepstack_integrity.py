"""Numerical integrity gates for LingBot/Qwen3-VL DeepStack.

The probe is diagnostic-only.  It measures the released visual feature
injections and a paired DeepStack-zero intervention without adding parameters
or changing the production graph.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

import torch

from picf_next.contracts import ContractError

DEEPSTACK_INTEGRITY_SCHEMA = "picf-next.lingbot-deepstack-integrity.v1"
DEEPSTACK_MIN_FEATURE_NONZERO_FRACTION = 0.01
DEEPSTACK_MIN_POSTERIOR_RELATIVE_RMS = 1.0e-4
DEEPSTACK_MIN_POSTERIOR_ABSOLUTE_RMS = 1.0e-7
DEEPSTACK_EFFECT_TO_REPEAT_RATIO = 100.0


def tensor_sha256(value: torch.Tensor) -> str:
    """Hash shape, dtype and exact tensor bytes without numerical conversion."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor digest requires a torch.Tensor")
    detached = value.detach().contiguous()
    header = f"{tuple(detached.shape)}\0{detached.dtype}\0".encode("ascii")
    payload = detached.view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(header + payload).hexdigest()


def tensor_numeric_summary(value: torch.Tensor) -> dict[str, object]:
    """Return stable FP32 statistics while retaining the source dtype and shape."""

    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor summary requires a torch.Tensor")
    if value.numel() <= 0:
        raise ValueError("tensor summary requires at least one element")
    finite = bool(torch.isfinite(value).all().item())
    numeric = value.detach().to(torch.float32)
    return {
        "shape": [int(item) for item in value.shape],
        "dtype": str(value.dtype),
        "numel": int(value.numel()),
        "all_finite": finite,
        "nonzero_fraction": float((numeric != 0).to(torch.float32).mean().item()),
        "mean": float(numeric.mean().item()),
        "std": float(numeric.std(unbiased=False).item()),
        "rms": float(numeric.square().mean().sqrt().item()),
        "l2": float(torch.linalg.vector_norm(numeric).item()),
        "max_abs": float(numeric.abs().max().item()),
        "sha256": tensor_sha256(value),
    }


def tensor_difference_summary(
    reference: torch.Tensor,
    candidate: torch.Tensor,
) -> dict[str, object]:
    """Measure a paired tensor intervention relative to one reference tensor."""

    if not isinstance(reference, torch.Tensor) or not isinstance(candidate, torch.Tensor):
        raise TypeError("tensor difference requires two torch.Tensor values")
    if reference.shape != candidate.shape:
        raise ValueError("paired tensor shapes differ")
    if reference.numel() <= 0:
        raise ValueError("tensor difference requires at least one element")
    left = reference.detach().to(torch.float32)
    right = candidate.detach().to(torch.float32)
    delta = right - left
    delta_rms = float(delta.square().mean().sqrt().item())
    reference_rms = float(left.square().mean().sqrt().item())
    relative_rms = delta_rms / max(reference_rms, torch.finfo(torch.float32).tiny)
    return {
        "shape": [int(item) for item in reference.shape],
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "bitwise_equal": bool(
            reference.dtype == candidate.dtype and torch.equal(reference, candidate)
        ),
        "all_finite": bool(
            torch.isfinite(reference).all().item()
            and torch.isfinite(candidate).all().item()
            and torch.isfinite(delta).all().item()
        ),
        "reference_rms": reference_rms,
        "delta_rms": delta_rms,
        "relative_rms": relative_rms,
        "mean_abs": float(delta.abs().mean().item()),
        "max_abs": float(delta.abs().max().item()),
        "reference_sha256": tensor_sha256(reference),
        "candidate_sha256": tensor_sha256(candidate),
    }


def _finite_real(value: object, *, name: str, minimum: float = 0.0) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < minimum
    ):
        raise ContractError(f"{name} must be finite and >= {minimum}")
    return float(value)


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ContractError(f"{name} must be a positive integer")
    return value


def _sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _injections(value: object, *, run_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list) or not value:
        raise ContractError(f"{run_name} must contain DeepStack injections")
    result: list[Mapping[str, Any]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping):
            raise ContractError(f"{run_name} injection {index} must be a mapping")
        result.append(raw)
    return tuple(result)


def _difference(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"{name} must be a tensor-difference mapping")
    for field in ("delta_rms", "relative_rms", "max_abs"):
        _finite_real(value.get(field), name=f"{name} {field}")
    if not isinstance(value.get("bitwise_equal"), bool):
        raise ContractError(f"{name} bitwise_equal must be boolean")
    if value.get("all_finite") is not True:
        raise ContractError(f"{name} contains non-finite values")
    _sha256(value.get("reference_sha256"), name=f"{name} reference digest")
    _sha256(value.get("candidate_sha256"), name=f"{name} candidate digest")
    return value


def deepstack_integrity_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    """Recompute every scientific gate from persisted numerical evidence."""

    expected_count = _positive_int(
        report.get("expected_deepstack_count"),
        name="expected DeepStack count",
    )
    runs = report.get("runs")
    if not isinstance(runs, Mapping) or set(runs) != {
        "normal",
        "normal_repeat",
        "zeroed",
    }:
        raise ContractError("DeepStack report must contain normal, repeat and zeroed runs")
    parsed_runs: dict[str, tuple[Mapping[str, Any], ...]] = {}
    expected_layers = tuple(range(expected_count))
    for run_name in ("normal", "normal_repeat", "zeroed"):
        raw_run = runs[run_name]
        if not isinstance(raw_run, Mapping) or raw_run.get("mode") != (
            "zeroed" if run_name == "zeroed" else "normal"
        ):
            raise ContractError(f"{run_name} has the wrong intervention mode")
        injections = _injections(raw_run.get("injections"), run_name=run_name)
        if len(injections) != expected_count:
            raise ContractError(f"{run_name} DeepStack injection count changed")
        layers = tuple(item.get("layer_index") for item in injections)
        if layers != expected_layers:
            raise ContractError(f"{run_name} DeepStack language-layer order changed")
        parsed_runs[run_name] = injections

    features_finite_nonzero = True
    feature_identity_stable = True
    normal_injection_exact = True
    zero_intervention_exact = True
    for layer_index in expected_layers:
        layer_records = {run_name: parsed_runs[run_name][layer_index] for run_name in parsed_runs}
        feature_hashes: list[str] = []
        for run_name, record in layer_records.items():
            feature = record.get("feature")
            if not isinstance(feature, Mapping):
                raise ContractError(f"{run_name} layer {layer_index} omitted feature statistics")
            feature_hashes.append(
                _sha256(
                    feature.get("sha256"),
                    name=f"{run_name} layer {layer_index} feature digest",
                )
            )
            feature_rms = _finite_real(
                feature.get("rms"),
                name=f"{run_name} layer {layer_index} feature rms",
            )
            feature_std = _finite_real(
                feature.get("std"),
                name=f"{run_name} layer {layer_index} feature std",
            )
            feature_nonzero = _finite_real(
                feature.get("nonzero_fraction"),
                name=f"{run_name} layer {layer_index} feature nonzero fraction",
            )
            if feature_nonzero > 1.0:
                raise ContractError("DeepStack feature nonzero fraction exceeds one")
            features_finite_nonzero &= bool(
                feature.get("all_finite") is True
                and feature_rms > 0.0
                and feature_std > 0.0
                and feature_nonzero >= DEEPSTACK_MIN_FEATURE_NONZERO_FRACTION
            )
            _positive_int(
                record.get("visual_position_count"),
                name=f"{run_name} layer {layer_index} visual position count",
            )
        feature_identity_stable &= len(set(feature_hashes)) == 1

        for run_name in ("normal", "normal_repeat"):
            record = layer_records[run_name]
            visual_rms = _finite_real(
                record.get("visual_delta_rms"),
                name=f"{run_name} layer {layer_index} visual delta rms",
            )
            expected_error = _finite_real(
                record.get("visual_expected_max_abs_error"),
                name=f"{run_name} layer {layer_index} expected visual error",
            )
            nonvisual_error = _finite_real(
                record.get("nonvisual_max_abs_delta"),
                name=f"{run_name} layer {layer_index} nonvisual error",
            )
            normal_injection_exact &= (
                visual_rms > 0.0 and expected_error == 0.0 and nonvisual_error == 0.0
            )

        zeroed = layer_records["zeroed"]
        zero_intervention_exact &= (
            _finite_real(
                zeroed.get("visual_delta_rms"),
                name=f"zeroed layer {layer_index} visual delta rms",
            )
            == 0.0
            and _finite_real(
                zeroed.get("nonvisual_max_abs_delta"),
                name=f"zeroed layer {layer_index} nonvisual error",
            )
            == 0.0
        )

    comparisons = report.get("comparisons")
    if not isinstance(comparisons, Mapping) or set(comparisons) != {
        "normal_repeat",
        "normal_zeroed",
    }:
        raise ContractError("DeepStack report comparisons are incomplete")
    normal_repeat = comparisons["normal_repeat"]
    normal_zeroed = comparisons["normal_zeroed"]
    if not isinstance(normal_repeat, Mapping) or not isinstance(normal_zeroed, Mapping):
        raise ContractError("DeepStack comparisons must be mappings")
    repeat_posterior = _difference(
        normal_repeat.get("posterior_rows"),
        name="normal-repeat posterior",
    )
    repeat_ownership = _difference(
        normal_repeat.get("relation_ownership"),
        name="normal-repeat relation ownership",
    )
    zero_posterior = _difference(
        normal_zeroed.get("posterior_rows"),
        name="normal-zeroed posterior",
    )
    zero_ownership = _difference(
        normal_zeroed.get("relation_ownership"),
        name="normal-zeroed relation ownership",
    )
    deterministic_repeat = bool(
        repeat_posterior["bitwise_equal"] and repeat_ownership["bitwise_equal"]
    )
    repeat_rms = float(repeat_posterior["delta_rms"])
    zero_rms = float(zero_posterior["delta_rms"])
    downstream_posterior_effect = bool(
        zero_rms >= DEEPSTACK_MIN_POSTERIOR_ABSOLUTE_RMS
        and float(zero_posterior["relative_rms"]) >= DEEPSTACK_MIN_POSTERIOR_RELATIVE_RMS
        and zero_rms
        >= max(
            DEEPSTACK_MIN_POSTERIOR_ABSOLUTE_RMS,
            DEEPSTACK_EFFECT_TO_REPEAT_RATIO * repeat_rms,
        )
        and (not zero_posterior["bitwise_equal"] or not zero_ownership["bitwise_equal"])
    )
    return {
        "feature_tensors_finite_nonzero": features_finite_nonzero,
        "feature_identity_stable_across_pair": feature_identity_stable,
        "normal_injection_exact_and_visual_only": normal_injection_exact,
        "zero_intervention_exact": zero_intervention_exact,
        "normal_repeat_deterministic": deterministic_repeat,
        "deepstack_reaches_picf_posterior": downstream_posterior_effect,
    }


def validate_deepstack_integrity_report(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate provenance-independent DeepStack numerical conclusions."""

    if not isinstance(value, Mapping):
        raise ContractError("DeepStack integrity report must be a mapping")
    if value.get("schema") != DEEPSTACK_INTEGRITY_SCHEMA:
        raise ContractError("DeepStack integrity report schema changed")
    expected_gates = deepstack_integrity_gates(value)
    if value.get("gates") != expected_gates:
        raise ContractError("DeepStack persisted gates differ from recomputation")
    expected_failures = sorted(name for name, passed in expected_gates.items() if not passed)
    if value.get("failures") != expected_failures:
        raise ContractError("DeepStack persisted failures differ from recomputation")
    expected_status = "PASS" if not expected_failures else "FAIL"
    if value.get("status") != expected_status:
        raise ContractError("DeepStack persisted status differs from numerical gates")
    return dict(value)

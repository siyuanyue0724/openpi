#!/usr/bin/env python3
"""Bind a compact M4 gate-update audit to a full-weight training checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_GATE_PATTERN = re.compile(
    r"^joint_bridge\.sequence_bridge\.policy\.action_layer_adapter\."
    r"(?P<family>dense|object)_branches\.(?P<index>[0-9]+)\.gate$"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--expected-factorization", choices=("A", "B", "C", "D"))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"metrics are not readable ASCII JSONL: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics JSON on line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_number} must be a JSON object")
        rows.append(row)
    if not rows:
        raise ValueError("metrics JSONL is empty")
    return rows


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("ascii")
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _gate_summary(values_by_index: Mapping[int, float], *, family: str) -> dict[str, Any]:
    if not values_by_index:
        raise ValueError(f"checkpoint contains no {family} residual gates")
    indices = sorted(values_by_index)
    if indices != list(range(len(indices))):
        raise ValueError(f"{family} residual gate indices are not contiguous from zero")
    values = [float(values_by_index[index]) for index in indices]
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{family} residual gates contain NaN or infinity")
    canonical = json.dumps(values, separators=(",", ":")).encode("ascii")
    absolute = [abs(value) for value in values]
    return {
        "abs_max": max(absolute),
        "abs_mean": sum(absolute) / len(absolute),
        "count": len(values),
        "nonzero_count": sum(value != 0.0 for value in values),
        "value_max": max(values),
        "value_min": min(values),
        "values_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _load_gate_summaries(model_path: Path) -> dict[str, dict[str, Any]]:
    from safetensors import safe_open

    values: dict[str, dict[int, float]] = {"dense": {}, "object": {}}
    with safe_open(model_path, framework="pt", device="cpu") as handle:
        # safetensors.safe_open exposes keys() but is intentionally not iterable.
        for name in handle.keys():  # noqa: SIM118
            match = _GATE_PATTERN.fullmatch(name)
            if match is None:
                continue
            tensor = handle.get_tensor(name)
            if tensor.numel() != 1 or not tensor.is_floating_point():
                raise ValueError(f"residual gate must be one floating-point scalar: {name}")
            family = match.group("family")
            index = int(match.group("index"))
            if index in values[family]:
                raise ValueError(f"duplicate {family} residual gate index: {index}")
            values[family][index] = float(tensor.item())
    summaries = {
        family: _gate_summary(family_values, family=family)
        for family, family_values in values.items()
    }
    if summaries["dense"]["count"] != summaries["object"]["count"]:
        raise ValueError("dense and object residual gate counts differ")
    return summaries


def _bound_model(checkpoint: Path, control: Mapping[str, Any]) -> dict[str, Any]:
    if control.get("schema") != "picf-next.checkpoint-control-manifest.v2":
        raise ValueError("unsupported checkpoint control schema")
    state_files = control.get("state_files")
    if not isinstance(state_files, dict):
        raise ValueError("checkpoint control has no state_files mapping")
    record = state_files.get("model.safetensors")
    if not isinstance(record, dict):
        raise ValueError("checkpoint control does not bind model.safetensors")
    expected_size = record.get("size_bytes")
    expected_sha256 = record.get("sha256")
    if (
        not isinstance(expected_size, int)
        or isinstance(expected_size, bool)
        or expected_size <= 0
        or not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
    ):
        raise ValueError("checkpoint model binding is malformed")
    model_path = checkpoint / "model.safetensors"
    if not model_path.is_file() or model_path.stat().st_size != expected_size:
        raise ValueError("checkpoint model file is absent or has the wrong size")
    actual_sha256 = _sha256(model_path)
    if actual_sha256 != expected_sha256:
        raise ValueError("checkpoint model hash differs from checkpoint control")
    return {
        "path": str(model_path.resolve()),
        "sha256": actual_sha256,
        "size_bytes": expected_size,
    }


def build_report(
    *,
    run_dir: Path,
    checkpoint: Path,
    expected_factorization: str | None = None,
) -> dict[str, Any]:
    run_dir = run_dir.expanduser().resolve()
    checkpoint = checkpoint.expanduser().resolve()
    try:
        checkpoint.relative_to(run_dir)
    except ValueError as error:
        raise ValueError("checkpoint must be inside run_dir") from error

    control_path = checkpoint / "picf_control.json"
    metrics_path = run_dir / "metrics.jsonl"
    static_path = run_dir / "static_preflight.json"
    plan_path = run_dir / "sample_plan.json"
    control = _read_json(control_path, "checkpoint control")
    static = _read_json(static_path, "static preflight")
    plan = _read_json(plan_path, "sample plan")
    metrics_rows = _read_metrics(metrics_path)
    model = _bound_model(checkpoint, control)
    gates = _load_gate_summaries(checkpoint / "model.safetensors")

    contract = control.get("contract")
    progress = control.get("progress")
    if not isinstance(contract, dict) or not isinstance(progress, dict):
        raise ValueError("checkpoint control lacks contract or progress")
    arm_config = contract.get("arm_config")
    if not isinstance(arm_config, dict):
        raise ValueError("checkpoint contract lacks arm_config")
    factorization = arm_config.get("causal_factorization")
    if not isinstance(factorization, dict):
        raise ValueError("checkpoint contract lacks causal_factorization")
    factorization_id = factorization.get("id")
    include_posterior = factorization.get("include_posterior_action_context")
    if factorization_id not in {"A", "B", "C", "D"} or not isinstance(include_posterior, bool):
        raise ValueError("checkpoint causal factorization is malformed")
    if expected_factorization is not None and factorization_id != expected_factorization:
        raise ValueError(
            f"checkpoint factorization {factorization_id} differs from expected "
            f"{expected_factorization}"
        )
    static_factorization = static.get("causal_factorization")
    if not isinstance(static_factorization, dict) or static_factorization.get("id") != (
        factorization_id
    ):
        raise ValueError("static preflight and checkpoint factorization differ")
    if control.get("plan_sha256") != plan.get("plan_sha256"):
        raise ValueError("checkpoint control and sample plan hashes differ")

    final_metrics = metrics_rows[-1]
    successful_steps = final_metrics.get("successful_optimizer_steps")
    if not isinstance(successful_steps, int) or isinstance(successful_steps, bool):
        raise ValueError("final metrics successful_optimizer_steps is malformed")
    metric_values = final_metrics.get("metrics")
    if not isinstance(metric_values, dict):
        raise ValueError("final metrics has no metrics mapping")

    dense_updated = gates["dense"]["nonzero_count"] > 0
    object_updated = gates["object"]["nonzero_count"] > 0
    checks = {
        "dense_route_updated_after_optimizer_step": successful_steps == 0 or dense_updated,
        "object_route_matches_declared_factorization": (
            successful_steps == 0
            or (include_posterior and object_updated)
            or (not include_posterior and not object_updated)
        ),
        "optimizer_step_not_skipped": final_metrics.get("optimizer_step_skipped") is False,
        "one_metrics_row_per_successful_step": len(metrics_rows) == successful_steps,
    }
    status = "PASS" if all(checks.values()) else "FAIL"
    return {
        "artifacts": {
            "checkpoint_control_sha256": _sha256(control_path),
            "metrics_sha256": _sha256(metrics_path),
            "model": model,
            "sample_plan_sha256": _sha256(plan_path),
            "static_preflight_sha256": _sha256(static_path),
        },
        "causal_factorization": factorization,
        "checks": checks,
        "checkpoint": str(checkpoint),
        "comparison_id": plan.get("metadata", {}).get("comparison_id"),
        "final_metrics": {
            "action_flow_loss": metric_values.get("action_flow_loss"),
            "learning_rate": metric_values.get("system_learning_rate_max"),
            "peak_allocated_bytes": metric_values.get("system_cuda_peak_allocated_bytes_rank_max"),
            "step_wall_seconds": metric_values.get("system_train_step_wall_seconds_rank_max"),
            "synchronized_grad_norm": metric_values.get("system_synchronized_grad_norm"),
        },
        "gates_after_optimizer_step": gates,
        "metrics_rows": len(metrics_rows),
        "run_dir": str(run_dir),
        "schema": "picf-next.molmoact2-m4-checkpoint-audit.v1",
        "status": status,
        "successful_optimizer_steps": successful_steps,
    }


def main() -> int:
    args = _parse_args()
    report = build_report(
        run_dir=args.run_dir,
        checkpoint=args.checkpoint,
        expected_factorization=args.expected_factorization,
    )
    output = args.output or args.run_dir / "smoke_checkpoint_audit.json"
    _atomic_json(output.expanduser().resolve(), report)
    print(json.dumps({"output": str(output), "status": report["status"]}, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

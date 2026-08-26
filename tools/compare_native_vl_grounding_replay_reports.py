#!/usr/bin/env python3
"""Compare a two-step no-retention replay with the frozen ADR-124 prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError

BASELINE_SCHEMA = "picf-next.lingbot-native-vl-grounding-adaptation.v5"
REPLAY_SCHEMA = "picf-next.lingbot-native-vl-grounding-adaptation.v6"
OUTPUT_SCHEMA = "picf-next.native-vl-no-retention-replay-comparison.v1"
REPLAY_STEP_COUNT = 2
GRADIENT_RELATIVE_TOLERANCE = 1e-6
GRADIENT_ABSOLUTE_TOLERANCE = 1e-8

_COMMON_TOP_LEVEL = (
    "cuda_allocator",
    "dataset_manifest_sha256",
    "fsdp2_placement",
    "initial_qwen",
    "native_vl_patch_sha256",
    "observation_mode",
    "optimizer",
    "optimizer_state_parameter_count",
    "optimizer_tied_parameter_name",
    "processor_lattices",
    "processor_snapshot_size",
    "source_commit",
    "teacher_prune",
    "trainable_scope",
    "training_plan",
    "world_size",
)
_COMMON_HYPERPARAMETERS = (
    "adam_beta1",
    "adam_beta2",
    "adam_eps",
    "learning_rate",
    "max_grad_norm",
    "schedule",
    "schedule_total_steps",
    "warmup_steps",
    "weight_decay",
)
_COMMON_STEP_FIELDS = (
    "curriculum_group_index",
    "curriculum_optimizer_step",
    "learning_rate",
    "observation_mode",
    "optimizer_step",
)
_COMMON_RANK_FIELDS = (
    "camera_name",
    "global_index",
    "image_grid_thw",
    "instruction",
    "loss_weight",
    "rank",
    "supervised_token_count",
    "target_identity_key",
    "task_key",
    "visual_lattice",
)
_EXACT_GRADIENT_FIELDS = (
    "all_finite",
    "frozen_gradient_elements",
    "trainable_gradient_elements",
)
_FLOAT_GRADIENT_FIELDS = (
    "clip_coefficient",
    "global_norm_before_clip",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-report", required=True, type=Path)
    parser.add_argument("--baseline-report-sha256", required=True)
    parser.add_argument("--replay-report", required=True, type=Path)
    parser.add_argument("--replay-report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"native VL replay comparison {name} must be one SHA-256")
    return value


def _require_git_revision(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"native VL replay comparison {name} must be one Git commit")
    return value


def _load_report(path: Path, *, expected_sha256: str, schema: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, name="report digest")
    if path.is_symlink() or not path.is_file():
        raise ContractError("native VL replay comparison input must be a regular file")
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ContractError("native VL replay comparison input cannot be read") from error
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError("native VL replay comparison report digest changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("native VL replay comparison input is invalid JSON") from error
    if not isinstance(value, dict) or value.get("schema") != schema:
        raise ContractError("native VL replay comparison input schema changed")
    if value.get("status") != "PASS":
        raise ContractError("native VL replay comparison requires successful executions")
    return cast(dict[str, Any], value)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"native VL replay comparison {name} must be a mapping")
    return value


def _steps(report: Mapping[str, Any], *, minimum: int, exact: bool) -> list[Mapping[str, Any]]:
    value = report.get("step_reports")
    if not isinstance(value, list) or len(value) < minimum or (exact and len(value) != minimum):
        raise ContractError("native VL replay comparison step coverage changed")
    if any(not isinstance(step, Mapping) for step in value):
        raise ContractError("native VL replay comparison step report is malformed")
    return cast(list[Mapping[str, Any]], value)


def _require_common_bindings(
    baseline: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> None:
    if any(baseline.get(field) != replay.get(field) for field in _COMMON_TOP_LEVEL):
        raise ContractError("native VL replay comparison top-level binding changed")
    baseline_hyper = _mapping(baseline.get("hyperparameters"), name="baseline hyperparameters")
    replay_hyper = _mapping(replay.get("hyperparameters"), name="replay hyperparameters")
    if any(
        baseline_hyper.get(field) != replay_hyper.get(field) for field in _COMMON_HYPERPARAMETERS
    ):
        raise ContractError("native VL replay comparison hyperparameters changed")
    if baseline_hyper.get("max_steps") != 64 or replay_hyper.get("max_steps") != REPLAY_STEP_COUNT:
        raise ContractError("native VL replay comparison training horizons changed")
    baseline_revision = _require_git_revision(
        baseline.get("picf_code_revision"),
        name="baseline PICF revision",
    )
    replay_revision = _require_git_revision(
        replay.get("picf_code_revision"),
        name="replay PICF revision",
    )
    if baseline_revision == replay_revision:
        raise ContractError("native VL replay comparison did not exercise a revised runner")
    retention = replay.get("public_vl_retention")
    if retention != {"enabled": False}:
        raise ContractError("native VL replay comparison replay enabled public retention")


def compare_native_vl_grounding_replay_reports(
    baseline: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, object]:
    """Apply the preregistered two-step no-retention equivalence rule."""

    if baseline.get("schema") != BASELINE_SCHEMA or replay.get("schema") != REPLAY_SCHEMA:
        raise ContractError("native VL replay comparison input schema changed")
    if baseline.get("status") != "PASS" or replay.get("status") != "PASS":
        raise ContractError("native VL replay comparison requires successful executions")
    _require_common_bindings(baseline, replay)
    baseline_steps = _steps(baseline, minimum=REPLAY_STEP_COUNT, exact=False)
    replay_steps = _steps(replay, minimum=REPLAY_STEP_COUNT, exact=True)

    exact_losses = True
    gradient_close = True
    maximum_gradient_absolute_delta = 0.0
    maximum_gradient_relative_delta = 0.0
    compared_rank_losses = 0
    compared_gradient_metrics = 0
    for baseline_step, replay_step in zip(
        baseline_steps[:REPLAY_STEP_COUNT],
        replay_steps,
        strict=True,
    ):
        if any(baseline_step.get(field) != replay_step.get(field) for field in _COMMON_STEP_FIELDS):
            raise ContractError("native VL replay comparison step identity changed")
        if replay_step.get("public_vl_retention") is not None:
            raise ContractError("native VL replay comparison step contains retention evidence")
        baseline_micro = baseline_step.get("microbatches")
        replay_micro = replay_step.get("microbatches")
        if not isinstance(baseline_micro, list) or not isinstance(replay_micro, list):
            raise ContractError("native VL replay comparison microbatches are malformed")
        if len(baseline_micro) != len(replay_micro):
            raise ContractError("native VL replay comparison microbatch count changed")
        for baseline_batch, replay_batch in zip(baseline_micro, replay_micro, strict=True):
            baseline_batch = _mapping(baseline_batch, name="baseline microbatch")
            replay_batch = _mapping(replay_batch, name="replay microbatch")
            if baseline_batch.get("visual_lattice") != replay_batch.get("visual_lattice"):
                raise ContractError("native VL replay comparison visual lattice changed")
            baseline_ranks = baseline_batch.get("ranks")
            replay_ranks = replay_batch.get("ranks")
            if not isinstance(baseline_ranks, list) or not isinstance(replay_ranks, list):
                raise ContractError("native VL replay comparison rank reports are malformed")
            if len(baseline_ranks) != len(replay_ranks):
                raise ContractError("native VL replay comparison rank count changed")
            for baseline_rank, replay_rank in zip(baseline_ranks, replay_ranks, strict=True):
                baseline_rank = _mapping(baseline_rank, name="baseline rank report")
                replay_rank = _mapping(replay_rank, name="replay rank report")
                if any(
                    baseline_rank.get(field) != replay_rank.get(field)
                    for field in _COMMON_RANK_FIELDS
                ):
                    raise ContractError("native VL replay comparison rank binding changed")
                baseline_loss = baseline_rank.get("loss")
                replay_loss = replay_rank.get("loss")
                if (
                    isinstance(baseline_loss, bool)
                    or not isinstance(baseline_loss, int | float)
                    or isinstance(replay_loss, bool)
                    or not isinstance(replay_loss, int | float)
                    or not math.isfinite(float(baseline_loss))
                    or not math.isfinite(float(replay_loss))
                ):
                    raise ContractError("native VL replay comparison loss is malformed")
                exact_losses = exact_losses and float(baseline_loss) == float(replay_loss)
                compared_rank_losses += 1

        baseline_gradient = _mapping(
            baseline_step.get("gradient_metrics"),
            name="baseline gradient metrics",
        )
        replay_gradient = _mapping(
            replay_step.get("gradient_metrics"),
            name="replay gradient metrics",
        )
        if set(baseline_gradient) != set(replay_gradient):
            raise ContractError("native VL replay comparison gradient metric fields changed")
        if any(
            baseline_gradient.get(field) != replay_gradient.get(field)
            for field in _EXACT_GRADIENT_FIELDS
        ):
            gradient_close = False
        for field in _FLOAT_GRADIENT_FIELDS:
            baseline_value = baseline_gradient.get(field)
            replay_value = replay_gradient.get(field)
            if (
                isinstance(baseline_value, bool)
                or not isinstance(baseline_value, int | float)
                or isinstance(replay_value, bool)
                or not isinstance(replay_value, int | float)
            ):
                raise ContractError("native VL replay comparison gradient metric is malformed")
            left = float(baseline_value)
            right = float(replay_value)
            if not math.isfinite(left) or not math.isfinite(right):
                raise ContractError("native VL replay comparison gradient metric is non-finite")
            absolute_delta = abs(right - left)
            relative_delta = absolute_delta / max(
                abs(left),
                abs(right),
                GRADIENT_ABSOLUTE_TOLERANCE,
            )
            maximum_gradient_absolute_delta = max(
                maximum_gradient_absolute_delta,
                absolute_delta,
            )
            maximum_gradient_relative_delta = max(
                maximum_gradient_relative_delta,
                relative_delta,
            )
            gradient_close = gradient_close and math.isclose(
                left,
                right,
                rel_tol=GRADIENT_RELATIVE_TOLERANCE,
                abs_tol=GRADIENT_ABSOLUTE_TOLERANCE,
            )
            compared_gradient_metrics += 1

    status = "PASS" if exact_losses and gradient_close else "FAIL"
    return {
        "baseline_picf_code_revision": baseline["picf_code_revision"],
        "compared_gradient_metrics": compared_gradient_metrics,
        "compared_rank_losses": compared_rank_losses,
        "exact_bf16_losses": exact_losses,
        "gradient_absolute_tolerance": GRADIENT_ABSOLUTE_TOLERANCE,
        "gradient_metrics_within_tolerance": gradient_close,
        "gradient_relative_tolerance": GRADIENT_RELATIVE_TOLERANCE,
        "maximum_gradient_absolute_delta": maximum_gradient_absolute_delta,
        "maximum_gradient_relative_delta": maximum_gradient_relative_delta,
        "replay_picf_code_revision": replay["picf_code_revision"],
        "replayed_step_count": REPLAY_STEP_COUNT,
        "schema": OUTPUT_SCHEMA,
        "status": status,
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    baseline = _load_report(
        args.baseline_report,
        expected_sha256=args.baseline_report_sha256,
        schema=BASELINE_SCHEMA,
    )
    replay = _load_report(
        args.replay_report,
        expected_sha256=args.replay_report_sha256,
        schema=REPLAY_SCHEMA,
    )
    report = compare_native_vl_grounding_replay_reports(baseline, replay)
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, sort_keys=True), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()

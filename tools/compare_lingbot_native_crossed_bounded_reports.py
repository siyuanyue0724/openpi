#!/usr/bin/env python3
"""Compare the frozen ADR-128 candidate/control training reports."""

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

INPUT_SCHEMA = "picf-next.lingbot-native-vl-crossed-adaptation.v1"
OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-crossed-comparison.v1"
EXPECTED_TRAINABLE_GRADIENT_ELEMENTS = 4_049_739_776

_COMMON_TOP_LEVEL = (
    "counterfactual_gradient_audit",
    "crossed_cpu_materialization",
    "cuda_allocator",
    "dataset_manifest_sha256",
    "fsdp2_placement",
    "hyperparameters",
    "initial_qwen",
    "native_vl_patch_sha256",
    "observation_mode",
    "optimizer",
    "optimizer_state_parameter_count",
    "optimizer_tied_parameter_name",
    "physical_sidecar_manifest_sha256",
    "picf_code_revision",
    "processor_lattices",
    "processor_snapshot_size",
    "public_vl_retention",
    "runtime_python_trees",
    "schema",
    "seed",
    "source_commit",
    "status",
    "teacher_prune",
    "trainable_scope",
    "world_size",
)
_STEP_IDENTITY_FIELDS = (
    "crossed_cell",
    "crossed_plan_optimizer_step",
    "learning_rate",
    "observation_mode",
    "optimizer_step",
)
_P_RANK_BINDINGS = (
    "assistant_text_sha256",
    "camera_name",
    "crossed_bbox_qwen_xyxy",
    "crossed_group_index",
    "crossed_instruction_sha256",
    "crossed_variant_index",
    "factor_name",
    "global_index",
    "image_grid_thw",
    "instruction",
    "loss_weight",
    "rank",
    "record_type",
    "source_rgb_sha256",
    "supervised_token_count",
    "target_identity_key",
    "task_key",
    "user_text_sha256",
    "visual_lattice",
)
_X_SEMANTIC_BINDINGS = (
    "camera_name",
    "factor_name",
    "loss_weight",
    "rank",
    "record_type",
    "target_identity_key",
    "task_key",
    "visual_lattice",
)
_PUBLIC_BINDINGS = (
    "assistant_text_sha256",
    "family",
    "grid_budget",
    "image_grid_thw",
    "image_height",
    "image_rgb_sha256",
    "image_width",
    "loss_weight",
    "rank",
    "record_id",
    "record_sha256",
    "source_row_index",
    "source_subindex",
    "supervised_token_count",
    "target_answer_sha256",
    "user_text",
    "user_text_sha256",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-report", required=True, type=Path)
    parser.add_argument("--candidate-report-sha256", required=True)
    parser.add_argument("--control-report", required=True, type=Path)
    parser.add_argument("--control-report-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def _require_sha256(value: object, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"ADR-128 comparison {name} must be one SHA-256")
    return value


def _load_report(path: Path, *, expected_sha256: str) -> dict[str, Any]:
    expected = _require_sha256(expected_sha256, name="report digest")
    if path.is_symlink() or not path.is_file():
        raise ContractError("ADR-128 comparison input must be a regular file")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected:
        raise ContractError("ADR-128 comparison report digest changed")
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ContractError("ADR-128 comparison input is invalid JSON") from error
    if not isinstance(value, dict):
        raise ContractError("ADR-128 comparison input must be a mapping")
    return cast(dict[str, Any], value)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ContractError(f"ADR-128 comparison {name} must be a mapping")
    return value


def _list_of_mappings(value: object, *, name: str) -> list[Mapping[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ContractError(f"ADR-128 comparison {name} must be a list of mappings")
    return cast(list[Mapping[str, Any]], value)


def _without_arm(value: object, *, name: str) -> dict[str, Any]:
    result = dict(_mapping(value, name=name))
    arm = result.pop("arm", result.pop("crossed_arm", None))
    if arm not in {"candidate", "control"}:
        raise ContractError(f"ADR-128 comparison {name} has no valid arm")
    return result


def _finite_loss(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ContractError(f"ADR-128 comparison {name} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ContractError(f"ADR-128 comparison {name} is not finite")
    return result


def _mean(values: list[float], *, name: str) -> float:
    if not values:
        raise ContractError(f"ADR-128 comparison {name} is empty")
    return sum(values) / len(values)


def _paired_curve_summary(
    rows: list[dict[str, object]],
    *,
    candidate_field: str,
    control_field: str,
) -> dict[str, object]:
    result = {}
    for cell in ("all", "P", "X"):
        selected = rows if cell == "all" else [row for row in rows if row["cell"] == cell]
        candidate = [float(row[candidate_field]) for row in selected]
        control = [float(row[control_field]) for row in selected]
        last_count = min(4, len(selected))
        candidate_mean = _mean(candidate, name=f"{cell} candidate curve")
        control_mean = _mean(control, name=f"{cell} control curve")
        candidate_last = _mean(candidate[-last_count:], name=f"{cell} candidate tail")
        control_last = _mean(control[-last_count:], name=f"{cell} control tail")
        result[cell] = {
            "candidate_last4_mean": candidate_last,
            "candidate_mean": candidate_mean,
            "candidate_minus_control_last4_mean": candidate_last - control_last,
            "candidate_minus_control_mean": candidate_mean - control_mean,
            "control_last4_mean": control_last,
            "control_mean": control_mean,
            "step_count": len(selected),
        }
    return result


def _rank_rows(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    batches = _list_of_mappings(step.get("microbatches"), name="microbatches")
    rows: list[Mapping[str, Any]] = []
    for batch in batches:
        factors = _list_of_mappings(batch.get("factors"), name="factors")
        for factor in factors:
            rows.extend(_list_of_mappings(factor.get("ranks"), name="factor ranks"))
    return rows


def _public_rows(step: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    public = _mapping(step.get("public_vl_retention"), name="public retention step")
    return _list_of_mappings(public.get("ranks"), name="public retention ranks")


def _require_bindings_equal(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    fields: tuple[str, ...],
    *,
    name: str,
) -> None:
    if any(candidate.get(field) != control.get(field) for field in fields):
        raise ContractError(f"ADR-128 comparison {name} changed")


def _require_gradient_contract(step: Mapping[str, Any]) -> None:
    gradient = _mapping(step.get("gradient_metrics"), name="gradient metrics")
    if (
        gradient.get("all_finite") is not True
        or gradient.get("frozen_gradient_elements") != 0
        or gradient.get("trainable_gradient_elements") != EXPECTED_TRAINABLE_GRADIENT_ELEMENTS
    ):
        raise ContractError("ADR-128 comparison gradient coverage failed")
    _finite_loss(gradient.get("clip_coefficient"), name="clip coefficient")
    _finite_loss(gradient.get("global_norm_before_clip"), name="gradient norm")


def compare_lingbot_native_crossed_bounded_reports(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
) -> dict[str, object]:
    """Validate matched execution without mistaking elapsed time for model state."""

    if candidate.get("schema") != INPUT_SCHEMA or control.get("schema") != INPUT_SCHEMA:
        raise ContractError("ADR-128 comparison input schema changed")
    if candidate.get("status") != "PASS" or control.get("status") != "PASS":
        raise ContractError("ADR-128 comparison requires two passing executions")
    if any(candidate.get(field) != control.get(field) for field in _COMMON_TOP_LEVEL):
        raise ContractError("ADR-128 comparison common execution binding changed")

    candidate_plan = _without_arm(candidate.get("training_plan"), name="candidate plan")
    control_plan = _without_arm(control.get("training_plan"), name="control plan")
    if candidate_plan != control_plan:
        raise ContractError("ADR-128 comparison frozen plans differ")
    if (
        candidate_plan.get("bounded_training_authorized") is not True
        or candidate_plan.get("long_training_authorized") is not False
    ):
        raise ContractError("ADR-128 comparison authorization boundary changed")

    candidate_factor = _without_arm(
        candidate.get("calvin_factor_contract"), name="candidate factor contract"
    )
    control_factor = _without_arm(
        control.get("calvin_factor_contract"), name="control factor contract"
    )
    if candidate_factor != control_factor:
        raise ContractError("ADR-128 comparison factor contracts differ")

    candidate_steps = _list_of_mappings(candidate.get("step_reports"), name="candidate steps")
    control_steps = _list_of_mappings(control.get("step_reports"), name="control steps")
    max_steps = _mapping(candidate.get("hyperparameters"), name="hyperparameters").get("max_steps")
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps not in {2, 64}
        or len(candidate_steps) != max_steps
        or len(control_steps) != max_steps
    ):
        raise ContractError("ADR-128 comparison step coverage changed")

    p_binding_rows = 0
    x_semantic_rows = 0
    public_binding_rows = 0
    loss_curves = []
    step_zero_exact_losses = True
    step_zero_exact_gradients = True
    for index, (candidate_step, control_step) in enumerate(
        zip(candidate_steps, control_steps, strict=True)
    ):
        _require_bindings_equal(
            candidate_step,
            control_step,
            _STEP_IDENTITY_FIELDS,
            name="step identity",
        )
        expected_cell = "P" if index % 2 == 0 else "X"
        if (
            candidate_step.get("crossed_cell") != expected_cell
            or candidate_step.get("optimizer_step") != index
            or candidate_step.get("crossed_plan_optimizer_step") != index
        ):
            raise ContractError("ADR-128 comparison frozen interleave changed")
        _require_gradient_contract(candidate_step)
        _require_gradient_contract(control_step)

        candidate_rows = _rank_rows(candidate_step)
        control_rows = _rank_rows(control_step)
        if len(candidate_rows) != 2 or len(control_rows) != 2:
            raise ContractError("ADR-128 comparison requires two target ranks per step")
        rank_fields = _P_RANK_BINDINGS if expected_cell == "P" else _X_SEMANTIC_BINDINGS
        for candidate_row, control_row in zip(candidate_rows, control_rows, strict=True):
            _require_bindings_equal(
                candidate_row,
                control_row,
                rank_fields,
                name=f"{expected_cell} rank binding",
            )
            _finite_loss(candidate_row.get("loss"), name="candidate target loss")
            _finite_loss(control_row.get("loss"), name="control target loss")
        candidate_target_mean = _mean(
            [_finite_loss(row.get("loss"), name="candidate target loss") for row in candidate_rows],
            name="candidate target step",
        )
        control_target_mean = _mean(
            [_finite_loss(row.get("loss"), name="control target loss") for row in control_rows],
            name="control target step",
        )
        if expected_cell == "P":
            p_binding_rows += len(candidate_rows)
        else:
            x_semantic_rows += len(candidate_rows)

        candidate_public = _public_rows(candidate_step)
        control_public = _public_rows(control_step)
        if len(candidate_public) != 2 or len(control_public) != 2:
            raise ContractError("ADR-128 comparison requires two public ranks per step")
        for candidate_row, control_row in zip(candidate_public, control_public, strict=True):
            _require_bindings_equal(
                candidate_row,
                control_row,
                _PUBLIC_BINDINGS,
                name="public record binding",
            )
            _finite_loss(candidate_row.get("loss"), name="candidate public loss")
            _finite_loss(control_row.get("loss"), name="control public loss")
        candidate_public_mean = _mean(
            [
                _finite_loss(row.get("loss"), name="candidate public loss")
                for row in candidate_public
            ],
            name="candidate public step",
        )
        control_public_mean = _mean(
            [_finite_loss(row.get("loss"), name="control public loss") for row in control_public],
            name="control public step",
        )
        loss_curves.append(
            {
                "candidate_public_mean": candidate_public_mean,
                "candidate_target_mean": candidate_target_mean,
                "cell": expected_cell,
                "control_public_mean": control_public_mean,
                "control_target_mean": control_target_mean,
                "optimizer_step": index,
            }
        )
        public_binding_rows += len(candidate_public)

        if index == 0:
            step_zero_exact_losses = all(
                candidate_row.get("loss") == control_row.get("loss")
                for candidate_row, control_row in (
                    *zip(candidate_rows, control_rows, strict=True),
                    *zip(candidate_public, control_public, strict=True),
                )
            )
            step_zero_exact_gradients = candidate_step.get("gradient_metrics") == control_step.get(
                "gradient_metrics"
            )

    if not step_zero_exact_losses or not step_zero_exact_gradients:
        raise ContractError("ADR-128 comparison deterministic step-zero parity failed")
    candidate_hashes = _mapping(
        candidate.get("candidate_model_file_sha256"), name="candidate model hashes"
    )
    control_hashes = _mapping(
        control.get("candidate_model_file_sha256"), name="control model hashes"
    )
    if not candidate_hashes or candidate_hashes == control_hashes:
        raise ContractError("ADR-128 comparison trained exports did not diverge")

    return {
        "candidate_model_file_count": len(candidate_hashes),
        "compared_step_count": max_steps,
        "control_model_file_count": len(control_hashes),
        "full_gradient_coverage_every_step": True,
        "loss_curves": loss_curves,
        "loss_summary": {
            "public": _paired_curve_summary(
                loss_curves,
                candidate_field="candidate_public_mean",
                control_field="control_public_mean",
            ),
            "target": _paired_curve_summary(
                loss_curves,
                candidate_field="candidate_target_mean",
                control_field="control_target_mean",
            ),
        },
        "p_exact_binding_rows": p_binding_rows,
        "public_exact_binding_rows": public_binding_rows,
        "schema": OUTPUT_SCHEMA,
        "status": "PASS",
        "step_zero_exact_gradients": step_zero_exact_gradients,
        "step_zero_exact_losses": step_zero_exact_losses,
        "x_matched_semantic_rows": x_semantic_rows,
    }


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    candidate = _load_report(
        args.candidate_report,
        expected_sha256=args.candidate_report_sha256,
    )
    control = _load_report(
        args.control_report,
        expected_sha256=args.control_report_sha256,
    )
    report = compare_lingbot_native_crossed_bounded_reports(candidate, control)
    write_text_durable_exclusive(
        args.output,
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

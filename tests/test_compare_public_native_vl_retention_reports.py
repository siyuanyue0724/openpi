from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from picf_next.contracts import ContractError
from picf_next.lingbot_native.lattice_feasibility import (
    native_lattice_shortest_edge,
    native_processor_area_budget_contract,
    native_processor_expected_grid,
    validate_native_processor_grid_budget,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_fixed_x_pair_geometry_metrics,
    native_vl_fixed_x_partition_summary,
)
from tools import compare_public_native_vl_retention_reports as comparator
from tools.compare_public_native_vl_retention_reports import (
    ADR125_CALVIN_DATASET_MANIFEST_SHA256,
    ADR125_CALVIN_EVALUATION_PLAN_ARTIFACT_SHA256,
    ADR125_CALVIN_EVALUATION_PLAN_FILE_SHA256,
    ADR125_CALVIN_PHYSICAL_SIDECAR_MANIFEST_SHA256,
    ADR125_CURRICULUM_ARTIFACT_SHA256,
    ADR125_CURRICULUM_FILE_SHA256,
    ADR125_PUBLIC_ARTIFACT_SHA256,
    ADR125_PUBLIC_MANIFEST_FILE_SHA256,
    ADR125_QWEN_LONGEST_EDGE_AREA,
    ADR125_TRAINING_HYPERPARAMETERS,
    FIXED_X_SCHEMA,
    TRAINING_SCHEMA,
    compare_public_native_vl_retention_reports,
)

_PRODUCTION_PUBLIC_HELDOUT_BINDING_SHA256 = comparator.ADR125_PUBLIC_HELDOUT_BINDING_SHA256
_PRODUCTION_PUBLIC_TRAINING_BINDING_SHA256 = comparator.ADR125_PUBLIC_TRAINING_BINDING_SHA256


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _write_report_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _processor_lattice(lattice: int) -> dict[str, int]:
    return {
        "lattice": lattice,
        "longest_edge_area": ADR125_QWEN_LONGEST_EDGE_AREA,
        "merge_size": 2,
        "patch_size": 16,
        "pixels_per_edge": lattice * 32,
        "shortest_edge_area": native_lattice_shortest_edge(lattice),
    }


_FAMILY_TASKS = {
    "block": ("lift_blue_block_table", "movable/block_blue"),
    "drawer": ("open_drawer", "part/table/drawer_link"),
    "slider": ("move_slider_left", "part/table/slide_link"),
    "led": ("turn_on_led", "part/table/button_link"),
    "lightbulb": ("turn_on_lightbulb", "part/table/switch_link"),
}


def _calvin_family_pairs() -> list[tuple[str, str]]:
    pairs = [
        ("block", family)
        for family, count in (("drawer", 27), ("slider", 27), ("led", 25), ("lightbulb", 28))
        for _ in range(count)
    ]
    first = ["drawer"] * 14 + ["slider"] * 14 + ["led"]
    second = ["led"] * 14 + ["lightbulb"] * 15
    pairs.extend(zip(first, second, strict=True))
    assert len(pairs) == 136
    return pairs


def _calvin_results(*, miss_one_led: bool) -> tuple[list[dict[str, Any]], dict[str, object]]:
    groups: list[dict[str, Any]] = []
    led_missed = False
    for group_index, families in enumerate(_calvin_family_pairs()):
        partition = "validation" if group_index < 68 else "heldout"
        ordinal = group_index if group_index < 68 else group_index - 68
        targets = ((100, 100, 220, 220), (700, 700, 820, 820))
        predictions: list[tuple[int, int, int, int] | None] = [targets[0], targets[1]]
        if miss_one_led and not led_missed:
            for variant_index, family in enumerate(families):
                if family == "led":
                    predictions[variant_index] = None
                    led_missed = True
                    break
        metrics = native_vl_fixed_x_pair_geometry_metrics(
            (predictions[0], predictions[1]),
            targets,
        )
        metric_variants = metrics["variants"]
        assert isinstance(metric_variants, list)
        variants = []
        for variant_index, family in enumerate(families):
            task_key, target_identity = _FAMILY_TASKS[family]
            instruction = f"{task_key} instruction {partition} {ordinal} {variant_index}"
            prediction = predictions[variant_index]
            variants.append(
                {
                    "camera_name": "static",
                    "generated_bbox_qwen_xyxy": (None if prediction is None else list(prediction)),
                    "generated_bbox_schema_valid": prediction is not None,
                    "generated_text": (
                        "generated"
                        if prediction is None
                        else json.dumps({"bbox_2d": list(prediction)})
                    ),
                    "instruction": instruction,
                    "instruction_sha256": _text_digest(instruction),
                    "target_bbox_qwen_xyxy": list(targets[variant_index]),
                    "target_identity_key": target_identity,
                    "task_key": task_key,
                    **metric_variants[variant_index],
                }
            )
        groups.append(
            {
                "ordinal": ordinal,
                "pair_metrics": {key: value for key, value in metrics.items() if key != "variants"},
                "partition": partition,
                "source_global_index": group_index,
                "source_rgb_sha256": f"{group_index + 1000:064x}",
                "source_state_sha256": f"{group_index + 2000:064x}",
                "variants": variants,
            }
        )
    summaries: dict[str, object] = {
        partition: native_vl_fixed_x_partition_summary(
            [group for group in groups if group["partition"] == partition]
        )
        for partition in ("validation", "heldout")
    }
    return groups, summaries


def _report(
    *,
    referring_nll: float,
    vqa_nll: float,
    qwen_digest: str,
) -> dict[str, Any]:
    image_height = 480
    image_width = 640
    image_grid_thw = native_processor_expected_grid(
        image_height=image_height,
        image_width=image_width,
        lattice=8,
    )
    grid_budget = validate_native_processor_grid_budget(image_grid_thw, lattice=8)
    rows = []
    for family in ("referring", "vqa"):
        for index in range(32):
            target_bbox = [100, 100, 200, 200]
            predicted_bbox = target_bbox if index < 20 else [300, 300, 400, 400]
            target_answer = (
                json.dumps({"bbox_2d": target_bbox}) if family == "referring" else "answer"
            )
            generated_text = (
                json.dumps({"bbox_2d": predicted_bbox})
                if family == "referring"
                else ("answer" if index < 10 else "different")
            )
            user_text = "question"
            row = {
                "family": family,
                "generated_text": generated_text,
                "grid_budget": deepcopy(grid_budget),
                "image_height": image_height,
                "image_rgb_sha256": f"{index + (0 if family == 'referring' else 100):064x}",
                "image_grid_thw": deepcopy(image_grid_thw),
                "image_width": image_width,
                "mean_token_nll": referring_nll if family == "referring" else vqa_nll,
                "record_id": f"{family}-heldout-{index:04d}",
                "record_sha256": f"{index + (200 if family == 'referring' else 300):064x}",
                "source_row_index": index,
                "source_subindex": 0,
                "supervised_token_count": 4,
                "target_answer": target_answer,
                "target_answer_sha256": _text_digest(target_answer),
                "user_text": user_text,
                "user_text_sha256": _text_digest(user_text),
            }
            if family == "referring":
                row.update(
                    {
                        "generated_bbox_qwen_xyxy": predicted_bbox,
                        "generated_bbox_schema_valid": True,
                        "target_bbox_qwen_xyxy": target_bbox,
                        "target_center_hit": index < 20,
                        "target_iou": 1.0 if index < 20 else 0.0,
                    }
                )
            else:
                row["normalized_exact_match"] = index < 10
            rows.append(row)
    calvin_results, calvin_summaries = _calvin_results(miss_one_led=qwen_digest == "7")
    return {
        "checkpoint_dir": "/checkpoint",
        "checkpoint_model_file_sha256": {"model.safetensors": "c" * 64},
        "dataset_manifest_sha256": ADR125_CALVIN_DATASET_MANIFEST_SHA256,
        "eligible_item_count": 136,
        "evaluation_plan_artifact_sha256": ADR125_CALVIN_EVALUATION_PLAN_ARTIFACT_SHA256,
        "evaluation_plan_file_sha256": ADR125_CALVIN_EVALUATION_PLAN_FILE_SHA256,
        "excluded_items": [],
        "item_limit_per_partition": 0,
        "max_new_tokens": 64,
        "native_vl_patch_sha256": comparator.ADR125_NATIVE_VL_PATCH_SHA256,
        "partition": "all",
        "picf_code_revision": comparator.ADR125_PICF_CODE_REVISION,
        "physical_sidecar_manifest_sha256": ADR125_CALVIN_PHYSICAL_SIDECAR_MANIFEST_SHA256,
        "processor_lattice": _processor_lattice(8),
        "public_vl_retention": {
            "artifact_sha256": ADR125_PUBLIC_ARTIFACT_SHA256,
            "enabled": True,
            "family_partition_counts": {
                "referring/heldout": 32,
                "vqa/heldout": 32,
            },
            "heldout_limit_per_family": 32,
            "manifest_file_sha256": ADR125_PUBLIC_MANIFEST_FILE_SHA256,
            "processor": native_processor_area_budget_contract(8),
            "results": rows,
            "summaries": {
                "referring": {
                    "generated_bbox_count": 32,
                    "generated_bbox_schema_valid_count": 32,
                    "mean_record_nll": referring_nll,
                    "mean_target_iou": 0.625,
                    "record_count": 32,
                    "supervised_token_count": 128,
                    "target_center_hit_count": 20,
                    "token_weighted_mean_nll": referring_nll,
                },
                "vqa": {
                    "mean_record_nll": vqa_nll,
                    "normalized_exact_match_count": 10,
                    "record_count": 32,
                    "supervised_token_count": 128,
                    "token_weighted_mean_nll": vqa_nll,
                },
            },
        },
        "results": calvin_results,
        "schema": FIXED_X_SCHEMA,
        "seed": comparator.ADR125_FIXED_X_SEED,
        "selected_item_count": 136,
        "source_commit": comparator.ADR125_LINGBOT_SOURCE_COMMIT,
        "qwen_restore": {
            "model_file_sha256": {"qwen.safetensors": qwen_digest * 64},
            "model_revision": (
                comparator.ADR125_INITIAL_QWEN_REVISION
                if qwen_digest == "7"
                else comparator.ADR125_PICF_CODE_REVISION
            ),
        },
        "summaries": calvin_summaries,
    }


def _training_report(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    control_restore = control["qwen_restore"]
    candidate_restore = candidate["qwen_restore"]
    public = control["public_vl_retention"]
    image_height = 480
    image_width = 640
    image_grid_thw = native_processor_expected_grid(
        image_height=image_height,
        image_width=image_width,
        lattice=8,
    )
    grid_budget = validate_native_processor_grid_budget(image_grid_thw, lattice=8)
    step_reports = [
        {
            "curriculum_group_index": step,
            "curriculum_optimizer_step": step,
            "gradient_metrics": {
                "all_finite": True,
                "clip_coefficient": 1.0 / (2.0 + 1e-6),
                "frozen_gradient_elements": 0,
                "global_norm_before_clip": 2.0,
                "trainable_gradient_elements": comparator.ADR125_TRAINABLE_NUMEL,
            },
            "learning_rate": 1e-6,
            "microbatches": [
                {
                    "ranks": [
                        {
                            "camera_name": "static",
                            "global_index": step,
                            "image_grid_thw": [[1, 16, 16]],
                            "instruction": f"instruction {step} rank {rank}",
                            "loss": 1.0,
                            "loss_weight": 1.0,
                            "rank": rank,
                            "supervised_token_count": 4,
                            "target_identity_key": "movable/block_blue",
                            "task_key": "lift_blue_block_table",
                            "visual_lattice": 8,
                        }
                        for rank in range(2)
                    ],
                    "visual_lattice": 8,
                }
            ],
            "observation_mode": "official_native_once",
            "optimizer_step": step,
            "public_vl_retention": {
                "ranks": [
                    {
                        "family": family,
                        "grid_budget": deepcopy(grid_budget),
                        "image_height": image_height,
                        "image_rgb_sha256": f"{step + rank * 1000:064x}",
                        "image_grid_thw": deepcopy(image_grid_thw),
                        "image_width": image_width,
                        "loss": 1.0,
                        "loss_weight": 0.1,
                        "rank": rank,
                        "record_id": f"{family}-train-{step:04d}",
                        "record_sha256": f"{step + rank * 1000 + 2000:064x}",
                        "source_row_index": step,
                        "source_subindex": 0,
                        "supervised_token_count": 4,
                        "target_answer_sha256": _text_digest("target"),
                        "user_text": "question",
                        "user_text_sha256": _text_digest("question"),
                    }
                    for rank, family in enumerate(("referring", "vqa"))
                ]
            },
        }
        for step in range(64)
    ]
    return {
        "candidate_model_file_sha256": candidate_restore["model_file_sha256"],
        "cuda_allocator": "expandable-segments",
        "dataset_manifest_sha256": control["dataset_manifest_sha256"],
        "fsdp2_placement": "gpu-sharded",
        "hyperparameters": dict(ADR125_TRAINING_HYPERPARAMETERS),
        "initial_qwen": {
            "model_file_sha256": control_restore["model_file_sha256"],
            "revision": control_restore["model_revision"],
        },
        "native_vl_patch_sha256": control["native_vl_patch_sha256"],
        "observation_mode": "official_native_once",
        "optimizer": "torch.optim.AdamW",
        "optimizer_state_parameter_count": comparator.ADR125_TRAINABLE_PARAMETER_COUNT,
        "picf_code_revision": control["picf_code_revision"],
        "processor_lattices": {"8": _processor_lattice(8)},
        "public_vl_retention": {
            "artifact_sha256": public["artifact_sha256"],
            "enabled": True,
            "global_loss_factors": {"referring": 0.05, "vqa": 0.05},
            "manifest_file_sha256": public["manifest_file_sha256"],
            "processor": native_processor_area_budget_contract(8),
            "rank_loss_weight": 0.1,
            "rank_streams": {"0": "referring", "1": "vqa"},
        },
        "schema": TRAINING_SCHEMA,
        "source_commit": control["source_commit"],
        "status": "PASS",
        "step_reports": step_reports,
        "trainable_scope": {
            "parameter_count": comparator.ADR125_TRAINABLE_PARAMETER_COUNT,
            "schema_sha256": comparator.ADR125_TRAINABLE_SCHEMA_SHA256,
            "trainable_numel": comparator.ADR125_TRAINABLE_NUMEL,
        },
        "training_plan": {
            "artifact_sha256": ADR125_CURRICULUM_ARTIFACT_SHA256,
            "file_sha256": ADR125_CURRICULUM_FILE_SHA256,
            "observation_mode": "official_native_once",
            "source_visual_lattices": [8, 14],
            "type": "official_native_once_curriculum",
            "visual_lattices": [8],
        },
        "world_size": 2,
    }


def _compare_with_training(
    control: dict[str, Any],
    candidate: dict[str, Any],
    training: dict[str, Any],
) -> dict[str, Any]:
    return compare_public_native_vl_retention_reports(
        control,
        candidate,
        training,
        control_report_sha256=_digest(control),
        candidate_report_sha256=_digest(candidate),
        candidate_training_report_sha256=_digest(training),
    )


def _compare(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    return _compare_with_training(control, candidate, _training_report(control, candidate))


def _set_calvin_prediction(
    report: dict[str, Any],
    *,
    partition: str,
    ordinal: int,
    variant_index: int,
    prediction: list[int] | None,
) -> None:
    group = next(
        row
        for row in report["results"]
        if row["partition"] == partition and row["ordinal"] == ordinal
    )
    group["variants"][variant_index]["generated_bbox_qwen_xyxy"] = prediction
    group["variants"][variant_index]["generated_bbox_schema_valid"] = prediction is not None
    group["variants"][variant_index]["generated_text"] = (
        "generated" if prediction is None else json.dumps({"bbox_2d": prediction})
    )
    variants = group["variants"]

    def bbox(value: list[int] | None) -> tuple[int, int, int, int] | None:
        return None if value is None else (value[0], value[1], value[2], value[3])

    predictions = (
        bbox(variants[0]["generated_bbox_qwen_xyxy"]),
        bbox(variants[1]["generated_bbox_qwen_xyxy"]),
    )
    target_0 = bbox(variants[0]["target_bbox_qwen_xyxy"])
    target_1 = bbox(variants[1]["target_bbox_qwen_xyxy"])
    assert target_0 is not None and target_1 is not None
    targets = (target_0, target_1)
    metrics = native_vl_fixed_x_pair_geometry_metrics(predictions, targets)
    group["pair_metrics"] = {key: value for key, value in metrics.items() if key != "variants"}
    metric_variants = metrics["variants"]
    assert isinstance(metric_variants, list)
    for variant, recomputed in zip(group["variants"], metric_variants, strict=True):
        variant.update(recomputed)
    report["summaries"] = {
        name: native_vl_fixed_x_partition_summary(
            [row for row in report["results"] if row["partition"] == name]
        )
        for name in ("validation", "heldout")
    }


def _recompute_public_summaries(report: dict[str, Any]) -> None:
    section = report["public_vl_retention"]
    section["summaries"] = comparator._recomputed_summaries(section)


def _set_referring_prediction(
    report: dict[str, Any],
    *,
    index: int,
    prediction: list[int] | None,
) -> None:
    row = report["public_vl_retention"]["results"][index]
    assert row["family"] == "referring"
    target_value = row["target_bbox_qwen_xyxy"]
    target = (target_value[0], target_value[1], target_value[2], target_value[3])
    predicted = (
        None if prediction is None else (prediction[0], prediction[1], prediction[2], prediction[3])
    )
    row["generated_text"] = "no bbox" if prediction is None else json.dumps({"bbox_2d": prediction})
    row["generated_bbox_qwen_xyxy"] = prediction
    row["generated_bbox_schema_valid"] = prediction is not None
    row["target_center_hit"] = (
        False if predicted is None else comparator.qwen_target_center_in_bbox(predicted, target)
    )
    row["target_iou"] = (
        0.0 if predicted is None else comparator.qwen_grounding_bbox_iou(predicted, target)
    )
    _recompute_public_summaries(report)


def _set_vqa_prediction(report: dict[str, Any], *, index: int, generated_text: str) -> None:
    row = report["public_vl_retention"]["results"][32 + index]
    assert row["family"] == "vqa"
    row["generated_text"] = generated_text
    row["normalized_exact_match"] = comparator.normalize_native_vl_answer(
        generated_text
    ) == comparator.normalize_native_vl_answer(row["target_answer"])
    _recompute_public_summaries(report)


def _bind_synthetic_frozen_banks_for_tests(monkeypatch: pytest.MonkeyPatch) -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    heldout = [
        [row[field] for field in comparator._PUBLIC_SCHEDULE_BINDINGS]
        for row in control["public_vl_retention"]["results"]
    ]
    training = _training_report(control, candidate)
    schedule = [
        [row[field] for field in comparator._PUBLIC_SCHEDULE_BINDINGS]
        for step in training["step_reports"]
        for row in step["public_vl_retention"]["ranks"]
    ]
    calvin_bindings = [comparator._calvin_group_binding(group) for group in control["results"]]
    calvin_schedule = [
        {
            "optimizer_step": step["optimizer_step"],
            "curriculum_group_index": step["curriculum_group_index"],
            "curriculum_optimizer_step": step["curriculum_optimizer_step"],
            "observation_mode": step["observation_mode"],
            "learning_rate": step["learning_rate"],
            "microbatches": [
                {
                    "visual_lattice": microbatch["visual_lattice"],
                    "ranks": [
                        {field: rank[field] for field in comparator._CALVIN_TRAINING_RANK_BINDINGS}
                        for rank in microbatch["ranks"]
                    ],
                }
                for microbatch in step["microbatches"]
            ],
        }
        for step in training["step_reports"]
    ]
    monkeypatch.setattr(
        comparator,
        "ADR125_PUBLIC_HELDOUT_BINDING_SHA256",
        _digest(heldout),
    )
    monkeypatch.setattr(
        comparator,
        "ADR125_PUBLIC_TRAINING_BINDING_SHA256",
        _digest(schedule),
    )
    monkeypatch.setattr(comparator, "ADR125_CALVIN_RECORD_BINDING_SHA256", _digest(calvin_bindings))
    monkeypatch.setattr(
        comparator,
        "ADR125_CALVIN_TRAINING_BINDING_SHA256",
        _digest(calvin_schedule),
    )
    monkeypatch.setattr(
        comparator,
        "ADR125_RELEASED_CHECKPOINT_ROOT_FILE_SHA256",
        {"model.safetensors": "c" * 64},
    )


@pytest.fixture(autouse=True)
def _bind_synthetic_frozen_banks(monkeypatch: pytest.MonkeyPatch) -> None:
    _bind_synthetic_frozen_banks_for_tests(monkeypatch)


def test_public_retention_comparison_passes_only_both_family_nll_improvement() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["checkpoint_dir"] = "/candidate-checkpoint"
    comparison = _compare(control, candidate)
    assert comparison["status"] == "PASS_PENDING_CALVIN_VISUAL_REVIEW"
    assert comparison["control_checkpoint_dir"] == "/checkpoint"
    assert comparison["candidate_checkpoint_dir"] == "/candidate-checkpoint"
    assert comparison["input_reports"] == {
        "candidate": {"schema": FIXED_X_SCHEMA, "sha256": _digest(candidate)},
        "candidate_training": {
            "schema": TRAINING_SCHEMA,
            "sha256": _digest(_training_report(control, candidate)),
        },
        "control": {"schema": FIXED_X_SCHEMA, "sha256": _digest(control)},
    }
    families = comparison["families"]
    assert isinstance(families, dict)
    assert families["referring"]["token_weighted_mean_nll_strictly_improves"] is True


def test_public_retention_comparison_fails_one_family_regression() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=1.1, qwen_digest="1")
    comparison = _compare(control, candidate)
    assert comparison["status"] == "FAIL"


def test_public_retention_comparison_rejects_changed_evaluation_binding() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = deepcopy(control)
    candidate["qwen_restore"]["model_file_sha256"] = {"qwen.safetensors": "1" * 64}
    candidate["processor_lattice"] = {"lattice": 14}
    with pytest.raises(ContractError, match="binding changed"):
        _compare(control, candidate)

    candidate = deepcopy(control)
    candidate["qwen_restore"]["model_file_sha256"] = {"qwen.safetensors": "1" * 64}
    candidate["public_vl_retention"]["results"][0]["record_id"] = "different"
    with pytest.raises(ContractError, match="held-out schedule changed"):
        _compare(control, candidate)


def test_public_retention_comparison_binds_public_processor_and_grids() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["processor"]["lattice"] = 14
    with pytest.raises(ContractError, match="public processor contract changed"):
        _compare(control, candidate)

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["results"][0]["image_grid_thw"] = [[1, 18, 18]]
    with pytest.raises(ContractError, match="row grid is invalid"):
        _compare(control, candidate)

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["results"][0]["grid_budget"]["merged_visual_tokens"] = 63
    with pytest.raises(ContractError, match="row grid budget changed"):
        _compare(control, candidate)


@pytest.mark.parametrize("invalid_limit", [0, 31, 33, None, "32", True])
def test_public_retention_comparison_requires_explicit_heldout_limit_32(
    invalid_limit: object,
) -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    control["public_vl_retention"]["heldout_limit_per_family"] = invalid_limit
    candidate["public_vl_retention"]["heldout_limit_per_family"] = invalid_limit

    with pytest.raises(ContractError, match="held-out limit must equal the frozen 32 records"):
        _compare(control, candidate)


def test_public_retention_comparison_accepts_explicit_heldout_limit_32() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")

    comparison = _compare(control, candidate)

    assert control["public_vl_retention"]["heldout_limit_per_family"] == 32
    assert candidate["public_vl_retention"]["heldout_limit_per_family"] == 32
    assert comparison["status"] == "PASS_PENDING_CALVIN_VISUAL_REVIEW"


def test_adr125_real_public_schedule_digests_include_supervised_token_count() -> None:
    assert comparator._PUBLIC_SCHEDULE_BINDINGS == (
        "family",
        "record_id",
        "record_sha256",
        "image_rgb_sha256",
        "image_height",
        "image_width",
        "source_row_index",
        "source_subindex",
        "supervised_token_count",
        "target_answer_sha256",
        "user_text_sha256",
    )
    assert (
        _PRODUCTION_PUBLIC_HELDOUT_BINDING_SHA256
        == "798cc30452ae35fb37167320c0e739884ef1d8fee32dc4ee4811cc52301fcbaa"
    )
    assert (
        _PRODUCTION_PUBLIC_TRAINING_BINDING_SHA256
        == "fc7c86dd9b9cf886589433b383525efa15d75ad850f4515e489ab266811e170d"
    )


def test_public_retention_comparison_binds_training_processor_and_every_step_grid() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    training = _training_report(control, candidate)
    training["public_vl_retention"]["processor"]["lattice"] = 999
    with pytest.raises(ContractError, match="training public processor changed"):
        _compare_with_training(control, candidate, training)

    training = _training_report(control, candidate)
    training["step_reports"][63]["public_vl_retention"]["ranks"][1]["image_grid_thw"] = [
        [1, 18, 18]
    ]
    with pytest.raises(ContractError, match="row grid is invalid"):
        _compare_with_training(control, candidate, training)


def test_public_retention_comparison_rejects_equal_model_content() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="7")
    candidate["checkpoint_dir"] = "/different-path-same-content"
    with pytest.raises(ContractError, match="models are equal"):
        _compare(control, candidate)


def test_public_retention_comparison_uses_effective_qwen_restore_content() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    shared_restore = {
        "model_file_sha256": {"qwen.safetensors": "7" * 64},
        "model_revision": comparator.ADR125_INITIAL_QWEN_REVISION,
    }
    control["qwen_restore"] = deepcopy(shared_restore)
    candidate["qwen_restore"] = deepcopy(shared_restore)
    with pytest.raises(ContractError, match="models are equal"):
        _compare(control, candidate)


def test_public_retention_comparison_rejects_invalid_metrics() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["summaries"]["referring"]["mean_record_nll"] = -0.1
    with pytest.raises(ContractError, match="summary differs from its result rows"):
        _compare(control, candidate)

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["summaries"]["vqa"]["normalized_exact_match_count"] = 33
    with pytest.raises(ContractError, match="summary differs from its result rows"):
        _compare(control, candidate)


def test_public_retention_comparison_recomputes_generation_metrics_from_text() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["results"][0]["generated_text"] = "no bbox"
    with pytest.raises(ContractError, match="referring text and bbox differ"):
        _compare(control, candidate)

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["results"][42]["generated_text"] = "answer"
    with pytest.raises(ContractError, match="VQA metric changed"):
        _compare(control, candidate)


@pytest.mark.parametrize(
    ("family", "index", "generated_text", "metric"),
    [
        ("referring", 0, json.dumps({"bbox_2d": [300, 300, 400, 400]}), "target_center_hit"),
        ("vqa", 0, "different", "normalized_exact_match"),
    ],
)
def test_public_retention_comparison_fails_on_free_generation_regression(
    family: str,
    index: int,
    generated_text: str,
    metric: str,
) -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    offset = 0 if family == "referring" else 32
    row = candidate["public_vl_retention"]["results"][offset + index]
    row["generated_text"] = generated_text
    if family == "referring":
        row["generated_bbox_qwen_xyxy"] = [300, 300, 400, 400]
        row["target_center_hit"] = False
        row["target_iou"] = 0.0
        summary = candidate["public_vl_retention"]["summaries"][family]
        summary["target_center_hit_count"] -= 1
        summary["mean_target_iou"] -= 1.0 / 32.0
    else:
        row["normalized_exact_match"] = False
        candidate["public_vl_retention"]["summaries"][family]["normalized_exact_match_count"] -= 1

    comparison = _compare(control, candidate)

    assert comparison["status"] == "FAIL"
    family_report = comparison["families"][family]
    assert family_report["generation_nonregression"] is False
    assert family_report["generation_checks"][f"{metric}_count_nonregression"] is False


def test_public_retention_comparison_rejects_referring_record_swap_cancellation() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    _set_referring_prediction(candidate, index=0, prediction=[300, 300, 400, 400])
    _set_referring_prediction(candidate, index=20, prediction=[100, 100, 200, 200])

    comparison = _compare(control, candidate)

    assert comparison["status"] == "FAIL"
    referring = comparison["families"]["referring"]
    assert referring["generation_metrics"]["target_center_hit_count"]["delta"] == 0
    assert referring["generation_checks"]["target_center_hit_count_nonregression"] is True
    assert referring["generation_checks"]["per_record_nonregression"] is False
    review = comparison["public_generation_visual_review_required"]
    assert {item["record_id"] for item in review} == {
        "referring-heldout-0000",
        "referring-heldout-0020",
    }
    regressed = next(item for item in review if item["record_id"].endswith("0000"))
    assert "generated_bbox_changed" in regressed["reasons"]
    assert "target_center_hit_flipped" in regressed["reasons"]


def test_public_retention_comparison_rejects_referring_presence_and_schema_regression() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    _set_referring_prediction(candidate, index=0, prediction=None)

    comparison = _compare(control, candidate)

    assert comparison["status"] == "FAIL"
    referring = comparison["families"]["referring"]
    assert referring["generation_checks"]["per_record_nonregression"] is False
    item = comparison["public_generation_visual_review_required"][0]
    assert set(item["reasons"]) >= {
        "generated_bbox_presence_flipped",
        "generated_bbox_schema_valid_flipped",
        "target_center_hit_flipped",
    }


def test_public_retention_comparison_reviews_every_referring_generation_change() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    _set_referring_prediction(candidate, index=20, prediction=[400, 400, 500, 500])

    comparison = _compare(control, candidate)

    assert comparison["status"] == "PASS_PENDING_CALVIN_VISUAL_REVIEW"
    assert comparison["public_generation_visual_review_required"] == [
        {
            "family": "referring",
            "reasons": ["generated_text_changed", "generated_bbox_changed"],
            "record_id": "referring-heldout-0020",
            "source_row_index": 20,
            "source_subindex": 0,
        }
    ]


def test_public_retention_comparison_rejects_vqa_record_swap_cancellation() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    _set_vqa_prediction(candidate, index=0, generated_text="different")
    _set_vqa_prediction(candidate, index=10, generated_text="answer")

    comparison = _compare(control, candidate)

    assert comparison["status"] == "FAIL"
    vqa = comparison["families"]["vqa"]
    assert vqa["generation_metrics"]["normalized_exact_match_count"]["delta"] == 0
    assert vqa["generation_checks"]["normalized_exact_match_count_nonregression"] is True
    assert vqa["generation_checks"]["per_record_nonregression"] is False


def test_public_retention_comparison_rejects_changed_base_checkpoint() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["checkpoint_model_file_sha256"] = {"model.safetensors": "2" * 64}
    with pytest.raises(ContractError, match=r"checkpoint(?: identity)? changed"):
        _compare(control, candidate)


def test_public_retention_comparison_rejects_shared_nonreleased_base_checkpoint() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    control["checkpoint_model_file_sha256"] = {"model.safetensors": "2" * 64}
    candidate["checkpoint_model_file_sha256"] = {"model.safetensors": "2" * 64}

    with pytest.raises(ContractError, match="released checkpoint identity changed"):
        _compare(control, candidate)


def test_public_retention_comparison_recomputes_summaries_from_rows() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["results"][0]["mean_token_nll"] = 9.0
    with pytest.raises(ContractError, match="summary differs from its result rows"):
        _compare(control, candidate)


def test_public_retention_comparison_binds_the_exact_training_output() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    training = _training_report(control, candidate)
    training["candidate_model_file_sha256"] = {"qwen.safetensors": "2" * 64}
    with pytest.raises(ContractError, match="candidate is not the training output"):
        _compare_with_training(control, candidate, training)

    training = _training_report(control, candidate)
    training["hyperparameters"]["learning_rate"] = 2e-6
    with pytest.raises(ContractError, match="training hyperparameters changed"):
        _compare_with_training(control, candidate, training)


def test_public_retention_comparison_rejects_manifest_and_training_schedule_mutations() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["public_vl_retention"]["manifest_file_sha256"] = "0" * 64
    with pytest.raises(ContractError, match="public manifest changed"):
        _compare(control, candidate)

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    training = _training_report(control, candidate)
    training["step_reports"][0]["public_vl_retention"]["ranks"][0].pop("record_id")
    with pytest.raises(ContractError, match="training schedule changed"):
        _compare_with_training(control, candidate, training)

    training = _training_report(control, candidate)
    training["step_reports"][0]["microbatches"] = []
    with pytest.raises(ContractError, match="CALVIN microbatch coverage changed"):
        _compare_with_training(control, candidate, training)

    training = _training_report(control, candidate)
    training["step_reports"][0]["gradient_metrics"]["trainable_gradient_elements"] = 1
    with pytest.raises(ContractError, match="CALVIN gradient coverage changed"):
        _compare_with_training(control, candidate, training)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 1),
        ("max_new_tokens", 1),
        ("picf_code_revision", "0" * 40),
        ("source_commit", "0" * 40),
        ("native_vl_patch_sha256", "0" * 64),
    ],
)
def test_public_retention_comparison_binds_frozen_protocol(field: str, value: object) -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    control[field] = value
    candidate[field] = value
    with pytest.raises(ContractError, match="frozen protocol changed"):
        _compare(control, candidate)


def test_public_retention_comparison_routes_broad_boxes_to_visual_review() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    _set_calvin_prediction(
        candidate,
        partition="validation",
        ordinal=0,
        variant_index=0,
        prediction=[0, 0, 1000, 1000],
    )
    comparison = _compare(control, candidate)
    review = comparison["calvin"]["visual_review_required"]
    item = next(
        row
        for row in review
        if row.get("partition") == "validation"
        and row.get("ordinal") == 0
        and row.get("variant_index") == 0
    )
    assert "prediction_covers_both_target_centers" in item["reasons"]
    assert "own_only_center_hit_regressed" in item["reasons"]


def test_public_retention_comparison_keeps_fixed_drawer_review_with_same_ordinal_change() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    group = next(
        row
        for row in candidate["results"]
        if row["partition"] == "heldout" and row["ordinal"] == 41
    )
    changed_index = next(
        index
        for index, variant in enumerate(group["variants"])
        if variant["target_identity_key"] != "part/table/drawer_link"
    )
    _set_calvin_prediction(
        candidate,
        partition="heldout",
        ordinal=41,
        variant_index=changed_index,
        prediction=[0, 0, 1000, 1000],
    )

    comparison = _compare(control, candidate)
    review = [
        item
        for item in comparison["calvin"]["visual_review_required"]
        if item.get("partition") == "heldout" and item.get("ordinal") == 41
    ]

    assert len(review) == 2
    assert any(item.get("variant_index") == changed_index for item in review)
    assert any(
        item
        == {
            "partition": "heldout",
            "ordinal": 41,
            "reason": "drawer prediction requires original-resolution broad-box review",
        }
        for item in review
    )


def test_public_retention_comparison_accepts_consistent_near_zero_gradient_norm() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    training = _training_report(control, candidate)
    training["step_reports"][0]["gradient_metrics"]["global_norm_before_clip"] = 1e-30
    training["step_reports"][0]["gradient_metrics"]["clip_coefficient"] = 1.0

    comparison = _compare_with_training(control, candidate, training)

    assert comparison["status"] == "PASS_PENDING_CALVIN_VISUAL_REVIEW"


@pytest.mark.parametrize(
    ("global_norm", "clip_coefficient"),
    [(2.0, 0.5), (1e-30, 0.999)],
)
def test_public_retention_comparison_rejects_inconsistent_clip_coefficient(
    global_norm: float,
    clip_coefficient: float,
) -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    training = _training_report(control, candidate)
    training["step_reports"][0]["gradient_metrics"]["global_norm_before_clip"] = global_norm
    training["step_reports"][0]["gradient_metrics"]["clip_coefficient"] = clip_coefficient

    with pytest.raises(ContractError, match="clip coefficient contradicts gradient norm"):
        _compare_with_training(control, candidate, training)


def test_public_retention_comparison_rejects_missing_or_self_inconsistent_calvin() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["results"] = []
    with pytest.raises(ContractError, match="CALVIN pair count changed"):
        _compare(control, candidate)

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    candidate["results"][0]["variants"][0]["generated_bbox_qwen_xyxy"] = None
    with pytest.raises(
        ContractError,
        match="CALVIN generated text and bbox differ",
    ):
        _compare(control, candidate)


@pytest.mark.parametrize("invalid", [None, {"lattice": 999}, {"lattice": 8}])
def test_public_retention_comparison_rejects_invalid_shared_calvin_lattice(
    invalid: object,
) -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    control["processor_lattice"] = invalid
    candidate["processor_lattice"] = invalid
    with pytest.raises(
        ContractError, match="CALVIN (processor lattice|processor contract|lattice)"
    ):
        _compare(control, candidate)


def test_public_retention_comparison_enforces_calvin_family_and_special_case_gates() -> None:
    control = _report(referring_nll=2.0, vqa_nll=1.0, qwen_digest="7")
    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    _set_calvin_prediction(
        candidate,
        partition="validation",
        ordinal=0,
        variant_index=0,
        prediction=None,
    )
    comparison = _compare(control, candidate)
    assert comparison["status"] == "FAIL"
    assert comparison["calvin"]["checks"]["block_exact"] is False

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    validation_16 = next(
        row
        for row in candidate["results"]
        if row["partition"] == "validation" and row["ordinal"] == 16
    )
    _set_calvin_prediction(
        candidate,
        partition="validation",
        ordinal=16,
        variant_index=1,
        prediction=validation_16["variants"][0]["generated_bbox_qwen_xyxy"],
    )
    comparison = _compare(control, candidate)
    assert comparison["calvin"]["checks"]["validation_item_16_prompt_switch"] is False
    assert comparison["status"] == "FAIL"

    candidate = _report(referring_nll=1.5, vqa_nll=0.9, qwen_digest="1")
    heldout_41 = next(
        row
        for row in candidate["results"]
        if row["partition"] == "heldout" and row["ordinal"] == 41
    )
    drawer_index = next(
        index
        for index, variant in enumerate(heldout_41["variants"])
        if variant["target_identity_key"] == "part/table/drawer_link"
    )
    other_index = 1 - drawer_index
    _set_calvin_prediction(
        candidate,
        partition="heldout",
        ordinal=41,
        variant_index=drawer_index,
        prediction=heldout_41["variants"][other_index]["target_bbox_qwen_xyxy"],
    )
    comparison = _compare(control, candidate)
    assert comparison["calvin"]["checks"]["heldout_item_41_drawer_hit"] is False
    assert comparison["status"] == "FAIL"


def test_report_loader_hashes_the_same_descriptor_bytes_it_parses(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = (json.dumps({"schema": FIXED_X_SCHEMA}, sort_keys=True) + "\n").encode()
    digest = _write_report_bytes(path, payload)

    report, actual = comparator._load_report(
        path,
        expected_sha256=digest,
        schema=FIXED_X_SCHEMA,
        require_pass=False,
    )

    assert report == {"schema": FIXED_X_SCHEMA}
    assert actual == digest
    with pytest.raises(ContractError, match="report digest changed"):
        comparator._load_report(
            path,
            expected_sha256="0" * 64,
            schema=FIXED_X_SCHEMA,
            require_pass=False,
        )


def test_report_loader_rejects_leaf_and_parent_symlinks(tmp_path: Path) -> None:
    real_dir = tmp_path / "real"
    path = real_dir / "report.json"
    payload = (json.dumps({"schema": FIXED_X_SCHEMA}, sort_keys=True) + "\n").encode()
    digest = _write_report_bytes(path, payload)
    leaf_link = tmp_path / "leaf.json"
    leaf_link.symlink_to(path)
    with pytest.raises(ContractError, match="non-symlink file"):
        comparator._load_report(
            leaf_link,
            expected_sha256=digest,
            schema=FIXED_X_SCHEMA,
            require_pass=False,
        )

    parent_link = tmp_path / "linked-parent"
    parent_link.symlink_to(real_dir, target_is_directory=True)
    with pytest.raises(ContractError, match="traverses a symlink"):
        comparator._load_report(
            parent_link / "report.json",
            expected_sha256=digest,
            schema=FIXED_X_SCHEMA,
            require_pass=False,
        )


def test_report_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    payload = (
        b'{"schema":"picf-next.lingbot-native-vl-fixed-x-g0.v3",'
        b'"schema":"picf-next.lingbot-native-vl-fixed-x-g0.v3"}'
    )
    digest = _write_report_bytes(path, payload)

    with pytest.raises(ContractError, match="duplicate key"):
        comparator._load_report(
            path,
            expected_sha256=digest,
            schema=FIXED_X_SCHEMA,
            require_pass=False,
        )

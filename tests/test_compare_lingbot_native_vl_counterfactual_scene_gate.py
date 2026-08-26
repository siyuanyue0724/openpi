from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    qwen_grounding_label,
)
from picf_next.lingbot_native.native_vl_fixed_x_metrics import (
    native_vl_fixed_x_pair_geometry_metrics,
)
from picf_next.lingbot_native.native_vl_scene_metrics import (
    native_vl_scene_bank_summary,
    normalize_scene_label,
)
from tools import compare_lingbot_native_vl_counterfactual_scene_gate as comparator

_RUNTIME_PYTHON_TREES = {
    "lingbot": dict(comparator.ADR127_LINGBOT_RUNTIME_PYTHON_TREE),
    "picf": {"file_count": 200, "tree_sha256": "b" * 64},
}


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _model_hash(marker: str) -> dict[str, str]:
    return {"model-00001-of-00001.safetensors": marker * 64}


def _factor_rank(
    *,
    rank: int,
    factor_name: str,
    source_digest: str,
    target: bool,
) -> dict[str, Any]:
    row = {
        "assistant_text_sha256": _digest(f"answer-{factor_name}-{rank}"),
        "camera_name": "static",
        "factor_name": factor_name,
        "global_index": 100,
        "image_grid_thw": [[1, 16, 16]],
        "loss": 1.0,
        "loss_weight": 0.5,
        "rank": rank,
        "record_type": "target" if target else "scene",
        "source_rgb_sha256": source_digest,
        "supervised_token_count": 8,
        "user_text_sha256": _digest(f"user-{factor_name}-{rank}"),
        "visual_lattice": 8,
    }
    if target:
        row.update(
            {
                "assistant_text_sha256": _digest(f"target-answer-{rank}"),
                "instruction": f"instruction-{rank}",
                "target_identity_key": "movable/block_blue",
                "task_key": "push_blue_block_right",
            }
        )
    else:
        order = (
            CALVIN_QWEN_SCENE_IDENTITY_ORDER
            if rank == 0
            else tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
        )
        row.update(
            {
                "absent_identity_keys": [
                    key for key in order if key in CALVIN_QWEN_SCENE_IDENTITY_ORDER[2:]
                ],
                "category_identity_order": list(order),
                "object_identity_keys": [
                    key for key in order if key == CALVIN_QWEN_SCENE_IDENTITY_ORDER[0]
                ],
                "subpatch_visible_identity_keys": [
                    key for key in order if key == CALVIN_QWEN_SCENE_IDENTITY_ORDER[1]
                ],
            }
        )
    return row


def _gram_summary(
    *,
    element_count: int,
    parameter_tensor_count: int,
    target_squared: float = 1.0,
    scene_squared: float = 1.0,
    public_squared: float = 1.0,
    target_scene: float = 0.0,
    target_public: float = 0.0,
    scene_public: float = 0.0,
) -> dict[str, Any]:
    squared = {
        "target": target_squared,
        "scene": scene_squared,
        "public": public_squared,
    }
    dots = {
        "target__scene": target_scene,
        "target__public": target_public,
        "scene__public": scene_public,
    }
    directional = {
        "target": 0.5 * target_squared + 0.5 * target_scene + 0.1 * target_public,
        "scene": 0.5 * target_scene + 0.5 * scene_squared + 0.1 * scene_public,
        "public": 0.5 * target_public + 0.5 * scene_public + 0.1 * public_squared,
    }
    cosines = {}
    for pair, (left, right) in {
        "target__scene": ("target", "scene"),
        "target__public": ("target", "public"),
        "scene__public": ("scene", "public"),
    }.items():
        denominator = math.sqrt(squared[left] * squared[right])
        cosines[pair] = None if denominator == 0.0 else dots[pair] / denominator
    mixed_squared = (
        0.25 * target_squared
        + 0.25 * scene_squared
        + 0.01 * public_squared
        + 0.5 * target_scene
        + 0.1 * target_public
        + 0.1 * scene_public
    )
    return {
        "element_count": element_count,
        "gradient_squared_norms": squared,
        "gradient_norms": {key: math.sqrt(value) for key, value in squared.items()},
        "mixed_gradient_descends": {key: value > 0.0 for key, value in directional.items()},
        "mixed_gradient_directional_inner_products": directional,
        "mixed_gradient_norm": math.sqrt(mixed_squared),
        "pairwise_dot_products": dots,
        "pairwise_cosines": cosines,
        "parameter_tensor_count": parameter_tensor_count,
    }


def _gradient_audit(steps: list[dict[str, Any]]) -> dict[str, Any]:
    reports = []
    for step in comparator.ADR127_GRADIENT_AUDIT_STEPS:
        objective_rank_reports = []
        for objective in comparator.ADR127_GRADIENT_OBJECTIVES:
            if objective == "public":
                source_ranks = steps[step]["public_vl_retention"]["ranks"]
                binding_fields = (
                    "assistant_text_sha256",
                    "image_rgb_sha256",
                    "image_grid_thw",
                    "rank",
                    "record_id",
                    "record_sha256",
                    "supervised_token_count",
                    "user_text_sha256",
                )
            else:
                factors = steps[step]["microbatches"][0]["factors"]
                source_ranks = next(
                    factor["ranks"] for factor in factors if factor["factor_name"] == objective
                )
                binding_fields = (
                    "assistant_text_sha256",
                    "camera_name",
                    "global_index",
                    "image_grid_thw",
                    "rank",
                    "record_type",
                    "source_rgb_sha256",
                    "supervised_token_count",
                    "user_text_sha256",
                    "visual_lattice",
                )
            objective_rank_reports.append(
                {
                    "objective": objective,
                    "ranks": [
                        {
                            **{field: source_ranks[rank][field] for field in binding_fields},
                            "elapsed_seconds": 1.0,
                            "grid_contract": {"lattice": 8},
                            "loss": 1.0,
                            "objective": objective,
                            "rank": rank,
                            "supervised_token_count": source_ranks[rank]["supervised_token_count"],
                        }
                        for rank in range(2)
                    ],
                }
            )
        reports.append(
            {
                "completed_updates_before_audit": step,
                "elapsed_seconds": 3.0,
                "objective_rank_reports": objective_rank_reports,
                "status": "PASS",
                "summary": {
                    "global": _gram_summary(
                        element_count=4_049_739_776,
                        parameter_tensor_count=404,
                    ),
                    "groups": {
                        "language_embedding_tied_lm_head": _gram_summary(
                            element_count=2_000_000_000,
                            parameter_tensor_count=200,
                            target_squared=0.4,
                            scene_squared=0.4,
                            public_squared=0.4,
                        ),
                        "visual_merger": _gram_summary(
                            element_count=2_049_739_776,
                            parameter_tensor_count=204,
                            target_squared=0.6,
                            scene_squared=0.6,
                            public_squared=0.6,
                        ),
                    },
                    "objective_weights": comparator.ADR127_GRADIENT_WEIGHTS,
                    "parameter_count": 404,
                },
            }
        )
    return {
        "enabled": True,
        "objective_weights": comparator.ADR127_GRADIENT_WEIGHTS,
        "reports": reports,
        "status": "PASS",
        "step_indices": list(comparator.ADR127_GRADIENT_AUDIT_STEPS),
    }


def _training_report(*, candidate: bool) -> dict[str, Any]:
    output_marker = "c" if candidate else "b"
    source_digest = "1" * 64
    steps = []
    for step in range(64):
        target_ranks = [
            _factor_rank(
                rank=rank,
                factor_name="target",
                source_digest=source_digest,
                target=True,
            )
            for rank in range(2)
        ]
        if candidate:
            second_name = "scene"
            second_ranks = [
                _factor_rank(
                    rank=rank,
                    factor_name=second_name,
                    source_digest=source_digest,
                    target=False,
                )
                for rank in range(2)
            ]
        else:
            second_name = "target_repeat"
            second_ranks = deepcopy(target_ranks)
            for row in second_ranks:
                row["factor_name"] = second_name
        factors = [
            {"factor_name": "target", "loss_weight": 0.5, "ranks": target_ranks},
            {"factor_name": second_name, "loss_weight": 0.5, "ranks": second_ranks},
        ]
        public = {
            "ranks": [
                {
                    "family": family,
                    "grid_budget": {"lattice": 8},
                    "image_height": 480,
                    "image_rgb_sha256": f"{step + rank + 1200:064x}",
                    "image_grid_thw": [[1, 16, 16]],
                    "image_width": 640,
                    "loss": 1.0,
                    "loss_weight": 0.1,
                    "rank": rank,
                    "record_id": f"{family}-{step}",
                    "record_sha256": f"{step + rank + 1300:064x}",
                    "source_row_index": step,
                    "source_subindex": 0,
                    "supervised_token_count": 4,
                    "target_answer_sha256": f"{step + rank + 1400:064x}",
                    "user_text": "question",
                    "user_text_sha256": _digest("question"),
                    "assistant_text_sha256": f"{step + rank + 1400:064x}",
                }
                for rank, family in enumerate(("referring", "vqa"))
            ]
        }
        steps.append(
            {
                "curriculum_group_index": step,
                "curriculum_optimizer_step": step,
                "gradient_metrics": {
                    "all_finite": True,
                    "frozen_gradient_elements": 0,
                    "trainable_gradient_elements": 4_049_739_776,
                },
                "learning_rate": 1e-6,
                "microbatches": [{"factors": factors, "visual_lattice": 8}],
                "observation_mode": "official_native_once",
                "optimizer_step": step,
                "public_vl_retention": public,
            }
        )
    return {
        "calvin_factor_contract": {
            "adr127_smoke": False,
            "mode": ("counterfactual_scene_candidate" if candidate else "target_repeat_control"),
            "rank_factor_weights_before_microbatch_average": (
                {"scene": 0.5, "target": 0.5}
                if candidate
                else {"target": 0.5, "target_repeat": 0.5}
            ),
        },
        "candidate_model_file_sha256": _model_hash(output_marker),
        "counterfactual_gradient_audit": (
            _gradient_audit(steps) if candidate else {"enabled": False}
        ),
        "cuda_allocator": "expandable-segments",
        "dataset_manifest_sha256": "2" * 64,
        "fsdp2_placement": "gpu-sharded",
        "hyperparameters": {
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "learning_rate": 1e-6,
            "max_grad_norm": 1.0,
            "max_steps": 64,
            "schedule": "linear-warmup-cosine-decay",
            "schedule_total_steps": 432,
            "warmup_steps": 0,
            "weight_decay": 0.0,
        },
        "initial_qwen": {
            "load_result": {"missing_keys": [], "unexpected_keys": []},
            "model_file_sha256": _model_hash("a"),
            "revision": comparator.ADR127_INITIAL_QWEN_REVISION,
        },
        "native_vl_patch_sha256": "3" * 64,
        "observation_mode": "official_native_once",
        "optimizer": "torch.optim.AdamW",
        "optimizer_state_parameter_count": 404,
        "physical_sidecar_manifest_sha256": "7" * 64,
        "picf_code_revision": "8" * 40,
        "processor_lattices": {"8": {"lattice": 8}},
        "processor_snapshot_size": 1,
        "public_vl_retention": {"artifact_sha256": "4" * 64, "enabled": True},
        "runtime_python_trees": deepcopy(_RUNTIME_PYTHON_TREES),
        "schema": comparator.TRAINING_SCHEMA,
        "seed": 20260801,
        "source_commit": "5" * 40,
        "status": "PASS",
        "step_reports": steps,
        "teacher_prune": {"removed": []},
        "trainable_scope": {"parameter_count": 404, "trainable_numel": 4_049_739_776},
        "training_plan": {"artifact_sha256": "6" * 64, "visual_lattices": [8]},
        "world_size": 2,
    }


_FAMILY_SPECS = (
    ("block", "push_blue_block_right", "movable/block_blue", "blue block", 107),
    ("drawer", "open_drawer", "part/table/drawer_link", "drawer", 41),
    ("slider", "move_slider_left", "part/table/slide_link", "sliding door", 41),
    ("led", "turn_on_led", "part/table/button_link", "push button", 40),
    (
        "lightbulb",
        "turn_on_lightbulb",
        "part/table/switch_link",
        "light switch",
        43,
    ),
)


def _target_specs() -> list[tuple[str, str, str, str]]:
    values = []
    for family, task, identity, label, count in _FAMILY_SPECS:
        if family == "block":
            colors = ("blue", "pink", "red")
            values.extend(
                (
                    family,
                    f"lift_{color}_block_table",
                    f"movable/block_{color}",
                    f"{color} block",
                )
                for color in (colors[index % len(colors)] for index in range(count))
            )
        else:
            values.extend((family, task, identity, label) for _ in range(count))
    assert len(values) == 272
    return values


def _target_variant(
    *, task: str, identity: str, label: str, target: list[int], prediction: list[int]
) -> dict[str, Any]:
    instruction = f"do {task}"
    request = "Locate the requested object."
    target_answer = json.dumps([{"label": label, "bbox_2d": target}], separators=(",", ":"))
    generated = json.dumps([{"label": label, "bbox_2d": prediction}], separators=(",", ":"))
    parsed_hit = target[0] <= (target[0] + target[2]) / 2 <= target[2]
    return {
        "camera_name": "static",
        "generated_bbox_qwen_xyxy": prediction,
        "generated_bbox_schema_valid": True,
        "generated_label": label,
        "generated_label_present": True,
        "generated_label_schema_valid": True,
        "generated_text": generated,
        "grounding_request": request,
        "grounding_request_sha256": _digest(request),
        "instruction": instruction,
        "instruction_sha256": _digest(instruction),
        "normalized_label_exact_match": True,
        "target_answer": target_answer,
        "target_answer_sha256": _digest(target_answer),
        "target_bbox_qwen_xyxy": target,
        "target_identity_key": identity,
        "target_label": label,
        "task_key": task,
        "own_target_center_hit": parsed_hit,
    }


def _target_groups(*, candidate: bool) -> list[dict[str, Any]]:
    specs = _target_specs()
    groups = []
    for ordinal in range(136):
        first = specs[2 * ordinal]
        second = specs[2 * ordinal + 1]
        global_index = 10_000 + ordinal
        if ordinal == 114:
            first = ("led", "turn_off_led", "part/table/button_link", "push button")
            second = (
                "lightbulb",
                "turn_on_lightbulb",
                "part/table/switch_link",
                "light switch",
            )
            global_index = comparator.ADR127_CLEAN_COLLISION_GLOBAL_INDEX
        targets = tuple(
            [400, 400, 600, 600]
            if spec[2] == "part/table/drawer_link"
            else ([100, 100, 200, 200] if index == 0 else [700, 700, 800, 800])
            for index, spec in enumerate((first, second))
        )
        predictions = [list(targets[0]), list(targets[1])]
        if ordinal == 114 and not candidate:
            predictions[1] = list(targets[0])
        variants = [
            _target_variant(
                task=spec[1],
                identity=spec[2],
                label=spec[3],
                target=list(target),
                prediction=prediction,
            )
            for spec, target, prediction in zip((first, second), targets, predictions, strict=True)
        ]
        metrics = native_vl_fixed_x_pair_geometry_metrics(
            (tuple(predictions[0]), tuple(predictions[1])),
            (tuple(targets[0]), tuple(targets[1])),
        )
        for variant, values in zip(variants, metrics["variants"], strict=True):
            variant.update(values)
        groups.append(
            {
                "ordinal": ordinal % 68,
                "pair_metrics": {key: value for key, value in metrics.items() if key != "variants"},
                "partition": "validation" if ordinal < 68 else "heldout",
                "source_global_index": global_index,
                "source_rgb_sha256": f"{ordinal + 10:064x}",
                "source_state_sha256": f"{ordinal + 200:064x}",
                "variants": variants,
            }
        )
    return groups


def _scene_rows(*, candidate: bool) -> list[dict[str, Any]]:
    rows = []
    for index in range(32):
        identity_keys = (
            CALVIN_QWEN_SCENE_IDENTITY_ORDER[index % len(CALVIN_QWEN_SCENE_IDENTITY_ORDER)],
            CALVIN_QWEN_SCENE_IDENTITY_ORDER[(index + 3) % len(CALVIN_QWEN_SCENE_IDENTITY_ORDER)],
        )
        targets = {
            identity_keys[0]: [100, 100, 200, 200],
            identity_keys[1]: [700, 700, 800, 800],
        }
        variants = []
        generated_maps = []
        for order_index in range(2):
            category_order = (
                CALVIN_QWEN_SCENE_IDENTITY_ORDER
                if order_index == 0
                else tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
            )
            ordered_identities = [item for item in category_order if item in targets]
            generated_boxes = {key: list(value) for key, value in targets.items()}
            if not candidate and index == 0 and order_index == 1:
                generated_boxes[ordered_identities[0]] = [300, 300, 400, 400]
            generated_items = [
                {
                    "label": qwen_grounding_label(identity_key),
                    "bbox_2d": generated_boxes[identity_key],
                }
                for identity_key in ordered_identities
            ]
            target_items = [
                {
                    "label": qwen_grounding_label(identity_key),
                    "bbox_2d": targets[identity_key],
                }
                for identity_key in ordered_identities
            ]
            generated = json.dumps(generated_items, separators=(",", ":"))
            request = f"scene request {order_index}"
            answer = json.dumps(target_items, separators=(",", ":"))
            generated_maps.append(
                {normalize_scene_label(item["label"]): item["bbox_2d"] for item in generated_items}
            )
            variants.append(
                {
                    "category_identity_order": list(category_order),
                    "expected_label_order": [
                        normalize_scene_label(item["label"]) for item in target_items
                    ],
                    "expected_object_count": 2,
                    "extra_labels": [],
                    "generated_label_order": [
                        normalize_scene_label(item["label"]) for item in generated_items
                    ],
                    "generated_object_count": 2,
                    "generated_text": generated,
                    "generated_text_sha256": _digest(generated),
                    "grounding_request": request,
                    "grounding_request_sha256": _digest(request),
                    "label_set_exact": True,
                    "missing_labels": [],
                    "objects": [
                        {
                            "center_selective": generated_boxes[identity_key]
                            == targets[identity_key],
                            "generated_bbox_qwen_xyxy": generated_boxes[identity_key],
                            "generated_center_hit": generated_boxes[identity_key]
                            == targets[identity_key],
                            "identity_key": identity_key,
                            "label": qwen_grounding_label(identity_key),
                            "label_found": True,
                            "target_bbox_qwen_xyxy": targets[identity_key],
                            "target_center_hit": generated_boxes[identity_key]
                            == targets[identity_key],
                            "target_iou": (
                                1.0
                                if generated_boxes[identity_key] == targets[identity_key]
                                else 0.0
                            ),
                            "unexpected_center_hit_labels": [],
                        }
                        for identity_key in ordered_identities
                    ],
                    "order_exact": True,
                    "order_variant": "canonical" if order_index == 0 else "reverse",
                    "schema_valid": True,
                    "target_answer": answer,
                    "target_answer_sha256": _digest(answer),
                }
            )
        exact_map = generated_maps[0] == generated_maps[1]
        pair_pass = exact_map and all(
            object_row["center_selective"]
            for variant in variants
            for object_row in variant["objects"]
        )
        rows.append(
            {
                "bank_index": index,
                "group_index": 500 + index,
                "pair_metrics": {
                    "label_box_map_exact": exact_map,
                    "pair_pass": pair_pass,
                    "variants": variants,
                },
                "source_global_index": 20_000 + index,
                "source_rgb_sha256": f"{index + 600:064x}",
                "task_keys": ["push_blue_block_left", "push_blue_block_right"],
            }
        )
    return rows


def _public_rows(*, nll: float = 0.1) -> list[dict[str, Any]]:
    rows = []
    for family in ("referring", "vqa"):
        for index in range(32):
            target = (
                json.dumps({"bbox_2d": [100, 100, 200, 200]}, separators=(",", ":"))
                if family == "referring"
                else "answer"
            )
            generated = target
            user = "question"
            row = {
                "family": family,
                "generated_text": generated,
                "image_height": 480,
                "image_rgb_sha256": f"{index + (700 if family == 'referring' else 800):064x}",
                "image_grid_thw": [[1, 16, 16]],
                "image_width": 640,
                "mean_token_nll": nll,
                "record_id": f"{family}-{index}",
                "record_sha256": f"{index + (900 if family == 'referring' else 1000):064x}",
                "source_row_index": index,
                "source_subindex": 0,
                "supervised_token_count": 4,
                "target_answer": target,
                "target_answer_sha256": _digest(target),
                "user_text": user,
                "user_text_sha256": _digest(user),
            }
            if family == "referring":
                row.update(
                    {
                        "generated_bbox_qwen_xyxy": [100, 100, 200, 200],
                        "generated_bbox_schema_valid": True,
                        "target_center_hit": True,
                    }
                )
            else:
                row["normalized_exact_match"] = True
            rows.append(row)
    return rows


def _fixed_x_report(*, candidate: bool, schema: str | None = None) -> dict[str, Any]:
    scene_rows = _scene_rows(candidate=candidate)
    return {
        "checkpoint_model_file_sha256": {"lingbot.safetensors": "d" * 64},
        "dataset_manifest_sha256": "2" * 64,
        "eligible_item_count": 136,
        "evaluation_plan_artifact_sha256": "e" * 64,
        "evaluation_plan_file_sha256": "f" * 64,
        "excluded_items": [],
        "item_limit_per_partition": 0,
        "max_new_tokens": 64,
        "native_vl_patch_sha256": "3" * 64,
        "partition": "all",
        "picf_code_revision": "8" * 40,
        "physical_sidecar_manifest_sha256": "7" * 64,
        "processor_lattice": {"lattice": 8},
        "runtime_python_trees": deepcopy(_RUNTIME_PYTHON_TREES),
        "public_vl_retention": {
            "artifact_sha256": "4" * 64,
            "enabled": True,
            "heldout_limit_per_family": 32,
            "manifest_file_sha256": "a" * 64,
            "results": _public_rows(),
        },
        "qwen_restore": {
            "model_file_sha256": _model_hash("c" if candidate else "b"),
            "model_revision": "8" * 40,
        },
        "results": _target_groups(candidate=candidate),
        "scene_evaluation": {
            "audit_artifact_sha256": "9" * 64,
            "audit_file_sha256": "0" * 64,
            "enabled": True,
            "generation_budget": {
                "configured_max_new_tokens": 512,
                "headroom_tokens": 241,
                "maximum_target_supervised_tokens": 271,
                "minimum_target_supervised_tokens": 79,
                "target_record_count": 64,
            },
            "max_new_tokens": 512,
            "results": scene_rows,
            "source_disjoint_scene_bank_count": 32,
            "summary": native_vl_scene_bank_summary(scene_rows),
        },
        "schema": schema or comparator.FIXED_X_SCHEMA,
        "seed": 20260802,
        "selected_item_count": 136,
        "source_commit": "5" * 40,
    }


def _reference_fixed_x(candidate: dict[str, Any]) -> dict[str, Any]:
    report = deepcopy(candidate)
    report["schema"] = comparator.ADR125_FIXED_X_SCHEMA
    return report


def _fixed_label_report(*, candidate: bool, reference: bool = False) -> dict[str, Any]:
    rows = []
    for index in range(16):
        is_switch = index < 3
        target_identity = "part/table/switch_link" if is_switch else "movable/block_blue"
        task = "turn_on_lightbulb" if is_switch else "push_blue_block_right"
        correct_mean = 1.0
        if candidate:
            distractor_mean = 2.0
        elif reference and is_switch:
            distractor_mean = 0.8
        else:
            distractor_mean = 0.9 if is_switch else 2.0
        correct = {
            "assistant_text": "correct",
            "bbox_xyxy": [1, 1, 2, 2],
            "mean_token_nll": correct_mean,
            "qwen_bbox_xyxy": [10, 10, 20, 20],
            "sequence_nll": correct_mean * 10,
            "supervised_token_count": 10,
        }
        distractor = {
            "assistant_text": "wrong",
            "bbox_xyxy": [3, 3, 4, 4],
            "distractor_identity_key": "part/table/button_link",
            "mean_token_nll": distractor_mean,
            "qwen_bbox_xyxy": [30, 30, 40, 40],
            "sequence_nll": distractor_mean * 10,
            "supervised_token_count": 10,
        }
        rows.append(
            {
                "bbox_xyxy": [1, 1, 2, 2],
                "camera_name": "static" if index % 2 == 0 else "gripper",
                "correct_answer": correct,
                "distractors": [distractor],
                "global_index": 30_000 + index,
                "host_image_key": "observation.images.image",
                "instruction": f"instruction {index}",
                "nll_margin": distractor_mean - correct_mean,
                "qwen_restore": {},
                "sequence_nll_margin": (distractor_mean - correct_mean) * 10,
                "source_rgb_sha256": f"{index + 1100:064x}",
                "target_identity_key": target_identity,
                "task_key": task,
            }
        )
    return {
        "dataset_manifest_sha256": "2" * 64,
        "native_vl_patch_sha256": "3" * 64,
        **({} if reference else {"picf_code_revision": "8" * 40}),
        "record_count": 16,
        "records": rows,
        "qwen_restore": {
            "model_file_sha256": _model_hash("c" if candidate else "b"),
            "model_revision": "8" * 40,
        },
        "runtime_python_trees": (None if reference else deepcopy(_RUNTIME_PYTHON_TREES)),
        "schema": (
            comparator.ADR126_FIXED_LABEL_SCHEMA if reference else comparator.FIXED_LABEL_SCHEMA
        ),
        "source_commit": "5" * 40,
        "uniform_vocabulary_nll": math.log(1000),
        "visual_sha256": {f"record-{index:02d}.png": f"{index + 1500:064x}" for index in range(16)},
        "vocabulary_size": 1000,
    }


def _inputs() -> dict[str, Any]:
    control_fixed_x = _fixed_x_report(candidate=False)
    candidate_fixed_x = _fixed_x_report(candidate=True)
    return {
        "adr125_reference_fixed_x": _reference_fixed_x(candidate_fixed_x),
        "adr126_reference_fixed_label": _fixed_label_report(candidate=False, reference=True),
        "control_training": _training_report(candidate=False),
        "candidate_training": _training_report(candidate=True),
        "control_fixed_x": control_fixed_x,
        "candidate_fixed_x": candidate_fixed_x,
        "control_fixed_label": _fixed_label_report(candidate=False),
        "candidate_fixed_label": _fixed_label_report(candidate=True),
        "expected_runtime_python_trees": deepcopy(_RUNTIME_PYTHON_TREES),
        "input_report_sha256": {
            "adr125_reference_fixed_x": comparator.ADR125_REFERENCE_FIXED_X_SHA256,
            "adr126_reference_fixed_label": comparator.ADR126_REFERENCE_FIXED_LABEL_SHA256,
            "control_training": "1" * 64,
            "candidate_training": "2" * 64,
            "control_fixed_x": "3" * 64,
            "candidate_fixed_x": "4" * 64,
            "control_fixed_label": "5" * 64,
            "candidate_fixed_label": "6" * 64,
        },
    }


def test_counterfactual_scene_gate_passes_only_complete_recomputed_evidence() -> None:
    report = comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**_inputs())

    assert report["schema"] == comparator.OUTPUT_SCHEMA
    assert report["status"] == "PASS"
    assert report["sections"]["scene"]["checks"]["all_candidate_scene_pairs_pass"] == {
        "actual": 32,
        "passed": True,
        "relation": "==",
        "threshold": 32,
    }
    assert (
        report["sections"]["clean_collision"]["checks"]["candidate_beats_control_clean_collision"][
            "passed"
        ]
        is True
    )


def test_counterfactual_scene_gate_returns_fail_with_explicit_failed_threshold() -> None:
    inputs = _inputs()
    for row in inputs["candidate_fixed_x"]["public_vl_retention"]["results"]:
        if row["family"] == "vqa":
            row["mean_token_nll"] = 0.5

    report = comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    assert report["status"] == "FAIL"
    check = report["sections"]["public"]["checks"]["vqa_token_weighted_nll_ceiling"]
    assert check == {
        "actual": 0.5,
        "passed": False,
        "relation": "<=",
        "threshold": 0.345908,
    }


def test_pixel_causal_gate_rejects_same_prompt_fixed_box_lookup() -> None:
    report = _fixed_x_report(candidate=True)
    for group in report["results"]:
        for variant in group["variants"]:
            if variant["target_identity_key"] != "movable/block_blue":
                continue
            variant["generated_text"] = json.dumps(
                [{"label": variant["target_label"], "bbox_2d": [100, 100, 200, 200]}],
                separators=(",", ":"),
            )

    metrics = comparator._pixel_causal_metrics(report)
    blue = next(row for row in metrics["sentinels"] if row["identity_key"] == "movable/block_blue")
    assert blue["sentinel_pass"] is False
    assert metrics["pass_count"] == metrics["sentinel_count"] - 1


def test_counterfactual_scene_gate_rejects_self_consistent_untrusted_runtime_tree() -> None:
    inputs = _inputs()
    wrong = {"file_count": 200, "tree_sha256": "f" * 64}
    for key in (
        "control_training",
        "candidate_training",
        "control_fixed_x",
        "candidate_fixed_x",
        "control_fixed_label",
        "candidate_fixed_label",
    ):
        inputs[key]["runtime_python_trees"]["picf"] = deepcopy(wrong)

    with pytest.raises(ContractError, match="trusted checkout"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_rejects_order_stable_wrong_scene_box() -> None:
    inputs = _inputs()
    scene = inputs["candidate_fixed_x"]["scene_evaluation"]
    row = scene["results"][0]
    wrong_box = [300, 300, 400, 400]
    wrong_label = normalize_scene_label(row["pair_metrics"]["variants"][0]["objects"][0]["label"])
    for variant in row["pair_metrics"]["variants"]:
        generated_items = json.loads(variant["generated_text"])
        generated_item = next(
            item for item in generated_items if normalize_scene_label(item["label"]) == wrong_label
        )
        generated_item["bbox_2d"] = wrong_box
        generated = json.dumps(generated_items, separators=(",", ":"))
        variant["generated_text"] = generated
        variant["generated_text_sha256"] = _digest(generated)
        object_row = next(
            item
            for item in variant["objects"]
            if normalize_scene_label(item["label"]) == wrong_label
        )
        object_row.update(
            {
                "center_selective": False,
                "generated_bbox_qwen_xyxy": wrong_box,
                "generated_center_hit": False,
                "target_center_hit": False,
                "target_iou": 0.0,
                "unexpected_center_hit_labels": [],
            }
        )
    row["pair_metrics"]["label_box_map_exact"] = True
    row["pair_metrics"]["pair_pass"] = False
    scene["summary"] = native_vl_scene_bank_summary(scene["results"])

    report = comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    assert report["status"] == "FAIL"
    checks = report["sections"]["scene"]["checks"]
    assert checks["all_candidate_scene_pairs_pass"]["passed"] is False
    assert checks["all_candidate_scene_objects_center_selective"]["passed"] is False


def test_counterfactual_scene_gate_rejects_single_object_scene() -> None:
    inputs = _inputs()
    variants = inputs["candidate_fixed_x"]["scene_evaluation"]["results"][0]["pair_metrics"][
        "variants"
    ]
    for variant in variants:
        target = json.dumps(json.loads(variant["target_answer"])[:1], separators=(",", ":"))
        variant["target_answer"] = target
        variant["target_answer_sha256"] = _digest(target)

    with pytest.raises(ContractError, match="scene evaluation is not multi-object"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_returns_fail_for_destructive_mixed_gradient() -> None:
    inputs = _inputs()
    audit = inputs["candidate_training"]["counterfactual_gradient_audit"]
    first = audit["reports"][0]
    first["summary"]["global"] = _gram_summary(
        element_count=4_049_739_776,
        parameter_tensor_count=404,
        target_squared=1.0,
        scene_squared=4.0,
        public_squared=1.0,
        target_scene=-2.0,
    )
    first["summary"]["groups"] = {
        "language_embedding_tied_lm_head": _gram_summary(
            element_count=2_000_000_000,
            parameter_tensor_count=200,
            target_squared=0.4,
            scene_squared=1.6,
            public_squared=0.4,
            target_scene=-0.8,
        ),
        "visual_merger": _gram_summary(
            element_count=2_049_739_776,
            parameter_tensor_count=204,
            target_squared=0.6,
            scene_squared=2.4,
            public_squared=0.6,
            target_scene=-1.2,
        ),
    }
    first["status"] = "FAIL"
    audit["status"] = "FAIL"

    report = comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    assert report["status"] == "FAIL"
    check = report["sections"]["gradient"]["checks"][
        "candidate_mixed_gradient_descends_target_and_scene"
    ]
    assert check["passed"] is False
    assert check["actual"] == {
        "report_statuses": ["FAIL", "PASS", "PASS", "PASS"],
        "status": "FAIL",
    }


def test_counterfactual_scene_gate_rejects_unrecomputed_gradient_flag() -> None:
    inputs = _inputs()
    first_global = inputs["candidate_training"]["counterfactual_gradient_audit"]["reports"][0][
        "summary"
    ]["global"]
    first_global["mixed_gradient_descends"]["scene"] = False

    with pytest.raises(ContractError, match="direction flag was not Gram-recomputed"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_rejects_unrecomputed_gradient_cosine() -> None:
    inputs = _inputs()
    first_global = inputs["candidate_training"]["counterfactual_gradient_audit"]["reports"][0][
        "summary"
    ]["global"]
    first_global["pairwise_cosines"]["target__scene"] = 0.5

    with pytest.raises(ContractError, match="cosine was not Gram-recomputed"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_rejects_pairwise_valid_non_psd_gram() -> None:
    inputs = _inputs()
    first = inputs["candidate_training"]["counterfactual_gradient_audit"]["reports"][0]
    first["summary"]["global"] = _gram_summary(
        element_count=4_049_739_776,
        parameter_tensor_count=404,
        target_scene=0.9,
        target_public=0.9,
        scene_public=-0.9,
    )

    with pytest.raises(ContractError, match="not positive semidefinite"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_rejects_group_moments_not_summing_to_global() -> None:
    inputs = _inputs()
    first_groups = inputs["candidate_training"]["counterfactual_gradient_audit"]["reports"][0][
        "summary"
    ]["groups"]
    first_groups["visual_merger"]["gradient_squared_norms"]["target"] = 0.7
    first_groups["visual_merger"]["gradient_norms"]["target"] = math.sqrt(0.7)
    first_groups["visual_merger"]["mixed_gradient_directional_inner_products"]["target"] = 0.35
    first_groups["visual_merger"]["mixed_gradient_descends"]["target"] = True
    first_groups["visual_merger"]["mixed_gradient_norm"] = math.sqrt(0.331)

    with pytest.raises(ContractError, match="moments do not sum to global"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_rejects_v4_and_mismatched_training_source() -> None:
    inputs = _inputs()
    inputs["candidate_fixed_x"]["schema"] = "picf-next.lingbot-native-vl-fixed-x-g0.v4"
    with pytest.raises(ContractError, match="fixed-X schema"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_training"]["step_reports"][1]["curriculum_group_index"] = 99
    with pytest.raises(ContractError, match="schedule/source"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_training"]["step_reports"][1]["gradient_metrics"][
        "trainable_gradient_elements"
    ] -= 1
    with pytest.raises(ContractError, match="training gradient contract"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["control_training"]["seed"] = 7
    inputs["candidate_training"]["seed"] = 7
    with pytest.raises(ContractError, match="training seed changed"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["control_training"]["physical_sidecar_manifest_sha256"] = "6" * 64
    inputs["candidate_training"]["physical_sidecar_manifest_sha256"] = "6" * 64
    with pytest.raises(ContractError, match="evaluation sidecar differs from training"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_rejects_model_scene_and_margin_tampering() -> None:
    inputs = _inputs()
    inputs["candidate_fixed_x"]["qwen_restore"]["model_file_sha256"] = _model_hash("f")
    with pytest.raises(ContractError, match="training output"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["scene_evaluation"]["summary"]["pair_pass_count"] = 0
    with pytest.raises(ContractError, match="scene summary"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["scene_evaluation"]["generation_budget"][
        "maximum_target_supervised_tokens"
    ] = 513
    with pytest.raises(ContractError, match="generation budget is unsound"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["scene_evaluation"]["generation_budget"]["headroom_tokens"] = 242
    with pytest.raises(ContractError, match="generation budget is unsound"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["scene_evaluation"]["generation_budget"]["extra"] = 1
    with pytest.raises(ContractError, match="generation budget fields are invalid"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["scene_evaluation"]["generation_budget"][
        "configured_max_new_tokens"
    ] = 513
    inputs["candidate_fixed_x"]["scene_evaluation"]["generation_budget"]["headroom_tokens"] = 242
    inputs["candidate_fixed_x"]["scene_evaluation"]["max_new_tokens"] = 513
    with pytest.raises(
        ContractError, match="different source-disjoint banks or generation budgets"
    ):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    variant = inputs["candidate_fixed_x"]["scene_evaluation"]["results"][0]["pair_metrics"][
        "variants"
    ][0]
    variant["grounding_request"] = "different request"
    variant["grounding_request_sha256"] = _digest(variant["grounding_request"])
    with pytest.raises(ContractError, match="different source-disjoint banks"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["scene_evaluation"]["results"][0]["pair_metrics"]["variants"][0][
        "objects"
    ][0]["identity_key"] = "part/table/button_link"
    with pytest.raises(ContractError, match="scene object target binding"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_label"]["records"][0]["nll_margin"] = 999.0
    with pytest.raises(ContractError, match="token margin"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_x"]["public_vl_retention"]["manifest_file_sha256"] = "f" * 64
    with pytest.raises(ContractError, match="public evaluation bank"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_label"]["qwen_restore"]["model_revision"] = "f" * 40
    with pytest.raises(ContractError, match="revision differs from training"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_counterfactual_scene_gate_binds_current_fixed_label_code_revision() -> None:
    inputs = _inputs()
    inputs["candidate_fixed_label"].pop("picf_code_revision")
    with pytest.raises(ContractError, match="fixed-label code revision"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)

    inputs = _inputs()
    inputs["candidate_fixed_label"]["picf_code_revision"] = "f" * 40
    with pytest.raises(ContractError, match="fixed-label code revision differs from training"):
        comparator.compare_lingbot_native_vl_counterfactual_scene_gate(**inputs)


def test_strict_loader_rejects_digest_duplicate_key_and_symlink(tmp_path: Path) -> None:
    report_path = tmp_path / "report.json"
    payload = json.dumps({"schema": comparator.FIXED_X_SCHEMA}).encode()
    report_path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    report, actual = comparator._load_report(
        report_path,
        expected_sha256=digest,
        schemas=comparator.FIXED_X_SCHEMA,
    )
    assert report["schema"] == comparator.FIXED_X_SCHEMA
    assert actual == digest

    with pytest.raises(ContractError, match="digest changed"):
        comparator._load_report(
            report_path,
            expected_sha256="0" * 64,
            schemas=comparator.FIXED_X_SCHEMA,
        )

    duplicate = b'{"schema":"picf-next.lingbot-native-vl-fixed-x-g0.v6","schema":"x"}'
    report_path.write_bytes(duplicate)
    with pytest.raises(ContractError, match="duplicate key"):
        comparator._load_report(
            report_path,
            expected_sha256=hashlib.sha256(duplicate).hexdigest(),
            schemas=comparator.FIXED_X_SCHEMA,
        )

    nonfinite = b'{"schema":"picf-next.lingbot-native-vl-fixed-x-g0.v6","value":NaN}'
    report_path.write_bytes(nonfinite)
    with pytest.raises(ContractError, match="invalid strict JSON"):
        comparator._load_report(
            report_path,
            expected_sha256=hashlib.sha256(nonfinite).hexdigest(),
            schemas=comparator.FIXED_X_SCHEMA,
        )

    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ContractError, match="real file"):
        comparator._load_report(
            link,
            expected_sha256=hashlib.sha256(b"{}").hexdigest(),
            schemas=comparator.FIXED_X_SCHEMA,
        )

"""Pure recomputation metrics for Qwen-native fixed-observation evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from picf_next.contracts import ContractError
from picf_next.eval.calvin_task_relevance import calvin_exact_task_loss_identities
from picf_next.lingbot_native.vl_cotraining import (
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
)

QwenBBox = tuple[int, int, int, int]
CALVIN_GROUNDING_FAMILIES = ("block", "drawer", "slider", "led", "lightbulb")
CALVIN_GROUNDING_FAMILY_VARIANT_COUNTS = {
    "block": 107,
    "drawer": 41,
    "slider": 41,
    "led": 40,
    "lightbulb": 43,
}


def normalize_native_vl_answer(value: str) -> str:
    """Normalize generated free-form answers for exact-match evaluation."""

    if not isinstance(value, str):
        raise TypeError("public retention answer must be text")
    normalized = value.replace("<|im_end|>", " ").replace("<|endoftext|>", " ")
    return " ".join(normalized.casefold().split())


def native_vl_calvin_task_family(task_key: str, target_identity_key: str) -> str:
    """Map one exact CALVIN task through the pinned physical-target protocol."""

    identities = calvin_exact_task_loss_identities(task_key)
    if identities is None or identities != (target_identity_key,):
        raise ContractError("fixed-X task and physical target identity differ")
    if target_identity_key.startswith("movable/block_"):
        return "block"
    families = {
        "part/table/drawer_link": "drawer",
        "part/table/slide_link": "slider",
        "part/table/button_link": "led",
        "part/table/switch_link": "lightbulb",
    }
    try:
        return families[target_identity_key]
    except KeyError as error:
        raise ContractError("fixed-X target identity has no frozen family") from error


def native_vl_fixed_x_pair_geometry_metrics(
    predictions: tuple[QwenBBox | None, QwenBBox | None],
    targets: tuple[QwenBBox, QwenBBox],
) -> dict[str, object]:
    """Recompute all same-image prompt-switch geometry from raw boxes."""

    rows = []
    for index, prediction in enumerate(predictions):
        own = targets[index]
        alternate = targets[1 - index]
        own_iou = qwen_grounding_bbox_iou(prediction, own) if prediction is not None else 0.0
        alternate_iou = (
            qwen_grounding_bbox_iou(prediction, alternate) if prediction is not None else 0.0
        )
        own_hit = qwen_target_center_in_bbox(prediction, own) if prediction is not None else False
        alternate_hit = (
            qwen_target_center_in_bbox(prediction, alternate) if prediction is not None else False
        )
        rows.append(
            {
                "alternate_target_center_hit": alternate_hit,
                "alternate_target_iou": alternate_iou,
                "diagonal_iou_advantage": own_iou - alternate_iou,
                "own_only_center_hit": own_hit and not alternate_hit,
                "own_target_center_hit": own_hit,
                "own_target_iou": own_iou,
            }
        )
    advantages = [float(row["diagonal_iou_advantage"]) for row in rows]
    return {
        "bidirectional_own_only_center_hit": all(bool(row["own_only_center_hit"]) for row in rows),
        "mean_diagonal_iou_advantage": sum(advantages) / len(advantages),
        "prediction_bbox_changed": (
            predictions[0] is not None
            and predictions[1] is not None
            and predictions[0] != predictions[1]
        ),
        "variants": rows,
    }


def native_vl_fixed_x_partition_summary(
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, object]:
    """Recompute one fixed-X partition summary from validated pair rows."""

    if not groups:
        raise ContractError("fixed-X partition summary requires pair rows")
    variants = []
    metrics = []
    for group in groups:
        group_variants = group.get("variants")
        pair_metrics = group.get("pair_metrics")
        if (
            not isinstance(group_variants, list)
            or len(group_variants) != 2
            or any(not isinstance(variant, Mapping) for variant in group_variants)
            or not isinstance(pair_metrics, Mapping)
        ):
            raise ContractError("fixed-X partition row is malformed")
        variants.extend(group_variants)
        metrics.append(pair_metrics)
    return {
        "bidirectional_own_only_center_hit_count": sum(
            bool(metric["bidirectional_own_only_center_hit"]) for metric in metrics
        ),
        "generated_bbox_count": sum(
            variant["generated_bbox_qwen_xyxy"] is not None for variant in variants
        ),
        "generated_bbox_schema_valid_count": sum(
            bool(variant["generated_bbox_schema_valid"]) for variant in variants
        ),
        "item_count": len(groups),
        "mean_diagonal_iou_advantage": sum(
            float(metric["mean_diagonal_iou_advantage"]) for metric in metrics
        )
        / len(metrics),
        "mean_own_target_iou": sum(float(variant["own_target_iou"]) for variant in variants)
        / len(variants),
        "own_target_center_hit_count": sum(
            bool(variant["own_target_center_hit"]) for variant in variants
        ),
        "prediction_bbox_changed_count": sum(
            bool(metric["prediction_bbox_changed"]) for metric in metrics
        ),
        "variant_count": len(variants),
    }

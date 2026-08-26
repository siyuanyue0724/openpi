"""Strict label-addressed metrics for native-Qwen multi-object grounding."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import cast

from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CalvinQwenSceneGroundingRecord,
    qwen_grounding_label,
)
from picf_next.lingbot_native.vl_cotraining import (
    NativeVLGeneratedSceneGrounding,
    qwen_grounding_bbox_iou,
    qwen_target_center_in_bbox,
)


def normalize_scene_label(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError("native VL scene metric label must be nonempty text")
    return " ".join(value.casefold().split())


def native_vl_scene_prediction_metrics(
    record: CalvinQwenSceneGroundingRecord,
    generated: NativeVLGeneratedSceneGrounding,
) -> dict[str, object]:
    """Match one generated list by normalized label, never by sequence position."""

    if not isinstance(record, CalvinQwenSceneGroundingRecord):
        raise TypeError("native VL scene metrics require one scene record")
    if not isinstance(generated, NativeVLGeneratedSceneGrounding):
        raise TypeError("native VL scene metrics require one parsed generation")
    expected = tuple(
        (normalize_scene_label(qwen_grounding_label(item.identity_key)), item)
        for item in record.objects
    )
    target_boxes = {label: record.qwen_bbox_for_object(item) for label, item in expected}
    predicted = {
        normalize_scene_label(item.label): item.bbox_qwen_xyxy for item in generated.objects
    }
    if len(predicted) != len(generated.objects):
        raise ContractError("native VL scene metrics received duplicate normalized labels")
    expected_labels = tuple(label for label, _item in expected)
    predicted_labels = tuple(normalize_scene_label(item.label) for item in generated.objects)
    rows = []
    for label, item in expected:
        prediction = predicted.get(label)
        target = target_boxes[label]
        target_center_hit = (
            False if prediction is None else qwen_target_center_in_bbox(prediction, target)
        )
        generated_center_hit = (
            False if prediction is None else qwen_target_center_in_bbox(target, prediction)
        )
        unexpected_center_hit_labels = []
        if prediction is not None:
            for other_label, other_target in target_boxes.items():
                if other_label == label:
                    continue
                ground_truth_overlaps_other_center = qwen_target_center_in_bbox(
                    target,
                    other_target,
                )
                prediction_hits_other_center = qwen_target_center_in_bbox(
                    prediction,
                    other_target,
                )
                if prediction_hits_other_center and not ground_truth_overlaps_other_center:
                    unexpected_center_hit_labels.append(other_label)
        rows.append(
            {
                "center_selective": (
                    target_center_hit and generated_center_hit and not unexpected_center_hit_labels
                ),
                "generated_bbox_qwen_xyxy": (None if prediction is None else list(prediction)),
                "generated_center_hit": generated_center_hit,
                "identity_key": item.identity_key,
                "label": qwen_grounding_label(item.identity_key),
                "label_found": prediction is not None,
                "target_bbox_qwen_xyxy": list(target),
                "target_center_hit": target_center_hit,
                "target_iou": (
                    0.0 if prediction is None else qwen_grounding_bbox_iou(prediction, target)
                ),
                "unexpected_center_hit_labels": unexpected_center_hit_labels,
            }
        )
    expected_set = set(expected_labels)
    predicted_set = set(predicted_labels)
    return {
        "expected_label_order": list(expected_labels),
        "expected_object_count": len(expected_labels),
        "extra_labels": sorted(predicted_set - expected_set),
        "generated_label_order": list(predicted_labels),
        "generated_object_count": len(predicted_labels),
        "label_set_exact": predicted_set == expected_set,
        "missing_labels": sorted(expected_set - predicted_set),
        "objects": rows,
        "order_exact": predicted_labels == expected_labels,
        "schema_valid": generated.schema_valid,
    }


def native_vl_scene_order_pair_metrics(
    records: tuple[CalvinQwenSceneGroundingRecord, CalvinQwenSceneGroundingRecord],
    generated: tuple[NativeVLGeneratedSceneGrounding, NativeVLGeneratedSceneGrounding],
) -> dict[str, object]:
    """Require order following and an invariant label-to-box map on the same image."""

    if (
        not isinstance(records, tuple)
        or len(records) != 2
        or any(not isinstance(item, CalvinQwenSceneGroundingRecord) for item in records)
        or not isinstance(generated, tuple)
        or len(generated) != 2
        or any(not isinstance(item, NativeVLGeneratedSceneGrounding) for item in generated)
    ):
        raise TypeError("native VL scene pair metrics require two typed records and generations")
    if (
        records[0].source_rgb_sha256 != records[1].source_rgb_sha256
        or records[0].camera_name != records[1].camera_name
        or records[0].global_index != records[1].global_index
    ):
        raise ContractError("native VL scene pair metrics require one byte-bound observation")
    metrics = tuple(
        native_vl_scene_prediction_metrics(record, prediction)
        for record, prediction in zip(records, generated, strict=True)
    )
    maps = tuple(
        {normalize_scene_label(item.label): item.bbox_qwen_xyxy for item in prediction.objects}
        for prediction in generated
    )
    pair_pass = (
        all(cast(int, item["expected_object_count"]) >= 2 for item in metrics)
        and all(bool(item["schema_valid"]) for item in metrics)
        and all(bool(item["label_set_exact"]) for item in metrics)
        and all(bool(item["order_exact"]) for item in metrics)
        and all(
            all(
                row.get("center_selective") is True
                for row in cast(list[Mapping[str, object]], item["objects"])
            )
            for item in metrics
        )
        and maps[0] == maps[1]
    )
    return {
        "label_box_map_exact": maps[0] == maps[1],
        "pair_pass": pair_pass,
        "variants": list(metrics),
    }


def native_vl_scene_bank_summary(
    pairs: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Aggregate strict pair and per-identity metrics from report-ready rows."""

    if not isinstance(pairs, Sequence) or isinstance(pairs, str | bytes) or not pairs:
        raise ContractError("native VL scene summary requires nonempty pair rows")
    pair_pass_count = 0
    schema_valid_count = 0
    generation_count = 0
    center_selective_count = 0
    generated_center_hit_count = 0
    object_prediction_count = 0
    unexpected_center_hit_count = 0
    identity_rows: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for pair in pairs:
        if not isinstance(pair, Mapping):
            raise ContractError("native VL scene summary pair is malformed")
        metrics = pair.get("pair_metrics")
        if not isinstance(metrics, Mapping):
            raise ContractError("native VL scene summary omits pair metrics")
        pair_pass_count += int(metrics.get("pair_pass") is True)
        variants = metrics.get("variants")
        if not isinstance(variants, list) or len(variants) != 2:
            raise ContractError("native VL scene summary requires two variants per pair")
        for variant in variants:
            if not isinstance(variant, Mapping):
                raise ContractError("native VL scene summary variant is malformed")
            generation_count += 1
            schema_valid_count += int(variant.get("schema_valid") is True)
            objects = variant.get("objects")
            if not isinstance(objects, list):
                raise ContractError("native VL scene summary object rows are malformed")
            for row in objects:
                if not isinstance(row, Mapping) or not isinstance(row.get("identity_key"), str):
                    raise ContractError("native VL scene summary object row is malformed")
                unexpected = row.get("unexpected_center_hit_labels")
                if (
                    not isinstance(unexpected, list)
                    or any(not isinstance(value, str) or not value for value in unexpected)
                    or len(unexpected) != len(set(unexpected))
                ):
                    raise ContractError(
                        "native VL scene summary unexpected-center rows are invalid"
                    )
                generated_center_hit = row.get("generated_center_hit")
                if not isinstance(generated_center_hit, bool):
                    raise ContractError("native VL scene summary generated-center flag is invalid")
                if row.get("center_selective") is not (
                    row.get("target_center_hit") is True and generated_center_hit and not unexpected
                ):
                    raise ContractError("native VL scene summary center-selective flag changed")
                object_prediction_count += 1
                center_selective_count += int(row.get("center_selective") is True)
                generated_center_hit_count += int(generated_center_hit)
                unexpected_center_hit_count += len(unexpected)
                identity_rows[str(row["identity_key"])].append(row)
    per_identity = {}
    for identity_key, rows in sorted(identity_rows.items()):
        ious = []
        for row in rows:
            value = row.get("target_iou")
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ContractError("native VL scene summary IoU is invalid")
            ious.append(float(value))
        per_identity[identity_key] = {
            "center_selective_count": sum(row.get("center_selective") is True for row in rows),
            "center_hit_count": sum(row.get("target_center_hit") is True for row in rows),
            "expected_count": len(rows),
            "generated_center_hit_count": sum(
                row.get("generated_center_hit") is True for row in rows
            ),
            "label_found_count": sum(row.get("label_found") is True for row in rows),
            "mean_iou": math.fsum(ious) / len(ious),
            "unexpected_center_hit_count": sum(
                len(cast(list[str], row["unexpected_center_hit_labels"])) for row in rows
            ),
        }
    return {
        "center_selective_count": center_selective_count,
        "generation_count": generation_count,
        "generated_center_hit_count": generated_center_hit_count,
        "object_prediction_count": object_prediction_count,
        "pair_count": len(pairs),
        "pair_pass_count": pair_pass_count,
        "per_identity": per_identity,
        "schema_valid_count": schema_valid_count,
        "unexpected_center_hit_count": unexpected_center_hit_count,
    }

from __future__ import annotations

import numpy as np
import pytest

from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.data.calvin_qwen_grounding import (
    CALVIN_QWEN_SCENE_IDENTITY_ORDER,
    CalvinQwenSceneGroundingRecord,
    CalvinQwenSceneObject,
    qwen_grounding_label,
)
from picf_next.lingbot_native.native_vl_scene_metrics import (
    native_vl_scene_bank_summary,
    native_vl_scene_order_pair_metrics,
    native_vl_scene_prediction_metrics,
)
from picf_next.lingbot_native.vl_cotraining import (
    NativeVLGeneratedSceneGrounding,
    NativeVLGeneratedSceneObject,
)


def _record(
    *, reverse: bool = False, empty: bool = False, single: bool = False
) -> CalvinQwenSceneGroundingRecord:
    image = np.zeros((200, 200, 3), dtype=np.uint8)
    image.setflags(write=False)
    by_identity = {
        "movable/block_blue": CalvinQwenSceneObject(
            identity_key="movable/block_blue",
            bbox_xyxy=(10, 20, 30, 40),
            visible_owner_pixels=400,
            projected_target_mass=1.0,
            positive_visual_token_count=2,
        ),
        "part/table/button_link": CalvinQwenSceneObject(
            identity_key="part/table/button_link",
            bbox_xyxy=(140, 150, 180, 190),
            visible_owner_pixels=1_600,
            projected_target_mass=2.0,
            positive_visual_token_count=4,
        ),
    }
    order = (
        tuple(reversed(CALVIN_QWEN_SCENE_IDENTITY_ORDER))
        if reverse
        else CALVIN_QWEN_SCENE_IDENTITY_ORDER
    )
    objects = () if empty else tuple(by_identity[key] for key in order if key in by_identity)
    if single:
        objects = objects[:1]
    visible_keys = {item.identity_key for item in objects}
    return CalvinQwenSceneGroundingRecord(
        global_index=5,
        camera_name="static",
        host_image_key="observation.images.image",
        category_identity_order=order,
        objects=objects,
        subpatch_objects=(),
        absent_identity_keys=tuple(key for key in order if key not in visible_keys),
        minimum_projected_target_mass=0.25,
        visual_lattice=8,
        image_grid_thw=(1, 16, 16),
        patch_size=16,
        merge_size=2,
        image=image,
        source_rgb_sha256=source_array_sha256("rgb_static", image),
    )


def _generated(
    record: CalvinQwenSceneGroundingRecord,
    *,
    coarse_blue_box: bool = False,
    full_image_boxes: bool = False,
    swap_boxes: bool = False,
) -> NativeVLGeneratedSceneGrounding:
    boxes = [record.qwen_bbox_for_object(item) for item in record.objects]
    if swap_boxes:
        boxes.reverse()
    if full_image_boxes:
        boxes = [(0, 0, 1000, 1000) for _item in record.objects]
    if coarse_blue_box:
        boxes = [
            (0, 0, 400, 400) if item.identity_key == "movable/block_blue" else box
            for item, box in zip(record.objects, boxes, strict=True)
        ]
    return NativeVLGeneratedSceneGrounding(
        objects=tuple(
            NativeVLGeneratedSceneObject(
                label=qwen_grounding_label(item.identity_key),
                bbox_qwen_xyxy=box,
            )
            for item, box in zip(record.objects, boxes, strict=True)
        ),
        schema_valid=True,
    )


def test_scene_metrics_match_by_label_and_require_order_equivariance() -> None:
    records = (_record(), _record(reverse=True))
    generated = (_generated(records[0]), _generated(records[1]))
    metrics = native_vl_scene_order_pair_metrics(records, generated)

    assert metrics["pair_pass"] is True
    assert metrics["label_box_map_exact"] is True
    for variant in metrics["variants"]:
        assert variant["schema_valid"] is True
        assert variant["label_set_exact"] is True
        assert variant["order_exact"] is True
        assert all(item["center_selective"] for item in variant["objects"])
        assert all(item["generated_center_hit"] for item in variant["objects"])
        assert all(item["target_center_hit"] for item in variant["objects"])
        assert all(item["target_iou"] == pytest.approx(1.0) for item in variant["objects"])
        assert all(not item["unexpected_center_hit_labels"] for item in variant["objects"])


def test_scene_metrics_expose_positional_box_shortcut_without_rematching_by_order() -> None:
    record = _record()
    metrics = native_vl_scene_prediction_metrics(record, _generated(record, swap_boxes=True))

    assert metrics["label_set_exact"] is True
    assert metrics["order_exact"] is True
    assert not any(item["target_center_hit"] for item in metrics["objects"])
    assert all(item["target_iou"] == 0.0 for item in metrics["objects"])


def test_scene_pair_rejects_order_stable_wrong_boxes_and_full_image_shortcut() -> None:
    records = (_record(), _record(reverse=True))

    wrong = native_vl_scene_order_pair_metrics(
        records,
        (_generated(records[0], swap_boxes=True), _generated(records[1], swap_boxes=True)),
    )
    assert wrong["label_box_map_exact"] is True
    assert wrong["pair_pass"] is False
    assert not any(
        row["center_selective"] for variant in wrong["variants"] for row in variant["objects"]
    )

    full_image = native_vl_scene_order_pair_metrics(
        records,
        (
            _generated(records[0], full_image_boxes=True),
            _generated(records[1], full_image_boxes=True),
        ),
    )
    assert full_image["label_box_map_exact"] is True
    assert full_image["pair_pass"] is False
    assert all(
        row["target_center_hit"] for variant in full_image["variants"] for row in variant["objects"]
    )
    assert not any(
        row["generated_center_hit"]
        for variant in full_image["variants"]
        for row in variant["objects"]
    )
    assert all(
        row["unexpected_center_hit_labels"]
        for variant in full_image["variants"]
        for row in variant["objects"]
    )

    coarse = native_vl_scene_order_pair_metrics(
        records,
        (
            _generated(records[0], coarse_blue_box=True),
            _generated(records[1], coarse_blue_box=True),
        ),
    )
    blue_rows = [
        row
        for variant in coarse["variants"]
        for row in variant["objects"]
        if row["identity_key"] == "movable/block_blue"
    ]
    assert coarse["label_box_map_exact"] is True
    assert coarse["pair_pass"] is False
    assert all(row["target_center_hit"] for row in blue_rows)
    assert not any(row["generated_center_hit"] for row in blue_rows)
    assert not any(row["unexpected_center_hit_labels"] for row in blue_rows)


def test_scene_metrics_reject_empty_and_single_object_pairs() -> None:
    empty_records = (_record(empty=True), _record(reverse=True, empty=True))
    empty_generation = NativeVLGeneratedSceneGrounding(objects=(), schema_valid=True)
    empty_metrics = native_vl_scene_order_pair_metrics(
        empty_records,
        (empty_generation, empty_generation),
    )
    assert empty_metrics["pair_pass"] is False

    single_records = (_record(single=True), _record(reverse=True, single=True))
    single_metrics = native_vl_scene_order_pair_metrics(
        single_records,
        (_generated(single_records[0]), _generated(single_records[1])),
    )
    assert single_metrics["pair_pass"] is False


def test_scene_metrics_summarize_bidirectional_center_evidence() -> None:

    records = (_record(), _record(reverse=True))
    pair_metrics = native_vl_scene_order_pair_metrics(
        records,
        (_generated(records[0]), _generated(records[1])),
    )
    summary = native_vl_scene_bank_summary([{"pair_metrics": pair_metrics}])
    assert summary["pair_count"] == 1
    assert summary["pair_pass_count"] == 1
    assert summary["generation_count"] == 2
    assert summary["generated_center_hit_count"] == 4
    assert summary["schema_valid_count"] == 2
    assert summary["center_selective_count"] == 4
    assert summary["object_prediction_count"] == 4
    assert summary["unexpected_center_hit_count"] == 0
    assert summary["per_identity"]["movable/block_blue"] == {
        "center_selective_count": 2,
        "center_hit_count": 2,
        "expected_count": 2,
        "generated_center_hit_count": 2,
        "label_found_count": 2,
        "mean_iou": pytest.approx(1.0),
        "unexpected_center_hit_count": 0,
    }

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from PIL import Image

from picf_next.lingbot_native import visual_audit
from picf_next.lingbot_native.calvin_objective import NativeCALVINObjectiveResult
from picf_next.lingbot_native.supervision import (
    NativeSequencePredictions,
    NativeSequenceTargets,
    SequenceAssignment,
)
from picf_next.lingbot_native.visual_audit import _atomic_png, render_native_relation_visuals
from picf_next.objective import ObjectiveTerm, UnifiedObjective
from tools.run_lingbot_vla2_native_full import _validate_full_visual_artifact


def _objective(
    *,
    target_track_count: int = 2,
    prediction_dtype: torch.dtype = torch.float32,
) -> NativeCALVINObjectiveResult:
    if target_track_count < 2:
        raise ValueError("test objective needs two valid target tracks")
    support = torch.full((1, 1, 8, 2), -5.0, dtype=prediction_dtype)
    support[0, 0, 1, 0] = 5.0
    support[0, 0, 4, 1] = 5.0
    ownership = torch.zeros(1, 1, 8, 3, dtype=prediction_dtype)
    ownership[..., -1] = 1.0
    ownership[0, 0, 1] = torch.tensor([0.9, 0.05, 0.05], dtype=prediction_dtype)
    ownership[0, 0, 4] = torch.tensor([0.05, 0.9, 0.05], dtype=prediction_dtype)
    predictions = NativeSequencePredictions(
        support_logits=support,
        ownership=ownership,
        existence_logits=torch.tensor([[[2.0, 1.0]]], dtype=prediction_dtype),
        task_relevance_logits=torch.tensor([[3.0, -2.0]], dtype=prediction_dtype),
        dense_task_grounding_logits=torch.zeros(1, 1, 8, dtype=prediction_dtype),
    )
    masks = torch.zeros(1, 1, target_track_count, 8)
    masks[0, 0, 0, 1] = 1.0
    masks[0, 0, 1, 4] = 1.0
    mask_valid = torch.zeros_like(masks, dtype=torch.bool)
    mask_valid[:, :, :2, [1, 4]] = True
    track_valid = torch.zeros(1, target_track_count, dtype=torch.bool)
    track_valid[:, :2] = True
    targets = NativeSequenceTargets(
        masks=masks,
        mask_valid=mask_valid,
        existence=track_valid[:, None].float(),
        existence_valid=track_valid[:, None].clone(),
        task_relevance=torch.nn.functional.pad(
            torch.tensor([[1.0, 0.0]]),
            (0, target_track_count - 2),
        ),
        task_valid=track_valid.clone(),
        track_valid=track_valid,
        capacity_censored=torch.zeros(1, target_track_count, dtype=torch.bool),
        token_observed_fraction=torch.tensor([[[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]]),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    zero = torch.tensor(0.0)
    return NativeCALVINObjectiveResult(
        objective=UnifiedObjective(total=zero, normalized_terms={}, valid_counts={}),
        predictions=predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.tensor([[0, 1]])),
        track_identity_keys_by_batch=(("block/blue", "button/black"),),
        row_bindings_by_batch=((("block/blue", 0), ("button/black", 1)),),
        predictive_terms=(),
        structural_terms=(),
    )


def _no_object_objective() -> NativeCALVINObjectiveResult:
    support = torch.full((1, 1, 8, 2), -5.0)
    ownership = torch.zeros(1, 1, 8, 3)
    ownership[..., -1] = 1.0
    predictions = NativeSequencePredictions(
        support_logits=support,
        ownership=ownership,
        existence_logits=torch.full((1, 1, 2), -5.0),
        task_relevance_logits=torch.full((1, 2), -5.0),
        dense_task_grounding_logits=torch.full((1, 1, 8), -5.0),
    )
    targets = NativeSequenceTargets(
        masks=torch.zeros(1, 1, 1, 8),
        mask_valid=torch.zeros(1, 1, 1, 8, dtype=torch.bool),
        existence=torch.zeros(1, 1, 1),
        existence_valid=torch.zeros(1, 1, 1, dtype=torch.bool),
        task_relevance=torch.zeros(1, 1),
        task_valid=torch.zeros(1, 1, dtype=torch.bool),
        track_valid=torch.zeros(1, 1, dtype=torch.bool),
        capacity_censored=torch.zeros(1, 1, dtype=torch.bool),
        token_observed_fraction=torch.ones(1, 1, 8),
        inventory_exhaustive=torch.ones(1, 1, dtype=torch.bool),
        exclusive_ownership=True,
    )
    zero = torch.tensor(0.0)
    return NativeCALVINObjectiveResult(
        objective=UnifiedObjective(total=zero, normalized_terms={}, valid_counts={}),
        predictions=predictions,
        targets=targets,
        assignment=SequenceAssignment(torch.full((1, 2), -1, dtype=torch.long)),
        track_identity_keys_by_batch=((),),
        row_bindings_by_batch=((),),
        predictive_terms=(),
        structural_terms=(),
    )


def _factorized_objective() -> NativeCALVINObjectiveResult:
    objective = _objective()
    predictions = objective.predictions
    task_row_probability = predictions.task_relevance_logits.sigmoid()
    task_row_probability_by_time = task_row_probability[:, None]
    task_object_probability = (
        task_row_probability_by_time[:, :, None] * predictions.ownership[..., :-1]
    )
    floor = torch.full_like(task_object_probability, torch.finfo(task_object_probability.dtype).min)
    task_object_log_probability = torch.where(
        task_object_probability > 0,
        task_object_probability.log(),
        floor,
    )
    task_event_distribution = torch.cat(
        (
            task_object_probability,
            1 - task_object_probability.sum(dim=-1, keepdim=True),
        ),
        dim=-1,
    )
    factorized_predictions = replace(
        predictions,
        task_object_log_probability=task_object_log_probability,
        task_object_probability=task_object_probability,
        task_event_distribution=task_event_distribution,
        task_row_probability=task_row_probability,
        task_row_probability_by_time=task_row_probability_by_time,
    )
    marker = ObjectiveTerm(
        name="set/task_row",
        values=torch.zeros(1),
        valid=torch.ones(1, dtype=torch.bool),
        weight=1.0,
    )
    return replace(
        objective,
        predictions=factorized_predictions,
        structural_terms=(marker,),
    )


def _host_item() -> dict[str, object]:
    static = torch.zeros(3, 18, 24)
    static[2] = 180
    gripper = torch.zeros(3, 12, 16)
    gripper[0] = 160
    return {
        "observation.images.camera_top": static,
        "observation.images.camera_wrist_left": gripper,
        "task": "move the blue block to the black button",
    }


def _model_inputs() -> dict[str, torch.Tensor]:
    return {
        "image_grid_thw": torch.tensor(
            [[[1, 2, 2], [1, 2, 2], [1, 2, 2]]],
            dtype=torch.long,
        ),
        "img_masks": torch.tensor([[True, True, False]]),
    }


def test_native_relation_visual_is_task_named_hash_bound_and_legible(tmp_path: Path) -> None:
    artifacts = render_native_relation_visuals(
        output_root=tmp_path,
        global_step=20,
        rank=1,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        objective=_objective(prediction_dtype=torch.bfloat16),
        structural_sensor_valid=torch.tensor(
            [[False, True, False, False, True, False, False, False]]
        ),
        sample_keys=("episode/3/20",),
        merge_size=2,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    path = tmp_path / artifact["path"]
    assert path.is_file() and artifact["bytes"] == path.stat().st_size
    assert len(artifact["sha256"]) == 64
    assert "task_move-the-blue-block-to-the-black-button" in path.name
    assert artifact["identity_keys"] == ["block/blue", "button/black"]
    assert artifact["source_time"] == 0
    assert artifact["source_side"] == "posterior"
    assert artifact["source_phase"] == 1
    assert artifact["binding_start_phase"] == [1, 1]
    assert artifact["source_binding_valid"] == [True, True]
    assert artifact["row_to_track"] == [0, 1]
    assert artifact["sequence_row_to_track"] == [0, 1]
    assert min(artifact["row_matched_soft_iou"]) > 0.85
    assert artifact["anchor_surface"] == "ownership_or_support_times_task_relevance.max(row)"
    assert artifact["input_weight_global_step"] == 19
    assert artifact["weight_boundary"] == "pre_update_forward"
    assert artifact["loss_only_labels_visible_to_model"] is False
    assert [view["name"] for view in artifact["views"]] == ["static", "gripper"]
    _validate_full_visual_artifact(
        artifact,
        run_root=tmp_path,
        expected_step=20,
        expected_rank=1,
    )
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width == 5 * 24
        assert image.height > 18 + 12
    with pytest.raises(FileExistsError, match="already exists"):
        render_native_relation_visuals(
            output_root=tmp_path,
            global_step=20,
            rank=1,
            host_items=(_host_item(),),
            model_inputs=_model_inputs(),
            objective=_objective(),
            structural_sensor_valid=torch.tensor(
                [[False, True, False, False, True, False, False, False]]
            ),
            sample_keys=("episode/3/20",),
            merge_size=2,
        )


def test_native_relation_visual_supports_released_checkpoint_evaluation(
    tmp_path: Path,
) -> None:
    artifacts = render_native_relation_visuals(
        output_root=tmp_path,
        global_step=0,
        input_weight_global_step=0,
        weight_boundary="checkpoint_evaluation",
        rank=0,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        objective=_objective(),
        structural_sensor_valid=torch.tensor(
            [[False, True, False, False, True, False, False, False]]
        ),
        sample_keys=("episode/3/0",),
        merge_size=2,
    )

    assert artifacts[0]["global_step"] == 0
    assert artifacts[0]["input_weight_global_step"] == 0
    assert artifacts[0]["weight_boundary"] == "checkpoint_evaluation"
    assert (tmp_path / artifacts[0]["path"]).is_file()

    with pytest.raises(ValueError, match="explicit weight boundary"):
        render_native_relation_visuals(
            output_root=tmp_path / "missing-boundary",
            global_step=0,
            rank=0,
            host_items=(_host_item(),),
            model_inputs=_model_inputs(),
            objective=_objective(),
            structural_sensor_valid=torch.tensor(
                [[False, True, False, False, True, False, False, False]]
            ),
            sample_keys=("episode/3/0",),
            merge_size=2,
        )


def test_exclusive_task_anchor_uses_categorical_ownership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[torch.Tensor] = []

    def capture_heat(rgb: object, heat: object) -> object:
        observed.append(torch.as_tensor(heat).clone())
        return rgb

    monkeypatch.setattr(visual_audit, "_heat_overlay", capture_heat)
    objective = _objective()
    render_native_relation_visuals(
        output_root=tmp_path,
        global_step=20,
        rank=0,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        objective=objective,
        structural_sensor_valid=torch.tensor(
            [[False, True, False, False, True, False, False, False]]
        ),
        sample_keys=("episode/3/20",),
        merge_size=2,
    )

    relevance = objective.predictions.task_relevance_logits[0].sigmoid()
    expected = torch.stack(
        (
            (objective.predictions.ownership[0, 0, 1, :-1] * relevance).max(),
            (objective.predictions.ownership[0, 0, 4, :-1] * relevance).max(),
        )
    )
    actual = torch.stack((observed[0].flatten()[0], observed[2].flatten()[0]))
    torch.testing.assert_close(actual, expected)
    expected_dense = objective.predictions.dense_task_grounding_logits[
        0,
        0,
        torch.tensor([1, 4]),
    ].sigmoid()
    actual_dense = torch.stack((observed[1].flatten()[0], observed[3].flatten()[0]))
    torch.testing.assert_close(actual_dense, expected_dense)


def test_factorized_visual_uses_exact_task_object_probability_surface(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[torch.Tensor] = []

    def capture_heat(rgb: object, heat: object) -> object:
        observed.append(torch.as_tensor(heat).clone())
        return rgb

    monkeypatch.setattr(visual_audit, "_heat_overlay", capture_heat)
    objective = _factorized_objective()
    artifacts = render_native_relation_visuals(
        output_root=tmp_path,
        global_step=21,
        rank=0,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        objective=objective,
        structural_sensor_valid=torch.tensor(
            [[False, True, False, False, True, False, False, False]]
        ),
        sample_keys=("episode/3/21",),
        merge_size=2,
    )

    assert artifacts[0]["anchor_surface"] == "task_object_probability.max(row)"
    expected = objective.predictions.task_object_probability
    assert expected is not None
    expected_anchor = expected[0, 0, torch.tensor([1, 4])].max(dim=-1).values
    actual_anchor = torch.stack((observed[0].flatten()[0], observed[2].flatten()[0]))
    torch.testing.assert_close(actual_anchor, expected_anchor)


def test_native_relation_visual_uses_explicit_structural_addresses(tmp_path: Path) -> None:
    with torch.no_grad():
        invalid = torch.zeros(1, 8, dtype=torch.bool)
    try:
        render_native_relation_visuals(
            output_root=tmp_path,
            global_step=1,
            rank=0,
            host_items=(_host_item(),),
            model_inputs=_model_inputs(),
            objective=_objective(),
            structural_sensor_valid=invalid,
            sample_keys=("episode/3/1",),
            merge_size=2,
        )
    except ValueError as error:
        assert "partition ended" in str(error)
    else:
        raise AssertionError("visual audit accepted an empty structural address map")


def test_native_relation_visual_handles_padded_and_empty_target_inventories(
    tmp_path: Path,
) -> None:
    common = {
        "output_root": tmp_path,
        "rank": 0,
        "host_items": (_host_item(),),
        "model_inputs": _model_inputs(),
        "structural_sensor_valid": torch.tensor(
            [[False, True, False, False, True, False, False, False]]
        ),
        "merge_size": 2,
    }
    padded = render_native_relation_visuals(
        **common,
        global_step=30,
        objective=_objective(target_track_count=3),
        sample_keys=("episode/3/30",),
    )
    empty = render_native_relation_visuals(
        **common,
        global_step=31,
        objective=_no_object_objective(),
        sample_keys=("episode/3/31",),
    )

    assert (tmp_path / padded[0]["path"]).is_file()
    assert empty[0]["identity_keys"] == []
    assert empty[0]["row_to_track"] == [-1, -1]
    with Image.open(tmp_path / empty[0]["path"]) as image:
        assert image.format == "PNG"
        assert image.width == 5 * 24

    misaligned = replace(
        _objective(),
        track_identity_keys_by_batch=(("block/blue",),),
    )
    with pytest.raises(ValueError, match="identities differ"):
        render_native_relation_visuals(
            **common,
            global_step=32,
            objective=misaligned,
            sample_keys=("episode/3/32",),
        )


def test_native_visual_publication_cleans_failed_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "visual.png"

    def fail_save(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected PNG failure")

    monkeypatch.setattr(Image.Image, "save", fail_save)
    with pytest.raises(OSError, match="injected PNG failure"):
        _atomic_png(Image.new("RGB", (4, 4)), output)

    assert not output.exists()
    assert not tuple(tmp_path.glob(".*.tmp-*"))


def test_native_visual_requires_official_three_slot_padded_geometry(tmp_path: Path) -> None:
    common = {
        "output_root": tmp_path,
        "global_step": 1,
        "rank": 0,
        "host_items": (_host_item(),),
        "objective": _objective(),
        "structural_sensor_valid": torch.tensor(
            [[False, True, False, False, True, False, False, False]]
        ),
        "sample_keys": ("episode/3/1",),
        "merge_size": 2,
    }
    with pytest.raises(ValueError, match="view geometry"):
        render_native_relation_visuals(
            **common,
            model_inputs={
                "image_grid_thw": torch.tensor([[[1, 2, 2], [1, 2, 2]]]),
                "img_masks": torch.ones(1, 2, dtype=torch.bool),
            },
        )

    wrong_padded_grid = _model_inputs()
    wrong_padded_grid["image_grid_thw"] = wrong_padded_grid["image_grid_thw"].clone()
    wrong_padded_grid["image_grid_thw"][0, 2] = torch.tensor([1, 4, 4])
    with pytest.raises(ValueError, match="padded Qwen grid"):
        render_native_relation_visuals(
            **common,
            model_inputs=wrong_padded_grid,
        )


def test_factorized_visual_rejects_missing_exact_task_object_field(tmp_path: Path) -> None:
    objective = replace(
        _objective(),
        structural_terms=(
            ObjectiveTerm(
                name="set/task_row",
                values=torch.zeros(1),
                valid=torch.ones(1, dtype=torch.bool),
                weight=1.0,
            ),
        ),
    )

    with pytest.raises(RuntimeError, match="exact task-object"):
        render_native_relation_visuals(
            output_root=tmp_path,
            global_step=1,
            rank=0,
            host_items=(_host_item(),),
            model_inputs=_model_inputs(),
            objective=objective,
            structural_sensor_valid=torch.tensor(
                [[False, True, False, False, True, False, False, False]]
            ),
            sample_keys=("episode/3/20",),
            merge_size=2,
        )

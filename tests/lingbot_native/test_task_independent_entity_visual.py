from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
from PIL import Image

from picf_next.lingbot_native.calvin_entity_set import PhysicalCALVINFrameTargetBundle
from picf_next.lingbot_native.entity_set_objective import (
    PhysicalFrameAssignment,
    PhysicalFrameTargets,
    PhysicalSetLoss,
)
from picf_next.lingbot_native.modalities import CALVIN_VIDEOMT_MASK_LAYOUT
from picf_next.lingbot_native.physical_relations import (
    PhysicalRelationOutput,
    PhysicalRelationSurfaceOutput,
)
from picf_next.lingbot_native.visual_audit import (
    TASK_INDEPENDENT_ENTITY_VISUAL_SCHEMA,
    _target_colors_in_row_gauge,
    render_task_independent_entity_visuals,
)


def test_loss_only_target_colors_follow_assigned_row_gauge() -> None:
    row_colors = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]).numpy()
    colors = _target_colors_in_row_gauge(
        (1, -1, 0),
        track_count=3,
        row_colors=row_colors,
    )

    assert colors.tolist() == [
        [70.0, 80.0, 90.0],
        [10.0, 20.0, 30.0],
        [160.0, 160.0, 160.0],
    ]


def _relation() -> PhysicalRelationOutput:
    support = torch.full((1, 8, 2), -5.0)
    support[0, 1, 0] = 5.0
    support[0, 4, 1] = 5.0
    ownership = torch.zeros(1, 8, 3)
    ownership[..., -1] = 1.0
    ownership[0, 1] = torch.tensor([0.9, 0.05, 0.05])
    ownership[0, 4] = torch.tensor([0.05, 0.9, 0.05])
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    existence_logits = torch.tensor([[2.0, 1.0]])
    return PhysicalRelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership,
        ownership_log_probability=ownership.clamp_min(1e-6).log(),
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        row_embeddings=torch.nn.functional.normalize(torch.ones(1, 2, 4), dim=-1),
        relation_temperature=torch.tensor(0.07),
        sensor_valid=sensor_valid,
        structural_sensor_valid=sensor_valid,
    )


def _targets() -> PhysicalCALVINFrameTargetBundle:
    masks = torch.zeros(1, 2, 8)
    masks[0, 0, 1] = 1.0
    masks[0, 1, 4] = 1.0
    mask_valid = torch.zeros_like(masks, dtype=torch.bool)
    mask_valid[:, :, [1, 4]] = True
    return PhysicalCALVINFrameTargetBundle(
        targets=PhysicalFrameTargets(
            masks=masks,
            mask_valid=mask_valid,
            existence=torch.ones(1, 2),
            existence_valid=torch.ones(1, 2, dtype=torch.bool),
            track_valid=torch.ones(1, 2, dtype=torch.bool),
            capacity_censored=torch.zeros(1, 2, dtype=torch.bool),
            token_observed_fraction=torch.tensor([[0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]),
            inventory_exhaustive=torch.ones(1, dtype=torch.bool),
            exclusive_ownership=True,
        ),
        identity_keys_by_batch=(("block/blue", "button/black"),),
    )


def _relation_with_native_surface() -> PhysicalRelationOutput:
    relation = _relation()
    tokens = 2 * 24 * 24
    support = torch.full((1, tokens, 2), -5.0)
    support[0, : 24 * 24, 0] = 5.0
    support[0, 24 * 24 :, 1] = 5.0
    ownership = torch.full((1, tokens, 3), 0.05)
    ownership[0, : 24 * 24, 0] = 0.9
    ownership[0, 24 * 24 :, 1] = 0.9
    surface = PhysicalRelationSurfaceOutput(
        name="vjepa",
        geometry_kind="image_grid",
        target_kind="calvin_vjepa21_visible_owner_v1",
        layout="vjepa21.calvin.static-gripper.24x24.v1",
        support_logits=support,
        ownership=ownership,
        ownership_log_probability=ownership.log(),
        sensor_valid=torch.ones(1, tokens, dtype=torch.bool),
        canonical_token_ids=torch.arange(tokens).unsqueeze(0),
    )
    return replace(relation, relation_surfaces=(surface,))


def _targets_with_native_surface() -> PhysicalCALVINFrameTargetBundle:
    bundle = _targets()
    targets = bundle.targets
    native_masks = torch.zeros(1, 2, 2 * 24 * 24)
    native_masks[0, 0, : 24 * 24] = 1.0
    native_masks[0, 1, 24 * 24 :] = 1.0
    return PhysicalCALVINFrameTargetBundle(
        targets=replace(
            targets,
            masks=torch.cat((targets.masks, native_masks), dim=-1),
            mask_valid=torch.cat(
                (
                    targets.mask_valid,
                    torch.ones_like(native_masks, dtype=torch.bool),
                ),
                dim=-1,
            ),
            token_observed_fraction=torch.cat(
                (
                    targets.token_observed_fraction,
                    torch.ones(1, 2 * 24 * 24),
                ),
                dim=-1,
            ),
        ),
        identity_keys_by_batch=bundle.identity_keys_by_batch,
    )


def _relation_with_videomt_surface() -> PhysicalRelationOutput:
    relation = _relation()
    tokens = 120 * 120
    support = torch.full((1, tokens, 2), -5.0)
    support[0, : tokens // 2, 0] = 5.0
    support[0, tokens // 2 :, 1] = 5.0
    ownership = torch.full((1, tokens, 3), 0.05)
    ownership[0, : tokens // 2, 0] = 0.9
    ownership[0, tokens // 2 :, 1] = 0.9
    donor_query = torch.zeros(1, tokens, 2)
    donor_query[0, : tokens // 2, 0] = 1.0
    donor_query[0, tokens // 2 :, 1] = 1.0
    query_ownership = torch.tensor([[[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]]])
    surface = PhysicalRelationSurfaceOutput(
        name="videomt_masks",
        geometry_kind="image_grid",
        target_kind="calvin_videomt_visible_owner_v1",
        layout=CALVIN_VIDEOMT_MASK_LAYOUT,
        support_logits=support,
        ownership=ownership,
        ownership_log_probability=ownership.log(),
        sensor_valid=torch.ones(1, tokens, dtype=torch.bool),
        grid_shape=(120, 120),
        donor_query_probability=donor_query,
        donor_context_probability=torch.zeros(1, tokens),
        contextual_query_ownership=query_ownership,
        query_valid=torch.ones(1, 2, dtype=torch.bool),
        canonical_query_ids=torch.arange(2).unsqueeze(0),
    )
    return replace(relation, relation_surfaces=(surface,))


def _targets_with_videomt_surface() -> PhysicalCALVINFrameTargetBundle:
    bundle = _targets()
    targets = bundle.targets
    tokens = 120 * 120
    native_masks = torch.zeros(1, 2, tokens)
    native_masks[0, 0, : tokens // 2] = 1.0
    native_masks[0, 1, tokens // 2 :] = 1.0
    return PhysicalCALVINFrameTargetBundle(
        targets=replace(
            targets,
            masks=torch.cat((targets.masks, native_masks), dim=-1),
            mask_valid=torch.cat(
                (targets.mask_valid, torch.ones_like(native_masks, dtype=torch.bool)),
                dim=-1,
            ),
            token_observed_fraction=torch.cat(
                (targets.token_observed_fraction, torch.ones(1, tokens)),
                dim=-1,
            ),
        ),
        identity_keys_by_batch=bundle.identity_keys_by_batch,
    )


def _set_loss() -> PhysicalSetLoss:
    zero = torch.tensor(0.0)
    return PhysicalSetLoss(
        total=zero,
        mask_focal=zero,
        mask_dice=zero,
        existence_focal=zero,
        ownership_nll=zero,
        assignment=PhysicalFrameAssignment(torch.tensor([[0, 1]])),
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


def test_task_independent_entity_visual_is_task_named_and_prompt_free(
    tmp_path: Path,
) -> None:
    artifacts = render_task_independent_entity_visuals(
        output_root=tmp_path,
        global_step=20,
        input_weight_global_step=19,
        rank=1,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        relation=_relation(),
        target_bundle=_targets(),
        set_loss=_set_loss(),
        sample_keys=("episode/3/20",),
        merge_size=2,
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["schema"] == TASK_INDEPENDENT_ENTITY_VISUAL_SCHEMA
    assert artifact["task_used_by_entity_objective"] is False
    assert artifact["loss_only_labels_visible_to_model"] is False
    assert artifact["row_to_track"] == [0, 1]
    assert min(artifact["row_matched_soft_iou"]) > 0.85
    assert artifact["anchor_surface"] == "task_independent_categorical_ownership"
    assert artifact["target_color_mode"] == "assigned_row_or_unassigned_gray_v1"
    assert [view["name"] for view in artifact["views"]] == ["static", "gripper"]
    path = tmp_path / artifact["path"]
    assert path.is_file() and len(artifact["sha256"]) == 64
    assert "task_move-the-blue-block-to-the-black-button" in path.name
    with Image.open(path) as image:
        assert image.format == "PNG"
        assert image.width == 4 * 24
        assert image.height > 18 + 12


def test_task_independent_entity_visual_accepts_explicit_step_zero_snapshot(
    tmp_path: Path,
) -> None:
    artifacts = render_task_independent_entity_visuals(
        output_root=tmp_path,
        global_step=0,
        input_weight_global_step=0,
        weight_boundary="fixed_checkpoint_evaluation",
        rank=0,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        relation=_relation(),
        target_bundle=_targets(),
        set_loss=_set_loss(),
        sample_keys=("episode/3/0",),
        merge_size=2,
    )

    assert artifacts[0]["global_step"] == 0
    assert artifacts[0]["input_weight_global_step"] == 0
    assert artifacts[0]["weight_boundary"] == "fixed_checkpoint_evaluation"
    assert (tmp_path / artifacts[0]["path"]).is_file()


def test_task_independent_visual_renders_vjepa_native_surface_separately(
    tmp_path: Path,
) -> None:
    artifacts = render_task_independent_entity_visuals(
        output_root=tmp_path,
        global_step=250,
        input_weight_global_step=249,
        rank=0,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        relation=_relation_with_native_surface(),
        target_bundle=_targets_with_native_surface(),
        set_loss=_set_loss(),
        sample_keys=("episode/3/250",),
        merge_size=2,
    )

    artifact = artifacts[0]
    assert artifact["anchor_surface"] == "shared_rows_qwen_plus_native_surfaces"
    assert min(artifact["row_matched_soft_iou"]) > 0.85
    assert artifact["relation_surfaces"] == [
        {
            "name": "vjepa",
            "view": "static",
            "layout": "vjepa21.calvin.static-gripper.24x24.v1",
            "native_grid": [24, 24],
            "token_count": 24 * 24,
            "available": True,
        },
        {
            "name": "vjepa",
            "view": "gripper",
            "layout": "vjepa21.calvin.static-gripper.24x24.v1",
            "native_grid": [24, 24],
            "token_count": 24 * 24,
            "available": True,
        },
    ]
    with Image.open(tmp_path / artifact["path"]) as image:
        assert image.width == 4 * 24
        assert image.height > 2 * (18 + 12)


def test_task_independent_visual_exposes_videomt_donor_and_host_composition(
    tmp_path: Path,
) -> None:
    artifacts = render_task_independent_entity_visuals(
        output_root=tmp_path,
        global_step=250,
        input_weight_global_step=249,
        rank=0,
        host_items=(_host_item(),),
        model_inputs=_model_inputs(),
        relation=_relation_with_videomt_surface(),
        target_bundle=_targets_with_videomt_surface(),
        set_loss=_set_loss(),
        sample_keys=("episode/3/250",),
        merge_size=2,
    )

    artifact = artifacts[0]
    surface = artifact["relation_surfaces"][0]
    assert surface["name"] == "videomt_masks"
    assert surface["view"] == "static"
    assert surface["native_grid"] == [120, 120]
    assert surface["token_count"] == 120 * 120
    assert [item["canonical_query_id"] for item in surface["donor_top_queries"]] == [0, 1]
    with Image.open(tmp_path / artifact["path"]) as image:
        assert image.width == 6 * 24
        assert image.height > 2 * (18 + 12)

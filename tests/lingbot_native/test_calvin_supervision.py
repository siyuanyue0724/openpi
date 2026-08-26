from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pytest
import torch

from picf_next.data.calvin_physical_supervision_schema import (
    CALVIN_OBJECT_GEOMETRY_CONTRACT,
)
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.calvin_target_request import NativeCALVINStructuralTargetRequest
from picf_next.lingbot_native.calvin import (
    CollatedNativeCALVINBatch,
    build_native_calvin_context,
)
from picf_next.lingbot_native.calvin_entity_set import (
    build_task_independent_calvin_targets,
    physical_frame_predictions_from_relation,
)
from picf_next.lingbot_native.calvin_objective import (
    NativeStructuralLossConfig,
    compose_native_calvin_objective,
)
from picf_next.lingbot_native.calvin_supervision import (
    build_native_calvin_sequence_target_bundle,
    stack_native_sequence_predictions,
)
from picf_next.lingbot_native.objective import NativeObjectiveConfig
from picf_next.lingbot_native.modalities import (
    CALVIN_VIDEOMT_MASK_LAYOUT,
    CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
    NativeObjectQuerySpatialRelation,
)
from picf_next.lingbot_native.physical_relations import (
    ContextualObjectQuerySpatialInput,
    PhysicalEntityReadout,
    PhysicalRelationOutput,
    PhysicalRelationSurfaceInput,
)
from picf_next.lingbot_native.relations import (
    HOST_NATIVE_MATCH_INTERFACE,
    RelationOutput,
    SharedRelationReadout,
)
from picf_next.lingbot_native.supervision import (
    TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
    TOKEN_MICRO_OWNERSHIP,
    NativeSequenceTargets,
)
from picf_next.lingbot_native.task_relation import (
    GLOBAL_MULTIPOSITIVE_TASK_RELATION,
    HOST_NATIVE_FACTORIZED_TASK_RELATION,
    HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION,
)
from picf_next.lingbot_native.training import run_native_policy_training_forward
from tests.lingbot_native.test_training_runtime import (
    _components,
    _controls,
    _model_inputs,
    _routing,
)

_HASHES = {
    "depth_gripper": "a" * 64,
    "depth_static": "b" * 64,
    "rgb_gripper": "c" * 64,
    "rgb_static": "d" * 64,
}


def _ownership_pair(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    log_probability = torch.log_softmax(logits.float(), dim=-1)
    return log_probability.exp().to(logits.dtype), log_probability


def _camera(
    name: str,
    owner: int,
    *,
    hash_override: str | None = None,
    unknown_bottom_right: bool = False,
) -> CalvinVisibleOwnerRaster:
    static = name == "static"
    shape = (200, 200) if static else (84, 84)
    host_key = "observation.images.image" if static else "observation.images.wrist_image"
    owner_index = np.full(shape, owner, dtype=np.uint8)
    supervised = np.ones(shape, dtype=np.bool_)
    if unknown_bottom_right:
        supervised[shape[0] // 2 :, shape[1] // 2 :] = False
    return CalvinVisibleOwnerRaster(
        camera_name=name,
        host_image_key=host_key,
        owner_index=owner_index,
        owner_supervised=supervised,
        source_rgb_sha256=hash_override or _HASHES[f"rgb_{name}"],
        source_depth_sha256=_HASHES[f"depth_{name}"],
        rgb_mae=0.0,
        depth_mae_m=0.0,
        depth_p95_m=0.0,
        depth_consistent_fraction=1.0,
    )


def _frame(
    identities: tuple[str, ...],
    *,
    static_owner: int = 1,
    gripper_owner: int = 2,
    hash_override: str | None = None,
) -> CalvinPhysicalSupervisionFrame:
    count = len(identities)
    return CalvinPhysicalSupervisionFrame(
        identity_keys=identities,
        geometry=torch.zeros(count, 3),
        geometry_variance=torch.zeros(count, 3),
        geometry_supervised=torch.ones(count, 3, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(
            _camera("static", static_owner, hash_override=hash_override),
            _camera("gripper", gripper_owner),
        ),
    )


class _Sidecar(CalvinPhysicalSupervisionSidecar):
    def __init__(self, frames: dict[int, CalvinPhysicalSupervisionFrame]) -> None:
        self.frames = frames

    def __call__(self, segment_index: int, global_index: int) -> CalvinPhysicalSupervisionFrame:
        assert segment_index == 3
        return self.frames[global_index]


def _request(index: int) -> NativeCALVINStructuralTargetRequest:
    return NativeCALVINStructuralTargetRequest(
        sample_key=f"episode/3/{index}",
        episode_key="episode/3",
        task_key="push_a",
        segment_index=3,
        source_global_index=index,
        source_sensor_sha256=tuple(sorted(_HASHES.items())),
    )


def _relation(*, task: tuple[float, float] = (0.0, 0.0)) -> RelationOutput:
    support = torch.zeros(1, 8, 2)
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    ownership, ownership_log_probability = _ownership_pair(torch.zeros(1, 8, 3))
    task_logits = torch.tensor([task])
    existence_logits = torch.zeros(1, 2)
    return RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership * sensor_valid.unsqueeze(-1),
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=torch.ones(1, 4),
        row_embeddings=torch.ones(1, 2, 4),
        relation_temperature=torch.ones(1),
        dense_task_grounding=torch.zeros(1, 8),
        dense_task_grounding_logits=torch.zeros(1, 8),
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
        ownership_log_probability=ownership_log_probability,
    )


def _physical_relation() -> PhysicalRelationOutput:
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    return PhysicalEntityReadout(4)(
        posterior_rows=torch.ones(1, 2, 4),
        sensor_hidden=torch.ones(1, 8, 4),
        sensor_valid=sensor_valid,
        structural_sensor_valid=sensor_valid,
    )


def _physical_relation_with_vjepa(*, available: bool = True) -> PhysicalRelationOutput:
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    surface_valid = torch.full((1, 2 * 24 * 24), available, dtype=torch.bool)
    canonical_ids = (
        torch.arange(surface_valid.shape[1]).unsqueeze(0)
        if available
        else torch.full(surface_valid.shape, -1, dtype=torch.long)
    )
    return PhysicalEntityReadout(4)(
        posterior_rows=torch.ones(1, 2, 4),
        sensor_hidden=torch.ones(1, 8, 4),
        sensor_valid=sensor_valid,
        structural_sensor_valid=sensor_valid,
        relation_surfaces=(
            PhysicalRelationSurfaceInput(
                name="vjepa",
                geometry_kind="image_grid",
                target_kind="calvin_vjepa21_visible_owner_v1",
                layout="vjepa21.calvin.static-gripper.24x24.v1",
                sensor_hidden=torch.ones(1, surface_valid.shape[1], 4),
                sensor_valid=surface_valid,
                canonical_token_ids=canonical_ids,
            ),
        ),
    )


def _physical_relation_with_videomt() -> PhysicalRelationOutput:
    query_valid = torch.ones(1, 2, dtype=torch.bool)
    relation = NativeObjectQuerySpatialRelation(
        name="videomt_masks",
        query_modality="videomt_queries",
        geometry_kind="image_grid",
        target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        layout=CALVIN_VIDEOMT_MASK_LAYOUT,
        object_logits=torch.ones(1, 2),
        mask_logits=torch.zeros(1, 2, 120 * 120),
        query_valid=query_valid,
        pixel_valid=torch.ones(1, 120 * 120, dtype=torch.bool),
        canonical_query_ids=torch.arange(2).unsqueeze(0),
        grid_shape=(120, 120),
    )
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    return PhysicalEntityReadout(4)(
        posterior_rows=torch.ones(1, 2, 4),
        sensor_hidden=torch.ones(1, 8, 4),
        sensor_valid=sensor_valid,
        structural_sensor_valid=sensor_valid,
        object_query_spatial_inputs=(
            ContextualObjectQuerySpatialInput(
                relation=relation,
                query_hidden=torch.ones(1, 2, 4),
            ),
        ),
    )


def _relation_with_object_rows(*, reversed_rows: bool) -> RelationOutput:
    support = torch.full((1, 8, 2), -8.0)
    support[0, 1, int(reversed_rows)] = 8.0
    support[0, 4, int(not reversed_rows)] = 8.0
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    ownership, ownership_log_probability = _ownership_pair(
        torch.cat((support, torch.zeros(1, 8, 1)), dim=-1)
    )
    task_logits = torch.zeros(1, 2)
    existence_logits = torch.full((1, 2), 8.0)
    return RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership * sensor_valid.unsqueeze(-1),
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=torch.ones(1, 4),
        row_embeddings=torch.ones(1, 2, 4),
        relation_temperature=torch.ones(1),
        dense_task_grounding=torch.zeros(1, 8),
        dense_task_grounding_logits=torch.zeros(1, 8),
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
        ownership_log_probability=ownership_log_probability,
    )


def _inputs(
    *,
    image_valid: tuple[bool, bool, bool] = (True, True, False),
    padded_grid: tuple[int, int, int] = (1, 2, 2),
) -> dict[str, torch.Tensor]:
    return {
        "images": torch.zeros(1, 3, 4, 3),
        "img_masks": torch.tensor([image_valid], dtype=torch.bool),
        "image_grid_thw": torch.tensor(
            [[[1, 2, 2], [1, 2, 2], list(padded_grid)]],
            dtype=torch.long,
        ),
    }


def _official_calvin_model_inputs() -> dict[str, torch.Tensor]:
    inputs = _model_inputs(1)
    inputs["images"] = torch.cat(
        (inputs["images"], torch.full_like(inputs["images"][:, :1], -1.0)),
        dim=1,
    )
    inputs["img_masks"] = torch.tensor([[True, True, False]])
    inputs["image_grid_thw"] = torch.cat(
        (inputs["image_grid_thw"], inputs["image_grid_thw"][:, :1]),
        dim=1,
    )
    return inputs


def test_native_calvin_targets_require_the_official_three_camera_slot_abi() -> None:
    common = {
        "requests_by_time": ((_request(10),),),
        "relations": (_relation(),),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
    }
    legacy_two_slot = _inputs()
    legacy_two_slot["images"] = legacy_two_slot["images"][:, :2]
    legacy_two_slot["img_masks"] = legacy_two_slot["img_masks"][:, :2]
    legacy_two_slot["image_grid_thw"] = legacy_two_slot["image_grid_thw"][:, :2]
    with pytest.raises(ValueError, match="views differ"):
        _targets(**common, model_inputs_by_time=(legacy_two_slot,))
    with pytest.raises(ValueError, match="validity differs"):
        _targets(
            **common,
            model_inputs_by_time=(_inputs(image_valid=(True, True, True)),),
        )
    with pytest.raises(ValueError, match="padded image grid differs"):
        _targets(
            **common,
            model_inputs_by_time=(_inputs(padded_grid=(1, 1, 4)),),
        )


def _targets(**kwargs: Any) -> NativeSequenceTargets:
    return build_native_calvin_sequence_target_bundle(**kwargs).targets


def test_native_calvin_targets_follow_view_order_and_stable_physical_identity() -> None:
    sidecar = _Sidecar(
        {
            10: _frame(("object/a", "object/b")),
            11: _frame(("object/b", "object/a")),
        }
    )
    relations = (_relation(task=(0.1, 0.2)), _relation(task=(0.3, 0.4)))
    targets = _targets(
        requests_by_time=((_request(10),), (_request(11),)),
        model_inputs_by_time=(_inputs(), _inputs()),
        relations=relations,
        physical_sidecar=sidecar,
        capacity=2,
        task_identity_resolver=lambda task: ("object/a",) if task == "push_a" else None,
        patch_size=1,
        merge_size=2,
    )

    assert targets.masks.shape == (1, 2, 2, 8)
    assert targets.masks[0, 0, 0, 1] == 1
    assert targets.masks[0, 0, 1, 4] == 1
    assert targets.masks[0, 1, 1, 1] == 1
    assert targets.masks[0, 1, 0, 4] == 1
    assert torch.equal(
        targets.token_observed_fraction,
        torch.tensor(
            [
                [
                    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ]
        ),
    )
    assert targets.mask_valid[0, :, :, [1, 4]].all()
    assert not targets.mask_valid[0, :, :, [0, 2, 3, 5, 6, 7]].any()
    assert targets.existence.bool().all()
    assert targets.existence_valid.all()
    assert torch.equal(targets.task_relevance, torch.tensor([[1.0, 0.0]]))
    assert targets.task_valid.all()

    predictions = stack_native_sequence_predictions(relations)
    assert predictions.support_logits.shape == (1, 2, 8, 2)
    assert torch.equal(predictions.task_relevance_logits, torch.tensor([[0.3, 0.4]]))

    bundle = build_native_calvin_sequence_target_bundle(
        requests_by_time=((_request(10),), (_request(11),)),
        model_inputs_by_time=(_inputs(), _inputs()),
        relations=relations,
        physical_sidecar=sidecar,
        capacity=2,
        task_identity_resolver=lambda task: ("object/a",) if task == "push_a" else None,
        patch_size=1,
        merge_size=2,
    )
    assert bundle.identity_keys_by_batch == (("object/a", "object/b"),)
    torch.testing.assert_close(bundle.targets.masks, targets.masks)


def test_task_independent_calvin_projection_is_invariant_to_task_key() -> None:
    relation = _physical_relation()
    request = _request(10)
    common = {
        "model_inputs_by_time": (_inputs(),),
        "relations": (relation,),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "patch_size": 1,
        "merge_size": 2,
    }
    first = build_task_independent_calvin_targets(
        **common,
        requests_by_time=((request,),),
    )[0]
    second = build_task_independent_calvin_targets(
        **common,
        requests_by_time=((replace(request, task_key="unseen_open_vocabulary_task"),),),
    )[0]

    assert first.identity_keys_by_batch == second.identity_keys_by_batch
    for field in (
        "masks",
        "mask_valid",
        "existence",
        "existence_valid",
        "track_valid",
        "capacity_censored",
        "token_observed_fraction",
        "inventory_exhaustive",
    ):
        torch.testing.assert_close(
            getattr(first.targets, field),
            getattr(second.targets, field),
        )


def test_vjepa_native_surface_extends_the_same_physical_target_and_row_gauge() -> None:
    relation = _physical_relation_with_vjepa()
    frame = build_task_independent_calvin_targets(
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        patch_size=1,
        merge_size=2,
    )[0]
    predictions = physical_frame_predictions_from_relation(relation)

    assert frame.targets.masks.shape == (1, 2, 8 + 2 * 24 * 24)
    assert predictions.support_logits.shape == (1, 8 + 2 * 24 * 24, 2)
    assert predictions.ownership_log_probability.shape == (1, 8 + 2 * 24 * 24, 3)
    static = slice(8, 8 + 24 * 24)
    gripper = slice(8 + 24 * 24, 8 + 2 * 24 * 24)
    assert frame.targets.mask_valid[..., static].all()
    assert frame.targets.mask_valid[..., gripper].all()
    assert torch.equal(frame.targets.masks[0, 0, static], torch.ones(24 * 24))
    assert not frame.targets.masks[0, 1, static].any()
    assert not frame.targets.masks[0, 0, gripper].any()
    assert torch.equal(frame.targets.masks[0, 1, gripper], torch.ones(24 * 24))
    measure = frame.targets.token_measure[0]
    torch.testing.assert_close(measure[1], torch.tensor(1.0))
    torch.testing.assert_close(measure[4], torch.tensor(1.0))
    torch.testing.assert_close(measure[static].sum(), torch.tensor(1.0))
    torch.testing.assert_close(measure[gripper].sum(), torch.tensor(1.0))
    torch.testing.assert_close(measure.sum(), torch.tensor(4.0))


def test_missing_vjepa_surface_preserves_shape_but_adds_no_supervision() -> None:
    relation = _physical_relation_with_vjepa(available=False)
    frame = build_task_independent_calvin_targets(
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        patch_size=1,
        merge_size=2,
    )[0]
    native = slice(8, 8 + 2 * 24 * 24)

    assert frame.targets.masks.shape == (1, 2, 8 + 2 * 24 * 24)
    assert not frame.targets.mask_valid[..., native].any()
    assert not frame.targets.token_observed_fraction[..., native].any()


def test_videomt_dense_mask_surface_extends_the_same_physical_row_gauge() -> None:
    relation = _physical_relation_with_videomt()
    frame = build_task_independent_calvin_targets(
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        patch_size=1,
        merge_size=2,
    )[0]
    predictions = physical_frame_predictions_from_relation(relation)
    dense = slice(8, 8 + 120 * 120)

    assert frame.targets.masks.shape == (1, 2, 8 + 120 * 120)
    assert predictions.support_logits.shape == (1, 8 + 120 * 120, 2)
    assert predictions.ownership_log_probability.shape == (1, 8 + 120 * 120, 3)
    assert frame.targets.mask_valid[..., dense].all()
    assert torch.equal(frame.targets.masks[0, 0, dense], torch.ones(120 * 120))
    assert not frame.targets.masks[0, 1, dense].any()
    torch.testing.assert_close(frame.targets.token_measure[0, dense].sum(), torch.tensor(1.0))


def test_task_independent_calvin_projection_is_invariant_to_opaque_id_rename() -> None:
    relation = _physical_relation()
    common = {
        "requests_by_time": ((_request(10),),),
        "model_inputs_by_time": (_inputs(),),
        "relations": (relation,),
        "capacity": 2,
        "patch_size": 1,
        "merge_size": 2,
    }
    named = build_task_independent_calvin_targets(
        **common,
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
    )[0]
    renamed = build_task_independent_calvin_targets(
        **common,
        physical_sidecar=_Sidecar({10: _frame(("opaque/7", "opaque/9"))}),
    )[0]

    assert named.identity_keys_by_batch != renamed.identity_keys_by_batch
    for field in (
        "masks",
        "mask_valid",
        "existence",
        "existence_valid",
        "track_valid",
        "capacity_censored",
        "token_observed_fraction",
        "inventory_exhaustive",
    ):
        torch.testing.assert_close(
            getattr(named.targets, field),
            getattr(renamed.targets, field),
        )


def test_native_calvin_default_keeps_partially_observed_qwen_tokens_with_mass_weight() -> None:
    frame = _frame(("object/a", "object/b"))
    frame = CalvinPhysicalSupervisionFrame(
        identity_keys=frame.identity_keys,
        geometry=frame.geometry,
        geometry_variance=frame.geometry_variance,
        geometry_supervised=frame.geometry_supervised,
        geometry_contract=frame.geometry_contract,
        cameras=(
            _camera("static", 1, unknown_bottom_right=True),
            _camera("gripper", 2),
        ),
    )
    targets = _targets(
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(_relation(),),
        physical_sidecar=_Sidecar({10: frame}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
    )

    torch.testing.assert_close(
        targets.token_observed_fraction[0, 0, 1],
        torch.tensor(0.75),
    )
    assert targets.mask_valid[0, 0, :, 1].all()
    torch.testing.assert_close(targets.masks[0, 0, 0, 1], torch.tensor(1.0))


def test_native_calvin_targets_fail_on_source_hash_drift() -> None:
    sidecar = _Sidecar({10: _frame(("object/a", "object/b"), hash_override="e" * 64)})
    with pytest.raises(ValueError, match="RGB supervision source hash differs"):
        _targets(
            requests_by_time=((_request(10),),),
            model_inputs_by_time=(_inputs(),),
            relations=(_relation(),),
            physical_sidecar=sidecar,
            capacity=2,
            task_identity_resolver=lambda _task: ("object/a",),
            patch_size=1,
            merge_size=2,
        )


def test_native_calvin_targets_fail_closed_or_censor_overflow_task_independently() -> None:
    sidecar = _Sidecar({10: _frame(("object/a", "object/b", "object/c"), gripper_owner=3)})
    common = {
        "requests_by_time": ((_request(10),),),
        "model_inputs_by_time": (_inputs(),),
        "relations": (_relation(),),
        "physical_sidecar": sidecar,
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
    }
    with pytest.raises(ValueError, match="capacity seed"):
        _targets(**common)

    targets = _targets(**common, capacity_seeds=(17,))
    assert targets.track_valid.sum() == 3
    assert targets.capacity_censored.sum() == 1
    other_task = dict(common)
    other_task["task_identity_resolver"] = lambda _task: ("object/b",)
    other_targets = _targets(
        **other_task,
        capacity_seeds=(17,),
    )
    assert torch.equal(targets.capacity_censored, other_targets.capacity_censored)


def test_native_calvin_targets_leave_ambiguous_task_labels_unknown() -> None:
    targets = _targets(
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(_relation(),),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: None,
        patch_size=1,
        merge_size=2,
    )
    assert not targets.task_valid.any()
    assert not targets.task_relevance.any()


def test_native_calvin_composer_backpropagates_one_unified_production_objective() -> None:
    support = torch.zeros(1, 8, 2, requires_grad=True)
    task_logits = torch.zeros(1, 2, requires_grad=True)
    existence_logits = torch.zeros(1, 2, requires_grad=True)
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    no_object = torch.zeros(1, 8, 1)
    ownership, ownership_log_probability = _ownership_pair(torch.cat((support, no_object), dim=-1))
    relation = RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership * sensor_valid.unsqueeze(-1),
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=torch.ones(1, 4),
        row_embeddings=torch.ones(1, 2, 4),
        relation_temperature=torch.ones(1),
        dense_task_grounding=torch.zeros(1, 8),
        dense_task_grounding_logits=torch.zeros(1, 8),
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
        ownership_log_probability=ownership_log_probability,
    )
    action_loss = torch.tensor(0.5, requires_grad=True)
    result = compose_native_calvin_objective(
        official_policy_loss=action_loss,
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=1.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )

    assert result.objective.valid_counts == {
        "action": 1,
        "set/support": 0,
        "set/existence": 2,
        "set/task": 1,
        "set/task_dense": 1,
        "set/ownership": 2,
        "set/ownership_nll": 2,
    }
    result.objective.total.backward()
    assert action_loss.grad is not None and action_loss.grad.abs() > 0
    assert support.grad is not None and support.grad.abs().sum() > 0
    assert task_logits.grad is not None and task_logits.grad.abs().sum() > 0
    assert existence_logits.grad is not None and existence_logits.grad.abs().sum() > 0


def test_native_calvin_composer_supports_action_free_representation_objective() -> None:
    support = torch.zeros(1, 8, 2, requires_grad=True)
    task_logits = torch.zeros(1, 2, requires_grad=True)
    dense_logits = torch.zeros(1, 8, requires_grad=True)
    existence_logits = torch.zeros(1, 2, requires_grad=True)
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    ownership, ownership_log_probability = _ownership_pair(
        torch.cat((support, torch.zeros(1, 8, 1)), dim=-1)
    )
    relation = RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership * sensor_valid.unsqueeze(-1),
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=torch.ones(1, 4),
        row_embeddings=torch.ones(1, 2, 4),
        relation_temperature=torch.ones(1),
        dense_task_grounding=dense_logits.sigmoid() * sensor_valid,
        dense_task_grounding_logits=dense_logits,
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
        ownership_log_probability=ownership_log_probability,
    )
    result = compose_native_calvin_objective(
        official_policy_loss=None,
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            action_weight=0.0,
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
        require_policy_loss_grad=False,
    )

    assert "action" not in result.objective.normalized_terms
    torch.testing.assert_close(result.objective.family_terms["action"], torch.tensor(0.0))
    result.objective.total.backward()
    for value in (support, task_logits, dense_logits, existence_logits):
        assert value.grad is not None and value.grad.abs().sum() > 0


def test_native_calvin_composer_replaces_local_task_bce_with_global_retrieval() -> None:
    support = torch.zeros(1, 8, 2, requires_grad=True)
    local_task_logits = torch.zeros(1, 2, requires_grad=True)
    task_embedding = torch.tensor([[0.6, 0.4]], requires_grad=True)
    row_embeddings = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0]]],
        requires_grad=True,
    )
    dense_logits = torch.zeros(1, 8, requires_grad=True)
    existence_logits = torch.zeros(1, 2, requires_grad=True)
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    ownership, ownership_log_probability = _ownership_pair(
        torch.cat((support, torch.zeros(1, 8, 1)), dim=-1)
    )
    relation = RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership * sensor_valid.unsqueeze(-1),
        task_relevance=local_task_logits.sigmoid(),
        task_relevance_logits=local_task_logits,
        task_embedding=task_embedding,
        row_embeddings=row_embeddings,
        relation_temperature=torch.ones(1),
        dense_task_grounding=dense_logits.sigmoid() * sensor_valid,
        dense_task_grounding_logits=dense_logits,
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
        ownership_log_probability=ownership_log_probability,
    )
    result = compose_native_calvin_objective(
        official_policy_loss=None,
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            action_weight=0.0,
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=1.0,
            dense_task_weight=0.0,
            ownership_weight=0.0,
            task_relation_estimator=GLOBAL_MULTIPOSITIVE_TASK_RELATION,
        ),
        require_policy_loss_grad=False,
    )

    task_terms = [term for term in result.structural_terms if term.name == "set/task"]
    assert len(task_terms) == 1
    assert result.objective.valid_counts["set/task"] == 1
    result.objective.total.backward()
    assert task_embedding.grad is not None and task_embedding.grad.abs().sum() > 0
    assert row_embeddings.grad is not None and row_embeddings.grad.abs().sum() > 0
    assert local_task_logits.grad is None


def test_native_calvin_composer_replaces_local_task_bce_with_host_row_competition() -> None:
    support = torch.zeros(1, 8, 2, requires_grad=True)
    task_logits = torch.tensor([[0.2, 0.8]], requires_grad=True)
    dense_logits = torch.zeros(1, 8, requires_grad=True)
    existence_logits = torch.zeros(1, 2, requires_grad=True)
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    ownership, ownership_log_probability = _ownership_pair(
        torch.cat((support, torch.zeros(1, 8, 1)), dim=-1)
    )
    relation = RelationOutput(
        support_logits=support,
        visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
        ownership=ownership * sensor_valid.unsqueeze(-1),
        task_relevance=task_logits.sigmoid(),
        task_relevance_logits=task_logits,
        task_embedding=None,
        row_embeddings=torch.zeros(1, 2, 2),
        relation_temperature=torch.ones(1),
        dense_task_grounding=dense_logits.sigmoid() * sensor_valid,
        dense_task_grounding_logits=dense_logits,
        existence=existence_logits.sigmoid(),
        existence_logits=existence_logits,
        sensor_valid=sensor_valid,
        match_embeddings=torch.zeros(1, 2, 2),
        task_interface=HOST_NATIVE_MATCH_INTERFACE,
        ownership_log_probability=ownership_log_probability,
    )
    result = compose_native_calvin_objective(
        official_policy_loss=None,
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            action_weight=0.0,
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=1.0,
            dense_task_weight=0.0,
            ownership_weight=0.0,
            task_relation_estimator=HOST_NATIVE_MULTIPOSITIVE_TASK_RELATION,
        ),
        require_policy_loss_grad=False,
    )

    task_terms = [term for term in result.structural_terms if term.name == "set/task"]
    assert len(task_terms) == 1
    assert result.objective.valid_counts["set/task"] == 1
    expected = -torch.log_softmax(task_logits.detach(), dim=1)[0, 0]
    torch.testing.assert_close(result.objective.normalized_terms["set/task"], expected)
    result.objective.total.backward()
    assert task_logits.grad is not None
    assert task_logits.grad[0, 0] < 0
    assert task_logits.grad[0, 1] > 0


def test_factorized_task_estimator_forbids_a_second_independent_dense_objective() -> None:
    with pytest.raises(ValueError, match="zero independent dense-task weight"):
        NativeStructuralLossConfig(
            support_weight=1.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
            task_relation_estimator=HOST_NATIVE_FACTORIZED_TASK_RELATION,
        )


def test_structural_config_rejects_unknown_ownership_estimator() -> None:
    with pytest.raises(ValueError, match="unknown ownership estimator"):
        NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
            ownership_estimator="unregistered",
        )


def test_factorized_task_estimator_requires_its_physical_probability_factor() -> None:
    with pytest.raises(ValueError, match="positive task and ownership"):
        NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=0.0,
            ownership_weight=0.0,
            task_relation_estimator=HOST_NATIVE_FACTORIZED_TASK_RELATION,
        )


def test_factorized_task_estimator_requires_its_semantic_probability_factor() -> None:
    with pytest.raises(ValueError, match="positive task and ownership"):
        NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
            task_relation_estimator=HOST_NATIVE_FACTORIZED_TASK_RELATION,
        )


def test_native_calvin_composer_factorizes_task_row_from_physical_ownership() -> None:
    torch.manual_seed(53)
    readout = SharedRelationReadout(4, temperature_init=0.1)
    rows = torch.randn(1, 2, 4, requires_grad=True)
    sensors = torch.randn(1, 8, 4, requires_grad=True)
    match = torch.randn(1, 2, 4, requires_grad=True)
    sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
    relation = readout(
        posterior_rows=rows,
        sensor_hidden=sensors,
        sensor_valid=sensor_valid,
        match_hidden=match,
    )

    result = compose_native_calvin_objective(
        official_policy_loss=None,
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            action_weight=0.0,
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=1.0,
            dense_task_weight=0.0,
            ownership_weight=1.0,
            task_relation_estimator=HOST_NATIVE_FACTORIZED_TASK_RELATION,
        ),
        require_policy_loss_grad=False,
    )

    names = tuple(term.name for term in result.structural_terms)
    assert names.count("set/task_row") == 1
    assert "set/task" not in names
    assert result.objective.valid_counts["set/task_row"] == 1
    result.objective.total.backward()
    assert rows.grad is not None and rows.grad.abs().sum() > 0
    assert sensors.grad is not None and sensors.grad.abs().sum() > 0
    assert match.grad is not None and match.grad.abs().sum() > 0
    assert readout.projection.weight.grad is not None
    assert readout.projection.weight.grad.abs().sum() > 0
    assert readout.match_projection.weight.grad is not None
    assert readout.match_projection.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    ("ownership_estimator", "ownership_bases", "expected_term_weight"),
    (
        (TOKEN_MICRO_OWNERSHIP, ("set/ownership",), 0.25),
        (
            TOKEN_MICRO_ENTITY_CONDITIONAL_OWNERSHIP,
            ("set/ownership", "set/ownership_entity"),
            0.125,
        ),
    ),
)
def test_native_calvin_depth_supervision_reuses_gauge_and_conserves_ownership_weight(
    ownership_estimator: str,
    ownership_bases: tuple[str, ...],
    expected_term_weight: float,
) -> None:
    support_leaves: list[torch.Tensor] = []

    def attached_relation(seed: int) -> RelationOutput:
        generator = torch.Generator().manual_seed(seed)
        support = torch.randn(1, 8, 2, generator=generator, requires_grad=True)
        support_leaves.append(support)
        sensor_valid = torch.tensor([[False, True, False, False, True, False, False, False]])
        ownership, ownership_log_probability = _ownership_pair(
            torch.cat((support, torch.zeros(1, 8, 1)), dim=-1)
        )
        task_logits = torch.zeros(1, 2, requires_grad=True)
        existence_logits = torch.zeros(1, 2, requires_grad=True)
        dense_logits = torch.zeros(1, 8, requires_grad=True)
        return RelationOutput(
            support_logits=support,
            visible_support=support.sigmoid() * sensor_valid.unsqueeze(-1),
            ownership=ownership * sensor_valid.unsqueeze(-1),
            task_relevance=task_logits.sigmoid(),
            task_relevance_logits=task_logits,
            task_embedding=torch.ones(1, 4),
            row_embeddings=torch.ones(1, 2, 4),
            relation_temperature=torch.ones(1),
            dense_task_grounding=dense_logits.sigmoid() * sensor_valid,
            dense_task_grounding_logits=dense_logits,
            existence=existence_logits.sigmoid(),
            existence_logits=existence_logits,
            sensor_valid=sensor_valid,
            ownership_log_probability=ownership_log_probability,
        )

    final = attached_relation(0)
    intermediates = {
        8: (attached_relation(1),),
        17: (attached_relation(2),),
        26: (attached_relation(3),),
    }
    result = compose_native_calvin_objective(
        official_policy_loss=torch.tensor(0.5, requires_grad=True),
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(final,),
        intermediate_relations_by_layer=intermediates,
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=0.1,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
            ownership_estimator=ownership_estimator,
        ),
    )

    terms = {term.name: term for term in result.structural_terms}
    ownership_names = tuple(
        f"{base}{'' if depth == 0 else f'_q{depth}'}"
        for depth in range(4)
        for base in ownership_bases
    )
    assert tuple(name for name in ownership_names if name in terms) == ownership_names
    assert sum(terms[name].weight for name in ownership_names) == pytest.approx(1.0)
    assert all(
        terms[name].weight == pytest.approx(expected_term_weight) for name in ownership_names
    )
    expected_structural = (
        result.objective.normalized_terms["set/existence"]
        + result.objective.normalized_terms["set/task"]
        + result.objective.normalized_terms["set/task_dense"]
        + sum(
            result.objective.normalized_terms[name] * expected_term_weight
            for name in ownership_names
        )
    ) / 4
    assert torch.allclose(
        result.objective.family_terms["structural"],
        expected_structural * 0.1,
    )
    result.objective.total.backward()
    assert all(leaf.grad is not None and leaf.grad.abs().sum() > 0 for leaf in support_leaves)


def test_native_calvin_depth_supervision_rejects_unsorted_or_incomplete_time_axes() -> None:
    common = {
        "official_policy_loss": torch.tensor(0.5, requires_grad=True),
        "requests_by_time": ((_request(10),),),
        "model_inputs_by_time": (_inputs(),),
        "relations": (_relation(),),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    }
    with pytest.raises(ValueError, match="sorted and unique"):
        compose_native_calvin_objective(
            **common,
            intermediate_relations_by_layer={17: (_relation(),), 8: (_relation(),)},
        )
    with pytest.raises(ValueError, match="final time axis"):
        compose_native_calvin_objective(
            **common,
            intermediate_relations_by_layer={8: ()},
        )


def test_native_calvin_composer_keeps_episode_gauge_when_current_cost_reverses() -> None:
    common = {
        "requests_by_time": ((_request(10),),),
        "model_inputs_by_time": (_inputs(),),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: ("object/a",),
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    }
    first = compose_native_calvin_objective(
        official_policy_loss=torch.tensor(0.5, requires_grad=True),
        relations=(_relation_with_object_rows(reversed_rows=False),),
        prior_row_bindings_by_batch=((),),
        **common,
    )
    unconstrained_second = compose_native_calvin_objective(
        official_policy_loss=torch.tensor(0.5, requires_grad=True),
        relations=(_relation_with_object_rows(reversed_rows=True),),
        **common,
    )
    bound_second = compose_native_calvin_objective(
        official_policy_loss=torch.tensor(0.5, requires_grad=True),
        relations=(_relation_with_object_rows(reversed_rows=True),),
        prior_row_bindings_by_batch=first.row_bindings_by_batch,
        **common,
    )

    assert not torch.equal(
        unconstrained_second.assignment.row_to_track,
        first.assignment.row_to_track,
    )
    assert torch.equal(
        bound_second.assignment.row_to_track,
        first.assignment.row_to_track,
    )
    assert bound_second.row_bindings_by_batch == first.row_bindings_by_batch


def test_native_calvin_objective_forward_is_one_real_post_forward_transaction() -> None:
    policy, _coordinator = _components()
    routing = _routing(0, optimizer_step=0, frame_index=0, episode_key="episode/3")
    controls = _controls(1, reset=True)
    request = _request(10)
    request = NativeCALVINStructuralTargetRequest(
        sample_key=routing.sample_keys[0],
        episode_key=routing.episode_keys[0],
        task_key=request.task_key,
        segment_index=request.segment_index,
        source_global_index=request.source_global_index,
        source_sensor_sha256=request.source_sensor_sha256,
    )
    batch = CollatedNativeCALVINBatch(
        model_inputs=_official_calvin_model_inputs(),
        controls=controls,
        routing=routing,
        source_digest="f" * 64,
        structural_target_requests=(request,),
    )
    context = build_native_calvin_context(batch, previous_state=None)
    forward = run_native_policy_training_forward(
        policy,
        model_inputs=batch.model_inputs,
        context=context,
    )
    relation = forward.context.relation_output
    assert isinstance(relation, RelationOutput)
    result = compose_native_calvin_objective(
        official_policy_loss=forward.official_total_loss,
        requests_by_time=((request,),),
        model_inputs_by_time=(batch.model_inputs,),
        relations=(relation,),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda task: ("object/a",) if task == "push_a" else None,
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=1.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
    )

    assert result.objective.valid_counts == {
        "action": 1,
        "set/support": 0,
        "set/existence": 2,
        "set/task": 1,
        "set/task_dense": 1,
        "set/ownership": 2,
        "set/ownership_nll": 2,
    }
    result.objective.total.backward()
    graph = policy.model.qwenvl_with_expert.picf_native_graph
    assert graph.object_queries.grad is not None
    assert graph.object_queries.grad.abs().sum() > 0
    assert graph.relation_readout.projection.weight.grad is not None
    assert graph.relation_readout.projection.weight.grad.abs().sum() > 0


def test_native_calvin_objective_rejects_silently_missing_predictive_family() -> None:
    with pytest.raises(ValueError, match="positive predictive family weight"):
        compose_native_calvin_objective(
            official_policy_loss=torch.tensor(0.5, requires_grad=True),
            requests_by_time=((_request(10),),),
            model_inputs_by_time=(_inputs(),),
            relations=(_relation(),),
            physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
            capacity=2,
            task_identity_resolver=lambda _task: ("object/a",),
            patch_size=1,
            merge_size=2,
            objective_config=NativeObjectiveConfig(
                predictive_weight=1.0,
                structural_weight=1.0,
            ),
            structural_config=NativeStructuralLossConfig(
                support_weight=1.0,
                existence_weight=1.0,
                task_weight=1.0,
                dense_task_weight=1.0,
                ownership_weight=1.0,
            ),
        )


def test_native_calvin_objective_allows_detached_policy_loss_only_when_declared() -> None:
    arguments = {
        "official_policy_loss": torch.tensor(0.5),
        "requests_by_time": ((_request(10),),),
        "model_inputs_by_time": (_inputs(),),
        "relations": (_relation(),),
        "physical_sidecar": _Sidecar({10: _frame(("object/a", "object/b"))}),
        "capacity": 2,
        "task_identity_resolver": lambda _task: None,
        "patch_size": 1,
        "merge_size": 2,
        "objective_config": NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=0.0,
        ),
        "structural_config": NativeStructuralLossConfig(
            support_weight=0.0,
            existence_weight=0.0,
            task_weight=0.0,
            dense_task_weight=0.0,
            ownership_weight=0.0,
        ),
    }

    with pytest.raises(ValueError, match="finite attached scalar"):
        compose_native_calvin_objective(**arguments)

    result = compose_native_calvin_objective(
        **arguments,
        require_policy_loss_grad=False,
    )
    assert not result.objective.family_terms["action"].requires_grad
    assert result.objective.family_terms["action"].item() == pytest.approx(0.5)


def test_native_calvin_predictive_factory_runs_only_after_independent_targets() -> None:
    observed_identity_keys: list[tuple[tuple[str, ...], ...]] = []

    def predictive_factory(bundle):
        observed_identity_keys.append(bundle.identity_keys_by_batch)
        return ()

    result = compose_native_calvin_objective(
        official_policy_loss=torch.tensor(0.5, requires_grad=True),
        requests_by_time=((_request(10),),),
        model_inputs_by_time=(_inputs(),),
        relations=(_relation(),),
        physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
        capacity=2,
        task_identity_resolver=lambda _task: ("object/a",),
        patch_size=1,
        merge_size=2,
        objective_config=NativeObjectiveConfig(
            predictive_weight=0.0,
            structural_weight=1.0,
        ),
        structural_config=NativeStructuralLossConfig(
            support_weight=1.0,
            existence_weight=1.0,
            task_weight=1.0,
            dense_task_weight=1.0,
            ownership_weight=1.0,
        ),
        predictive_input_factory=predictive_factory,
    )

    assert observed_identity_keys == [(("object/a", "object/b"),)]
    assert result.predictive_terms == ()


def test_native_calvin_objective_rejects_two_predictive_input_paths() -> None:
    with pytest.raises(ValueError, match="directly or through one factory"):
        compose_native_calvin_objective(
            official_policy_loss=torch.tensor(0.5, requires_grad=True),
            requests_by_time=((_request(10),),),
            model_inputs_by_time=(_inputs(),),
            relations=(_relation(),),
            physical_sidecar=_Sidecar({10: _frame(("object/a", "object/b"))}),
            capacity=2,
            task_identity_resolver=lambda _task: None,
            patch_size=1,
            merge_size=2,
            objective_config=NativeObjectiveConfig(
                predictive_weight=0.0,
                structural_weight=0.0,
            ),
            structural_config=NativeStructuralLossConfig(
                support_weight=0.0,
                existence_weight=0.0,
                task_weight=0.0,
                dense_task_weight=0.0,
                ownership_weight=0.0,
            ),
            predictive_inputs=(object(),),
            predictive_input_factory=lambda _bundle: (),
        )

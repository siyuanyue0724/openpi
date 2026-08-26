from __future__ import annotations

# ruff: noqa: E402
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("olmo.hf_model.modeling_molmoact2")

from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinPhysicalSupervisionSidecar,
    CalvinVisibleOwnerRaster,
)
from picf_next.hosts.molmoact2 import (
    MOLMO_VISION_PATCH_MODALITY,
    MolmoAct2ImagePatchSpan,
    MolmoAct2VisionPatchLayout,
)
from picf_next.hosts.molmoact2_training import (
    CalvinSourceFrameLossTargetRequest,
    CalvinStatefulLossTargetLayout,
    CalvinStatefulLossTargetRequest,
    CalvinVisibleObjectTargetBuilder,
)
from picf_next.models.evidence import ModalityTokenSpan


def _pooling() -> tuple[tuple[int, ...], ...]:
    rows = []
    for pooled_y in range(14):
        for pooled_x in range(14):
            support = []
            for dy in range(2):
                for dx in range(2):
                    y = 2 * pooled_y + dy
                    x = 2 * pooled_x + dx
                    support.append(y * 27 + x if y < 27 and x < 27 else -1)
            rows.append(tuple(support))
    return tuple(rows)


class _PhysicalFixture(CalvinPhysicalSupervisionSidecar):
    def __init__(
        self,
        *,
        camera_names: tuple[str, ...] = ("static", "gripper"),
        unknown_owner: int | None = None,
        token_unmeasurable_owner: int | None = None,
        subpatch_owner: int | None = None,
        cascading_overlap: bool = False,
    ) -> None:
        self.camera_names = camera_names
        self.unknown_owner = unknown_owner
        self.token_unmeasurable_owner = token_unmeasurable_owner
        self.subpatch_owner = subpatch_owner
        self.cascading_overlap = cascading_overlap

    def __call__(self, segment_index: int, global_index: int) -> CalvinPhysicalSupervisionFrame:
        assert (segment_index, global_index) == (0, 10)
        return self.source_frame(global_index)

    def source_frame(self, global_index: int) -> CalvinPhysicalSupervisionFrame:
        assert global_index == 10
        static = np.zeros((200, 200), dtype=np.uint8)
        static[:, :80] = 1
        static[40:120, 120:190] = 2
        if self.subpatch_owner is not None:
            static[static == self.subpatch_owner] = 0
            static[100, 100] = self.subpatch_owner
        gripper = np.zeros((84, 84), dtype=np.uint8)
        gripper[10:70, 20:75] = 2
        if self.cascading_overlap:
            static.fill(0)
            static[82:88, 82:88] = 2
            static[84, 84] = 1
            gripper.fill(0)
        cameras = []
        camera_values = {
            "static": ("observation.images.image", static),
            "gripper": ("observation.images.wrist_image", gripper),
        }
        for name in self.camera_names:
            key, owner = camera_values[name]
            owner.setflags(write=False)
            owner_supervised = np.ones(owner.shape, dtype=np.bool_)
            if self.unknown_owner is not None:
                owner_supervised[owner == self.unknown_owner] = False
            if self.token_unmeasurable_owner is not None:
                owner_mask = owner == self.token_unmeasurable_owner
                owner_supervised[owner_mask] = False
                owner_pixels = np.argwhere(owner_mask)
                if owner_pixels.size:
                    y, x = owner_pixels[0]
                    owner_supervised[y, x] = True
            owner_supervised.setflags(write=False)
            cameras.append(
                CalvinVisibleOwnerRaster(
                    camera_name=name,
                    host_image_key=key,
                    owner_index=owner,
                    owner_supervised=owner_supervised,
                    source_rgb_sha256="a" * 64,
                    source_depth_sha256="b" * 64,
                    rgb_mae=1.0,
                    depth_mae_m=0.001,
                    depth_p95_m=0.002,
                    depth_consistent_fraction=float(owner_supervised.mean()),
                )
            )
        return CalvinPhysicalSupervisionFrame(
            identity_keys=("object/red", "part/button", "object/occluded"),
            geometry=torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
            geometry_variance=torch.zeros(3, 3),
            geometry_supervised=torch.ones(3, 3, dtype=torch.bool),
            geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
            cameras=tuple(cameras),
        )


def _layout(*, semantic: bool = True) -> CalvinStatefulLossTargetLayout:
    pooling = _pooling()
    rows = (
        (
            MolmoAct2ImagePatchSpan(
                image_key="observation.images.image",
                start=0,
                stop=729,
                image_num_crops=1,
                patches_per_crop=729,
                image_grid=(14, 14, 0, 0),
                image_token_pooling=pooling,
            ),
            MolmoAct2ImagePatchSpan(
                image_key="observation.images.wrist_image",
                start=729,
                stop=1458,
                image_num_crops=1,
                patches_per_crop=729,
                image_grid=(14, 14, 0, 0),
                image_token_pooling=pooling,
            ),
        ),
    )
    vision = MolmoAct2VisionPatchLayout(
        rows=rows,
        tokens_per_row=1458,
        semantic_image_keys=semantic,
    )
    return CalvinStatefulLossTargetLayout(
        token_valid=torch.ones(1, 1460, dtype=torch.bool),
        spans=(
            ModalityTokenSpan("proprio", 0, 2),
            ModalityTokenSpan(MOLMO_VISION_PATCH_MODALITY, 2, 1460),
        ),
        target_dtype=torch.float32,
        rollout_input_dtype=torch.float32,
        vision_patch_layout=vision,
    )


def _request(
    *,
    overrides: dict[str, str] | None = None,
) -> CalvinStatefulLossTargetRequest:
    hashes = {
        "depth_gripper": "b" * 64,
        "depth_static": "b" * 64,
        "rgb_gripper": "a" * 64,
        "rgb_static": "a" * 64,
    }
    hashes.update(overrides or {})
    return CalvinStatefulLossTargetRequest(
        "sample",
        0,
        10,
        0,
        source_sensor_sha256=tuple(sorted(hashes.items())),
    )


def _source_request() -> CalvinSourceFrameLossTargetRequest:
    hashes = {
        "depth_gripper": "b" * 64,
        "depth_static": "b" * 64,
        "rgb_gripper": "a" * 64,
        "rgb_static": "a" * 64,
    }
    return CalvinSourceFrameLossTargetRequest(
        "source-frame",
        10,
        0,
        source_sensor_sha256=tuple(sorted(hashes.items())),
    )


def test_visible_sets_and_occluded_lifecycle_are_separate() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture())
    targets = builder(
        (_request(),),
        _layout(),
    )

    assert targets.set_targets is not None
    assert targets.lifecycle_targets is not None
    target = targets.set_targets[0]
    lifecycle = targets.lifecycle_targets[0]
    assert target.temporal_identity_keys == ("object/red", "part/button")
    assert target.num_objects == 2
    assert target.geometry is not None and target.geometry.shape == (2, 3)
    assert target.geometry_variance is not None
    torch.testing.assert_close(target.geometry_variance, torch.zeros(2, 3))
    assert target.object_inventory_complete
    assert not target.token_supervised[:2].any()
    assert target.token_supervised[2:].all()
    torch.testing.assert_close(
        target.ownership[target.token_supervised].sum(dim=-1),
        torch.ones_like(target.ownership[target.token_supervised, 0]),
    )
    assert lifecycle is not None
    assert lifecycle.alive_identity_keys == (
        "object/red",
        "part/button",
        "object/occluded",
    )
    assert lifecycle.inventory_complete
    assert lifecycle.visibility is not None
    assert lifecycle.visibility.tolist() == [1.0, 1.0, 0.0]
    assert lifecycle.visibility_supervised is not None
    assert lifecycle.visibility_supervised.all()


def test_all_source_visible_targets_do_not_require_language_segment_identity() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture())
    targets = builder.source_frames(
        (_source_request(),),
        _layout(),
    )

    assert targets.set_targets is not None
    assert targets.set_targets[0].temporal_identity_keys == ("object/red", "part/button")
    with pytest.raises(TypeError, match="stateful loss-target requests"):
        builder((_source_request(),), _layout())  # type: ignore[arg-type]


def test_intervened_measurement_frame_uses_same_set_likelihood_without_lifecycle() -> None:
    fixture = _PhysicalFixture()
    builder = CalvinVisibleObjectTargetBuilder(fixture)
    factual = fixture.source_frame(10)
    removed_cameras = []
    for camera in factual.cameras:
        owner = camera.owner_index.copy()
        owner[owner == 1] = 0
        owner.setflags(write=False)
        removed_cameras.append(replace(camera, owner_index=owner))
    removed = replace(factual, cameras=tuple(removed_cameras))
    hashes = tuple(
        sorted(
            {
                "depth_gripper": "b" * 64,
                "depth_static": "b" * 64,
                "rgb_gripper": "a" * 64,
                "rgb_static": "a" * 64,
            }.items()
        )
    )

    targets = builder.measurement_frames((removed,), (hashes,), _layout())

    assert len(targets) == 1
    assert targets[0].object_inventory_complete
    assert targets[0].temporal_identity_keys == ("part/button",)
    assert "object/red" not in targets[0].temporal_identity_keys


def test_depth_inconsistent_ownership_is_unknown_but_detection_absence_is_supervised() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture(unknown_owner=1))
    targets = builder((_request(),), _layout())

    assert targets.set_targets is not None
    assert targets.lifecycle_targets is not None
    target = targets.set_targets[0]
    lifecycle = targets.lifecycle_targets[0]
    assert target.temporal_identity_keys == ("part/button",)
    assert lifecycle is not None
    assert lifecycle.visibility is not None
    assert lifecycle.visibility.tolist() == [0.0, 1.0, 0.0]
    assert lifecycle.visibility_supervised is not None
    assert lifecycle.visibility_supervised.tolist() == [True, True, True]
    assert not target.token_supervised.all()
    assert (target.ownership[~target.token_supervised] == 0.0).all()


def test_raw_visible_unmeasurable_ownership_is_unknown_but_detection_is_absent() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture(token_unmeasurable_owner=1))
    targets = builder((_request(),), _layout())

    assert targets.set_targets is not None
    assert targets.lifecycle_targets is not None
    target = targets.set_targets[0]
    lifecycle = targets.lifecycle_targets[0]
    assert target.temporal_identity_keys == ("part/button",)
    assert lifecycle is not None
    assert lifecycle.visibility is not None
    assert lifecycle.visibility.tolist() == [0.0, 1.0, 0.0]
    assert lifecycle.visibility_supervised is not None
    assert lifecycle.visibility_supervised.tolist() == [True, True, True]
    assert (target.ownership[target.token_supervised, :-1].sum(dim=0) > 0.0).all()


def test_positive_subpatch_ownership_remains_a_soft_current_detection() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture(subpatch_owner=1))
    targets = builder((_request(),), _layout())

    assert targets.set_targets is not None
    assert targets.lifecycle_targets is not None
    target = targets.set_targets[0]
    lifecycle = targets.lifecycle_targets[0]
    assert target.temporal_identity_keys == ("object/red", "part/button")
    assert target.num_objects == 2
    assert lifecycle is not None
    assert lifecycle.visibility is not None
    assert lifecycle.visibility.tolist() == [1.0, 1.0, 0.0]
    assert lifecycle.visibility_supervised is not None
    assert lifecycle.visibility_supervised.tolist() == [True, True, True]
    assert int((~target.token_supervised).sum()) == 2
    assert (target.ownership[~target.token_supervised] == 0.0).all()
    assert (target.ownership[target.token_supervised, :-1].sum(dim=0) > 0.0).all()
    torch.testing.assert_close(
        target.ownership[target.token_supervised].sum(dim=-1),
        torch.ones_like(target.ownership[target.token_supervised, 0]),
    )


def test_mixed_subpatch_owners_remain_one_exclusive_soft_target() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture(cascading_overlap=True))
    targets = builder((_request(),), _layout())

    assert targets.set_targets is not None
    assert targets.lifecycle_targets is not None
    target = targets.set_targets[0]
    lifecycle = targets.lifecycle_targets[0]
    assert target.temporal_identity_keys == ("object/red", "part/button")
    assert target.num_objects == 2
    assert lifecycle is not None
    assert lifecycle.visibility is not None
    assert lifecycle.visibility.tolist() == [1.0, 1.0, 0.0]
    assert lifecycle.visibility_supervised is not None
    assert lifecycle.visibility_supervised.tolist() == [True, True, True]
    assert int((~target.token_supervised).sum()) == 2
    assert (target.ownership[~target.token_supervised] == 0.0).all()
    assert (target.ownership[target.token_supervised, :-1].sum(dim=0) > 0.0).all()
    torch.testing.assert_close(
        target.ownership[target.token_supervised].sum(dim=-1),
        torch.ones_like(target.ownership[target.token_supervised, 0]),
    )


def test_visible_builder_rejects_implicit_camera_names() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture())
    with pytest.raises(ValueError, match="semantic image keys"):
        builder(
            (_request(),),
            _layout(semantic=False),
        )


def test_visible_builder_rejects_source_sensor_drift() -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture())
    with pytest.raises(ValueError, match="source sensor hash differs"):
        builder(
            (_request(overrides={"rgb_static": "c" * 64}),),
            _layout(),
        )


@pytest.mark.parametrize("camera_names", [("static",), ("static", "static")])
def test_visible_builder_rejects_incomplete_or_duplicate_cameras(
    camera_names: tuple[str, ...],
) -> None:
    builder = CalvinVisibleObjectTargetBuilder(_PhysicalFixture(camera_names=camera_names))
    with pytest.raises(ValueError, match="camera"):
        builder(
            (_request(),),
            _layout(),
        )

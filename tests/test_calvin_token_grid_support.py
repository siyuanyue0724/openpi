from __future__ import annotations

import numpy as np
import pytest
import torch

from picf_next.contracts import ContractError
from picf_next.data.calvin_geometry_schema import CALVIN_OBJECT_GEOMETRY_CONTRACT
from picf_next.data.calvin_physical_supervision_sidecar import (
    CalvinPhysicalSupervisionFrame,
    CalvinVisibleOwnerRaster,
)
from picf_next.data.calvin_token_grid_support import (
    project_calvin_token_grid_identity_support,
)


def _projection() -> dict[str, object]:
    def view(source_field: str, source_shape: list[int], digit: str) -> dict[str, object]:
        return {
            "source_field": source_field,
            "source_shape": source_shape,
            "image_grid_thw": [1, 4, 4],
            "merged_grid_hw": [2, 2],
            "raw_patch_count": 16,
            "merged_token_count": 4,
            "pixel_values_shape": [16, 6],
            "source_rgb_sha256": [digit * 64] * 3,
        }

    return {
        "schema": "picf-next.lingbot-calvin-qwen-projection.v1",
        "status": "PASS",
        "runtime_input": False,
        "processor_id": "Qwen/Qwen3-VL-4B-Instruct",
        "processor_revision": "a" * 40,
        "processor_assets_sha256": "b" * 64,
        "processor_config_sha256": "c" * 64,
        "processor_preprocessor_config_sha256": "d" * 64,
        "dataset_manifest_sha256": "e" * 64,
        "dataset_tree_sha256": "f" * 64,
        "source_frame_count": 20,
        "sample_global_indices": [0, 10, 19],
        "patch_size": 1,
        "merge_size": 2,
        "temporal_patch_size": 2,
        "views": {
            "static": view("rgb_static", [200, 200, 3], "1"),
            "gripper": view("rgb_gripper", [84, 84, 3], "2"),
        },
        "transformers_version": "5.0.0",
    }


def _camera(name: str, owner: np.ndarray) -> CalvinVisibleOwnerRaster:
    return CalvinVisibleOwnerRaster(
        camera_name=name,
        host_image_key=(
            "observation.images.image" if name == "static" else "observation.images.wrist_image"
        ),
        owner_index=owner,
        owner_supervised=np.ones_like(owner, dtype=np.bool_),
        source_rgb_sha256=("1" if name == "static" else "2") * 64,
        source_depth_sha256=("3" if name == "static" else "4") * 64,
        rgb_mae=0.0,
        depth_mae_m=0.0,
        depth_p95_m=0.0,
        depth_consistent_fraction=1.0,
    )


def _frame(static: np.ndarray, gripper: np.ndarray) -> CalvinPhysicalSupervisionFrame:
    dimension = CALVIN_OBJECT_GEOMETRY_CONTRACT.dimension
    return CalvinPhysicalSupervisionFrame(
        identity_keys=("object/one", "object/two"),
        geometry=torch.zeros(2, dimension),
        geometry_variance=torch.zeros(2, dimension),
        geometry_supervised=torch.ones(2, dimension, dtype=torch.bool),
        geometry_contract=CALVIN_OBJECT_GEOMETRY_CONTRACT,
        cameras=(
            _camera("static", static),
            _camera("gripper", gripper),
        ),
    )


def test_token_grid_support_uses_projected_instance_id_mapping_per_view() -> None:
    static = np.zeros((200, 200), dtype=np.uint8)
    static[:100, :100] = 1
    gripper = np.zeros((84, 84), dtype=np.uint8)
    gripper[42:, 42:] = 2

    supports = project_calvin_token_grid_identity_support(
        _frame(static, gripper),
        projection=_projection(),
    )
    by_identity = {value.identity_key: value for value in supports}

    assert by_identity["object/one"].measurable
    assert by_identity["object/two"].measurable
    one_views = {value.camera_name: value for value in by_identity["object/one"].views}
    two_views = {value.camera_name: value for value in by_identity["object/two"].views}
    assert one_views["static"].target_mass > 0
    assert one_views["gripper"].target_mass == 0
    assert two_views["static"].target_mass == 0
    assert two_views["gripper"].target_mass > 0
    assert by_identity["object/one"].object_row_addressable
    assert by_identity["object/two"].object_row_addressable
    assert by_identity["object/one"].strict_categorical_winner_token_count > 0
    assert by_identity["object/two"].strict_categorical_winner_token_count > 0


def test_token_grid_support_reports_raw_visible_but_downsampled_away_identity() -> None:
    static = np.zeros((200, 200), dtype=np.uint8)
    static[:100, :100] = 1
    static[-1, -1] = 2
    gripper = np.zeros((84, 84), dtype=np.uint8)

    supports = project_calvin_token_grid_identity_support(
        _frame(static, gripper),
        projection=_projection(),
    )

    assert supports[0].measurable
    assert not supports[1].measurable
    assert not supports[1].object_row_addressable
    assert supports[1].target_mass == 0
    assert supports[1].positive_token_count == 0


def test_token_grid_support_rejects_tied_physical_object_rows() -> None:
    static = np.zeros((200, 200), dtype=np.uint8)
    static[:100, :50] = 1
    static[:100, 50:100] = 2
    gripper = np.zeros((84, 84), dtype=np.uint8)

    supports = project_calvin_token_grid_identity_support(
        _frame(static, gripper),
        projection=_projection(),
    )

    assert supports[0].measurable
    assert supports[1].measurable
    assert not supports[0].object_row_addressable
    assert not supports[1].object_row_addressable


def test_token_grid_support_keeps_unique_object_evidence_separate_from_context() -> None:
    static = np.zeros((200, 200), dtype=np.uint8)
    static[:50, :50] = 1
    gripper = np.zeros((84, 84), dtype=np.uint8)

    support = project_calvin_token_grid_identity_support(
        _frame(static, gripper),
        projection=_projection(),
    )[0]

    assert support.measurable
    assert support.object_row_addressable
    assert support.strict_categorical_winner_token_count == 0


def test_token_grid_support_rejects_projection_camera_drift() -> None:
    static = np.zeros((200, 200), dtype=np.uint8)
    gripper = np.zeros((84, 84), dtype=np.uint8)
    projection = _projection()
    del projection["views"]["gripper"]  # type: ignore[index]

    with pytest.raises(ContractError, match="projection views"):
        project_calvin_token_grid_identity_support(
            _frame(static, gripper),
            projection=projection,
        )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True])
def test_token_grid_support_rejects_invalid_supervision_fraction(value: object) -> None:
    static = np.zeros((200, 200), dtype=np.uint8)
    gripper = np.zeros((84, 84), dtype=np.uint8)

    with pytest.raises(ContractError, match="fraction"):
        project_calvin_token_grid_identity_support(
            _frame(static, gripper),
            projection=_projection(),
            minimum_supervised_fraction=value,  # type: ignore[arg-type]
        )

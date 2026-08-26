from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from picf_next.lingbot_native.wsa_da3_loss import (
    WSA_DA3_QUERY_DIM,
    WSA_DA3_TEACHER_LAYERS,
    WSA_DA3_TOKENS_PER_VIEW,
)
from picf_next.lingbot_native.wsa_da3_teacher_runtime import (
    OnlineWSADA3TeacherRuntime,
    prepare_official_wsa_da3_future_views,
)


class _FakeTeacher(nn.Module):
    feature_dim = WSA_DA3_QUERY_DIM
    teacher_layers = WSA_DA3_TEACHER_LAYERS

    def __init__(self, **_: object) -> None:
        super().__init__()
        self.register_parameter("sentinel", nn.Parameter(torch.zeros(())))
        self.seen_shape: tuple[int, ...] | None = None

    def forward(self, images: torch.Tensor) -> list[torch.Tensor]:
        self.seen_shape = tuple(images.shape)
        batch = images.shape[0]
        return [
            images.new_full(
                (batch, 2 * WSA_DA3_TOKENS_PER_VIEW, WSA_DA3_QUERY_DIM),
                float(layer),
                dtype=torch.bfloat16,
            )
            for layer in WSA_DA3_TEACHER_LAYERS
        ]


def _observation(*, wrist: bool = True) -> dict[str, np.ndarray | None]:
    return {
        "observation.images.camera_top": np.full((200, 200, 3), 128, dtype=np.uint8),
        "observation.images.camera_wrist_left": (
            np.full((3, 84, 84), 64, dtype=np.uint8) if wrist else None
        ),
    }


def test_prepare_future_views_resizes_both_cameras_and_preserves_validity() -> None:
    images, valid = prepare_official_wsa_da3_future_views(
        (_observation(), _observation(wrist=False))
    )

    assert images.shape == (2, 2, 3, 504, 504)
    assert valid.tolist() == [[True, True], [True, False]]
    assert torch.equal(images[1, 0], images[1, 1])


def test_prepare_future_views_uses_explicit_dataset_camera_contract() -> None:
    static = np.full((200, 200, 3), 17, dtype=np.uint8)
    wrist = np.full((84, 84, 3), 29, dtype=np.uint8)
    images, valid = prepare_official_wsa_da3_future_views(
        (
            {
                "observation.images.image": static,
                "observation.images.wrist_image": wrist,
            },
        ),
        camera_keys=(
            "observation.images.image",
            "observation.images.wrist_image",
        ),
    )

    assert images.shape == (1, 2, 3, 504, 504)
    assert valid.tolist() == [[True, True]]
    assert images[0, 0].mean().item() == pytest.approx(17.0)
    assert images[0, 1].mean().item() == pytest.approx(29.0)


@pytest.mark.parametrize(
    "camera_keys",
    [(), ("one",), ("same", "same"), ("one", "two", "three")],
)
def test_prepare_future_views_rejects_ambiguous_camera_contract(
    camera_keys: tuple[str, ...],
) -> None:
    with pytest.raises(ValueError, match="exactly two distinct"):
        prepare_official_wsa_da3_future_views(
            (_observation(),),
            camera_keys=camera_keys,
        )


def test_online_runtime_builds_complete_targets_and_offloads_teacher(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    source_dir = tmp_path / "source"
    wsa_dir = tmp_path / "wsa"
    model_dir.mkdir()
    source_dir.mkdir()
    wsa_dir.mkdir()
    runtime = OnlineWSADA3TeacherRuntime.from_official_source(
        wsa_source_root=wsa_dir,
        da3_model_dir=model_dir,
        da3_source_root=source_dir,
        teacher_factory=_FakeTeacher,
    )

    targets, receipt = runtime.build_targets(
        future_observations=(_observation(),),
        future_source_global_indices=(123,),
        device="cpu",
    )

    assert tuple(runtime.teacher.parameters())[0].device.type == "cpu"
    assert runtime.teacher.seen_shape == (1, 2, 3, 504, 504)
    assert len(targets.layers) == 4
    assert targets.layers[0].shape == (1, 2592, 2048)
    assert targets.layers[0].dtype is torch.bfloat16
    assert receipt.future_source_global_indices == (123,)
    assert receipt.valid_view_count == 2


def test_future_views_reject_sample_without_real_camera() -> None:
    with pytest.raises(ValueError, match="no valid future camera"):
        prepare_official_wsa_da3_future_views(
            (
                {
                    "observation.images.camera_top": None,
                    "observation.images.camera_wrist_left": None,
                },
            )
        )

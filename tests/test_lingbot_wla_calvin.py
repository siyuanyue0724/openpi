from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from picf_next.lingbot_wla_calvin import (
    build_wla_calvin_target_batch,
    build_wla_calvin_target_batch_from_source_indices,
)


class _FakeIndex:
    def __init__(self) -> None:
        self.requests: list[tuple[int, tuple[str, ...]]] = []

    def validated_source_frame_arrays(self, global_index: int, *, fields: tuple[str, ...]):
        self.requests.append((global_index, fields))
        return {"rgb_static": np.full((200, 200, 3), global_index % 251, dtype=np.uint8)}

    @staticmethod
    def source_episode(global_index: int):
        return SimpleNamespace(start=0, end=100)


class _FakeDataset:
    def __init__(self) -> None:
        self.index = _FakeIndex()

    @staticmethod
    def source_global_index_by_key(sample_key: str) -> int:
        return {"a": 10, "b": 30}[sample_key]


def test_wla_calvin_target_is_exact_horizon_and_not_model_input() -> None:
    dataset = _FakeDataset()

    def transform(image):
        value = torch.as_tensor(np.asarray(image).copy()).permute(2, 0, 1).float()
        return torch.nn.functional.interpolate(
            value.unsqueeze(0),
            size=(512, 512),
            mode="nearest",
        ).squeeze(0)

    target = build_wla_calvin_target_batch(
        dataset,
        ("a", "b"),
        action_horizon=8,
        target_transform=transform,
    )
    assert target.images.shape == (2, 3, 512, 512)
    assert target.source_global_indices == (18, 38)
    assert dataset.index.requests == [
        (18, ("rgb_static",)),
        (38, ("rgb_static",)),
    ]
    assert len(target.source_rgb_sha256) == 2
    assert all(len(value) == 64 for value in target.source_rgb_sha256)


def test_wla_calvin_source_index_boundary_is_key_namespace_independent() -> None:
    index = _FakeIndex()

    def transform(image):
        value = torch.as_tensor(np.asarray(image).copy()).permute(2, 0, 1).float()
        return torch.nn.functional.interpolate(
            value.unsqueeze(0), size=(512, 512), mode="nearest"
        ).squeeze(0)

    target = build_wla_calvin_target_batch_from_source_indices(
        index,
        (10, 30),
        action_horizon=8,
        target_transform=transform,
    )

    assert target.source_global_indices == (18, 38)
    assert index.requests == [(18, ("rgb_static",)), (38, ("rgb_static",))]

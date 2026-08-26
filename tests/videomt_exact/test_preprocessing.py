from __future__ import annotations

import numpy as np
import torch

from picf_next.videomt_exact.preprocessing import (
    official_topk_query_classes,
    prepare_rgb_frames,
    resize_query_masks_to_original,
    unique_query_topk,
)


def test_released_preprocessing_contract() -> None:
    frames = [np.zeros((200, 320, 3), dtype=np.uint8) for _ in range(2)]
    prepared = prepare_rgb_frames(frames)
    assert prepared.original_sizes == ((200, 320), (200, 320))
    assert prepared.resized_sizes == ((480, 768), (480, 768))
    assert prepared.padded_size == (480, 768)
    assert prepared.model_input.shape == (2, 3, 480, 768)
    assert torch.isfinite(prepared.model_input).all()


def test_official_and_unique_query_selection_contract() -> None:
    logits = torch.full((2, 3, 5), -5.0)
    logits[:, 0, 2] = 5.0
    logits[:, 1, 1] = 4.0
    logits[:, 2, 4] = 6.0
    scores, queries, classes = official_topk_query_classes(logits, topk=2)
    assert set(zip(queries.tolist(), classes.tolist(), strict=True)) == {(0, 2), (1, 1)}
    unique_scores, unique_queries, unique_classes = unique_query_topk(logits, topk=2)
    assert unique_queries.tolist() == [0, 1]
    assert unique_classes.tolist() == [2, 1]
    assert torch.all(unique_scores[:-1] >= unique_scores[1:])


def test_two_stage_mask_resize_contract() -> None:
    masks = torch.randn(2, 3, 8, 12)
    resized = resize_query_masks_to_original(
        masks,
        padded_size=(32, 48),
        resized_size=(30, 45),
        original_size=(20, 30),
    )
    assert resized.shape == (2, 3, 20, 30)
    assert torch.isfinite(resized).all()

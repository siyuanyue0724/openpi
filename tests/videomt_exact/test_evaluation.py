from __future__ import annotations

import numpy as np
import torch

from picf_next.videomt_exact.calvin_targets import PreparedCalvinVidEoMTClip
from picf_next.videomt_exact.evaluation import _video_iou, evaluate_videomt_anchors
from picf_next.videomt_exact.preprocessing import PreparedVidEoMTFrames
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput


def test_video_level_matching_recovers_distinct_physical_queries() -> None:
    time, height, width = 2, 16, 16
    rgb = tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(time))
    frames = PreparedVidEoMTFrames(
        model_input=torch.zeros(time, 3, height, width),
        resized_rgb=rgb,
        original_sizes=((height, width),) * time,
        resized_sizes=((height, width),) * time,
        padded_size=(height, width),
    )
    targets = torch.zeros(2, time, height, width)
    targets[0, :, 2:7, 1:6] = 1
    targets[1, :, 9:14, 10:15] = 1
    clip = PreparedCalvinVidEoMTClip(
        frames=frames,
        target={
            "labels": torch.zeros(2, dtype=torch.long),
            "ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
            "masks": targets,
        },
        identity_keys=("left", "right"),
        camera_name="static",
    )

    class_logits = torch.full((1, time, 200, 41), -5.0)
    class_logits[..., 40] = 5.0
    class_logits[:, :, :2, 0] = 5.0
    class_logits[:, :, :2, 40] = -5.0
    mask_logits = torch.full((1, 200, time, height, width), -10.0)
    mask_logits[0, 0, :, 2:7, 1:6] = 10.0
    mask_logits[0, 1, :, 9:14, 10:15] = 10.0
    output = ExactVidEoMTOutput(
        class_logits=class_logits,
        mask_logits=mask_logits,
        query_embeddings=torch.zeros(1, time, 200, 1024),
        propagated_queries=torch.zeros(1, 200, 1024),
        auxiliary_outputs=(),
    )

    result = evaluate_videomt_anchors(output, clip)

    assert result.query_indices == (0, 1)
    assert result.mean_binary_iou == 1.0
    assert result.recall_at_50 == 1.0
    assert result.mean_foreground_probability > 0.99
    assert all(value.recall_at_50 == 1.0 for value in result.ranked_proposals)
    assert all(value.query_indices[:2] == (0, 1) for value in result.ranked_proposals)
    assert dict(result.foreground_query_counts) == {0.1: 2, 0.25: 2, 0.5: 2}


def test_ranked_metrics_expose_oracle_only_proposals() -> None:
    time, height, width = 2, 16, 16
    rgb = tuple(np.zeros((height, width, 3), dtype=np.uint8) for _ in range(time))
    frames = PreparedVidEoMTFrames(
        model_input=torch.zeros(time, 3, height, width),
        resized_rgb=rgb,
        original_sizes=((height, width),) * time,
        resized_sizes=((height, width),) * time,
        padded_size=(height, width),
    )
    targets = torch.zeros(2, time, height, width)
    targets[0, :, 2:7, 1:6] = 1
    targets[1, :, 9:14, 10:15] = 1
    clip = PreparedCalvinVidEoMTClip(
        frames=frames,
        target={
            "labels": torch.zeros(2, dtype=torch.long),
            "ids": torch.tensor([[0, 0], [1, 1]], dtype=torch.long),
            "masks": targets,
        },
        identity_keys=("left", "right"),
        camera_name="static",
    )
    class_logits = torch.full((1, time, 200, 41), -5.0)
    class_logits[..., 40] = 5.0
    class_logits[:, :, 2:14, 0] = 5.0
    class_logits[:, :, 2:14, 40] = -5.0
    class_logits[:, :, :2, 0] = 0.0
    class_logits[:, :, :2, 40] = 0.0
    mask_logits = torch.full((1, 200, time, height, width), -10.0)
    mask_logits[0, 0, :, 2:7, 1:6] = 10.0
    mask_logits[0, 1, :, 9:14, 10:15] = 10.0
    output = ExactVidEoMTOutput(
        class_logits=class_logits,
        mask_logits=mask_logits,
        query_embeddings=torch.zeros(1, time, 200, 1024),
        propagated_queries=torch.zeros(1, 200, 1024),
        auxiliary_outputs=(),
    )

    result = evaluate_videomt_anchors(output, clip)

    assert result.mean_binary_iou == 1.0
    by_k = {value.top_k: value for value in result.ranked_proposals}
    assert by_k[10].recall_at_50 == 0.0
    assert by_k[10].query_indices == (2, 3)
    assert by_k[25].recall_at_50 == 1.0
    assert by_k[25].query_indices == (0, 1)


def test_video_iou_disables_outer_autocast_for_large_pixel_reductions() -> None:
    predictions = torch.ones(2, 2, 256, 256)
    targets = torch.ones(1, 2, 256, 256)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _video_iou(predictions, targets)

    assert result.dtype == torch.float32
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, torch.ones_like(result))

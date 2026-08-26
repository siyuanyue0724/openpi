from __future__ import annotations

from pathlib import Path

import torch

from picf_next.videomt_exact.joint_visual import (
    NATIVE_VIDEOMT_QUERY_VISUAL_SCHEMA,
    render_native_videomt_query_visuals,
)
from picf_next.videomt_exact.runtime import ExactVidEoMTOutput


def test_native_query_visual_preserves_query_ids_without_training_selection(
    tmp_path: Path,
) -> None:
    height = width = 32
    class_logits = torch.full((1, 1, 200, 41), -6.0)
    class_logits[..., -1] = 6.0
    class_logits[0, 0, :2, 0] = 6.0
    class_logits[0, 0, :2, -1] = -6.0
    mask_logits = torch.full((1, 200, 1, 8, 8), -8.0)
    mask_logits[0, 0, 0, 1:4, 1:4] = 8.0
    mask_logits[0, 1, 0, 5:7, 5:7] = 8.0
    output = ExactVidEoMTOutput(
        class_logits=class_logits,
        mask_logits=mask_logits,
        query_embeddings=torch.zeros(1, 1, 200, 1024),
        propagated_queries=torch.zeros(1, 200, 1024),
        auxiliary_outputs=(),
    )
    targets = torch.zeros(2, 5, height, width)
    targets[0, :, 4:16, 4:16] = 1
    targets[1, :, 20:28, 20:28] = 1
    artifacts = render_native_videomt_query_visuals(
        output_root=tmp_path,
        global_step=250,
        input_weight_global_step=249,
        rank=0,
        normalized_padded_rgb=torch.zeros(1, 5, 3, height, width),
        clip_targets=(
            {
                "labels": torch.zeros(2, dtype=torch.long),
                "ids": torch.zeros(2, 5, dtype=torch.long),
                "masks": targets,
                "valid_pixels": torch.ones(5, height, width, dtype=torch.bool),
            },
        ),
        identity_keys=(("left", "right"),),
        source_output=output,
        sample_keys=("episode/frame",),
    )

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["schema"] == NATIVE_VIDEOMT_QUERY_VISUAL_SCHEMA
    assert artifact["query_count"] == 200
    assert artifact["selection_used_by_training"] is False
    assert set(artifact["visible_query_ids"]) == {0, 1}
    assert Path(str(artifact["path"])).is_file()

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from picf_contact_motion_sidecar_precompute import _proposal_from_static_scores
from picf_contact_motion_sidecar_precompute import _save_preview


def test_proposal_from_static_scores_emits_sparse_soft_masks() -> None:
    height, width = 120, 160
    yy, xx = np.mgrid[0:height, 0:width]
    pixels = np.stack([xx.reshape(-1), yy.reshape(-1)], axis=1).astype(np.float32)
    scores = np.zeros((height, width), dtype=np.float32)
    scores[((xx - 40) ** 2 + (yy - 40) ** 2) < 7**2] = 0.95
    scores[((xx - 115) ** 2 + (yy - 80) ** 2) < 8**2] = 0.85

    result = _proposal_from_static_scores(
        pixels_static=pixels,
        scores_static=scores.reshape(-1),
        image_hw=(height, width),
        top_fraction=0.02,
        min_top_points=20,
        min_score=0.1,
        box_pad_px=4,
        max_proposals=3,
        component_radius_px=10,
        component_min_points=6,
        box_percentile_low=12,
        box_percentile_high=88,
        mask_samples_per_proposal=32,
    )

    assert result is not None
    centers, boxes, objectness, mask_xy, mask_weights, mask_offsets, _stats = result
    assert centers.shape[0] >= 2
    assert boxes.shape == (centers.shape[0], 4)
    assert objectness.shape == (centers.shape[0],)
    assert mask_offsets.shape == (centers.shape[0] + 1,)
    assert int(mask_offsets[-1]) == int(mask_xy.shape[0]) == int(mask_weights.shape[0])
    assert np.all((mask_xy >= 0.0) & (mask_xy <= 1.0))
    assert np.all(mask_weights >= 0.0)


def test_preview_accepts_multiple_boxes_and_mask_samples() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        rgb = np.zeros((64, 96, 3), dtype=np.uint8)
        centers = np.asarray([[0.25, 0.25], [0.75, 0.65]], dtype=np.float32)
        boxes = np.asarray([[0.15, 0.15, 0.35, 0.35], [0.62, 0.52, 0.88, 0.82]], dtype=np.float32)
        mask_xy = np.asarray([[0.25, 0.25], [0.26, 0.24], [0.75, 0.65]], dtype=np.float32)
        _save_preview(
            Path(tmp_dir),
            rgb_static=rgb,
            proposal_center=centers,
            proposal_box=boxes,
            proposal_mask_xy=mask_xy,
            step_id=1,
            segment_id=2,
            text="test prompt",
            stats={"objectness": 0.7},
        )
        files = list(Path(tmp_dir).glob("*.png"))
        assert len(files) == 1
        Image.open(files[0]).verify()

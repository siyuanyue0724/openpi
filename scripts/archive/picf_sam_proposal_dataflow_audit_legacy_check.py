from __future__ import annotations

import numpy as np

from scripts.archive import picf_sam_proposal_dataflow_audit_legacy as audit
from scripts.archive import picf_sam_proposal_precompute_legacy as precompute


def test_sam_proposal_arrays_normalize_and_rank_quality() -> None:
    masks = [
        {"bbox": [10, 20, 30, 40], "predicted_iou": 0.9, "stability_score": 0.9},
        {"bbox": [0, 0, 20, 20], "predicted_iou": 0.9, "stability_score": 0.1},
    ]
    payload = precompute._proposal_arrays_from_masks(masks, image_hw=(100, 200), view_id=1, max_proposals=8)
    assert payload["proposal_centers_xy"].shape == (2, 2)
    assert payload["proposal_boxes_xyxy"].shape == (2, 4)
    assert np.all(payload["proposal_boxes_xyxy"] >= 0.0)
    assert np.all(payload["proposal_boxes_xyxy"] <= 1.0)
    assert payload["proposal_view_ids"].tolist() == [1, 1]
    assert payload["proposal_source_ids"].tolist() == [5, 5]
    assert float(payload["proposal_objectness"][0]) > float(payload["proposal_objectness"][1])


def test_sam_proposal_dataflow_audit_without_external_code_passes_core_checks() -> None:
    checks = audit.run_checks(external_code_root=None)
    assert checks
    assert all(check.ok for check in checks)

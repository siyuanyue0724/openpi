from __future__ import annotations

import json
from pathlib import Path

from scripts.picf_owm_same_object_probe import run_overlay_probe
from scripts.picf_owm_same_object_probe import run_probe


def test_same_object_probe_reads_anchor_debug_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "anchor_debug.jsonl"
    rows = []
    for step, offset in ((0, 0.0), (1, 0.01)):
        rows.append(
            {
                "episode": 1,
                "step": step,
                "goal": "probe",
                "anchor_debug": {
                    "observation": {
                        "xyz": [[offset, 0.0, 0.5], [0.4 + offset, 0.0, 0.5]],
                        "pixel": [[10.0 + offset, 10.0], [90.0 + offset, 10.0]],
                        "role_ids": [0, 0],
                        "support_signature": [[1.0, 0.0], [0.0, 1.0]],
                        "binding_signature": [[1.0, 0.0], [0.0, 1.0]],
                    },
                    "mapg": {
                        "visual_priors": {
                            "topk": [
                                [{"index": 1, "weight": 1.0}],
                                [{"index": 8, "weight": 1.0}],
                            ]
                        },
                        "point_priors": {
                            "topk": [
                                [{"index": 11, "weight": 1.0}],
                                [{"index": 18, "weight": 1.0}],
                            ]
                        },
                    },
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    metrics = run_probe(path, pos_xyz_m=0.04, neg_xyz_m=0.12, pos_px=12.0, neg_px=35.0)

    assert metrics["source_format"] == "anchor_debug_jsonl"
    assert metrics["binding_signature_cos_auc"] == 1.0
    assert metrics["pair_examples"]["positive"] == 2
    assert metrics["pair_examples"]["negative"] == 2


def test_same_object_probe_trains_quadratic_binding_probe(tmp_path: Path) -> None:
    path = tmp_path / "anchor_debug.jsonl"
    rows = []
    for step in range(6):
        offset = 0.005 * step
        rows.append(
            {
                "episode": 1,
                "step": step,
                "goal": "probe",
                "anchor_debug": {
                    "observation": {
                        "xyz": [[offset, 0.0, 0.5], [0.45 + offset, 0.0, 0.5], [0.9 + offset, 0.0, 0.5]],
                        "pixel": [[10.0 + offset, 10.0], [80.0 + offset, 10.0], [150.0 + offset, 10.0]],
                        "role_ids": [0, 0, 0],
                        "support_signature": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                        "binding_signature": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                    },
                    "mapg": {
                        "visual_priors": {
                            "topk": [
                                [{"index": 1, "weight": 1.0}],
                                [{"index": 8, "weight": 1.0}],
                                [{"index": 15, "weight": 1.0}],
                            ]
                        },
                        "point_priors": {
                            "topk": [
                                [{"index": 11, "weight": 1.0}],
                                [{"index": 18, "weight": 1.0}],
                                [{"index": 25, "weight": 1.0}],
                            ]
                        },
                    },
                },
            }
        )
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    metrics = run_probe(
        path,
        pos_xyz_m=0.04,
        neg_xyz_m=0.12,
        pos_px=12.0,
        neg_px=35.0,
        quadratic_probe_modes=("diag_quadratic", "low_rank_quadratic", "full_quadratic"),
        quadratic_probe_rank=2,
        quadratic_probe_epochs=80,
        quadratic_probe_lr=0.05,
        quadratic_probe_seed=7,
    )

    assert metrics["binding_signature_diag_quadratic_trained_auc"] == 1.0
    assert metrics["binding_signature_low_rank_quadratic_trained_auc"] == 1.0
    assert metrics["binding_signature_full_quadratic_trained_auc"] == 1.0
    assert metrics["trained_quadratic_probes"]["diag_quadratic"]["status"] == "ok"


def test_same_object_probe_reads_training_anchor_overlays(tmp_path: Path) -> None:
    overlay_dir = tmp_path / "anchor_overlays"
    overlay_dir.mkdir()
    for step, offset in ((100, 0.0), (200, 0.01)):
        payload = {
            "segment_id": 3,
            "step": step,
            "anchors": [
                {
                    "source": "posterior",
                    "index": 0,
                    "role": 0,
                    "world_xyz": [offset, 0.0, 0.5],
                    "pixel_xy": [20.0 + offset, 12.0],
                    "binding_signature": [1.0, 0.0],
                    "support_signature": [1.0, 0.0],
                },
                {
                    "source": "posterior",
                    "index": 1,
                    "role": 0,
                    "world_xyz": [0.4 + offset, 0.0, 0.5],
                    "pixel_xy": [100.0 + offset, 12.0],
                    "binding_signature": [0.0, 1.0],
                    "support_signature": [0.0, 1.0],
                },
            ],
        }
        (overlay_dir / f"step_{step:06d}.json").write_text(json.dumps(payload))

    metrics = run_overlay_probe(
        overlay_dir,
        source="posterior",
        pos_xyz_m=0.04,
        neg_xyz_m=0.12,
        pos_px=12.0,
        neg_px=35.0,
    )

    assert metrics["source_format"] == "anchor_overlay_json:posterior"
    assert metrics["binding_signature_cos_auc"] == 1.0
    assert metrics["pair_examples"]["positive"] == 2
    assert metrics["pair_examples"]["negative"] == 2

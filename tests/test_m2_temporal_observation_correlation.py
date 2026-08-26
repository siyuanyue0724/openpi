from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.audit_m2_temporal_observation_correlation import (
    build_report,
    consecutive_pairs,
    summarize_temporal_correlation,
)


def _row(
    index: int,
    *,
    prediction: float,
    target: float,
    split: str = "train",
    group: int = 0,
    identity: str = "block",
    axis: str = "x",
    arm: str = "actual",
) -> dict[str, object]:
    return {
        "model_arm": arm,
        "split": split,
        "global_index": index,
        "group_kind": "episode",
        "group_index": group,
        "identity_key": identity,
        "axis": axis,
        "predicted_mean_normalized": prediction,
        "target_normalized": target,
        "residual_normalized": prediction - target,
    }


def test_pairing_requires_same_stream_identity_axis_and_adjacent_frame() -> None:
    rows = [
        _row(10, prediction=1.0, target=0.0),
        _row(11, prediction=1.1, target=0.1),
        _row(13, prediction=1.2, target=0.2),
        _row(11, prediction=2.0, target=0.0, identity="drawer"),
        _row(12, prediction=2.1, target=0.1, identity="drawer"),
        _row(12, prediction=9.0, target=0.0, arm="control"),
    ]

    pairs = consecutive_pairs(rows)

    assert [(left["global_index"], right["global_index"]) for left, right in pairs] == [
        (10, 11),
        (11, 12),
    ]


def test_summary_recovers_persistent_error_and_optimal_innovation_gain() -> None:
    # The observation error persists at 1.0 while the target moves by 0.1.
    # Reusing z[t-1] has error 0.9 and adding the current innovation with gain
    # one recovers z[t], whose error remains 1.0. The fitted gain is therefore
    # negative here; the audit reports evidence and does not clip it into a
    # preferred filtering story.
    rows = [
        _row(0, prediction=1.0, target=0.0),
        _row(1, prediction=1.1, target=0.1),
        _row(2, prediction=1.2, target=0.2),
    ]

    summary = summarize_temporal_correlation(rows)
    axis = summary["splits"]["train"]["axes"]["x"]

    assert axis["pair_count"] == 2
    assert axis["current_observation_mse"] == pytest.approx(1.0)
    assert axis["zero_transition_prior_mse"] == pytest.approx(0.81)
    assert axis["target_delta_mse"] == pytest.approx(0.01)
    assert axis["optimal_current_innovation_gain"] == pytest.approx(-9.0)
    assert axis["optimal_linear_fused_mse"] == pytest.approx(0.0, abs=1e-14)


def test_report_validates_residual_and_hashes_exact_source(tmp_path: Path) -> None:
    path = tmp_path / "coordinate_rows.jsonl"
    rows = [
        _row(0, prediction=0.2, target=0.0),
        _row(1, prediction=0.3, target=0.1),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="ascii")

    report = build_report(path)

    assert report["schema"] == "picf.m2-temporal-observation-correlation.v1"
    assert report["source"]["path"] == str(path.resolve())
    assert len(report["source"]["sha256"]) == 64

    rows[1]["residual_normalized"] = 99.0
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="ascii")
    with pytest.raises(ValueError, match="inconsistent residual"):
        build_report(path)

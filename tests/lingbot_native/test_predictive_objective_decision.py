from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.predictive_decision import (
    IMPLEMENTED_PREDICTIVE_OBJECTIVE,
    IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
    PREDICTIVE_OBJECTIVE_POSTERIOR_FUTURE_IDENTITY,
    load_predictive_objective_decision,
    validate_predictive_objective_decision,
)
from tools.build_lingbot_predictive_objective_decision import (
    build_predictive_objective_decision,
)


def _decision(tmp_path: Path) -> tuple[Path, dict[str, object]]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    record = tmp_path / "ADR-82.md"
    record.write_text("owner-reviewed predictive objective")
    value = build_predictive_objective_decision(
        reviewer="owner-review",
        temporal_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
        visible_support_weighting=IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
        minimum_visible_fraction=0.015625,
        decision_record=record,
    )
    path = tmp_path / "predictive-objective.json"
    path.write_text(json.dumps(value, sort_keys=True))
    return path, value


def test_predictive_objective_decision_binds_semantics_support_and_record(
    tmp_path: Path,
) -> None:
    path, value = _decision(tmp_path)
    payload = path.read_bytes()
    decision = load_predictive_objective_decision(
        path,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_temporal_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
        expected_visible_support_weighting=IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
        expected_minimum_visible_fraction=0.015625,
    )
    assert decision.temporal_objective == IMPLEMENTED_PREDICTIVE_OBJECTIVE
    assert decision.minimum_visible_fraction == 0.015625
    assert validate_predictive_objective_decision(value) == decision


def test_predictive_objective_decision_rejects_unimplemented_or_tampered_semantics(
    tmp_path: Path,
) -> None:
    path, value = _decision(tmp_path)
    with pytest.raises(ValueError, match="another temporal objective"):
        load_predictive_objective_decision(
            path,
            expected_temporal_objective=PREDICTIVE_OBJECTIVE_POSTERIOR_FUTURE_IDENTITY,
        )

    value["claim_scope"] = "controlled-future-dynamics/v1"
    with pytest.raises(ValueError, match="claim scope"):
        validate_predictive_objective_decision(value)


def test_predictive_objective_decision_rejects_record_or_threshold_tampering(
    tmp_path: Path,
) -> None:
    path, value = _decision(tmp_path)
    decision_record = Path(value["decision_record"]["path"])  # type: ignore[index]
    decision_record.write_text("edited after approval")
    with pytest.raises(ValueError, match="reviewed content"):
        load_predictive_objective_decision(path)

    _path, threshold_value = _decision(tmp_path / "threshold")
    threshold_value["visible_support"]["minimum_visible_fraction_hex"] = "0x1p-2"  # type: ignore[index]
    with pytest.raises(ValueError, match="canonical"):
        validate_predictive_objective_decision(threshold_value)


@pytest.mark.parametrize("minimum", [False, float("nan"), -0.01, 1.0])
def test_predictive_objective_decision_builder_rejects_invalid_threshold_types(
    tmp_path: Path,
    minimum: object,
) -> None:
    record = tmp_path / "ADR-82.md"
    record.write_text("owner-reviewed predictive objective")
    with pytest.raises(ValueError, match=r"\[0,1\)"):
        build_predictive_objective_decision(
            reviewer="owner-review",
            temporal_objective=IMPLEMENTED_PREDICTIVE_OBJECTIVE,
            visible_support_weighting=IMPLEMENTED_PREDICTIVE_VISIBLE_SUPPORT_WEIGHTING,
            minimum_visible_fraction=minimum,  # type: ignore[arg-type]
            decision_record=record,
        )

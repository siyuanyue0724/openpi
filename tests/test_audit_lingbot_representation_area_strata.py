from __future__ import annotations

from copy import deepcopy

import pytest

from picf_next.contracts import ContractError
from tools.audit_lingbot_representation_area_strata import build_area_strata_audit


def _sample(key: str, area: float, ownership: float, margin: float, rank_one: bool) -> dict:
    return {
        "factual_ownership_summary": {"target_soft_iou": ownership},
        "factual_target_identity_keys": [f"object/{key}"],
        "factual_task_instruction_sha256": key * 64,
        "factual_task_row_diagnostic": {
            "all_targets_beat_known_negatives": rank_one,
            "target_vs_hardest_negative_logit_margin": margin * 10,
            "target_vs_hardest_negative_probability_margin": margin,
        },
        "factual_token_evidence": {"metrics": {"eligible": True, "target_area_fraction": area}},
        "partition": "heldout",
        "sample_key": key,
        "task_key": f"task/{key}",
    }


def _snapshot(samples: list[dict]) -> dict:
    return {"artifact_sha256": "a" * 64, "samples": samples, "status": "PASS"}


def test_area_strata_audit_preserves_small_object_failures() -> None:
    baseline = _snapshot(
        [
            _sample("a", 0.01, 0.01, -0.4, False),
            _sample("b", 0.03, 0.10, -0.2, False),
            _sample("c", 0.10, 0.40, 0.2, True),
        ]
    )
    candidate = deepcopy(baseline)
    candidate["artifact_sha256"] = "b" * 64
    candidate["samples"][0]["factual_ownership_summary"]["target_soft_iou"] = 0.05
    candidate["samples"][0]["factual_task_row_diagnostic"][
        "target_vs_hardest_negative_logit_margin"
    ] = -1.0
    candidate["samples"][0]["factual_task_row_diagnostic"][
        "target_vs_hardest_negative_probability_margin"
    ] = -0.1
    candidate["samples"][0]["factual_task_row_diagnostic"]["all_targets_beat_known_negatives"] = (
        True
    )

    report = build_area_strata_audit(baseline, candidate, partition="heldout")

    assert report["diagnostic_only"] is True
    assert report["eligible_pair_count"] == 3
    small = report["strata"]["lt_2_percent"]
    assert small["count"] == 1
    assert small["metrics"]["ownership"]["mean_delta"] == pytest.approx(0.04)
    assert small["metrics"]["margin_logit"]["mean_delta"] == pytest.approx(3.0)
    assert small["metrics"]["margin_probability"]["mean_delta"] == pytest.approx(0.3)
    assert small["metrics"]["rank_one_rate"]["candidate_mean"] == 1.0


def test_area_strata_audit_rejects_changed_targets() -> None:
    baseline = _snapshot(
        [
            _sample("a", 0.01, 0.01, -0.4, False),
            _sample("b", 0.03, 0.10, -0.2, False),
            _sample("c", 0.10, 0.40, 0.2, True),
        ]
    )
    candidate = deepcopy(baseline)
    candidate["samples"][0]["factual_token_evidence"]["metrics"]["target_area_fraction"] = 0.011

    with pytest.raises(ContractError, match="paired target area differs"):
        build_area_strata_audit(baseline, candidate, partition="heldout")


def test_area_strata_audit_rejects_missing_pairs() -> None:
    samples = [
        _sample("a", 0.01, 0.01, -0.4, False),
        _sample("b", 0.03, 0.10, -0.2, False),
        _sample("c", 0.10, 0.40, 0.2, True),
    ]
    baseline = _snapshot(samples)
    candidate = _snapshot(deepcopy(samples[:-1]))

    with pytest.raises(ContractError, match="sample-key sets differ"):
        build_area_strata_audit(baseline, candidate, partition="heldout")


def test_area_strata_audit_ignores_ineligible_null_diagnostics() -> None:
    samples = [
        _sample("a", 0.01, 0.01, -0.4, False),
        _sample("b", 0.03, 0.10, -0.2, False),
        _sample("c", 0.10, 0.40, 0.2, True),
        _sample("d", 0.0, 0.0, 0.0, False),
    ]
    samples[-1]["factual_token_evidence"]["metrics"]["eligible"] = False
    samples[-1]["factual_ownership_summary"]["target_soft_iou"] = None
    samples[-1]["factual_task_row_diagnostic"]["all_targets_beat_known_negatives"] = None
    samples[-1]["factual_task_row_diagnostic"]["target_vs_hardest_negative_logit_margin"] = None
    samples[-1]["factual_task_row_diagnostic"]["target_vs_hardest_negative_probability_margin"] = (
        None
    )

    report = build_area_strata_audit(
        _snapshot(samples),
        _snapshot(deepcopy(samples)),
        partition="heldout",
    )

    assert report["eligible_pair_count"] == 3

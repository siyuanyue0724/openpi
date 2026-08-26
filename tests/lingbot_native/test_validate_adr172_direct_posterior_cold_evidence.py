from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.validate_adr172_direct_posterior_cold_evidence import (
    DIRECT_ACTION_SURFACE,
    REPORT_SCHEMA,
    ValidationInputError,
    validate_adr172_direct_posterior_cold_evidence,
)


def _source_contract(commit: str) -> dict[str, object]:
    return {
        "schema": "picf-next.g3-picf-source-contract.v1",
        "repository_commit": commit * 40,
        "repository_tree": "2" * 40,
        "worktree_clean": True,
        "critical_file_sha256": {"runner.py": "3" * 64},
    }


def _arm_receipts() -> list[dict[str, object]]:
    return [
        {
            "arm_name": "factual",
            "arm_kind": "factual",
            "row_index": None,
            "source_visibility_sha256": "1" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "a" * 64,
        },
        {
            "arm_name": "factual-repeat",
            "arm_kind": "factual",
            "row_index": None,
            "source_visibility_sha256": "1" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "a" * 64,
        },
        {
            "arm_name": "remove-row-0",
            "arm_kind": "row-removal",
            "row_index": 0,
            "source_visibility_sha256": "2" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "b" * 64,
        },
        {
            "arm_name": "remove-row-1",
            "arm_kind": "row-removal",
            "row_index": 1,
            "source_visibility_sha256": "3" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "c" * 64,
        },
        {
            "arm_name": "blocked",
            "arm_kind": "blocked",
            "row_index": None,
            "source_visibility_sha256": "4" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "d" * 64,
        },
        {
            "arm_name": "blocked-remove-row-0",
            "arm_kind": "blocked-row-removal",
            "row_index": 0,
            "source_visibility_sha256": "4" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "d" * 64,
        },
        {
            "arm_name": "blocked-remove-row-1",
            "arm_kind": "blocked-row-removal",
            "row_index": 1,
            "source_visibility_sha256": "4" * 64,
            "active_action_mask_sha256": "5" * 64,
            "action_output_sha256": "d" * 64,
        },
    ]


def _prompt(
    *,
    target: str,
    distractor: str,
    target_row: int,
    distractor_row: int,
) -> dict[str, object]:
    return {
        "prompt_name": f"{target}-prompt",
        "target_identity": target,
        "matched_distractor_identity": distractor,
        "target_row": target_row,
        "matched_distractor_row": distractor_row,
        "bindings": [["object-a", 0], ["object-b", 1]],
        "independent_bindings": [["object-a", 0], ["object-b", 1]],
        "arm_receipts": _arm_receipts(),
        "score": {
            "prompt_name": f"{target}-prompt",
            "sample_keys": ["episode-key"],
            "active_action_counts": [7],
            "blocked_placebo_integrity_verified": True,
            "replay_floor_rms": [0.0],
            "factual_all_posterior_block_effect_rms": [0.4],
            "factual_target_effect_rms": [0.2],
            "factual_distractor_effect_rms": [0.05],
            "factual_target_minus_distractor": [0.15],
            "factual_target_effect_over_all_posterior_block": [0.5],
            "factual_distractor_effect_over_all_posterior_block": [0.125],
            "factual_selectivity_over_all_posterior_block": [0.375],
            "mean_factual_all_posterior_block_effect_rms": 0.4,
            "mean_factual_target_minus_distractor": 0.15,
            "mean_factual_selectivity_over_all_posterior_block": 0.375,
        },
    }


def _scene(*, partition: str, rank: int, index: int) -> dict[str, object]:
    return {
        "item_id": f"{partition}-{rank}-{index}",
        "sample_key": f"sample-{partition}-{rank}-{index}",
        "prompt_count": 2,
        "target_identities": ["object-a", "object-b"],
        "canonical_bindings": [["object-a", 0], ["object-b", 1]],
        "independent_bindings_by_prompt": [
            [["object-a", 0], ["object-b", 1]],
            [["object-a", 0], ["object-b", 1]],
        ],
        "shared_row_gauge": True,
        "physical_prompt_drift_max_abs": 0.0,
        "prompts": [
            _prompt(
                target="object-a",
                distractor="object-b",
                target_row=0,
                distractor_row=1,
            ),
            _prompt(
                target="object-b",
                distractor="object-a",
                target_row=1,
                distractor_row=0,
            ),
        ],
        "score": {
            "sample_keys": ["episode-key"],
            "active_action_counts": [7],
            "blocked_placebo_integrity_verified": True,
            "replay_floor_rms": [0.0, 0.0],
            "max_replay_floor_rms": 0.0,
            "prompt_mean_factual_all_posterior_block_effect_rms": [0.4, 0.4],
            "minimum_prompt_factual_all_posterior_block_effect_rms": 0.4,
            "crossed_prompt_target_selectivity": [0.3],
            "crossed_prompt_selectivity_over_all_posterior_block": [0.375],
            "mean_crossed_prompt_target_selectivity": 0.3,
            "mean_crossed_prompt_selectivity_over_all_posterior_block": 0.375,
            "positive_crossed_prompt_target_selectivity_count": 1,
            "sample_count": 1,
        },
    }


def _summary(partition: str) -> dict[str, object]:
    return {
        "partition": partition,
        "status": "PASS",
        "failures": [],
        "scene_count": 8,
        "positive_scene_fraction_minimum": 0.75,
        "minimum_positive_scene_count": 6,
        "positive_crossed_prompt_scene_count": 8,
        "positive_normalized_crossed_prompt_scene_count": 8,
        "positive_all_posterior_block_scene_count": 8,
        "joint_positive_scene_count": 8,
        "mean_crossed_prompt_target_selectivity": 0.3,
        "mean_crossed_prompt_selectivity_over_all_posterior_block": 0.375,
        "mean_minimum_prompt_factual_all_posterior_block_effect_rms": 0.4,
        "max_replay_floor_rms": 0.0,
        "scenes": [],
    }


def _report() -> dict[str, object]:
    rank_reports = []
    for rank in range(2):
        history = {}
        for partition in ("validation", "heldout"):
            scenes = [_scene(partition=partition, rank=rank, index=index) for index in range(4)]
            history[partition] = {
                "scene_count": 4,
                "prompt_count": 8,
                "max_replay_floor_rms": 0.0,
                "scenes": scenes,
            }
        rank_reports.append(
            {
                "rank": rank,
                "direct_action_causal_surface": DIRECT_ACTION_SURFACE,
                "history": [history],
            }
        )
    return {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "phase": "evaluation",
        "world_size": 2,
        "capacity": 2,
        "direct_action_causal_surface": DIRECT_ACTION_SURFACE,
        "thresholds": {
            "bitwise_factual_replay": True,
            "blocked_row_placebo_bitwise_equality": True,
            "shared_canonical_row_gauge": True,
            "mean_crossed_prompt_target_selectivity_strictly_positive": True,
            "mean_normalized_crossed_prompt_selectivity_strictly_positive": True,
            "mean_all_posterior_block_effect_strictly_positive": True,
            "joint_positive_scene_requires_normalized_selectivity": True,
            "joint_positive_scene_fraction_minimum": 0.75,
        },
        "causal_adoption_contract": {"exclusive_visual_path_claim": False},
        "action_inference_contract": {"active_action_surface": "joint_mask AND NOT action_is_pad"},
        "picf_source_contract": _source_contract("1"),
        "trained_picf_source_contract": _source_contract("9"),
        "cold_causal_summary": {
            "validation": _summary("validation"),
            "heldout": _summary("heldout"),
        },
        "rank_reports": rank_reports,
    }


def _write_report(tmp_path: Path, report: dict[str, object]) -> Path:
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="ascii")
    return path


def _replace_active_action_counts(value: object, replacement: object) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "active_action_counts":
                value[key] = [replacement]
            else:
                _replace_active_action_counts(item, replacement)
    elif isinstance(value, list):
        for item in value:
            _replace_active_action_counts(item, replacement)


def test_adr172_independent_validator_accepts_recomputed_causal_report(tmp_path: Path) -> None:
    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, _report()))

    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert result["partitions"]["validation"]["joint_positive_scene_count"] == 8


def test_adr172_independent_validator_accepts_exact_integral_legacy_counts(
    tmp_path: Path,
) -> None:
    report = _report()
    _replace_active_action_counts(report, 7.0)

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "PASS"


@pytest.mark.parametrize("invalid_count", [7.5, True])
def test_adr172_independent_validator_rejects_nondiscrete_counts(
    tmp_path: Path,
    invalid_count: object,
) -> None:
    report = _report()
    _replace_active_action_counts(report, invalid_count)

    with pytest.raises(ValidationInputError, match="must be an integer count"):
        validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))


def test_adr172_independent_validator_rejects_blocked_placebo_hash_drift(
    tmp_path: Path,
) -> None:
    report = _report()
    report["rank_reports"][0]["history"][0]["validation"]["scenes"][0]["prompts"][0][
        "arm_receipts"
    ][5]["action_output_sha256"] = "e" * 64

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "FAIL"
    assert any("blocked placebo action hash differs" in value for value in result["failures"])


def test_adr172_independent_validator_rejects_aggregate_tampering(tmp_path: Path) -> None:
    report = _report()
    report["cold_causal_summary"]["validation"]["mean_crossed_prompt_target_selectivity"] = 9.0

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "FAIL"
    assert any(
        "serialized mean_crossed_prompt_target_selectivity" in value for value in result["failures"]
    )


def test_adr172_independent_validator_rejects_float32_significant_tampering(
    tmp_path: Path,
) -> None:
    report = _report()
    report["rank_reports"][0]["history"][0]["validation"]["scenes"][0]["prompts"][0]["score"][
        "factual_target_effect_over_all_posterior_block"
    ][0] += 1.0e-4

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "FAIL"
    assert any(
        "serialized factual_target_effect_over_all_posterior_block" in value
        for value in result["failures"]
    )


def test_adr172_independent_validator_rejects_row_gauge_drift(tmp_path: Path) -> None:
    report = _report()
    report["rank_reports"][0]["history"][0]["validation"]["scenes"][0][
        "independent_bindings_by_prompt"
    ][1] = [["object-a", 1], ["object-b", 0]]

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "FAIL"
    assert any("physical row gauge changed" in value for value in result["failures"])


def test_adr172_independent_validator_rejects_stale_fraction(tmp_path: Path) -> None:
    report = _report()
    report["thresholds"]["joint_positive_scene_fraction_minimum"] = 0.625

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "FAIL"
    assert any("registered 0.75 gate" in value for value in result["failures"])


def test_adr172_independent_validator_requires_normalized_scene_support(
    tmp_path: Path,
) -> None:
    report = _report()
    report["thresholds"]["joint_positive_scene_requires_normalized_selectivity"] = False

    result = validate_adr172_direct_posterior_cold_evidence(_write_report(tmp_path, report))

    assert result["status"] == "FAIL"
    assert any("omitted normalized selectivity" in value for value in result["failures"])

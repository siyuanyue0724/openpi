from __future__ import annotations

import copy
import hashlib
import json
import math
import sys
from collections.abc import Callable
from typing import Any

import pytest

from picf_next.contracts import ContractError
from tools.compare_lingbot_vla2_ltop_g3_retention_reports import (
    G2_REPRESENTATION_SCHEMA,
    G3_RETENTION_SCHEMA,
    OUTPUT_SCHEMA,
    compare_lingbot_vla2_ltop_g3_retention_reports,
    main,
)

CAPACITY = 16


def _digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _scene(
    item_id: str,
    *,
    row_by_identity: dict[str, int],
    prompt_coverages: tuple[tuple[float, float], tuple[float, float]],
    physical_set_loss: float,
) -> dict[str, Any]:
    identities = [f"{item_id}/first", f"{item_id}/second"]
    bindings = [[identity, row] for identity, row in sorted(row_by_identity.items())]
    prompts = []
    for index, (target_coverage, alternate_coverage) in enumerate(prompt_coverages):
        target_row = row_by_identity[identities[index]]
        alternate_row = row_by_identity[identities[1 - index]]
        distribution = [0.01] * CAPACITY
        distribution[target_row] = 0.60
        distribution[alternate_row] = 0.20
        prompts.append(
            {
                "alternate_coverage": alternate_coverage,
                "alternate_row": alternate_row,
                "margin": target_coverage - alternate_coverage,
                "mean_row_distribution": distribution,
                "target_coverage": target_coverage,
                "target_row": target_row,
                "top_row": target_row,
            }
        )
    margins = [float(prompt["margin"]) for prompt in prompts]
    return {
        "bindings_by_prompt": [copy.deepcopy(bindings), copy.deepcopy(bindings)],
        "independent_bindings_by_prompt": [copy.deepcopy(bindings), copy.deepcopy(bindings)],
        "item_id": item_id,
        "mean_margin": sum(margins) / len(margins),
        "mean_physical_set_loss": physical_set_loss,
        "mean_target_nll": sum(
            -math.log(max(float(prompt["target_coverage"]), 1.0e-30)) for prompt in prompts
        )
        / len(prompts),
        "metric_self_checks": {"matched_row_permutation_max_abs_error": 0.0},
        "physical_prompt_drift_max_abs": 0.0,
        "positive_margin_count": sum(margin > 0.0 for margin in margins),
        "prompt_distribution_cosine": 0.9,
        "prompt_distribution_mean_l1": 0.1,
        "prompts": prompts,
        "sample_key": f"sample/{item_id}",
        "shared_row_gauge": True,
        "target_identities": identities,
        "target_rows": [row_by_identity[identity] for identity in identities],
    }


def _partition(scenes: list[dict[str, Any]]) -> dict[str, Any]:
    prompts = [prompt for scene in scenes for prompt in scene["prompts"]]
    return {
        "mean_margin": sum(float(prompt["margin"]) for prompt in prompts) / len(prompts),
        "mean_physical_set_loss": sum(float(scene["mean_physical_set_loss"]) for scene in scenes)
        / len(scenes),
        "mean_target_nll": sum(float(scene["mean_target_nll"]) for scene in scenes) / len(scenes),
        "metric_self_checks": {"matched_row_permutation_max_abs_error": 0.0},
        "physical_prompt_drift_max_abs": 0.0,
        "positive_margin_count": sum(float(prompt["margin"]) > 0.0 for prompt in prompts),
        "prompt_count": len(prompts),
        "prompts": copy.deepcopy(prompts),
        "scene_count": len(scenes),
        "scenes": scenes,
        "shared_row_gauge": True,
    }


def _reports(*, g3_runner_status: str = "FAIL") -> tuple[dict[str, Any], dict[str, Any], str]:
    g2_ranks = []
    g3_ranks = []
    for rank in range(2):
        g2_partitions: dict[str, Any] = {}
        g3_partitions: dict[str, Any] = {}
        local_items: dict[str, Any] = {}
        for partition in ("validation", "heldout"):
            g2_scenes = []
            g3_scenes = []
            for scene_index in range(4):
                item_id = f"{partition}-r{rank}-s{scene_index}"
                first = f"{item_id}/first"
                second = f"{item_id}/second"
                context = f"{item_id}/context"
                g2_scenes.append(
                    _scene(
                        item_id,
                        row_by_identity={first: 0, second: 3, context: 7},
                        prompt_coverages=((0.70, 0.20), (0.64, 0.24)),
                        physical_set_loss=0.40 + 0.01 * scene_index,
                    )
                )
                g3_scenes.append(
                    _scene(
                        item_id,
                        row_by_identity={first: 5, second: 1, context: 9},
                        prompt_coverages=((0.72, 0.18), (0.67, 0.21)),
                        physical_set_loss=0.36 + 0.01 * scene_index,
                    )
                )
            g2_partitions[partition] = _partition(g2_scenes)
            g3_partitions[partition] = _partition(g3_scenes)
            local_items[partition] = [
                {
                    "item_id": scene["item_id"],
                    "sample_key": scene["sample_key"],
                    "target_identities": scene["target_identities"],
                }
                for scene in g2_scenes
            ]
        g2_ranks.append(
            {
                "history": [{"step": 128, **g2_partitions}],
                "local_items": local_items,
                "rank": rank,
            }
        )
        g3_ranks.append({"history": [{"step": 128, **g3_partitions}], "rank": rank})

    common = {
        "architecture_identity": "lingbot_task_query_object_value_read_v1",
        "capacity": CAPACITY,
        "dataset_contract": {"schema": "frozen-calvin-v1", "sha256": "d" * 64},
        "task_query_count": 4,
        "world_size": 2,
    }
    g2 = {
        **common,
        "failures": [],
        "input_sha256": {
            "dataset_manifest": "4" * 64,
            "execution_contract": "1" * 64,
            "normalization": "5" * 64,
            "offline_labels": "2" * 64,
            "physical_sidecar_manifest": "3" * 64,
        },
        "rank_reports": g2_ranks,
        "schema": G2_REPRESENTATION_SCHEMA,
        "status": "PASS",
        "steps": 128,
        "training_scope": "representation",
    }
    g2_digest = _digest(g2)
    g3 = {
        **common,
        "execution_contract_sha256": "1" * 64,
        "failures": [] if g3_runner_status == "PASS" else ["absolute runner floor failed"],
        "g2_report_sha256": g2_digest,
        "offline_labels_sha256": "2" * 64,
        "phase": "retention",
        "physical_sidecar_manifest_sha256": "3" * 64,
        "rank_reports": g3_ranks,
        "representation_retention_contract": {
            "crossed_prompts_per_scene": 2,
            "optimizer_updates": 0,
            "reference": "accepted G2b full-scene representation gate",
            "scenes_per_rank_per_partition": 4,
            "scientific_action_evidence": False,
        },
        "schema": G3_RETENTION_SCHEMA,
        "status": g3_runner_status,
        "steps": 128,
    }
    return g2, g3, g2_digest


def _compare(
    g2: dict[str, Any],
    g3: dict[str, Any],
    g2_digest: str,
) -> dict[str, Any]:
    return compare_lingbot_vla2_ltop_g3_retention_reports(
        g2,
        g3,
        g2_report_sha256=g2_digest,
        g3_retention_report_sha256=_digest(g3),
        bootstrap_seed=17,
        bootstrap_samples=200,
    )


def test_comparison_pairs_full_scene_axis_and_aligns_permuted_gauges() -> None:
    g2, g3, g2_digest = _reports(g3_runner_status="FAIL")

    report = _compare(g2, g3, g2_digest)

    assert report["schema"] == OUTPUT_SCHEMA
    assert report["comparison_status"] == "COMPLETE"
    assert report["input_reports"]["g3_retention"]["runner_status"] == "FAIL"
    assert report["decision"]["scientific_conclusion"] == ("NOT_AUTHORIZED_BY_COMPARISON_ALONE")
    assert report["decision"]["default_positive_prompt_count_nonregression_satisfied"]
    validation = report["partitions"]["validation"]
    assert validation["scene_count"] == 8
    assert validation["prompt_count"] == 16
    first_scene = validation["scenes"][0]
    assert first_scene["raw_target_rows_equal"] is False
    assert first_scene["gauge_permutation_g2_row_to_g3_row"] == {"0": 5, "3": 1, "7": 9}
    assert first_scene["mean_margin_delta_g3_minus_g2"] > 0.0
    assert first_scene["mean_target_nll_delta_g3_minus_g2"] < 0.0
    assert first_scene["mean_physical_set_loss_delta_g3_minus_g2"] < 0.0
    assert validation["prompts"][0]["physical_set_loss_note"].startswith(
        "source schemas expose only the scene mean"
    )


def test_runner_pass_is_recorded_but_never_promoted_to_scientific_pass() -> None:
    g2, g3, g2_digest = _reports(g3_runner_status="PASS")

    report = _compare(g2, g3, g2_digest)

    assert report["input_reports"]["g3_retention"]["runner_status"] == "PASS"
    assert report["decision"]["runner_status_is_not_scientific_conclusion"] is True
    assert "scientific_status" not in report
    assert "PASS" not in {
        report["comparison_status"],
        report["decision"]["scientific_conclusion"],
    }


def test_default_count_gate_uses_observed_g2_counts_without_fallback() -> None:
    g2, g3, g2_digest = _reports()
    scene = g3["rank_reports"][0]["history"][0]["validation"]["scenes"][0]
    scene["prompts"][0]["target_coverage"] = 0.10
    scene["prompts"][0]["alternate_coverage"] = 0.30
    scene["prompts"][0]["margin"] = -0.20
    scene["mean_margin"] = sum(prompt["margin"] for prompt in scene["prompts"]) / 2
    scene["positive_margin_count"] = 1
    scene["mean_target_nll"] = (
        sum(-math.log(max(prompt["target_coverage"], 1.0e-30)) for prompt in scene["prompts"]) / 2
    )
    partition = g3["rank_reports"][0]["history"][0]["validation"]
    partition["prompts"] = [
        prompt for current_scene in partition["scenes"] for prompt in current_scene["prompts"]
    ]
    partition["mean_margin"] = sum(prompt["margin"] for prompt in partition["prompts"]) / 8
    partition["positive_margin_count"] = sum(
        prompt["margin"] > 0 for prompt in partition["prompts"]
    )
    partition["mean_target_nll"] = (
        sum(current_scene["mean_target_nll"] for current_scene in partition["scenes"]) / 4
    )

    report = _compare(g2, g3, g2_digest)

    gate = report["partitions"]["validation"]["positive_prompt_count_nonregression"]
    assert gate == {
        "g2_observed_positive_prompt_count": 16,
        "g3_retention_observed_positive_prompt_count": 15,
        "maximum_allowed_regression": 0,
        "required_g3_minimum": 16,
        "satisfied": False,
    }
    assert report["comparison_status"] == "COMPLETE"
    assert report["decision"]["default_positive_prompt_count_nonregression_satisfied"] is False
    assert len(report["partitions"]["validation"]["prompts"]) == 16


def test_scene_bootstrap_is_deterministic() -> None:
    g2, g3, g2_digest = _reports()

    first = _compare(g2, g3, g2_digest)
    second = _compare(g2, g3, g2_digest)

    assert (
        first["partitions"]["validation"]["bootstrap"]
        == second["partitions"]["validation"]["bootstrap"]
    )
    assert (
        first["partitions"]["heldout"]["bootstrap"] == second["partitions"]["heldout"]["bootstrap"]
    )


Mutation = Callable[[dict[str, Any], dict[str, Any]], None]


def _change_sample_key(_g2: dict[str, Any], g3: dict[str, Any]) -> None:
    g3["rank_reports"][0]["history"][0]["validation"]["scenes"][0]["sample_key"] = "wrong/sample"


def _change_prompt_identity(_g2: dict[str, Any], g3: dict[str, Any]) -> None:
    scene = g3["rank_reports"][0]["history"][0]["validation"]["scenes"][0]
    scene["target_identities"] = list(reversed(scene["target_identities"]))
    scene["target_rows"] = list(reversed(scene["target_rows"]))
    scene["prompts"] = list(reversed(scene["prompts"]))
    partition = g3["rank_reports"][0]["history"][0]["validation"]
    partition["prompts"] = [
        prompt for current_scene in partition["scenes"] for prompt in current_scene["prompts"]
    ]


def _break_shared_gauge(_g2: dict[str, Any], g3: dict[str, Any]) -> None:
    g3["rank_reports"][0]["history"][0]["validation"]["scenes"][0]["shared_row_gauge"] = False


def _change_identity_set(_g2: dict[str, Any], g3: dict[str, Any]) -> None:
    scene = g3["rank_reports"][0]["history"][0]["validation"]["scenes"][0]
    for field in ("bindings_by_prompt", "independent_bindings_by_prompt"):
        for bindings in scene[field]:
            context = next(pair for pair in bindings if pair[0].endswith("/context"))
            context[0] = "replacement/context"


def _change_execution_contract(_g2: dict[str, Any], g3: dict[str, Any]) -> None:
    g3["execution_contract_sha256"] = "f" * 64


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_change_sample_key, "sample_key changed"),
        (_change_prompt_identity, "prompt identity/order changed"),
        (_break_shared_gauge, "shared_row_gauge is not true"),
        (_change_identity_set, "physical identity set changed"),
        (_change_execution_contract, "frozen execution_contract changed"),
    ],
)
def test_strict_alignment_rejects_unpaired_reports(mutation: Mutation, message: str) -> None:
    g2, g3, g2_digest = _reports()
    mutation(g2, g3)

    with pytest.raises(ContractError, match=message):
        _compare(g2, g3, g2_digest)


def test_exact_g2_file_digest_binding_is_required() -> None:
    g2, g3, g2_digest = _reports()
    g3["g2_report_sha256"] = "0" * 64

    with pytest.raises(ContractError, match="exact G2 report"):
        _compare(g2, g3, g2_digest)


def test_inconsistent_reported_metric_is_rejected() -> None:
    g2, g3, g2_digest = _reports()
    g3["rank_reports"][0]["history"][0]["heldout"]["scenes"][0]["mean_target_nll"] += 0.1

    with pytest.raises(ContractError, match="mean_target_nll is inconsistent"):
        _compare(g2, g3, g2_digest)


def test_cli_binds_exact_file_bytes_and_publishes_immutable_report(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    g2, g3, _g2_digest = _reports()
    g2_payload = json.dumps(g2, sort_keys=True, separators=(",", ":")).encode()
    g3["g2_report_sha256"] = hashlib.sha256(g2_payload).hexdigest()
    g3_payload = json.dumps(g3, sort_keys=True, separators=(",", ":")).encode()
    g2_path = tmp_path / "g2.json"
    g3_path = tmp_path / "g3.json"
    output = tmp_path / "comparison.json"
    g2_path.write_bytes(g2_payload)
    g3_path.write_bytes(g3_payload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_ltop_retention",
            "--g2-report",
            str(g2_path),
            "--g3-retention-report",
            str(g3_path),
            "--output",
            str(output),
            "--bootstrap-samples",
            "100",
        ],
    )

    main()

    published = json.loads(output.read_text())
    assert published["schema"] == OUTPUT_SCHEMA
    assert published["input_reports"]["g2"]["sha256"] == hashlib.sha256(g2_payload).hexdigest()
    assert (
        published["input_reports"]["g3_retention"]["sha256"]
        == hashlib.sha256(g3_payload).hexdigest()
    )
    assert json.loads(capsys.readouterr().out)["comparison_status"] == "COMPLETE"
    with pytest.raises(FileExistsError):
        main()

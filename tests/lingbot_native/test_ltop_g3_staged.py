from __future__ import annotations

import json
from pathlib import Path

import pytest

from picf_next.lingbot_native.ltop_g3_staged import compose_staged_g3


def _common() -> dict[str, object]:
    return {
        "status": "PASS",
        "failures": [],
        "mode": "gate",
        "architecture_identity": "lingbot",
        "runtime_source_contract": {
            "source_commit": "1" * 40,
            "native_patch_sha256": "2" * 64,
            "runtime_hotfix_sha256": "3" * 64,
        },
        "world_size": 2,
        "steps": 128,
        "eval_every": 32,
        "seed": 7,
        "capacity": 16,
        "task_query_count": 4,
        "stage_checkpoint": "/mnt/g2",
        "g2_report_sha256": "a" * 64,
        "dataset_contract": {"tree": "b" * 64},
        "execution_contract_sha256": "c" * 64,
        "offline_labels_sha256": "d" * 64,
        "physical_sidecar_manifest_sha256": "e" * 64,
    }


def _score() -> dict[str, object]:
    return {
        "sample_keys": ["sample"],
        "mean_factual_target_minus_distractor": 1.0,
        "mean_blocked_path_difference_in_differences": 1.0,
        "positive_factual_count": 1,
        "positive_blocked_path_did_count": 1,
    }


def _history() -> list[dict[str, object]]:
    entry: dict[str, object] = {"step": 128}
    for partition in ("validation", "heldout"):
        entry[partition] = {
            "max_replay_floor_rms": 0.0,
            "scenes": [{"score": _score()}],
        }
    return [entry]


def _write_reports(tmp_path: Path) -> tuple[Path, Path]:
    checkpoint = str(tmp_path / "checkpoint")
    training = {
        **_common(),
        "schema": "picf-next.ltop-g3-training-phase.v1",
        "phase": "training",
        "checkpoint": {"path": checkpoint, "optimizer_saved": False},
        "training_contract": {"optimizer": "released"},
        "rank_reports": [
            {
                "rank": rank,
                "runtime_schedule_sha256": "f" * 64,
                "action_losses": [1.0] * 112 + [0.8] * 16,
                "history": [],
            }
            for rank in (0, 1)
        ],
    }
    evaluation = {
        **_common(),
        "schema": "picf-next.ltop-g3-evaluation-phase.v1",
        "phase": "evaluation",
        "trained_checkpoint": checkpoint,
        "action_inference_contract": {"surface": "policy.sample_actions"},
        "thresholds": {"positive_sample_fraction_minimum": 0.625},
        "rank_reports": [
            {
                "rank": rank,
                "runtime_schedule_sha256": "f" * 64,
                "history": _history(),
                "cuda_memory_bytes": {"peak_allocated": 1},
            }
            for rank in (0, 1)
        ],
    }
    training_path = tmp_path / "training.json"
    evaluation_path = tmp_path / "evaluation.json"
    training_path.write_text(json.dumps(training), encoding="ascii")
    evaluation_path.write_text(json.dumps(evaluation), encoding="ascii")
    return training_path, evaluation_path


def test_compose_staged_g3_preserves_registered_final_abi(tmp_path: Path) -> None:
    training, evaluation = _write_reports(tmp_path)

    report = compose_staged_g3(training_path=training, evaluation_path=evaluation)

    assert report["schema"] == "picf-next.ltop-g3-production-action-mediation.v1"
    assert report["status"] == "PASS"
    assert report["phase"] == "fresh-process-composed"
    assert report["runtime_source_contract"] == _common()["runtime_source_contract"]
    assert len(report["rank_reports"]) == 2
    assert report["staged_evidence"]["training_and_evaluation_processes_are_disjoint"]


def test_compose_staged_g3_rejects_identity_drift(tmp_path: Path) -> None:
    training, evaluation = _write_reports(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="ascii"))
    payload["seed"] = 8
    evaluation.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="seed"):
        compose_staged_g3(training_path=training, evaluation_path=evaluation)


def test_compose_staged_g3_rejects_runtime_source_drift(tmp_path: Path) -> None:
    training, evaluation = _write_reports(tmp_path)
    payload = json.loads(evaluation.read_text(encoding="ascii"))
    payload["runtime_source_contract"]["runtime_hotfix_sha256"] = "4" * 64
    evaluation.write_text(json.dumps(payload), encoding="ascii")

    with pytest.raises(ValueError, match="runtime_source_contract"):
        compose_staged_g3(training_path=training, evaluation_path=evaluation)

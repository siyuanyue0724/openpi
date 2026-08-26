from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from picf_next.lingbot_native.task_address_graph import TaskAddressActionInformationSet
from tools.validate_ltop_g3_mediator_trial import (
    ARM_LABELS,
    ARM_VALUES,
    FINAL_REPORT_SCHEMA,
    JOURNAL_SCHEMA,
    OUTPUT_SCHEMA,
    main,
    validate_ltop_g3_mediator_trial,
)

SCHEDULE_SHA256 = "a" * 64


def test_validator_arm_values_match_the_typed_training_contract() -> None:
    assert tuple(arm.value for arm in TaskAddressActionInformationSet) == ARM_VALUES


def _arm_for_step(step: int) -> str:
    return ARM_VALUES[(step - 1) % len(ARM_VALUES)]


def _arm_occurrence(step: int) -> int:
    return (step + 1) // 2


def _action_loss(step: int, *, bad_arm: str | None = None) -> float:
    arm = _arm_for_step(step)
    occurrence = _arm_occurrence(step)
    if occurrence <= 16:
        return 1.0
    if occurrence > 112:
        return 0.98 if arm == bad_arm else 0.80
    return 0.90


def _records(rank: int, *, digest: str = SCHEDULE_SHA256, bad_arm: str | None = None):
    records = []
    for step in range(1, 257):
        arm = _arm_for_step(step)
        records.append(
            {
                "action_loss": _action_loss(step, bad_arm=bad_arm),
                "arm": arm,
                "cycle_index": (step - 1) // 16,
                "global_step": step,
                "physical_set_loss": 0.3,
                "prompt_index": (step - 1) % 2,
                "prompt_key": f"prompt-{(step - 1) % 2}",
                "rank": rank,
                "sample_keys": [f"sample-{rank}"],
                "scene_index": ((step - 1) // 2) % 8,
                "scene_key": f"scene-{((step - 1) // 2) % 8}",
                "schedule_sha256": digest,
                "schema": JOURNAL_SCHEMA,
                "task_address_loss": 0.2,
                "total_loss": 1.5,
            }
        )
    return records


def _write_journals(
    root: Path,
    *,
    bad_arm_by_rank: dict[int, str] | None = None,
    digest_by_rank: dict[int, str] | None = None,
) -> tuple[Path, dict[int, list[dict[str, Any]]]]:
    journal_dir = root / "rank_journal"
    journal_dir.mkdir()
    records_by_rank = {}
    for rank in range(2):
        records = _records(
            rank,
            digest=(digest_by_rank or {}).get(rank, SCHEDULE_SHA256),
            bad_arm=(bad_arm_by_rank or {}).get(rank),
        )
        records_by_rank[rank] = records
        payload = "".join(
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n" for record in records
        )
        (journal_dir / f"rank_{rank}.jsonl").write_text(payload, encoding="ascii")
    return journal_dir, records_by_rank


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _final_report(journal_dir: Path, records_by_rank: dict[int, list[dict[str, Any]]]):
    rank_reports = []
    for rank, records in records_by_rank.items():
        rank_reports.append(
            {
                "action_information_set_counts": {
                    arm_value: sum(record["arm"] == arm_value for record in records)
                    for arm_value in ARM_VALUES
                },
                "action_information_set_history": [
                    {
                        key: record[key]
                        for key in (
                            "global_step",
                            "cycle_index",
                            "scene_index",
                            "scene_key",
                            "prompt_index",
                            "prompt_key",
                            "arm",
                        )
                    }
                    for record in records
                ],
                "action_information_set_schedule_sha256": SCHEDULE_SHA256,
                "action_losses": [record["action_loss"] for record in records],
                "all_gradients_finite": True,
                "arm_journal": {
                    "file_sha256": _file_sha256(journal_dir / f"rank_{rank}.jsonl"),
                    "rank": rank,
                    "record_count": len(records),
                    "schema": "picf-next.ltop-g3-arm-journal-receipt.v1",
                },
                "rank": rank,
            }
        )
    return {
        "failures": [],
        "mode": "mediator-trial",
        "phase": "training",
        "rank_reports": rank_reports,
        "schema": FINAL_REPORT_SCHEMA,
        "status": "PASS",
        "steps": 256,
        "training_contract": {
            "action_information_set_trial": {
                "schedule": {"sha256": SCHEDULE_SHA256},
            }
        },
        "world_size": 2,
    }


def test_validator_accepts_each_rank_and_pooled_global_windows(tmp_path: Path) -> None:
    journal_dir, records_by_rank = _write_journals(tmp_path)
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(_final_report(journal_dir, records_by_rank)), encoding="ascii"
    )

    result = validate_ltop_g3_mediator_trial(
        journal_dir=journal_dir,
        report_path=report_path,
    )

    assert result["schema"] == OUTPUT_SCHEMA
    assert result["status"] == "PASS"
    assert result["failures"] == []
    assert [rank["rank"] for rank in result["ranks"]] == [0, 1]
    assert all(rank["balanced_arms_pass"] for rank in result["ranks"])
    assert all(rank["window_gates_pass"] for rank in result["ranks"])
    assert result["global"]["record_count"] == 512
    assert result["global"]["aggregation"] == "pooled-rank-local-arm-windows"
    for arm_label in ARM_LABELS:
        assert result["global"]["arms"][arm_label]["count"] == 256
        assert result["global"]["arms"][arm_label]["optimizer_step_count"] == 128
        assert result["global"]["arms"][arm_label]["balanced_optimizer_steps_pass"] is True
        assert result["global"]["arms"][arm_label]["first_window"]["count"] == 32
        assert result["global"]["arms"][arm_label]["last_window"]["count"] == 32
        assert result["global"]["arms"][arm_label]["last_to_first_ratio"] == 0.8
        assert result["global"]["arms"][arm_label]["relative_improvement"] == pytest.approx(0.2)
    assert result["final_report"]["consistent"] is True


def test_validator_rejects_one_rank_arm_even_when_pooled_trend_is_good(tmp_path: Path) -> None:
    journal_dir, _ = _write_journals(
        tmp_path,
        bad_arm_by_rank={0: "mediator-required"},
    )

    result = validate_ltop_g3_mediator_trial(journal_dir=journal_dir)

    assert result["status"] == "FAIL"
    assert result["ranks"][0]["arms"]["MEDIATOR_REQUIRED"]["window_gate_pass"] is False
    assert result["ranks"][1]["arms"]["MEDIATOR_REQUIRED"]["window_gate_pass"] is True
    assert any(
        failure.startswith("rank 0 MEDIATOR_REQUIRED: last-16") for failure in result["failures"]
    )


def test_validator_rejects_imbalance_and_nonfinite_loss(tmp_path: Path) -> None:
    journal_dir, _ = _write_journals(tmp_path)
    rank_one_path = journal_dir / "rank_1.jsonl"
    records = [json.loads(line) for line in rank_one_path.read_text().splitlines()]
    records.pop()
    records[-1]["total_loss"] = float("nan")
    rank_one_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="ascii",
    )

    result = validate_ltop_g3_mediator_trial(journal_dir=journal_dir)

    assert result["status"] == "FAIL"
    assert result["ranks"][1]["balanced_arms_pass"] is False
    assert result["ranks"][1]["finite_pass"] is False
    assert result["global"]["balanced_arms_pass"] is False
    assert result["global"]["finite_pass"] is False


def test_validator_rejects_rank_schedule_digest_disagreement(tmp_path: Path) -> None:
    journal_dir, _ = _write_journals(
        tmp_path,
        digest_by_rank={1: "b" * 64},
    )

    result = validate_ltop_g3_mediator_trial(journal_dir=journal_dir)

    assert result["status"] == "FAIL"
    assert result["global"]["schedule"]["consistent"] is False
    assert result["global"]["schedule"]["rank_digests"] == ["a" * 64, "b" * 64]
    assert "rank journals do not share one schedule digest" in result["failures"]


def test_validator_rejects_cross_rank_schedule_entries_despite_matching_digest(
    tmp_path: Path,
) -> None:
    journal_dir, _ = _write_journals(tmp_path)
    rank_one_path = journal_dir / "rank_1.jsonl"
    records = [json.loads(line) for line in rank_one_path.read_text().splitlines()]
    records[0]["scene_key"] = "different-scene"
    rank_one_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records),
        encoding="ascii",
    )

    result = validate_ltop_g3_mediator_trial(journal_dir=journal_dir)

    assert result["status"] == "FAIL"
    assert result["global"]["schedule"]["consistent"] is True
    assert result["global"]["schedule"]["entries_consistent_across_ranks"] is False
    assert (
        "rank journals do not contain the same counterbalanced schedule entries"
        in result["failures"]
    )


def test_validator_rejects_final_report_that_disagrees_with_journal(tmp_path: Path) -> None:
    journal_dir, records_by_rank = _write_journals(tmp_path)
    report = _final_report(journal_dir, records_by_rank)
    report["rank_reports"][1]["action_losses"][3] = 99.0
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="ascii")

    result = validate_ltop_g3_mediator_trial(
        journal_dir=journal_dir,
        report_path=report_path,
    )

    assert result["status"] == "FAIL"
    assert result["final_report"]["consistent"] is False
    assert "final report rank 1 action losses differ from journal" in result["failures"]


def test_cli_outputs_stable_json_and_nonzero_on_failure(tmp_path: Path, capsys) -> None:
    journal_dir, _ = _write_journals(
        tmp_path,
        bad_arm_by_rank={0: "factual"},
    )
    output = tmp_path / "validation.json"

    first_status = main(["--journal-dir", str(journal_dir), "--output", str(output)])
    first_stdout = capsys.readouterr().out
    first_file = output.read_text(encoding="ascii")
    second_status = main(["--journal-dir", str(journal_dir), "--output", str(output)])
    second_stdout = capsys.readouterr().out

    assert first_status == second_status == 1
    assert first_stdout == second_stdout == first_file == output.read_text(encoding="ascii")
    assert json.loads(first_stdout)["status"] == "FAIL"


def test_cli_returns_stable_failure_json_for_malformed_input(tmp_path: Path, capsys) -> None:
    journal_dir, _ = _write_journals(tmp_path)
    path = journal_dir / "rank_0.jsonl"
    original = path.read_text(encoding="ascii")
    path.write_text("not-json\n" + original, encoding="ascii")

    status = main(["--journal-dir", str(journal_dir)])
    payload = json.loads(capsys.readouterr().out)

    assert status == 1
    assert payload["schema"] == OUTPUT_SCHEMA
    assert payload["status"] == "FAIL"
    assert payload["failures"][0].startswith("ValidationInputError:")


def test_report_validation_does_not_mutate_input(tmp_path: Path) -> None:
    journal_dir, records_by_rank = _write_journals(tmp_path)
    report_path = tmp_path / "report.json"
    report = _final_report(journal_dir, records_by_rank)
    report_path.write_text(json.dumps(report), encoding="ascii")
    journal_before = {path: path.read_bytes() for path in sorted(journal_dir.glob("rank_*.jsonl"))}
    report_before = report_path.read_bytes()

    validate_ltop_g3_mediator_trial(journal_dir=journal_dir, report_path=report_path)

    assert {path: path.read_bytes() for path in journal_before} == journal_before
    assert report_path.read_bytes() == report_before

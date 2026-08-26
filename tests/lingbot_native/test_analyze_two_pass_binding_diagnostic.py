from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "adr152" / "tools" / "analyze_two_pass_binding_diagnostic.py"


def _variant(*, action: float, state_l2: float) -> dict[str, Any]:
    return {
        "official_action_loss": action,
        "factual_official_action_loss": action - 0.01,
        "entity_loss": 0.5,
        "predictive_family_loss": 0.01,
        "relative_state_manipulation": state_l2,
    }


def _rank(rank: int) -> dict[str, Any]:
    action = 0.2 + rank * 0.01
    return {
        "rank": rank,
        "eligible": True,
        "variants": {
            "factual": _variant(action=action, state_l2=0.0),
            "zero": _variant(action=action + 0.02, state_l2=1.0),
            "wrong_time": _variant(action=action + 0.001, state_l2=0.1),
            "cross_batch": _variant(action=action, state_l2=0.9),
            "wrong_row": _variant(action=action, state_l2=1.1),
        },
        "control_intervention": {
            "direct_prior_intervention": {
                "arms": {
                    "factual": {
                        "prior_relative_l2": 0.0,
                        "official_action_loss_delta": 0.0,
                    },
                    "zero": {
                        "prior_relative_l2": 1.0,
                        "official_action_loss_delta": 0.02,
                    },
                    "cross_batch": {
                        "prior_relative_l2": 0.1,
                        "official_action_loss_delta": 0.0,
                    },
                    "wrong_row": {
                        "prior_relative_l2": 0.2,
                        "official_action_loss_delta": 0.0,
                    },
                }
            }
        },
    }


def _write_diagnostic(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "picf-next.adr149-two-pass-filter-diagnostic/v1",
                "global_step": 200,
                "rank_reports": [_rank(0), _rank(1)],
            }
        ),
        encoding="utf-8",
    )


def test_binding_summary_detects_row_invariant_action(tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.json"
    output = tmp_path / "summary.json"
    _write_diagnostic(diagnostic)

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--diagnostic",
            str(diagnostic),
            "--output",
            str(output),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["status"] == "PASS"
    report = json.loads(output.read_text(encoding="utf-8"))
    flags = report["exact_flags"]
    assert flags["all_ranks_eligible"] is True
    assert flags["wrong_row_changes_state_on_every_rank"] is True
    assert flags["routed_action_exactly_invariant_to_wrong_row_on_every_rank"] is True
    assert flags["direct_action_exactly_invariant_to_wrong_row_on_every_rank"] is True
    assert flags["direct_action_changes_under_zero_prior_on_any_rank"] is True


def test_binding_summary_rejects_missing_registered_arm(tmp_path: Path) -> None:
    diagnostic = tmp_path / "diagnostic.json"
    output = tmp_path / "summary.json"
    _write_diagnostic(diagnostic)
    payload = json.loads(diagnostic.read_text(encoding="utf-8"))
    del payload["rank_reports"][0]["variants"]["wrong_row"]
    diagnostic.write_text(json.dumps(payload), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--diagnostic",
            str(diagnostic),
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "omits a registered state arm" in completed.stderr
    assert not output.exists()

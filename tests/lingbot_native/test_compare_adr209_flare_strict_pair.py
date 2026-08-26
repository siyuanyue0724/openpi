from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.compare_adr209_flare_strict_pair import (
    CANDIDATE_OBJECTIVE,
    CANDIDATE_PROFILE,
    CONTROL_OBJECTIVE,
    CONTROL_PROFILE,
    _canonical_sha256,
    _validate_and_summarize_journals,
    _validate_manifests,
    _validate_snapshot_pair,
)


def _manifest(*, candidate: bool) -> dict[str, object]:
    contract = {
        "future_latent_objective_scale": 1.0 if candidate else 0.0,
        "objective_profile": CANDIDATE_OBJECTIVE if candidate else CONTROL_OBJECTIVE,
        "picf_architecture_profile": CANDIDATE_PROFILE if candidate else CONTROL_PROFILE,
        "shared_contract": {"future_token_count": 128, "capture_layer_index": 26},
    }
    return {
        "execution_contract": contract,
        "execution_contract_sha256": _canonical_sha256(contract),
        "implementation_sha256": "1" * 64,
        "shared_identity": "2" * 64,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="ascii")


def _row(*, step: int, candidate: bool) -> dict[str, object]:
    raw_loss = 1.0 - 0.05 * (step - 1) if candidate else 1.0
    weighted_loss = 0.2 * raw_loss
    source_total = 7.0 - 0.1 * step
    action_loss = 0.5 - 0.01 * step
    return {
        "global_step": step,
        "sample_keys": [f"sample-{step}"],
        "augmentation_seeds": [step],
        "flow_noise_seeds": [step + 10],
        "flow_timestep_seeds": [step + 20],
        "source_digest": f"source-{step}",
        "temporal_plan_sha256": "3" * 64,
        "frame_indices": [step - 1],
        "lane_ids": [0],
        "reset": step == 1,
        "official_action_loss": action_loss,
        "official_moe_regularizer": 0.002,
        "objective_total": (
            source_total + action_loss + (weighted_loss if candidate else 0.0)
        ),
        "future_latent_alignment": {
            "action_layer_count": 36,
            "capture_layer_index": 26,
            "future_token_count": 128,
            "objective_scale": 1.0 if candidate else 0.0,
            "objective_contribution": weighted_loss if candidate else 0.0,
            "weighted_loss": weighted_loss,
            "raw_loss": raw_loss,
            "mean_cosine": 1.0 - raw_loss,
            "target_manifest_sha256": "4" * 64,
        },
        "gradient_metrics": {
            "host_all_finite": True,
            "source_all_finite_and_present": True,
        },
        "videomt_source_objective": {"total": source_total},
        "peak_cuda_reserved_bytes": 32 * 1024**3,
    }


def _snapshot(*, action_offset: float = 0.0) -> dict[str, object]:
    samples = [
        {
            "sample_key": "sample-0",
            "partition": "heldout",
            "source_digest": "4" * 64,
            "model_inputs_sha256": "5" * 64,
            "native_source_rgb_sha256": "6" * 64,
            "action_loss": 0.4 + action_offset,
        },
        {
            "sample_key": "sample-1",
            "partition": "validation",
            "source_digest": "7" * 64,
            "model_inputs_sha256": "8" * 64,
            "native_source_rgb_sha256": "9" * 64,
            "action_loss": 0.5 + action_offset,
        },
    ]
    return {
        "status": "PASS",
        "checkpoint_global_step": 0,
        "evaluation_input_sha256": "a" * 64,
        "samples": samples,
        "partition_summaries": {
            "heldout": {"mean_action_loss": 0.4 + action_offset},
            "validation": {"mean_action_loss": 0.5 + action_offset},
        },
    }


def test_strict_manifests_differ_only_at_registered_lambda_surface(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    control_path = tmp_path / "control.json"
    _write_json(candidate_path, _manifest(candidate=True))
    _write_json(control_path, _manifest(candidate=False))

    _, _, differences = _validate_manifests(candidate_path, control_path)

    assert differences == [
        "execution_contract.future_latent_objective_scale",
        "execution_contract.objective_profile",
        "execution_contract.picf_architecture_profile",
        "execution_contract_sha256",
    ]


def test_strict_manifests_reject_an_extra_difference(tmp_path: Path) -> None:
    candidate = _manifest(candidate=True)
    control = _manifest(candidate=False)
    control["shared_identity"] = "f" * 64
    candidate_path = tmp_path / "candidate.json"
    control_path = tmp_path / "control.json"
    _write_json(candidate_path, candidate)
    _write_json(control_path, control)

    with pytest.raises(ValueError, match="unexpected manifest differences"):
        _validate_manifests(candidate_path, control_path)


def test_journal_comparison_requires_the_exact_stream() -> None:
    candidate = {0: [_row(step=step, candidate=True) for step in range(1, 6)]}
    control = {0: [_row(step=step, candidate=False) for step in range(1, 6)]}

    curve, summary = _validate_and_summarize_journals(
        candidate,
        control,
        terminal_step=5,
    )

    assert len(curve) == 5
    assert summary["action_auc_delta_candidate_minus_control"] == pytest.approx(0.0)
    assert summary["candidate_flare_final"] < summary["control_flare_final"]

    mismatched = copy.deepcopy(control)
    mismatched[0][2]["flow_noise_seeds"] = [999]
    with pytest.raises(ValueError, match="stream differs for flow_noise_seeds"):
        _validate_and_summarize_journals(candidate, mismatched, terminal_step=5)


def test_step_zero_action_snapshot_must_match_exactly(tmp_path: Path) -> None:
    candidate_path = tmp_path / "candidate.json"
    control_path = tmp_path / "control.json"
    _write_json(candidate_path, _snapshot())
    _write_json(control_path, _snapshot())

    _, _, summaries = _validate_snapshot_pair(
        candidate_path,
        control_path,
        step=0,
        require_equal_action=True,
    )

    assert summaries["heldout"]["mean_paired_delta_candidate_minus_control"] == 0.0

    _write_json(control_path, _snapshot(action_offset=0.01))
    with pytest.raises(ValueError, match="initial action loss differs"):
        _validate_snapshot_pair(
            candidate_path,
            control_path,
            step=0,
            require_equal_action=True,
        )

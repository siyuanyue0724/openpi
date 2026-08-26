# ruff: noqa: E402  # Optional torch gate must precede safetensors imports.
from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")
from safetensors.torch import save_file

from tools.audit_molmoact2_m4_checkpoint import _gate_summary, build_report


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, factorization: str, object_gate: float):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints" / "step-00000001"
    checkpoint.mkdir(parents=True)
    model = checkpoint / "model.safetensors"
    prefix = "joint_bridge.sequence_bridge.policy.action_layer_adapter"
    save_file(
        {
            f"{prefix}.dense_branches.0.gate": torch.tensor(1e-5),
            f"{prefix}.dense_branches.1.gate": torch.tensor(-1e-5),
            f"{prefix}.object_branches.0.gate": torch.tensor(object_gate),
            f"{prefix}.object_branches.1.gate": torch.tensor(-object_gate),
            "unrelated.depth_gate": torch.ones(3),
        },
        model,
    )
    include_posterior = factorization in {"C", "D"}
    control = {
        "schema": "picf-next.checkpoint-control-manifest.v2",
        "plan_sha256": "p" * 64,
        "contract": {
            "arm_config": {
                "causal_factorization": {
                    "id": factorization,
                    "include_posterior_action_context": include_posterior,
                }
            }
        },
        "progress": {"successful_optimizer_steps": 1},
        "state_files": {
            "model.safetensors": {
                "sha256": _sha256(model),
                "size_bytes": model.stat().st_size,
            }
        },
    }
    (checkpoint / "picf_control.json").write_text(json.dumps(control), encoding="ascii")
    (run / "static_preflight.json").write_text(
        json.dumps({"causal_factorization": {"id": factorization}}), encoding="ascii"
    )
    (run / "sample_plan.json").write_text(
        json.dumps(
            {
                "metadata": {"comparison_id": "matched"},
                "plan_sha256": "p" * 64,
            }
        ),
        encoding="ascii",
    )
    (run / "metrics.jsonl").write_text(
        json.dumps(
            {
                "metrics": {"action_flow_loss": 0.5},
                "optimizer_step_skipped": False,
                "successful_optimizer_steps": 1,
            }
        )
        + "\n",
        encoding="ascii",
    )
    return run, checkpoint


def test_gate_summary_requires_contiguous_finite_values() -> None:
    summary = _gate_summary({0: 0.0, 1: -0.25}, family="dense")

    assert summary["count"] == 2
    assert summary["nonzero_count"] == 1
    assert summary["abs_mean"] == pytest.approx(0.125)
    with pytest.raises(ValueError, match="contiguous"):
        _gate_summary({1: 0.0}, family="dense")
    with pytest.raises(ValueError, match="NaN"):
        _gate_summary({0: float("nan")}, family="dense")


@pytest.mark.parametrize(("factorization", "object_gate"), (("A", 0.0), ("C", 1e-5)))
def test_checkpoint_audit_accepts_gate_state_for_factorization(
    tmp_path, factorization: str, object_gate: float
) -> None:
    run, checkpoint = _fixture(tmp_path, factorization=factorization, object_gate=object_gate)

    report = build_report(
        run_dir=run,
        checkpoint=checkpoint,
        expected_factorization=factorization,
    )

    assert report["status"] == "PASS"
    assert report["gates_after_optimizer_step"]["dense"]["nonzero_count"] == 2
    assert report["gates_after_optimizer_step"]["object"]["nonzero_count"] == (
        0 if factorization == "A" else 2
    )


def test_checkpoint_audit_rejects_zero_object_route_for_posterior_arm(tmp_path) -> None:
    run, checkpoint = _fixture(tmp_path, factorization="C", object_gate=0.0)

    report = build_report(run_dir=run, checkpoint=checkpoint)

    assert report["status"] == "FAIL"
    assert report["checks"]["object_route_matches_declared_factorization"] is False

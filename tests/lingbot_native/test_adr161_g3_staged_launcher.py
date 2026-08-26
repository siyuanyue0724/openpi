from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "adr161/run_ltop_g3_staged_evaluation_2gpu.sh").read_text(encoding="utf-8")


def test_staged_launcher_requires_a_registered_training_pass_and_checkpoint() -> None:
    source = _source()

    assert '"schema": "picf-next.ltop-g3-training-phase.v1"' in source
    assert '"status": "PASS"' in source
    assert '"phase": "training"' in source
    assert '"steps": 128' in source
    assert '"eval_every": 32' in source
    assert 'checkpoint.get("optimizer_saved") is not False' in source
    assert "checkpoint_path.is_dir()" in source


def test_staged_launcher_cold_evaluates_then_composes_the_final_gate() -> None:
    source = _source()

    assert "--nproc_per_node=2" in source
    assert "--phase evaluation" in source
    assert '--trained-checkpoint "$trained_checkpoint"' in source
    assert "compose_ltop_g3_staged.py" in source
    assert "load_accepted_g3_gate" in source
    assert "timeout --signal=TERM --kill-after=60s" in source
    assert "ltop_g3_evaluation_runtime_failure.json" in source
    assert source.index("--phase evaluation") < source.index("compose_ltop_g3_staged.py")


def test_staged_launcher_preserves_the_verified_muon_runtime_hotfix() -> None:
    source = _source()

    assert "PICF_LINGBOT_RUNTIME_HOTFIX" in source
    assert "lingbot_vla2_distributed_muon_collective_alignment.patch" in source
    assert '--runtime-hotfix "$runtime_hotfix"' in source

from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "adr161/run_ltop_core_pilot_2gpu.sh").read_text(encoding="utf-8")


def test_adr161_launcher_is_two_gpu_persistent_and_g3_gated() -> None:
    source = _source()

    assert "--nproc_per_node=2" in source
    assert "ltop_g3_action_mediation_report.json" in source
    assert "[smoke|pilot]" in source
    assert 'if [[ "$run_dir" != /mnt/* ]]' in source
    assert "--stage-checkpoint /mnt/picf-next/checkpoints/adr160-g2b-confirm-49eac80-v3" in source
    assert "--action-information-set-policy factual-only" in source


def test_adr161_launcher_uses_the_frozen_interleaved_2k_contract() -> None:
    source = _source()

    assert "calvin-two-gpu-2k-interleave8-v2" in source
    assert "0481025ca66430ac91562f9356bc60e3fd82bedea00ae34a6a5fa6e8708a74cf" in source
    assert "38a5919be926db83d4cd43be1f6192da92e917a4848790a5bc7d8ea1875b38f0" in source
    assert "24003a1707f6aff1324bbae5a96e5c88448bc47c0b737388503d358e15001244" in source


def test_adr161_launcher_exposes_only_the_registered_pair() -> None:
    source = _source()

    assert "ltop-ec-factual|ltop-ec-blocked" in source
    assert "official-lbot" not in source
    assert "run_lingbot_vla2_ltop_core_pilot.py" in source


def test_adr161_launcher_preserves_the_verified_muon_runtime_hotfix() -> None:
    source = _source()

    assert "PICF_LINGBOT_RUNTIME_HOTFIX" in source
    assert "lingbot_vla2_distributed_muon_collective_alignment.patch" in source
    assert '--runtime-hotfix "$runtime_hotfix"' in source

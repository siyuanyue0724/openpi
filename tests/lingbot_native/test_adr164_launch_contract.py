from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "adr164/run_ltop_core_long_2gpu.sh").read_text(encoding="utf-8")


def _restart_smoke_source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "adr164/run_ltop_core_restart_smoke_2gpu.sh").read_text(encoding="utf-8")


def test_adr164_launcher_is_two_gpu_persistent_and_g3_gated() -> None:
    source = _source()

    assert "--nproc_per_node=2" in source
    assert "PICF_G3_ACCEPTANCE_REPORT must point to an ADR170 source-aligned acceptance" in source
    assert 'if [[ "$run_dir" != /mnt/* ]]' in source
    assert "mode=${PICF_LTOP_MODE:-long}" in source
    assert '--mode "$mode"' in source
    assert "--arm ltop-ec-factual" in source
    assert "--action-information-set-policy rank-step-counterbalanced-50-50" in source


def test_adr164_launcher_pins_the_frozen_30k_contract() -> None:
    source = _source()

    assert "calvin-two-gpu-30k-interleave8-v2" in source
    assert "d35b4c587fa30e6d23029da4ef2f6cccf08faa83b0bd937ab15379aeb1e69d71" in source
    assert "0852f5bed788da25b857c0bf3e6e9009ab9887ea44784f366fa9bef0de2904fe" in source
    assert "e873da94f941bf706629329287d3a9f850041cb6c2dc2fc60a47d85023e473d3" in source


def test_adr164_launcher_preserves_the_verified_runtime_and_start_state() -> None:
    source = _source()

    assert "lingbot_vla2_distributed_muon_collective_alignment.patch" in source
    assert "adr160-g2b-confirm-49eac80-v3" in source
    assert "PICF_G3_ACCEPTANCE_REPORT" in source
    assert "ADR170 source-aligned acceptance" in source
    assert "adr165-g3-mediator-acceptance" not in source
    assert "--cuda-allocator expandable-segments" in source


def test_adr164_launcher_uses_independent_2k_processes_and_final_cold_load() -> None:
    source = _source()

    assert "segment_steps=2000" in source
    assert "current_phase=resume" in source
    assert '--phase "$current_phase"' in source
    assert '--load-global-step "$current_step"' in source
    assert '--stop-after-step "$stop_step"' in source
    assert '--load-global-step "$total_steps"' in source
    assert '--stop-after-step "$total_steps"' in source
    assert '2>&1 | tee "$log_path"' in source
    assert "latest_checkpoint_step=0" in source
    assert "resume must use latest complete LTOP checkpoint" in source


def test_adr164_restart_smoke_reuses_the_production_launcher() -> None:
    source = _restart_smoke_source()

    assert "PICF_LTOP_MODE=restart-smoke" in source
    assert 'exec "$repository_root/adr164/run_ltop_core_long_2gpu.sh" "$@"' in source

from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "adr161/continue_after_g3_2gpu.sh").read_text(encoding="utf-8")


def test_continuation_waits_for_training_then_runs_isolated_g3_evaluation() -> None:
    source = _source()

    assert "ltop_g3_training_report.json" in source
    assert "G3 training process exited without a final report" in source
    assert source.count("pgrep -f 'run_lingbot_vla2_ltop_g3_action_mediation.py") == 2
    assert "Avoid overlapping two full-model process groups" in source
    assert "run_ltop_g3_staged_evaluation_2gpu.sh" in source
    assert "cold-starting isolated evaluation" in source


def test_continuation_smokes_then_cold_starts_the_formal_factual_arm() -> None:
    source = _source()

    smoke = source.index('ltop-ec-factual "$smoke_run" smoke')
    validate = source.index("engineering smoke omitted its transactional checkpoint or metrics")
    pilot = source.index('ltop-ec-factual "$pilot_run" pilot')
    assert smoke < validate < pilot
    assert "global_step_2" in source
    assert "steps_00000001_00000002.json" in source
    assert "len(diagnostics) != 2 or len(visuals) != 2" in source
    assert "exec env PICF_REPOSITORY_ROOT" in source


def test_continuation_uses_persistent_outputs_and_revision_names() -> None:
    source = _source()

    assert 'repository_root" != /mnt/*' in source
    assert 'git -C "$repository_root" rev-parse --verify HEAD' in source
    assert "/mnt/picf-next/runs/adr161-ltop-ec-factual-smoke-" in source
    assert "/mnt/picf-next/runs/adr161-ltop-ec-factual-2k-" in source
    assert "/mnt/picf-next/runs/adr160-g3-evaluation-" in source
    assert "/mnt/picf-next/runs/adr160-g3-composed-" in source


def test_continuation_binds_smoke_and_pilot_to_the_composed_g3_report() -> None:
    source = _source()

    assert source.count('PICF_G3_RUN_ROOT="$g3_composed_run"') == 2


def test_continuation_discards_only_the_validated_disposable_smoke_checkpoint() -> None:
    source = _source()

    validation = source.index("engineering smoke omitted a rank diagnostic or entity visual")
    discard = source.index("shutil.rmtree(checkpoint_directory)")
    pilot = source.index('ltop-ec-factual "$pilot_run" pilot')
    assert validation < discard < pilot
    assert "checkpoint_manifest_sha256" in source
    assert "DISCARDED_AFTER_PASS" in source
    assert "write_text_durable_exclusive" in source

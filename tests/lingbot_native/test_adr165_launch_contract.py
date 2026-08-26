from __future__ import annotations

from pathlib import Path


def _source() -> str:
    root = Path(__file__).resolve().parents[2]
    return (root / "adr165/run_ltop_g3_mediator_trial_2gpu_256.sh").read_text(
        encoding="utf-8"
    )


def test_adr165_launcher_is_fixed_two_gpu_256_step_training() -> None:
    source = _source()

    assert "--nproc_per_node=2" in source
    assert "--mode mediator-trial" in source
    assert "--phase training" in source
    assert "--steps 256" in source
    assert "--eval-every 32" in source


def test_adr165_launcher_is_clean_mnt_only_and_reuses_g2b() -> None:
    source = _source()

    assert '[[ "$repository_root" == /mnt/* && "$source_checkout" == /mnt/* ]]' in source
    assert '[[ "$run_root" == /mnt/* && ! -e "$run_root" && ! -L "$run_root" ]]' in source
    assert "status --porcelain=v1 --untracked-files=all" in source
    assert "adr160-g2b-confirm-49eac80-v3" in source


def test_adr165_launcher_publishes_model_only_and_arm_journals() -> None:
    source = _source()

    assert '--checkpoint-output "$run_root/checkpoint-model-only"' in source
    assert '--journal-dir "$run_root/rank_journal"' in source
    assert '--output "$run_root/ltop_g3_mediator_trial_training_report.json"' in source

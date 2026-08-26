from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "adr150/launch_complete_pipeline_when_caches_ready.sh"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_complete_pipeline_waits_for_every_final_modality_artifact() -> None:
    script = _script()
    assert (
        "CALIBRATION_RECEIPT_SHA="
        "3bb381922df52c2cd561a2acb1824bc17c78dcacc2ca971727736c62c4baca65"
    ) in script
    for value in (
        '"$ANYTOUCH/manifest.json"',
        '"$ANYTOUCH.receipt.json"',
        '"$SONATA/manifest.json"',
        '"$SONATA.receipt.json"',
        '"$root/manifest.json"',
        '"$root.receipt.json"',
    ):
        assert value in script
    assert script.count("$CACHE_ROOT/vjepa-parts/p") == 8


def test_complete_pipeline_is_fail_closed_and_orders_promotion_gates() -> None:
    script = _script()
    ordered = (
        "stage MERGING_VJEPA",
        "stage RUNNING_SEMANTIC_AUDIT",
        "stage RUNNING_SOURCE_AUDIT_AND_FOUR_GPU_ACCEPTANCE",
        'run_full_modal_acceptance_suite_4gpu.sh" "$SUITE_ROOT"',
        "stage RUNNING_MATCHED_LBOT_200",
        'run_matched_lbot_4gpu.sh" "$LBOT_RUN" 200',
        'wait "$SOURCE_PID"',
        "stage FREEZING_INPUTS",
        'freeze_inputs.sh" "$LBOT_REPORT"',
        "stage STARTING_FULL_MODAL_2K",
        'launch_four_gpu_initial_2k.sh" "$TRAIN_RUN" "$LBOT_REPORT"',
    )
    positions = tuple(script.index(value) for value in ordered)
    assert positions == tuple(sorted(positions))
    assert "set -euo pipefail" in script
    assert '[[ -x "$path" && -f "$(readlink -f "$path")" ]]' in script
    assert "run_lingbot_vla2_task_independent_full.py" not in script

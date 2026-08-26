import subprocess
from pathlib import Path

from tools.run_lingbot_vla2_task_independent_full import IMPLEMENTATION_FILES

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "adr152/run_posterior_adoption_route_4gpu.sh"
ACCEPTANCE = ROOT / "adr150/run_full_modal_acceptance_4gpu.sh"


def test_adr152_route_launcher_is_syntactically_valid_and_implementation_bound() -> None:
    subprocess.run(("bash", "-n", str(LAUNCHER)), check=True)
    assert str(LAUNCHER.relative_to(ROOT)) in IMPLEMENTATION_FILES


def test_adr152_route_launcher_binds_exact_lbot_and_one_registered_treatment() -> None:
    launcher = LAUNCHER.read_text(encoding="utf-8")
    acceptance = ACCEPTANCE.read_text(encoding="utf-8")
    for required in (
        '"status": "PASS"',
        '"steps": 200',
        '"seed": 20260721',
        '"world_size": 4',
        '"picf_graph_installed": False',
        '"posterior_present": False',
        '"task_scorer_present": False',
        '"physical_sidecar_read": False',
        '"registered_evaluation_steps": [0, 20, 100, 200]',
        "evaluation_input_sha256",
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
        'posterior-route "$RUN_DIR"',
    ):
        assert required in launcher
    assert "ACCEPTANCE_MODE=posterior-adoption-route" in acceptance
    assert "STOP_AFTER_STEP=500" in acceptance


def test_adr152_route_launcher_fails_closed_without_persistent_inputs(tmp_path: Path) -> None:
    result = subprocess.run(
        ("bash", str(LAUNCHER), str(tmp_path / "run"), str(tmp_path / "lbot")),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "under /mnt" in result.stderr

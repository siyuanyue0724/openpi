import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ARM_SCRIPT = ROOT / "configs/cloud/adr135_matched_medium_horizon_arm.sh"
BASELINE_SCRIPT = ROOT / "configs/cloud/adr135_step_zero_baseline.sh"
BUNDLE_SCRIPT = ROOT / "configs/cloud/adr135_publish_bundle.sh"


def test_adr135_shell_scripts_are_syntactically_valid() -> None:
    subprocess.run(
        ["bash", "-n", str(ARM_SCRIPT), str(BASELINE_SCRIPT)],
        check=True,
        cwd=ROOT,
    )


def test_adr135_bundle_publisher_is_syntactically_valid_and_atomic() -> None:
    subprocess.run(["bash", "-n", str(BUNDLE_SCRIPT)], check=True, cwd=ROOT)
    source = BUNDLE_SCRIPT.read_text(encoding="ascii")

    assert "refusing to replace ADR135 bundle" in source
    assert 'mv "$TEMP" "$OUTPUT"' in source
    assert "RUNNER_SHA256" in source
    assert "ARM_SCRIPT_SHA256" in source
    assert "BASELINE_SCRIPT_SHA256" in source
    assert "STEP_ZERO_BASELINE_SHA256" in source
    assert "picf-next.adr135-matched-medium-horizon-bundle.v1" in source


def test_adr135_arms_share_one_command_and_differ_only_by_registered_estimator() -> None:
    source = ARM_SCRIPT.read_text(encoding="ascii")

    assert "M) OWNERSHIP_ESTIMATOR=token_micro_categorical" in source
    assert "E) OWNERSHIP_ESTIMATOR=token_micro_entity_conditional_equal" in source
    assert source.count('--ownership-estimator "$OWNERSHIP_ESTIMATOR"') == 1
    assert "--training-stage representation" in source
    assert "--checkpoint-publication always" in source
    assert "--total-planned-steps 1000" in source
    assert "--evidence-profile matched_medium_horizon" in source
    assert "--gradient-audit-steps 18,34,50,100,200,500,1000" in source
    assert "--visual-audit-every 200" in source
    assert "--authorization-manifest" not in source
    assert "ADR135 runner changed after bundle publication" in source
    assert "ADR135 arm launcher changed after bundle publication" in source


def test_adr135_arm_script_exposes_only_the_registered_segments() -> None:
    source = ARM_SCRIPT.read_text(encoding="ascii")
    for segment in (
        "fresh:0:1:0",
        "resume:1:199:200",
        "resume:200:300:500",
        "resume:500:500:1000",
    ):
        assert segment in source
    assert source.count("unregistered ADR135 segment") == 1


def test_adr135_baseline_is_released_weight_step_zero_only() -> None:
    source = BASELINE_SCRIPT.read_text(encoding="ascii")

    assert "--phase fresh" in source
    assert "--invocation-steps 1" in source
    assert "--total-planned-steps 1000" in source
    assert "--representation-evaluation-steps 0" in source
    assert "--evidence-profile loss_visual_trial" in source
    assert "--visual-audit-every 0" in source
    assert "--lane-interleave-factor 1" in source
    assert "--reset-mixture-numerator" not in source
    assert "ADR135 runner changed after bundle publication" in source
    assert "ADR135 baseline launcher changed after bundle publication" in source

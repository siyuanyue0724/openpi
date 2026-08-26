import re
import subprocess
from pathlib import Path

import tools.run_lingbot_vla2_native_full as full_runner

ROOT = Path(__file__).resolve().parents[2]
ARM_SCRIPT = ROOT / "configs/cloud/adr136_content_addressed_set_arm.sh"
BASELINE_SCRIPT = ROOT / "configs/cloud/adr136_step_zero_baseline.sh"
BUNDLE_SCRIPT = ROOT / "configs/cloud/adr136_publish_bundle.sh"


def test_adr136_shell_scripts_are_syntactically_valid() -> None:
    subprocess.run(
        ["bash", "-n", str(ARM_SCRIPT), str(BASELINE_SCRIPT), str(BUNDLE_SCRIPT)],
        check=True,
        cwd=ROOT,
    )


def test_adr136_bundle_freezes_architecture_and_is_atomic() -> None:
    source = BUNDLE_SCRIPT.read_text(encoding="ascii")

    assert "refusing to replace ADR136 bundle" in source
    assert 'mv "$TEMP" "$OUTPUT"' in source
    assert "picf-next.adr136-content-addressed-set-bundle.v1" in source
    assert "content_addressed_set_v1" in source
    for digest in (
        "RUNNER_SHA256",
        "PUBLISH_SCRIPT_SHA256",
        "ARM_SCRIPT_SHA256",
        "BASELINE_SCRIPT_SHA256",
        "HOST_SHA256",
        "GRAPH_SHA256",
        "RELATIONS_SHA256",
        "SUPERVISION_SHA256",
        "TASK_RELATION_SHA256",
        "TEMPORAL_SHA256",
        "FULL_TRAINING_SHA256",
        "STEP_ZERO_BASELINE_SHA256",
    ):
        assert digest in source
    assert "load_representation_evaluation_baseline" in source
    assert "validate_representation_baseline_plan" in source
    assert '"$STEP_ZERO_BASELINE" "$RESET_EVALUATION_PLAN"' in source


def test_adr136_arm_changes_only_the_registered_transition_candidate() -> None:
    source = ARM_SCRIPT.read_text(encoding="ascii")

    assert "OBJECT_TRANSITION" in source
    assert "content_addressed_set_v1" in source
    assert "--training-stage representation" in source
    assert "--checkpoint-publication always" in source
    assert "--total-planned-steps 1000" in source
    assert "--evidence-profile matched_medium_horizon" in source
    assert "--gradient-audit-steps 18,34,50,100,200,500,1000" in source
    assert "--visual-audit-every 200" in source
    assert "--ownership-estimator token_micro_categorical" in source
    assert "token_micro_entity_conditional_equal" not in source
    assert "--relation-supervision-layers" not in source
    assert "--authorization-manifest" not in source


def test_adr136_arm_exposes_only_bounded_falsification_segments() -> None:
    source = ARM_SCRIPT.read_text(encoding="ascii")
    match = re.search(
        r'case "\$PHASE:\$LOAD_GLOBAL_STEP:\$INVOCATION_STEPS:\$EVALUATION_STEP" in\s*'
        r"(?P<segments>[^)]*)\) ;;",
        source,
    )
    assert match is not None
    shell_segments = {segment.strip() for segment in match.group("segments").split("|")}
    runner_segments = {
        f"{phase}:{load_step}:{invocation_steps}:{evaluation_steps[0]}"
        for phase, load_step, invocation_steps, evaluation_steps in (
            full_runner.CONTENT_ADDRESSED_SET_MEDIUM_HORIZON_SEGMENTS
        )
    }
    assert shell_segments == runner_segments
    for forbidden in ("resume:200", "resume:500", "1000:1000"):
        assert forbidden not in source
    assert source.count("unregistered ADR136 segment") == 1


def test_adr136_baseline_publishes_step_zero_without_writing_a_checkpoint() -> None:
    source = BASELINE_SCRIPT.read_text(encoding="ascii")

    assert "--phase fresh" in source
    assert "--checkpoint-publication never" in source
    assert "--invocation-steps 1" in source
    assert "--total-planned-steps 1000" in source
    assert "--representation-evaluation-steps 0,1" in source
    assert "--evidence-profile loss_visual_trial" in source
    assert "--visual-audit-every 0" in source
    assert "--lane-interleave-factor 1" in source
    assert "--reset-mixture-numerator" not in source

    invocation_steps = int(source.split("--invocation-steps ", 1)[1].split()[0])
    evaluation_steps = {
        int(value)
        for value in source.split("--representation-evaluation-steps ", 1)[1].split()[0].split(",")
    }
    assert {0, invocation_steps} <= evaluation_steps

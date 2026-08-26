from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.build_lingbot_fixed_observation_launch import (
    ADR122_ESTIMATOR,
    ARM_NR_DONOR_SHA256,
    FIXED_OBSERVATION_OPTION_PAIRS,
    FIXED_OBSERVATION_OPTIONS,
    derive_fixed_observation_launch,
)
from tools.build_lingbot_representation_natural_prompt_launch import parse_launch_text

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/build_lingbot_fixed_observation_launch.py"
DONOR = ROOT / "references" / "experiments" / "lingbot-representation-adr121-arm-nr-launch.sh"
CANDIDATE_CHECKOUT = "/mnt/picf-next/audit-checkouts/adr123-b932f0d-git-20260731"
CANDIDATE_REVISION = "b932f0d" + "0" * 33
CANDIDATE_TREE = "1" * 40
RUN_DIR = "/mnt/picf-next/runs/adr123-fixed-observation-b932f0d-20260731"
LOG = "/mnt/picf-next/logs/adr123-fixed-observation-b932f0d-20260731.log"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _bindings() -> dict[str, str]:
    result: dict[str, str] = {}
    for index, (path_option, digest_option) in enumerate(
        FIXED_OBSERVATION_OPTION_PAIRS,
        start=1,
    ):
        result[path_option] = f"/mnt/picf-next/probes/adr123-fixed-x-{index}.json"
        result[digest_option] = f"{index:x}" * 64
    return result


def _derive(
    donor_text: str,
    *,
    bindings: dict[str, str] | None = None,
) -> tuple[str, dict[str, object]]:
    return derive_fixed_observation_launch(
        donor_text,
        baseline_sha256=_sha256(donor_text),
        candidate_checkout=CANDIDATE_CHECKOUT,
        candidate_revision=CANDIDATE_REVISION,
        candidate_tree=CANDIDATE_TREE,
        run_dir=RUN_DIR,
        log=LOG,
        bindings=_bindings() if bindings is None else bindings,
    )


def test_fixed_observation_launch_has_only_preregistered_deltas() -> None:
    donor_text = DONOR.read_text(encoding="utf-8")
    assert _sha256(donor_text) == ARM_NR_DONOR_SHA256
    candidate_text, report = _derive(donor_text)
    donor = parse_launch_text(donor_text)
    candidate = parse_launch_text(candidate_text)

    assert len(donor.option_pairs) == 58
    assert len(candidate.option_pairs) == 68
    assert candidate.options["--task-relation-estimator"] == ADR122_ESTIMATOR
    assert (
        tuple(
            option
            for option, _value in candidate.option_pairs
            if option in FIXED_OBSERVATION_OPTIONS
        )
        == FIXED_OBSERVATION_OPTIONS
    )
    assert candidate.options["--reset-mixture-numerator"] == "1"
    assert candidate.options["--reset-mixture-denominator"] == "2"
    assert candidate.options["--representation-evaluation-steps"] == "0,200"
    assert candidate.options["--checkpoint-publication"] == "never"
    assert "CUDA_VISIBLE_DEVICES=0,1" in candidate.prefix
    assert "--nproc_per_node=2" in candidate.prefix
    assert candidate.runner == (f"{CANDIDATE_CHECKOUT}/tools/run_lingbot_vla2_native_full.py")
    for option in ("--patch", "--robot-config", "--data-config"):
        assert candidate.options[option].startswith(f"{CANDIDATE_CHECKOUT}/")
    assert report["status"] == "PASS"
    assert report["authorizes_action_or_long_training"] is False
    assert report["unchanged_control_option_count"] == 57
    assert report["candidate_launch_sha256"] == _sha256(candidate_text)
    assert report["candidate_revision"] == CANDIDATE_REVISION
    assert report["candidate_tree"] == CANDIDATE_TREE


def test_fixed_observation_launch_rejects_donor_and_binding_drift() -> None:
    donor_text = DONOR.read_text(encoding="utf-8")
    changed_schedule = donor_text.replace(
        "--gradient-audit-steps 18,34,50,100,200",
        "--gradient-audit-steps 50,100,200",
    )
    with pytest.raises(ValueError, match="not the pinned archived launcher"):
        _derive(changed_schedule)

    with pytest.raises(ValueError, match="not the pinned archived launcher"):
        derive_fixed_observation_launch(
            donor_text,
            baseline_sha256="0" * 64,
            candidate_checkout=CANDIDATE_CHECKOUT,
            candidate_revision=CANDIDATE_REVISION,
            candidate_tree=CANDIDATE_TREE,
            run_dir=RUN_DIR,
            log=LOG,
            bindings=_bindings(),
        )

    incomplete = _bindings()
    del incomplete["--fixed-observation-heldout-audit-sha256"]
    with pytest.raises(ValueError, match="bindings differ"):
        _derive(donor_text, bindings=incomplete)

    duplicate = _bindings()
    duplicate["--fixed-observation-heldout-audit"] = duplicate[
        "--fixed-observation-validation-audit"
    ]
    with pytest.raises(ValueError, match="paths must be distinct"):
        _derive(donor_text, bindings=duplicate)

    uppercase = _bindings()
    uppercase["--fixed-observation-heldout-audit-sha256"] = "A" * 64
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        _derive(donor_text, bindings=uppercase)


def test_fixed_observation_launch_rejects_aliases_and_unsafe_paths() -> None:
    donor_text = DONOR.read_text(encoding="utf-8")
    donor = parse_launch_text(donor_text)
    kwargs = {
        "baseline_sha256": _sha256(donor_text),
        "run_dir": RUN_DIR,
        "log": LOG,
        "bindings": _bindings(),
        "candidate_revision": CANDIDATE_REVISION,
        "candidate_tree": CANDIDATE_TREE,
    }
    donor_pythonpath = next(token for token in donor.prefix if token.startswith("PYTHONPATH="))
    donor_checkout = donor_pythonpath.removeprefix("PYTHONPATH=").split(":")[1]

    with pytest.raises(ValueError, match="must not alias Arm-NR code"):
        derive_fixed_observation_launch(
            donor_text,
            candidate_checkout=donor_checkout,
            **kwargs,
        )
    with pytest.raises(ValueError, match="canonical persistent"):
        derive_fixed_observation_launch(
            donor_text,
            candidate_checkout="/mnt/picf-next/audit-checkouts/../adr123",
            **kwargs,
        )
    with pytest.raises(ValueError, match="output paths must not alias"):
        derive_fixed_observation_launch(
            donor_text,
            candidate_checkout=CANDIDATE_CHECKOUT,
            run_dir=donor.run_dir,
            log=LOG,
            baseline_sha256=kwargs["baseline_sha256"],
            bindings=kwargs["bindings"],
            candidate_revision=CANDIDATE_REVISION,
            candidate_tree=CANDIDATE_TREE,
        )
    with pytest.raises(ValueError, match="does not contain its revision prefix"):
        derive_fixed_observation_launch(
            donor_text,
            candidate_checkout=CANDIDATE_CHECKOUT,
            candidate_revision="a" * 40,
            candidate_tree=CANDIDATE_TREE,
            run_dir=RUN_DIR,
            log=LOG,
            baseline_sha256=kwargs["baseline_sha256"],
            bindings=kwargs["bindings"],
        )


def test_fixed_observation_launch_cli_writes_executable_contract(tmp_path: Path) -> None:
    donor_text = DONOR.read_text(encoding="utf-8")
    output = tmp_path / "adr123.sh"
    report = tmp_path / "adr123.launch-contract.json"
    command = [
        sys.executable,
        str(TOOL),
        "--baseline-launch",
        str(DONOR),
        "--baseline-launch-sha256",
        _sha256(donor_text),
        "--candidate-checkout",
        CANDIDATE_CHECKOUT,
        "--candidate-revision",
        CANDIDATE_REVISION,
        "--candidate-tree",
        CANDIDATE_TREE,
        "--run-dir",
        RUN_DIR,
        "--log",
        LOG,
    ]
    bindings = _bindings()
    for option in FIXED_OBSERVATION_OPTIONS:
        command.extend((option, bindings[option]))
    command.extend(("--output", str(output), "--report", str(report)))

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert output.stat().st_mode & 0o111
    assert subprocess.run(["bash", "-n", str(output)], check=False).returncode == 0
    report_value = json.loads(report.read_text(encoding="utf-8"))
    assert (
        report_value["candidate_launch_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    )

    repeated = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert repeated.returncode != 0
    assert "already exists" in repeated.stderr

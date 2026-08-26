#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Derive the bounded ADR-123 fixed-observation launch from Arm-NR evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
from collections.abc import Mapping
from pathlib import Path
from typing import Any

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot fixed-observation launch builder",
)

from picf_next.artifact_io import write_text_durable_exclusive
from tools.build_lingbot_representation_natural_prompt_launch import (
    ParsedLaunch,
    parse_launch_text,
)


REPORT_SCHEMA = "picf-next.lingbot-fixed-observation-launch.v1"
ARM_NR_DONOR_SHA256 = "781f7a135857bab39ad47249717014d64b190a81f4cc94297bd561cef44627c0"
DONOR_ESTIMATOR = "local_balanced_sigmoid"
ADR122_ESTIMATOR = "host_native_multi_positive_softmax"
FIXED_OBSERVATION_OPTION_PAIRS = (
    (
        "--fixed-observation-pair-plan",
        "--fixed-observation-pair-plan-sha256",
    ),
    (
        "--fixed-observation-training-audit",
        "--fixed-observation-training-audit-sha256",
    ),
    (
        "--fixed-observation-evaluation-plan",
        "--fixed-observation-evaluation-plan-sha256",
    ),
    (
        "--fixed-observation-validation-audit",
        "--fixed-observation-validation-audit-sha256",
    ),
    (
        "--fixed-observation-heldout-audit",
        "--fixed-observation-heldout-audit-sha256",
    ),
)
FIXED_OBSERVATION_OPTIONS = tuple(
    option for pair in FIXED_OBSERVATION_OPTION_PAIRS for option in pair
)
FROZEN_ARM_NR_OPTIONS = {
    "--phase": "fresh",
    "--training-stage": "representation",
    "--checkpoint-publication": "never",
    "--representation-evaluation-steps": "0,200",
    "--load-global-step": "0",
    "--invocation-steps": "200",
    "--total-planned-steps": "200",
    "--seed": "20260721",
    "--capacity": "16",
    "--maximum-control-tokens": "8",
    "--maximum-optimizer-lag": "16",
    "--lane-interleave-factor": "8",
    "--reset-mixture-numerator": "1",
    "--reset-mixture-denominator": "2",
    "--maximum-peak-reserved-gib": "39",
    "--fsdp2-placement": "selective-embedding-offload",
    "--cuda-allocator": "expandable-segments",
    "--predictive-weight": "0.004",
    "--structural-weight": "0.004",
    "--gradient-audit-steps": "18,34,50,100,200",
    "--source-prediction-mode": "omitted_static",
    "--visual-audit-every": "1",
    "--task-relation-estimator": DONOR_ESTIMATOR,
}
CHECKOUT_BOUND_OPTIONS = (
    "--patch",
    "--robot-config",
    "--data-config",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _require_git_object(name: str, value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a full lowercase Git object ID")
    return value


def _validate_persistent_path(
    value: str,
    *,
    name: str,
    prefix: str,
    suffix: str = "",
) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or (suffix and not value.endswith(suffix))
        or posixpath.normpath(value) != value
        or any(
            character.isspace() or ord(character) < 32 or ord(character) > 126
            for character in value
        )
    ):
        raise ValueError(f"{name} is not a canonical persistent {prefix} path")
    return value


def _donor_checkout(launch: ParsedLaunch) -> str:
    pythonpath_tokens = tuple(token for token in launch.prefix if token.startswith("PYTHONPATH="))
    if len(pythonpath_tokens) != 1:
        raise ValueError("Arm-NR donor must contain exactly one PYTHONPATH binding")
    values = pythonpath_tokens[0].removeprefix("PYTHONPATH=").split(":")
    if len(values) != 2 or values[0] != f"{values[1]}/src":
        raise ValueError("Arm-NR donor PYTHONPATH is not one exact checkout")
    checkout = values[1]
    _validate_persistent_path(
        checkout,
        name="Arm-NR donor checkout",
        prefix="/mnt/picf-next/audit-checkouts/",
    )
    if launch.runner != f"{checkout}/tools/run_lingbot_vla2_native_full.py":
        raise ValueError("Arm-NR donor runner is outside its declared checkout")
    for option in CHECKOUT_BOUND_OPTIONS:
        value = launch.options.get(option)
        if value is None or not value.startswith(f"{checkout}/"):
            raise ValueError(f"Arm-NR donor {option} is outside its declared checkout")
    return checkout


def _validate_arm_nr_donor(launch: ParsedLaunch) -> str:
    options = launch.options
    for name, expected in FROZEN_ARM_NR_OPTIONS.items():
        if options.get(name) != expected:
            raise ValueError(
                f"Arm-NR donor {name} differs from the frozen control: "
                f"{options.get(name)!r} != {expected!r}"
            )
    required_options = (
        "--representation-split",
        "--representation-split-sha256",
        "--representation-evaluation-plan",
        "--representation-evaluation-plan-sha256",
        "--representation-warm-evaluation-plan",
        "--representation-warm-evaluation-plan-sha256",
        "--representation-evaluation-baseline",
        "--representation-evaluation-baseline-sha256",
    )
    if any(name not in options for name in required_options):
        raise ValueError("Arm-NR donor lacks its complete split/evaluation evidence")
    if any(name in options for name in FIXED_OBSERVATION_OPTIONS):
        raise ValueError("Arm-NR donor already contains fixed-observation options")
    if "--representation-task-intervention-plan" in options:
        raise ValueError("Arm-NR donor contains a legacy task intervention")
    if "--nproc_per_node=2" not in launch.prefix:
        raise ValueError("Arm-NR donor is not the frozen two-rank launch")
    if "CUDA_VISIBLE_DEVICES=0,1" not in launch.prefix:
        raise ValueError("Arm-NR donor is not bound to exactly GPU 0 and GPU 1")
    for environment_name in (
        "PICF_LINGBOT_TRAINING_AUTHORIZATION",
        "PICF_LINGBOT_TRAINING_AUTHORIZATION_SHA256",
    ):
        if not any(
            launch.prefix[index : index + 2] == ("-u", environment_name)
            for index in range(len(launch.prefix) - 1)
        ):
            raise ValueError(f"Arm-NR donor does not unset {environment_name}")
    return _donor_checkout(launch)


def _validate_fixed_bindings(bindings: Mapping[str, str]) -> dict[str, str]:
    if set(bindings) != set(FIXED_OBSERVATION_OPTIONS):
        missing = sorted(set(FIXED_OBSERVATION_OPTIONS) - set(bindings))
        extra = sorted(set(bindings) - set(FIXED_OBSERVATION_OPTIONS))
        raise ValueError(f"fixed-observation bindings differ: missing={missing}, extra={extra}")
    normalized = dict(bindings)
    paths: list[str] = []
    for path_option, digest_option in FIXED_OBSERVATION_OPTION_PAIRS:
        paths.append(
            _validate_persistent_path(
                normalized[path_option],
                name=path_option,
                prefix="/mnt/picf-next/",
                suffix=".json",
            )
        )
        _require_sha256(digest_option, normalized[digest_option])
    if len(set(paths)) != len(paths):
        raise ValueError("fixed-observation artifact paths must be distinct")
    return normalized


def _replace_checkout(value: str, *, donor: str, candidate: str) -> str:
    return value.replace(donor, candidate) if donor in value else value


def _derive_candidate_text(
    baseline_text: str,
    *,
    donor_checkout: str,
    candidate_checkout: str,
    run_dir: str,
    log: str,
    bindings: Mapping[str, str],
) -> str:
    if not baseline_text.endswith("\n"):
        raise ValueError("Arm-NR donor launch must end with a newline")
    counts = {
        "RUN_DIR": 0,
        "LOG": 0,
        "estimator": 0,
        "insertion": 0,
    }
    candidate_lines: list[str] = []
    for line in baseline_text.splitlines(keepends=True):
        stripped = line.strip()
        if line.startswith("RUN_DIR="):
            counts["RUN_DIR"] += 1
            candidate_lines.append(f"RUN_DIR={shlex.quote(run_dir)}\n")
            continue
        if line.startswith("LOG="):
            counts["LOG"] += 1
            candidate_lines.append(f"LOG={shlex.quote(log)}\n")
            continue
        if stripped.startswith("--task-relation-estimator "):
            counts["estimator"] += 1
            indentation = line[: len(line) - len(line.lstrip())]
            candidate_lines.append(
                f"{indentation}--task-relation-estimator {ADR122_ESTIMATOR} \\\n"
            )
            continue
        if stripped.startswith("--representation-evaluation-plan "):
            counts["insertion"] += 1
            indentation = line[: len(line) - len(line.lstrip())]
            for option in FIXED_OBSERVATION_OPTIONS:
                candidate_lines.append(
                    f"{indentation}{option} {shlex.quote(bindings[option])} \\\n"
                )
        candidate_lines.append(line.replace(donor_checkout, candidate_checkout))
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"Arm-NR textual launch structure differs: {counts}")
    return "".join(candidate_lines)


def _expected_candidate_pairs(
    baseline: ParsedLaunch,
    *,
    donor_checkout: str,
    candidate_checkout: str,
    bindings: Mapping[str, str],
) -> tuple[tuple[str, str], ...]:
    expected: list[tuple[str, str]] = []
    for option, value in baseline.option_pairs:
        if option == "--representation-evaluation-plan":
            expected.extend((name, bindings[name]) for name in FIXED_OBSERVATION_OPTIONS)
        if option == "--task-relation-estimator":
            value = ADR122_ESTIMATOR
        value = _replace_checkout(
            value,
            donor=donor_checkout,
            candidate=candidate_checkout,
        )
        expected.append((option, value))
    return tuple(expected)


def derive_fixed_observation_launch(
    baseline_text: str,
    *,
    baseline_sha256: str,
    candidate_checkout: str,
    candidate_revision: str,
    candidate_tree: str,
    run_dir: str,
    log: str,
    bindings: Mapping[str, str],
) -> tuple[str, dict[str, Any]]:
    baseline_sha256 = _require_sha256("baseline launch SHA-256", baseline_sha256)
    if baseline_sha256 != ARM_NR_DONOR_SHA256:
        raise ValueError("Arm-NR donor is not the pinned archived launcher")
    if _sha256_bytes(baseline_text.encode("utf-8")) != baseline_sha256:
        raise ValueError("Arm-NR donor launch SHA-256 differs")
    baseline = parse_launch_text(baseline_text)
    donor_checkout = _validate_arm_nr_donor(baseline)
    candidate_checkout = _validate_persistent_path(
        candidate_checkout,
        name="ADR-123 candidate checkout",
        prefix="/mnt/picf-next/audit-checkouts/",
    )
    if candidate_checkout == donor_checkout:
        raise ValueError("ADR-123 candidate checkout must not alias Arm-NR code")
    candidate_revision = _require_git_object(
        "ADR-123 candidate revision",
        candidate_revision,
    )
    candidate_tree = _require_git_object("ADR-123 candidate tree", candidate_tree)
    if candidate_revision[:7] not in posixpath.basename(candidate_checkout):
        raise ValueError("ADR-123 checkout name does not contain its revision prefix")
    run_dir = _validate_persistent_path(
        run_dir,
        name="ADR-123 run directory",
        prefix="/mnt/picf-next/runs/",
    )
    log = _validate_persistent_path(
        log,
        name="ADR-123 log",
        prefix="/mnt/picf-next/logs/",
        suffix=".log",
    )
    if run_dir == baseline.run_dir or log == baseline.log:
        raise ValueError("ADR-123 output paths must not alias Arm-NR evidence")
    normalized_bindings = _validate_fixed_bindings(bindings)

    candidate_text = _derive_candidate_text(
        baseline_text,
        donor_checkout=donor_checkout,
        candidate_checkout=candidate_checkout,
        run_dir=run_dir,
        log=log,
        bindings=normalized_bindings,
    )
    candidate = parse_launch_text(candidate_text)
    expected_pairs = _expected_candidate_pairs(
        baseline,
        donor_checkout=donor_checkout,
        candidate_checkout=candidate_checkout,
        bindings=normalized_bindings,
    )
    if candidate.option_pairs != expected_pairs:
        raise RuntimeError("ADR-123 changed an unregistered runner option or ordering")
    expected_prefix = tuple(
        _replace_checkout(
            token,
            donor=donor_checkout,
            candidate=candidate_checkout,
        )
        for token in baseline.prefix
    )
    if candidate.prefix != expected_prefix:
        raise RuntimeError("ADR-123 changed its environment or distributed topology")
    expected_runner = _replace_checkout(
        baseline.runner,
        donor=donor_checkout,
        candidate=candidate_checkout,
    )
    if candidate.runner != expected_runner:
        raise RuntimeError("ADR-123 changed its native runner outside the candidate checkout")
    if candidate.options["--task-relation-estimator"] != ADR122_ESTIMATOR:
        raise RuntimeError("ADR-123 did not restore the frozen ADR-122 estimator")

    candidate_sha256 = _sha256_bytes(candidate_text.encode("utf-8"))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "baseline_launch_sha256": baseline_sha256,
        "candidate_launch_sha256": candidate_sha256,
        "candidate_run_dir": run_dir,
        "candidate_log": log,
        "donor_checkout": donor_checkout,
        "candidate_checkout": candidate_checkout,
        "candidate_revision": candidate_revision,
        "candidate_tree": candidate_tree,
        "donor_estimator": DONOR_ESTIMATOR,
        "candidate_estimator": ADR122_ESTIMATOR,
        "added_options": list(FIXED_OBSERVATION_OPTIONS),
        "fixed_observation_bindings": {
            name: normalized_bindings[name] for name in FIXED_OBSERVATION_OPTIONS
        },
        "baseline_runner_option_count": len(baseline.option_pairs),
        "candidate_runner_option_count": len(candidate.option_pairs),
        "unchanged_control_option_count": len(baseline.option_pairs) - 1,
        "world_size": 2,
        "optimizer_updates_per_rank": 200,
        "evaluation_steps": [0, 200],
        "checkpoint_publication": "never",
        "scientific_training_delta": "truth_audited_same_observation_prompt_target_pairing",
        "authorizes_action_or_long_training": False,
    }
    return candidate_text, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-launch", type=Path, required=True)
    parser.add_argument("--baseline-launch-sha256", required=True)
    parser.add_argument("--candidate-checkout", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--candidate-tree", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", required=True)
    for path_option, digest_option in FIXED_OBSERVATION_OPTION_PAIRS:
        parser.add_argument(path_option, required=True)
        parser.add_argument(digest_option, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.report.exists()
        or args.report.is_symlink()
    ):
        raise FileExistsError("ADR-123 launch or report output already exists")
    bindings = {
        option: getattr(args, option.removeprefix("--").replace("-", "_"))
        for option in FIXED_OBSERVATION_OPTIONS
    }
    candidate_text, report = derive_fixed_observation_launch(
        args.baseline_launch.read_text(encoding="utf-8"),
        baseline_sha256=args.baseline_launch_sha256,
        candidate_checkout=args.candidate_checkout,
        candidate_revision=args.candidate_revision,
        candidate_tree=args.candidate_tree,
        run_dir=args.run_dir,
        log=args.log,
        bindings=bindings,
    )
    write_text_durable_exclusive(args.output, candidate_text)
    os.chmod(args.output, 0o700)
    write_text_durable_exclusive(
        args.report,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()

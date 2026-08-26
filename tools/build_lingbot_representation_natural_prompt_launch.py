#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Derive the ADR-120 natural-prompt launch from the archived ADR-117 launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__:
    from tools.repository_import import bind_entrypoint_to_own_repository
else:
    from repository_import import bind_entrypoint_to_own_repository

bind_entrypoint_to_own_repository(
    __file__,
    entrypoint_name="LingBot representation natural-prompt launch builder",
)

from picf_next.artifact_io import write_text_durable_exclusive


REPORT_SCHEMA = "picf-next.lingbot-representation-natural-prompt-launch.v1"
RUNNER_SUFFIX = "/tools/run_lingbot_vla2_native_full.py"
INTERVENTION_OPTIONS = (
    "--representation-task-intervention-plan",
    "--representation-task-intervention-plan-sha256",
)
FROZEN_ARM_N_OPTIONS = {
    "--phase": "fresh",
    "--training-stage": "representation",
    "--checkpoint-publication": "never",
    "--representation-evaluation-steps": "0,200",
    "--load-global-step": "0",
    "--invocation-steps": "200",
    "--total-planned-steps": "200",
    "--seed": "20260721",
    "--capacity": "16",
    "--maximum-optimizer-lag": "8",
    "--lane-interleave-factor": "8",
    "--gradient-audit-steps": "9,17,20,50,100,200",
    "--visual-audit-every": "1",
}


@dataclass(frozen=True)
class ParsedLaunch:
    run_dir: str
    log: str
    prefix: tuple[str, ...]
    runner: str
    option_pairs: tuple[tuple[str, str], ...]

    @property
    def options(self) -> dict[str, str]:
        return dict(self.option_pairs)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if len(value) != 64:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    try:
        decoded = bytes.fromhex(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a lowercase SHA-256") from error
    if decoded.hex() != value:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _logical_lines(text: str) -> tuple[str, ...]:
    if "\r" in text:
        raise ValueError("launch script must use LF line endings")
    lines = tuple(line for line in text.replace("\\\n", " ").splitlines() if line.strip())
    if len(lines) != 6:
        raise ValueError("launch script must contain the frozen six-command structure")
    return lines


def _assignment(line: str, name: str) -> str:
    tokens = shlex.split(line, posix=True)
    prefix = f"{name}="
    if len(tokens) != 1 or not tokens[0].startswith(prefix):
        raise ValueError(f"launch script has a malformed {name} assignment")
    value = tokens[0][len(prefix) :]
    if not value:
        raise ValueError(f"launch script has an empty {name} assignment")
    return value


def parse_launch_text(text: str) -> ParsedLaunch:
    lines = _logical_lines(text)
    if lines[0] != "#!/usr/bin/env bash" or lines[1] != "set -euo pipefail":
        raise ValueError("launch script lost its fail-closed shell prologue")
    run_dir = _assignment(lines[2], "RUN_DIR")
    log = _assignment(lines[3], "LOG")
    if shlex.split(lines[4], posix=True) != ["echo", "$$", ">", "$RUN_DIR/launcher.pid"]:
        raise ValueError("launch script lost its launcher PID publication")

    tokens = shlex.split(lines[5], posix=True)
    if len(tokens) < 8 or tokens[:2] != ["exec", "env"]:
        raise ValueError("launch script must exec the frozen environment command")
    if tokens[-2:] != [">$LOG", "2>&1"]:
        raise ValueError("launch script lost its exact log redirection")
    runner_indices = [
        index for index, token in enumerate(tokens[:-2]) if token.endswith(RUNNER_SUFFIX)
    ]
    if len(runner_indices) != 1:
        raise ValueError("launch script must invoke exactly one native full runner")
    runner_index = runner_indices[0]
    prefix = tuple(tokens[1:runner_index])
    runner = tokens[runner_index]
    argument_tokens = tokens[runner_index + 1 : -2]
    if len(argument_tokens) % 2:
        raise ValueError("native full runner arguments must be option/value pairs")
    option_pairs = tuple(
        (argument_tokens[index], argument_tokens[index + 1])
        for index in range(0, len(argument_tokens), 2)
    )
    if any(not option.startswith("--") for option, _value in option_pairs):
        raise ValueError("native full runner contains a positional or malformed argument")
    option_names = tuple(option for option, _value in option_pairs)
    if len(set(option_names)) != len(option_names):
        raise ValueError("native full runner contains duplicate options")
    options = dict(option_pairs)
    if options.get("--run-dir") != "$RUN_DIR":
        raise ValueError("native full runner must bind --run-dir to RUN_DIR")
    return ParsedLaunch(
        run_dir=run_dir,
        log=log,
        prefix=prefix,
        runner=runner,
        option_pairs=option_pairs,
    )


def _validate_adr117_baseline(launch: ParsedLaunch) -> None:
    options = launch.options
    if any(options.get(name) is None for name in INTERVENTION_OPTIONS):
        raise ValueError("ADR-117 baseline lacks its complete donor intervention")
    for name, expected in FROZEN_ARM_N_OPTIONS.items():
        if options.get(name) != expected:
            raise ValueError(
                f"ADR-117 baseline {name} differs from the frozen Arm-N control: "
                f"{options.get(name)!r} != {expected!r}"
            )
    if "--nproc_per_node=2" not in launch.prefix:
        raise ValueError("ADR-117 baseline is not the frozen two-rank launch")
    if "CUDA_VISIBLE_DEVICES=0,1" not in launch.prefix:
        raise ValueError("ADR-117 baseline is not bound to exactly GPU 0 and GPU 1")


def _validate_persistent_output_path(
    value: str,
    *,
    name: str,
    prefix: str,
    suffix: str = "",
) -> None:
    if (
        not value.startswith(prefix)
        or (suffix and not value.endswith(suffix))
        or posixpath.normpath(value) != value
        or any(
            character.isspace() or ord(character) < 32 or ord(character) > 126
            for character in value
        )
    ):
        raise ValueError(f"Arm-N {name} is not a canonical persistent {prefix} path")


def _derive_candidate_text(
    baseline_text: str,
    *,
    run_dir: str,
    log: str,
) -> str:
    if not baseline_text.endswith("\n"):
        raise ValueError("ADR-117 baseline launch must end with a newline")
    counts = {"RUN_DIR": 0, "LOG": 0, **dict.fromkeys(INTERVENTION_OPTIONS, 0)}
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
        removed = next(
            (option for option in INTERVENTION_OPTIONS if stripped.startswith(f"{option} ")),
            None,
        )
        if removed is not None:
            counts[removed] += 1
            continue
        candidate_lines.append(line)
    if any(count != 1 for count in counts.values()):
        raise ValueError(f"ADR-117 textual launch structure differs: {counts}")
    return "".join(candidate_lines)


def derive_natural_prompt_launch(
    baseline_text: str,
    *,
    baseline_sha256: str,
    run_dir: str,
    log: str,
) -> tuple[str, dict[str, Any]]:
    baseline_sha256 = _require_sha256("baseline launch SHA-256", baseline_sha256)
    observed_baseline_sha256 = _sha256_bytes(baseline_text.encode("utf-8"))
    if observed_baseline_sha256 != baseline_sha256:
        raise ValueError("ADR-117 baseline launch SHA-256 differs")
    baseline = parse_launch_text(baseline_text)
    _validate_adr117_baseline(baseline)

    _validate_persistent_output_path(
        run_dir,
        name="run directory",
        prefix="/mnt/picf-next/runs/",
    )
    _validate_persistent_output_path(
        log,
        name="log path",
        prefix="/mnt/picf-next/logs/",
        suffix=".log",
    )
    if run_dir == baseline.run_dir or log == baseline.log:
        raise ValueError("Arm-N output paths must not alias ADR-117 evidence")

    candidate_pairs = tuple(
        pair for pair in baseline.option_pairs if pair[0] not in INTERVENTION_OPTIONS
    )
    candidate_text = _derive_candidate_text(
        baseline_text,
        run_dir=run_dir,
        log=log,
    )
    reparsed = parse_launch_text(candidate_text)

    baseline_without_intervention = tuple(
        pair for pair in baseline.option_pairs if pair[0] not in INTERVENTION_OPTIONS
    )
    if reparsed.option_pairs != baseline_without_intervention:
        raise RuntimeError("Arm-N changed a runner option other than donor intervention")
    if reparsed.prefix != baseline.prefix or reparsed.runner != baseline.runner:
        raise RuntimeError("Arm-N changed its environment, launcher, topology, or runner")
    if any(name in reparsed.options for name in INTERVENTION_OPTIONS):
        raise RuntimeError("Arm-N retained a donor intervention option")

    candidate_sha256 = _sha256_bytes(candidate_text.encode("utf-8"))
    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "status": "PASS",
        "baseline_launch_sha256": baseline_sha256,
        "candidate_launch_sha256": candidate_sha256,
        "candidate_run_dir": run_dir,
        "candidate_log": log,
        "removed_options": list(INTERVENTION_OPTIONS),
        "unchanged_runner_option_count": len(candidate_pairs),
        "gradient_audit_steps": reparsed.options["--gradient-audit-steps"],
        "visual_audit_every": int(reparsed.options["--visual-audit-every"]),
        "world_size": 2,
        "training_state_delta": "donor_intervention_absent_use_natural_source_prompt",
        "other_runner_delta_count": 0,
    }
    return candidate_text, report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-launch", type=Path, required=True)
    parser.add_argument("--baseline-launch-sha256", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError("Arm-N launch or report output already exists")
    baseline_text = args.baseline_launch.read_text(encoding="utf-8")
    candidate_text, report = derive_natural_prompt_launch(
        baseline_text,
        baseline_sha256=args.baseline_launch_sha256,
        run_dir=args.run_dir,
        log=args.log,
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

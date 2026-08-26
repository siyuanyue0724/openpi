#!/usr/bin/env python3
"""Audit one matched MolmoAct2 M4 Arm-A/Arm-C training prefix.

The report is descriptive evidence for the bounded M4 gate. It validates the
frozen sample/randomness contract, retains every aligned loss point, uses a
moving-block bootstrap for the temporally correlated trajectory, reconstructs
task/episode provenance, and combines the curve with a read-only posterior
intervention report. It does not authorize longer training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-a-run", type=Path, required=True)
    parser.add_argument("--arm-c-run", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--dataset-split-root", type=Path, required=True)
    parser.add_argument("--intervention-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-steps", type=int, default=20)
    parser.add_argument("--bootstrap-block-steps", type=int, default=20)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=17_291)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} is not valid ASCII JSON: {path}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="ascii").splitlines()
    except (OSError, UnicodeDecodeError) as error:
        raise ValueError(f"metrics are not readable ASCII JSONL: {path}") from error
    rows = []
    for line_number, line in enumerate(lines, start=1):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid metrics JSON on line {line_number}: {path}") from error
        if not isinstance(row, dict):
            raise ValueError(f"metrics line {line_number} must contain one JSON object")
        rows.append(row)
    if not rows:
        raise ValueError(f"metrics JSONL is empty: {path}")
    return rows


def _finite_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _git_revision(root: Path) -> str:
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if len(revision) != 40:
        raise RuntimeError("audit source revision is not one full commit")
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if dirty:
        raise RuntimeError("paired M4 audit requires one clean committed worktree")
    return revision


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "ascii"
    )
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validated_curve(rows: Sequence[Mapping[str, Any]], *, arm: str) -> list[float]:
    curve = []
    for index, row in enumerate(rows, start=1):
        if row.get("attempted_optimizer_steps") != index:
            raise ValueError(f"Arm {arm} attempted-step sequence is not contiguous")
        if row.get("successful_optimizer_steps") != index:
            raise ValueError(f"Arm {arm} successful-step sequence is not contiguous")
        if row.get("optimizer_step_skipped") is not False:
            raise ValueError(f"Arm {arm} contains a skipped optimizer step")
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError(f"Arm {arm} metrics row {index} has no metrics mapping")
        loss = _finite_float(metrics.get("action_flow_loss"), f"Arm {arm} action loss")
        for alias in ("loss", "picf_loss_action", "system_optimizer_loss"):
            if _finite_float(metrics.get(alias), f"Arm {arm} {alias}") != loss:
                raise ValueError(f"Arm {arm} action-loss aliases differ at step {index}")
        curve.append(loss)
    return curve


def _curve_summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        raise ValueError("curve summary requires at least one value")
    return {
        "final": values[-1],
        "maximum": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "minimum": min(values),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile inputs are invalid")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _moving_block_mean_interval(
    deltas: Sequence[float],
    *,
    block_steps: int,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    if (
        not deltas
        or not isinstance(block_steps, int)
        or isinstance(block_steps, bool)
        or not 1 <= block_steps <= len(deltas)
        or not isinstance(replicates, int)
        or isinstance(replicates, bool)
        or replicates < 1_000
        or not isinstance(seed, int)
        or isinstance(seed, bool)
        or seed < 0
    ):
        raise ValueError("moving-block bootstrap configuration is invalid")
    rng = random.Random(seed)
    block_count = math.ceil(len(deltas) / block_steps)
    last_start = len(deltas) - block_steps
    means = []
    for _ in range(replicates):
        resample = []
        for _ in range(block_count):
            start = rng.randrange(last_start + 1)
            resample.extend(deltas[start : start + block_steps])
        means.append(statistics.fmean(resample[: len(deltas)]))
    means.sort()
    return {
        "block_steps": block_steps,
        "interpretation": "descriptive_temporal_trajectory_interval_not_generalization_ci",
        "lower_95": _quantile(means, 0.025),
        "replicates": replicates,
        "seed": seed,
        "upper_95": _quantile(means, 0.975),
    }


def _paired_summary(
    arm_a: Sequence[float],
    arm_c: Sequence[float],
    *,
    window_steps: int,
    block_steps: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(arm_a) != len(arm_c) or not arm_a:
        raise ValueError("paired curves must have one nonempty aligned length")
    if not 1 <= window_steps <= len(arm_a):
        raise ValueError("paired window size is invalid")
    deltas = [c_value - a_value for a_value, c_value in zip(arm_a, arm_c, strict=True)]
    windows = []
    for start in range(0, len(deltas), window_steps):
        stop = min(start + window_steps, len(deltas))
        selected = deltas[start:stop]
        windows.append(
            {
                "arm_c_wins": sum(value < 0.0 for value in selected),
                "arm_c_losses": sum(value > 0.0 for value in selected),
                "exact_ties": sum(value == 0.0 for value in selected),
                "mean_loss_delta_c_minus_a": statistics.fmean(selected),
                "start_step_one_based": start + 1,
                "stop_step_one_based_inclusive": stop,
            }
        )
    mean_delta = statistics.fmean(deltas)
    return {
        "arm_c_losses": sum(value > 0.0 for value in deltas),
        "arm_c_wins": sum(value < 0.0 for value in deltas),
        "descriptive_moving_block_bootstrap": _moving_block_mean_interval(
            deltas,
            block_steps=block_steps,
            replicates=bootstrap_replicates,
            seed=bootstrap_seed,
        ),
        "exact_ties": sum(value == 0.0 for value in deltas),
        "mean_loss_delta_c_minus_a": mean_delta,
        "median_loss_delta_c_minus_a": statistics.median(deltas),
        "relative_mean_delta_fraction_of_arm_a": mean_delta / statistics.fmean(arm_a),
        "windows": windows,
    }


def _metric_values(
    rows: Sequence[Mapping[str, Any]],
    name: str,
    *,
    start: int = 0,
) -> list[float]:
    values = []
    for row in rows[start:]:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("metrics row has no metrics mapping")
        values.append(_finite_float(metrics.get(name), name))
    return values


def _system_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    wall = _metric_values(rows, "system_train_step_wall_seconds_rank_max", start=1)
    if not wall:
        wall = _metric_values(rows, "system_train_step_wall_seconds_rank_max")
    peak = _metric_values(rows, "system_cuda_peak_allocated_bytes_rank_max")
    return {
        "peak_allocated_bytes_max": max(peak),
        "step_wall_seconds_mean_excluding_first": statistics.fmean(wall),
        "step_wall_seconds_median_excluding_first": statistics.median(wall),
    }


def _arm_run(run_dir: Path, *, expected_arm: str) -> dict[str, Any]:
    run = run_dir.expanduser().resolve()
    paths = {
        "checkpoint_audit": run / "smoke_checkpoint_audit.json",
        "metrics": run / "metrics.jsonl",
        "plan": run / "sample_plan.json",
        "static": run / "static_preflight.json",
    }
    if any(not path.is_file() for path in paths.values()):
        missing = [name for name, path in paths.items() if not path.is_file()]
        raise ValueError(f"Arm {expected_arm} run is incomplete: {missing}")
    static = _read_json(paths["static"], f"Arm {expected_arm} static preflight")
    plan = _read_json(paths["plan"], f"Arm {expected_arm} sample plan")
    checkpoint_audit = _read_json(paths["checkpoint_audit"], f"Arm {expected_arm} checkpoint audit")
    factorization = static.get("causal_factorization")
    expected_posterior = expected_arm == "C"
    if (
        not isinstance(factorization, dict)
        or factorization.get("id") != expected_arm
        or factorization.get("include_causal_video") is not False
        or factorization.get("include_posterior_action_context") is not expected_posterior
    ):
        raise ValueError(f"Arm {expected_arm} causal factorization is not the matched A/C arm")
    if checkpoint_audit.get("status") != "PASS":
        raise ValueError(f"Arm {expected_arm} checkpoint audit did not pass")
    if checkpoint_audit.get("causal_factorization") != factorization:
        raise ValueError(f"Arm {expected_arm} checkpoint/static factorization differs")
    rows = _read_metrics(paths["metrics"])
    curve = _validated_curve(rows, arm=expected_arm)
    metadata = plan.get("metadata")
    if (
        not isinstance(metadata, dict)
        or metadata.get("total_steps") != len(rows)
        or plan.get("plan_sha256") != static.get("plan_sha256")
        or checkpoint_audit.get("successful_optimizer_steps") != len(rows)
    ):
        raise ValueError(f"Arm {expected_arm} plan, run, and checkpoint lengths differ")
    return {
        "checkpoint_audit": checkpoint_audit,
        "curve": curve,
        "factorization": factorization,
        "paths": paths,
        "plan": plan,
        "rows": rows,
        "run": run,
        "static": static,
    }


def _paired_contract(arm_a: Mapping[str, Any], arm_c: Mapping[str, Any]) -> dict[str, Any]:
    if arm_a["plan"] != arm_c["plan"]:
        raise ValueError("Arm A and Arm C sample/randomness plans differ")
    common_static_fields = (
        "artifacts",
        "dataset_samples",
        "episode_count",
        "m0_report_validated",
        "plan_sha256",
        "recipe_sha256",
        "schema",
        "stationary_temporal_core_validated",
        "stationary_temporal_initialization",
        "vjepa2_cache",
    )
    for name in common_static_fields:
        if arm_a["static"].get(name) != arm_c["static"].get(name):
            raise ValueError(f"Arm A and Arm C static field differs: {name}")
    return {
        "comparison_id": arm_a["plan"]["metadata"].get("comparison_id"),
        "plan": arm_a["plan"],
        "recipe_sha256": arm_a["static"]["recipe_sha256"],
        "stationary_temporal_initialization": arm_a["static"]["stationary_temporal_initialization"],
    }


def _task_contexts(
    *,
    recipe_path: Path,
    split_root: Path,
    paired_contract: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], str]:
    from picf_next.training.molmoact2_calvin import (
        build_calvin_episode_stream_plan,
        load_calvin_training_assets,
    )
    from picf_next.training.recipe import load_training_recipe

    root = Path(__file__).resolve().parents[1]
    recipe = load_training_recipe(recipe_path.expanduser().resolve())
    if recipe.recipe_sha256 != paired_contract["recipe_sha256"]:
        raise ValueError("task-context recipe differs from paired training runs")
    assets = load_calvin_training_assets(
        recipe,
        repository_root=root,
        split_root=split_root.expanduser().resolve(),
    )
    metadata = paired_contract["plan"]["metadata"]
    plan = build_calvin_episode_stream_plan(
        recipe,
        assets.dataset,
        comparison_id=str(metadata["comparison_id"]),
        seed=int(metadata["seed"]),
        global_batch_size=int(metadata["global_batch_size"]),
        total_steps=int(metadata["total_steps"]),
    )
    if plan.plan_sha256 != paired_contract["plan"]["plan_sha256"]:
        raise ValueError("reconstructed task-context plan differs from paired run")
    if len(assets.dataset.episode_manifest) != len(assets.dataset.index.segments):
        raise ValueError("CALVIN episode and language-segment manifests differ")
    segment_by_episode = {
        episode.episode_key: segment
        for episode, segment in zip(
            assets.dataset.episode_manifest,
            assets.dataset.index.segments,
            strict=True,
        )
    }
    contexts = []
    for step in range(plan.total_steps):
        lanes = []
        for transition in plan.global_batch(step).transitions:
            segment = segment_by_episode[transition.episode_key]
            lanes.append(
                {
                    "episode_instance_id": transition.episode_instance_id,
                    "lane_id": transition.lane_id,
                    "sample_key": transition.sample.sample_key,
                    "task": segment.instruction,
                    "task_index": segment.index,
                    "task_key": segment.task_key,
                    "transition_index": transition.transition_index,
                }
            )
        contexts.append({"lanes": lanes, "optimizer_step_one_based": step + 1})
    return contexts, plan.plan_sha256


def _task_segments(
    contexts: Sequence[Mapping[str, Any]],
    deltas: Sequence[float],
) -> list[dict[str, Any]]:
    if len(contexts) != len(deltas) or not contexts:
        raise ValueError("task contexts must align with paired deltas")
    segments = []
    start = 0

    def identity(context: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
        lanes = context.get("lanes")
        if not isinstance(lanes, list) or not lanes:
            raise ValueError("task context has no lane metadata")
        return tuple((str(lane["lane_id"]), str(lane["episode_instance_id"])) for lane in lanes)

    def publish(stop: int) -> None:
        first = contexts[start]
        last = contexts[stop - 1]
        first_lanes = first["lanes"]
        last_lanes = last["lanes"]
        lanes = []
        for begin, end in zip(first_lanes, last_lanes, strict=True):
            if begin["lane_id"] != end["lane_id"] or begin["task"] != end["task"]:
                raise ValueError("task-segment lane identity changed inside one segment")
            lanes.append(
                {
                    "episode_instance_id": begin["episode_instance_id"],
                    "lane_id": begin["lane_id"],
                    "start_transition_index": begin["transition_index"],
                    "stop_transition_index_inclusive": end["transition_index"],
                    "task": begin["task"],
                    "task_index": begin["task_index"],
                    "task_key": begin["task_key"],
                }
            )
        selected = deltas[start:stop]
        segments.append(
            {
                "arm_c_losses": sum(value > 0.0 for value in selected),
                "arm_c_wins": sum(value < 0.0 for value in selected),
                "exact_ties": sum(value == 0.0 for value in selected),
                "lanes": lanes,
                "mean_loss_delta_c_minus_a": statistics.fmean(selected),
                "start_step_one_based": start + 1,
                "stop_step_one_based_inclusive": stop,
            }
        )

    current = identity(contexts[0])
    for index in range(1, len(contexts)):
        next_identity = identity(contexts[index])
        if next_identity != current:
            publish(index)
            start = index
            current = next_identity
    publish(len(contexts))
    return segments


def _intervention_summary(
    report: Mapping[str, Any],
    *,
    paired_contract: Mapping[str, Any],
    completed_steps: int,
) -> tuple[dict[str, Any], dict[str, bool]]:
    if report.get("schema") != "picf-next.m4-action-intervention-audit.v1":
        raise ValueError("unsupported M4 action intervention schema")
    aggregate = report.get("aggregate")
    checkpoint = report.get("checkpoint")
    plan = report.get("plan")
    if (
        not isinstance(aggregate, dict)
        or not isinstance(checkpoint, dict)
        or not isinstance(plan, dict)
    ):
        raise ValueError("M4 action intervention report is incomplete")
    if checkpoint.get("completed_optimizer_steps") != completed_steps:
        raise ValueError("M4 intervention checkpoint length differs from paired curves")
    if plan.get("checkpoint_plan_sha256") != paired_contract["plan"]["plan_sha256"]:
        raise ValueError("M4 intervention plan differs from paired curves")
    if report.get("recipe_sha256") != paired_contract["recipe_sha256"]:
        raise ValueError("M4 intervention recipe differs from paired curves")
    conditions = aggregate.get("conditions")
    rank_count = aggregate.get("rank_count")
    if not isinstance(conditions, dict) or not isinstance(rank_count, int) or rank_count <= 0:
        raise ValueError("M4 intervention aggregate is malformed")

    def all_ranks_worse(name: str) -> bool:
        condition = conditions.get(name)
        if not isinstance(condition, dict):
            raise ValueError(f"M4 intervention omits {name}")
        delta = _finite_float(condition.get("loss_delta_from_baseline"), f"{name} loss delta")
        return delta > 0.0 and condition.get("positive_loss_delta_ranks") == rank_count

    joint = conditions.get("joint_row_permutation")
    if not isinstance(joint, dict):
        raise ValueError("M4 intervention omits joint row permutation")
    checks = {
        "baseline_replay_exact_all_ranks": aggregate.get("all_baseline_replays_exact") is True,
        "joint_row_permutation_exact": (
            _finite_float(joint.get("loss_delta_from_baseline"), "joint loss delta") == 0.0
            and _finite_float(joint.get("velocity_rms_from_baseline"), "joint velocity RMS") == 0.0
        ),
        "posterior_effect_measurable": (
            _finite_float(aggregate.get("maximum_causal_velocity_rms"), "causal velocity RMS") > 0.0
        ),
        "correct_beats_no_posterior_all_ranks": all_ranks_worse("without_posterior"),
        "correct_beats_wrong_address_all_ranks": all_ranks_worse("wrong_address"),
        "correct_beats_removed_max_prior_all_ranks": all_ranks_worse("remove_max_prior_row"),
        "correct_beats_stale_previous_all_ranks": all_ranks_worse("stale_previous_frame"),
        "oracle_intervention_present": "oracle" in conditions,
    }
    return {"aggregate": aggregate, "gates": report.get("gates")}, checks


def build_report(
    *,
    arm_a_run: Path,
    arm_c_run: Path,
    recipe: Path,
    dataset_split_root: Path,
    intervention_report: Path,
    window_steps: int,
    bootstrap_block_steps: int,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    arm_a = _arm_run(arm_a_run, expected_arm="A")
    arm_c = _arm_run(arm_c_run, expected_arm="C")
    contract = _paired_contract(arm_a, arm_c)
    contexts, reconstructed_plan_sha256 = _task_contexts(
        recipe_path=recipe,
        split_root=dataset_split_root,
        paired_contract=contract,
    )
    paired = _paired_summary(
        arm_a["curve"],
        arm_c["curve"],
        window_steps=window_steps,
        block_steps=bootstrap_block_steps,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_seed=bootstrap_seed,
    )
    deltas = [
        c_value - a_value for a_value, c_value in zip(arm_a["curve"], arm_c["curve"], strict=True)
    ]
    intervention_path = intervention_report.expanduser().resolve()
    intervention_payload = _read_json(intervention_path, "M4 action intervention report")
    intervention, intervention_checks = _intervention_summary(
        intervention_payload,
        paired_contract=contract,
        completed_steps=len(deltas),
    )
    arm_a_system = _system_summary(arm_a["rows"])
    arm_c_system = _system_summary(arm_c["rows"])
    c_checkpoint_gates = arm_c["checkpoint_audit"].get("gates_after_optimizer_step")
    if not isinstance(c_checkpoint_gates, dict):
        raise ValueError("Arm C checkpoint audit omits residual gates")
    object_gates = c_checkpoint_gates.get("object")
    if not isinstance(object_gates, dict):
        raise ValueError("Arm C checkpoint audit omits object residual gates")
    checks = {
        "arm_c_mean_action_loss_not_worse": paired["mean_loss_delta_c_minus_a"] <= 0.0,
        "arm_c_object_route_nonzero_all_layers": (
            object_gates.get("nonzero_count") == object_gates.get("count")
            and isinstance(object_gates.get("count"), int)
            and object_gates["count"] > 0
        ),
        "paired_plan_reconstructed_exactly": (
            reconstructed_plan_sha256 == contract["plan"]["plan_sha256"]
        ),
        **intervention_checks,
    }
    required_for_m4 = (
        "arm_c_mean_action_loss_not_worse",
        "arm_c_object_route_nonzero_all_layers",
        "baseline_replay_exact_all_ranks",
        "joint_row_permutation_exact",
        "posterior_effect_measurable",
        "correct_beats_no_posterior_all_ranks",
        "correct_beats_wrong_address_all_ranks",
        "correct_beats_removed_max_prior_all_ranks",
        "correct_beats_stale_previous_all_ranks",
        "oracle_intervention_present",
    )
    m4_accepted = all(checks[name] for name in required_for_m4)
    artifacts = {}
    for arm, payload in (("A", arm_a), ("C", arm_c)):
        artifacts[arm] = {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in payload["paths"].items()
        }
    artifacts["intervention"] = {
        "path": str(intervention_path),
        "sha256": _sha256(intervention_path),
    }
    return {
        "artifacts": artifacts,
        "audit_source_revision": _git_revision(Path(__file__).resolve().parents[1]),
        "checks": checks,
        "curves": {
            "arm_a": _curve_summary(arm_a["curve"]),
            "arm_c": _curve_summary(arm_c["curve"]),
            "paired": paired,
            "step_count": len(deltas),
        },
        "intervention": intervention,
        "m4_acceptance": "PASS" if m4_accepted else "FAIL",
        "paired_contract": contract,
        "schema": "picf-next.m4-paired-curve-audit.v1",
        "system": {
            "arm_a": arm_a_system,
            "arm_c": arm_c_system,
            "arm_c_overhead": {
                "peak_allocated_bytes": (
                    arm_c_system["peak_allocated_bytes_max"]
                    - arm_a_system["peak_allocated_bytes_max"]
                ),
                "wall_mean_fraction": (
                    arm_c_system["step_wall_seconds_mean_excluding_first"]
                    / arm_a_system["step_wall_seconds_mean_excluding_first"]
                    - 1.0
                ),
                "wall_median_fraction": (
                    arm_c_system["step_wall_seconds_median_excluding_first"]
                    / arm_a_system["step_wall_seconds_median_excluding_first"]
                    - 1.0
                ),
            },
        },
        "task_segments": _task_segments(contexts, deltas),
    }


def main() -> None:
    args = _parse_args()
    report = build_report(
        arm_a_run=args.arm_a_run,
        arm_c_run=args.arm_c_run,
        recipe=args.recipe,
        dataset_split_root=args.dataset_split_root,
        intervention_report=args.intervention_report,
        window_steps=args.window_steps,
        bootstrap_block_steps=args.bootstrap_block_steps,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = args.output.expanduser().resolve()
    if Path("/mnt") not in output.parents:
        raise ValueError("cloud M4 paired audit output must be a strict descendant of /mnt")
    _atomic_json(output, report)
    print(json.dumps({"output": str(output), "sha256": _sha256(output)}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Compare matched WLA-LBOT masked and PICF-full training curves.

This is an early optimization diagnostic, not an action-capability gate.  It
fails closed unless both arms used the same implementation, stream, random
draws, WLA targets, and parameter graph.  A held-out action evaluation and a
CALVIN rollout remain necessary before long-training promotion.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "picf-next.adr224-wla-host-evidence-curve-comparison/v1"
FULL_ARM = "picf_full"
MASKED_ARM = "wla_lbot_masked"
WLA_ACTION_BACKEND = "wla_complete"

_MANIFEST_INVARIANTS = (
    "schema",
    "world_size",
    "global_batch_size",
    "implementation_sha256",
    "model_family_sha256",
    "stream_plan_sha256",
    "representation_split_artifact_sha256",
    "evaluation_plan_artifact_sha256",
    "parameter_manifest",
    "objective",
    "trainable_scope",
    "action_fsdp2_topology",
    "vlm_fsdp2_topology",
)
_RECORD_IDENTITIES = (
    "global_step",
    "sample_keys",
    "source_digest",
    "temporal_plan_sha256",
    "augmentation_seeds",
    "flow_noise_seeds",
    "flow_timestep_seeds",
    "frame_indices",
    "lane_ids",
    "reset",
    "local_bptt_steps",
    "optimizer_lags",
)
_HELDOUT_SAMPLE_IDENTITIES = (
    "partition",
    "ordinal",
    "task_key",
    "segment_index",
    "source_episode_index",
    "source_global_index",
    "transition_index",
    "sample_key",
    "source_digest",
    "model_inputs_sha256",
    "native_source_rgb_sha256",
    "native_source_query_count",
    "prior_control_chunk_count",
)


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode(
            "ascii"
        )
    ).hexdigest()


def _parse_window(value: str) -> tuple[int, int]:
    try:
        start, end = (int(item) for item in value.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("window must be START:END") from error
    if start <= 0 or end < start:
        raise argparse.ArgumentTypeError("window bounds must satisfy 0 < START <= END")
    return start, end


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-run-dir", type=Path, required=True)
    parser.add_argument("--masked-run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=_parse_window, action="append", default=[])
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument("--minimum-relative-lead", type=float, default=0.05)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _normalized_execution_contract(
    manifest: Mapping[str, Any], *, expected_arm: str
) -> dict[str, Any]:
    contract = manifest.get("execution_contract")
    if not isinstance(contract, dict):
        raise ValueError("run manifest has no execution contract")
    normalized = copy.deepcopy(contract)
    arm = normalized.pop("wla_host_evidence_arm", None)
    if arm != expected_arm:
        raise ValueError(f"expected {expected_arm!r} arm, found {arm!r}")
    return normalized


def validate_matched_manifests(
    full: Mapping[str, Any], masked: Mapping[str, Any]
) -> dict[str, Any]:
    if full.get("status") != "PASS" or masked.get("status") != "PASS":
        raise ValueError("both run manifests must pass")
    for field in _MANIFEST_INVARIANTS:
        if full.get(field) != masked.get(field):
            raise ValueError(f"matched run manifests differ at {field}")
    if full.get("early_stop_step") != masked.get("early_stop_step"):
        raise ValueError("matched runs stop at different steps")
    full_contract = _normalized_execution_contract(full, expected_arm=FULL_ARM)
    masked_contract = _normalized_execution_contract(masked, expected_arm=MASKED_ARM)
    if full_contract != masked_contract:
        raise ValueError("execution contracts differ beyond the evidence arm")
    return {
        "implementation_sha256": full["implementation_sha256"],
        "model_family_sha256": full["model_family_sha256"],
        "stream_plan_sha256": full["stream_plan_sha256"],
        "execution_contract_without_arm_sha256": _canonical_sha256(full_contract),
    }


def _load_rank_journals(run_dir: Path, *, expected_world_size: int) -> dict[tuple[int, int], dict[str, Any]]:
    root = run_dir / "metrics" / "rank_journal"
    records: dict[tuple[int, int], dict[str, Any]] = {}
    for rank in range(expected_world_size):
        path = root / f"rank_{rank}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(path)
        previous_step = 0
        for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"rank journal line is not an object: {path}:{line_number}")
            step = value.get("global_step")
            if not isinstance(step, int) or step != previous_step + 1:
                raise ValueError(f"rank journal is not contiguous: {path}:{line_number}")
            records[(step, rank)] = value
            previous_step = step
    return records


def _nested(record: Mapping[str, Any], *keys: str) -> Any:
    value: Any = record
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"record omits {'.'.join(keys)}")
        value = value[key]
    return value


def _positive_loss(record: Mapping[str, Any], name: str) -> float:
    value = float(_nested(record, "wla_action_world", "metrics", name))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _validate_record_pair(
    full: Mapping[str, Any], masked: Mapping[str, Any]
) -> None:
    if full.get("wla_host_evidence_arm") != FULL_ARM:
        raise ValueError("full journal record names another evidence arm")
    if masked.get("wla_host_evidence_arm") != MASKED_ARM:
        raise ValueError("masked journal record names another evidence arm")
    for field in _RECORD_IDENTITIES:
        if full.get(field) != masked.get(field):
            raise ValueError(f"matched journal records differ at {field}")
    for path in (
        ("wla_action_world", "target_source_global_indices"),
        ("wla_action_world", "target_source_rgb_sha256"),
        ("wla_action_world", "world_loss_weight"),
        ("wla_action_world", "optimizer_contract"),
        ("videomt_source_objective", "global_indices"),
        ("videomt_source_objective", "query_count"),
    ):
        if _nested(full, *path) != _nested(masked, *path):
            raise ValueError(f"matched journal records differ at {'.'.join(path)}")


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _moving_block_interval(
    values: Sequence[float], *, replicates: int, seed: int
) -> list[float]:
    if not values or replicates <= 0:
        raise ValueError("block bootstrap requires values and positive replicates")
    block_length = max(1, min(len(values), round(math.sqrt(len(values)))))
    starts = range(max(1, len(values) - block_length + 1))
    generator = random.Random(seed)
    means: list[float] = []
    for _ in range(replicates):
        sample: list[float] = []
        while len(sample) < len(values):
            start = generator.choice(starts)
            sample.extend(values[start : start + block_length])
        means.append(sum(sample[: len(values)]) / len(values))
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _episode_bootstrap_interval(
    values: Sequence[tuple[int, float]], *, replicates: int, seed: int
) -> list[float]:
    grouped: dict[int, list[float]] = {}
    for episode, value in values:
        grouped.setdefault(episode, []).append(value)
    episode_means = [sum(group) / len(group) for group in grouped.values()]
    if not episode_means or replicates <= 0:
        raise ValueError("episode bootstrap requires values and positive replicates")
    generator = random.Random(seed)
    means = [
        sum(generator.choice(episode_means) for _ in episode_means) / len(episode_means)
        for _ in range(replicates)
    ]
    return [_percentile(means, 0.025), _percentile(means, 0.975)]


def _validate_heldout_sample_pair(
    full: Mapping[str, Any], masked: Mapping[str, Any]
) -> tuple[int, float, float]:
    for field in _HELDOUT_SAMPLE_IDENTITIES:
        if full.get(field) != masked.get(field):
            raise ValueError(f"matched held-out samples differ at {field}")
    if full.get("action_backend") != WLA_ACTION_BACKEND or masked.get(
        "action_backend"
    ) != WLA_ACTION_BACKEND:
        raise ValueError("matched held-out samples did not execute complete WLA action")
    episode = full.get("source_episode_index")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("held-out source episode identity is invalid")
    full_loss = float(full.get("action_loss", math.nan))
    masked_loss = float(masked.get("action_loss", math.nan))
    if (
        not math.isfinite(full_loss)
        or not math.isfinite(masked_loss)
        or full_loss <= 0
        or masked_loss <= 0
    ):
        raise ValueError("held-out action losses must be finite and positive")
    return episode, full_loss, masked_loss


def _validate_heldout_temporal_pair(
    initial: Mapping[str, Any], later: Mapping[str, Any]
) -> tuple[int, float, float]:
    for field in _HELDOUT_SAMPLE_IDENTITIES:
        if initial.get(field) != later.get(field):
            raise ValueError(f"held-out temporal samples differ at {field}")
    if initial.get("action_backend") != WLA_ACTION_BACKEND or later.get(
        "action_backend"
    ) != WLA_ACTION_BACKEND:
        raise ValueError("held-out temporal samples did not execute complete WLA action")
    episode = initial.get("source_episode_index")
    if isinstance(episode, bool) or not isinstance(episode, int) or episode < 0:
        raise ValueError("held-out temporal source episode identity is invalid")
    initial_loss = float(initial.get("action_loss", math.nan))
    later_loss = float(later.get("action_loss", math.nan))
    if (
        not math.isfinite(initial_loss)
        or not math.isfinite(later_loss)
        or initial_loss <= 0
        or later_loss <= 0
    ):
        raise ValueError("held-out temporal action losses must be finite and positive")
    return episode, initial_loss, later_loss


def _heldout_step_summary(
    full: Mapping[str, Any],
    masked: Mapping[str, Any],
    *,
    step: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    for field in (
        "schema",
        "status",
        "checkpoint_global_step",
        "architecture_identity",
        "state_mode",
        "implementation_sha256",
        "model_family_sha256",
        "lingbot_base_family_sha256",
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
        "evaluation_input_sha256",
    ):
        if full.get(field) != masked.get(field):
            raise ValueError(f"matched held-out snapshots differ at {field}")
    if full.get("status") != "PASS" or full.get("checkpoint_global_step") != step:
        raise ValueError("held-out snapshot is not a passing registered step")
    full_samples = full.get("samples")
    masked_samples = masked.get("samples")
    if (
        not isinstance(full_samples, list)
        or not isinstance(masked_samples, list)
        or not full_samples
        or len(full_samples) != len(masked_samples)
    ):
        raise ValueError("matched held-out snapshots have different sample counts")
    rows = [
        _validate_heldout_sample_pair(full_sample, masked_sample)
        for full_sample, masked_sample in zip(full_samples, masked_samples, strict=True)
    ]

    def summarize(indices: Sequence[int], *, seed: int) -> dict[str, Any]:
        selected = [rows[index] for index in indices]
        log_ratios = [
            (episode, math.log(full_loss / masked_loss))
            for episode, full_loss, masked_loss in selected
        ]
        deltas = [
            (episode, full_loss - masked_loss)
            for episode, full_loss, masked_loss in selected
        ]
        mean_log_ratio = sum(value for _, value in log_ratios) / len(log_ratios)
        log_interval = _episode_bootstrap_interval(
            log_ratios, replicates=bootstrap_replicates, seed=seed
        )
        delta_interval = _episode_bootstrap_interval(
            deltas, replicates=bootstrap_replicates, seed=seed + 1
        )
        return {
            "sample_count": len(selected),
            "source_episode_count": len({episode for episode, _, _ in selected}),
            "full_mean_action_loss": (
                sum(full_loss for _, full_loss, _ in selected) / len(selected)
            ),
            "masked_mean_action_loss": (
                sum(masked_loss for _, _, masked_loss in selected) / len(selected)
            ),
            "paired_geometric_relative_delta": math.exp(mean_log_ratio) - 1.0,
            "paired_geometric_relative_delta_episode_bootstrap_95": [
                math.exp(bound) - 1.0 for bound in log_interval
            ],
            "paired_absolute_delta": sum(value for _, value in deltas) / len(deltas),
            "paired_absolute_delta_episode_bootstrap_95": delta_interval,
            "full_sample_wins": sum(full_loss < masked_loss for _, full_loss, masked_loss in selected),
            "ties": sum(full_loss == masked_loss for _, full_loss, masked_loss in selected),
            "masked_sample_wins": sum(full_loss > masked_loss for _, full_loss, masked_loss in selected),
        }

    partitions = sorted({str(sample.get("partition")) for sample in full_samples})
    return {
        "checkpoint_global_step": step,
        "overall": summarize(range(len(rows)), seed=20261000 + step),
        "partitions": {
            partition: summarize(
                [
                    index
                    for index, sample in enumerate(full_samples)
                    if sample.get("partition") == partition
                ],
                seed=(
                    20262000
                    + step
                    + sum(
                        (index + 1) * ord(character)
                        for index, character in enumerate(partition)
                    )
                ),
            )
            for partition in partitions
        },
    }


def _heldout_learning_difference_in_differences(
    full_initial: Mapping[str, Any],
    full_later: Mapping[str, Any],
    masked_initial: Mapping[str, Any],
    masked_later: Mapping[str, Any],
    *,
    initial_step: int,
    later_step: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if later_step <= initial_step:
        raise ValueError("held-out learning comparison requires a later checkpoint")
    snapshots = (full_initial, full_later, masked_initial, masked_later)
    for snapshot, step in zip(
        snapshots,
        (initial_step, later_step, initial_step, later_step),
        strict=True,
    ):
        if snapshot.get("status") != "PASS" or snapshot.get("checkpoint_global_step") != step:
            raise ValueError("held-out learning snapshot is not a passing registered step")
    full_initial_samples = full_initial.get("samples")
    full_later_samples = full_later.get("samples")
    masked_initial_samples = masked_initial.get("samples")
    masked_later_samples = masked_later.get("samples")
    sample_lists = (
        full_initial_samples,
        full_later_samples,
        masked_initial_samples,
        masked_later_samples,
    )
    if (
        any(not isinstance(samples, list) for samples in sample_lists)
        or not full_initial_samples
        or len({len(samples) for samples in sample_lists}) != 1
    ):
        raise ValueError("held-out learning snapshots have different sample counts")

    rows: list[tuple[int, float, float, float, float]] = []
    for full_0, full_t, masked_0, masked_t in zip(*sample_lists, strict=True):
        _validate_heldout_sample_pair(full_0, masked_0)
        _validate_heldout_sample_pair(full_t, masked_t)
        full_episode, full_initial_loss, full_later_loss = (
            _validate_heldout_temporal_pair(full_0, full_t)
        )
        masked_episode, masked_initial_loss, masked_later_loss = (
            _validate_heldout_temporal_pair(masked_0, masked_t)
        )
        if full_episode != masked_episode:
            raise ValueError("held-out learning source episode identities differ")
        rows.append(
            (
                full_episode,
                full_initial_loss,
                full_later_loss,
                masked_initial_loss,
                masked_later_loss,
            )
        )

    def summarize(indices: Sequence[int], *, seed: int) -> dict[str, Any]:
        selected = [rows[index] for index in indices]
        full_changes = [
            (episode, math.log(full_later_loss / full_initial_loss))
            for episode, full_initial_loss, full_later_loss, _, _ in selected
        ]
        masked_changes = [
            (episode, math.log(masked_later_loss / masked_initial_loss))
            for episode, _, _, masked_initial_loss, masked_later_loss in selected
        ]
        log_differences = [
            (episode, full_change[1] - masked_change[1])
            for full_change, masked_change in zip(full_changes, masked_changes, strict=True)
            for episode in (full_change[0],)
        ]
        absolute_differences = [
            (
                episode,
                (full_later_loss - full_initial_loss)
                - (masked_later_loss - masked_initial_loss),
            )
            for (
                episode,
                full_initial_loss,
                full_later_loss,
                masked_initial_loss,
                masked_later_loss,
            ) in selected
        ]
        mean_full_change = sum(value for _, value in full_changes) / len(full_changes)
        mean_masked_change = sum(value for _, value in masked_changes) / len(masked_changes)
        mean_log_difference = sum(value for _, value in log_differences) / len(log_differences)
        log_interval = _episode_bootstrap_interval(
            log_differences, replicates=bootstrap_replicates, seed=seed
        )
        absolute_interval = _episode_bootstrap_interval(
            absolute_differences,
            replicates=bootstrap_replicates,
            seed=seed + 1,
        )
        return {
            "sample_count": len(selected),
            "source_episode_count": len({row[0] for row in selected}),
            "full_paired_geometric_relative_change": math.exp(mean_full_change) - 1.0,
            "masked_paired_geometric_relative_change": math.exp(mean_masked_change) - 1.0,
            "learning_ratio_of_ratios_delta": math.exp(mean_log_difference) - 1.0,
            "learning_ratio_of_ratios_delta_episode_bootstrap_95": [
                math.exp(bound) - 1.0 for bound in log_interval
            ],
            "learning_absolute_difference_in_differences": (
                sum(value for _, value in absolute_differences) / len(absolute_differences)
            ),
            "learning_absolute_difference_in_differences_episode_bootstrap_95": (
                absolute_interval
            ),
            "full_learning_wins": sum(value < 0 for _, value in log_differences),
            "ties": sum(value == 0 for _, value in log_differences),
            "masked_learning_wins": sum(value > 0 for _, value in log_differences),
        }

    partitions = sorted(
        {str(sample.get("partition")) for sample in full_initial_samples}
    )
    return {
        "initial_checkpoint_global_step": initial_step,
        "later_checkpoint_global_step": later_step,
        "overall": summarize(range(len(rows)), seed=20263000 + later_step),
        "partitions": {
            partition: summarize(
                [
                    index
                    for index, sample in enumerate(full_initial_samples)
                    if sample.get("partition") == partition
                ],
                seed=(
                    20264000
                    + later_step
                    + sum(
                        (index + 1) * ord(character)
                        for index, character in enumerate(partition)
                    )
                ),
            )
            for partition in partitions
        },
    }


def compare_heldout_snapshots(
    *,
    full_snapshots: Mapping[int, Mapping[str, Any]],
    masked_snapshots: Mapping[int, Mapping[str, Any]],
    bootstrap_replicates: int,
) -> dict[str, Any]:
    if full_snapshots.keys() != masked_snapshots.keys() or not full_snapshots:
        raise ValueError("matched runs have different held-out checkpoint steps")
    steps = sorted(full_snapshots)
    initial_step = steps[0]
    return {
        "evidence_scope": "fixed_heldout_wla_action_loss",
        "steps": [
            _heldout_step_summary(
                full_snapshots[step],
                masked_snapshots[step],
                step=step,
                bootstrap_replicates=bootstrap_replicates,
            )
            for step in steps
        ],
        "learning_difference_in_differences": [
            _heldout_learning_difference_in_differences(
                full_snapshots[initial_step],
                full_snapshots[step],
                masked_snapshots[initial_step],
                masked_snapshots[step],
                initial_step=initial_step,
                later_step=step,
                bootstrap_replicates=bootstrap_replicates,
            )
            for step in steps[1:]
        ],
    }


def _load_heldout_snapshots(run_dir: Path) -> dict[int, dict[str, Any]]:
    root = run_dir / "action_evaluations"
    snapshots: dict[int, dict[str, Any]] = {}
    for path in sorted(root.glob("step_*/distributed.json")):
        try:
            step = int(path.parent.name.removeprefix("step_"))
        except ValueError as error:
            raise ValueError(f"invalid held-out checkpoint directory: {path.parent}") from error
        snapshots[step] = _load_json(path)
    if not snapshots:
        raise FileNotFoundError(f"no fixed held-out action snapshots under {root}")
    return snapshots


def _window_summary(
    rows: Sequence[tuple[int, int, float, float]],
    *,
    start: int,
    end: int,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    selected = [row for row in rows if start <= row[0] <= end]
    steps = sorted({row[0] for row in selected})
    if steps != list(range(start, end + 1)):
        raise ValueError(f"curve does not completely cover window {start}:{end}")
    per_step_log_ratios: list[float] = []
    per_step_deltas: list[float] = []
    for step in steps:
        step_rows = [row for row in selected if row[0] == step]
        per_step_log_ratios.append(
            sum(math.log(full / masked) for _, _, full, masked in step_rows)
            / len(step_rows)
        )
        per_step_deltas.append(
            sum(full - masked for _, _, full, masked in step_rows) / len(step_rows)
        )
    log_interval = _moving_block_interval(
        per_step_log_ratios, replicates=bootstrap_replicates, seed=seed
    )
    delta_interval = _moving_block_interval(
        per_step_deltas, replicates=bootstrap_replicates, seed=seed + 1
    )
    full_values = [row[2] for row in selected]
    masked_values = [row[3] for row in selected]
    mean_log_ratio = sum(per_step_log_ratios) / len(per_step_log_ratios)
    return {
        "start_step": start,
        "end_step": end,
        "step_count": len(steps),
        "paired_rank_step_count": len(selected),
        "full_mean_action_loss": sum(full_values) / len(full_values),
        "masked_mean_action_loss": sum(masked_values) / len(masked_values),
        "ratio_of_means": (sum(full_values) / sum(masked_values)),
        "paired_geometric_relative_delta": math.exp(mean_log_ratio) - 1.0,
        "paired_geometric_relative_delta_block_bootstrap_95": [
            math.exp(bound) - 1.0 for bound in log_interval
        ],
        "paired_absolute_delta": sum(per_step_deltas) / len(per_step_deltas),
        "paired_absolute_delta_block_bootstrap_95": delta_interval,
        "full_rank_step_wins": sum(full < masked for _, _, full, masked in selected),
        "ties": sum(full == masked for _, _, full, masked in selected),
        "masked_rank_step_wins": sum(full > masked for _, _, full, masked in selected),
    }


def compare_records(
    *,
    full_records: Mapping[tuple[int, int], Mapping[str, Any]],
    masked_records: Mapping[tuple[int, int], Mapping[str, Any]],
    windows: Sequence[tuple[int, int]],
    bootstrap_replicates: int,
    minimum_relative_lead: float,
) -> dict[str, Any]:
    if full_records.keys() != masked_records.keys() or not full_records:
        raise ValueError("matched runs have different rank-step keys")
    if not 0 < minimum_relative_lead < 1:
        raise ValueError("minimum relative lead must lie in (0, 1)")
    rows: list[tuple[int, int, float, float]] = []
    for step_rank in sorted(full_records):
        full = full_records[step_rank]
        masked = masked_records[step_rank]
        _validate_record_pair(full, masked)
        rows.append(
            (
                step_rank[0],
                step_rank[1],
                _positive_loss(full, "loss_action"),
                _positive_loss(masked, "loss_action"),
            )
        )
    maximum_step = max(step for step, _, _, _ in rows)
    registered_windows = tuple(windows) or ((1, maximum_step),)
    summaries = [
        _window_summary(
            rows,
            start=start,
            end=end,
            bootstrap_replicates=bootstrap_replicates,
            seed=20260826 + index * 2,
        )
        for index, (start, end) in enumerate(registered_windows)
    ]
    overall = _window_summary(
        rows,
        start=1,
        end=maximum_step,
        bootstrap_replicates=bootstrap_replicates,
        seed=20260926,
    )
    lead = -minimum_relative_lead
    upper = overall["paired_geometric_relative_delta_block_bootstrap_95"][1]
    if maximum_step < 100:
        decision = "INSUFFICIENT_EARLY_CURVE"
    elif overall["paired_geometric_relative_delta"] >= 0.02:
        decision = "PICF_EARLY_OPTIMIZATION_REGRESSION"
    elif overall["paired_geometric_relative_delta"] <= lead and upper < 0:
        decision = "PICF_EARLY_MATERIAL_OPTIMIZATION_LEAD"
    else:
        decision = "PICF_EARLY_OPTIMIZATION_INCONCLUSIVE"
    return {
        "schema": SCHEMA,
        "status": "PASS",
        "decision": decision,
        "evidence_scope": "matched_training_action_loss_only",
        "authorizes_long_train": False,
        "required_remaining_gates": [
            "fixed_heldout_action_curve",
            "CALVIN_closed_loop_rollout",
        ],
        "maximum_step": maximum_step,
        "minimum_relative_lead": minimum_relative_lead,
        "overall": overall,
        "windows": summaries,
    }


def main() -> None:
    args = _parse_args()
    if args.bootstrap_replicates <= 0:
        raise ValueError("bootstrap replicates must be positive")
    full_manifest = _load_json(args.full_run_dir / "run_manifest.json")
    masked_manifest = _load_json(args.masked_run_dir / "run_manifest.json")
    contract = validate_matched_manifests(full_manifest, masked_manifest)
    world_size = int(full_manifest["world_size"])
    report = compare_records(
        full_records=_load_rank_journals(
            args.full_run_dir, expected_world_size=world_size
        ),
        masked_records=_load_rank_journals(
            args.masked_run_dir, expected_world_size=world_size
        ),
        windows=args.window,
        bootstrap_replicates=args.bootstrap_replicates,
        minimum_relative_lead=args.minimum_relative_lead,
    )
    heldout = compare_heldout_snapshots(
        full_snapshots=_load_heldout_snapshots(args.full_run_dir),
        masked_snapshots=_load_heldout_snapshots(args.masked_run_dir),
        bootstrap_replicates=args.bootstrap_replicates,
    )
    payload = {**report, "contract": contract, "fixed_heldout": heldout}
    payload["artifact_sha256"] = _canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="ascii") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()

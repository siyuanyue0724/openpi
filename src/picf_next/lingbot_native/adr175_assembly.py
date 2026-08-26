"""Deterministic assembly of raw ADR-175 runs into strict arm reports."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from picf_next.lingbot_native.adr175_contract import Adr175BroadSupportContract
from picf_next.lingbot_native.adr175_validation import (
    ADR175_AMBIGUOUS_TASKS,
    ADR175_ARM_REPORT_SCHEMA,
    ADR175_ARMS,
    ADR175_EXACT_TASK_TARGETS,
    ADR175_MILESTONES,
    ADR175_TOTAL_STEPS,
    canonical_sha256,
    seal_adr175_arm_report,
)

ADR175_EVALUATION_SNAPSHOT_SCHEMA = "picf-next.adr175-evaluation-snapshot.v1"
ADR175_ASSEMBLY_SCHEMA = "picf-next.adr175-arm-assembly.v1"
ADR175_BOOTSTRAP_SEED = 20260816
ADR175_BOOTSTRAP_REPLICATES = 10_000


@dataclass(frozen=True, slots=True)
class Adr175AssemblyContractIdentity:
    """The frozen broad-support fields required to audit a completed 2k run."""

    artifact_sha256: str
    matched_arm_input_sha256: str
    dataset_manifest_sha256: str
    stream_plan_sha256: str
    representation_split_artifact_sha256: str
    entity_evaluation_plan_artifact_sha256: str
    training_prefix_steps: int
    plan_total_steps: int
    global_batch_size: int
    training_prefix_sample_count: int
    training_prefix_sample_keys_sha256: str
    training_prefix_prompt_receipts_sha256: str

    @classmethod
    def from_contract(
        cls,
        contract: Adr175BroadSupportContract,
    ) -> Adr175AssemblyContractIdentity:
        return cls(
            artifact_sha256=contract.artifact_sha256,
            matched_arm_input_sha256=contract.matched_arm_input_sha256,
            dataset_manifest_sha256=contract.dataset_manifest_sha256,
            stream_plan_sha256=contract.stream_plan_sha256,
            representation_split_artifact_sha256=(contract.representation_split_artifact_sha256),
            entity_evaluation_plan_artifact_sha256=(
                contract.entity_evaluation_plan_artifact_sha256
            ),
            training_prefix_steps=contract.training_prefix_steps,
            plan_total_steps=contract.plan_total_steps,
            global_batch_size=contract.global_batch_size,
            training_prefix_sample_count=contract.training_prefix_sample_count,
            training_prefix_sample_keys_sha256=(contract.training_prefix_sample_keys_sha256),
            training_prefix_prompt_receipts_sha256=(
                contract.training_prefix_prompt_receipts_sha256
            ),
        )

    def validate_for_adr175(self) -> None:
        for name, value in (
            ("contract artifact", self.artifact_sha256),
            ("matched-arm input", self.matched_arm_input_sha256),
            ("dataset manifest", self.dataset_manifest_sha256),
            ("stream plan", self.stream_plan_sha256),
            ("representation split", self.representation_split_artifact_sha256),
            ("entity evaluation plan", self.entity_evaluation_plan_artifact_sha256),
            ("training sample keys", self.training_prefix_sample_keys_sha256),
            ("training prompt receipts", self.training_prefix_prompt_receipts_sha256),
        ):
            _sha256(value, name=name)
        if (
            self.training_prefix_steps != ADR175_TOTAL_STEPS
            or self.plan_total_steps != ADR175_TOTAL_STEPS
            or self.global_batch_size != 2
            or self.training_prefix_sample_count != ADR175_TOTAL_STEPS * 2
        ):
            raise ValueError(
                "ADR-175 assembly requires the exact 2000-step, global-batch-two prefix"
            )


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _list(value: object, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be nonempty text")
    return value


def _sha256(value: object, *, name: str) -> str:
    digest = _text(value, name=name)
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _unit_interval(value: object, *, name: str) -> float:
    result = _finite(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} lies outside [0,1]")
    return result


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError(f"{name} requires nonempty finite values")
    return math.fsum(values) / len(values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_adr175_snapshot(receipt: object) -> dict[str, Any]:
    """Load and verify one runner-published evaluation snapshot receipt."""

    payload = _mapping(receipt, name="ADR-175 snapshot receipt")
    path = Path(_text(payload.get("path"), name="snapshot path"))
    expected_file_sha256 = _sha256(
        payload.get("file_sha256"),
        name="snapshot file SHA-256",
    )
    if _file_sha256(path) != expected_file_sha256:
        raise ValueError(f"ADR-175 snapshot file changed: {path}")
    snapshot = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(snapshot, dict):
        raise TypeError("ADR-175 snapshot root must be an object")
    artifact_sha256 = _sha256(
        snapshot.pop("artifact_sha256", None),
        name="snapshot artifact SHA-256",
    )
    if artifact_sha256 != _sha256(
        payload.get("artifact_sha256"),
        name="snapshot receipt artifact SHA-256",
    ):
        raise ValueError("ADR-175 snapshot receipt and artifact identity differ")
    if canonical_sha256(snapshot) != artifact_sha256:
        raise ValueError("ADR-175 snapshot semantic SHA-256 differs")
    return {**snapshot, "artifact_sha256": artifact_sha256}


def _combined_rank_digest(
    rank_receipts: Sequence[tuple[int, Mapping[str, Any]]],
    field: str,
) -> str:
    return canonical_sha256(
        [
            {
                "rank": rank,
                "sha256": _sha256(receipt.get(field), name=f"rank {rank} {field}"),
            }
            for rank, receipt in rank_receipts
        ]
    )


def _step_receipts(
    raw: Mapping[str, Any],
    *,
    arm: str,
    contract: Adr175AssemblyContractIdentity,
) -> list[dict[str, object]]:
    rank_reports = _list(raw.get("rank_reports"), name="ADR-175 rank reports")
    if sorted(int(item["rank"]) for item in rank_reports) != [0, 1]:
        raise ValueError("ADR-175 raw report must contain ranks zero and one")
    by_rank = {int(item["rank"]): _list(item["steps"], name="rank steps") for item in rank_reports}
    if any(len(steps) != ADR175_TOTAL_STEPS for steps in by_rank.values()):
        raise ValueError("ADR-175 every rank must publish exactly 2000 steps")
    result: list[dict[str, object]] = []
    fields = (
        "sample_sha256",
        "action_target_sha256",
        "noise_sha256",
        "time_sha256",
        "prompt_sha256",
    )
    prefix_sample_keys: list[str] = []
    prefix_prompt_receipts: list[str] = []
    maximum_peak_reserved_bytes = int(raw.get("maximum_peak_reserved_bytes", -1))
    if maximum_peak_reserved_bytes <= 0:
        raise ValueError("ADR-175 raw report omits its positive memory ceiling")
    for step_index in range(ADR175_TOTAL_STEPS):
        rank_receipts: list[tuple[int, Mapping[str, Any]]] = []
        rank_execution_inputs: list[dict[str, object]] = []
        for rank in (0, 1):
            step = _mapping(by_rank[rank][step_index], name=f"rank {rank} step")
            if step.get("global_step") != step_index + 1:
                raise ValueError("ADR-175 rank steps are not contiguous")
            if step.get("adr175_arm") != arm:
                raise ValueError(f"ADR-175 rank {rank} step arm differs from its raw report")
            if step.get("policy_forward_absent") is not False:
                raise ValueError(f"ADR-175 rank {rank} omitted the policy forward")
            expected_picf_active = arm != "lbot"
            if step.get("picf_graph_active") is not expected_picf_active:
                raise ValueError(f"ADR-175 rank {rank} PICF activation differs from its arm")
            if _list(step.get("frame_indices"), name="ADR-175 frame indices") != [0]:
                raise ValueError("ADR-175 frozen sample plan must remain one-frame reset-only")
            if _list(step.get("reset"), name="ADR-175 reset flags") != [True]:
                raise ValueError("ADR-175 frozen sample plan must reset every occurrence")
            optimizer_lags = _list(step.get("optimizer_lags"), name="optimizer lags")
            if optimizer_lags != [0]:
                raise ValueError("ADR-175 optimizer update lag differs from zero")
            gradient_metrics = _mapping(
                step.get("gradient_metrics"),
                name=f"rank {rank} gradient metrics",
            )
            if gradient_metrics.get("all_finite") is not True:
                raise ValueError(f"ADR-175 rank {rank} published a nonfinite gradient")
            for metric_name in (
                "preclip_global_norm",
                "vlm_host_norm",
                "action_expert_norm",
            ):
                _finite(
                    gradient_metrics.get(metric_name),
                    name=f"rank {rank} {metric_name}",
                )
            if expected_picf_active:
                _finite(
                    gradient_metrics.get("native_graph_norm"),
                    name=f"rank {rank} native graph norm",
                )
            peak_reserved = int(step.get("peak_cuda_reserved_bytes", -1))
            if peak_reserved <= 0 or peak_reserved > maximum_peak_reserved_bytes:
                raise ValueError(f"ADR-175 rank {rank} exceeded its registered memory ceiling")
            for metric_name in (
                "objective_total",
                "official_action_loss",
                "official_policy_loss",
                "action_family",
            ):
                _finite(step.get(metric_name), name=f"rank {rank} {metric_name}")

            sample_keys = _list(step.get("sample_keys"), name="ADR-175 rank sample keys")
            if len(sample_keys) != 1:
                raise ValueError("ADR-175 global-batch-two run requires one sample per rank")
            sample_key = _text(sample_keys[0], name="ADR-175 sample key")
            source_digest = _sha256(step.get("source_digest"), name="source digest")
            receipt = _mapping(
                step.get("adr175_input_receipt"),
                name=f"rank {rank} ADR-175 input receipt",
            )
            expected_sample_receipt = canonical_sha256(
                {"sample_keys": [sample_key], "source_digest": source_digest}
            )
            if receipt.get("sample_sha256") != expected_sample_receipt:
                raise ValueError(f"ADR-175 rank {rank} sample receipt differs from its step")
            prefix_sample_keys.append(sample_key)
            prefix_prompt_receipts.append(
                _sha256(receipt.get("prompt_sha256"), name=f"rank {rank} prompt receipt")
            )
            rank_receipts.append((rank, receipt))
            rank_execution_inputs.append(
                {
                    "frame_indices": step.get("frame_indices"),
                    "lane_ids": step.get("lane_ids"),
                    "optimizer_lags": step.get("optimizer_lags"),
                    "previous_state_ages": step.get("previous_state_ages"),
                    "previous_state_input_absent": step.get("previous_state_input_absent"),
                    "rank": rank,
                    "reset": step.get("reset"),
                    "sample_keys": step.get("sample_keys"),
                    "source_digest": step.get("source_digest"),
                    "visual_lattice": step.get("visual_lattice"),
                    "visual_lattice_contract": step.get("visual_lattice_contract"),
                }
            )
        result.append(
            {
                "global_step": step_index + 1,
                "execution_input_sha256": canonical_sha256(rank_execution_inputs),
                **{field: _combined_rank_digest(rank_receipts, field) for field in fields},
            }
        )
    if len(prefix_sample_keys) != contract.training_prefix_sample_count:
        raise ValueError("ADR-175 raw steps do not cover the frozen prefix sample count")
    if canonical_sha256(prefix_sample_keys) != contract.training_prefix_sample_keys_sha256:
        raise ValueError("ADR-175 raw sample order differs from the frozen 2k prefix")
    if canonical_sha256(prefix_prompt_receipts) != (
        contract.training_prefix_prompt_receipts_sha256
    ):
        raise ValueError("ADR-175 raw prompt receipts differ from the frozen 2k prefix")
    return result


def _task_macro(samples: Sequence[Mapping[str, Any]], field: str) -> float:
    values_by_task: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        value = sample.get(field)
        if value is None:
            continue
        values_by_task[_text(sample.get("task_key"), name="evaluation task")].append(
            _finite(value, name=field)
        )
    if not values_by_task:
        raise ValueError(f"ADR-175 task macro has no {field} values")
    return _mean(
        [
            _mean(values, name=f"{task_key} {field}")
            for task_key, values in sorted(values_by_task.items())
        ],
        name=f"task-macro {field}",
    )


def _sample_entity_score(sample: Mapping[str, Any]) -> float | None:
    evidence = sample.get("entity_evidence")
    if evidence is None:
        return None
    rows = _list(_mapping(evidence, name="entity evidence").get("rows"), name="entity rows")
    if not rows:
        return None
    return _mean(
        [
            _finite(
                _mapping(row, name="entity row").get("support_soft_iou_efficiency"),
                name="support soft-IoU efficiency",
            )
            for row in rows
        ],
        name="sample entity-set score",
    )


def _entity_task_macro(samples: Sequence[Mapping[str, Any]]) -> float:
    enriched = [dict(sample, entity_set_score=_sample_entity_score(sample)) for sample in samples]
    score = _task_macro(enriched, "entity_set_score")
    if not 0.0 <= score <= 1.0:
        raise ValueError("ADR-175 entity-set score lies outside [0,1]")
    return score


def _partition_values(
    samples: Sequence[Mapping[str, Any]],
    field: str,
    *,
    entity: bool = False,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for partition in ("validation", "heldout"):
        selected = [sample for sample in samples if sample.get("partition") == partition]
        if not selected:
            raise ValueError(f"ADR-175 snapshot omits {partition}")
        result[partition] = _entity_task_macro(selected) if entity else _task_macro(selected, field)
    return result


def _attention_partition_values(
    samples: Sequence[Mapping[str, Any]],
    field: str,
    *,
    observable_by_sample: Mapping[str, bool],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for partition in ("validation", "heldout"):
        enriched: list[dict[str, object]] = []
        for sample in samples:
            if sample.get("partition") != partition:
                continue
            sample_key = _text(sample.get("sample_key"), name="evaluation sample key")
            if sample_key not in observable_by_sample:
                continue
            value = _resolved_attention_value(
                sample,
                field=field,
                target_observable=observable_by_sample[sample_key],
            )
            if value is not None:
                enriched.append(
                    {
                        "task_key": _text(sample.get("task_key"), name="evaluation task"),
                        field: value,
                    }
                )
        result[partition] = _task_macro(enriched, field)
    return result


def _snapshot_sample_identity_sha256(snapshot: Mapping[str, Any]) -> str:
    return canonical_sha256(
        [
            {
                "ordinal": sample.get("ordinal"),
                "partition": sample.get("partition"),
                "rank": sample.get("rank"),
                "sample_key": sample.get("sample_key"),
                "segment_index": sample.get("segment_index"),
                "source_digest": sample.get("source_digest"),
                "source_episode_index": sample.get("source_episode_index"),
                "source_global_index": sample.get("source_global_index"),
                "task_key": sample.get("task_key"),
                "transition_index": sample.get("transition_index"),
            }
            for sample in _list(snapshot.get("samples"), name="snapshot samples")
        ]
    )


def _validate_evaluation_topology(snapshot: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    samples = [
        _mapping(sample, name="evaluation sample")
        for sample in _list(snapshot.get("samples"), name="snapshot samples")
    ]
    expected_tasks = {
        *(task_key for task_key, _target_keys in ADR175_EXACT_TASK_TARGETS),
        *ADR175_AMBIGUOUS_TASKS,
    }
    expected_counts = {"validation": 1, "heldout": 2}
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    sample_keys: set[str] = set()
    for sample in samples:
        partition = _text(sample.get("partition"), name="evaluation partition")
        task_key = _text(sample.get("task_key"), name="evaluation task")
        sample_key = _text(sample.get("sample_key"), name="evaluation sample key")
        if partition not in expected_counts or task_key not in expected_tasks:
            raise ValueError("ADR-175 evaluation task or partition inventory changed")
        if sample_key in sample_keys:
            raise ValueError(f"ADR-175 duplicate evaluation sample key: {sample_key}")
        sample_keys.add(sample_key)
        grouped[(task_key, partition)].append(sample)
    expected_groups = {
        (task_key, partition) for task_key in expected_tasks for partition in expected_counts
    }
    if set(grouped) != expected_groups:
        raise ValueError("ADR-175 evaluation task/partition coverage changed")
    for (task_key, partition), selected in grouped.items():
        expected_count = expected_counts[partition]
        if len(selected) != expected_count:
            raise ValueError(f"ADR-175 {task_key}/{partition} requires {expected_count} samples")
        source_episodes = [sample.get("source_episode_index") for sample in selected]
        if any(isinstance(value, bool) or not isinstance(value, int) for value in source_episodes):
            raise TypeError("ADR-175 source episode index must be an integer")
        if len(set(source_episodes)) != expected_count:
            raise ValueError(f"ADR-175 {task_key}/{partition} requires source-distinct samples")
    return samples


def _exact_observability(
    snapshot: Mapping[str, Any],
) -> tuple[dict[str, bool], dict[str, dict[str, object]], str]:
    target_keys_by_task = dict(ADR175_EXACT_TASK_TARGETS)
    observable_by_sample: dict[str, bool] = {}
    receipts_by_sample: dict[str, dict[str, object]] = {}
    for sample in _validate_evaluation_topology(snapshot):
        task_key = _text(sample.get("task_key"), name="evaluation task")
        if task_key not in target_keys_by_task:
            continue
        sample_key = _text(sample.get("sample_key"), name="evaluation sample key")
        evidence = _mapping(sample.get("entity_evidence"), name="entity evidence")
        rows = [
            _mapping(row, name="entity evidence row")
            for row in _list(evidence.get("rows"), name="entity evidence rows")
        ]
        observed_identity_keys = tuple(
            sorted(_text(row.get("identity_key"), name="entity identity") for row in rows)
        )
        if len(observed_identity_keys) != len(set(observed_identity_keys)):
            raise ValueError(f"ADR-175 duplicate entity identity: {sample_key}")
        target_visible_count = evidence.get("target_visible_count")
        if (
            isinstance(target_visible_count, bool)
            or not isinstance(target_visible_count, int)
            or target_visible_count != len(observed_identity_keys)
        ):
            raise ValueError(f"ADR-175 target-visible count changed: {sample_key}")
        target_identity_keys = target_keys_by_task[task_key]
        target_observable = set(target_identity_keys).issubset(observed_identity_keys)
        target_resolved = sample.get("target_valid")
        if not isinstance(target_resolved, bool):
            raise TypeError(f"ADR-175 target resolution is not boolean: {sample_key}")
        if target_resolved and not target_observable:
            raise ValueError(f"ADR-175 resolved an unobservable exact target: {sample_key}")
        score = sample.get("conditional_selectivity")
        adoption = sample.get("posterior_adoption")
        if target_resolved:
            _unit_interval(score, name=f"{sample_key} conditional selectivity")
            _unit_interval(adoption, name=f"{sample_key} posterior adoption")
        elif score is not None or adoption is not None:
            raise ValueError(f"ADR-175 unresolved target published attention scores: {sample_key}")
        receipt = {
            "partition": _text(sample.get("partition"), name="evaluation partition"),
            "sample_key": sample_key,
            "source_episode_index": sample.get("source_episode_index"),
            "source_global_index": sample.get("source_global_index"),
            "task_key": task_key,
            "target_identity_keys": list(target_identity_keys),
            "observed_identity_keys": list(observed_identity_keys),
            "target_observable": target_observable,
        }
        observable_by_sample[sample_key] = target_observable
        receipts_by_sample[sample_key] = receipt
    expected_exact_samples = len(ADR175_EXACT_TASK_TARGETS) * 3
    if len(observable_by_sample) != expected_exact_samples:
        raise ValueError("ADR-175 exact observability sample count changed")
    digest = canonical_sha256(
        [receipts_by_sample[sample_key] for sample_key in sorted(receipts_by_sample)]
    )
    return observable_by_sample, receipts_by_sample, digest


def _resolved_attention_value(
    sample: Mapping[str, Any],
    *,
    field: str,
    target_observable: bool,
) -> float | None:
    target_resolved = sample.get("target_valid")
    if not isinstance(target_resolved, bool):
        raise TypeError("ADR-175 target resolution must be boolean")
    if not target_observable:
        if target_resolved:
            raise ValueError("ADR-175 resolved an unobservable target")
        return None
    if not target_resolved:
        if sample.get(field) is not None:
            raise ValueError(f"ADR-175 unresolved target published {field}")
        return 0.0
    return _unit_interval(sample.get(field), name=field)


def _load_snapshots(
    raw: Mapping[str, Any],
    *,
    snapshot_loader: Callable[[object], Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    receipts = _list(raw.get("evaluation_snapshots"), name="evaluation snapshots")
    snapshots: dict[int, Mapping[str, Any]] = {}
    arm = _text(_mapping(raw.get("adr175"), name="ADR-175 raw metadata").get("arm"), name="arm")
    for receipt in receipts:
        receipt_payload = _mapping(receipt, name="ADR-175 snapshot receipt")
        snapshot = _mapping(snapshot_loader(receipt_payload), name="ADR-175 snapshot")
        if snapshot.get("schema") != ADR175_EVALUATION_SNAPSHOT_SCHEMA:
            raise ValueError("ADR-175 evaluation snapshot schema changed")
        if snapshot.get("status") != "PASS" or snapshot.get("arm") != arm:
            raise ValueError("ADR-175 evaluation snapshot status or arm differs")
        step = int(snapshot.get("checkpoint_global_step"))
        if int(receipt_payload.get("checkpoint_global_step", -1)) != step:
            raise ValueError("ADR-175 snapshot receipt milestone differs from its artifact")
        if receipt_payload.get("evaluation_input_sha256") != snapshot.get(
            "evaluation_input_sha256"
        ):
            raise ValueError("ADR-175 snapshot receipt input differs from its artifact")
        if canonical_sha256(receipt_payload.get("partition_summaries")) != canonical_sha256(
            snapshot.get("partition_summaries")
        ):
            raise ValueError("ADR-175 snapshot receipt summaries differ from its artifact")
        for snapshot_field, raw_field in (
            ("implementation_sha256", "implementation_sha256"),
            ("model_family_sha256", "model_family_sha256"),
            ("stream_plan_sha256", "plan_sha256"),
            ("representation_split_sha256", "representation_split_sha256"),
            ("entity_evaluation_plan_sha256", "entity_evaluation_plan_sha256"),
        ):
            if snapshot.get(snapshot_field) != raw.get(raw_field):
                raise ValueError(
                    f"ADR-175 snapshot {snapshot_field} differs from its parent raw run"
                )
        if step in snapshots:
            raise ValueError("ADR-175 evaluation milestone appears twice")
        _validate_evaluation_topology(snapshot)
        snapshots[step] = snapshot
    if tuple(sorted(snapshots)) != ADR175_MILESTONES:
        raise ValueError("ADR-175 snapshots must be exactly 0/250/500/1000/2000")
    if len({snapshot["evaluation_input_sha256"] for snapshot in snapshots.values()}) != 1:
        raise ValueError("ADR-175 evaluation inputs changed across milestones")
    sample_identities = {
        _snapshot_sample_identity_sha256(snapshot) for snapshot in snapshots.values()
    }
    if len(sample_identities) != 1:
        raise ValueError("ADR-175 evaluation sample identities changed across milestones")
    return snapshots


def _milestones(
    arm: str,
    snapshots: Mapping[int, Mapping[str, Any]],
    *,
    observable_by_sample: Mapping[str, bool] | None,
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for global_step in ADR175_MILESTONES:
        samples = [
            _mapping(sample, name="evaluation sample")
            for sample in _list(snapshots[global_step].get("samples"), name="snapshot samples")
        ]
        treatment = arm != "lbot"
        if treatment and observable_by_sample is None:
            raise ValueError("ADR-175 treatment milestones require observability")
        result.append(
            {
                "global_step": global_step,
                "posterior_adoption": (
                    _attention_partition_values(
                        samples,
                        "posterior_adoption",
                        observable_by_sample=observable_by_sample,
                    )
                    if treatment and observable_by_sample is not None
                    else None
                ),
                "conditional_selectivity": (
                    _attention_partition_values(
                        samples,
                        "conditional_selectivity",
                        observable_by_sample=observable_by_sample,
                    )
                    if treatment and observable_by_sample is not None
                    else None
                ),
                "action_loss": _partition_values(samples, "official_action_loss"),
                "entity_set_score": (
                    _partition_values(samples, "entity_set_score", entity=True)
                    if treatment
                    else None
                ),
            }
        )
    return result


def _exact_strata(
    snapshot: Mapping[str, Any],
    *,
    observable_by_sample: Mapping[str, bool],
    observability_receipts_by_sample: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    samples = [
        _mapping(sample, name="evaluation sample")
        for sample in _list(snapshot.get("samples"), name="snapshot samples")
    ]
    outcomes: list[dict[str, object]] = []
    for task_key, target_identity_keys in ADR175_EXACT_TASK_TARGETS:
        values: dict[str, float] = {}
        censored: dict[str, bool] = {}
        sample_counts: dict[str, int] = {}
        observable_counts: dict[str, int] = {}
        task_receipts: list[Mapping[str, object]] = []
        for partition in ("validation", "heldout"):
            selected = [
                sample
                for sample in samples
                if sample.get("task_key") == task_key and sample.get("partition") == partition
            ]
            expected_count = 1 if partition == "validation" else 2
            if len(selected) != expected_count:
                raise ValueError(
                    f"ADR-175 exact {task_key}/{partition} requires {expected_count} samples"
                )
            sample_keys = [
                _text(sample.get("sample_key"), name="evaluation sample key") for sample in selected
            ]
            if any(sample_key not in observable_by_sample for sample_key in sample_keys):
                raise ValueError("ADR-175 exact observability map omitted a sample")
            observable = [observable_by_sample[sample_key] for sample_key in sample_keys]
            sample_counts[partition] = len(selected)
            observable_counts[partition] = sum(observable)
            censored[partition] = not all(observable)
            task_receipts.extend(
                observability_receipts_by_sample[sample_key] for sample_key in sample_keys
            )
            if censored[partition]:
                values[partition] = 0.0
            else:
                resolved_values = [
                    _resolved_attention_value(
                        sample,
                        field="conditional_selectivity",
                        target_observable=True,
                    )
                    for sample in selected
                ]
                if any(value is None for value in resolved_values):
                    raise RuntimeError("observable exact target produced no resolved score")
                values[partition] = _mean(
                    [float(value) for value in resolved_values],
                    name=f"{task_key}/{partition} resolved selectivity",
                )
        outcomes.append(
            {
                "stratum_id": canonical_sha256(
                    {
                        "task_key": task_key,
                        "target_identity_keys": list(target_identity_keys),
                    }
                ),
                "task_key": task_key,
                "target_identity_keys": list(target_identity_keys),
                "validation_score": values["validation"],
                "heldout_score": values["heldout"],
                "validation_censored": censored["validation"],
                "heldout_censored": censored["heldout"],
                "validation_sample_count": sample_counts["validation"],
                "heldout_sample_count": sample_counts["heldout"],
                "validation_observable_sample_count": observable_counts["validation"],
                "heldout_observable_sample_count": observable_counts["heldout"],
                "observability_receipt_sha256": canonical_sha256(
                    sorted(task_receipts, key=lambda item: str(item["sample_key"]))
                ),
            }
        )
    return sorted(outcomes, key=lambda item: str(item["stratum_id"]))


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _task_macro_pairs(rows: Sequence[tuple[str, float, float]]) -> tuple[float, float]:
    raw_by_task: dict[str, list[float]] = defaultdict(list)
    normalized_by_task: dict[str, list[float]] = defaultdict(list)
    for task_key, physical, native in rows:
        difference = native - physical
        raw_by_task[task_key].append(difference)
        normalized_by_task[task_key].append(difference / max(1.0 - physical, 1.0e-6))
    raw = _mean(
        [_mean(values, name=f"{task} raw") for task, values in sorted(raw_by_task.items())],
        name="bootstrap raw task macro",
    )
    normalized = _mean(
        [
            _mean(values, name=f"{task} normalized")
            for task, values in sorted(normalized_by_task.items())
        ],
        name="bootstrap normalized task macro",
    )
    return raw, normalized


def _heldout_bootstrap(
    physical_snapshot: Mapping[str, Any],
    native_snapshot: Mapping[str, Any],
    *,
    observable_by_sample: Mapping[str, bool],
) -> dict[str, object]:
    def keyed(snapshot: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        result: dict[str, Mapping[str, Any]] = {}
        for sample in (
            _mapping(value, name="heldout sample")
            for value in _list(snapshot.get("samples"), name="snapshot samples")
        ):
            if (
                sample.get("partition") != "heldout"
                or sample.get("task_key") in ADR175_AMBIGUOUS_TASKS
            ):
                continue
            sample_key = _text(sample.get("sample_key"), name="heldout sample key")
            if sample_key in result:
                raise ValueError(f"ADR-175 duplicate heldout sample key: {sample_key}")
            result[sample_key] = sample
        return result

    physical = keyed(physical_snapshot)
    native = keyed(native_snapshot)
    if physical.keys() != native.keys():
        raise ValueError("ADR-175 treatment heldout sample identities differ")
    clusters_by_task: dict[str, dict[int, list[tuple[str, float, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for sample_key in sorted(physical):
        reference = physical[sample_key]
        candidate = native[sample_key]
        if observable_by_sample.get(sample_key) is not True:
            raise ValueError("ADR-175 exact heldout target is not externally observable")
        if reference.get("task_key") != candidate.get("task_key") or reference.get(
            "source_episode_index"
        ) != candidate.get("source_episode_index"):
            raise ValueError("ADR-175 heldout pairing metadata differs")
        task_key = str(reference["task_key"])
        physical_score = _resolved_attention_value(
            reference,
            field="conditional_selectivity",
            target_observable=True,
        )
        native_score = _resolved_attention_value(
            candidate,
            field="conditional_selectivity",
            target_observable=True,
        )
        if physical_score is None or native_score is None:
            raise RuntimeError("observable heldout target produced no selectivity score")
        clusters_by_task[task_key][int(reference["source_episode_index"])].append(
            (task_key, physical_score, native_score)
        )
    expected_tasks = {task_key for task_key, _target_keys in ADR175_EXACT_TASK_TARGETS}
    if set(clusters_by_task) != expected_tasks:
        raise ValueError("ADR-175 bootstrap exact-task inventory changed")
    if any(len(clusters) != 2 for clusters in clusters_by_task.values()):
        raise ValueError("ADR-175 bootstrap requires exactly two heldout episodes per task")
    if any(len(rows) != 1 for clusters in clusters_by_task.values() for rows in clusters.values()):
        raise ValueError("ADR-175 bootstrap requires one fixed sample per task/episode")
    global_cluster_ids = sorted(
        {cluster_id for task_clusters in clusters_by_task.values() for cluster_id in task_clusters}
    )
    cluster_count = len(global_cluster_ids)
    full_rows = [
        row
        for task_key in sorted(clusters_by_task)
        for cluster_id in sorted(clusters_by_task[task_key])
        for row in clusters_by_task[task_key][cluster_id]
    ]
    raw_estimate, normalized_estimate = _task_macro_pairs(full_rows)
    generator = random.Random(ADR175_BOOTSTRAP_SEED)
    raw_replicates: list[float] = []
    normalized_replicates: list[float] = []
    for _ in range(ADR175_BOOTSTRAP_REPLICATES):
        cluster_weights = {
            cluster_id: generator.expovariate(1.0) for cluster_id in global_cluster_ids
        }
        raw_task_values: list[float] = []
        normalized_task_values: list[float] = []
        for task_key in sorted(clusters_by_task):
            task_clusters = clusters_by_task[task_key]
            weighted_raw = 0.0
            weighted_normalized = 0.0
            total_weight = 0.0
            for cluster_id in sorted(task_clusters):
                weight = cluster_weights[cluster_id]
                rows = task_clusters[cluster_id]
                cluster_raw, cluster_normalized = _task_macro_pairs(rows)
                weighted_raw += weight * cluster_raw
                weighted_normalized += weight * cluster_normalized
                total_weight += weight
            raw_task_values.append(weighted_raw / total_weight)
            normalized_task_values.append(weighted_normalized / total_weight)
        raw_replicates.append(_mean(raw_task_values, name="Bayesian raw task macro"))
        normalized_replicates.append(
            _mean(normalized_task_values, name="Bayesian normalized task macro")
        )
    return {
        "cluster_unit": "source_episode",
        "cluster_count": cluster_count,
        "confidence_level": 0.95,
        "resampling_scheme": "paired_global_source_episode_bayesian",
        "replicates": ADR175_BOOTSTRAP_REPLICATES,
        "seed": ADR175_BOOTSTRAP_SEED,
        "reference_arm": "physical-set",
        "candidate_arm": "native-attention",
        "raw_estimate": raw_estimate,
        "raw_lower_bound": _percentile(raw_replicates, 0.025),
        "normalized_estimate": normalized_estimate,
        "normalized_lower_bound": _percentile(normalized_replicates, 0.025),
    }


def assemble_adr175_arm_reports(
    raw_reports: Mapping[str, Mapping[str, Any]],
    *,
    broad_support_contract: Adr175AssemblyContractIdentity,
    raw_report_file_sha256_by_arm: Mapping[str, str],
    snapshot_loader: Callable[[object], Mapping[str, Any]] = load_adr175_snapshot,
) -> dict[str, dict[str, object]]:
    """Assemble and seal all three arm reports from raw runner evidence."""

    broad_support_contract.validate_for_adr175()
    if tuple(sorted(raw_reports)) != tuple(sorted(ADR175_ARMS)):
        raise ValueError("ADR-175 assembly requires exactly lbot/physical-set/native-attention")
    if tuple(sorted(raw_report_file_sha256_by_arm)) != tuple(sorted(ADR175_ARMS)):
        raise ValueError("ADR-175 assembly requires one raw-report file SHA per arm")

    snapshots_by_arm: dict[str, dict[int, Mapping[str, Any]]] = {}
    step_receipts_by_arm: dict[str, list[dict[str, object]]] = {}
    shared_by_arm: dict[str, dict[str, object]] = {}
    treatment_contract_by_arm: dict[str, str] = {}
    observability_by_arm: dict[str, dict[str, bool]] = {}
    observability_receipts_by_arm: dict[str, dict[str, dict[str, object]]] = {}
    observability_sha256_by_arm: dict[str, str] = {}
    for arm in ADR175_ARMS:
        raw = _mapping(raw_reports[arm], name=f"{arm} raw report")
        if raw.get("status") != "PASS" or raw.get("steps") != ADR175_TOTAL_STEPS:
            raise ValueError(f"ADR-175 raw {arm} run is incomplete")
        metadata = _mapping(raw.get("adr175"), name=f"{arm} ADR-175 metadata")
        if metadata.get("arm") != arm or metadata.get("contract_artifact_sha256") != (
            broad_support_contract.artifact_sha256
        ):
            raise ValueError(f"ADR-175 raw {arm} contract identity differs")
        if metadata.get("matched_arm_input_sha256") != (
            broad_support_contract.matched_arm_input_sha256
        ):
            raise ValueError(f"ADR-175 raw {arm} matched-prefix identity differs")
        if raw.get("plan_sha256") != broad_support_contract.stream_plan_sha256:
            raise ValueError(f"ADR-175 raw {arm} stream plan differs from its contract")
        if raw.get("representation_split_sha256") != (
            broad_support_contract.representation_split_artifact_sha256
        ):
            raise ValueError(f"ADR-175 raw {arm} representation split differs")
        if raw.get("entity_evaluation_plan_sha256") != (
            broad_support_contract.entity_evaluation_plan_artifact_sha256
        ):
            raise ValueError(f"ADR-175 raw {arm} entity evaluation plan differs")
        dataset_contract = _mapping(raw.get("dataset_contract"), name="dataset contract")
        dataset_validation = _mapping(
            dataset_contract.get("validation"),
            name="dataset validation",
        )
        if (
            dataset_contract.get("status") != "PASS"
            or dataset_validation.get("dataset_tree_sha256")
            != broad_support_contract.dataset_manifest_sha256
        ):
            raise ValueError(f"ADR-175 raw {arm} dataset tree differs from its contract")
        optimizer_manifest = _mapping(
            metadata.get("shared_optimizer_manifest"),
            name=f"{arm} shared optimizer manifest",
        )
        optimizer_sha256 = _sha256(
            metadata.get("shared_optimizer_contract_sha256"),
            name=f"{arm} shared optimizer SHA-256",
        )
        if canonical_sha256(optimizer_manifest) != optimizer_sha256:
            raise ValueError(f"ADR-175 raw {arm} optimizer manifest digest differs")
        if optimizer_manifest.get("expected_update_count") != ADR175_TOTAL_STEPS:
            raise ValueError(f"ADR-175 raw {arm} optimizer update count differs")
        implementation_files = _mapping(
            raw.get("implementation_files"),
            name="implementation files",
        )
        implementation_sha256 = _sha256(
            raw.get("implementation_sha256"),
            name="implementation",
        )
        if canonical_sha256(implementation_files) != implementation_sha256:
            raise ValueError(f"ADR-175 raw {arm} implementation digest differs")
        runtime_contract = {
            "action_suffix_executed": raw.get("action_suffix_executed"),
            "alignment_teacher_prune": raw.get("alignment_teacher_prune"),
            "checkpoint_published": raw.get("checkpoint_published"),
            "cuda_allocator": raw.get("cuda_allocator"),
            "curve_mode": raw.get("curve_mode"),
            "fsdp2_placement": raw.get("fsdp2_placement"),
            "gradient_checkpointing": raw.get("gradient_checkpointing"),
            "maximum_peak_reserved_bytes": raw.get("maximum_peak_reserved_bytes"),
            "parameter_scope": raw.get("parameter_scope"),
            "posterior_input_mode": raw.get("posterior_input_mode"),
            "registered_evaluation_steps": raw.get("registered_evaluation_steps"),
            "representation_parameter_scope": raw.get("representation_parameter_scope"),
            "seed": raw.get("seed"),
            "schema": raw.get("schema"),
            "task_scorer_present": raw.get("task_scorer_present"),
            "world_size": raw.get("world_size"),
        }
        shared_by_arm[arm] = {
            "broad_support_contract_sha256": broad_support_contract.artifact_sha256,
            "broad_support_contract_file_sha256": _sha256(
                metadata.get("contract_file_sha256"),
                name="broad-support contract file",
            ),
            "matched_arm_input_sha256": broad_support_contract.matched_arm_input_sha256,
            "dataset_contract_sha256": canonical_sha256(dataset_contract),
            "physical_sidecar_manifest_sha256": _sha256(
                raw.get("physical_sidecar_manifest_sha256"),
                name="physical sidecar manifest",
            ),
            "stream_plan_sha256": _sha256(raw.get("plan_sha256"), name="stream plan"),
            "representation_split_sha256": _sha256(
                raw.get("representation_split_sha256"),
                name="representation split",
            ),
            "evaluation_plan_sha256": _sha256(
                raw.get("entity_evaluation_plan_sha256"),
                name="evaluation plan",
            ),
            "shared_initialization_sha256": _sha256(
                metadata.get("shared_initialization_sha256"),
                name="shared initialization",
            ),
            "shared_optimizer_contract_sha256": optimizer_sha256,
            "source_commit": _text(raw.get("source_commit"), name="source commit"),
            "source_patch_sha256": _sha256(
                raw.get("source_patch_sha256"),
                name="source patch",
            ),
            "patched_source_sha256": canonical_sha256(
                _mapping(raw.get("patched_source_sha256"), name="patched source")
            ),
            "implementation_sha256": implementation_sha256,
            "checkpoint_contract_sha256": canonical_sha256(
                {
                    "assets": raw.get("checkpoint_assets"),
                    "revision": raw.get("checkpoint_revision"),
                }
            ),
            "processor_contract_sha256": canonical_sha256(
                {
                    "assets": raw.get("processor_assets"),
                    "revision": raw.get("processor_revision"),
                }
            ),
            "objective_sha256": canonical_sha256(_mapping(raw.get("objective"), name="objective")),
            "vision_geometry_sha256": canonical_sha256(
                _mapping(raw.get("qwen_vision_geometry"), name="vision geometry")
            ),
            "runtime_contract_sha256": canonical_sha256(runtime_contract),
            "total_steps": ADR175_TOTAL_STEPS,
        }
        step_receipts_by_arm[arm] = _step_receipts(
            raw,
            arm=arm,
            contract=broad_support_contract,
        )
        if arm != "lbot":
            treatment_contract_by_arm[arm] = canonical_sha256(
                {
                    "architecture_identity": raw.get("architecture_identity"),
                    "graph": raw.get("graph"),
                    "model_family_sha256": raw.get("model_family_sha256"),
                    "parameter_manifest": raw.get("parameter_manifest"),
                    "parameter_scope": raw.get("parameter_scope"),
                    "parameter_storage": raw.get("parameter_storage"),
                    "picf_graph_sha256": metadata.get("picf_graph_sha256"),
                    "picf_initialization_sha256": metadata.get("picf_initialization_sha256"),
                    "relation_interface": raw.get("relation_interface"),
                }
            )
        snapshots_by_arm[arm] = _load_snapshots(raw, snapshot_loader=snapshot_loader)
        if arm != "lbot":
            milestone_observability = [
                _exact_observability(snapshot) for snapshot in snapshots_by_arm[arm].values()
            ]
            if len({item[2] for item in milestone_observability}) != 1:
                raise ValueError(f"ADR-175 {arm} observability changed across milestones")
            observability_by_arm[arm] = milestone_observability[0][0]
            observability_receipts_by_arm[arm] = milestone_observability[0][1]
            observability_sha256_by_arm[arm] = milestone_observability[0][2]
            for snapshot in snapshots_by_arm[arm].values():
                for sample in (
                    _mapping(value, name="ambiguous validity sample")
                    for value in _list(snapshot.get("samples"), name="snapshot samples")
                ):
                    if (
                        sample.get("task_key") in ADR175_AMBIGUOUS_TASKS
                        and sample.get("target_valid") is not False
                    ):
                        raise ValueError(
                            f"ADR-175 {arm} enabled a target row for an ambiguous task"
                        )

    if len({canonical_sha256(value) for value in shared_by_arm.values()}) != 1:
        raise ValueError("ADR-175 shared contracts differ across arms")
    if len(set(treatment_contract_by_arm.values())) != 1:
        raise ValueError("ADR-175 physical-set/native-attention treatment contracts differ")
    if len({canonical_sha256(value) for value in step_receipts_by_arm.values()}) != 1:
        raise ValueError("ADR-175 step receipts differ across arms")
    if len(set(observability_sha256_by_arm.values())) != 1:
        raise ValueError("ADR-175 treatment observability differs across arms")
    evaluation_inputs = {snapshots_by_arm[arm][0]["evaluation_input_sha256"] for arm in ADR175_ARMS}
    if len(evaluation_inputs) != 1:
        raise ValueError("ADR-175 evaluation inputs differ across arms")
    if (
        len({_snapshot_sample_identity_sha256(snapshots_by_arm[arm][0]) for arm in ADR175_ARMS})
        != 1
    ):
        raise ValueError("ADR-175 evaluation sample provenance differs across arms")

    bootstrap = _heldout_bootstrap(
        snapshots_by_arm["physical-set"][ADR175_TOTAL_STEPS],
        snapshots_by_arm["native-attention"][ADR175_TOTAL_STEPS],
        observable_by_sample=observability_by_arm["physical-set"],
    )
    sealed: dict[str, dict[str, object]] = {}
    for arm in ADR175_ARMS:
        raw_metadata = _mapping(raw_reports[arm]["adr175"], name=f"{arm} metadata")
        treatment = arm != "lbot"
        unsigned = {
            "schema": ADR175_ARM_REPORT_SCHEMA,
            "status": "COMPLETE",
            "arm": arm,
            "raw_report_file_sha256": _sha256(
                raw_report_file_sha256_by_arm[arm],
                name=f"{arm} raw-report file",
            ),
            "evaluation_evidence_sha256": canonical_sha256(
                [
                    {
                        key: receipt[key]
                        for key in (
                            "artifact_sha256",
                            "checkpoint_global_step",
                            "evaluation_input_sha256",
                            "file_sha256",
                            "partition_summaries",
                        )
                    }
                    for receipt in _list(
                        raw_reports[arm].get("evaluation_snapshots"),
                        name="evaluation snapshot receipts",
                    )
                ]
            ),
            "picf_treatment_contract_sha256": (
                treatment_contract_by_arm[arm] if treatment else None
            ),
            "shared_contract": shared_by_arm[arm],
            "picf_graph_sha256": (
                _sha256(raw_metadata.get("picf_graph_sha256"), name="PICF graph")
                if treatment
                else None
            ),
            "picf_initialization_sha256": (
                _sha256(
                    raw_metadata.get("picf_initialization_sha256"),
                    name="PICF initialization",
                )
                if treatment
                else None
            ),
            "exact_observability_sha256": (observability_sha256_by_arm[arm] if treatment else None),
            "ambiguous_target_validity": [
                {"task_key": task_key, "target_valid": False} for task_key in ADR175_AMBIGUOUS_TASKS
            ],
            "step_receipts": step_receipts_by_arm[arm],
            "milestones": _milestones(
                arm,
                snapshots_by_arm[arm],
                observable_by_sample=(observability_by_arm[arm] if treatment else None),
            ),
            "exact_strata": (
                _exact_strata(
                    snapshots_by_arm[arm][ADR175_TOTAL_STEPS],
                    observable_by_sample=observability_by_arm[arm],
                    observability_receipts_by_sample=observability_receipts_by_arm[arm],
                )
                if treatment
                else None
            ),
            "heldout_selectivity_bootstrap": (bootstrap if arm == "native-attention" else None),
        }
        sealed[arm] = seal_adr175_arm_report(unsigned)
    return sealed

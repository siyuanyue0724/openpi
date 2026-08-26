"""Fail-closed composition for process-isolated LTOP G3 evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final

G3_FINAL_SCHEMA: Final = "picf-next.ltop-g3-production-action-mediation.v1"
G3_TRAINING_SCHEMA: Final = "picf-next.ltop-g3-training-phase.v1"
G3_EVALUATION_SCHEMA: Final = "picf-next.ltop-g3-evaluation-phase.v1"

_MATCHED_FIELDS: Final = (
    "mode",
    "architecture_identity",
    "runtime_source_contract",
    "world_size",
    "steps",
    "eval_every",
    "seed",
    "capacity",
    "task_query_count",
    "stage_checkpoint",
    "g2_report_sha256",
    "dataset_contract",
    "execution_contract_sha256",
    "offline_labels_sha256",
    "physical_sidecar_manifest_sha256",
)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_phase(path: Path, *, schema: str, phase: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"G3 {phase} report is absent or not a regular file: {path}")
    payload = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(payload, dict):
        raise TypeError(f"G3 {phase} report must be an object")
    expected = {
        "schema": schema,
        "status": "PASS",
        "failures": [],
        "phase": phase,
        "mode": "gate",
        "world_size": 2,
        "steps": 128,
        "eval_every": 32,
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ValueError(f"G3 {phase} report {field} differs from the registered contract")
    return payload


def _rank_map(report: dict[str, Any], *, phase: str) -> dict[int, dict[str, Any]]:
    reports = report.get("rank_reports")
    if not isinstance(reports, list) or len(reports) != 2:
        raise ValueError(f"G3 {phase} report must contain two ranks")
    result: dict[int, dict[str, Any]] = {}
    for item in reports:
        if not isinstance(item, dict):
            raise TypeError(f"G3 {phase} rank report is not an object")
        rank = item.get("rank")
        if isinstance(rank, bool) or not isinstance(rank, int) or rank not in (0, 1):
            raise ValueError(f"G3 {phase} rank is invalid")
        if rank in result:
            raise ValueError(f"G3 {phase} report contains a duplicate rank")
        result[rank] = item
    if set(result) != {0, 1}:
        raise ValueError(f"G3 {phase} report rank set is incomplete")
    return result


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("G3 staged composer cannot average an empty sequence")
    return sum(values) / len(values)


def compose_staged_g3(*, training_path: Path, evaluation_path: Path) -> dict[str, Any]:
    """Compose separate training and action-evaluation processes into the G3 ABI."""

    training = _load_phase(training_path, schema=G3_TRAINING_SCHEMA, phase="training")
    evaluation = _load_phase(
        evaluation_path,
        schema=G3_EVALUATION_SCHEMA,
        phase="evaluation",
    )
    for field in _MATCHED_FIELDS:
        if training.get(field) != evaluation.get(field):
            raise ValueError(f"G3 staged phases differ at {field}")

    checkpoint = training.get("checkpoint")
    if not isinstance(checkpoint, dict) or checkpoint.get("optimizer_saved") is not False:
        raise ValueError("G3 training report omits its model-only checkpoint")
    checkpoint_path = checkpoint.get("path")
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("G3 training checkpoint path is invalid")
    if evaluation.get("trained_checkpoint") != checkpoint_path:
        raise ValueError("G3 evaluation did not restore the training-phase checkpoint")

    train_ranks = _rank_map(training, phase="training")
    eval_ranks = _rank_map(evaluation, phase="evaluation")
    for rank in (0, 1):
        if train_ranks[rank].get("runtime_schedule_sha256") != eval_ranks[rank].get(
            "runtime_schedule_sha256"
        ):
            raise ValueError(f"G3 rank {rank} runtime schedule changed between phases")
        if len(train_ranks[rank].get("action_losses", ())) != 128:
            raise ValueError(f"G3 rank {rank} training loss trace is incomplete")
        if len(eval_ranks[rank].get("history", ())) != 1:
            raise ValueError(f"G3 rank {rank} evaluation receipt is incomplete")

    first_losses = [
        float(value) for rank in (0, 1) for value in train_ranks[rank]["action_losses"][:16]
    ]
    last_losses = [
        float(value) for rank in (0, 1) for value in train_ranks[rank]["action_losses"][-16:]
    ]
    if _mean(last_losses) >= 0.95 * _mean(first_losses):
        raise ValueError("G3 staged action loss did not improve by at least five percent")

    for partition in ("validation", "heldout"):
        scores = [
            scene["score"]
            for rank in (0, 1)
            for scene in eval_ranks[rank]["history"][0][partition]["scenes"]
        ]
        if _mean([float(score["mean_factual_target_minus_distractor"]) for score in scores]) <= 0:
            raise ValueError(f"G3 staged {partition} target-row action effect is nonpositive")
        if (
            _mean([float(score["mean_blocked_path_difference_in_differences"]) for score in scores])
            <= 0
        ):
            raise ValueError(f"G3 staged {partition} blocked-path DID is nonpositive")
        sample_count = sum(len(score["sample_keys"]) for score in scores)
        required = math.ceil(0.625 * sample_count)
        if sum(int(score["positive_factual_count"]) for score in scores) < required:
            raise ValueError(f"G3 staged {partition} factual count is below threshold")
        if sum(int(score["positive_blocked_path_did_count"]) for score in scores) < required:
            raise ValueError(f"G3 staged {partition} blocked DID count is below threshold")

    rank_reports = []
    for rank in (0, 1):
        merged = dict(train_ranks[rank])
        merged["history"] = eval_ranks[rank]["history"]
        merged["staged_evaluation_cuda_memory_bytes"] = eval_ranks[rank]["cuda_memory_bytes"]
        rank_reports.append(merged)
    return {
        "schema": G3_FINAL_SCHEMA,
        "status": "PASS",
        "failures": [],
        "phase": "fresh-process-composed",
        **{field: training[field] for field in _MATCHED_FIELDS},
        "checkpoint": checkpoint,
        "action_inference_contract": evaluation["action_inference_contract"],
        "training_contract": training["training_contract"],
        "thresholds": evaluation["thresholds"],
        "staged_evidence": {
            "training_report": str(training_path.resolve()),
            "training_report_sha256": _file_sha256(training_path),
            "evaluation_report": str(evaluation_path.resolve()),
            "evaluation_report_sha256": _file_sha256(evaluation_path),
            "training_and_evaluation_processes_are_disjoint": True,
        },
        "rank_reports": rank_reports,
    }

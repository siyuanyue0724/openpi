#!/usr/bin/env python3
# ruff: noqa: E402, I001
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run one registered two-GPU LTOP exact-cache execution.

Historical pilot/smoke executions restore the accepted G2b state for their
frozen engineering comparison. Long and restart-smoke executions require the
complete ADR172 fixed-head training, cold-causal and retention evidence, then
strictly cold-load the accepted model-only checkpoint before optimizer
construction. Production training keeps the direct-posterior action route
factual; ``BLOCKED`` remains an evaluation-only causal intervention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _repository_import_path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _repository_import_text = str(_repository_import_path)
    while _repository_import_text in sys.path:
        sys.path.remove(_repository_import_text)
    sys.path.insert(0, _repository_import_text)

from tools.cuda_allocator_bootstrap import (
    CUDA_ALLOCATOR_MODES,
    bootstrap_cuda_allocator,
    configure_cuda_allocator as _configure_cuda_allocator,
)

_BOOTSTRAPPED_CUDA_ALLOCATOR = (
    bootstrap_cuda_allocator(sys.argv[1:]) if __name__ == "__main__" else None
)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.capacity import (
    require_checkpoint_write_capacity,
    require_persistent_run_root,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from picf_next.lingbot_native.ltop_core_pilot import (
    AcceptedG3MediatorGate,
    LTOP_CORE_PILOT_SCHEMA,
    LTOP_CORE_PILOT_WORLD_SIZE,
    LTOP_CORE_PILOT_MODES,
    LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
    LTOP_CORE_LONG_TOTAL_STEPS,
    LTOPCorePilotArm,
    LTOPCorePilotCadence,
    LTOPCoreLongCadence,
    LTOPCoreRestartSmokeCadence,
    LTOPCorePilotSmokeCadence,
    load_accepted_g3_gate,
    matched_arm_contract,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from picf_next.training.run_lease import acquire_distributed_run_lease
from tools.bootstrap_lingbot_vla2_native import CHECKOUT_RELATIVE_PATH, PATCH_RELATIVE_PATH
from tools.lingbot_vla2_ltop_stage_runtime import (
    LingBotVLA2LTOPStageRequest,
    ltop_stage_runtime_source_contract,
    open_lingbot_vla2_ltop_stage_runtime,
    prepare_lingbot_vla2_ltop_stage_transfer,
)
from tools.lingbot_vla2_runtime_helpers import (
    _resolve_training_config,
    build_lingbot_official_optimizer,
    clip_lingbot_distributed_l2_grad_norm_,
    require_lingbot_exact_resume_contract,
)
from tools.run_lingbot_vla2_ltop_g2_core import (
    G2_ARCHITECTURE,
    G2_CAPACITY,
    G2_TASK_QUERY_COUNT,
    _episode_ids,
    _sha256,
)
from tools.run_lingbot_vla2_native_g0 import (
    _canonical,
    _capture_rank_rng,
    _checkpoint_boundary,
    _distributed_gradient_metrics,
    _distributed_rank_local_call,
    _fsync_tree,
    _git_output,
    _model_local_state_digest,
    _move_model_inputs,
    _rank_rng_digest,
    _restore_rank_rng,
    _update_tensor_digest,
    _validate_optimizer_state,
    _write_text_durable,
)


CORE_PILOT_CHECKPOINT_SCHEMA = "picf-next.ltop-core-pilot-checkpoint.v2"
CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA = "picf-next.ltop-core-pilot-checkpoint-extra.v1"
CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA = "picf-next.ltop-core-pilot-checkpoint-provenance.v3"
CORE_PILOT_METRICS_SCHEMA = "picf-next.ltop-core-pilot-metrics.v1"
CORE_PILOT_DIAGNOSTIC_SCHEMA = "picf-next.ltop-core-pilot-diagnostic.v1"
CORE_PILOT_PROGRESS_SCHEMA = "picf-next.ltop-core-pilot-progress.v1"
CORE_PILOT_INPUT_RECEIPT_SCHEMA = "picf-next.ltop-core-pilot-input-receipt.v1"
CORE_PILOT_JOURNAL_RECEIPT_SCHEMA = "picf-next.ltop-core-pilot-rank-journal.v1"
CORE_PILOT_OPTIMIZER_INITIALIZATION_SCHEMA = "picf-next.ltop-core-pilot-optimizer-initialization.v1"
CORE_PILOT_RUNTIME_ENVIRONMENT_SCHEMA = "picf-next.ltop-core-pilot-runtime-environment.v1"
CORE_PILOT_SOURCE_IDENTITY_SCHEMA = "picf-next.ltop-core-pilot-source-identity.v1"
CORE_PILOT_COLD_RESUME_SCHEMA = "picf-next.ltop-core-pilot-cold-resume.v1"
CORE_LONG_PRUNED_CHECKPOINT_SCHEMA = "picf-next.ltop-long-pruned-checkpoint.v1"
CORE_LONG_ACTION_INFORMATION_SET_SCHEDULE_SCHEMA = (
    "picf-next.ltop-core-long-action-information-set-schedule.v1"
)
CORE_LONG_MINIMUM_CHECKPOINT_WRITE_BYTES = 80 * 2**30
CORE_LONG_CHECKPOINT_SAFETY_MARGIN_BYTES = 16 * 2**30
CORE_ACTION_INFORMATION_SET_POLICIES = (
    "factual-only",
    LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
)
ADR174_FIXED_HEAD_OBJECTIVE_SCHEMA = "picf-next.adr174-fixed-head-objective.v1"
ADR174_FIXED_HEAD_ROUTE = "native-task-independent-direct-posterior"
ADR174_FIXED_HEAD_SCOPE = "guidedvla-fixed-object-heads-0-1"
ADR174_FIXED_HEAD_INDICES = (0, 1)
ADR174_FIXED_HEAD_LAYERS = (32, 35)
ADR174_FIXED_HEAD_WEIGHT = 0.001
ADR172_COLD_REPORT_SCHEMA = "picf-next.adr172-direct-action-posterior-evaluation.v1"
ADR172_COLD_VALIDATION_SCHEMA = "picf-next.adr172-direct-posterior-cold-validation.v1"
ADR172_RETENTION_REPORT_SCHEMA = "picf-next.adr172-direct-action-posterior-retention.v1"
ADR172_TRAINING_REPORT_SCHEMA = "picf-next.adr172-direct-action-posterior-training.v1"
ADR172_TRAINING_CHECKPOINT_SCHEMA = "picf-next.adr172-direct-posterior-training-checkpoint.v1"
ADR172_TRAINING_CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
ADR172_TRAINING_MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
ADR172_ACTION_SUPERVISION_SCHEMA = "picf-next.task-action-supervision.v1"
ADR172_PICF_SOURCE_CONTRACT_SCHEMA = "picf-next.g3-picf-source-contract.v1"
ADR172_GUIDEDVLA_UPSTREAM_CONTRACT = {
    "repository": "GuidedVLA",
    "repository_commit": "04be059e0d6bd448be5cb45fdbafc775f7eb5e38",
    "config_name": "pi0_libero_object_depth_skill",
    "object_use_control": False,
    "object_head_indices": list(ADR174_FIXED_HEAD_INDICES),
    "object_loss_head_aggregation": "mean_heads",
    "object_loss_weight": ADR174_FIXED_HEAD_WEIGHT,
    "weight_source": "named LIBERO full object-depth-skill recipe",
    "critical_file_sha256": {
        "src/openpi/models/pi0_config.py": (
            "9189bbe8bd2dd3d92d67775c9a6a8abc5b02f3132cca017ff5847a8b8fb492eb"
        ),
        "src/openpi/models_pytorch/pi0_pytorch.py": (
            "2c476c60c4365cef36a8f7c3fe4a4945e0c7f1f208e0341b1479a3dd2798391a"
        ),
        "src/openpi/training/config.py": (
            "2ff2258d14c2e56a3cf286c67e087be91b96316aeab185c99f6d619f49812c9e"
        ),
        "scripts/train_pytorch.py": (
            "b8da08fab3275b9d642bdb1a111bc186787dca9834c0c2582d297e71c70c8836"
        ),
    },
}
ADR172_PICF_CRITICAL_SOURCE_FILES = (
    "tools/run_lingbot_vla2_ltop_adr172_direct_posterior.py",
    "src/picf_next/lingbot_native/action_posterior_receipt.py",
    "src/picf_next/lingbot_native/action_posterior_collector.py",
    "src/picf_next/lingbot_native/action_posterior_learning.py",
    "src/picf_next/lingbot_native/graph.py",
    "src/picf_next/lingbot_native/host.py",
    "src/picf_next/lingbot_native/ltop_action_mediation.py",
    "src/picf_next/lingbot_native/task_address_target.py",
    "src/picf_next/lingbot_native/task_action_supervision.py",
)

_CHECKPOINT_BOUNDARY_KEYS = frozenset(
    {
        "lane_snapshot_sha256",
        "model_local_state_sha256",
        "optimizer_local_state_sha256",
        "rank_rng_state_sha256",
    }
)
_CHECKPOINT_EXTRA_KEYS = frozenset(
    {
        "boundary_sha256",
        "global_step",
        "lane_snapshot",
        "next_optimizer_step",
        "optimizer_local_moment_elements",
        "optimizer_state_entries",
        "provenance",
        "provenance_sha256",
        "rank",
        "rank_rng_state",
        "schema",
        "source_digest",
        "world_size",
    }
)

_CONTROL_TENSOR_FIELDS = (
    "values",
    "field_valid",
    "token_valid",
    "delta_time",
    "reset",
    "acknowledged",
)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None else Path(value)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=_environment_path("PICF_LINGBOT_NATIVE_SOURCE")
        or _REPOSITORY_ROOT / CHECKOUT_RELATIVE_PATH,
    )
    parser.add_argument(
        "--patch",
        type=Path,
        default=_REPOSITORY_ROOT / PATCH_RELATIVE_PATH,
    )
    parser.add_argument("--runtime-hotfix", type=Path, default=None)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=_REPOSITORY_ROOT / "configs/lingbot/calvin_robot.yaml",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=_environment_path("PICF_CHECKPOINT_DIR"),
    )
    parser.add_argument(
        "--processor-dir",
        type=Path,
        default=_environment_path("PICF_PROCESSOR_DIR"),
    )
    parser.add_argument("--stage-checkpoint", type=Path, required=True)
    parser.add_argument("--g2-report", type=Path, required=True)
    parser.add_argument("--g3-report", type=Path, required=True)
    parser.add_argument("--adr172-cold-report", type=Path, default=None)
    parser.add_argument("--adr172-cold-validation", type=Path, default=None)
    parser.add_argument("--adr172-physical-retention-report", type=Path, default=None)
    parser.add_argument(
        "--action-information-set-policy",
        choices=CORE_ACTION_INFORMATION_SET_POLICIES,
        required=True,
    )
    parser.add_argument(
        "--dataset-split",
        type=Path,
        default=_environment_path("PICF_DATASET_DIR"),
    )
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_environment_path("PICF_DATASET_MANIFEST"),
    )
    parser.add_argument(
        "--norm-stats",
        type=Path,
        default=_environment_path("PICF_LINGBOT_NORM_STATS"),
    )
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--stream-plan", type=Path, required=True)
    parser.add_argument("--stream-plan-sha256", required=True)
    parser.add_argument("--representation-split", type=Path, required=True)
    parser.add_argument("--representation-split-sha256", required=True)
    parser.add_argument("--evaluation-plan", type=Path, required=True)
    parser.add_argument("--evaluation-plan-sha256", required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--offline-labels", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=tuple(arm.value for arm in LTOPCorePilotArm),
        required=True,
    )
    parser.add_argument("--mode", choices=LTOP_CORE_PILOT_MODES, default="pilot")
    parser.add_argument("--phase", choices=("fresh", "resume"), default="fresh")
    parser.add_argument("--load-global-step", type=int, default=0)
    parser.add_argument("--stop-after-step", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--capacity", type=int, default=G2_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G2_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--maximum-grad-norm", type=float, default=1.0)
    parser.add_argument("--physical-set-weight", type=float, default=1.0)
    parser.add_argument("--official-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
    parser.add_argument(
        "--cuda-allocator",
        choices=CUDA_ALLOCATOR_MODES,
        default="native",
    )
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    required_paths = {
        "source checkout": args.source_checkout,
        "patch": args.patch,
        "training config": args.training_config,
        "robot config": args.robot_config,
        "checkpoint": args.checkpoint_dir,
        "processor": args.processor_dir,
        "G2b stage checkpoint": args.stage_checkpoint,
        "G2b report": args.g2_report,
        "G3 PASS report": args.g3_report,
        "dataset split": args.dataset_split,
        "dataset manifest": args.dataset_manifest,
        "normalization": args.norm_stats,
        "physical sidecar": args.physical_sidecar_root,
        "physical sidecar manifest": args.physical_sidecar_manifest,
        "stream plan": args.stream_plan,
        "representation split": args.representation_split,
        "evaluation plan": args.evaluation_plan,
        "execution contract": args.execution_contract,
        "offline labels": args.offline_labels,
    }
    if args.runtime_hotfix is not None:
        required_paths["runtime optimizer hotfix"] = args.runtime_hotfix
    if args.mode in {"long", "restart-smoke"}:
        required_paths.update(
            {
                "ADR172 cold report": args.adr172_cold_report,
                "ADR172 independent cold validation": args.adr172_cold_validation,
                "ADR172 physical retention report": args.adr172_physical_retention_report,
            }
        )
    missing = [name for name, path in required_paths.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"LTOP core-pilot required paths are absent: {missing}")
    if args.run_dir.is_symlink():
        raise ValueError("LTOP core-pilot run directory cannot be a symbolic link")
    integer_fields = (
        "seed",
        "capacity",
        "task_query_count",
        "maximum_control_tokens",
    )
    for name in integer_fields:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"LTOP core-pilot {name} must be a positive integer")
    if args.capacity != G2_CAPACITY or args.task_query_count != G2_TASK_QUERY_COUNT:
        raise ValueError("LTOP core pilot must preserve the accepted G2b graph shape")
    expected_information_set_policy = "factual-only"
    if args.action_information_set_policy != expected_information_set_policy:
        raise ValueError("LTOP action-information-set policy differs from the selected mode")
    if args.arm != LTOPCorePilotArm.FACTUAL.value:
        raise ValueError("ADR174 fixed-head production keeps the action route factual")
    for name in (
        "physical_sidecar_manifest_sha256",
        "stream_plan_sha256",
        "representation_split_sha256",
        "evaluation_plan_sha256",
    ):
        value = getattr(args, name)
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"LTOP core-pilot {name} must be one lowercase SHA-256")
    for name in (
        "maximum_grad_norm",
        "physical_set_weight",
        "official_loss_weight",
    ):
        value = getattr(args, name)
        if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"LTOP core-pilot {name} must be finite and positive")


def _cadence_for_mode(mode: str) -> Any:
    if mode == "smoke":
        return LTOPCorePilotSmokeCadence()
    if mode == "restart-smoke":
        return LTOPCoreRestartSmokeCadence()
    if mode == "pilot":
        return LTOPCorePilotCadence()
    if mode == "long":
        return LTOPCoreLongCadence()
    raise ValueError(f"unsupported LTOP core-pilot mode: {mode}")


def _resolve_run_interval(args: argparse.Namespace, cadence: Any) -> tuple[int, int]:
    load_step = args.load_global_step
    if isinstance(load_step, bool) or not isinstance(load_step, int) or load_step < 0:
        raise ValueError("LTOP load-global-step must be a non-negative integer")
    if args.phase == "fresh":
        if load_step != 0:
            raise ValueError("fresh LTOP execution must start at global step zero")
    elif load_step <= 0 or not cadence.checkpoint_due(load_step):
        raise ValueError("resume LTOP execution must load a registered checkpoint boundary")
    stop_step = cadence.total_steps if args.stop_after_step is None else args.stop_after_step
    if isinstance(stop_step, bool) or not isinstance(stop_step, int):
        raise ValueError("LTOP stop-after-step must be an integer")
    if stop_step < load_step or stop_step > cadence.total_steps:
        raise ValueError("LTOP stop-after-step is outside the selected execution interval")
    if stop_step == load_step:
        if args.phase != "resume":
            raise ValueError("only a resume process may perform checkpoint verification only")
    elif not cadence.checkpoint_due(stop_step):
        raise ValueError("LTOP execution must stop on a registered checkpoint boundary")
    return load_step, stop_step


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_json_atomic_replace(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(_canonical_json(payload) + "\n", encoding="ascii")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _score_to_json(score: Any) -> dict[str, Any]:
    payload = asdict(score)
    for name, value in tuple(payload.items()):
        if hasattr(value, "detach"):
            payload[name] = value.detach().float().cpu().tolist()
    return payload


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("LTOP core pilot cannot average an empty sequence")
    return sum(values) / len(values)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fixed_head_objective_contract() -> dict[str, Any]:
    return {
        "schema": ADR174_FIXED_HEAD_OBJECTIVE_SCHEMA,
        "route": ADR174_FIXED_HEAD_ROUTE,
        "registered_layer_indices": list(ADR174_FIXED_HEAD_LAYERS),
        "head_scope": ADR174_FIXED_HEAD_SCOPE,
        "head_indices": list(ADR174_FIXED_HEAD_INDICES),
        "head_aggregation": "mean_heads",
        "loss": "negative-log-full-key-target-posterior-mass",
        "loss_weight": ADR174_FIXED_HEAD_WEIGHT,
        "single_forward_per_optimizer_step": True,
        "deploy_time_module_added": False,
        "replaces_old_task_address_objective": True,
        "upstream_contract": json.loads(_canonical_json(ADR172_GUIDEDVLA_UPSTREAM_CONTRACT)),
    }


def _read_regular_json(path: Path, *, name: str) -> tuple[dict[str, Any], str]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{name} is absent or not a regular file: {path}")
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise TypeError(f"{name} must be a JSON object")
    return value, hashlib.sha256(raw).hexdigest()


def _require_failure_free_report(
    report: Mapping[str, Any],
    *,
    name: str,
    schema: str,
    phase: str,
) -> None:
    if report.get("schema") != schema:
        raise ValueError(f"{name} schema differs")
    if report.get("status") != "PASS" or report.get("failures") != []:
        raise ValueError(f"{name} is not a failure-free PASS")
    if report.get("mode") != "gate" or report.get("phase") != phase:
        raise ValueError(f"{name} must use mode=gate phase={phase}")
    if report.get("world_size") != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError(f"{name} uses another distributed topology")
    if report.get("architecture_identity") != G2_ARCHITECTURE:
        raise ValueError(f"{name} belongs to another architecture")
    if report.get("capacity") != G2_CAPACITY:
        raise ValueError(f"{name} changed posterior capacity")
    if report.get("task_query_count") != G2_TASK_QUERY_COUNT:
        raise ValueError(f"{name} changed task-query count")


def _require_fixed_head_training_contract(report: Mapping[str, Any], *, name: str) -> None:
    training = report.get("training_contract")
    if not isinstance(training, Mapping):
        raise ValueError(f"{name} omits its training contract")
    adoption = training.get("direct_posterior_adoption")
    expected = _fixed_head_objective_contract()
    expected_adoption = {
        "route": expected["route"],
        "registered_layer_indices": expected["registered_layer_indices"],
        "head_scope": expected["head_scope"],
        "head_indices": expected["head_indices"],
        "upstream_contract": expected["upstream_contract"],
        "single_forward_per_optimizer_step": True,
        "deploy_time_module_added": False,
    }
    if adoption != expected_adoption:
        raise ValueError(f"{name} fixed-head adoption contract differs")
    weights = training.get("loss_weights")
    if not isinstance(weights, Mapping) or weights.get("direct_grounding") != (
        ADR174_FIXED_HEAD_WEIGHT
    ):
        raise ValueError(f"{name} fixed-head loss weight differs")


def _adr172_rank_checkpoint_identity(
    report: Mapping[str, Any],
    *,
    name: str,
) -> dict[str, Any]:
    rank_reports = report.get("rank_reports")
    if not isinstance(rank_reports, list) or len(rank_reports) != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError(f"{name} omits distributed checkpoint identity")
    ordered = sorted(rank_reports, key=lambda value: value.get("rank", -1))
    if [value.get("rank") for value in ordered] != list(range(LTOP_CORE_PILOT_WORLD_SIZE)):
        raise ValueError(f"{name} checkpoint identity rank set differs")
    trees = {value.get("trained_checkpoint_model_tree_sha256") for value in ordered}
    if len(trees) != 1:
        raise ValueError(f"{name} ranks disagree on checkpoint model tree")
    tree = next(iter(trees))
    model_digests = [value.get("trained_model_local_state_sha256") for value in ordered]
    for label, digest in (("model tree", tree), *(('model state', item) for item in model_digests)):
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(f"{name} {label} digest is malformed")
    return {
        "model_tree_sha256": tree,
        "model_local_state_sha256_by_rank": model_digests,
    }


def _load_accepted_adr172_fixed_head_training(path: Path) -> AcceptedG3MediatorGate:
    """Load the exact ADR172 model-only checkpoint authorized for production."""

    report, report_sha256 = _read_regular_json(path, name="ADR172 training report")
    if report.get("schema") != ADR172_TRAINING_REPORT_SCHEMA:
        raise ValueError("ADR172 training report schema differs")
    if report.get("status") != "PASS" or report.get("failures") != []:
        raise ValueError("ADR172 training report is not a failure-free PASS")
    expected_run = {
        "mode": "direct-trial",
        "phase": "training",
        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
        "steps": 256,
        "eval_every": 32,
        "architecture_identity": G2_ARCHITECTURE,
        "capacity": G2_CAPACITY,
        "task_query_count": G2_TASK_QUERY_COUNT,
    }
    for field, expected in expected_run.items():
        if report.get(field) != expected:
            raise ValueError(f"ADR172 training report changed {field}")
    _require_fixed_head_training_contract(report, name="ADR172 training report")

    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("ADR172 training report omits its checkpoint receipt")
    expected_checkpoint = {
        "format": ADR172_TRAINING_CHECKPOINT_FORMAT,
        "optimizer_saved": False,
        "action_supervision_schema": ADR172_ACTION_SUPERVISION_SCHEMA,
        "direct_grounding_weight": ADR174_FIXED_HEAD_WEIGHT,
        "direct_posterior_head_scope": ADR174_FIXED_HEAD_SCOPE,
        "direct_posterior_head_indices": list(ADR174_FIXED_HEAD_INDICES),
        "direct_posterior_registered_layer_indices": list(ADR174_FIXED_HEAD_LAYERS),
        "model_tree_schema": ADR172_TRAINING_MODEL_TREE_SCHEMA,
    }
    for field, expected in expected_checkpoint.items():
        if checkpoint.get(field) != expected:
            raise ValueError(f"ADR172 training checkpoint changed {field}")
    checkpoint_value = checkpoint.get("path")
    if not isinstance(checkpoint_value, str) or not Path(checkpoint_value).is_absolute():
        raise ValueError("ADR172 training checkpoint path must be absolute")
    checkpoint_path = Path(checkpoint_value)
    if checkpoint_path.is_symlink() or not checkpoint_path.is_dir():
        raise FileNotFoundError("ADR172 training checkpoint is absent or not a real directory")
    checkpoint_manifest_path = checkpoint_path / "ltop_g3_training_checkpoint.json"
    manifest, manifest_sha256 = _read_regular_json(
        checkpoint_manifest_path,
        name="ADR172 training checkpoint manifest",
    )
    if checkpoint.get("manifest_sha256") != manifest_sha256:
        raise ValueError("ADR172 training report is not byte-bound to its checkpoint manifest")

    identity = _adr172_rank_checkpoint_identity(report, name="ADR172 training report")
    model_tree_sha256 = identity["model_tree_sha256"]
    model_local_state_sha256_by_rank = identity["model_local_state_sha256_by_rank"]
    expected_manifest = {
        "schema": ADR172_TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "format": ADR172_TRAINING_CHECKPOINT_FORMAT,
        "optimizer_saved": False,
        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
        "global_step": 256,
        "action_supervision_schema": ADR172_ACTION_SUPERVISION_SCHEMA,
        "direct_action_causal_surface": report.get("direct_action_causal_surface"),
        "direct_grounding_weight": ADR174_FIXED_HEAD_WEIGHT,
        "direct_posterior_head_scope": ADR174_FIXED_HEAD_SCOPE,
        "direct_posterior_head_indices": list(ADR174_FIXED_HEAD_INDICES),
        "direct_posterior_registered_layer_indices": list(ADR174_FIXED_HEAD_LAYERS),
        "direct_grounding_upstream_contract": ADR172_GUIDEDVLA_UPSTREAM_CONTRACT,
        "source_stage_checkpoint": report.get("stage_checkpoint"),
        "g2_report_sha256": report.get("g2_report_sha256"),
        "runtime_source_contract": report.get("runtime_source_contract"),
        "picf_source_contract": report.get("trained_picf_source_contract"),
        "model_tree_schema": ADR172_TRAINING_MODEL_TREE_SCHEMA,
        "model_tree_sha256": model_tree_sha256,
        "training_final_model_local_state_sha256_by_rank": (
            model_local_state_sha256_by_rank
        ),
    }
    for field, expected in expected_manifest.items():
        if manifest.get(field) != expected:
            raise ValueError(f"ADR172 training checkpoint manifest changed {field}")
    if checkpoint.get("model_tree_sha256") != model_tree_sha256:
        raise ValueError("ADR172 training report checkpoint model tree differs")
    if checkpoint.get("training_final_model_local_state_sha256_by_rank") != (
        model_local_state_sha256_by_rank
    ):
        raise ValueError("ADR172 training report rank-local model identity differs")
    if checkpoint.get("picf_source_contract") != report.get("trained_picf_source_contract"):
        raise ValueError("ADR172 training checkpoint source provenance differs")

    return AcceptedG3MediatorGate(
        path=path.resolve(),
        file_sha256=report_sha256,
        report=report,
        checkpoint_path=checkpoint_path.resolve(),
        training_final_model_local_state_sha256_by_rank=(
            model_local_state_sha256_by_rank[0],
            model_local_state_sha256_by_rank[1],
        ),
        checkpoint_model_tree_sha256=model_tree_sha256,
    )


def _load_adr172_fixed_head_evidence(
    *,
    cold_report_path: Path,
    cold_validation_path: Path,
    retention_report_path: Path,
) -> dict[str, Any]:
    cold, cold_sha256 = _read_regular_json(cold_report_path, name="ADR172 cold report")
    validation, validation_sha256 = _read_regular_json(
        cold_validation_path,
        name="ADR172 independent cold validation",
    )
    retention, retention_sha256 = _read_regular_json(
        retention_report_path,
        name="ADR172 physical retention report",
    )
    _require_failure_free_report(
        cold,
        name="ADR172 cold report",
        schema=ADR172_COLD_REPORT_SCHEMA,
        phase="evaluation",
    )
    _require_failure_free_report(
        retention,
        name="ADR172 physical retention report",
        schema=ADR172_RETENTION_REPORT_SCHEMA,
        phase="retention",
    )
    if validation.get("schema") != ADR172_COLD_VALIDATION_SCHEMA:
        raise ValueError("ADR172 independent cold validation schema differs")
    if validation.get("status") != "PASS" or validation.get("failures") != []:
        raise ValueError("ADR172 independent cold validation is not a failure-free PASS")
    if validation.get("source_report_sha256") != cold_sha256:
        raise ValueError("ADR172 independent cold validation is not byte-bound to the report")
    source_report = validation.get("source_report")
    if not isinstance(source_report, str) or Path(source_report).resolve() != (
        cold_report_path.resolve()
    ):
        raise ValueError("ADR172 independent cold validation names another report")
    _require_fixed_head_training_contract(cold, name="ADR172 cold report")
    _require_fixed_head_training_contract(retention, name="ADR172 physical retention report")
    cold_checkpoint_identity = _adr172_rank_checkpoint_identity(
        cold,
        name="ADR172 cold report",
    )
    retention_checkpoint_identity = _adr172_rank_checkpoint_identity(
        retention,
        name="ADR172 physical retention report",
    )
    if cold_checkpoint_identity != retention_checkpoint_identity:
        raise ValueError("ADR172 cold and retention checkpoint identities differ")

    shared_fields = (
        "architecture_identity",
        "capacity",
        "task_query_count",
        "stage_checkpoint",
        "trained_checkpoint",
        "g2_report_sha256",
        "runtime_source_contract",
        "picf_source_contract",
        "trained_picf_source_contract",
        "dataset_contract",
        "execution_contract_sha256",
        "offline_labels_sha256",
        "physical_sidecar_manifest_sha256",
        "training_contract",
    )
    for field in shared_fields:
        if cold.get(field) != retention.get(field):
            raise ValueError(f"ADR172 cold and retention evidence differ at {field}")
    trained_source = cold.get("trained_picf_source_contract")
    if not isinstance(trained_source, Mapping):
        raise ValueError("ADR172 evidence omits trained source provenance")
    if trained_source.get("schema") != ADR172_PICF_SOURCE_CONTRACT_SCHEMA:
        raise ValueError("ADR172 trained source provenance schema differs")
    critical_files = trained_source.get("critical_file_sha256")
    if not isinstance(critical_files, Mapping) or set(critical_files) != set(
        ADR172_PICF_CRITICAL_SOURCE_FILES
    ):
        raise ValueError("ADR172 trained source critical-file set differs")
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in critical_files.values()
    ):
        raise ValueError("ADR172 trained source contains a malformed file digest")
    return {
        "objective": _fixed_head_objective_contract(),
        "cold_report": str(cold_report_path.resolve()),
        "cold_report_sha256": cold_sha256,
        "cold_validation": str(cold_validation_path.resolve()),
        "cold_validation_sha256": validation_sha256,
        "physical_retention_report": str(retention_report_path.resolve()),
        "physical_retention_report_sha256": retention_sha256,
        "trained_checkpoint": cold["trained_checkpoint"],
        "trained_checkpoint_identity": cold_checkpoint_identity,
        "runtime_source_contract": cold["runtime_source_contract"],
        "dataset_contract": cold["dataset_contract"],
        "stage_checkpoint": cold["stage_checkpoint"],
        "g2_report_sha256": cold["g2_report_sha256"],
        "execution_contract_sha256": cold["execution_contract_sha256"],
        "offline_labels_sha256": cold["offline_labels_sha256"],
        "physical_sidecar_manifest_sha256": cold["physical_sidecar_manifest_sha256"],
        "training_contract": cold["training_contract"],
        "trained_picf_source_contract": dict(trained_source),
    }


def _validate_adr172_training_evidence_binding(
    accepted: AcceptedG3MediatorGate,
    evidence: Mapping[str, Any],
) -> None:
    identity = {
        "model_tree_sha256": accepted.checkpoint_model_tree_sha256,
        "model_local_state_sha256_by_rank": list(
            accepted.training_final_model_local_state_sha256_by_rank
        ),
    }
    expected = {
        "trained_checkpoint": str(accepted.checkpoint_path),
        "trained_checkpoint_identity": identity,
        "runtime_source_contract": accepted.report.get("runtime_source_contract"),
        "dataset_contract": accepted.report.get("dataset_contract"),
        "stage_checkpoint": accepted.report.get("stage_checkpoint"),
        "g2_report_sha256": accepted.report.get("g2_report_sha256"),
        "execution_contract_sha256": accepted.report.get("execution_contract_sha256"),
        "offline_labels_sha256": accepted.report.get("offline_labels_sha256"),
        "physical_sidecar_manifest_sha256": accepted.report.get(
            "physical_sidecar_manifest_sha256"
        ),
        "training_contract": accepted.report.get("training_contract"),
        "trained_picf_source_contract": accepted.report.get("trained_picf_source_contract"),
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(f"ADR172 training and gate evidence differ at {field}")


def _validate_adr172_fixed_head_runtime_binding(
    evidence: Mapping[str, Any],
    *,
    args: argparse.Namespace,
    stage_contract: Any,
    runtime_dataset_contract: Mapping[str, Any] | None = None,
) -> None:
    expected = {
        "stage_checkpoint": str(args.stage_checkpoint.resolve()),
        "g2_report_sha256": stage_contract.g2_report_sha256,
        "runtime_source_contract": ltop_stage_runtime_source_contract(stage_contract),
        "execution_contract_sha256": _sha256(args.execution_contract),
        "offline_labels_sha256": _sha256(args.offline_labels),
        "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
    }
    for field, value in expected.items():
        if evidence.get(field) != value:
            raise ValueError(f"ADR172 fixed-head evidence differs from runtime at {field}")
    if runtime_dataset_contract is not None and evidence.get("dataset_contract") != dict(
        runtime_dataset_contract
    ):
        raise ValueError("ADR172 fixed-head evidence belongs to another runtime dataset")
    source = evidence.get("trained_picf_source_contract")
    if not isinstance(source, Mapping):
        raise ValueError("ADR172 fixed-head source provenance is absent")
    files = source.get("critical_file_sha256")
    if not isinstance(files, Mapping):
        raise ValueError("ADR172 fixed-head source file provenance is absent")
    for relative in ADR172_PICF_CRITICAL_SOURCE_FILES:
        path = (_REPOSITORY_ROOT / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"ADR172 fixed-head critical source is absent: {relative}")
        if _file_sha256(path) != files[relative]:
            raise ValueError(f"ADR172 fixed-head critical source changed: {relative}")


def _canonical_sha256(domain: str, value: object) -> str:
    if not isinstance(domain, str) or not domain:
        raise ValueError("LTOP core-pilot digest domain cannot be empty")
    digest = hashlib.sha256()
    encoded_domain = domain.encode("ascii")
    digest.update(len(encoded_domain).to_bytes(8, "big"))
    digest.update(encoded_domain)
    encoded_value = _canonical_json(value).encode("ascii")
    digest.update(len(encoded_value).to_bytes(8, "big"))
    digest.update(encoded_value)
    return digest.hexdigest()


def _tensor_mapping_sha256(values: Mapping[str, Any], *, torch_module: Any) -> str:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("LTOP core-pilot model inputs must be one non-empty mapping")
    digest = hashlib.sha256()
    digest.update(b"picf-next.ltop-core-pilot-model-inputs.v1")
    for name in sorted(values):
        if not isinstance(name, str) or not name:
            raise ValueError("LTOP core-pilot model-input names cannot be empty")
        _update_tensor_digest(
            digest,
            name=name,
            tensor=values[name],
            torch_module=torch_module,
        )
    return digest.hexdigest()


def _executed_control_chain_sha256(
    chunks: Sequence[Any],
    *,
    torch_module: Any,
    domain: str,
) -> str:
    if not chunks:
        raise ValueError("LTOP core-pilot executed-control chain cannot be empty")
    digest = hashlib.sha256()
    digest.update(domain.encode("ascii"))
    for chunk_index, chunk in enumerate(chunks):
        for field in _CONTROL_TENSOR_FIELDS:
            _update_tensor_digest(
                digest,
                name=f"chunk_{chunk_index}.{field}",
                tensor=getattr(chunk, field),
                torch_module=torch_module,
            )
    return digest.hexdigest()


def _batch_input_receipt(batch: Any, *, torch_module: Any) -> dict[str, Any]:
    structural_targets = [asdict(value) for value in batch.structural_target_requests]
    return {
        "schema": CORE_PILOT_INPUT_RECEIPT_SCHEMA,
        "model_input_sha256": _tensor_mapping_sha256(
            batch.model_inputs,
            torch_module=torch_module,
        ),
        "controls_sha256": _executed_control_chain_sha256(
            (batch.controls,),
            torch_module=torch_module,
            domain="picf-next.ltop-core-pilot-controls.v1",
        ),
        "prior_controls_sha256": _executed_control_chain_sha256(
            batch.effective_prior_control_chunks,
            torch_module=torch_module,
            domain="picf-next.ltop-core-pilot-prior-controls.v1",
        ),
        "structural_targets_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-structural-targets.v1",
            structural_targets,
        ),
    }


def _forward_input_receipt(
    input_receipt: Mapping[str, Any],
    *,
    intervention: str,
    action_information_set: str = "factual",
) -> dict[str, str]:
    if intervention not in {"factual", "blocked"}:
        raise ValueError("LTOP core-pilot forward used an unknown action intervention")
    if action_information_set not in {"factual", "mediator-required"}:
        raise ValueError("LTOP core-pilot forward used an unknown action information set")
    shared = {
        "input_receipt": dict(input_receipt),
        "action_information_set": action_information_set,
        "object_read_action_intervention": "typed-treatment-slot",
    }
    executed = dict(shared)
    executed["object_read_action_intervention"] = intervention
    return {
        "normalized_forward_input_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-normalized-forward.v1",
            shared,
        ),
        "forward_input_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-executed-forward.v1",
            executed,
        ),
    }


def _action_information_set_for_step(
    *,
    policy: str,
    optimizer_step: int,
    rank: int,
    factual: Any,
    mediator_required: Any,
) -> Any:
    """Return the typed per-rank information set without another model forward."""

    if policy == "factual-only":
        return factual
    if policy != LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY:
        raise ValueError("unknown LTOP action-information-set policy")
    if rank not in range(LTOP_CORE_PILOT_WORLD_SIZE):
        raise ValueError("LTOP action-information-set schedule received an invalid rank")
    if isinstance(optimizer_step, bool) or not isinstance(optimizer_step, int):
        raise TypeError("LTOP optimizer step must be an integer")
    if optimizer_step < 0:
        raise ValueError("LTOP optimizer step cannot be negative")
    return mediator_required if (optimizer_step + rank) % 2 else factual


def _expected_action_information_set_counts(
    *,
    policy: str,
    load_global_step: int,
    stop_global_step: int,
    rank: int,
) -> dict[str, int]:
    for name, value in (
        ("load_global_step", load_global_step),
        ("stop_global_step", stop_global_step),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"LTOP {name} must be an integer")
        if value < 0:
            raise ValueError(f"LTOP {name} cannot be negative")
    if stop_global_step < load_global_step:
        raise ValueError("LTOP action-information-set interval is reversed")
    if rank not in range(LTOP_CORE_PILOT_WORLD_SIZE):
        raise ValueError("LTOP action-information-set schedule received an invalid rank")

    length = stop_global_step - load_global_step
    counts = {"factual": 0, "mediator-required": 0}
    if policy == "factual-only":
        counts["factual"] = length
        return counts
    if policy != LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY:
        raise ValueError("unknown LTOP action-information-set policy")
    counts["factual"] = length // 2
    counts["mediator-required"] = length // 2
    if length % 2:
        first = _action_information_set_for_step(
            policy=policy,
            optimizer_step=load_global_step,
            rank=rank,
            factual="factual",
            mediator_required="mediator-required",
        )
        counts[first] += 1
    return counts


def _long_action_information_set_schedule_contract() -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema": CORE_LONG_ACTION_INFORMATION_SET_SCHEDULE_SCHEMA,
        "policy": LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
        "formula": {
            "expression": "(optimizer_step + rank) % 2",
            "remainder_to_executed_information_set": {
                "0": "factual",
                "1": "mediator-required",
            },
        },
        "zero_based_domain": {
            "optimizer_step": {
                "start_inclusive": 0,
                "stop_exclusive": LTOP_CORE_LONG_TOTAL_STEPS,
            },
            "rank": {
                "start_inclusive": 0,
                "stop_exclusive": LTOP_CORE_PILOT_WORLD_SIZE,
            },
        },
        "total_steps": LTOP_CORE_LONG_TOTAL_STEPS,
        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
        "per_rank_counts": [
            {
                "rank": rank,
                **_expected_action_information_set_counts(
                    policy=LTOP_CORE_LONG_ACTION_INFORMATION_SET_POLICY,
                    load_global_step=0,
                    stop_global_step=LTOP_CORE_LONG_TOTAL_STEPS,
                    rank=rank,
                ),
            }
            for rank in range(LTOP_CORE_PILOT_WORLD_SIZE)
        ],
        "per_optimizer_step_counts": {
            "factual": 1,
            "mediator-required": 1,
        },
    }
    return {
        **body,
        "canonical_sha256": _canonical_sha256(
            CORE_LONG_ACTION_INFORMATION_SET_SCHEDULE_SCHEMA,
            body,
        ),
    }


def _validate_long_action_information_set_schedule_contract(value: Any) -> dict[str, Any]:
    required = {
        "canonical_sha256",
        "formula",
        "per_optimizer_step_counts",
        "per_rank_counts",
        "policy",
        "schema",
        "total_steps",
        "world_size",
        "zero_based_domain",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("LTOP long action-information-set schedule contract is incomplete")
    body = dict(value)
    canonical_sha256 = body.pop("canonical_sha256")
    _require_sha256("long action-information-set schedule", canonical_sha256)
    if canonical_sha256 != _canonical_sha256(
        CORE_LONG_ACTION_INFORMATION_SET_SCHEDULE_SCHEMA,
        body,
    ):
        raise ValueError("LTOP long action-information-set schedule digest differs")
    if value != _long_action_information_set_schedule_contract():
        raise ValueError("LTOP long action-information-set schedule contract differs")
    return value


def _arm_contract_for_mode(arm: LTOPCorePilotArm, mode: str) -> dict[str, Any]:
    if mode not in LTOP_CORE_PILOT_MODES:
        raise ValueError("unknown LTOP execution mode")
    start_state = (
        "same-accepted-mediator-g3-model-only-checkpoint"
        if mode in {"long", "restart-smoke"}
        else "same-accepted-g2b-model-only-checkpoint"
    )
    contract = matched_arm_contract(arm, start_state=start_state)
    contract.update(
        {
            "training_objective": (
                "released-action-moe+task-free-physical-set+fixed-head-direct-posterior"
            ),
            "object_read_action_intervention": "not-applicable-direct-posterior-route",
            "only_permitted_pair_difference": "none-production-factual-only",
            "fixed_head_objective": _fixed_head_objective_contract(),
        }
    )
    return contract


def _require_accepted_g3_dataset_contract(
    *,
    accepted_dataset_contract: Any,
    runtime_dataset_contract: Any,
) -> dict[str, Any]:
    if not isinstance(accepted_dataset_contract, dict) or not accepted_dataset_contract:
        raise ValueError("accepted G3 dataset contract is absent")
    if not isinstance(runtime_dataset_contract, dict) or not runtime_dataset_contract:
        raise ValueError("runtime dataset contract is absent")
    if accepted_dataset_contract != runtime_dataset_contract:
        raise ValueError("accepted G3 dataset contract differs from runtime")
    return runtime_dataset_contract


def _action_information_set_metric_summary(
    rank_windows: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    fields: Sequence[str],
) -> dict[str, Any]:
    if len(rank_windows) != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("LTOP metric summary requires one window per rank")
    if not fields or any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("LTOP metric summary fields are invalid")
    if len(set(fields)) != len(fields):
        raise ValueError("LTOP metric summary fields are duplicated")

    records_by_information_set: dict[str, list[Mapping[str, Any]]] = {
        "factual": [],
        "mediator-required": [],
    }
    seen_ranks: set[int] = set()
    for rank_window in rank_windows:
        if not isinstance(rank_window, Mapping) or set(rank_window) != {"rank", "steps"}:
            raise ValueError("LTOP metric rank window is malformed")
        rank = rank_window["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in range(LTOP_CORE_PILOT_WORLD_SIZE)
            or rank in seen_ranks
        ):
            raise ValueError("LTOP metric rank window has an invalid or duplicate rank")
        seen_ranks.add(rank)
        steps = rank_window["steps"]
        if not isinstance(steps, Sequence) or isinstance(steps, (str, bytes)):
            raise ValueError("LTOP metric rank window steps are malformed")
        for record in steps:
            if not isinstance(record, Mapping):
                raise ValueError("LTOP metric step record is malformed")
            global_step = record.get("global_step")
            if (
                isinstance(global_step, bool)
                or not isinstance(global_step, int)
                or global_step <= 0
            ):
                raise ValueError("LTOP metric global step is invalid")
            executed = record.get("executed_action_information_set")
            expected = _action_information_set_for_step(
                policy=policy,
                optimizer_step=global_step - 1,
                rank=rank,
                factual="factual",
                mediator_required="mediator-required",
            )
            if executed != expected:
                raise ValueError("LTOP metric executed information set differs from schedule")
            for field in fields:
                if field not in record:
                    raise ValueError(f"LTOP metric step record lacks {field}")
                float(record[field])
            records_by_information_set[executed].append(record)
    if seen_ranks != set(range(LTOP_CORE_PILOT_WORLD_SIZE)):
        raise ValueError("LTOP metric rank set differs")

    return {
        "policy": policy,
        "arms": {
            information_set: {
                "count": len(records),
                "means": {
                    field: (
                        _mean([float(record[field]) for record in records]) if records else None
                    )
                    for field in fields
                },
            }
            for information_set, records in records_by_information_set.items()
        },
    }


def _scientific_boundary_for_mode(mode: str) -> str:
    if mode not in LTOP_CORE_PILOT_MODES:
        raise ValueError("unknown LTOP execution mode")
    if mode in {"smoke", "restart-smoke"}:
        return "Engineering smoke only; no capability claim is permitted."
    if mode == "long":
        return (
            "This report validates one factual direct-posterior training run using the "
            "byte-bound ADR172 fixed-head objective. Deployment benefit still requires "
            "separately registered factual deployment evaluation."
        )
    return (
        "This historical pilot report validates one arm's execution integrity. Mediator "
        "benefit requires the separately registered two-arm typed-intervention paired-curve "
        "comparison; released-LingBot superiority additionally requires LBOT-JOINT calibration."
    )


def _source_identity(
    *,
    source_checkout: Path,
    patch: Path,
    runtime_hotfix: Path | None,
) -> dict[str, Any]:
    tracked_status = _git_output(
        _REPOSITORY_ROOT,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    if tracked_status:
        raise RuntimeError("LTOP core-pilot source checkout contains tracked modifications")
    upstream_status = _git_output(
        source_checkout,
        "status",
        "--porcelain",
        "--untracked-files=no",
    )
    upstream_diff = _git_output(source_checkout, "diff", "--binary", "HEAD", "--")
    identity = {
        "schema": CORE_PILOT_SOURCE_IDENTITY_SCHEMA,
        "picf_commit": _git_output(_REPOSITORY_ROOT, "rev-parse", "HEAD"),
        "picf_tree": _git_output(_REPOSITORY_ROOT, "rev-parse", "HEAD^{tree}"),
        "picf_tracked_worktree_clean": True,
        "runner_sha256": _file_sha256(Path(__file__).resolve()),
        "lingbot_commit": _git_output(source_checkout, "rev-parse", "HEAD"),
        "lingbot_tree": _git_output(source_checkout, "rev-parse", "HEAD^{tree}"),
        "lingbot_tracked_status_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-lingbot-status.v1",
            upstream_status,
        ),
        "lingbot_tracked_diff_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-lingbot-diff.v1",
            upstream_diff,
        ),
        "patch_sha256": _file_sha256(patch.resolve()),
    }
    identity["runtime_hotfix_sha256"] = (
        None if runtime_hotfix is None else _file_sha256(runtime_hotfix.resolve())
    )
    return identity


def _optimizer_option(value: Any, *, torch_module: Any) -> Any:
    if torch_module.is_tensor(value):
        raise TypeError("LTOP core-pilot optimizer options cannot contain tensors")
    if isinstance(value, Mapping):
        return {
            str(name): _optimizer_option(child, torch_module=torch_module)
            for name, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_optimizer_option(child, torch_module=torch_module) for child in value]
    return _canonical(value)


def _optimizer_initialization_receipt(
    *,
    rank: int,
    optimizer: Any,
    policy: Any,
    optimizer_manifest: Any,
    model_local_state_sha256: str,
    rank_rng_sha256: str,
    torch_module: Any,
) -> dict[str, Any]:
    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping) or state:
        raise RuntimeError("LTOP core-pilot optimizer must begin with an empty state")
    names_by_id: dict[int, list[str]] = {}
    for name, parameter in policy.named_parameters(remove_duplicate=False):
        if parameter.requires_grad:
            names_by_id.setdefault(id(parameter), []).append(f"policy.{name}")
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise TypeError("LTOP core-pilot optimizer parameter group is malformed")
        parameter_names: list[str] = []
        for parameter in parameters:
            aliases = names_by_id.get(id(parameter))
            if not aliases:
                raise RuntimeError("LTOP core-pilot optimizer owns an unnamed parameter")
            parameter_names.append(sorted(set(aliases))[0])
        options = {
            str(name): _optimizer_option(value, torch_module=torch_module)
            for name, value in sorted(group.items())
            if name != "params"
        }
        groups.append(
            {
                "group_index": group_index,
                "parameter_names": parameter_names,
                "options": options,
            }
        )
    return {
        "schema": CORE_PILOT_OPTIMIZER_INITIALIZATION_SCHEMA,
        "rank": rank,
        "fresh_zero_state": True,
        "state_entry_count": 0,
        "parameter_groups_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-optimizer-groups.v1",
            groups,
        ),
        "optimizer_state_sha256": _canonical_sha256(
            "picf-next.ltop-core-pilot-empty-optimizer-state.v1",
            {"state_entry_count": 0},
        ),
        "parameter_manifest_sha256": optimizer_manifest.schema_sha256,
        "model_local_state_sha256": model_local_state_sha256,
        "rank_rng_state_sha256": rank_rng_sha256,
    }


def _runtime_environment_contract(
    *,
    torch_module: Any,
    device: Any,
    fsdp2_placement: str,
) -> dict[str, Any]:
    nccl_namespace = getattr(torch_module.cuda, "nccl", None)
    nccl_version = None
    if nccl_namespace is not None and callable(getattr(nccl_namespace, "version", None)):
        raw_nccl_version = nccl_namespace.version()
        nccl_version = (
            list(raw_nccl_version)
            if isinstance(raw_nccl_version, (list, tuple))
            else raw_nccl_version
        )
    return {
        "schema": CORE_PILOT_RUNTIME_ENVIRONMENT_SCHEMA,
        "python_version": platform.python_version(),
        "torch_version": str(torch_module.__version__),
        "cuda_runtime_version": str(torch_module.version.cuda),
        "cudnn_version": torch_module.backends.cudnn.version(),
        "nccl_version": nccl_version,
        "gpu_name": torch_module.cuda.get_device_name(device),
        "gpu_compute_capability": list(torch_module.cuda.get_device_capability(device)),
        "deterministic_algorithms": bool(torch_module.are_deterministic_algorithms_enabled()),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "matmul_allow_tf32": bool(torch_module.backends.cuda.matmul.allow_tf32),
        "fsdp2_placement": fsdp2_placement,
        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_sha256(name: str, value: Any) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"LTOP {name} is not one lowercase SHA-256 digest")
    return value


def _detached_prior_boundary(step: int) -> bytes:
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise ValueError("LTOP detached-prior boundary requires a positive step")
    return _canonical_json(
        {
            "schema": "picf-next.ltop-core-pilot-detached-prior.v1",
            "persistent_lane_bank": False,
            "prior_source": "frozen-batch-effective-prior-control-chunks",
            "next_optimizer_step": step,
        }
    ).encode("ascii")


def _validate_checkpoint_boundary(value: Any) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_BOUNDARY_KEYS:
        raise ValueError("LTOP checkpoint boundary hashes are incomplete")
    for name, digest in value.items():
        _require_sha256(f"checkpoint {name}", digest)
    return value


def _checkpoint_provenance_sha256(value: Any) -> str:
    if not isinstance(value, dict) or value.get("schema") != (
        CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA
    ):
        raise ValueError("LTOP checkpoint provenance contract is malformed")
    return _canonical_sha256(CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA, value)


def _validate_checkpoint_provenance_rank_receipts(
    value: Any,
    *,
    expected_provenance_sha256: str,
) -> list[dict[str, Any]]:
    _require_sha256("checkpoint provenance", expected_provenance_sha256)
    if not isinstance(value, list) or len(value) != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("LTOP checkpoint provenance rank receipts are incomplete")
    seen_ranks: set[int] = set()
    receipts: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != {
            "checkpoint_provenance_sha256",
            "rank",
        }:
            raise ValueError("LTOP checkpoint provenance rank receipt is malformed")
        rank = item["rank"]
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank not in range(LTOP_CORE_PILOT_WORLD_SIZE)
            or rank in seen_ranks
        ):
            raise ValueError("LTOP checkpoint provenance rank receipt has an invalid rank")
        digest = item["checkpoint_provenance_sha256"]
        _require_sha256("rank checkpoint provenance", digest)
        if digest != expected_provenance_sha256:
            raise RuntimeError("LTOP checkpoint provenance differs across ranks")
        seen_ranks.add(rank)
        receipts.append(dict(item))
    if seen_ranks != set(range(LTOP_CORE_PILOT_WORLD_SIZE)):
        raise ValueError("LTOP checkpoint provenance receipt rank set differs")
    return sorted(receipts, key=lambda item: item["rank"])


def _all_gather_checkpoint_provenance_rank_receipts(
    *,
    distributed: Any,
    rank: int,
    checkpoint_provenance_sha256: str,
) -> list[dict[str, Any]]:
    local = {
        "rank": rank,
        "checkpoint_provenance_sha256": checkpoint_provenance_sha256,
    }
    gathered: list[Any] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
    distributed.all_gather_object(gathered, local)
    return _validate_checkpoint_provenance_rank_receipts(
        gathered,
        expected_provenance_sha256=checkpoint_provenance_sha256,
    )


def _validate_checkpoint_manifest(
    value: Any,
    *,
    expected_global_step: int,
    expected_arm: str,
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    required = {
        "arm",
        "global_step",
        "next_optimizer_step",
        "provenance",
        "provenance_rank_receipts",
        "provenance_sha256",
        "rank_boundaries",
        "schema",
        "status",
        "world_size",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise ValueError("LTOP checkpoint manifest is incomplete")
    if value["schema"] != CORE_PILOT_CHECKPOINT_SCHEMA or value["status"] != "PASS":
        raise ValueError("LTOP checkpoint manifest is not a passing registered checkpoint")
    if value["global_step"] != expected_global_step or (
        value["next_optimizer_step"] != expected_global_step
    ):
        raise ValueError("LTOP checkpoint manifest optimizer boundary differs")
    if value["arm"] != expected_arm or value["world_size"] != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("LTOP checkpoint manifest execution topology differs")
    if value["provenance"] != expected_provenance:
        raise ValueError("LTOP checkpoint manifest provenance differs")
    expected_provenance_sha256 = _checkpoint_provenance_sha256(expected_provenance)
    if value["provenance_sha256"] != expected_provenance_sha256:
        raise ValueError("LTOP checkpoint manifest provenance digest differs")
    _validate_checkpoint_provenance_rank_receipts(
        value["provenance_rank_receipts"],
        expected_provenance_sha256=expected_provenance_sha256,
    )
    boundaries = value["rank_boundaries"]
    if not isinstance(boundaries, list) or len(boundaries) != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("LTOP checkpoint manifest rank boundaries are incomplete")
    seen_ranks: set[int] = set()
    for item in boundaries:
        if not isinstance(item, dict) or set(item) != {"boundary", "rank"}:
            raise ValueError("LTOP checkpoint manifest rank boundary is malformed")
        rank = item["rank"]
        if isinstance(rank, bool) or not isinstance(rank, int) or rank in seen_ranks:
            raise ValueError("LTOP checkpoint manifest rank boundary is duplicated")
        seen_ranks.add(rank)
        _validate_checkpoint_boundary(item["boundary"])
    if seen_ranks != set(range(LTOP_CORE_PILOT_WORLD_SIZE)):
        raise ValueError("LTOP checkpoint manifest rank set differs")
    return value


def _validate_resume_extra(
    value: Any,
    *,
    expected_global_step: int,
    expected_source_digest: str,
    expected_provenance: dict[str, Any],
    rank: int,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CHECKPOINT_EXTRA_KEYS:
        raise ValueError("LTOP checkpoint extra state is incomplete")
    if value["schema"] != CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA:
        raise ValueError("LTOP checkpoint extra-state schema differs")
    if value["global_step"] != expected_global_step or (
        value["next_optimizer_step"] != expected_global_step
    ):
        raise ValueError("LTOP checkpoint optimizer boundary differs")
    if value["rank"] != rank or value["world_size"] != LTOP_CORE_PILOT_WORLD_SIZE:
        raise ValueError("LTOP checkpoint topology differs")
    if value["source_digest"] != expected_source_digest:
        raise ValueError("LTOP checkpoint frozen stream boundary differs")
    if value["provenance"] != expected_provenance:
        raise ValueError("LTOP checkpoint extra-state provenance differs")
    provenance_sha256 = _checkpoint_provenance_sha256(expected_provenance)
    if value["provenance_sha256"] != provenance_sha256:
        raise ValueError("LTOP checkpoint extra-state provenance digest differs")
    if not isinstance(value["lane_snapshot"], bytes) or not value["lane_snapshot"]:
        raise ValueError("LTOP checkpoint detached-prior boundary is absent")
    expected_lane_snapshot = _detached_prior_boundary(expected_global_step)
    if value["lane_snapshot"] != expected_lane_snapshot:
        raise ValueError("LTOP checkpoint detached-prior boundary differs")
    if not isinstance(value["rank_rng_state"], dict):
        raise ValueError("LTOP checkpoint rank RNG state is absent")
    for name in ("optimizer_state_entries", "optimizer_local_moment_elements"):
        field = value[name]
        if isinstance(field, bool) or not isinstance(field, int) or field <= 0:
            raise ValueError("LTOP checkpoint optimizer summary is invalid")
    _validate_checkpoint_boundary(value["boundary_sha256"])
    return value


def _prepare_rank_metric_journal(
    path: Path,
    *,
    phase: str,
    load_global_step: int,
    expected_boundary_source_digest: str | None = None,
) -> Any:
    """Reuse the native-full append/truncate protocol with contiguous-step validation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if phase == "fresh":
        return path.open("x", encoding="ascii", buffering=1)
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"resume metric journal is absent: {path}")
    if not expected_boundary_source_digest:
        raise ValueError("resume metric journal requires its frozen stream boundary")
    retained: list[str] = []
    retained_boundary: dict[str, Any] | None = None
    previous_step = 0
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        try:
            payload = json.loads(line)
            step = payload["global_step"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"metric journal line {line_number} is malformed") from error
        if isinstance(step, bool) or not isinstance(step, int) or step != previous_step + 1:
            raise ValueError("metric journal steps must be contiguous and strictly increasing")
        previous_step = step
        if step <= load_global_step:
            retained.append(line)
            retained_boundary = payload
    if retained_boundary is None or retained_boundary["global_step"] != load_global_step:
        raise ValueError("metric journal does not reach the restored checkpoint boundary")
    if retained_boundary.get("source_digest") != expected_boundary_source_digest:
        raise ValueError("metric journal frozen stream boundary differs from the checkpoint")
    staging = path.with_name(f".{path.name}.resume-{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    _write_text_durable(staging, "\n".join(retained) + "\n")
    os.replace(staging, path)
    _fsync_directory(path.parent)
    return path.open("a", encoding="ascii", buffering=1)


def _prune_resume_publications(run_dir: Path, *, load_global_step: int) -> None:
    metrics = run_dir / "metrics"
    if metrics.is_dir():
        for path in sorted(metrics.glob("steps_*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError("resume metric publication must be a regular file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            end_step = payload.get("end_step")
            if isinstance(end_step, bool) or not isinstance(end_step, int):
                raise ValueError("resume metric publication has no integer end step")
            if end_step > load_global_step:
                path.unlink()
        _fsync_directory(metrics)
    diagnostics = run_dir / "diagnostics"
    if diagnostics.is_dir():
        for path in sorted(diagnostics.glob("step_*")):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("resume diagnostic publication must be one real directory")
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError as error:
                raise ValueError("resume diagnostic step directory is malformed") from error
            if step > load_global_step:
                shutil.rmtree(path)
        _fsync_directory(diagnostics)
    invocations = run_dir / "invocations"
    if invocations.is_dir():
        for path in sorted(invocations.glob("ltop_core_*.json")):
            if path.is_symlink() or not path.is_file():
                raise ValueError("resume invocation report must be one regular file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            stop_step = payload.get("stop_global_step")
            if isinstance(stop_step, bool) or not isinstance(stop_step, int):
                raise ValueError("resume invocation report has no integer stop step")
            if stop_step > load_global_step:
                path.unlink()
        _fsync_directory(invocations)
    terminal_report = run_dir / "ltop_core_pilot_report.json"
    if terminal_report.exists() or terminal_report.is_symlink():
        if terminal_report.is_symlink() or not terminal_report.is_file():
            raise ValueError("resume terminal report must be one regular file")
        payload = json.loads(terminal_report.read_text(encoding="utf-8"))
        stop_step = payload.get("stop_global_step")
        if isinstance(stop_step, bool) or not isinstance(stop_step, int):
            raise ValueError("resume terminal report has no integer stop step")
        if stop_step > load_global_step:
            terminal_report.unlink()
    progress = run_dir / "progress.json"
    if progress.exists() or progress.is_symlink():
        if progress.is_symlink() or not progress.is_file():
            raise ValueError("resume progress report must be one regular file")
        payload = json.loads(progress.read_text(encoding="utf-8"))
        completed_steps = payload.get("completed_steps")
        if isinstance(completed_steps, bool) or not isinstance(completed_steps, int):
            raise ValueError("resume progress report has no integer completed step")
        if completed_steps > load_global_step:
            progress.unlink()
    _fsync_directory(run_dir)


def _checkpoint_tree_size(path: Path) -> int:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("LTOP checkpoint size reference must be one real directory")
    total = 0
    for root, directories, files in os.walk(path, followlinks=False):
        for name in directories:
            if (Path(root) / name).is_symlink():
                raise ValueError("LTOP checkpoint tree cannot contain symbolic links")
        for name in files:
            candidate = Path(root) / name
            if candidate.is_symlink() or not candidate.is_file():
                raise ValueError("LTOP checkpoint tree must contain regular files only")
            total += candidate.stat().st_size
    return total


def _require_rolling_checkpoint_capacity(checkpoint_root: Path) -> dict[str, int | None]:
    free_bytes = require_checkpoint_write_capacity(checkpoint_root)
    complete = [
        path
        for path in checkpoint_root.glob("global_step_*")
        if path.is_dir() and not path.is_symlink()
    ]
    reference_bytes = max((_checkpoint_tree_size(path) for path in complete), default=0)
    required_bytes = max(
        CORE_LONG_MINIMUM_CHECKPOINT_WRITE_BYTES,
        reference_bytes + CORE_LONG_CHECKPOINT_SAFETY_MARGIN_BYTES,
    )
    if free_bytes < required_bytes:
        raise RuntimeError(
            "LTOP rolling checkpoint filesystem has "
            f"{free_bytes / 2**30:.2f} GiB free; "
            f"{required_bytes / 2**30:.2f} GiB is required from the latest measured "
            "checkpoint plus the 16-GiB publication margin"
        )
    return {
        "free_bytes": free_bytes,
        "reference_checkpoint_bytes": reference_bytes or None,
        "required_free_bytes": required_bytes,
    }


def _write_or_validate_cold_resume_receipt(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file():
            raise ValueError("LTOP cold-resume receipt is not one regular file")
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError("LTOP cold-resume receipt conflicts with this verified load")
        return
    write_text_durable_exclusive(
        path,
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )


def _prune_superseded_long_checkpoints(
    checkpoint_root: Path,
    *,
    verified_step: int,
    verification_receipt: Path,
) -> list[dict[str, Any]]:
    """Prune predecessors only after a successor passed a cold-process load."""

    if verification_receipt.is_symlink() or not verification_receipt.is_file():
        raise FileNotFoundError("LTOP cold-resume verification receipt is absent")
    verification = json.loads(verification_receipt.read_text(encoding="utf-8"))
    if (
        not isinstance(verification, dict)
        or verification.get("schema") != CORE_PILOT_COLD_RESUME_SCHEMA
        or verification.get("status") != "PASS"
        or verification.get("global_step") != verified_step
    ):
        raise ValueError("LTOP checkpoint pruning lacks a matching cold-resume PASS")
    successor = checkpoint_root / f"global_step_{verified_step}"
    successor_manifest = successor / "ltop_core_pilot_checkpoint.json"
    if (
        successor.is_symlink()
        or not successor.is_dir()
        or (successor_manifest.is_symlink() or not successor_manifest.is_file())
    ):
        raise FileNotFoundError("LTOP verified successor checkpoint is absent")
    if verification.get("checkpoint_manifest_sha256") != _file_sha256(successor_manifest):
        raise ValueError("LTOP cold-resume receipt belongs to another successor checkpoint")

    receipts: list[dict[str, Any]] = []
    receipt_root = checkpoint_root / "pruned_receipts"
    receipt_root.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[int, Path, Path]] = []
    for candidate in checkpoint_root.glob("global_step_*"):
        if candidate.is_symlink() or not candidate.is_dir():
            raise ValueError(f"unexpected LTOP long checkpoint path: {candidate}")
        try:
            candidate_step = int(candidate.name.removeprefix("global_step_"))
        except ValueError as error:
            raise ValueError(f"invalid LTOP long checkpoint name: {candidate.name}") from error
        if candidate_step < verified_step:
            tombstone = checkpoint_root / (
                f".global_step_{candidate_step}.pruned_by_{verified_step}"
            )
            candidates.append((candidate_step, candidate, tombstone))
    for tombstone in checkpoint_root.glob(f".global_step_*.pruned_by_{verified_step}"):
        if tombstone.is_symlink() or not tombstone.is_dir():
            raise ValueError("LTOP checkpoint prune tombstone is malformed")
        prefix = tombstone.name.removeprefix(".global_step_")
        candidate_text, separator, successor_text = prefix.partition(".pruned_by_")
        if separator != ".pruned_by_" or successor_text != str(verified_step):
            raise ValueError("LTOP checkpoint prune tombstone boundary differs")
        try:
            candidate_step = int(candidate_text)
        except ValueError as error:
            raise ValueError("LTOP checkpoint prune tombstone step is malformed") from error
        if candidate_step >= verified_step:
            raise ValueError("LTOP checkpoint prune tombstone order differs")
        if not any(step == candidate_step for step, _candidate, _tombstone in candidates):
            candidates.append(
                (
                    candidate_step,
                    checkpoint_root / f"global_step_{candidate_step}",
                    tombstone,
                )
            )
    for candidate_step, candidate, tombstone in sorted(
        candidates,
        key=lambda value: value[0],
    ):
        receipt = receipt_root / f"global_step_{candidate_step}.json"
        if candidate.exists() or candidate.is_symlink():
            if candidate.is_symlink() or not candidate.is_dir():
                raise ValueError("LTOP predecessor checkpoint is malformed")
            if tombstone.exists() or tombstone.is_symlink():
                raise FileExistsError("LTOP checkpoint prune tombstone already exists")
            os.replace(candidate, tombstone)
            _fsync_directory(checkpoint_root)
        if not tombstone.exists() and not receipt.exists():
            raise RuntimeError("LTOP predecessor disappeared before its prune receipt")
        if receipt.exists() or receipt.is_symlink():
            if receipt.is_symlink() or not receipt.is_file():
                raise ValueError("LTOP checkpoint prune receipt is malformed")
            payload = json.loads(receipt.read_text(encoding="utf-8"))
            if (
                payload.get("schema") != CORE_LONG_PRUNED_CHECKPOINT_SCHEMA
                or payload.get("status") != "PRUNED_AFTER_COLD_SUCCESSOR_PASS"
                or payload.get("global_step") != candidate_step
                or payload.get("successor_global_step") != verified_step
                or payload.get("successor_verification_receipt_sha256")
                != _file_sha256(verification_receipt)
            ):
                raise ValueError("LTOP checkpoint prune receipt conflicts with recovery")
        else:
            manifest = tombstone / "ltop_core_pilot_checkpoint.json"
            if manifest.is_symlink() or not manifest.is_file():
                raise ValueError(f"superseded checkpoint lacks its manifest: {tombstone}")
            payload = {
                "schema": CORE_LONG_PRUNED_CHECKPOINT_SCHEMA,
                "status": "PRUNED_AFTER_COLD_SUCCESSOR_PASS",
                "global_step": candidate_step,
                "successor_global_step": verified_step,
                "successor_verification_receipt_sha256": _file_sha256(verification_receipt),
                "checkpoint_manifest_sha256": _file_sha256(manifest),
                "checkpoint_path": str(candidate.resolve()),
            }
            write_text_durable_exclusive(
                receipt,
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
        if tombstone.exists():
            shutil.rmtree(tombstone)
            _fsync_directory(checkpoint_root)
        receipts.append(
            {
                "global_step": candidate_step,
                "receipt_path": str(receipt.resolve()),
                "receipt_sha256": _file_sha256(receipt),
            }
        )
    return receipts


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("CUDA allocator pre-bootstrap differs from parsed arguments")
    cadence = _cadence_for_mode(args.mode)
    load_global_step, stop_global_step = _resolve_run_interval(args, cadence)
    arm = LTOPCorePilotArm(args.arm)
    mediator_g3_required = args.mode in {"long", "restart-smoke"}
    execution_arm_contract = _arm_contract_for_mode(arm, args.mode)
    action_information_set_schedule_contract = None
    adr172_evidence = None
    if mediator_g3_required:
        if (
            args.adr172_cold_report is None
            or args.adr172_cold_validation is None
            or args.adr172_physical_retention_report is None
        ):
            raise ValueError("long/restart-smoke requires complete ADR172 fixed-head evidence")
        adr172_evidence = _load_adr172_fixed_head_evidence(
            cold_report_path=args.adr172_cold_report,
            cold_validation_path=args.adr172_cold_validation,
            retention_report_path=args.adr172_physical_retention_report,
        )
    accepted_g3 = (
        _load_accepted_adr172_fixed_head_training(args.g3_report)
        if mediator_g3_required
        else load_accepted_g3_gate(args.g3_report)
    )
    if adr172_evidence is not None:
        _validate_adr172_training_evidence_binding(accepted_g3, adr172_evidence)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = require_persistent_run_root(args.run_dir)
    request = LingBotVLA2LTOPStageRequest(
        source_checkout=args.source_checkout,
        patch=args.patch,
        training_config=args.training_config,
        checkpoint_dir=args.checkpoint_dir,
        processor_dir=args.processor_dir,
        stage_checkpoint=args.stage_checkpoint,
        g2_report=args.g2_report,
        runtime_hotfix=args.runtime_hotfix,
        seed=args.seed,
        maximum_control_tokens=args.maximum_control_tokens,
        fsdp2_placement=args.fsdp2_placement,
    )
    stage_contract = prepare_lingbot_vla2_ltop_stage_transfer(request)
    if adr172_evidence is not None:
        _validate_adr172_fixed_head_runtime_binding(
            adr172_evidence,
            args=args,
            stage_contract=stage_contract,
        )
    if accepted_g3.report.get("stage_checkpoint") != str(args.stage_checkpoint.resolve()):
        raise ValueError("G3 PASS report belongs to another G2b stage checkpoint")
    if accepted_g3.report.get("g2_report_sha256") != stage_contract.g2_report_sha256:
        raise ValueError("G3 PASS report belongs to another G2b report")
    if accepted_g3.report.get("runtime_source_contract") != ltop_stage_runtime_source_contract(
        stage_contract
    ):
        raise ValueError("G3 PASS report belongs to another runtime source contract")
    if accepted_g3.report.get("architecture_identity") != G2_ARCHITECTURE:
        raise ValueError("G3 PASS report belongs to another architecture")
    if accepted_g3.report.get("capacity") != G2_CAPACITY:
        raise ValueError("G3 PASS report changed the accepted object-row capacity")
    if accepted_g3.report.get("task_query_count") != G2_TASK_QUERY_COUNT:
        raise ValueError("G3 PASS report changed the accepted task-query count")
    if accepted_g3.report.get("execution_contract_sha256") != _sha256(args.execution_contract):
        raise ValueError("G3 PASS report belongs to another execution contract")
    if accepted_g3.report.get("offline_labels_sha256") != _sha256(args.offline_labels):
        raise ValueError("G3 PASS report belongs to another offline-label contract")
    if (
        accepted_g3.report.get("physical_sidecar_manifest_sha256")
        != args.physical_sidecar_manifest_sha256
    ):
        raise ValueError("G3 PASS report belongs to another physical sidecar")
    source_identity = _source_identity(
        source_checkout=args.source_checkout.resolve(),
        patch=args.patch.resolve(),
        runtime_hotfix=(None if args.runtime_hotfix is None else args.runtime_hotfix.resolve()),
    )

    with open_lingbot_vla2_ltop_stage_runtime(stage_contract) as runtime:
        torch = runtime.runtime_modules.torch
        dist = runtime.runtime_modules.dist
        rank = runtime.rank
        device = runtime.device
        runtime_environment = _runtime_environment_contract(
            torch_module=torch,
            device=device,
            fsdp2_placement=args.fsdp2_placement,
        )

        import numpy as np
        from lingbotvla.checkpoint import build_checkpointer
        from lingbotvla.data import VLADataCollatorWithPacking
        from lingbotvla.data.vla_data.utils import FeatureTransform
        from lingbotvla.models import build_processor
        from lingbotvla.models.vla.lingbot_vla.moe_load_balance import (
            build_moe_load_balance_hook,
        )
        from lingbotvla.optim import build_muon_optimizer

        from picf_next.data.calvin import (
            CalvinDatasetIndex,
            CalvinPhysicalTransitionDataset,
            CalvinStatefulTransitionDataset,
        )
        from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
        from picf_next.data.calvin_physical_supervision_sidecar import (
            CalvinPhysicalSupervisionSidecar,
        )
        from picf_next.data.dataset_manifest import (
            load_dataset_file_manifest,
            validate_dataset_runtime_binding,
        )
        from picf_next.eval.calvin_task_relevance import calvin_exact_task_loss_identities
        from picf_next.lingbot_native.calvin import (
            CollatedNativeCALVINBatch,
            build_native_calvin_physical_episode_domain,
            build_planned_native_calvin_batch,
            collate_native_calvin_training_batch,
            materialize_native_flow_randomness,
        )
        from picf_next.lingbot_native.calvin_entity_set import (
            build_task_independent_calvin_targets,
            physical_frame_predictions_from_relation,
            physical_frame_row_bindings,
        )
        from picf_next.lingbot_native.action_posterior_collector import (
            RegisteredActionPosteriorReceiptCollector,
        )
        from picf_next.lingbot_native.action_posterior_learning import (
            action_posterior_target_mass_loss,
        )
        from picf_next.lingbot_native.entity_set_objective import (
            eligible_physical_tracks,
            match_physical_frame_entities,
            physical_frame_set_loss,
        )
        from picf_next.lingbot_native.entity_evaluation_plan import (
            EntityEvaluationPlan,
            build_entity_evaluation_plan,
        )
        from picf_next.lingbot_native.host import (
            LingBotNativePriorStepper,
            ObjectReadActionIntervention,
            native_context_from_prior_trace,
        )
        from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
        from picf_next.lingbot_native.representation_split import RepresentationTrialSplit
        from picf_next.lingbot_native.state import AddressedLayerwisePriorTrace
        from picf_next.lingbot_native.task_address_graph import (
            TaskAddressActionInformationSet,
        )
        from picf_next.lingbot_native.task_address_target import (
            resolve_task_address_target_row,
        )
        from picf_next.lingbot_native.torch_dcp_compat import (
            install_torch_2_8_sparse_optimizer_state_backport,
        )
        from picf_next.lingbot_native.training import (
            audit_native_optimizer_coverage,
            run_native_policy_training_forward,
        )
        from picf_next.lingbot_native.visual_audit import (
            render_task_independent_entity_visuals,
        )
        from picf_next.training.control import load_frozen_episode_stream_plan

        install_torch_2_8_sparse_optimizer_state_backport(torch)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        torch.cuda.reset_peak_memory_stats(device)

        run_lease = acquire_distributed_run_lease(args.run_dir, rank=rank, distributed=dist)
        manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
        norm_stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
        validate_lingbot_calvin_norm_stats(norm_stats)
        norm_source = norm_stats["source"]
        if (
            norm_source["dataset_id"] != manifest.dataset_id
            or norm_source["dataset_revision"] != manifest.dataset_revision
            or norm_source["dataset_tree_sha256"] != manifest.tree_sha256
            or manifest.split_name != args.dataset_split.name
        ):
            raise ValueError("LTOP core-pilot CALVIN manifest and normalization differ")
        dataset_contract = validate_dataset_runtime_binding(
            manifest,
            args.dataset_split,
            dataset_id=norm_source["dataset_id"],
            dataset_revision=norm_source["dataset_revision"],
            split_name=args.dataset_split.name,
        )
        _require_accepted_g3_dataset_contract(
            accepted_dataset_contract=accepted_g3.report.get("dataset_contract"),
            runtime_dataset_contract=dataset_contract,
        )
        if adr172_evidence is not None:
            _validate_adr172_fixed_head_runtime_binding(
                adr172_evidence,
                args=args,
                stage_contract=stage_contract,
                runtime_dataset_contract=dataset_contract,
            )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        evaluation_dataset = CalvinStatefulTransitionDataset(
            index,
            action_horizon=runtime.model_config.chunk_size,
        )
        dataset = CalvinPhysicalTransitionDataset(
            index,
            action_horizon=runtime.model_config.chunk_size,
        )
        sidecar = CalvinPhysicalSupervisionSidecar(
            args.physical_sidecar_root,
            index,
            manifest_path=args.physical_sidecar_manifest,
            expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        )
        expected_file_hashes = {
            args.stream_plan: args.stream_plan_sha256,
            args.representation_split: args.representation_split_sha256,
            args.evaluation_plan: args.evaluation_plan_sha256,
        }
        for path, expected_sha256 in expected_file_hashes.items():
            if _file_sha256(path) != expected_sha256:
                raise ValueError(f"LTOP core-pilot contract file SHA-256 differs: {path}")
        representation_split = RepresentationTrialSplit.load(args.representation_split)
        plan = load_frozen_episode_stream_plan(
            args.stream_plan,
            episodes=build_native_calvin_physical_episode_domain(
                dataset,
                excluded_source_episode_indices=(
                    representation_split.stream_domain_excluded_source_episode_indices
                ),
            ),
        )
        evaluation_plan = EntityEvaluationPlan.load(args.evaluation_plan)
        if plan.total_steps < cadence.total_steps:
            raise ValueError("LTOP core-pilot stream is shorter than the requested prefix")
        if args.mode in {"pilot", "long"} and plan.total_steps != cadence.total_steps:
            raise ValueError("LTOP production stream must match its registered optimizer budget")
        if plan.global_batch_size != LTOP_CORE_PILOT_WORLD_SIZE:
            raise ValueError("LTOP core-pilot stream has the wrong global batch")
        if representation_split.stream_plan_sha256 != plan.plan_sha256:
            raise ValueError("LTOP core-pilot split and stream plan differ")
        if representation_split.training_steps != plan.total_steps:
            raise ValueError("LTOP core-pilot split does not cover the complete stream")
        if evaluation_plan.representation_split_sha256 != representation_split.artifact_sha256:
            raise ValueError("LTOP core-pilot evaluation plan belongs to another split")
        if (
            build_entity_evaluation_plan(
                representation_split,
                evaluation_dataset,
                world_size=LTOP_CORE_PILOT_WORLD_SIZE,
            )
            != evaluation_plan
        ):
            raise ValueError("LTOP core-pilot evaluation plan is not reproducible")
        evaluation_sources = {item.source_episode_index for item in evaluation_plan.items}
        if evaluation_sources.intersection(representation_split.training_source_episode_indices):
            raise ValueError("LTOP core-pilot evaluation overlaps a training source episode")
        _merged, data_mapping = _resolve_training_config(
            runtime.training_config,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=cadence.total_steps,
        )
        processor = build_processor(str(args.processor_dir.resolve()))
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            official_lingbot_data_config(data_mapping),
            runtime.model_config,
            processor,
            chunk_size=runtime.model_config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )

        policy = runtime.policy
        graph = runtime.graph
        initial_model_local_state_sha256 = runtime.actual_model_local_state_sha256
        initialization_checkpoint: dict[str, Any] = (
            {
                "kind": "accepted-mediator-g3-model-only",
                "path": str(accepted_g3.checkpoint_path),
                "accepted_g3_report": str(accepted_g3.path),
                "accepted_g3_report_sha256": accepted_g3.file_sha256,
                "expected_rank_local_model_sha256": list(
                    accepted_g3.training_final_model_local_state_sha256_by_rank
                ),
                "checkpoint_model_tree_sha256": (accepted_g3.checkpoint_model_tree_sha256),
                "strict_model_only_restore": True,
            }
            if mediator_g3_required
            else {
                "kind": "accepted-g2b-stage",
                "path": str(args.stage_checkpoint.resolve()),
                "accepted_g3_report": str(accepted_g3.path),
                "accepted_g3_report_sha256": accepted_g3.file_sha256,
            }
        )
        if mediator_g3_required and args.phase == "fresh":
            restored = {"model": policy}
            initialization_checkpointer = build_checkpointer(
                dist_backend="fsdp2",
                ckpt_manager="dcp",
            )
            _distributed_rank_local_call(
                action=lambda: initialization_checkpointer.load(
                    str(accepted_g3.checkpoint_path),
                    restored,
                    allow_partial_load=False,
                ),
                phase="ltop-core-mediator-g3-initialization-cold-load",
                rank=rank,
                dist_module=dist,
            )
            if set(restored) != {"model"} or restored["model"] is not policy:
                raise RuntimeError(
                    "LTOP mediator initialization changed the model-only restore boundary"
                )
            initial_model_local_state_sha256 = _distributed_rank_local_call(
                action=lambda: _model_local_state_digest(policy, torch),
                phase="ltop-core-mediator-g3-initialization-model-digest",
                rank=rank,
                dist_module=dist,
            )
            expected_model_sha256 = accepted_g3.training_final_model_local_state_sha256_by_rank[
                rank
            ]
            if initial_model_local_state_sha256 != expected_model_sha256:
                raise RuntimeError(
                    "LTOP mediator initialization model digest differs from cold acceptance"
                )
        policy.requires_grad_(True)
        policy.train()
        graph.train()
        require_lingbot_exact_resume_contract(runtime.optimizer_contract)
        optimizer = build_lingbot_official_optimizer(
            policy,
            runtime.optimizer_contract,
            build_muon_optimizer=build_muon_optimizer,
            build_moe_load_balance_hook=build_moe_load_balance_hook,
        )
        optimizer_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        optimizer_initialization = (
            _optimizer_initialization_receipt(
                rank=rank,
                optimizer=optimizer,
                policy=policy,
                optimizer_manifest=optimizer_manifest,
                model_local_state_sha256=initial_model_local_state_sha256,
                rank_rng_sha256=_rank_rng_digest(_capture_rank_rng(torch, np, device=device)),
                torch_module=torch,
            )
            if args.phase == "fresh"
            else None
        )
        checkpoint_provenance = {
            "schema": CORE_PILOT_CHECKPOINT_PROVENANCE_SCHEMA,
            "architecture_identity": G2_ARCHITECTURE,
            "arm_contract": execution_arm_contract,
            "action_information_set_policy": args.action_information_set_policy,
            "action_information_set_schedule_contract": None,
            "fixed_head_objective": _fixed_head_objective_contract(),
            "adr172_fixed_head_evidence": adr172_evidence,
            "cadence": asdict(cadence),
            "capacity": args.capacity,
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "cuda_allocator": args.cuda_allocator,
            "dataset_contract": dataset_contract,
            "evaluation_plan_sha256": evaluation_plan.artifact_sha256,
            "execution_contract_sha256": _sha256(args.execution_contract),
            "g2_report_sha256": stage_contract.g2_report_sha256,
            "g3_report_sha256": accepted_g3.file_sha256,
            "initialization_checkpoint": initialization_checkpoint,
            "maximum_control_tokens": args.maximum_control_tokens,
            "maximum_grad_norm": float(args.maximum_grad_norm).hex(),
            "mode": args.mode,
            "norm_stats_sha256": _file_sha256(args.norm_stats),
            "offline_labels_sha256": _sha256(args.offline_labels),
            "optimizer_contract": runtime.optimizer_contract.metadata,
            "optimizer_parameter_manifest_sha256": optimizer_manifest.schema_sha256,
            "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
            "processor_dir": str(args.processor_dir.resolve()),
            "representation_split_sha256": representation_split.artifact_sha256,
            "robot_config_sha256": _file_sha256(args.robot_config),
            "seed": args.seed,
            "source_identity": source_identity,
            "fsdp2_placement": args.fsdp2_placement,
            "stage_checkpoint_inventory": stage_contract.checkpoint_inventory,
            "stage_model_identity": stage_contract.model_identity,
            "stage_checkpoint": str(args.stage_checkpoint.resolve()),
            "stream_plan_sha256": plan.plan_sha256,
            "task_query_count": args.task_query_count,
            "training_config_sha256": _file_sha256(args.training_config),
            "training_loss_weights": {
                "official": float(args.official_loss_weight).hex(),
                "physical_set": float(args.physical_set_weight).hex(),
                "action_posterior": float(ADR174_FIXED_HEAD_WEIGHT).hex(),
            },
            "runtime_environment_contract": runtime_environment,
        }
        checkpoint_provenance_sha256 = _checkpoint_provenance_sha256(checkpoint_provenance)
        checkpoint_provenance_rank_receipts = _all_gather_checkpoint_provenance_rank_receipts(
            distributed=dist,
            rank=rank,
            checkpoint_provenance_sha256=checkpoint_provenance_sha256,
        )
        loaded_boundary: dict[str, str] | None = None
        resume_runtime_rng_verified = False
        resume_receipt_report: dict[str, Any] | None = None
        pruned_checkpoints: list[dict[str, Any]] = []
        journal_expected_source_digest: str | None = None
        if args.phase == "resume":
            checkpoint_dir = args.run_dir / "checkpoints" / f"global_step_{load_global_step}"
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                raise FileNotFoundError(checkpoint_dir)
            checkpoint_manifest_path = checkpoint_dir / "ltop_core_pilot_checkpoint.json"
            if checkpoint_manifest_path.is_symlink() or not checkpoint_manifest_path.is_file():
                raise ValueError("LTOP resume checkpoint lacks its immutable manifest")
            checkpoint_manifest = _validate_checkpoint_manifest(
                json.loads(checkpoint_manifest_path.read_text(encoding="utf-8")),
                expected_global_step=load_global_step,
                expected_arm=arm.value,
                expected_provenance=checkpoint_provenance,
            )
            checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
            state = {"model": policy, "optimizer": optimizer, "extra_state": {}}
            checkpointer.load(str(checkpoint_dir), state)
            prior_planned = build_planned_native_calvin_batch(
                plan,
                dataset,
                optimizer_step=load_global_step - 1,
                rank=rank,
                world_size=LTOP_CORE_PILOT_WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                device=device,
                dtype=torch.bfloat16,
                maximum_control_tokens=args.maximum_control_tokens,
            )
            resume_extra = _validate_resume_extra(
                state["extra_state"],
                expected_global_step=load_global_step,
                expected_source_digest=prior_planned.source_digest,
                expected_provenance=checkpoint_provenance,
                rank=rank,
            )
            journal_expected_source_digest = resume_extra["source_digest"]
            optimizer_state = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=load_global_step,
            )
            if any(
                optimizer_state[name] != resume_extra[name]
                for name in ("optimizer_state_entries", "optimizer_local_moment_elements")
            ):
                raise RuntimeError("LTOP restored optimizer summary differs")
            loaded_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=resume_extra["lane_snapshot"],
                rank_rng_state=resume_extra["rank_rng_state"],
                torch_module=torch,
            )
            if loaded_boundary != resume_extra["boundary_sha256"]:
                raise RuntimeError("LTOP restored checkpoint boundary differs")
            manifest_boundary = next(
                item["boundary"]
                for item in checkpoint_manifest["rank_boundaries"]
                if item["rank"] == rank
            )
            if loaded_boundary != manifest_boundary:
                raise RuntimeError("LTOP DCP state differs from its immutable manifest")
            _restore_rank_rng(
                resume_extra["rank_rng_state"],
                torch,
                np,
                device=device,
            )
            resume_runtime_rng_verified = (
                _rank_rng_digest(_capture_rank_rng(torch, np, device=device))
                == loaded_boundary["rank_rng_state_sha256"]
            )
            if not resume_runtime_rng_verified:
                raise RuntimeError("LTOP runtime RNG restore differs from its checkpoint")

        journal_dir = args.run_dir / "metrics" / "rank_journal"
        journal_path = journal_dir / f"rank_{rank}.jsonl"
        journal_result: list[Any] = [None]
        try:
            metric_handle = _prepare_rank_metric_journal(
                journal_path,
                phase=args.phase,
                load_global_step=load_global_step,
                expected_boundary_source_digest=journal_expected_source_digest,
            )
            journal_result[0] = {"rank": rank, "status": "PASS"}
        except BaseException as error:
            metric_handle = None
            journal_result[0] = {
                "rank": rank,
                "status": "FAIL",
                "error": f"{type(error).__name__}: {error}",
            }
        gathered_journal_results: list[Any] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
        dist.all_gather_object(gathered_journal_results, journal_result[0])
        journal_failures = [
            value for value in gathered_journal_results if value.get("status") != "PASS"
        ]
        if journal_failures:
            if metric_handle is not None:
                metric_handle.close()
            raise RuntimeError(f"LTOP metric journal recovery failed: {journal_failures}")
        if metric_handle is None:
            raise RuntimeError("LTOP metric journal handle is absent after recovery")

        if args.phase == "resume":
            gathered_loads: list[Any] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
            dist.all_gather_object(
                gathered_loads,
                {
                    "rank": rank,
                    "boundary": loaded_boundary,
                    "optimizer_state": optimizer_state,
                    "runtime_rng_verified": resume_runtime_rng_verified,
                },
            )
            resume_publication: list[Any] = [None]
            if rank == 0:
                try:
                    _prune_resume_publications(
                        args.run_dir,
                        load_global_step=load_global_step,
                    )
                    checkpoint_manifest_path = (
                        args.run_dir
                        / "checkpoints"
                        / f"global_step_{load_global_step}"
                        / "ltop_core_pilot_checkpoint.json"
                    )
                    receipt_payload = {
                        "schema": CORE_PILOT_COLD_RESUME_SCHEMA,
                        "status": "PASS",
                        "global_step": load_global_step,
                        "checkpoint_path": str(checkpoint_manifest_path.parent.resolve()),
                        "checkpoint_manifest_sha256": _file_sha256(checkpoint_manifest_path),
                        "provenance_sha256": checkpoint_provenance_sha256,
                        "rank_loads": sorted(
                            gathered_loads,
                            key=lambda value: value["rank"],
                        ),
                    }
                    receipt_path = (
                        args.run_dir
                        / "checkpoints"
                        / "cold_resume_receipts"
                        / f"global_step_{load_global_step}.json"
                    )
                    receipt_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_or_validate_cold_resume_receipt(
                        receipt_path,
                        receipt_payload,
                    )
                    _fsync_directory(receipt_path.parent)
                    pruned_checkpoints = (
                        _prune_superseded_long_checkpoints(
                            args.run_dir / "checkpoints",
                            verified_step=load_global_step,
                            verification_receipt=receipt_path,
                        )
                        if args.mode in {"long", "restart-smoke"}
                        else []
                    )
                    resume_publication[0] = {
                        "path": str(receipt_path.resolve()),
                        "file_sha256": _file_sha256(receipt_path),
                        "checkpoint_manifest_sha256": receipt_payload["checkpoint_manifest_sha256"],
                        "pruned_checkpoints": pruned_checkpoints,
                    }
                except BaseException as error:
                    resume_publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(resume_publication, src=0)
            if not isinstance(resume_publication[0], dict) or ("error" in resume_publication[0]):
                metric_handle.close()
                raise RuntimeError(f"LTOP cold-resume publication failed: {resume_publication[0]}")
            resume_receipt_report = resume_publication[0]
            dist.barrier()

        def collate(candidate: Any) -> CollatedNativeCALVINBatch:
            value = collate_native_calvin_training_batch(
                candidate.training,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=candidate.augmentation_seeds,
                source_digest=candidate.source_digest,
            )
            value = CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    value.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=value.controls,
                routing=value.routing,
                source_digest=value.source_digest,
                structural_target_requests=value.structural_target_requests,
                modalities=None,
                prior_control_chunks=value.prior_control_chunks,
            )
            return materialize_native_flow_randomness(value, candidate)

        prior_stepper = LingBotNativePriorStepper(policy, graph)

        def build_prior(batch: CollatedNativeCALVINBatch) -> AddressedLayerwisePriorTrace:
            episode_ids = _episode_ids(
                batch.routing.episode_keys,
                torch_module=torch,
                device=device,
            )
            prior: Any | None = None
            valid = torch.zeros(batch.routing.batch_size, dtype=torch.bool, device=device)
            with torch.no_grad():
                for controls in batch.effective_prior_control_chunks:
                    prior = prior_stepper(
                        prior,
                        controls,
                        previous_memory_valid=valid,
                        episode_ids=episode_ids,
                    )
                    valid = torch.ones_like(valid)
            if not isinstance(prior, AddressedLayerwisePriorTrace):
                raise RuntimeError("LTOP core-pilot prior rollout omitted addressed rows")
            return prior

        if graph.config.num_layers != 36:
            raise ValueError("ADR174 fixed-head objective is registered for 36 LingBot layers")
        registered_layer_indices = ADR174_FIXED_HEAD_LAYERS
        fixed_head_indices = torch.tensor(
            ADR174_FIXED_HEAD_INDICES,
            dtype=torch.long,
            device=device,
        )

        def training_forward(
            batch: CollatedNativeCALVINBatch,
            prior: AddressedLayerwisePriorTrace,
        ) -> tuple[Any, tuple[Any, ...]]:
            context = native_context_from_prior_trace(
                controls=batch.controls,
                prior_trace=prior,
                modalities=None,
                posterior_adoption_route=torch.ones(
                    batch.routing.batch_size,
                    dtype=torch.bool,
                    device=device,
                ),
            )
            collector = RegisteredActionPosteriorReceiptCollector(
                registered_layer_indices=registered_layer_indices
            )
            result = run_native_policy_training_forward(
                policy,
                model_inputs=batch.model_inputs,
                context=context,
                action_attention_callback=collector,
            )
            receipts = collector.finalize()
            if any(not bool(receipt.posterior_attention.requires_grad) for receipt in receipts):
                raise RuntimeError("ADR174 fixed-head receipt left the training graph")
            if any(
                receipt.layer_index != expected
                for receipt, expected in zip(receipts, registered_layer_indices, strict=True)
            ):
                raise RuntimeError("ADR174 fixed-head receipt layer identity differs")
            if any(
                receipt.posterior_attention.shape[1] <= max(ADR174_FIXED_HEAD_INDICES)
                for receipt in receipts
            ):
                raise RuntimeError("ADR174 native attention omits a registered object head")
            return result, receipts

        vision_config = runtime.model_config.vision_config
        patch_size = int(vision_config.patch_size)
        merge_size = int(vision_config.spatial_merge_size)

        def physical_supervision(
            *,
            context: Any,
            batch: CollatedNativeCALVINBatch,
            target_identity: str | None,
            canonical_assignment: Any | None = None,
            canonical_identity_keys: Any | None = None,
        ) -> dict[str, Any]:
            relation = context.relation_output
            if not isinstance(relation, PhysicalRelationOutput):
                raise RuntimeError("LTOP core-pilot observation omitted physical relations")
            target_bundle = build_task_independent_calvin_targets(
                requests_by_time=(batch.structural_target_requests,),
                model_inputs_by_time=(batch.model_inputs,),
                relations=(relation,),
                physical_sidecar=sidecar,
                capacity=args.capacity,
                patch_size=patch_size,
                merge_size=merge_size,
            )[0]
            if (
                canonical_identity_keys is not None
                and target_bundle.identity_keys_by_batch != canonical_identity_keys
            ):
                raise RuntimeError(
                    "LTOP core-pilot crossed prompts changed the physical identity axis"
                )
            predictions = physical_frame_predictions_from_relation(relation)
            matched_assignment = match_physical_frame_entities(predictions, target_bundle.targets)
            assignment = (
                matched_assignment if canonical_assignment is None else canonical_assignment
            )
            set_loss = physical_frame_set_loss(
                predictions,
                target_bundle.targets,
                assignment=assignment,
            )
            bindings = physical_frame_row_bindings(
                target_bundle,
                assignment,
                capacity=args.capacity,
            )[0]
            matched_bindings = physical_frame_row_bindings(
                target_bundle,
                matched_assignment,
                capacity=args.capacity,
            )[0]
            eligible_tracks = eligible_physical_tracks(target_bundle.targets, 0)
            target_row, target_row_reason = resolve_task_address_target_row(
                target_identity=target_identity,
                identity_keys=target_bundle.identity_keys_by_batch[0],
                eligible_track_indices=eligible_tracks.detach().cpu().tolist(),
                bindings=bindings,
                allow_unobservable=True,
            )
            return {
                "target_bundle": target_bundle,
                "set_loss": set_loss,
                "assignment": assignment,
                "matched_assignment": matched_assignment,
                "bindings": bindings,
                "matched_bindings": matched_bindings,
                "identity_keys_by_batch": target_bundle.identity_keys_by_batch,
                "target_row": target_row,
                "target_row_reason": target_row_reason,
            }

        def task_target_identity(batch: CollatedNativeCALVINBatch) -> str | None:
            if batch.routing.batch_size != 1 or len(batch.structural_target_requests) != 1:
                raise RuntimeError("LTOP core pilot requires one local CALVIN sample per rank")
            identities = calvin_exact_task_loss_identities(
                batch.structural_target_requests[0].task_key
            )
            if identities is None:
                return None
            if len(identities) != 1:
                raise RuntimeError("LTOP core pilot exact task target is not unique")
            return identities[0]

        def objective_for_batch(
            batch: CollatedNativeCALVINBatch,
        ) -> dict[str, Any]:
            prior = build_prior(batch)
            result, posterior_receipts = training_forward(batch, prior)
            target_identity = task_target_identity(batch)
            physical = physical_supervision(
                context=result.context,
                batch=batch,
                target_identity=target_identity,
            )
            target_row_weights = torch.zeros(
                (batch.routing.batch_size, args.capacity),
                dtype=posterior_receipts[0].posterior_attention.dtype,
                device=device,
            )
            target_valid = torch.zeros(
                batch.routing.batch_size,
                dtype=torch.bool,
                device=device,
            )
            if physical["target_row"] is not None:
                target_row_weights[:, physical["target_row"]] = 1.0
                target_valid[:] = True
            posterior_targets = tuple(
                action_posterior_target_mass_loss(
                    receipt.posterior_attention,
                    target_row_weights=target_row_weights,
                    target_valid=target_valid,
                    head_indices=fixed_head_indices,
                )
                for receipt in posterior_receipts
            )
            action_posterior_loss = torch.stack(
                tuple(value.loss for value in posterior_targets)
            ).mean()
            total_loss = (
                args.official_loss_weight * result.official_total_loss
                + args.physical_set_weight * physical["set_loss"].total
                + ADR174_FIXED_HEAD_WEIGHT * action_posterior_loss
            )
            return {
                "prior": prior,
                "result": result,
                "target_identity": target_identity,
                "physical": physical,
                "posterior_receipts": posterior_receipts,
                "posterior_targets": posterior_targets,
                "target_valid": target_valid,
                "action_posterior_loss": action_posterior_loss,
                "total_loss": total_loss,
            }

        def publish_training_visual(
            *,
            step: int,
            planned: Any,
            batch: CollatedNativeCALVINBatch,
            objective: dict[str, Any],
        ) -> dict[str, Any]:
            """Render the already-executed training forward without another FSDP lifecycle."""

            visual_artifacts = render_task_independent_entity_visuals(
                output_root=args.run_dir,
                global_step=step,
                input_weight_global_step=step - 1,
                weight_boundary="pre_update_training_forward",
                rank=rank,
                host_items=planned.training.host_items,
                model_inputs=batch.model_inputs,
                relation=objective["result"].context.relation_output,
                target_bundle=objective["physical"]["target_bundle"],
                set_loss=objective["physical"]["set_loss"],
                sample_keys=batch.routing.sample_keys,
                merge_size=merge_size,
            )
            report = {
                "schema": CORE_PILOT_DIAGNOSTIC_SCHEMA,
                "rank": rank,
                "global_step": step,
                "weight_boundary": "pre_update_training_forward",
                "extra_model_forward": False,
                "source_disjoint": False,
                "sample_keys": list(batch.routing.sample_keys),
                "source_digest": batch.source_digest,
                "executed_action_information_set": "factual",
                "target_identity": objective["target_identity"],
                "target_row": objective["physical"]["target_row"],
                "action_posterior_supervision_reason": (
                    objective["physical"]["target_row_reason"]
                ),
                "official_total_loss": float(
                    objective["result"].official_total_loss.detach().float().item()
                ),
                "official_action_loss": float(
                    objective["result"].official_action_loss.detach().float().item()
                ),
                "physical_set_loss": float(
                    objective["physical"]["set_loss"].total.detach().float().item()
                ),
                "action_posterior_loss": float(
                    objective["action_posterior_loss"].detach().float().item()
                ),
                "fixed_head_objective": _fixed_head_objective_contract(),
                "action_posterior_receipts": [
                    {
                        "layer_index": receipt.layer_index,
                        "head_indices": list(ADR174_FIXED_HEAD_INDICES),
                        "target_mass_mean": float(
                            target.target_mass.detach().float().mean().item()
                        ),
                        "total_posterior_mass_mean": float(
                            target.total_posterior_mass.detach().float().mean().item()
                        ),
                    }
                    for receipt, target in zip(
                        objective["posterior_receipts"],
                        objective["posterior_targets"],
                        strict=True,
                    )
                ],
                "visual_artifacts": visual_artifacts,
            }
            destination = args.run_dir / "diagnostics" / f"step_{step:08d}" / f"rank_{rank}.json"
            write_text_durable_exclusive(
                destination,
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            return report

        def publish_metrics_window(step: int, local_window: list[dict[str, Any]]) -> dict[str, Any]:
            if len(local_window) != cadence.metrics_every:
                raise RuntimeError("LTOP core-pilot metric window is incomplete")
            gathered_windows: list[Any] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
            dist.all_gather_object(
                gathered_windows,
                {"rank": rank, "steps": tuple(local_window)},
            )
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    ordered = sorted(gathered_windows, key=lambda value: value["rank"])
                    expected_steps = list(range(step - cadence.metrics_every + 1, step + 1))
                    for rank_window in ordered:
                        measured = [value["global_step"] for value in rank_window["steps"]]
                        if measured != expected_steps:
                            raise RuntimeError("LTOP core-pilot metric steps are not contiguous")
                    fields = (
                        "total_loss",
                        "action_loss",
                        "moe_regularizer",
                        "physical_set_loss",
                        "action_posterior_loss",
                        "step_time_s",
                    )
                    action_information_set_summary = _action_information_set_metric_summary(
                        ordered,
                        policy=args.action_information_set_policy,
                        fields=fields,
                    )
                    payload = {
                        "schema": CORE_PILOT_METRICS_SCHEMA,
                        "arm": arm.value,
                        "start_step": expected_steps[0],
                        "end_step": expected_steps[-1],
                        "sample_count": sum(
                            len(value["sample_keys"])
                            for rank_window in ordered
                            for value in rank_window["steps"]
                        ),
                        "means": {
                            field: _mean(
                                [
                                    float(value[field])
                                    for rank_window in ordered
                                    for value in rank_window["steps"]
                                ]
                            )
                            for field in fields
                        },
                        "action_information_set_summary": action_information_set_summary,
                        "rank_windows": ordered,
                    }
                    destination = (
                        args.run_dir / "metrics" / f"steps_{expected_steps[0]:08d}_{step:08d}.json"
                    )
                    write_text_durable_exclusive(
                        destination,
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    )
                    publication[0] = {
                        "path": str(destination),
                        "file_sha256": _file_sha256(destination),
                        "start_step": expected_steps[0],
                        "end_step": step,
                        "means": payload["means"],
                        "action_information_set_summary": action_information_set_summary,
                    }
                except BaseException as error:
                    publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(f"LTOP core-pilot metric publication failed: {publication[0]}")
            dist.barrier()
            return publication[0]

        def publish_checkpoint(step: int, *, source_digest: str) -> dict[str, Any]:
            if not cadence.checkpoint_due(step):
                raise ValueError("LTOP core-pilot checkpoint is outside its registered boundary")
            save_provenance_rank_receipts = _all_gather_checkpoint_provenance_rank_receipts(
                distributed=dist,
                rank=rank,
                checkpoint_provenance_sha256=checkpoint_provenance_sha256,
            )
            if save_provenance_rank_receipts != checkpoint_provenance_rank_receipts:
                raise RuntimeError("LTOP checkpoint provenance changed after initialization")
            checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
            rng = _capture_rank_rng(torch, np, device=device)
            lane_snapshot = _detached_prior_boundary(step)
            boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=rng,
                torch_module=torch,
            )
            optimizer_state = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=step,
            )
            extra = {
                "schema": CORE_PILOT_CHECKPOINT_EXTRA_SCHEMA,
                "rank": rank,
                "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
                "global_step": step,
                "next_optimizer_step": step,
                "source_digest": source_digest,
                "provenance": checkpoint_provenance,
                "provenance_sha256": checkpoint_provenance_sha256,
                "rank_rng_state": rng,
                "lane_snapshot": lane_snapshot,
                "boundary_sha256": boundary,
                **optimizer_state,
            }
            gathered_boundaries: list[Any] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
            dist.all_gather_object(
                gathered_boundaries,
                {"rank": rank, "boundary": boundary},
            )
            checkpoint_root = args.run_dir / "checkpoints"
            output = checkpoint_root / f"global_step_{step}"
            staging = checkpoint_root / f".global_step_{step}.incomplete"
            preflight: list[str | None] = [None]
            capacity_report: dict[str, int | None] | None = None
            if rank == 0:
                try:
                    checkpoint_root.mkdir(parents=True, exist_ok=True)
                    if output.exists() or output.is_symlink():
                        raise FileExistsError(output)
                    if staging.is_symlink():
                        raise ValueError("checkpoint staging path cannot be a symbolic link")
                    if staging.exists():
                        if not staging.is_dir():
                            raise ValueError("checkpoint staging path is not a directory")
                        shutil.rmtree(staging)
                        _fsync_directory(checkpoint_root)
                    capacity_report = _require_rolling_checkpoint_capacity(checkpoint_root)
                except BaseException as error:
                    preflight[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(preflight, src=0)
            if preflight[0] is not None:
                raise RuntimeError(f"LTOP core-pilot checkpoint preflight failed: {preflight[0]}")
            checkpointer.save(
                str(staging),
                {"model": policy, "optimizer": optimizer, "extra_state": extra},
                global_steps=None,
            )
            post_save_rng = _capture_rank_rng(torch, np, device=device)
            if _rank_rng_digest(post_save_rng) != boundary["rank_rng_state_sha256"]:
                raise RuntimeError("LTOP DCP save consumed rank RNG state")
            dist.barrier()
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    payload = {
                        "schema": CORE_PILOT_CHECKPOINT_SCHEMA,
                        "status": "PASS",
                        "global_step": step,
                        "next_optimizer_step": step,
                        "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
                        "arm": arm.value,
                        "provenance": checkpoint_provenance,
                        "provenance_rank_receipts": save_provenance_rank_receipts,
                        "provenance_sha256": checkpoint_provenance_sha256,
                        "rank_boundaries": sorted(
                            gathered_boundaries,
                            key=lambda value: value["rank"],
                        ),
                    }
                    _validate_checkpoint_manifest(
                        payload,
                        expected_global_step=step,
                        expected_arm=arm.value,
                        expected_provenance=checkpoint_provenance,
                    )
                    _write_text_durable(
                        staging / "ltop_core_pilot_checkpoint.json",
                        json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    )
                    _fsync_tree(staging)
                    os.replace(staging, output)
                    _fsync_directory(checkpoint_root)
                    try:
                        _validate_checkpoint_manifest(
                            json.loads(
                                (output / "ltop_core_pilot_checkpoint.json").read_text(
                                    encoding="utf-8"
                                )
                            ),
                            expected_global_step=step,
                            expected_arm=arm.value,
                            expected_provenance=checkpoint_provenance,
                        )
                    except BaseException:
                        os.replace(output, staging)
                        _fsync_directory(checkpoint_root)
                        raise
                    publication[0] = {
                        "path": str(output),
                        "manifest_sha256": _file_sha256(output / "ltop_core_pilot_checkpoint.json"),
                        "provenance_rank_receipts": save_provenance_rank_receipts,
                        "capacity": capacity_report,
                        "pruned_checkpoints": [],
                    }
                except BaseException as error:
                    publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(
                    f"LTOP core-pilot checkpoint publication failed: {publication[0]}"
                )
            dist.barrier()
            return publication[0]

        invocation_report_path = (
            args.run_dir
            / "invocations"
            / (f"ltop_core_{args.phase}_from_{load_global_step:08d}_to_{stop_global_step:08d}.json")
        )
        publish_terminal_report = (
            stop_global_step == cadence.total_steps and stop_global_step > load_global_step
        )
        terminal_report_path = args.run_dir / "ltop_core_pilot_report.json"
        preflight_error: list[str | None] = [None]
        if rank == 0:
            try:
                if invocation_report_path.exists() or invocation_report_path.is_symlink():
                    raise FileExistsError(invocation_report_path)
                if publish_terminal_report and (
                    terminal_report_path.exists() or terminal_report_path.is_symlink()
                ):
                    raise FileExistsError(terminal_report_path)
                if stop_global_step > load_global_step:
                    checkpoint = args.run_dir / "checkpoints" / f"global_step_{stop_global_step}"
                    if checkpoint.exists() or checkpoint.is_symlink():
                        raise FileExistsError("LTOP output checkpoint already exists")
                    staging_checkpoint = (
                        args.run_dir / "checkpoints" / f".global_step_{stop_global_step}.incomplete"
                    )
                    if staging_checkpoint.is_symlink():
                        raise ValueError("LTOP output checkpoint staging is a symbolic link")
                    if staging_checkpoint.exists():
                        if not staging_checkpoint.is_dir():
                            raise ValueError("LTOP output checkpoint staging is malformed")
                        shutil.rmtree(staging_checkpoint)
                        _fsync_directory(staging_checkpoint.parent)
                    _require_rolling_checkpoint_capacity(args.run_dir / "checkpoints")
            except BaseException as error:
                preflight_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(preflight_error, src=0)
        if preflight_error[0] is not None:
            raise RuntimeError(f"LTOP core-pilot output preflight failed: {preflight_error[0]}")

        metric_window: list[dict[str, Any]] = []
        metric_reports: list[dict[str, Any]] = []
        action_loss_first_window: list[float] = []
        action_loss_last_window: list[float] = []
        action_information_set_counts = {
            value.value: 0 for value in TaskAddressActionInformationSet
        }
        diagnostic_index: list[dict[str, Any]] = []
        all_gradients_finite = True
        checkpoint_report: dict[str, Any] | None = (
            {
                "path": str(
                    (args.run_dir / "checkpoints" / f"global_step_{load_global_step}").resolve()
                ),
                "manifest_sha256": resume_receipt_report["checkpoint_manifest_sha256"],
                "cold_resume": resume_receipt_report,
                "verification_only": True,
            }
            if stop_global_step == load_global_step and resume_receipt_report is not None
            else None
        )
        train_started = time.perf_counter()

        for optimizer_step in range(load_global_step, stop_global_step):
            step = optimizer_step + 1
            planned = build_planned_native_calvin_batch(
                plan,
                dataset,
                optimizer_step=optimizer_step,
                rank=rank,
                world_size=LTOP_CORE_PILOT_WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                device=device,
                dtype=torch.bfloat16,
                maximum_control_tokens=args.maximum_control_tokens,
            )
            batch = collate(planned)
            input_receipt = _batch_input_receipt(batch, torch_module=torch)
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_started = time.perf_counter()
            action_information_set = TaskAddressActionInformationSet.FACTUAL
            objective = objective_for_batch(batch)
            executed_intervention = objective["result"].context.object_read_action_intervention
            if not isinstance(executed_intervention, ObjectReadActionIntervention):
                raise RuntimeError("LTOP core-pilot forward omitted its typed intervention")
            expected_intervention = (
                ObjectReadActionIntervention.FACTUAL
                if arm is LTOPCorePilotArm.FACTUAL
                else ObjectReadActionIntervention.BLOCKED
            )
            if executed_intervention is not expected_intervention:
                raise RuntimeError("LTOP core-pilot executed the wrong intervention arm")
            executed_information_sets = tuple(objective["result"].context.action_information_sets)
            expected_information_sets = (action_information_set,) * batch.routing.batch_size
            if executed_information_sets != expected_information_sets:
                raise RuntimeError("LTOP core-pilot executed the wrong action information set")
            forward_receipt = _forward_input_receipt(
                input_receipt,
                intervention=executed_intervention.value,
                action_information_set=action_information_set.value,
            )
            objective["total_loss"].backward()
            gradient_metrics = _distributed_gradient_metrics(
                policy,
                (
                    ("native_graph", "picf_native_graph"),
                    ("task_query", "task_query_embeddings"),
                    ("shared_host", "qwenvl_with_expert.qwen"),
                    ("action_output", "action_out_proj"),
                ),
                device=device,
                dist=dist,
                torch_module=torch,
            )
            finite = bool(gradient_metrics["all_finite"])
            all_gradients_finite &= finite
            if not finite:
                raise FloatingPointError("LTOP core pilot produced a non-finite gradient")
            clipped = clip_lingbot_distributed_l2_grad_norm_(
                tuple(policy.parameters()),
                args.maximum_grad_norm,
                device=device,
                dist_module=dist,
                torch_module=torch,
                error_if_nonfinite=True,
            )
            gradient_metrics["preclip_global_norm"] = clipped
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            step_record = {
                "global_step": step,
                "sample_keys": list(batch.routing.sample_keys),
                "lane_ids": list(batch.routing.lane_ids),
                "frame_indices": list(batch.routing.frame_indices),
                "reset": list(batch.routing.reset),
                "source_digest": batch.source_digest,
                "augmentation_seeds": list(planned.augmentation_seeds),
                "flow_noise_seeds": list(planned.flow_noise_seeds),
                "flow_timestep_seeds": list(planned.flow_timestep_seeds),
                "model_input_sha256": input_receipt["model_input_sha256"],
                "controls_sha256": input_receipt["controls_sha256"],
                "prior_controls_sha256": input_receipt["prior_controls_sha256"],
                "structural_targets_sha256": input_receipt["structural_targets_sha256"],
                **forward_receipt,
                "executed_object_read_action_intervention": executed_intervention.value,
                "executed_action_information_set": action_information_set.value,
                "total_loss": float(objective["total_loss"].detach().float().item()),
                "action_loss": float(
                    objective["result"].official_action_loss.detach().float().item()
                ),
                "moe_regularizer": float(
                    objective["result"].official_moe_regularizer.detach().float().item()
                ),
                "physical_set_loss": float(
                    objective["physical"]["set_loss"].total.detach().float().item()
                ),
                "action_posterior_loss": float(
                    objective["action_posterior_loss"].detach().float().item()
                ),
                "target_identity": objective["target_identity"],
                "target_row": objective["physical"]["target_row"],
                "action_posterior_supervision_reason": (
                    objective["physical"]["target_row_reason"]
                ),
                "fixed_head_objective_sha256": _canonical_sha256(
                    ADR174_FIXED_HEAD_OBJECTIVE_SCHEMA,
                    _fixed_head_objective_contract(),
                ),
                "action_posterior_objective_schema": ADR174_FIXED_HEAD_OBJECTIVE_SCHEMA,
                "action_posterior_loss_weight": ADR174_FIXED_HEAD_WEIGHT,
                "action_posterior_receipts": [
                    {
                        "layer_index": receipt.layer_index,
                        "head_indices": list(ADR174_FIXED_HEAD_INDICES),
                        "target_mass_mean": float(
                            target.target_mass.detach().float().mean().item()
                        ),
                        "total_posterior_mass_mean": float(
                            target.total_posterior_mass.detach().float().mean().item()
                        ),
                    }
                    for receipt, target in zip(
                        objective["posterior_receipts"],
                        objective["posterior_targets"],
                        strict=True,
                    )
                ],
                "gradient_metrics": gradient_metrics,
                "step_time_s": time.perf_counter() - step_started,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            }
            metric_handle.write(_canonical_json(step_record) + "\n")
            action_information_set_counts[action_information_set.value] += 1
            metric_window.append(step_record)
            if len(action_loss_first_window) < cadence.metrics_every:
                action_loss_first_window.append(step_record["action_loss"])
            action_loss_last_window.append(step_record["action_loss"])
            if len(action_loss_last_window) > cadence.metrics_every:
                del action_loss_last_window[0]

            if cadence.metrics_due(step):
                metric_handle.flush()
                os.fsync(metric_handle.fileno())
                metric_reports.append(publish_metrics_window(step, metric_window))
                metric_window.clear()
            if cadence.diagnostics_due(step):
                local_diagnostic = publish_training_visual(
                    step=step,
                    planned=planned,
                    batch=batch,
                    objective=objective,
                )
                gathered_diagnostics: list[Any] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
                dist.all_gather_object(gathered_diagnostics, local_diagnostic)
                if rank == 0:
                    diagnostic_index.append(
                        {
                            "global_step": step,
                            "ranks": sorted(
                                gathered_diagnostics,
                                key=lambda value: value["rank"],
                            ),
                        }
                    )
            if cadence.checkpoint_due(step):
                checkpoint_report = publish_checkpoint(
                    step,
                    source_digest=batch.source_digest,
                )
            if rank == 0 and (step % 8 == 0 or step == stop_global_step):
                elapsed = time.perf_counter() - train_started
                completed_in_invocation = step - load_global_step
                mean_step = elapsed / completed_in_invocation
                _write_json_atomic_replace(
                    args.run_dir / "progress.json",
                    {
                        "schema": CORE_PILOT_PROGRESS_SCHEMA,
                        "arm": arm.value,
                        "completed_steps": step,
                        "total_steps": cadence.total_steps,
                        "invocation_start_step": load_global_step,
                        "invocation_stop_step": stop_global_step,
                        "elapsed_s": elapsed,
                        "mean_elapsed_per_completed_step_s": mean_step,
                        "estimated_remaining_s": mean_step * (cadence.total_steps - step),
                        "updated_unix_s": time.time(),
                    },
                )

        metric_handle.flush()
        os.fsync(metric_handle.fileno())
        metric_handle.close()
        _fsync_directory(journal_dir)
        with journal_path.open("r", encoding="ascii") as journal_stream:
            journal_record_count = sum(1 for line in journal_stream if line.rstrip("\n"))
        if journal_record_count != stop_global_step:
            raise RuntimeError("LTOP core-pilot rank journal has the wrong record count")
        journal_receipt = {
            "schema": CORE_PILOT_JOURNAL_RECEIPT_SCHEMA,
            "rank": rank,
            "path": str(journal_path.resolve()),
            "file_sha256": _file_sha256(journal_path),
            "record_count": journal_record_count,
        }
        if metric_window:
            raise RuntimeError("LTOP core-pilot ended with a partial metric window")
        expected_information_set_counts = _expected_action_information_set_counts(
            policy=args.action_information_set_policy,
            load_global_step=load_global_step,
            stop_global_step=stop_global_step,
            rank=rank,
        )
        if action_information_set_counts != expected_information_set_counts:
            raise RuntimeError(
                "LTOP action-information-set counts differ from the registered schedule"
            )
        if checkpoint_report is None:
            raise RuntimeError(
                "LTOP core-pilot did not publish or cold-verify its terminal checkpoint"
            )

        torch.cuda.synchronize(device)
        duration = time.perf_counter() - train_started
        final_optimizer_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        if final_optimizer_manifest != optimizer_manifest:
            raise RuntimeError("LTOP core-pilot optimizer ownership changed during execution")
        executed_steps = stop_global_step - load_global_step
        rank_report = {
            "rank": rank,
            "metric_reports": metric_reports,
            "diagnostics": diagnostic_index if rank == 0 else [],
            "all_gradients_finite": all_gradients_finite,
            "action_loss_first_window_mean": (
                _mean(action_loss_first_window) if action_loss_first_window else None
            ),
            "action_loss_last_window_mean": (
                _mean(action_loss_last_window) if action_loss_last_window else None
            ),
            "action_information_set_counts": action_information_set_counts,
            "optimizer_parameter_manifest": asdict(optimizer_manifest),
            "optimizer_initialization": optimizer_initialization,
            "loaded_boundary_sha256": loaded_boundary,
            "resume_runtime_rng_verified": resume_runtime_rng_verified,
            "cold_resume_receipt": resume_receipt_report,
            "stage_restore": runtime.rank_report(),
            "journal": journal_receipt,
            "checkpoint": checkpoint_report,
            "timings": {
                "train_checkpoint_diagnostics_and_terminal_eval_s": duration,
                "mean_wall_s_per_optimizer_step": (
                    duration / executed_steps if executed_steps else None
                ),
            },
            "cuda_memory_bytes": {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            },
        }
        gathered: list[dict[str, Any] | None] = [None] * LTOP_CORE_PILOT_WORLD_SIZE
        dist.all_gather_object(gathered, rank_report)
        publication: list[Any] = [None]
        if rank == 0:
            try:
                rank_reports = sorted(
                    (value for value in gathered if value is not None),
                    key=lambda value: value["rank"],
                )
                failures: list[str] = []
                if len(rank_reports) != LTOP_CORE_PILOT_WORLD_SIZE:
                    failures.append("one or more distributed rank reports are absent")
                if any(not value["all_gradients_finite"] for value in rank_reports):
                    failures.append("one or more ranks produced non-finite gradients")
                expected_metric_windows = executed_steps // cadence.metrics_every
                if any(
                    len(value["metric_reports"]) != expected_metric_windows
                    for value in rank_reports
                ):
                    failures.append("one or more ranks omitted an invocation metric window")
                report = {
                    "schema": LTOP_CORE_PILOT_SCHEMA,
                    "status": "PASS" if not failures else "FAIL",
                    "failures": failures,
                    "mode": args.mode,
                    "phase": args.phase,
                    "arm": arm.value,
                    "arm_contract": execution_arm_contract,
                    "architecture_identity": G2_ARCHITECTURE,
                    "source_identity": source_identity,
                    "runtime_environment_contract": runtime_environment,
                    "world_size": LTOP_CORE_PILOT_WORLD_SIZE,
                    "steps": cadence.total_steps,
                    "load_global_step": load_global_step,
                    "stop_global_step": stop_global_step,
                    "executed_optimizer_steps": executed_steps,
                    "cadence": asdict(cadence),
                    "seed": args.seed,
                    "capacity": args.capacity,
                    "task_query_count": args.task_query_count,
                    "stage_checkpoint": str(args.stage_checkpoint.resolve()),
                    "g2_report_sha256": stage_contract.g2_report_sha256,
                    "g3_report_sha256": accepted_g3.file_sha256,
                    "initialization_checkpoint": initialization_checkpoint,
                    "dataset_contract": dataset_contract,
                    "stream_plan_sha256": plan.plan_sha256,
                    "representation_split_sha256": representation_split.artifact_sha256,
                    "evaluation_plan_sha256": evaluation_plan.artifact_sha256,
                    "execution_contract_sha256": _sha256(args.execution_contract),
                    "offline_labels_sha256": _sha256(args.offline_labels),
                    "physical_sidecar_manifest_sha256": (args.physical_sidecar_manifest_sha256),
                    "checkpoint_provenance": checkpoint_provenance,
                    "checkpoint_provenance_sha256": checkpoint_provenance_sha256,
                    "checkpoint_provenance_rank_receipts": (checkpoint_provenance_rank_receipts),
                    "cold_resume_receipt": resume_receipt_report,
                    "pruned_checkpoints": pruned_checkpoints,
                    "action_inference_contract": {
                        "surface": "separate-fresh-process-evaluator",
                        "executed_in_training_process": False,
                        "reason": "FSDP train/diagnostic lifecycle isolation",
                    },
                    "training_contract": {
                        "optimizer": runtime.optimizer_contract.metadata,
                        "fresh_optimizer_after_strict_model_only_restore": (args.phase == "fresh"),
                        "full_dcp_model_optimizer_extra_restore": args.phase == "resume",
                        "deploy_time_module_added": False,
                        "action_information_set_policy": (args.action_information_set_policy),
                        "action_information_set_schedule_contract": (
                            action_information_set_schedule_contract
                        ),
                        "fixed_head_objective": _fixed_head_objective_contract(),
                        "adr172_fixed_head_evidence": adr172_evidence,
                        "loss_weights": {
                            "official": args.official_loss_weight,
                            "physical_set": args.physical_set_weight,
                            "action_posterior": ADR174_FIXED_HEAD_WEIGHT,
                        },
                    },
                    "checkpoint": checkpoint_report,
                    "scientific_boundary": _scientific_boundary_for_mode(args.mode),
                    "rank_reports": rank_reports,
                }
                invocation_report_path.parent.mkdir(parents=True, exist_ok=True)
                write_text_durable_exclusive(
                    invocation_report_path,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                report_paths = [str(invocation_report_path.resolve())]
                if publish_terminal_report:
                    write_text_durable_exclusive(
                        terminal_report_path,
                        json.dumps(report, indent=2, sort_keys=True) + "\n",
                    )
                    report_paths.append(str(terminal_report_path.resolve()))
                publication[0] = {
                    "status": report["status"],
                    "failures": failures,
                    "paths": report_paths,
                }
            except BaseException as error:
                publication[0] = {"error": f"{type(error).__name__}: {error}"}
        dist.broadcast_object_list(publication, src=0)
        if not isinstance(publication[0], dict) or "error" in publication[0]:
            raise RuntimeError(f"LTOP core-pilot publication failed: {publication[0]}")
        if publication[0]["status"] != "PASS":
            raise RuntimeError(f"LTOP core pilot failed: {publication[0]['failures']}")
        if run_lease is not None:
            run_lease.close()


if __name__ == "__main__":
    main()

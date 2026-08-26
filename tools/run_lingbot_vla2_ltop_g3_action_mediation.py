#!/usr/bin/env python3
# ruff: noqa: E402, I001
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the bounded two-GPU LTOP G3 production action-mediation gate.

G3 restores the accepted G2b model-only checkpoint, creates a fresh released
LingBot optimizer, and trains the existing shared host/native graph/action
surface.  It adds no model, head, selector, scorer, or lifecycle rule.

Evaluation is label closed: every production ``sample_actions`` receipt is
sealed under fixed flow noise before loss-side physical identities are opened.
The scorer then compares target-row removal with the crossed-prompt physical
row and verifies that blocking ``OBJECT_READ -> ACTION`` removes that effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
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

from picf_next.artifact_io import (
    directory_tree_sha256,
    publish_prepared_directory_durable_exclusive,
    write_text_durable_exclusive,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from picf_next.lingbot_native.task_address_graph import (
    TaskAddressActionInformationSet,
)
from picf_next.lingbot_native.task_address_learning import (
    action_consumable_task_address,
    action_consumable_task_address_depth_contract,
)
from picf_next.lingbot_native.task_address_target import (
    resolve_task_address_target_row as _resolve_physical_target_row,
)
from picf_next.lingbot_native.task_action_supervision import (
    TASK_ACTION_SUPERVISION_SCHEMA,
    TaskActionSupervisionScope,
    require_factual_action_supervision,
    task_action_supervision_receipt,
)
from tools.bootstrap_lingbot_vla2_native import CHECKOUT_RELATIVE_PATH, PATCH_RELATIVE_PATH
from tools.lingbot_vla2_ltop_stage_runtime import (
    LingBotVLA2LTOPStageRequest,
    ltop_stage_runtime_source_contract,
    open_lingbot_vla2_ltop_stage_runtime,
    prepare_lingbot_vla2_ltop_stage_transfer,
)
from tools.lingbot_vla2_runtime_helpers import (
    LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
    _resolve_training_config,
    _tensor_sha256,
    build_lingbot_official_optimizer,
    clip_lingbot_distributed_l2_grad_norm_,
    require_lingbot_released_action_sampling_steps,
    select_lingbot_deterministic_moe_backend,
)
from tools.run_lingbot_vla2_ltop_g1 import (
    _RUNTIME_MODEL_FIELDS,
    _tensor_manifest,
    apply_ltop_g1_inference_contract,
)
from tools.run_lingbot_vla2_ltop_g2_core import (
    G2_ARCHITECTURE,
    G2_CAPACITY,
    G2_TASK_QUERY_COUNT,
    G2_WORLD_SIZE,
    _episode_ids,
    _load_contracts,
    _local_representation_contract_items,
    _physical_relation_prompt_drift,
    _prompt_variant,
    _scene_level_robustness,
    _scene_metrics,
    _sha256,
    _validate_representation_execution_provenance,
    _validate_representation_item_source,
)
from tools.run_lingbot_vla2_native_g0 import (
    _distributed_gradient_metrics,
    _distributed_rank_local_call,
    _fsync_tree,
    _model_local_state_digest,
    _move_model_inputs,
)


G3_SCHEMA = "picf-next.ltop-g3-production-action-mediation.v1"
G3_TRAINING_SCHEMA = "picf-next.ltop-g3-training-phase.v1"
G3_EVALUATION_SCHEMA = "picf-next.ltop-g3-evaluation-phase.v1"
G3_RETENTION_SCHEMA = "picf-next.ltop-g3-representation-retention.v1"
G3_MODES = ("smoke", "gate", "mediator-trial")
G3_PHASES = ("combined", "training", "evaluation", "retention")
G3_DEFAULT_STEPS = 128
G3_DEFAULT_EVAL_EVERY = 32
G3_MEDIATOR_TRIAL_STEPS = 256
G3_MEDIATOR_TRIAL_EVAL_EVERY = 32
G3_MEDIATOR_TRIAL_SCENES = 8
G3_PROMPTS_PER_SCENE = 2
G3_MEDIATOR_TRIAL_CYCLE_STEPS = G3_MEDIATOR_TRIAL_SCENES * G3_PROMPTS_PER_SCENE
G3_MEDIATOR_TRIAL_SCHEDULE_SCHEMA = "picf-next.ltop-g3-mediator-required-counterbalance.v1"
G3_SOURCE_ACTION_SCHEDULE_SCHEMA = "picf-next.ltop-g3-source-action-counterbalance.v1"
G3_ARM_JOURNAL_SCHEMA = "picf-next.ltop-g3-action-information-set-step.v1"
G3_EVALUATION_SCENES_PER_PARTITION = 1
G3_TRAINING_CHECKPOINT_SCHEMA = "picf-next.ltop-g3-training-checkpoint.v5"
G3_MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
G3_MODEL_ONLY_CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
G3_PICF_SOURCE_CONTRACT_SCHEMA = "picf-next.g3-picf-source-contract.v1"
G3_PICF_CRITICAL_SOURCE_FILES = (
    "tools/run_lingbot_vla2_ltop_g3_action_mediation.py",
    "src/picf_next/lingbot_native/task_address_learning.py",
    "src/picf_next/lingbot_native/task_action_supervision.py",
)
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _git_output(*arguments: str) -> str:
    completed = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), *arguments),
        capture_output=True,
        check=False,
        text=True,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: "
            f"{completed.stderr[-1_000:]!r}"
        )
    return completed.stdout.strip()


def _git_bytes(*arguments: str) -> bytes:
    completed = subprocess.run(
        ("git", "-C", str(_REPOSITORY_ROOT), *arguments),
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(
            f"git {' '.join(arguments)} failed with exit {completed.returncode}: "
            f"{completed.stderr[-1_000:]!r}"
        )
    return completed.stdout


def _picf_source_contract() -> dict[str, object]:
    if _git_output("rev-parse", "--is-inside-work-tree") != "true":
        raise ValueError("PICF source root is not one Git worktree")
    if Path(_git_output("rev-parse", "--show-toplevel")).resolve() != _REPOSITORY_ROOT:
        raise ValueError("PICF source root differs from the Git worktree root")
    commit = _git_output("rev-parse", "--verify", "HEAD^{commit}")
    tree = _git_output("rev-parse", "--verify", "HEAD^{tree}")
    if not _GIT_OBJECT_PATTERN.fullmatch(commit) or not _GIT_OBJECT_PATTERN.fullmatch(tree):
        raise ValueError("PICF source identity contains a malformed Git object")
    if _git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise ValueError("PICF source worktree is not exactly clean")
    files: dict[str, str] = {}
    for relative in G3_PICF_CRITICAL_SOURCE_FILES:
        path = (_REPOSITORY_ROOT / relative).resolve(strict=True)
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"PICF critical source is absent or indirect: {relative}")
        if path.relative_to(_REPOSITORY_ROOT).as_posix() != relative:
            raise ValueError(f"PICF critical source escaped the repository: {relative}")
        working = path.read_bytes()
        committed = _git_bytes("show", f"{commit}:{relative}")
        if working != committed:
            raise ValueError(f"PICF critical source differs from HEAD: {relative}")
        files[relative] = hashlib.sha256(working).hexdigest()
    return {
        "schema": G3_PICF_SOURCE_CONTRACT_SCHEMA,
        "repository_commit": commit,
        "repository_tree": tree,
        "worktree_clean": True,
        "critical_file_sha256": files,
    }


def _validate_picf_source_contract(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {
        "schema",
        "repository_commit",
        "repository_tree",
        "worktree_clean",
        "critical_file_sha256",
    }:
        raise ValueError("G3 PICF source contract fields differ")
    if value.get("schema") != G3_PICF_SOURCE_CONTRACT_SCHEMA:
        raise ValueError("G3 PICF source contract schema differs")
    for field in ("repository_commit", "repository_tree"):
        item = value.get(field)
        if not isinstance(item, str) or not _GIT_OBJECT_PATTERN.fullmatch(item):
            raise ValueError(f"G3 PICF source contract {field} is malformed")
    if value.get("worktree_clean") is not True:
        raise ValueError("G3 PICF source contract is not clean")
    files = value.get("critical_file_sha256")
    if not isinstance(files, dict) or set(files) != set(G3_PICF_CRITICAL_SOURCE_FILES):
        raise ValueError("G3 PICF critical source set differs")
    if any(
        not isinstance(item, str) or not _SHA256_PATTERN.fullmatch(item) for item in files.values()
    ):
        raise ValueError("G3 PICF critical source digest is malformed")
    return value


def _validate_g3_training_checkpoint_manifest(
    manifest: object,
    *,
    expected_layer_count: int | None = None,
    expected_picf_source_contract: dict[str, object] | None = None,
) -> tuple[list[str], str]:
    """Validate the cheap, content-addressed portion of the cold-load ABI."""

    if not isinstance(manifest, dict):
        raise ValueError("G3 trained checkpoint manifest must be a JSON object")
    expected = {
        "schema": G3_TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": G3_MEDIATOR_TRIAL_STEPS,
        "optimizer_saved": False,
        "format": G3_MODEL_ONLY_CHECKPOINT_FORMAT,
        "world_size": G2_WORLD_SIZE,
        "model_tree_schema": G3_MODEL_TREE_SCHEMA,
        "action_supervision_schema": TASK_ACTION_SUPERVISION_SCHEMA,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"G3 trained checkpoint manifest violates {field}")
    depth_contract = manifest.get("task_address_supervision_depth")
    if not isinstance(depth_contract, dict):
        raise ValueError("G3 checkpoint omits its task-address supervision depth")
    try:
        expected_depth_contract = action_consumable_task_address_depth_contract(
            depth_contract.get("layer_count")
        )
    except (TypeError, ValueError) as error:
        raise ValueError("G3 checkpoint task-address supervision depth is malformed") from error
    if depth_contract != expected_depth_contract:
        raise ValueError("G3 checkpoint task-address supervision depth differs")
    if expected_layer_count is not None and depth_contract["layer_count"] != expected_layer_count:
        raise ValueError("G3 checkpoint task-address depth differs from the loaded host graph")
    source_contract = _validate_picf_source_contract(manifest.get("picf_source_contract"))
    if (
        expected_picf_source_contract is not None
        and source_contract != expected_picf_source_contract
    ):
        raise ValueError("G3 checkpoint PICF source identity differs from the loaded runner")
    model_tree_sha256 = manifest.get("model_tree_sha256")
    if not isinstance(model_tree_sha256, str) or not _SHA256_PATTERN.fullmatch(model_tree_sha256):
        raise ValueError("G3 checkpoint model-tree SHA-256 is malformed")
    schedule_sha256 = manifest.get("action_information_set_schedule_sha256")
    if not isinstance(schedule_sha256, str) or not _SHA256_PATTERN.fullmatch(schedule_sha256):
        raise ValueError("G3 checkpoint action-information schedule SHA-256 is malformed")
    expected_digests = manifest.get("training_final_model_local_state_sha256_by_rank")
    if (
        not isinstance(expected_digests, list)
        or len(expected_digests) != G2_WORLD_SIZE
        or any(
            not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value)
            for value in expected_digests
        )
    ):
        raise ValueError("G3 checkpoint omits its per-rank training terminal digests")
    return list(expected_digests), model_tree_sha256


def _require_canonical_bindings_applied(
    *,
    physical: dict[str, Any],
    canonical_bindings: Any | None,
) -> None:
    """Reject only a mismatch in the assignment actually supplied to the loss."""

    if canonical_bindings is not None and physical["bindings"] != canonical_bindings:
        raise RuntimeError(
            "LTOP G3 crossed prompt failed to apply the canonical physical row gauge"
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
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--offline-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, default=None)
    parser.add_argument("--trained-checkpoint", type=Path, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--journal-dir", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument("--mode", choices=G3_MODES, default="gate")
    parser.add_argument("--phase", choices=G3_PHASES, default="combined")
    parser.add_argument(
        "--evaluation-action-information-set",
        choices=tuple(value.value for value in TaskAddressActionInformationSet),
        default=TaskAddressActionInformationSet.FACTUAL.value,
    )
    parser.add_argument("--steps", type=int, default=G3_DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=G3_DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--capacity", type=int, default=G2_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G2_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--maximum-grad-norm", type=float, default=1.0)
    parser.add_argument("--physical-set-weight", type=float, default=1.0)
    parser.add_argument("--task-address-weight", type=float, default=1.0)
    parser.add_argument("--official-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--evaluation-scenes-per-partition",
        type=int,
        default=G3_EVALUATION_SCENES_PER_PARTITION,
    )
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
    if args.journal_dir is None and args.phase in {"combined", "training"}:
        args.journal_dir = args.output.parent / "rank_journal"
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
        "dataset split": args.dataset_split,
        "dataset manifest": args.dataset_manifest,
        "normalization": args.norm_stats,
        "physical sidecar": args.physical_sidecar_root,
        "physical sidecar manifest": args.physical_sidecar_manifest,
        "execution contract": args.execution_contract,
        "offline labels": args.offline_labels,
    }
    if args.runtime_hotfix is not None:
        required_paths["runtime optimizer hotfix"] = args.runtime_hotfix
    missing = [name for name, path in required_paths.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"LTOP G3 required paths are absent: {missing}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.phase == "training":
        if args.checkpoint_output is None:
            raise ValueError("LTOP G3 training phase requires --checkpoint-output")
        if args.checkpoint_output.exists() or args.checkpoint_output.is_symlink():
            raise FileExistsError(args.checkpoint_output)
        if args.trained_checkpoint is not None:
            raise ValueError("G3 training phase cannot consume --trained-checkpoint")
    elif args.phase in {"evaluation", "retention"}:
        if args.checkpoint_output is not None:
            raise ValueError("G3 cold evaluation phase cannot publish --checkpoint-output")
        if (
            args.trained_checkpoint is None
            or args.trained_checkpoint.is_symlink()
            or not args.trained_checkpoint.is_dir()
        ):
            raise FileNotFoundError("G3 cold evaluation phase requires --trained-checkpoint")
        checkpoint_manifest = args.trained_checkpoint / "ltop_g3_training_checkpoint.json"
        if not checkpoint_manifest.is_file() or checkpoint_manifest.is_symlink():
            raise FileNotFoundError("G3 trained checkpoint manifest is absent")
        model_directory = args.trained_checkpoint / "model"
        if model_directory.is_symlink() or not model_directory.is_dir():
            raise FileNotFoundError("G3 trained checkpoint model directory is absent")
        model_names = {path.name for path in model_directory.iterdir() if path.is_file()}
        if ".metadata" not in model_names or not any(
            name.endswith(".distcp") for name in model_names
        ):
            raise ValueError("G3 trained checkpoint omits its DCP model payload")
        _validate_g3_training_checkpoint_manifest(
            json.loads(checkpoint_manifest.read_text(encoding="ascii"))
        )
    elif args.checkpoint_output is not None or args.trained_checkpoint is not None:
        raise ValueError("staged checkpoint arguments require a staged G3 phase")
    if args.phase in {"evaluation", "retention"}:
        if args.journal_dir is not None:
            raise ValueError("G3 cold evaluation phase cannot publish a training journal")
    elif args.journal_dir is None:
        raise ValueError("G3 training requires one rank-journal directory")
    elif args.journal_dir.exists() or args.journal_dir.is_symlink():
        raise FileExistsError(args.journal_dir)
    if (
        args.phase != "evaluation"
        and args.evaluation_action_information_set != TaskAddressActionInformationSet.FACTUAL.value
    ):
        raise ValueError("evaluation action information-set override belongs only to evaluation")
    if args.progress_output is not None and args.progress_output == args.output:
        raise ValueError("LTOP G3 progress and final report paths must differ")
    integer_fields = (
        "steps",
        "eval_every",
        "progress_every",
        "seed",
        "capacity",
        "task_query_count",
        "maximum_control_tokens",
        "evaluation_scenes_per_partition",
    )
    for name in integer_fields:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"LTOP G3 {name} must be a positive integer")
    if args.capacity != G2_CAPACITY or args.task_query_count != G2_TASK_QUERY_COUNT:
        raise ValueError("LTOP G3 must preserve the accepted G2b graph shape")
    if args.eval_every > args.steps or args.steps % args.eval_every:
        raise ValueError("LTOP G3 eval-every must divide the positive step budget")
    if args.mode == "gate" and (
        args.steps != G3_DEFAULT_STEPS or args.eval_every != G3_DEFAULT_EVAL_EVERY
    ):
        raise ValueError("LTOP G3 gate requires the registered 128/32 schedule")
    if args.mode == "mediator-trial" and (
        args.phase != "training"
        or args.steps != G3_MEDIATOR_TRIAL_STEPS
        or args.eval_every != G3_MEDIATOR_TRIAL_EVAL_EVERY
    ):
        raise ValueError("LTOP G3 mediator trial requires the staged training-only 256/32 schedule")
    for name in (
        "maximum_grad_norm",
        "physical_set_weight",
        "task_address_weight",
        "official_loss_weight",
    ):
        value = getattr(args, name)
        if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"LTOP G3 {name} must be finite and positive")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_g3_mediator_counterbalanced_schedule(
    *,
    scene_prompt_keys: Sequence[tuple[str, Sequence[str]]],
    steps: int,
) -> dict[str, Any]:
    """Build the fixed 8-scene x 2-prompt single-forward crossover schedule."""

    normalized = tuple((scene, tuple(prompts)) for scene, prompts in scene_prompt_keys)
    if len(normalized) != G3_MEDIATOR_TRIAL_SCENES:
        raise ValueError("G3 mediator schedule requires exactly eight validation scenes")
    if len({scene for scene, _prompts in normalized}) != len(normalized):
        raise ValueError("G3 mediator schedule scene keys must be unique")
    if any(
        not isinstance(scene, str)
        or not scene
        or len(prompts) != G3_PROMPTS_PER_SCENE
        or any(not isinstance(prompt, str) or not prompt for prompt in prompts)
        or len(set(prompts)) != len(prompts)
        for scene, prompts in normalized
    ):
        raise ValueError("G3 mediator schedule requires two distinct prompt keys per scene")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or steps % G3_MEDIATOR_TRIAL_CYCLE_STEPS
    ):
        raise ValueError("G3 mediator schedule steps must be a positive multiple of 16")

    entries: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        zero_based = step - 1
        cycle_index, cycle_offset = divmod(
            zero_based,
            G3_MEDIATOR_TRIAL_CYCLE_STEPS,
        )
        scene_index, prompt_index = divmod(cycle_offset, G3_PROMPTS_PER_SCENE)
        required = bool((scene_index + prompt_index + cycle_index) % 2)
        scene_key, prompt_keys = normalized[scene_index]
        entries.append(
            {
                "global_step": step,
                "cycle_index": cycle_index,
                "cycle_offset": cycle_offset,
                "scene_index": scene_index,
                "scene_key": scene_key,
                "prompt_index": prompt_index,
                "prompt_key": prompt_keys[prompt_index],
                "arm": (
                    TaskAddressActionInformationSet.MEDIATOR_REQUIRED.value
                    if required
                    else TaskAddressActionInformationSet.FACTUAL.value
                ),
            }
        )

    arm_counts = {
        arm.value: sum(entry["arm"] == arm.value for entry in entries)
        for arm in TaskAddressActionInformationSet
    }
    cell_arm_counts: dict[str, dict[str, int]] = {}
    for entry in entries:
        cell_key = f"{entry['scene_key']}::{entry['prompt_key']}"
        counts = cell_arm_counts.setdefault(
            cell_key,
            {arm.value: 0 for arm in TaskAddressActionInformationSet},
        )
        counts[entry["arm"]] += 1
    payload: dict[str, Any] = {
        "schema": G3_MEDIATOR_TRIAL_SCHEDULE_SCHEMA,
        "design": "scene-prompt-stratified-two-period-crossover",
        "single_forward_per_optimizer_step": True,
        "assignment": "(scene_index + prompt_index + cycle_index) mod 2",
        "steps": steps,
        "scene_count": len(normalized),
        "prompts_per_scene": G3_PROMPTS_PER_SCENE,
        "cycle_steps": G3_MEDIATOR_TRIAL_CYCLE_STEPS,
        "arm_counts": arm_counts,
        "cell_arm_counts": cell_arm_counts,
        "entries": entries,
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return payload


def build_g3_source_action_counterbalanced_schedule(
    *,
    scene_source_keys: Sequence[tuple[str, str]],
    steps: int,
) -> dict[str, Any]:
    """Counterbalance factual/required arms over immutable source task-action pairs."""

    normalized = tuple((scene, task) for scene, task in scene_source_keys)
    if not normalized:
        raise ValueError("G3 source-action schedule requires at least one validation scene")
    if len({scene for scene, _task in normalized}) != len(normalized):
        raise ValueError("G3 source-action schedule scene keys must be unique")
    if any(
        not isinstance(scene, str) or not scene or not isinstance(task, str) or not task
        for scene, task in normalized
    ):
        raise ValueError("G3 source-action schedule keys must be non-empty strings")
    if (
        isinstance(steps, bool)
        or not isinstance(steps, int)
        or steps <= 0
        or steps % (2 * len(normalized))
    ):
        raise ValueError("G3 source-action schedule must complete every two-period scene cell")

    entries: list[dict[str, Any]] = []
    cycle_steps = 2 * len(normalized)
    for step in range(1, steps + 1):
        zero_based = step - 1
        cycle_index, cycle_offset = divmod(zero_based, cycle_steps)
        period_index, scene_index = divmod(cycle_offset, len(normalized))
        required = bool((scene_index + period_index) % 2)
        scene_key, source_task_key = normalized[scene_index]
        entries.append(
            {
                "global_step": step,
                "cycle_index": cycle_index,
                "cycle_offset": cycle_offset,
                "period_index": period_index,
                "scene_index": scene_index,
                "scene_key": scene_key,
                "source_task_key": source_task_key,
                "arm": (
                    TaskAddressActionInformationSet.MEDIATOR_REQUIRED.value
                    if required
                    else TaskAddressActionInformationSet.FACTUAL.value
                ),
            }
        )

    arm_counts = {
        arm.value: sum(entry["arm"] == arm.value for entry in entries)
        for arm in TaskAddressActionInformationSet
    }
    scene_arm_counts = {
        scene: {
            arm.value: sum(
                entry["scene_key"] == scene and entry["arm"] == arm.value for entry in entries
            )
            for arm in TaskAddressActionInformationSet
        }
        for scene, _task in normalized
    }
    payload: dict[str, Any] = {
        "schema": G3_SOURCE_ACTION_SCHEDULE_SCHEMA,
        "design": "source-task-action-scene-stratified-two-period-crossover",
        "single_forward_per_optimizer_step": True,
        "action_labels": "immutable-source-trajectory-only",
        "crossed_prompts_used_for_action_loss": False,
        "assignment": "(scene_index + period_index) mod 2",
        "steps": steps,
        "scene_count": len(normalized),
        "periods_per_cycle": 2,
        "cycle_steps": cycle_steps,
        "arm_counts": arm_counts,
        "scene_arm_counts": scene_arm_counts,
        "entries": entries,
    }
    payload["sha256"] = hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()
    return payload


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_targets_sha256(batch: Any) -> str:
    # Hash the released LingBot targets after its canonical feature transform.
    # The raw host names are ``action.lingbot`` and ``action.lingbot_is_pad``;
    # the forward ABI intentionally exposes them as ``actions`` and
    # ``action_is_pad``.
    names = ("actions", "action_is_pad")
    missing = [name for name in names if name not in batch.model_inputs]
    if missing:
        raise KeyError(f"G3 action supervision omitted official targets: {missing}")
    payload = {name: _tensor_sha256(batch.model_inputs[name]) for name in names}
    return hashlib.sha256(_canonical_json(payload).encode("ascii")).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
        raise ValueError("LTOP G3 cannot average an empty sequence")
    return sum(values) / len(values)


def _gate_failures(rank_reports: list[dict[str, Any]], *, mode: str) -> list[str]:
    failures: list[str] = []
    if len(rank_reports) != G2_WORLD_SIZE:
        return ["G3 omitted one or more distributed rank reports"]
    for report in rank_reports:
        rank = report["rank"]
        if not report["all_gradients_finite"]:
            failures.append(f"rank {rank}: non-finite gradients")
        if report["cuda_memory_bytes"]["peak_allocated"] >= 39 * 1024**3:
            failures.append(f"rank {rank}: peak allocated memory reached the A100 safety bound")
        required_gradients = (
            "native_graph_norm",
            "task_query_norm",
            "shared_host_norm",
            "action_output_norm",
        )
        if any(
            any(float(step.get(name, 0.0)) <= 0 for name in required_gradients)
            for step in report["gradient_metrics_history"]
        ):
            failures.append(
                f"rank {rank}: one required large-model/action gradient surface was zero"
            )
        final = report["history"][-1]
        partitions = ("validation",) if mode == "smoke" else ("validation", "heldout")
        for partition in partitions:
            if float(final[partition]["max_replay_floor_rms"]) != 0.0:
                failures.append(f"rank {rank}: {partition} factual replay was not bitwise stable")
    if mode == "smoke":
        return failures

    first_losses = [value for report in rank_reports for value in report["action_losses"][:16]]
    last_losses = [value for report in rank_reports for value in report["action_losses"][-16:]]
    if _mean(last_losses) >= 0.95 * _mean(first_losses):
        failures.append("G3 official action loss did not improve by at least five percent")
    final_scores = {
        partition: [
            scene["score"]
            for report in rank_reports
            for scene in report["history"][-1][partition]["scenes"]
        ]
        for partition in ("validation", "heldout")
    }
    for partition, scores in final_scores.items():
        factual = [float(score["mean_factual_target_minus_distractor"]) for score in scores]
        did = [float(score["mean_blocked_path_difference_in_differences"]) for score in scores]
        if _mean(factual) <= 0:
            failures.append(f"G3 {partition} target-row action effect did not beat distractors")
        if _mean(did) <= 0:
            failures.append(
                f"G3 {partition} blocked-path difference-in-differences was nonpositive"
            )
        sample_count = sum(len(score["sample_keys"]) for score in scores)
        positive_factual = sum(int(score["positive_factual_count"]) for score in scores)
        positive_did = sum(int(score["positive_blocked_path_did_count"]) for score in scores)
        minimum = math.ceil(0.625 * sample_count)
        if positive_factual < minimum:
            failures.append(f"G3 {partition} positive factual count {positive_factual} < {minimum}")
        if positive_did < minimum:
            failures.append(f"G3 {partition} positive blocked DID count {positive_did} < {minimum}")
    return failures


def _training_failures(rank_reports: list[dict[str, Any]], *, mode: str) -> list[str]:
    failures: list[str] = []
    if len(rank_reports) != G2_WORLD_SIZE:
        return ["G3 training phase omitted one or more distributed rank reports"]
    for report in rank_reports:
        rank = report["rank"]
        if not report["all_gradients_finite"]:
            failures.append(f"rank {rank}: non-finite gradients")
        if report["cuda_memory_bytes"]["peak_allocated"] >= 39 * 1024**3:
            failures.append(f"rank {rank}: peak allocated memory reached the A100 safety bound")
        required_gradients = (
            "native_graph_norm",
            "task_query_norm",
            "shared_host_norm",
            "action_output_norm",
        )
        if any(
            any(float(step.get(name, 0.0)) <= 0 for name in required_gradients)
            for step in report["gradient_metrics_history"]
        ):
            failures.append(
                f"rank {rank}: one required large-model/action gradient surface was zero"
            )
        supervision = report.get("action_supervision_history")
        if not isinstance(supervision, list) or len(supervision) != len(
            report.get("action_losses", ())
        ):
            failures.append(f"rank {rank}: action supervision history is incomplete")
        elif any(
            not isinstance(receipt, dict)
            or receipt.get("schema") != TASK_ACTION_SUPERVISION_SCHEMA
            or receipt.get("scope") != TaskActionSupervisionScope.FACTUAL_ACTION.value
            or receipt.get("official_action_loss_enabled") is not True
            or receipt.get("source_task_key") != receipt.get("candidate_task_key")
            or receipt.get("source_instruction_sha256")
            != receipt.get("candidate_instruction_sha256")
            for receipt in supervision
        ):
            failures.append(f"rank {rank}: official action loss used non-factual supervision")
        if mode in {"mediator-trial", "smoke"}:
            expected_per_arm = len(report.get("action_losses", ())) // 2
            expected_counts = {
                TaskAddressActionInformationSet.FACTUAL.value: expected_per_arm,
                TaskAddressActionInformationSet.MEDIATOR_REQUIRED.value: expected_per_arm,
            }
            if report.get("action_information_set_counts") != expected_counts:
                failures.append(f"rank {rank}: source-action arm counts are not balanced")
            history = report.get("action_information_set_history")
            expected_steps = (
                G3_MEDIATOR_TRIAL_STEPS
                if mode == "mediator-trial"
                else len(report.get("action_losses", ()))
            )
            if not isinstance(history, list) or len(history) != expected_steps:
                failures.append(f"rank {rank}: source-action arm history is incomplete")
            elif any(
                not isinstance(item, dict)
                or not isinstance(item.get("prompt_key"), str)
                or not item["prompt_key"].startswith("source-task/")
                for item in history
            ):
                failures.append(f"rank {rank}: source-action history contains a crossed prompt")
            else:
                scene_keys = {item.get("scene_key") for item in history}
                if None in scene_keys or not scene_keys or expected_steps % (2 * len(scene_keys)):
                    failures.append(f"rank {rank}: source-action scene schedule is malformed")
                else:
                    expected_scene_arm_count = expected_steps // (2 * len(scene_keys))
                    if any(
                        sum(
                            item.get("scene_key") == scene_key and item.get("arm") == arm.value
                            for item in history
                        )
                        != expected_scene_arm_count
                        for scene_key in scene_keys
                        for arm in TaskAddressActionInformationSet
                    ):
                        failures.append(
                            f"rank {rank}: source-action scene/arm cells are not balanced"
                        )
            journal = report.get("arm_journal")
            if not isinstance(journal, dict) or journal.get("record_count") != expected_steps:
                failures.append(f"rank {rank}: source-action arm journal is incomplete")
    if mode != "smoke":
        first_losses = [value for report in rank_reports for value in report["action_losses"][:16]]
        last_losses = [value for report in rank_reports for value in report["action_losses"][-16:]]
        if _mean(last_losses) >= 0.95 * _mean(first_losses):
            failures.append("G3 official action loss did not improve by at least five percent")
    if mode == "mediator-trial":
        schedule_digests = {
            report.get("action_information_set_schedule_sha256") for report in rank_reports
        }
        if len(schedule_digests) != 1 or None in schedule_digests:
            failures.append("G3 mediator-trial ranks used different counterbalance schedules")
    return failures


def _evaluation_failures(rank_reports: list[dict[str, Any]], *, mode: str) -> list[str]:
    failures: list[str] = []
    if len(rank_reports) != G2_WORLD_SIZE:
        return ["G3 evaluation phase omitted one or more distributed rank reports"]
    partitions = ("validation",) if mode == "smoke" else ("validation", "heldout")
    for report in rank_reports:
        rank = report["rank"]
        if report["cuda_memory_bytes"]["peak_allocated"] >= 39 * 1024**3:
            failures.append(f"rank {rank}: peak allocated memory reached the A100 safety bound")
        if len(report["history"]) != 1:
            failures.append(f"rank {rank}: staged evaluation did not publish exactly one receipt")
            continue
        final = report["history"][0]
        for partition in partitions:
            if float(final[partition]["max_replay_floor_rms"]) != 0.0:
                failures.append(f"rank {rank}: {partition} factual replay was not bitwise stable")
    if failures or mode == "smoke":
        return failures
    final_scores = {
        partition: [
            scene["score"]
            for report in rank_reports
            for scene in report["history"][0][partition]["scenes"]
        ]
        for partition in partitions
    }
    for partition, scores in final_scores.items():
        factual = [float(score["mean_factual_target_minus_distractor"]) for score in scores]
        did = [float(score["mean_blocked_path_difference_in_differences"]) for score in scores]
        if _mean(factual) <= 0:
            failures.append(f"G3 {partition} target-row action effect did not beat distractors")
        if _mean(did) <= 0:
            failures.append(
                f"G3 {partition} blocked-path difference-in-differences was nonpositive"
            )
        sample_count = sum(len(score["sample_keys"]) for score in scores)
        positive_factual = sum(int(score["positive_factual_count"]) for score in scores)
        positive_did = sum(int(score["positive_blocked_path_did_count"]) for score in scores)
        minimum = math.ceil(0.625 * sample_count)
        if positive_factual < minimum:
            failures.append(f"G3 {partition} positive factual count {positive_factual} < {minimum}")
        if positive_did < minimum:
            failures.append(f"G3 {partition} positive blocked DID count {positive_did} < {minimum}")
    return failures


def _retention_failures(rank_reports: list[dict[str, Any]]) -> list[str]:
    """Apply the accepted G2b representation floors after G3 joint training."""

    if len(rank_reports) != G2_WORLD_SIZE:
        return ["G3 representation retention omitted one or more distributed ranks"]
    failures: list[str] = []
    final = {partition: [] for partition in ("validation", "heldout")}
    for report in rank_reports:
        rank = report["rank"]
        if report["cuda_memory_bytes"]["peak_allocated"] >= 39 * 1024**3:
            failures.append(f"rank {rank}: peak allocated memory reached the A100 safety bound")
        if len(report["history"]) != 1:
            failures.append(f"rank {rank}: retention phase did not publish exactly one receipt")
            continue
        for partition in final:
            partition_report = report["history"][0][partition]
            if partition_report["scene_count"] != 4 or partition_report["prompt_count"] != 8:
                failures.append(f"rank {rank}: {partition} retention scene axis is incomplete")
            if not partition_report["shared_row_gauge"]:
                failures.append(f"rank {rank}: {partition} row gauge changed across prompts")
            if float(partition_report["physical_prompt_drift_max_abs"]) > 1.0e-5:
                failures.append(f"rank {rank}: {partition} physical rows became prompt dependent")
            if (
                float(
                    partition_report["metric_self_checks"]["matched_row_permutation_max_abs_error"]
                )
                > 1.0e-6
            ):
                failures.append(f"rank {rank}: {partition} row-permutation self-check failed")
            final[partition].append(partition_report)
    if failures:
        return failures

    def aggregate(partition: str, field: str) -> float:
        values = final[partition]
        return sum(float(value[field]) for value in values) / len(values)

    validation_prompts = [prompt for value in final["validation"] for prompt in value["prompts"]]
    heldout_prompts = [prompt for value in final["heldout"] for prompt in value["prompts"]]
    if sum(float(prompt["margin"]) > 0 for prompt in validation_prompts) < 12:
        failures.append("G3 retention validation target margin is positive for fewer than 12/16")
    if aggregate("validation", "mean_margin") < 0.02:
        failures.append("G3 retention validation mean target margin is below 0.02")
    if sum(float(prompt["margin"]) > 0 for prompt in heldout_prompts) < 10:
        failures.append("G3 retention heldout target margin is positive for fewer than 10/16")
    if aggregate("heldout", "mean_margin") <= 0:
        failures.append("G3 retention heldout mean target margin is nonpositive")
    return failures


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    picf_source_contract = _picf_source_contract()
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("CUDA allocator pre-bootstrap differs from parsed arguments")
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
    execution, labels = _load_contracts(
        args.execution_contract,
        args.offline_labels,
        expected_item_count=16,
    )

    with open_lingbot_vla2_ltop_stage_runtime(stage_contract) as runtime:
        torch = runtime.runtime_modules.torch
        dist = runtime.runtime_modules.dist
        rank = runtime.rank
        device = runtime.device

        import numpy as np
        from lingbotvla.checkpoint import build_checkpointer
        from lingbotvla.data import VLADataCollatorWithPacking
        from lingbotvla.data.vla_data.utils import FeatureTransform
        from lingbotvla.models import build_processor
        from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
        from lingbotvla.models.vla.lingbot_vla.moe_load_balance import (
            build_moe_load_balance_hook,
        )
        from lingbotvla.ops import fused_moe
        from lingbotvla.optim import build_muon_optimizer

        from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
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
            build_native_calvin_replay_batch,
            collate_native_calvin_training_batch,
            materialize_native_flow_randomness,
        )
        from picf_next.lingbot_native.calvin_entity_set import (
            build_task_independent_calvin_targets,
            physical_frame_predictions_from_relation,
            physical_frame_row_bindings,
        )
        from picf_next.lingbot_native.entity_set_objective import (
            eligible_physical_tracks,
            match_physical_frame_entities,
            physical_frame_set_loss,
        )
        from picf_next.lingbot_native.host import (
            LingBotNativePriorStepper,
            native_context_from_prior_trace,
        )
        from picf_next.lingbot_native.ltop_action_mediation import (
            OfflineLTOPActionTargets,
            build_label_blind_ltop_action_arms,
            score_offline_ltop_action_mediation,
            seal_ltop_action_receipt,
        )
        from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
        from picf_next.lingbot_native.state import AddressedLayerwisePriorTrace
        from picf_next.lingbot_native.task_address_learning import (
            task_address_row_coverage,
            task_address_target_coverage,
        )
        from picf_next.lingbot_native.task_address_receipt import task_address_attention_receipt
        from picf_next.lingbot_native.torch_dcp_compat import (
            install_torch_2_8_sparse_optimizer_state_backport,
        )
        from picf_next.lingbot_native.training import (
            audit_native_optimizer_coverage,
            run_native_policy_diagnostic_forward,
            run_native_policy_training_forward,
        )

        install_torch_2_8_sparse_optimizer_state_backport(torch)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        torch.cuda.reset_peak_memory_stats(device)

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
            raise ValueError("LTOP G3 CALVIN manifest and normalization differ")
        dataset_contract = validate_dataset_runtime_binding(
            manifest,
            args.dataset_split,
            dataset_id=norm_source["dataset_id"],
            dataset_revision=norm_source["dataset_revision"],
            split_name=args.dataset_split.name,
        )
        _validate_representation_execution_provenance(
            execution,
            dataset_manifest_file_sha256=_sha256(args.dataset_manifest),
            dataset_tree_sha256=manifest.tree_sha256,
        )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        dataset = CalvinStatefulTransitionDataset(
            index,
            action_horizon=runtime.model_config.chunk_size,
        )
        sidecar = CalvinPhysicalSupervisionSidecar(
            args.physical_sidecar_root,
            index,
            manifest_path=args.physical_sidecar_manifest,
            expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        )
        _merged, data_mapping = _resolve_training_config(
            runtime.training_config,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=args.steps,
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
        optimizer = None
        optimizer_manifest = None
        cold_loaded_model_local_state_sha256: str | None = None
        expected_training_model_local_state_sha256: str | None = None
        trained_checkpoint_model_tree_sha256: str | None = None
        if args.phase in {"evaluation", "retention"}:
            if args.trained_checkpoint is None:
                raise AssertionError("validated G3 trained checkpoint disappeared")
            checkpoint_manifest_path = (
                args.trained_checkpoint.resolve() / "ltop_g3_training_checkpoint.json"
            )
            checkpoint_manifest = json.loads(checkpoint_manifest_path.read_text(encoding="ascii"))
            expected_digests, expected_model_tree_sha256 = (
                _validate_g3_training_checkpoint_manifest(
                    checkpoint_manifest,
                    expected_layer_count=graph.config.num_layers,
                    expected_picf_source_contract=picf_source_contract,
                )
            )
            expected_training_model_local_state_sha256 = expected_digests[rank]
            checkpoint_tree_identity: list[Any] = [None]
            if rank == 0:
                try:
                    checkpoint_tree_identity[0] = directory_tree_sha256(
                        args.trained_checkpoint.resolve() / "model",
                        schema=G3_MODEL_TREE_SCHEMA,
                    )
                except BaseException as error:
                    checkpoint_tree_identity[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(checkpoint_tree_identity, src=0)
            if isinstance(checkpoint_tree_identity[0], dict):
                raise RuntimeError(
                    f"G3 trained checkpoint model-tree digest failed: {checkpoint_tree_identity[0]}"
                )
            trained_checkpoint_model_tree_sha256 = checkpoint_tree_identity[0]
            if trained_checkpoint_model_tree_sha256 != expected_model_tree_sha256:
                raise ValueError("G3 trained checkpoint model tree changed after publication")
            restored = {"model": policy}
            checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
            _distributed_rank_local_call(
                action=lambda: checkpointer.load(
                    str(args.trained_checkpoint.resolve()),
                    restored,
                    allow_partial_load=False,
                ),
                phase="ltop-g3-staged-evaluation-cold-load",
                rank=rank,
                dist_module=dist,
            )
            if set(restored) != {"model"} or restored["model"] is not policy:
                raise RuntimeError("G3 staged evaluation changed the model-only restore boundary")
            cold_loaded_model_local_state_sha256 = _distributed_rank_local_call(
                action=lambda: _model_local_state_digest(policy, torch),
                phase="ltop-g3-staged-evaluation-cold-loaded-model-digest",
                rank=rank,
                dist_module=dist,
            )
            if cold_loaded_model_local_state_sha256 != expected_training_model_local_state_sha256:
                raise RuntimeError(
                    "G3 cold-loaded model differs from the training terminal rank state"
                )
            policy.requires_grad_(False)
            if args.phase == "retention":
                # The diagnostic forward deliberately executes the same train-mode
                # surface used by the accepted G2b representation gate, but under
                # no-grad with every parameter frozen. Action sampling remains an
                # eval-mode contract in the separate evaluation phase.
                policy.train()
                graph.train()
            else:
                policy.eval()
                graph.eval()
            torch.cuda.synchronize(device)
        else:
            policy.requires_grad_(True)
            policy.train()
            graph.train()
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
        moe_inference_backend = select_lingbot_deterministic_moe_backend(
            action_expert_module=qwen2_action_expert,
            fused_moe_module=fused_moe,
        )

        local_items, runtime_schedule = _local_representation_contract_items(
            execution,
            labels,
            rank=rank,
        )
        gathered_contract_items: list[Any] = [None] * G2_WORLD_SIZE
        dist.all_gather_object(gathered_contract_items, local_items)
        global_items_by_id: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
        for rank_items in gathered_contract_items:
            if not isinstance(rank_items, tuple):
                raise RuntimeError("LTOP G3 rank contract exchange changed type")
            for item, label in rank_items:
                item_id = item["item_id"]
                if item_id in global_items_by_id:
                    raise RuntimeError("LTOP G3 rank contract exchange duplicated an item")
                global_items_by_id[item_id] = (item, label)
        global_items = tuple(
            sorted(
                global_items_by_id.values(),
                key=lambda value: (value[0]["partition"], value[0]["ordinal"]),
            )
        )
        if len(global_items) != 16:
            raise RuntimeError("LTOP G3 rank contract exchange did not recover all 16 items")
        scene_contract_items = (
            tuple(value for value in global_items if value[0]["partition"] == "validation")
            if args.mode == "mediator-trial"
            else local_items
        )

        def collate_host(candidate: Any) -> CollatedNativeCALVINBatch:
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
                    device=torch.device("cpu"),
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

        def batch_to_device(batch: CollatedNativeCALVINBatch) -> CollatedNativeCALVINBatch:
            return CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    batch.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=batch.controls,
                routing=batch.routing,
                source_digest=batch.source_digest,
                structural_target_requests=batch.structural_target_requests,
                modalities=None,
                prior_control_chunks=batch.prior_control_chunks,
            )

        scenes: dict[str, list[dict[str, Any]]] = {"validation": [], "heldout": []}
        for item, label in scene_contract_items:
            source = build_native_calvin_replay_batch(
                dataset,
                sample_key=item["sample_key"],
                lane_id=rank,
                episode_instance_id=f"ltop-g3/{item['item_id']}",
                optimizer_step=0,
                replay_seed=item["replay_seed"],
                device=device,
                dtype=torch.bfloat16,
            )
            request = source.training.structural_target_requests[0]
            _validate_representation_item_source(
                item,
                request=request,
                canonical_source_global_index=dataset.source_global_index_by_key(
                    item["sample_key"]
                ),
                sidecar_source_state_sha256=sidecar.source_state_sha256(
                    request.source_global_index
                ),
            )
            source_batch = collate_host(source)
            source_host_item = source.training.host_items[0]
            source_instruction = source_host_item.get("task")
            if not isinstance(source_instruction, str) or not source_instruction:
                raise RuntimeError("LTOP G3 source action batch omitted its complete task text")
            source_task_key = request.task_key
            source_action_targets_sha256 = _action_targets_sha256(source_batch)
            source_action_supervision = task_action_supervision_receipt(
                sample_key=item["sample_key"],
                source_task_key=source_task_key,
                source_instruction=source_instruction,
                candidate_task_key=source_task_key,
                candidate_instruction=source_instruction,
                source_action_targets_sha256=source_action_targets_sha256,
                candidate_action_targets_sha256=source_action_targets_sha256,
            )
            require_factual_action_supervision(source_action_supervision)

            prompt_candidates = tuple(_prompt_variant(source, prompt) for prompt in item["prompts"])
            batches = tuple(collate_host(candidate) for candidate in prompt_candidates)
            prompt_supervision = []
            for prompt, batch in zip(item["prompts"], batches, strict=True):
                receipt = task_action_supervision_receipt(
                    sample_key=item["sample_key"],
                    source_task_key=source_task_key,
                    source_instruction=source_instruction,
                    candidate_task_key=prompt["task_key"],
                    candidate_instruction=prompt["instruction"],
                    source_action_targets_sha256=source_action_targets_sha256,
                    candidate_action_targets_sha256=_action_targets_sha256(batch),
                )
                if receipt.scope is not TaskActionSupervisionScope.REPRESENTATION_ONLY:
                    raise RuntimeError(
                        "LTOP G3 crossed prompt unexpectedly retained factual action scope"
                    )
                prompt_supervision.append(receipt)
            target_identities: list[str] = []
            for prompt, prompt_label in zip(item["prompts"], label["prompts"], strict=True):
                identities = calvin_exact_task_loss_identities(prompt["task_key"])
                if identities is None or len(identities) != 1:
                    raise RuntimeError("LTOP G3 requires one exact task identity")
                identity = identities[0]
                if identity != prompt_label["target_identity_key"]:
                    raise RuntimeError("LTOP G3 exact task identity differs from offline label")
                target_identities.append(identity)
            if target_identities[0] == target_identities[1]:
                raise RuntimeError("LTOP G3 crossed prompts require distinct physical targets")
            source_identities = calvin_exact_task_loss_identities(source_task_key)
            source_target_identity = (
                source_identities[0]
                if source_identities is not None and len(source_identities) == 1
                else None
            )
            scenes[item["partition"]].append(
                {
                    "item": item,
                    "batches": batches,
                    "source_batch": source_batch,
                    "source_task_key": source_task_key,
                    "source_instruction": source_instruction,
                    "source_target_identity": source_target_identity,
                    "source_action_supervision": source_action_supervision,
                    "prompt_supervision": tuple(prompt_supervision),
                    "target_identities": tuple(target_identities),
                }
            )

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
                raise RuntimeError("LTOP G3 prior rollout omitted addressed rows")
            return prior

        joint_host = policy.model.qwenvl_with_expert
        original_attention_interface = joint_host.attention_interface
        active_capture: dict[str, Any] | None = None

        def attention_interface_with_receipt(
            query_states: Any,
            key_states: Any,
            value_states: Any,
            attention_mask: Any,
        ) -> Any:
            nonlocal active_capture
            if active_capture is not None:
                context = active_capture["context"]
                layout = context.task_address_attention_layout
                if layout is not None and active_capture["layer_count"] < graph.config.num_layers:
                    if query_states.shape[:2] != (layout.batch_size, layout.query_count):
                        raise RuntimeError(
                            "LTOP G3 expanded-prefix attention differs from its executed layout"
                        )
                    receipt = task_address_attention_receipt(
                        query_states=query_states,
                        key_states=key_states,
                        attention_mask=attention_mask,
                        object_read_slice=layout.object_read_slice,
                        prior_slice=layout.prior_slice,
                        posterior_slice=layout.posterior_slice,
                        capacity=layout.capacity,
                    )
                    layer_index = active_capture["layer_count"]
                    active_capture["layer_count"] = layer_index + 1
                    if layer_index == graph.config.num_layers - 2:
                        active_capture["action_consumable_row_mass"] = receipt.row_mass
            return original_attention_interface(
                query_states,
                key_states,
                value_states,
                attention_mask,
            )

        joint_host.attention_interface = attention_interface_with_receipt

        def training_forward(
            batch: CollatedNativeCALVINBatch,
            prior: AddressedLayerwisePriorTrace,
            *,
            require_grad: bool,
            action_information_sets: tuple[TaskAddressActionInformationSet, ...],
        ) -> tuple[Any, Any]:
            nonlocal active_capture
            visible = torch.ones(
                (batch.routing.batch_size, args.capacity),
                dtype=torch.bool,
                device=device,
            )
            context = native_context_from_prior_trace(
                controls=batch.controls,
                prior_trace=prior,
                modalities=None,
                action_information_sets=action_information_sets,
                object_read_source_row_visible=visible,
            )
            active_capture = {
                "context": context,
                "layer_count": 0,
                "action_consumable_row_mass": None,
            }
            try:
                result = (
                    run_native_policy_training_forward(
                        policy,
                        model_inputs=batch.model_inputs,
                        context=context,
                    )
                    if require_grad
                    else run_native_policy_diagnostic_forward(
                        policy,
                        model_inputs=batch.model_inputs,
                        context=context,
                    )
                )
            finally:
                captured = active_capture
                active_capture = None
            if captured is None or captured["layer_count"] != graph.config.num_layers:
                raise RuntimeError("LTOP G3 did not capture every shared-host layer")
            row_mass = captured["action_consumable_row_mass"]
            if row_mass is None:
                raise RuntimeError("LTOP G3 omitted the action-consumable address layer")
            address_depth = action_consumable_task_address(
                row_mass,
                layer_count=graph.config.num_layers,
            )
            row_mass = address_depth.row_mass
            if row_mass is None or require_grad != bool(row_mass.requires_grad):
                raise RuntimeError("LTOP G3 task-address attachment differs from phase")
            return result, row_mass

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
            allow_unobservable_target: bool = False,
        ) -> dict[str, Any]:
            relation = context.relation_output
            if not isinstance(relation, PhysicalRelationOutput):
                raise RuntimeError("LTOP G3 observation omitted physical relations")
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
                raise RuntimeError("LTOP G3 crossed prompts changed the physical identity axis")
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
            target_row, target_row_reason = _resolve_physical_target_row(
                target_identity=target_identity,
                identity_keys=target_bundle.identity_keys_by_batch[0],
                eligible_track_indices=eligible_tracks.detach().cpu().tolist(),
                bindings=bindings,
                allow_unobservable=allow_unobservable_target,
            )
            return {
                "relation": relation,
                "set_loss": set_loss,
                "assignment": assignment,
                "matched_assignment": matched_assignment,
                "bindings": bindings,
                "matched_bindings": matched_bindings,
                "identity_keys_by_batch": target_bundle.identity_keys_by_batch,
                "target_row": target_row,
                "target_row_reason": target_row_reason,
            }

        def evaluate_scene(scene: dict[str, Any]) -> list[dict[str, Any]]:
            outputs: list[dict[str, Any]] = []
            policy_was_training = policy.training
            graph_was_training = graph.training
            original_config = {
                name: getattr(runtime.model_config, name)
                for name in (
                    "use_cache",
                    "use_compile",
                    "attention_implementation",
                    "vit_attn_implementation",
                    "num_steps",
                )
            }
            policy.eval()
            graph.eval()
            apply_ltop_g1_inference_contract(runtime.model_config)
            runtime.model_config.num_steps = LINGBOT_RELEASED_ACTION_SAMPLING_STEPS
            require_lingbot_released_action_sampling_steps(runtime.model_config)
            evaluation_information_set = TaskAddressActionInformationSet(
                args.evaluation_action_information_set
            )
            try:
                prompt_batches = scene["batches"][:1] if args.mode == "smoke" else scene["batches"]
                for prompt_index, host_batch in enumerate(prompt_batches):
                    batch = batch_to_device(host_batch)
                    prior = build_prior(batch)
                    runtime_inputs = {
                        name: batch.model_inputs[name] for name in sorted(_RUNTIME_MODEL_FIELDS)
                    }
                    _input_manifest, deploy_inputs_sha256 = _tensor_manifest(runtime_inputs)
                    noise = batch.model_inputs["noise"]
                    noise_sha256 = _tensor_sha256(noise)
                    arms = build_label_blind_ltop_action_arms(
                        batch_size=batch.routing.batch_size,
                        capacity=args.capacity,
                        device=device,
                    )
                    receipts = []
                    factual_context = None
                    for arm in arms:
                        context = native_context_from_prior_trace(
                            controls=batch.controls,
                            prior_trace=prior,
                            modalities=None,
                            action_information_sets=(evaluation_information_set,)
                            * batch.routing.batch_size,
                            object_read_action_intervention=arm.object_read_action_intervention,
                            object_read_source_row_visible=arm.object_read_source_row_visible,
                        )
                        with torch.no_grad():
                            action = policy.sample_actions(
                                **runtime_inputs,
                                noise=noise.clone(),
                                picf_native_context=context,
                            )
                        receipts.append(
                            seal_ltop_action_receipt(
                                prompt_name=scene["item"]["prompts"][prompt_index]["name"],
                                sample_keys=batch.routing.episode_keys,
                                arm=arm,
                                deploy_inputs_sha256=deploy_inputs_sha256,
                                inference_randomness_sha256=noise_sha256,
                                action_output=action,
                            )
                        )
                        if arm.name == "factual":
                            factual_context = context
                    if factual_context is None:
                        raise RuntimeError("LTOP G3 factual action arm was not executed")

                    # The action boundary is now closed. Only this block opens
                    # loss-side identities and the physical sidecar.
                    target_identity = scene["target_identities"][prompt_index]
                    distractor_identity = scene["target_identities"][1 - prompt_index]
                    physical = physical_supervision(
                        context=factual_context,
                        batch=batch,
                        target_identity=target_identity,
                    )
                    binding_map = dict(physical["bindings"])
                    if distractor_identity not in binding_map:
                        raise RuntimeError("LTOP G3 crossed-prompt distractor is unbound")
                    score = score_offline_ltop_action_mediation(
                        receipts,
                        targets=OfflineLTOPActionTargets(
                            prompt_name=scene["item"]["prompts"][prompt_index]["name"],
                            sample_keys=batch.routing.episode_keys,
                            target_rows=torch.tensor(
                                [physical["target_row"]],
                                dtype=torch.long,
                                device=device,
                            ),
                            matched_distractor_rows=torch.tensor(
                                [binding_map[distractor_identity]],
                                dtype=torch.long,
                                device=device,
                            ),
                        ),
                        capacity=args.capacity,
                    )
                    outputs.append(
                        {
                            "prompt_name": scene["item"]["prompts"][prompt_index]["name"],
                            "target_identity": target_identity,
                            "matched_distractor_identity": distractor_identity,
                            "target_row": physical["target_row"],
                            "matched_distractor_row": binding_map[distractor_identity],
                            "score": _score_to_json(score),
                        }
                    )
                    del (
                        action,
                        arm,
                        batch,
                        context,
                        factual_context,
                        noise,
                        physical,
                        prior,
                        receipts,
                        runtime_inputs,
                        score,
                    )
            finally:
                for name, value in original_config.items():
                    setattr(runtime.model_config, name, value)
                policy.train(policy_was_training)
                graph.train(graph_was_training)
            return outputs

        def evaluate_retention_scene(scene: dict[str, Any]) -> dict[str, Any]:
            """Re-evaluate the accepted G2b representation contract after G3."""

            distributions = []
            target_rows: list[int] = []
            bindings_by_prompt = []
            independent_bindings_by_prompt = []
            set_losses = []
            relation_predictions = []
            canonical_assignment = None
            canonical_bindings = None
            canonical_identity_keys = None
            first_batch = batch_to_device(scene["batches"][0])
            prior = build_prior(first_batch)
            with torch.no_grad():
                for prompt_index, host_batch in enumerate(scene["batches"]):
                    batch = first_batch if prompt_index == 0 else batch_to_device(host_batch)
                    result, row_mass = training_forward(
                        batch,
                        prior,
                        require_grad=False,
                        action_information_sets=(TaskAddressActionInformationSet.FACTUAL,)
                        * batch.routing.batch_size,
                    )
                    physical = physical_supervision(
                        context=result.context,
                        batch=batch,
                        target_identity=scene["target_identities"][prompt_index],
                        canonical_assignment=canonical_assignment,
                        canonical_identity_keys=canonical_identity_keys,
                    )
                    if canonical_assignment is None:
                        canonical_assignment = physical["matched_assignment"]
                        canonical_bindings = physical["matched_bindings"]
                        canonical_identity_keys = physical["identity_keys_by_batch"]
                    if canonical_bindings is None:
                        raise RuntimeError("G3 retention did not establish a canonical row gauge")
                    target_row = physical["target_row"]
                    target_rows.append(target_row)
                    bindings_by_prompt.append([list(value) for value in physical["bindings"]])
                    independent_bindings_by_prompt.append(
                        [list(value) for value in physical["matched_bindings"]]
                    )
                    set_losses.append(float(physical["set_loss"].total.float().item()))
                    relation_predictions.append(physical["relation"])
                    distributions.append(
                        task_address_target_coverage(
                            row_mass,
                            torch.tensor([target_row], dtype=torch.long, device=device),
                        ).conditional_distribution
                    )
                    if prompt_index != 0:
                        del batch
                    del result
            metrics = _scene_metrics(
                (distributions[0], distributions[1]),
                (target_rows[0], target_rows[1]),
                task_address_row_coverage=task_address_row_coverage,
                torch_module=torch,
            )
            return {
                **metrics,
                "item_id": scene["item"]["item_id"],
                "sample_key": scene["item"]["sample_key"],
                "target_identities": list(scene["target_identities"]),
                "target_rows": target_rows,
                "bindings_by_prompt": bindings_by_prompt,
                "independent_bindings_by_prompt": independent_bindings_by_prompt,
                "shared_row_gauge": all(
                    tuple(tuple(value) for value in bindings) == canonical_bindings
                    for bindings in independent_bindings_by_prompt
                ),
                "mean_physical_set_loss": sum(set_losses) / len(set_losses),
                "physical_prompt_drift_max_abs": _physical_relation_prompt_drift(
                    relation_predictions[0],
                    relation_predictions[1],
                ),
            }

        def evaluate_retention_partition(partition: str) -> dict[str, Any]:
            per_scene = [evaluate_retention_scene(scene) for scene in scenes[partition]]
            prompts = [prompt for scene in per_scene for prompt in scene["prompts"]]
            return {
                "scene_count": len(per_scene),
                "prompt_count": len(prompts),
                "mean_margin": sum(float(prompt["margin"]) for prompt in prompts) / len(prompts),
                "positive_margin_count": sum(float(prompt["margin"]) > 0 for prompt in prompts),
                "mean_target_nll": sum(float(scene["mean_target_nll"]) for scene in per_scene)
                / len(per_scene),
                "mean_physical_set_loss": sum(
                    float(scene["mean_physical_set_loss"]) for scene in per_scene
                )
                / len(per_scene),
                "physical_prompt_drift_max_abs": max(
                    float(scene["physical_prompt_drift_max_abs"]) for scene in per_scene
                ),
                "shared_row_gauge": all(bool(scene["shared_row_gauge"]) for scene in per_scene),
                "metric_self_checks": {
                    "matched_row_permutation_max_abs_error": max(
                        float(scene["metric_self_checks"]["matched_row_permutation_max_abs_error"])
                        for scene in per_scene
                    )
                },
                "prompts": prompts,
                "scenes": per_scene,
            }

        evaluation_scenes = (
            {"validation": scenes["validation"][:1]}
            if args.mode == "smoke"
            else {
                partition: partition_scenes[: args.evaluation_scenes_per_partition]
                for partition, partition_scenes in scenes.items()
            }
        )
        history: list[dict[str, Any]] = []

        def record(step: int) -> None:
            entry: dict[str, Any] = {"step": step}
            for partition, partition_scenes in evaluation_scenes.items():
                scene_outputs = []
                for scene in partition_scenes:
                    prompts = evaluate_scene(scene)
                    scene_outputs.append(
                        {
                            "item_id": scene["item"]["item_id"],
                            "sample_key": scene["item"]["sample_key"],
                            "prompts": prompts,
                            "score": {
                                "prompt_name": f"{scene['item']['item_id']}/aggregate",
                                "sample_keys": [scene["item"]["sample_key"] for _ in prompts],
                                "replay_floor_rms": [
                                    value
                                    for prompt in prompts
                                    for value in prompt["score"]["replay_floor_rms"]
                                ],
                                "mean_factual_target_minus_distractor": _mean(
                                    [
                                        prompt["score"]["mean_factual_target_minus_distractor"]
                                        for prompt in prompts
                                    ]
                                ),
                                "mean_blocked_path_difference_in_differences": _mean(
                                    [
                                        prompt["score"][
                                            "mean_blocked_path_difference_in_differences"
                                        ]
                                        for prompt in prompts
                                    ]
                                ),
                                "positive_factual_count": sum(
                                    prompt["score"]["positive_factual_count"] for prompt in prompts
                                ),
                                "positive_blocked_path_did_count": sum(
                                    prompt["score"]["positive_blocked_path_did_count"]
                                    for prompt in prompts
                                ),
                            },
                        }
                    )
                replay = [
                    value for scene in scene_outputs for value in scene["score"]["replay_floor_rms"]
                ]
                entry[partition] = {
                    "scenes": scene_outputs,
                    "max_replay_floor_rms": max(replay, default=math.inf),
                }
            history.append(entry)

        action_losses: list[float] = []
        total_losses: list[float] = []
        physical_losses: list[float] = []
        address_losses: list[float] = []
        gradient_metrics_history: list[dict[str, Any]] = []
        action_information_set_history: list[dict[str, Any]] = []
        action_information_set_counts = {arm.value: 0 for arm in TaskAddressActionInformationSet}
        action_supervision_history: list[dict[str, Any]] = []
        task_address_supervision_history: list[dict[str, Any]] = []
        all_gradients_finite = True
        mediator_schedule: dict[str, Any] | None = None
        if args.mode in {"mediator-trial", "smoke"}:
            mediator_schedule = build_g3_source_action_counterbalanced_schedule(
                scene_source_keys=tuple(
                    (
                        scene["item"]["item_id"],
                        scene["source_task_key"],
                    )
                    for scene in scenes["validation"]
                ),
                steps=args.steps,
            )
        journal_handle = None
        journal_path = None
        trace_handle = None
        trace_path = None
        if args.phase in {"combined", "training"}:
            if args.journal_dir is None:
                raise AssertionError("validated G3 journal directory disappeared")
            if rank == 0:
                args.journal_dir.mkdir(parents=True, exist_ok=False)
                _fsync_directory(args.journal_dir.parent)
            dist.barrier()
            journal_path = args.journal_dir / f"rank_{rank}.jsonl"
            journal_handle = journal_path.open("x", encoding="ascii", buffering=1)
            if args.mode == "smoke":
                trace_path = args.journal_dir / f"rank_{rank}.trace.jsonl"
                trace_handle = trace_path.open("x", encoding="ascii", buffering=1)

        def trace_training_stage(
            *,
            step: int,
            stage: str,
            scene_key: str,
            information_set: TaskAddressActionInformationSet,
            synchronize: bool = False,
        ) -> None:
            if trace_handle is None:
                return
            if synchronize:
                torch.cuda.synchronize(device)
            trace_handle.write(
                _canonical_json(
                    {
                        "schema": "picf-next.ltop-g3-smoke-stage-trace.v1",
                        "rank": rank,
                        "step": step,
                        "stage": stage,
                        "scene_key": scene_key,
                        "arm": information_set.value,
                        "monotonic_s": time.perf_counter(),
                    }
                )
                + "\n"
            )
            trace_handle.flush()
            os.fsync(trace_handle.fileno())

        if args.phase == "combined" and args.mode != "smoke":
            record(0)
        train_started = time.perf_counter()
        training_steps = 0 if args.phase in {"evaluation", "retention"} else args.steps
        for step in range(1, training_steps + 1):
            if optimizer is None:
                raise AssertionError("LTOP G3 training phase omitted its optimizer")
            optimizer.zero_grad(set_to_none=True)
            schedule_entry = (
                mediator_schedule["entries"][step - 1] if mediator_schedule is not None else None
            )
            if schedule_entry is None:
                scene_index = (step - 1) % len(scenes["validation"])
                action_information_set = TaskAddressActionInformationSet.FACTUAL
            else:
                scene_index = int(schedule_entry["scene_index"])
                action_information_set = TaskAddressActionInformationSet(schedule_entry["arm"])
            scene = scenes["validation"][scene_index]
            if schedule_entry is not None and (
                schedule_entry["scene_key"] != scene["item"]["item_id"]
                or schedule_entry["source_task_key"] != scene["source_task_key"]
            ):
                raise RuntimeError("LTOP G3 mediator schedule differs from the materialized scene")
            supervision = scene["source_action_supervision"]
            require_factual_action_supervision(supervision)
            trace_training_stage(
                step=step,
                stage="batch-begin",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            batch = batch_to_device(scene["source_batch"])
            trace_training_stage(
                step=step,
                stage="prior-begin",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            prior = build_prior(scene["source_batch"])
            action_information_sets = (action_information_set,) * batch.routing.batch_size
            trace_training_stage(
                step=step,
                stage="forward-begin",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            result, row_mass = training_forward(
                batch,
                prior,
                require_grad=True,
                action_information_sets=action_information_sets,
            )
            trace_training_stage(
                step=step,
                stage="forward-done",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            physical = physical_supervision(
                context=result.context,
                batch=batch,
                target_identity=scene["source_target_identity"],
                allow_unobservable_target=True,
            )
            trace_training_stage(
                step=step,
                stage="physical-done",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            target_row = physical["target_row"]
            address = (
                task_address_target_coverage(
                    row_mass,
                    torch.tensor([target_row], dtype=torch.long, device=device),
                )
                if target_row is not None
                else None
            )
            address_loss = row_mass.sum() * 0.0 if address is None else address.loss
            loss = (
                args.official_loss_weight * result.official_total_loss
                + args.physical_set_weight * physical["set_loss"].total
                + args.task_address_weight * address_loss
            )
            trace_training_stage(
                step=step,
                stage="backward-begin",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            loss.backward()
            trace_training_stage(
                step=step,
                stage="backward-done",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            metrics = _distributed_gradient_metrics(
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
            trace_training_stage(
                step=step,
                stage="gradient-metrics-done",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            finite = bool(metrics["all_finite"])
            all_gradients_finite &= finite
            if not finite:
                raise FloatingPointError("LTOP G3 produced a non-finite gradient")
            gradient_metrics_history.append(metrics)
            clip_lingbot_distributed_l2_grad_norm_(
                tuple(policy.parameters()),
                args.maximum_grad_norm,
                device=device,
                dist_module=dist,
                torch_module=torch,
                error_if_nonfinite=True,
            )
            trace_training_stage(
                step=step,
                stage="gradient-clip-done",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            optimizer.step()
            trace_training_stage(
                step=step,
                stage="optimizer-done",
                scene_key=scene["item"]["item_id"],
                information_set=action_information_set,
                synchronize=True,
            )
            action_loss_value = float(result.official_action_loss.detach().float().item())
            total_loss_value = float(loss.detach().float().item())
            physical_loss_value = float(physical["set_loss"].total.detach().float().item())
            address_loss_value = float(address_loss.detach().float().item())
            action_losses.append(action_loss_value)
            total_losses.append(total_loss_value)
            physical_losses.append(physical_loss_value)
            address_losses.append(address_loss_value)
            action_information_set_counts[action_information_set.value] += 1
            arm_record = {
                "global_step": step,
                "cycle_index": (None if schedule_entry is None else schedule_entry["cycle_index"]),
                "scene_index": scene_index,
                "scene_key": scene["item"]["item_id"],
                "prompt_index": 0,
                "prompt_key": f"source-task/{scene['source_task_key']}",
                "arm": action_information_set.value,
            }
            action_information_set_history.append(arm_record)
            action_supervision_history.append(supervision.to_dict())
            task_address_supervision_history.append(
                {
                    "global_step": step,
                    "scene_key": scene["item"]["item_id"],
                    "source_task_key": scene["source_task_key"],
                    "source_target_identity": scene["source_target_identity"],
                    "enabled": target_row is not None,
                    "reason": physical["target_row_reason"],
                }
            )
            if journal_handle is None:
                raise RuntimeError("LTOP G3 training journal handle is absent")
            journal_handle.write(
                _canonical_json(
                    {
                        "schema": G3_ARM_JOURNAL_SCHEMA,
                        "rank": rank,
                        **arm_record,
                        "schedule_sha256": (
                            None if mediator_schedule is None else mediator_schedule["sha256"]
                        ),
                        "sample_keys": list(batch.routing.episode_keys),
                        "action_loss": action_loss_value,
                        "total_loss": total_loss_value,
                        "physical_set_loss": physical_loss_value,
                        "task_address_loss": address_loss_value,
                        "source_task_key": scene["source_task_key"],
                        "source_target_identity": scene["source_target_identity"],
                        "task_address_supervision_enabled": target_row is not None,
                        "task_address_supervision_reason": physical["target_row_reason"],
                        "action_supervision": supervision.to_dict(),
                    }
                )
                + "\n"
            )
            if step % args.progress_every == 0 or step == args.steps:
                journal_handle.flush()
                os.fsync(journal_handle.fileno())
            if args.phase == "combined" and (step % args.eval_every == 0 or step == args.steps):
                record(step)
            del address, batch, loss, physical, prior, result, row_mass
            if (
                rank == 0
                and args.progress_output is not None
                and (step % args.progress_every == 0 or step == args.steps)
            ):
                elapsed = time.perf_counter() - train_started
                mean_step = elapsed / step
                _write_json_atomic_replace(
                    args.progress_output,
                    {
                        "schema": "picf-next.ltop-g3-progress.v1",
                        "completed_steps": step,
                        "total_steps": args.steps,
                        "elapsed_s": elapsed,
                        "mean_elapsed_per_completed_step_s": mean_step,
                        "estimated_remaining_s": mean_step * (args.steps - step),
                        "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                        "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
                        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                        "action_information_set_counts": action_information_set_counts,
                        "schedule_sha256": (
                            None if mediator_schedule is None else mediator_schedule["sha256"]
                        ),
                        "updated_unix_s": time.time(),
                    },
                )

        if trace_handle is not None:
            trace_handle.flush()
            os.fsync(trace_handle.fileno())
            trace_handle.close()
            if trace_path is None:
                raise AssertionError("G3 smoke trace path disappeared")
            _fsync_directory(trace_path.parent)

        journal_receipt = None
        if journal_handle is not None:
            journal_handle.flush()
            os.fsync(journal_handle.fileno())
            journal_handle.close()
            if journal_path is None:
                raise AssertionError("G3 journal path disappeared")
            _fsync_directory(journal_path.parent)
            with journal_path.open("r", encoding="ascii") as stream:
                record_count = sum(1 for line in stream if line.rstrip("\n"))
            if record_count != training_steps:
                raise RuntimeError("LTOP G3 rank journal record count differs from training steps")
            journal_receipt = {
                "schema": "picf-next.ltop-g3-arm-journal-receipt.v1",
                "rank": rank,
                "path": str(journal_path.resolve()),
                "file_sha256": _file_sha256(journal_path),
                "record_count": record_count,
            }

        if args.phase == "evaluation":
            record(args.steps)
        elif args.phase == "retention":
            history.append(
                {
                    "step": args.steps,
                    "validation": evaluate_retention_partition("validation"),
                    "heldout": evaluate_retention_partition("heldout"),
                }
            )

        torch.cuda.synchronize(device)
        duration = time.perf_counter() - train_started
        joint_host.attention_interface = original_attention_interface
        if optimizer is not None:
            final_optimizer_manifest = audit_native_optimizer_coverage(
                modules={"policy": policy},
                optimizer=optimizer,
            )
            if final_optimizer_manifest != optimizer_manifest:
                raise RuntimeError("LTOP G3 optimizer ownership changed during execution")
        post_evaluation_model_local_state_sha256 = (
            _distributed_rank_local_call(
                action=lambda: _model_local_state_digest(policy, torch),
                phase="ltop-g3-staged-evaluation-post-forward-model-digest",
                rank=rank,
                dist_module=dist,
            )
            if args.phase in {"evaluation", "retention"}
            else None
        )
        if (
            post_evaluation_model_local_state_sha256 is not None
            and post_evaluation_model_local_state_sha256 != cold_loaded_model_local_state_sha256
        ):
            raise RuntimeError("G3 cold evaluation mutated persistent model state")
        training_final_model_local_state_sha256 = (
            _distributed_rank_local_call(
                action=lambda: _model_local_state_digest(policy, torch),
                phase="ltop-g3-training-terminal-model-digest",
                rank=rank,
                dist_module=dist,
            )
            if args.phase == "training"
            else None
        )
        checkpoint_report: dict[str, Any] | None = None
        if args.phase == "training":
            if args.checkpoint_output is None:
                raise AssertionError("validated G3 training checkpoint output disappeared")
            checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
            checkpoint_output = args.checkpoint_output.resolve()
            checkpoint_staging = checkpoint_output.with_name(
                f".{checkpoint_output.name}.incomplete"
            )
            checkpoint_error: list[str | None] = [None]
            if rank == 0:
                try:
                    checkpoint_output.parent.mkdir(parents=True, exist_ok=True)
                    if checkpoint_output.exists() or checkpoint_output.is_symlink():
                        raise FileExistsError(checkpoint_output)
                    if checkpoint_staging.is_symlink():
                        raise ValueError("G3 checkpoint staging path cannot be a symbolic link")
                    if checkpoint_staging.exists():
                        if not checkpoint_staging.is_dir():
                            raise ValueError("G3 checkpoint staging path is not a directory")
                        shutil.rmtree(checkpoint_staging)
                except BaseException as error:
                    checkpoint_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(checkpoint_error, src=0)
            if checkpoint_error[0] is not None:
                raise RuntimeError(f"LTOP G3 checkpoint preflight failed: {checkpoint_error[0]}")
            gathered_training_digests: list[Any] = [None] * G2_WORLD_SIZE
            dist.all_gather_object(
                gathered_training_digests,
                {
                    "rank": rank,
                    "sha256": training_final_model_local_state_sha256,
                    "action_information_set_counts": dict(action_information_set_counts),
                },
            )
            ordered_training_states = sorted(
                gathered_training_digests,
                key=lambda item: item["rank"],
            )
            if [item.get("rank") for item in ordered_training_states] != list(range(G2_WORLD_SIZE)):
                raise RuntimeError("G3 training terminal rank state exchange is incomplete")
            ordered_training_digests = [item["sha256"] for item in ordered_training_states]
            if len(ordered_training_digests) != G2_WORLD_SIZE or any(
                not isinstance(value, str) or len(value) != 64 for value in ordered_training_digests
            ):
                raise RuntimeError("G3 training terminal model digest exchange is incomplete")
            checkpointer.save(
                str(checkpoint_staging),
                {"model": policy},
                global_steps=None,
            )
            post_checkpoint_save_model_local_state_sha256 = _distributed_rank_local_call(
                action=lambda: _model_local_state_digest(policy, torch),
                phase="ltop-g3-training-post-checkpoint-save-model-digest",
                rank=rank,
                dist_module=dist,
            )
            if (
                post_checkpoint_save_model_local_state_sha256
                != training_final_model_local_state_sha256
            ):
                raise RuntimeError("G3 DCP save mutated the training terminal model state")
            dist.barrier()
            checkpoint_publication: list[Any] = [None]
            if rank == 0:
                try:
                    model_tree_sha256 = directory_tree_sha256(
                        checkpoint_staging / "model",
                        schema=G3_MODEL_TREE_SCHEMA,
                    )
                    manifest = {
                        "schema": G3_TRAINING_CHECKPOINT_SCHEMA,
                        "status": "PASS",
                        "global_step": args.steps,
                        "optimizer_saved": False,
                        "format": G3_MODEL_ONLY_CHECKPOINT_FORMAT,
                        "world_size": G2_WORLD_SIZE,
                        "model_tree_schema": G3_MODEL_TREE_SCHEMA,
                        "model_tree_sha256": model_tree_sha256,
                        "action_supervision_schema": TASK_ACTION_SUPERVISION_SCHEMA,
                        "picf_source_contract": picf_source_contract,
                        "task_address_supervision_depth": (
                            action_consumable_task_address_depth_contract(graph.config.num_layers)
                        ),
                        "training_final_model_local_state_sha256_by_rank": (
                            ordered_training_digests
                        ),
                        "action_information_set_schedule_sha256": (
                            None if mediator_schedule is None else mediator_schedule["sha256"]
                        ),
                        "action_information_set_counts_by_rank": [
                            item["action_information_set_counts"]
                            for item in ordered_training_states
                        ],
                        "source_stage_checkpoint": str(args.stage_checkpoint.resolve()),
                        "g2_report_sha256": stage_contract.g2_report_sha256,
                        "runtime_source_contract": ltop_stage_runtime_source_contract(
                            stage_contract
                        ),
                    }
                    write_text_durable_exclusive(
                        checkpoint_staging / "ltop_g3_training_checkpoint.json",
                        _canonical_json(manifest) + "\n",
                    )
                    manifest_sha256 = _file_sha256(
                        checkpoint_staging / "ltop_g3_training_checkpoint.json"
                    )
                    _fsync_tree(checkpoint_staging)
                    publish_prepared_directory_durable_exclusive(
                        checkpoint_staging,
                        checkpoint_output,
                    )
                    checkpoint_publication[0] = {
                        "path": str(checkpoint_output),
                        "format": G3_MODEL_ONLY_CHECKPOINT_FORMAT,
                        "optimizer_saved": False,
                        "manifest_sha256": manifest_sha256,
                        "model_tree_schema": G3_MODEL_TREE_SCHEMA,
                        "model_tree_sha256": model_tree_sha256,
                        "action_supervision_schema": TASK_ACTION_SUPERVISION_SCHEMA,
                        "picf_source_contract": picf_source_contract,
                        "task_address_supervision_depth": (
                            action_consumable_task_address_depth_contract(graph.config.num_layers)
                        ),
                        "training_final_model_local_state_sha256_by_rank": (
                            ordered_training_digests
                        ),
                    }
                except BaseException as error:
                    checkpoint_publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(checkpoint_publication, src=0)
            if (
                not isinstance(checkpoint_publication[0], dict)
                or "error" in checkpoint_publication[0]
            ):
                raise RuntimeError(
                    f"LTOP G3 checkpoint publication failed: {checkpoint_publication[0]}"
                )
            checkpoint_report = checkpoint_publication[0]
            dist.barrier()
        rank_report = {
            "rank": rank,
            "history": history,
            "action_losses": action_losses,
            "total_losses": total_losses,
            "physical_losses": physical_losses,
            "task_address_losses": address_losses,
            "action_information_set_history": action_information_set_history,
            "action_information_set_counts": action_information_set_counts,
            "action_supervision_history": action_supervision_history,
            "action_supervision_schema": TASK_ACTION_SUPERVISION_SCHEMA,
            "task_address_supervision_history": task_address_supervision_history,
            "action_information_set_schedule_sha256": (
                None if mediator_schedule is None else mediator_schedule["sha256"]
            ),
            "arm_journal": journal_receipt,
            "gradient_metrics_history": gradient_metrics_history,
            "all_gradients_finite": all_gradients_finite,
            "runtime_schedule_sha256": runtime_schedule["sha256"],
            "cuda_allocator": args.cuda_allocator,
            "optimizer_parameter_manifest": (
                None if optimizer_manifest is None else asdict(optimizer_manifest)
            ),
            "training_final_model_local_state_sha256": (training_final_model_local_state_sha256),
            "post_checkpoint_save_model_local_state_sha256": (
                post_checkpoint_save_model_local_state_sha256 if args.phase == "training" else None
            ),
            "cold_loaded_model_local_state_sha256": (cold_loaded_model_local_state_sha256),
            "post_evaluation_model_local_state_sha256": (post_evaluation_model_local_state_sha256),
            "trained_checkpoint_model_tree_sha256": (trained_checkpoint_model_tree_sha256),
            "trained_model_local_state_sha256": cold_loaded_model_local_state_sha256,
            "stage_restore": runtime.rank_report(),
            "timings": {
                "train_and_eval_s": duration,
                "mean_optimizer_step_including_eval_s": (
                    None if training_steps == 0 else duration / training_steps
                ),
            },
            "cuda_memory_bytes": {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            },
        }
        gathered: list[dict[str, Any] | None] = [None] * G2_WORLD_SIZE
        dist.all_gather_object(gathered, rank_report)
        outcome: list[object] = [None, None]
        if rank == 0:
            rank_reports = sorted(
                (value for value in gathered if value is not None),
                key=lambda value: value["rank"],
            )
            if args.phase == "training":
                failures = _training_failures(rank_reports, mode=args.mode)
            elif args.phase == "evaluation":
                failures = _evaluation_failures(rank_reports, mode=args.mode)
            elif args.phase == "retention":
                failures = _retention_failures(rank_reports)
            else:
                failures = _gate_failures(rank_reports, mode=args.mode)
            report = {
                "schema": (
                    G3_TRAINING_SCHEMA
                    if args.phase == "training"
                    else G3_EVALUATION_SCHEMA
                    if args.phase == "evaluation"
                    else G3_RETENTION_SCHEMA
                    if args.phase == "retention"
                    else G3_SCHEMA
                ),
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
                "mode": args.mode,
                "phase": args.phase,
                "architecture_identity": G2_ARCHITECTURE,
                "world_size": G2_WORLD_SIZE,
                "steps": args.steps,
                "eval_every": args.eval_every,
                "seed": args.seed,
                "capacity": args.capacity,
                "task_query_count": args.task_query_count,
                "stage_checkpoint": str(args.stage_checkpoint.resolve()),
                "trained_checkpoint": (
                    None
                    if args.trained_checkpoint is None
                    else str(args.trained_checkpoint.resolve())
                ),
                "evaluation_action_information_set": (
                    args.evaluation_action_information_set if args.phase == "evaluation" else None
                ),
                "evaluation_scenes_per_partition": (
                    args.evaluation_scenes_per_partition if args.phase == "evaluation" else None
                ),
                "evaluation_scope": (
                    ("quick" if args.evaluation_scenes_per_partition == 1 else "full")
                    if args.phase == "evaluation"
                    else None
                ),
                "g2_report_sha256": stage_contract.g2_report_sha256,
                "runtime_source_contract": ltop_stage_runtime_source_contract(stage_contract),
                "picf_source_contract": picf_source_contract,
                "dataset_contract": dataset_contract,
                "execution_contract_sha256": _sha256(args.execution_contract),
                "offline_labels_sha256": _sha256(args.offline_labels),
                "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
                "action_inference_contract": (
                    None
                    if args.phase == "retention"
                    else {
                        "surface": "policy.sample_actions",
                        "fixed_noise": True,
                        "released_denoise_steps": LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
                        "arms": (
                            "factual/repeat/every-row removal/blocked/every-row blocked removal"
                        ),
                        "labels_opened_after_all_forward_receipts": True,
                        "moe_backend": moe_inference_backend,
                    }
                ),
                "representation_retention_contract": (
                    {
                        "purpose": "diagnose representation forgetting versus action bypass",
                        "optimizer_updates": 0,
                        "scenes_per_rank_per_partition": 4,
                        "crossed_prompts_per_scene": 2,
                        "reference": "accepted G2b full-scene representation gate",
                        "scientific_action_evidence": False,
                    }
                    if args.phase == "retention"
                    else None
                ),
                "training_contract": {
                    "optimizer": runtime.optimizer_contract.metadata,
                    "fresh_optimizer_after_strict_model_only_restore": True,
                    "deploy_time_module_added": False,
                    "action_supervision": {
                        "schema": TASK_ACTION_SUPERVISION_SCHEMA,
                        "official_action_loss": "immutable-source-task-action-pairs-only",
                        "crossed_prompt_action_loss": False,
                        "crossed_prompts": "representation-and-causal-evaluation-only",
                        "ambiguous_source_task_address_loss": False,
                        "unobservable_source_target_address_loss": False,
                        "unobservable_source_target_policy": (
                            "disable-address-only-with-explicit-loss-side-receipt"
                        ),
                    },
                    "task_address_supervision_depth": (
                        action_consumable_task_address_depth_contract(graph.config.num_layers)
                    ),
                    "action_information_set_trial": {
                        "single_forward_per_optimizer_step": True,
                        "schedule": mediator_schedule,
                        "evaluation_intervention_enum_is_separate": True,
                    },
                    "loss_weights": {
                        "official": args.official_loss_weight,
                        "physical_set": args.physical_set_weight,
                        "task_address": args.task_address_weight,
                    },
                },
                "checkpoint": checkpoint_report,
                "thresholds": (
                    {
                        "validation_positive_prompt_margins_global_minimum": 12,
                        "validation_mean_margin_minimum": 0.02,
                        "heldout_positive_prompt_margins_global_minimum": 10,
                        "heldout_mean_margin_strictly_positive": True,
                        "physical_prompt_drift_max_abs": 1.0e-5,
                        "shared_row_gauge_required": True,
                    }
                    if args.phase == "retention"
                    else {
                        "bitwise_factual_replay": True,
                        "action_loss_improvement_ratio_maximum": 0.95,
                        "mean_factual_target_minus_distractor_strictly_positive": True,
                        "mean_blocked_path_did_strictly_positive": True,
                        "positive_sample_fraction_minimum": 0.625,
                    }
                ),
                "rank_reports": rank_reports,
            }
            if args.phase == "retention":
                report["scene_level_robustness"] = {
                    partition: _scene_level_robustness(
                        rank_reports,
                        partition=partition,
                        seed=args.seed + offset,
                    )
                    for offset, partition in enumerate(("validation", "heldout"))
                }
            write_text_durable_exclusive(args.output, _canonical_json(report) + "\n")
            outcome[:] = [report["status"], report["failures"]]
        dist.broadcast_object_list(outcome, src=0)
        if outcome[0] != "PASS":
            raise RuntimeError(f"LTOP G3 failed: {outcome[1]}")


if __name__ == "__main__":
    main()

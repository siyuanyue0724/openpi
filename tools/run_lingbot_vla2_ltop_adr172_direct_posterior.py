#!/usr/bin/env python3
# ruff: noqa: E402, I001
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the bounded two-GPU ADR172 direct action-to-posterior gate.

The runner preserves the already exercised CALVIN, FSDP2, optimizer,
checkpoint, and journal shell from ADR170.  Its scientific route is different:
scene evidence reaches the released action suffix only through current
POSTERIOR rows, and a parameter-free receipt supervises native action attention
to the loss-side physical target row.  It adds no selector or deploy-time head.
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
from picf_next.lingbot_native.action_posterior_collector import (
    RegisteredActionPosteriorReceiptCollector,
)
from picf_next.lingbot_native.action_posterior_learning import (
    aggregate_action_posterior_distribution,
    action_posterior_target_mass_loss,
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


G3_SCHEMA = "picf-next.adr172-direct-action-posterior.v1"
G3_TRAINING_SCHEMA = "picf-next.adr172-direct-action-posterior-training.v1"
G3_EVALUATION_SCHEMA = "picf-next.adr172-direct-action-posterior-evaluation.v1"
G3_RETENTION_SCHEMA = "picf-next.adr172-direct-action-posterior-retention.v1"
G3_MODES = ("smoke", "gate", "direct-trial")
G3_PHASES = ("combined", "training", "evaluation", "retention")
G3_DEFAULT_STEPS = 128
G3_DEFAULT_EVAL_EVERY = 32
G3_DIRECT_TRIAL_STEPS = 256
G3_DIRECT_TRIAL_EVAL_EVERY = 32
G3_SOURCE_ACTION_SCHEDULE_SCHEMA = "picf-next.adr172-direct-source-schedule.v1"
G3_ROUTE_JOURNAL_SCHEMA = "picf-next.adr172-direct-route-step.v1"
G3_DIRECT_ROUTE = "direct-posterior-required"
G3_EVALUATION_SCENES_PER_PARTITION = 4
G3_TRAINING_CHECKPOINT_SCHEMA = "picf-next.adr172-direct-posterior-training-checkpoint.v1"
G3_MODEL_TREE_SCHEMA = "picf-next.ltop-g3-model-dcp-tree.v1"
G3_MODEL_ONLY_CHECKPOINT_FORMAT = "lingbot-fsdp2-dcp-model-only"
G3_DIRECT_ACTION_CAUSAL_SURFACE = "native-action-to-current-posterior-row-kv"
G3_PICF_SOURCE_CONTRACT_SCHEMA = "picf-next.g3-picf-source-contract.v1"
G3_COLD_POSITIVE_SCENE_FRACTION = 0.75
G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE = 1.0e-6
G3_PHYSICAL_PROMPT_DRIFT_MAX_ABS = 1.0e-5
G3_PHYSICAL_SET_LOSS_COMPONENTS = (
    "mask_focal",
    "mask_dice",
    "existence_focal",
    "ownership_nll",
)
G3_PICF_CRITICAL_SOURCE_FILES = (
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
ADR172_REGISTERED_LAYER_OFFSETS = (-4, -1)
ADR172_ACTION_HEAD_SCOPE = "all-action-heads"
ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE = "guidedvla-fixed-object-heads-0-1"
ADR172_ACTION_HEAD_SCOPES = (
    ADR172_ACTION_HEAD_SCOPE,
    ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE,
)
ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES = (0, 1)
ADR172_DIRECT_GROUNDING_WEIGHT_BY_HEAD_SCOPE = {
    ADR172_ACTION_HEAD_SCOPE: 1.0,
    ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE: 0.001,
}
ADR172_GUIDEDVLA_UPSTREAM_CONTRACT = {
    "repository": "GuidedVLA",
    "repository_commit": "04be059e0d6bd448be5cb45fdbafc775f7eb5e38",
    "config_name": "pi0_libero_object_depth_skill",
    "object_use_control": False,
    "object_head_indices": list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES),
    "object_loss_head_aggregation": "mean_heads",
    "object_loss_weight": 0.001,
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
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_GIT_OBJECT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


def _direct_posterior_registered_layer_indices(layer_count: int) -> tuple[int, ...]:
    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("ADR172 direct-posterior host layer count must be positive")
    indices = tuple(layer_count + offset for offset in ADR172_REGISTERED_LAYER_OFFSETS)
    if any(not 0 <= layer < layer_count for layer in indices):
        raise ValueError("ADR172 registered layer offsets are outside the shared host")
    return indices


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


def _validate_runtime_source_contract(value: object) -> dict[str, object]:
    expected_fields = {
        "native_patch_sha256",
        "runtime_hotfix_sha256",
        "runtime_patched_source_sha256",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise ValueError("G3 checkpoint runtime source contract fields differ")
    native_patch = value.get("native_patch_sha256")
    if not isinstance(native_patch, str) or not _SHA256_PATTERN.fullmatch(native_patch):
        raise ValueError("G3 checkpoint native patch SHA-256 is malformed")
    runtime_hotfix = value.get("runtime_hotfix_sha256")
    if runtime_hotfix is not None and (
        not isinstance(runtime_hotfix, str) or not _SHA256_PATTERN.fullmatch(runtime_hotfix)
    ):
        raise ValueError("G3 checkpoint runtime hotfix SHA-256 is malformed")
    patched_sources = value.get("runtime_patched_source_sha256")
    if (
        not isinstance(patched_sources, dict)
        or not patched_sources
        or any(not isinstance(path, str) or not path for path in patched_sources)
        or any(
            not isinstance(digest, str) or not _SHA256_PATTERN.fullmatch(digest)
            for digest in patched_sources.values()
        )
    ):
        raise ValueError("G3 checkpoint runtime patched-source digests are malformed")
    return value


def _validate_g3_training_checkpoint_manifest(
    manifest: object,
    *,
    expected_layer_count: int | None = None,
    expected_head_scope: str = ADR172_ACTION_HEAD_SCOPE,
    expected_picf_source_contract: dict[str, object] | None = None,
    expected_source_stage_checkpoint: Path | str | None = None,
    expected_g2_report_sha256: str | None = None,
    expected_runtime_source_contract: dict[str, object] | None = None,
) -> tuple[list[str], str]:
    """Validate the cheap, content-addressed portion of the cold-load ABI."""

    if not isinstance(manifest, dict):
        raise ValueError("G3 trained checkpoint manifest must be a JSON object")
    expected = {
        "schema": G3_TRAINING_CHECKPOINT_SCHEMA,
        "status": "PASS",
        "global_step": G3_DIRECT_TRIAL_STEPS,
        "optimizer_saved": False,
        "format": G3_MODEL_ONLY_CHECKPOINT_FORMAT,
        "world_size": G2_WORLD_SIZE,
        "model_tree_schema": G3_MODEL_TREE_SCHEMA,
        "action_supervision_schema": TASK_ACTION_SUPERVISION_SCHEMA,
        "direct_action_causal_surface": G3_DIRECT_ACTION_CAUSAL_SURFACE,
        "direct_route": G3_DIRECT_ROUTE,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"G3 trained checkpoint manifest violates {field}")
    if "task_address_supervision_depth" in manifest:
        raise ValueError("ADR172 checkpoint carries the rejected two-hop depth contract")
    if any(
        field in manifest
        for field in (
            "action_information_set_schedule_sha256",
            "action_information_set_counts_by_rank",
        )
    ):
        raise ValueError("ADR172 checkpoint carries rejected action-information route metadata")
    registered_layers = manifest.get("direct_posterior_registered_layer_indices")
    if (
        not isinstance(registered_layers, list)
        or len(registered_layers) != len(ADR172_REGISTERED_LAYER_OFFSETS)
        or any(
            isinstance(layer, bool) or not isinstance(layer, int) or layer < 0
            for layer in registered_layers
        )
        or registered_layers != sorted(set(registered_layers))
    ):
        raise ValueError("G3 checkpoint direct-posterior registered layers are malformed")
    if expected_head_scope not in ADR172_ACTION_HEAD_SCOPES:
        raise ValueError("G3 expected direct-posterior head scope is unknown")
    if manifest.get("direct_posterior_head_scope") != expected_head_scope:
        raise ValueError("G3 checkpoint direct-posterior head scope differs")
    expected_indices = (
        None
        if expected_head_scope == ADR172_ACTION_HEAD_SCOPE
        else list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES)
    )
    if manifest.get("direct_posterior_head_indices") not in (None, expected_indices):
        raise ValueError("G3 checkpoint direct-posterior head indices differ")
    expected_weight = ADR172_DIRECT_GROUNDING_WEIGHT_BY_HEAD_SCOPE[expected_head_scope]
    serialized_weight = manifest.get("direct_grounding_weight")
    if serialized_weight is not None and serialized_weight != expected_weight:
        raise ValueError("G3 checkpoint direct-grounding weight differs")
    expected_upstream = _direct_grounding_upstream_contract(expected_head_scope)
    serialized_upstream = manifest.get("direct_grounding_upstream_contract")
    if expected_head_scope == ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE:
        if manifest.get("direct_posterior_head_indices") != expected_indices:
            raise ValueError("G3 GuidedVLA checkpoint omits fixed object-head indices")
        if serialized_weight != expected_weight:
            raise ValueError("G3 GuidedVLA checkpoint omits its registered grounding weight")
        if serialized_upstream != expected_upstream:
            raise ValueError("G3 GuidedVLA checkpoint upstream contract differs")
    elif serialized_upstream is not None:
        raise ValueError("G3 all-head checkpoint carries an unexpected upstream contract")
    if expected_layer_count is not None and registered_layers != list(
        _direct_posterior_registered_layer_indices(expected_layer_count)
    ):
        raise ValueError("G3 checkpoint direct-posterior layers differ from the loaded host graph")
    source_contract = _validate_picf_source_contract(manifest.get("picf_source_contract"))
    if (
        expected_picf_source_contract is not None
        and source_contract != expected_picf_source_contract
    ):
        raise ValueError("G3 checkpoint PICF source identity differs from the loaded runner")
    source_stage_checkpoint = manifest.get("source_stage_checkpoint")
    if (
        not isinstance(source_stage_checkpoint, str)
        or not source_stage_checkpoint
        or not Path(source_stage_checkpoint).is_absolute()
    ):
        raise ValueError("G3 checkpoint source stage checkpoint is malformed")
    if expected_source_stage_checkpoint is not None and source_stage_checkpoint != str(
        Path(expected_source_stage_checkpoint).resolve()
    ):
        raise ValueError("G3 checkpoint source stage checkpoint differs from the current stage")
    g2_report_sha256 = manifest.get("g2_report_sha256")
    if not isinstance(g2_report_sha256, str) or not _SHA256_PATTERN.fullmatch(g2_report_sha256):
        raise ValueError("G3 checkpoint G2 report SHA-256 is malformed")
    if expected_g2_report_sha256 is not None and g2_report_sha256 != expected_g2_report_sha256:
        raise ValueError("G3 checkpoint G2 report differs from the current accepted G2 report")
    runtime_source_contract = _validate_runtime_source_contract(
        manifest.get("runtime_source_contract")
    )
    if (
        expected_runtime_source_contract is not None
        and runtime_source_contract != expected_runtime_source_contract
    ):
        raise ValueError("G3 checkpoint runtime source contract differs from the current source")
    model_tree_sha256 = manifest.get("model_tree_sha256")
    if not isinstance(model_tree_sha256, str) or not _SHA256_PATTERN.fullmatch(model_tree_sha256):
        raise ValueError("G3 checkpoint model-tree SHA-256 is malformed")
    schedule_sha256 = manifest.get("direct_route_schedule_sha256")
    if not isinstance(schedule_sha256, str) or not _SHA256_PATTERN.fullmatch(schedule_sha256):
        raise ValueError("G3 checkpoint direct-route schedule SHA-256 is malformed")
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
    parser.add_argument("--steps", type=int, default=G3_DEFAULT_STEPS)
    parser.add_argument("--eval-every", type=int, default=G3_DEFAULT_EVAL_EVERY)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--capacity", type=int, default=G2_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G2_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--maximum-grad-norm", type=float, default=1.0)
    parser.add_argument("--physical-set-weight", type=float, default=1.0)
    parser.add_argument("--direct-grounding-weight", type=float, default=1.0)
    parser.add_argument(
        "--direct-posterior-head-scope",
        choices=ADR172_ACTION_HEAD_SCOPES,
        default=ADR172_ACTION_HEAD_SCOPE,
    )
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
            json.loads(checkpoint_manifest.read_text(encoding="ascii")),
            expected_head_scope=args.direct_posterior_head_scope,
            expected_source_stage_checkpoint=args.stage_checkpoint,
            expected_g2_report_sha256=_sha256(args.g2_report),
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
    if (
        args.phase == "evaluation"
        and args.mode == "gate"
        and args.evaluation_scenes_per_partition != 4
    ):
        raise ValueError("formal ADR172 cold evaluation requires four scenes per partition")
    if args.mode == "direct-trial" and (
        args.phase != "training"
        or args.steps != G3_DIRECT_TRIAL_STEPS
        or args.eval_every != G3_DIRECT_TRIAL_EVAL_EVERY
    ):
        raise ValueError("ADR172 direct trial requires the staged training-only 256/32 schedule")
    for name in (
        "maximum_grad_norm",
        "physical_set_weight",
        "direct_grounding_weight",
        "official_loss_weight",
    ):
        value = getattr(args, name)
        if not isinstance(value, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"LTOP G3 {name} must be finite and positive")
    expected_grounding_weight = ADR172_DIRECT_GROUNDING_WEIGHT_BY_HEAD_SCOPE[
        args.direct_posterior_head_scope
    ]
    if args.direct_grounding_weight != expected_grounding_weight:
        raise ValueError(
            "ADR172 direct grounding weight differs from its registered head-scope profile"
        )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def build_g3_direct_source_schedule(
    *,
    scene_source_keys: Sequence[tuple[str, str]],
    steps: int,
) -> dict[str, Any]:
    """Cycle immutable source task-action pairs under the one promoted direct route."""

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
        or steps % len(normalized)
    ):
        raise ValueError("ADR172 direct schedule must complete every scene cycle")

    entries: list[dict[str, Any]] = []
    cycle_steps = len(normalized)
    for step in range(1, steps + 1):
        zero_based = step - 1
        cycle_index, cycle_offset = divmod(zero_based, cycle_steps)
        scene_index = cycle_offset
        scene_key, source_task_key = normalized[scene_index]
        entries.append(
            {
                "global_step": step,
                "cycle_index": cycle_index,
                "cycle_offset": cycle_offset,
                "scene_index": scene_index,
                "scene_key": scene_key,
                "source_task_key": source_task_key,
                "route": G3_DIRECT_ROUTE,
            }
        )

    scene_counts = {
        scene: sum(entry["scene_key"] == scene for entry in entries) for scene, _task in normalized
    }
    payload: dict[str, Any] = {
        "schema": G3_SOURCE_ACTION_SCHEDULE_SCHEMA,
        "design": "source-task-action-scene-stratified-direct-posterior",
        "single_forward_per_optimizer_step": True,
        "action_labels": "immutable-source-trajectory-only",
        "crossed_prompts_used_for_action_loss": False,
        "route": G3_DIRECT_ROUTE,
        "steps": steps,
        "scene_count": len(normalized),
        "cycle_steps": cycle_steps,
        "scene_counts": scene_counts,
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
            tensor = value.detach().cpu()
            tensor = tensor.long() if name == "active_action_counts" else tensor.float()
            payload[name] = tensor.tolist()
    return payload


def _g2_physical_retention_reference(
    report: dict[str, Any],
) -> dict[int, dict[str, dict[str, Any]]]:
    """Extract paired physical-set endpoints from the accepted G2b report."""

    if report.get("status") != "PASS":
        raise ValueError("ADR172 requires an accepted G2b representation report")
    rank_reports = report.get("rank_reports")
    if not isinstance(rank_reports, list) or len(rank_reports) != G2_WORLD_SIZE:
        raise ValueError("G2b representation report omits one or more ranks")
    reference: dict[int, dict[str, dict[str, Any]]] = {}
    for rank_report in rank_reports:
        rank = rank_report.get("rank")
        history = rank_report.get("history")
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or rank in reference
            or not isinstance(history, list)
            or not history
        ):
            raise ValueError("G2b representation report has an invalid rank history")
        final = history[-1]
        if not isinstance(final, dict):
            raise ValueError("G2b representation report has an invalid final rank receipt")
        partition_reference: dict[str, dict[str, Any]] = {}
        for partition in ("validation", "heldout"):
            partition_report = final.get(partition)
            if not isinstance(partition_report, dict):
                raise ValueError("G2b physical-set endpoint is invalid")
            value = partition_report.get("mean_physical_set_loss")
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError("G2b physical-set endpoint is invalid")
            scenes = partition_report.get("scenes")
            if not isinstance(scenes, list) or len(scenes) != G3_EVALUATION_SCENES_PER_PARTITION:
                raise ValueError("G2b physical-set endpoint omits its paired scene axis")
            scene_reference: list[dict[str, Any]] = []
            scene_keys: set[tuple[str, str]] = set()
            component_presence: list[bool] = []
            for scene in scenes:
                if not isinstance(scene, dict):
                    raise ValueError("G2b physical-set scene endpoint is invalid")
                item_id = scene.get("item_id")
                sample_key = scene.get("sample_key")
                scene_loss = scene.get("mean_physical_set_loss")
                if (
                    not isinstance(item_id, str)
                    or not item_id
                    or not isinstance(sample_key, str)
                    or not sample_key
                    or isinstance(scene_loss, bool)
                    or not isinstance(scene_loss, int | float)
                    or not math.isfinite(float(scene_loss))
                    or float(scene_loss) < 0
                    or (item_id, sample_key) in scene_keys
                ):
                    raise ValueError("G2b physical-set scene endpoint is invalid")
                scene_keys.add((item_id, sample_key))
                components = scene.get("physical_set_loss_components")
                component_presence.append(components is not None)
                if components is not None and (
                    not isinstance(components, dict)
                    or set(components) != set(G3_PHYSICAL_SET_LOSS_COMPONENTS)
                    or any(
                        isinstance(component, bool)
                        or not isinstance(component, int | float)
                        or not math.isfinite(float(component))
                        or float(component) < 0
                        for component in components.values()
                    )
                ):
                    raise ValueError("G2b physical-set component endpoint is invalid")
                scene_reference.append(
                    {
                        "item_id": item_id,
                        "sample_key": sample_key,
                        "mean_physical_set_loss": float(scene_loss),
                        "physical_set_loss_components": (
                            None
                            if components is None
                            else {
                                name: float(components[name])
                                for name in G3_PHYSICAL_SET_LOSS_COMPONENTS
                            }
                        ),
                    }
                )
            if any(component_presence) and not all(component_presence):
                raise ValueError("G2b physical-set component endpoint is only partially published")
            scene_mean = _mean(
                [float(scene["mean_physical_set_loss"]) for scene in scene_reference]
            )
            if abs(float(value) - scene_mean) > G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE:
                raise ValueError("G2b physical-set partition mean differs from its scene endpoints")
            components_available = all(component_presence)
            partition_reference[partition] = {
                "mean_physical_set_loss": float(value),
                "scenes": scene_reference,
                "component_gate": {
                    "available_in_g2_reference": components_available,
                    "components": (
                        list(G3_PHYSICAL_SET_LOSS_COMPONENTS) if components_available else []
                    ),
                    "gap": (
                        None
                        if components_available
                        else (
                            "accepted G2b report does not publish per-scene physical loss "
                            "components"
                        )
                    ),
                },
            }
        reference[rank] = partition_reference
    if set(reference) != set(range(G2_WORLD_SIZE)):
        raise ValueError("G2b representation ranks are not the registered distributed axis")
    for partition in ("validation", "heldout"):
        scene_keys = [
            (scene["item_id"], scene["sample_key"])
            for rank in reference
            for scene in reference[rank][partition]["scenes"]
        ]
        if len(scene_keys) != len(set(scene_keys)):
            raise ValueError("G2b physical-set scene endpoints repeat across ranks")
        component_availability = {
            bool(reference[rank][partition]["component_gate"]["available_in_g2_reference"])
            for rank in reference
        }
        if len(component_availability) != 1:
            raise ValueError("G2b physical-set component endpoint differs across ranks")
    return reference


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


def _training_failures(
    rank_reports: list[dict[str, Any]],
    *,
    mode: str,
    head_scope: str = ADR172_ACTION_HEAD_SCOPE,
) -> list[str]:
    failures: list[str] = []
    expected_head_indices = (
        None
        if head_scope == ADR172_ACTION_HEAD_SCOPE
        else list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES)
    )
    if len(rank_reports) != G2_WORLD_SIZE:
        return ["ADR172 training phase omitted one or more distributed rank reports"]
    for report in rank_reports:
        rank = report["rank"]
        action_losses = report.get("action_losses")
        grounding_history = report.get("direct_grounding_history")
        grounding_losses = report.get("direct_grounding_losses")
        grounding_supervision = report.get("task_address_supervision_history")
        if not report["all_gradients_finite"]:
            failures.append(f"rank {rank}: non-finite gradients")
        if report["cuda_memory_bytes"]["peak_allocated"] >= 39 * 1024**3:
            failures.append(f"rank {rank}: peak allocated memory reached the A100 safety bound")
        required_gradients = (
            "native_graph_norm",
            "shared_host_norm",
            "shared_q_projection_norm",
            "shared_k_projection_norm",
            "action_output_norm",
        )
        if any(
            any(float(step.get(name, 0.0)) <= 0 for name in required_gradients)
            for step in report["gradient_metrics_history"]
        ):
            failures.append(f"rank {rank}: one required graph/Q/K/action gradient surface was zero")
        supervision = report.get("action_supervision_history")
        if not isinstance(action_losses, list) or not action_losses:
            failures.append(f"rank {rank}: action loss history is absent")
        if not isinstance(supervision, list) or len(supervision) != len(action_losses or ()):
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
        if (
            not isinstance(grounding_history, list)
            or not isinstance(grounding_losses, list)
            or not isinstance(grounding_supervision, list)
            or len(grounding_history) != len(action_losses or ())
            or len(grounding_losses) != len(grounding_history)
            or len(grounding_supervision) != len(grounding_history)
        ):
            failures.append(f"rank {rank}: direct grounding history is incomplete")
        else:
            allowed_mask_reasons = {
                "no-singleton-source-target",
                "unobservable-current-frame-target",
            }
            for item, grounding_loss, address in zip(
                grounding_history,
                grounding_losses,
                grounding_supervision,
                strict=True,
            ):
                layers = item.get("layers")
                registered_layers = item.get("registered_layer_indices")
                target_valid = item.get("target_valid")
                base_valid = (
                    isinstance(item, dict)
                    and item.get("head_scope") == head_scope
                    and item.get("head_indices") in (None, expected_head_indices)
                    and (
                        head_scope == ADR172_ACTION_HEAD_SCOPE
                        or item.get("head_indices") == expected_head_indices
                    )
                    and isinstance(registered_layers, list)
                    and bool(registered_layers)
                    and isinstance(layers, list)
                    and bool(layers)
                    and [layer.get("layer_index") for layer in layers] == registered_layers
                    and all(
                        math.isfinite(float(layer.get("total_posterior_mass_mean", math.nan)))
                        and float(layer.get("total_posterior_mass_mean", 0.0)) > 0.0
                        and math.isfinite(float(layer.get("target_mass_mean", math.nan)))
                        for layer in layers
                    )
                    and math.isfinite(float(grounding_loss))
                    and isinstance(address, dict)
                )
                visible_valid = (
                    target_valid is True
                    and item.get("target_row") is not None
                    and address.get("enabled") is True
                    and address.get("reason") == "bound-current-frame-target"
                    and float(grounding_loss) > 0.0
                    and all(float(layer["target_mass_mean"]) > 0.0 for layer in layers or ())
                )
                masked_valid = (
                    target_valid is False
                    and item.get("target_row") is None
                    and address.get("enabled") is False
                    and address.get("reason") in allowed_mask_reasons
                    and float(grounding_loss) == 0.0
                    and all(float(layer["target_mass_mean"]) == 0.0 for layer in layers or ())
                )
                if not base_valid or not (visible_valid or masked_valid):
                    failures.append(
                        f"rank {rank}: direct posterior receipt is invalid or inconsistently masked"
                    )
                    break
        journal = report.get("arm_journal")
        if not isinstance(journal, dict) or journal.get("record_count") != len(action_losses or ()):
            failures.append(f"rank {rank}: training journal is incomplete")
    if mode != "smoke":
        first_losses = [value for report in rank_reports for value in report["action_losses"][:16]]
        last_losses = [value for report in rank_reports for value in report["action_losses"][-16:]]
        if _mean(last_losses) >= 0.95 * _mean(first_losses):
            failures.append("ADR172 official action loss did not improve by at least five percent")
        first_grounding = [
            value for report in rank_reports for value in report["direct_grounding_losses"][:16]
        ]
        last_grounding = [
            value for report in rank_reports for value in report["direct_grounding_losses"][-16:]
        ]
        if _mean(last_grounding) >= _mean(first_grounding):
            failures.append("ADR172 direct grounding loss did not improve")
    return failures


def _cold_finite_float(value: object, *, nonnegative: bool = False) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    result = float(value)
    if not math.isfinite(result) or (nonnegative and result < 0.0):
        return None
    return result


def _direct_posterior_head_indices(
    head_scope: str,
    *,
    head_count: int,
) -> tuple[int, ...] | None:
    if head_scope == ADR172_ACTION_HEAD_SCOPE:
        return None
    if head_scope != ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE:
        raise ValueError(f"unknown ADR172 direct-posterior head scope: {head_scope!r}")
    if head_count <= max(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES):
        raise ValueError("native action attention omits one registered GuidedVLA object head")
    return ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES


def _direct_grounding_upstream_contract(head_scope: str) -> dict[str, object] | None:
    if head_scope == ADR172_ACTION_HEAD_SCOPE:
        return None
    if head_scope != ADR172_GUIDEDVLA_ACTION_HEAD_SCOPE:
        raise ValueError(f"unknown ADR172 direct-posterior head scope: {head_scope!r}")
    return json.loads(_canonical_json(ADR172_GUIDEDVLA_UPSTREAM_CONTRACT))


def _cold_float_vector(value: object, *, nonnegative: bool = False) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    result = [_cold_finite_float(item, nonnegative=nonnegative) for item in value]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result]


def _cold_causal_partition_evaluation(
    rank_reports: list[dict[str, Any]],
    *,
    partition: str,
    expected_scenes_per_rank: int,
    apply_scientific_gate: bool,
) -> dict[str, Any]:
    """Recompute the scene-level crossed-prompt posterior-action contract."""

    failures: list[str] = []
    scene_units: list[dict[str, Any]] = []
    seen_scene_keys: set[tuple[str, str]] = set()
    seen_sample_keys: set[str] = set()
    for report in rank_reports:
        rank = report.get("rank")
        history = report.get("history")
        if not isinstance(history, list) or len(history) != 1 or not isinstance(history[0], dict):
            failures.append(f"rank {rank}: staged evaluation did not publish exactly one receipt")
            continue
        partition_report = history[0].get(partition)
        if not isinstance(partition_report, dict):
            failures.append(f"rank {rank}: {partition} evaluation receipt is invalid")
            continue
        scenes = partition_report.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != expected_scenes_per_rank:
            failures.append(f"rank {rank}: {partition} expected {expected_scenes_per_rank} scenes")
            continue
        if partition_report.get("scene_count") != len(scenes):
            failures.append(f"rank {rank}: {partition} scene count differs from its evidence")
        if partition_report.get("prompt_count") != 2 * len(scenes):
            failures.append(f"rank {rank}: {partition} crossed-prompt axis is incomplete")
        for scene_index, scene in enumerate(scenes):
            context = f"rank {rank}: {partition} scene {scene_index}"
            if not isinstance(scene, dict):
                failures.append(f"{context}: evidence is invalid")
                continue
            item_id = scene.get("item_id")
            sample_key = scene.get("sample_key")
            if (
                not isinstance(item_id, str)
                or not item_id
                or not isinstance(sample_key, str)
                or not sample_key
            ):
                failures.append(f"{context}: item/sample identity is invalid")
                continue
            scene_key = (item_id, sample_key)
            if scene_key in seen_scene_keys or sample_key in seen_sample_keys:
                failures.append(f"{context}: sample identity repeats within {partition}")
                continue
            seen_scene_keys.add(scene_key)
            seen_sample_keys.add(sample_key)
            prompts = scene.get("prompts")
            if not isinstance(prompts, list) or len(prompts) != 2:
                failures.append(f"{context}: exactly two crossed prompts are required")
                continue
            first, second = prompts
            if not isinstance(first, dict) or not isinstance(second, dict):
                failures.append(f"{context}: crossed prompt evidence is invalid")
                continue
            if (
                first.get("target_identity") != second.get("matched_distractor_identity")
                or second.get("target_identity") != first.get("matched_distractor_identity")
                or first.get("target_row") != second.get("matched_distractor_row")
                or second.get("target_row") != first.get("matched_distractor_row")
            ):
                failures.append(f"{context}: crossed prompts do not reverse one canonical row pair")
            if scene.get("shared_row_gauge") is not True:
                failures.append(f"{context}: physical row gauge changed across prompts")
            score = scene.get("score")
            if not isinstance(score, dict):
                failures.append(f"{context}: scene score is invalid")
                continue
            if score.get("blocked_placebo_integrity_verified") is not True:
                failures.append(f"{context}: blocked-row placebo integrity was not verified")
            replay = _cold_float_vector(score.get("replay_floor_rms"), nonnegative=True)
            crossed = _cold_float_vector(score.get("crossed_prompt_target_selectivity"))
            normalized = _cold_float_vector(
                score.get("crossed_prompt_selectivity_over_all_posterior_block")
            )
            prompt_all_block = _cold_float_vector(
                score.get("prompt_mean_factual_all_posterior_block_effect_rms"),
                nonnegative=True,
            )
            if (
                replay is None
                or crossed is None
                or normalized is None
                or prompt_all_block is None
                or len(prompt_all_block) != 2
                or len(crossed) != len(normalized)
            ):
                failures.append(f"{context}: causal vectors are invalid")
                continue
            declared_sample_count = score.get("sample_count")
            if (
                isinstance(declared_sample_count, bool)
                or not isinstance(declared_sample_count, int)
                or declared_sample_count != len(crossed)
            ):
                failures.append(f"{context}: causal sample count differs from its vectors")
                continue
            max_replay = max(replay)
            mean_crossed = _mean(crossed)
            mean_normalized = _mean(normalized)
            minimum_all_block = min(prompt_all_block)
            declared_values = {
                "max_replay_floor_rms": max_replay,
                "mean_crossed_prompt_target_selectivity": mean_crossed,
                "mean_crossed_prompt_selectivity_over_all_posterior_block": mean_normalized,
                "minimum_prompt_factual_all_posterior_block_effect_rms": minimum_all_block,
            }
            for field, recomputed in declared_values.items():
                declared = _cold_finite_float(
                    score.get(field),
                    nonnegative=field
                    in {
                        "max_replay_floor_rms",
                        "minimum_prompt_factual_all_posterior_block_effect_rms",
                    },
                )
                if declared is None or not math.isclose(
                    declared,
                    recomputed,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-12,
                ):
                    failures.append(f"{context}: serialized {field} differs from raw evidence")
            positive_crossed_count = sum(value > 0.0 for value in crossed)
            if score.get("positive_crossed_prompt_target_selectivity_count") != (
                positive_crossed_count
            ):
                failures.append(f"{context}: positive crossed-prompt count differs")
            scene_units.append(
                {
                    "rank": rank,
                    "item_id": item_id,
                    "sample_key": sample_key,
                    "replay_floor_rms": max_replay,
                    "mean_crossed_prompt_target_selectivity": mean_crossed,
                    "mean_crossed_prompt_selectivity_over_all_posterior_block": (mean_normalized),
                    "minimum_prompt_factual_all_posterior_block_effect_rms": (minimum_all_block),
                    "positive_crossed_prompt": mean_crossed > 0.0,
                    "positive_normalized_crossed_prompt": mean_normalized > 0.0,
                    "positive_all_posterior_block": minimum_all_block > 0.0,
                    "joint_positive": (
                        mean_crossed > 0.0 and mean_normalized > 0.0 and minimum_all_block > 0.0
                    ),
                }
            )

    if not scene_units:
        failures.append(f"{partition}: no valid causal scene evidence")
        return {
            "partition": partition,
            "status": "FAIL",
            "failures": failures,
            "scene_count": 0,
            "scenes": [],
        }
    mean_crossed = _mean(
        [float(scene["mean_crossed_prompt_target_selectivity"]) for scene in scene_units]
    )
    mean_normalized = _mean(
        [
            float(scene["mean_crossed_prompt_selectivity_over_all_posterior_block"])
            for scene in scene_units
        ]
    )
    mean_all_block = _mean(
        [
            float(scene["minimum_prompt_factual_all_posterior_block_effect_rms"])
            for scene in scene_units
        ]
    )
    positive_crossed = sum(bool(scene["positive_crossed_prompt"]) for scene in scene_units)
    positive_normalized = sum(
        bool(scene["positive_normalized_crossed_prompt"]) for scene in scene_units
    )
    positive_all_block = sum(bool(scene["positive_all_posterior_block"]) for scene in scene_units)
    joint_positive = sum(bool(scene["joint_positive"]) for scene in scene_units)
    minimum_positive = math.ceil(G3_COLD_POSITIVE_SCENE_FRACTION * len(scene_units))
    max_replay = max(float(scene["replay_floor_rms"]) for scene in scene_units)
    if max_replay != 0.0:
        failures.append(f"{partition}: factual replay was not bitwise stable")
    if apply_scientific_gate:
        if mean_crossed <= 0.0:
            failures.append(f"{partition}: mean crossed-prompt row selectivity was nonpositive")
        if mean_normalized <= 0.0:
            failures.append(
                f"{partition}: normalized crossed-prompt row selectivity was nonpositive"
            )
        if mean_all_block <= 0.0:
            failures.append(f"{partition}: all-posterior block had no executable-action effect")
        if joint_positive < minimum_positive:
            failures.append(
                f"{partition}: jointly positive causal scenes {joint_positive} < {minimum_positive}"
            )
    return {
        "partition": partition,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "scene_count": len(scene_units),
        "positive_scene_fraction_minimum": G3_COLD_POSITIVE_SCENE_FRACTION,
        "minimum_positive_scene_count": minimum_positive,
        "positive_crossed_prompt_scene_count": positive_crossed,
        "positive_normalized_crossed_prompt_scene_count": positive_normalized,
        "positive_all_posterior_block_scene_count": positive_all_block,
        "joint_positive_scene_count": joint_positive,
        "mean_crossed_prompt_target_selectivity": mean_crossed,
        "mean_crossed_prompt_selectivity_over_all_posterior_block": mean_normalized,
        "mean_minimum_prompt_factual_all_posterior_block_effect_rms": mean_all_block,
        "max_replay_floor_rms": max_replay,
        "scenes": scene_units,
    }


def _evaluation_failures(rank_reports: list[dict[str, Any]], *, mode: str) -> list[str]:
    if len(rank_reports) != G2_WORLD_SIZE:
        return ["G3 evaluation phase omitted one or more distributed rank reports"]
    ranks = [report.get("rank") for report in rank_reports]
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks) or set(
        ranks
    ) != set(range(G2_WORLD_SIZE)):
        return ["G3 evaluation ranks differ from the distributed axis"]
    failures: list[str] = []
    for report in rank_reports:
        rank = report["rank"]
        if report.get("direct_action_causal_surface") != G3_DIRECT_ACTION_CAUSAL_SURFACE:
            failures.append(f"rank {rank}: evaluation did not use direct posterior-row action")
    partitions = ("validation",) if mode == "smoke" else ("validation", "heldout")
    expected_scenes = 1 if mode == "smoke" else G3_EVALUATION_SCENES_PER_PARTITION
    summaries = {
        partition: _cold_causal_partition_evaluation(
            rank_reports,
            partition=partition,
            expected_scenes_per_rank=expected_scenes,
            apply_scientific_gate=mode != "smoke",
        )
        for partition in partitions
    }
    for summary in summaries.values():
        failures.extend(summary["failures"])
    if mode != "smoke" and not failures:
        partition_samples = {
            partition: {scene["sample_key"] for scene in summary["scenes"]}
            for partition, summary in summaries.items()
        }
        overlap = partition_samples["validation"] & partition_samples["heldout"]
        if overlap:
            failures.append("G3 validation and heldout causal samples overlap")
    return failures


def _nonnegative_finite_float(value: object) -> float | None:
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or not math.isfinite(float(value))
        or float(value) < 0
    ):
        return None
    return float(value)


def _physical_scene_map(value: object) -> dict[tuple[str, str], dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    scenes: dict[tuple[str, str], dict[str, Any]] = {}
    for scene in value:
        if not isinstance(scene, dict):
            return None
        item_id = scene.get("item_id")
        sample_key = scene.get("sample_key")
        loss = _nonnegative_finite_float(scene.get("mean_physical_set_loss"))
        if (
            not isinstance(item_id, str)
            or not item_id
            or not isinstance(sample_key, str)
            or not sample_key
            or loss is None
            or (item_id, sample_key) in scenes
        ):
            return None
        scenes[(item_id, sample_key)] = scene
    return scenes


def _physical_component_values(value: object) -> dict[str, float] | None:
    if not isinstance(value, dict) or set(value) != set(G3_PHYSICAL_SET_LOSS_COMPONENTS):
        return None
    components = {name: _nonnegative_finite_float(value.get(name)) for name in value}
    if any(component is None for component in components.values()):
        return None
    return {name: float(components[name]) for name in G3_PHYSICAL_SET_LOSS_COMPONENTS}


def _physical_retention_partition_evaluation(
    rank_reports: list[dict[str, Any]],
    *,
    partition: str,
) -> dict[str, Any]:
    failures: list[str] = []
    partition_pairs: list[dict[str, Any]] = []
    scene_pairs: list[dict[str, Any]] = []
    component_pairs: list[dict[str, Any]] = []
    component_reference_states: list[bool] = []
    for report in rank_reports:
        rank = report.get("rank")
        history = report.get("history")
        if not isinstance(history, list) or len(history) != 1 or not isinstance(history[0], dict):
            failures.append(f"rank {rank}: retention phase did not publish exactly one receipt")
            continue
        partition_report = history[0].get(partition)
        reference_root = report.get("g2_physical_retention_reference")
        reference = None if not isinstance(reference_root, dict) else reference_root.get(partition)
        if not isinstance(partition_report, dict):
            failures.append(f"rank {rank}: {partition} retention receipt is invalid")
            continue
        if not isinstance(reference, dict):
            failures.append(f"rank {rank}: {partition} G2b physical reference is invalid")
            continue
        if (
            partition_report.get("scene_count") != G3_EVALUATION_SCENES_PER_PARTITION
            or partition_report.get("prompt_count") != 2 * G3_EVALUATION_SCENES_PER_PARTITION
        ):
            failures.append(f"rank {rank}: {partition} retention scene axis is incomplete")
        if partition_report.get("shared_row_gauge") is not True:
            failures.append(f"rank {rank}: {partition} row gauge changed across prompts")
        prompt_drift = _nonnegative_finite_float(
            partition_report.get("physical_prompt_drift_max_abs")
        )
        if prompt_drift is None:
            failures.append(f"rank {rank}: {partition} physical prompt drift is invalid")
        elif prompt_drift > G3_PHYSICAL_PROMPT_DRIFT_MAX_ABS:
            failures.append(f"rank {rank}: {partition} physical rows became prompt dependent")

        current_mean = _nonnegative_finite_float(partition_report.get("mean_physical_set_loss"))
        reference_mean = _nonnegative_finite_float(reference.get("mean_physical_set_loss"))
        if current_mean is None:
            failures.append(f"rank {rank}: {partition} physical set loss is invalid")
        if reference_mean is None:
            failures.append(f"rank {rank}: {partition} G2b physical reference is invalid")
        if current_mean is not None and reference_mean is not None:
            delta = current_mean - reference_mean
            passed = delta <= G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE
            partition_pairs.append(
                {
                    "rank": rank,
                    "current_mean_physical_set_loss": current_mean,
                    "g2_mean_physical_set_loss": reference_mean,
                    "delta": delta,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    f"rank {rank}: {partition} mean physical set loss regressed from accepted "
                    f"G2b by {delta:.9g} (tolerance "
                    f"{G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE:.9g})"
                )

        current_scenes = _physical_scene_map(partition_report.get("scenes"))
        reference_scenes = _physical_scene_map(reference.get("scenes"))
        if current_scenes is None or len(current_scenes) != G3_EVALUATION_SCENES_PER_PARTITION:
            failures.append(f"rank {rank}: {partition} physical scene evidence is invalid")
            continue
        if reference_scenes is None or len(reference_scenes) != G3_EVALUATION_SCENES_PER_PARTITION:
            failures.append(f"rank {rank}: {partition} G2b scene reference is invalid")
            continue
        if set(current_scenes) != set(reference_scenes):
            failures.append(
                f"rank {rank}: {partition} physical scenes do not pair with accepted G2b"
            )
            continue
        if current_mean is not None:
            observed_mean = _mean(
                [float(scene["mean_physical_set_loss"]) for scene in current_scenes.values()]
            )
            if abs(current_mean - observed_mean) > G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE:
                failures.append(
                    f"rank {rank}: {partition} physical partition mean differs from its scenes"
                )
        if reference_mean is not None:
            observed_reference_mean = _mean(
                [float(scene["mean_physical_set_loss"]) for scene in reference_scenes.values()]
            )
            if (
                abs(reference_mean - observed_reference_mean)
                > G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE
            ):
                failures.append(
                    f"rank {rank}: {partition} G2b partition mean differs from its scenes"
                )

        component_gate = reference.get("component_gate")
        if not isinstance(component_gate, dict) or not isinstance(
            component_gate.get("available_in_g2_reference"), bool
        ):
            failures.append(f"rank {rank}: {partition} G2b component gate metadata is invalid")
            continue
        components_available = bool(component_gate["available_in_g2_reference"])
        component_reference_states.append(components_available)
        for key in sorted(reference_scenes):
            current_scene = current_scenes[key]
            reference_scene = reference_scenes[key]
            current_loss = float(current_scene["mean_physical_set_loss"])
            reference_loss = float(reference_scene["mean_physical_set_loss"])
            delta = current_loss - reference_loss
            passed = delta <= G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE
            scene_pairs.append(
                {
                    "rank": rank,
                    "item_id": key[0],
                    "sample_key": key[1],
                    "current_mean_physical_set_loss": current_loss,
                    "g2_mean_physical_set_loss": reference_loss,
                    "delta": delta,
                    "passed": passed,
                }
            )
            if not passed:
                failures.append(
                    f"rank {rank}: {partition} scene {key[0]} physical set loss regressed "
                    f"from accepted G2b by {delta:.9g} (tolerance "
                    f"{G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE:.9g})"
                )
            if not components_available:
                continue
            current_components = _physical_component_values(
                current_scene.get("physical_set_loss_components")
            )
            reference_components = _physical_component_values(
                reference_scene.get("physical_set_loss_components")
            )
            if current_components is None or reference_components is None:
                failures.append(
                    f"rank {rank}: {partition} scene {key[0]} physical component evidence "
                    "is invalid"
                )
                continue
            for name in G3_PHYSICAL_SET_LOSS_COMPONENTS:
                component_delta = current_components[name] - reference_components[name]
                component_passed = component_delta <= G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE
                component_pairs.append(
                    {
                        "rank": rank,
                        "item_id": key[0],
                        "sample_key": key[1],
                        "component": name,
                        "current": current_components[name],
                        "g2": reference_components[name],
                        "delta": component_delta,
                        "passed": component_passed,
                    }
                )
                if not component_passed:
                    failures.append(
                        f"rank {rank}: {partition} scene {key[0]} physical component "
                        f"{name} regressed from accepted G2b by {component_delta:.9g} "
                        f"(tolerance {G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE:.9g})"
                    )

    component_state = (
        "ENFORCED"
        if component_reference_states and all(component_reference_states)
        else "NOT_ENFORCED_G2_REFERENCE_GAP"
        if component_reference_states and not any(component_reference_states)
        else "INVALID"
    )
    if component_state == "INVALID":
        failures.append(f"{partition}: G2b physical component coverage is inconsistent")
    return {
        "partition": partition,
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "absolute_tolerance": G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE,
        "partition_pairs": partition_pairs,
        "scene_pairs": scene_pairs,
        "component_gate": {
            "status": component_state,
            "components": (
                list(G3_PHYSICAL_SET_LOSS_COMPONENTS) if component_state == "ENFORCED" else []
            ),
            "gap": (
                "accepted G2b report does not publish per-scene physical loss components"
                if component_state == "NOT_ENFORCED_G2_REFERENCE_GAP"
                else None
            ),
            "pairs": component_pairs,
        },
        "action_diagnostic_gating": False,
    }


def _physical_retention_summary(
    rank_reports: list[dict[str, Any]],
    *,
    partition: str,
) -> dict[str, Any]:
    """Summarize only the physical metrics used by the retention decision."""

    evaluation = _physical_retention_partition_evaluation(
        rank_reports,
        partition=partition,
    )
    partition_pairs = evaluation["partition_pairs"]
    scene_pairs = evaluation["scene_pairs"]
    return {
        **evaluation,
        "maximum_partition_loss_delta": (
            None if not partition_pairs else max(float(pair["delta"]) for pair in partition_pairs)
        ),
        "maximum_scene_loss_delta": (
            None if not scene_pairs else max(float(pair["delta"]) for pair in scene_pairs)
        ),
        "all_partition_means_within_tolerance": bool(partition_pairs)
        and all(bool(pair["passed"]) for pair in partition_pairs),
        "all_paired_scenes_within_tolerance": bool(scene_pairs)
        and all(bool(pair["passed"]) for pair in scene_pairs),
    }


def _retention_failures(rank_reports: list[dict[str, Any]]) -> list[str]:
    """Require paired physical posterior retention after direct-action training."""

    if len(rank_reports) != G2_WORLD_SIZE:
        return ["G3 representation retention omitted one or more distributed ranks"]
    ranks = [report.get("rank") for report in rank_reports]
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in ranks) or set(
        ranks
    ) != set(range(G2_WORLD_SIZE)):
        return ["G3 representation retention ranks differ from the distributed axis"]
    failures: list[str] = []
    for partition in ("validation", "heldout"):
        failures.extend(
            _physical_retention_partition_evaluation(
                rank_reports,
                partition=partition,
            )["failures"]
        )
    return failures


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    picf_source_contract = _picf_source_contract()
    g2_physical_retention_reference = _g2_physical_retention_reference(
        json.loads(args.g2_report.read_text(encoding="utf-8"))
    )
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
            direct_posterior_action_row_visibility,
            score_offline_ltop_action_mediation,
            seal_ltop_action_receipt,
        )
        from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
        from picf_next.lingbot_native.state import AddressedLayerwisePriorTrace
        from picf_next.lingbot_native.task_address_learning import (
            task_address_row_coverage,
        )
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
        trained_picf_source_contract: dict[str, object] | None = None
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
                    expected_head_scope=args.direct_posterior_head_scope,
                    expected_source_stage_checkpoint=args.stage_checkpoint,
                    expected_g2_report_sha256=stage_contract.g2_report_sha256,
                    expected_runtime_source_contract=ltop_stage_runtime_source_contract(
                        stage_contract
                    ),
                )
            )
            trained_picf_source_contract = _validate_picf_source_contract(
                checkpoint_manifest.get("picf_source_contract")
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
        # FSDP ranks are also the two data-parallel members of the formal
        # global batch.  The retired paired-arm trial intentionally replayed
        # one global scene on both ranks, but the promoted single direct route
        # must preserve the rank-disjoint source partitions.
        scene_contract_items = local_items

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
        registered_layer_indices = _direct_posterior_registered_layer_indices(
            graph.config.num_layers
        )

        def training_forward(
            batch: CollatedNativeCALVINBatch,
            prior: AddressedLayerwisePriorTrace,
            *,
            require_grad: bool,
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
            result = (
                run_native_policy_training_forward(
                    policy,
                    model_inputs=batch.model_inputs,
                    context=context,
                    action_attention_callback=collector,
                )
                if require_grad
                else run_native_policy_diagnostic_forward(
                    policy,
                    model_inputs=batch.model_inputs,
                    context=context,
                    action_attention_callback=collector,
                )
            )
            receipts = collector.finalize()
            if any(
                require_grad != bool(receipt.posterior_attention.requires_grad)
                for receipt in receipts
            ):
                raise RuntimeError("ADR172 action-posterior receipt differs from forward phase")
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

        def evaluate_scene(scene: dict[str, Any]) -> dict[str, Any]:
            outputs: list[dict[str, Any]] = []
            relation_predictions = []
            canonical_assignment = None
            canonical_bindings = None
            canonical_identity_keys = None
            independent_bindings_by_prompt = []
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
            try:
                for prompt_index, host_batch in enumerate(scene["batches"]):
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
                        posterior_row_visible = direct_posterior_action_row_visibility(arm)
                        context = native_context_from_prior_trace(
                            controls=batch.controls,
                            prior_trace=prior,
                            modalities=None,
                            posterior_adoption_route=torch.ones(
                                batch.routing.batch_size,
                                dtype=torch.bool,
                                device=device,
                            ),
                            posterior_action_row_visible=posterior_row_visible,
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
                                joint_mask=batch.model_inputs["joint_mask"],
                                action_is_pad=batch.model_inputs["action_is_pad"],
                                executed_source_row_visible=posterior_row_visible,
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
                        canonical_assignment=canonical_assignment,
                        canonical_identity_keys=canonical_identity_keys,
                    )
                    if canonical_assignment is None:
                        canonical_assignment = physical["matched_assignment"]
                        canonical_bindings = physical["matched_bindings"]
                        canonical_identity_keys = physical["identity_keys_by_batch"]
                    if canonical_bindings is None:
                        raise RuntimeError(
                            "LTOP G3 evaluation did not establish a canonical row gauge"
                        )
                    relation_predictions.append(physical["relation"])
                    independent_bindings_by_prompt.append(
                        [list(value) for value in physical["matched_bindings"]]
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
                            "bindings": [list(value) for value in physical["bindings"]],
                            "independent_bindings": [
                                list(value) for value in physical["matched_bindings"]
                            ],
                            "arm_receipts": [
                                {
                                    "arm_name": receipt.arm_name,
                                    "arm_kind": receipt.arm_kind.value,
                                    "row_index": receipt.row_index,
                                    "source_visibility_sha256": (receipt.source_visibility_sha256),
                                    "active_action_mask_sha256": (
                                        receipt.active_action_mask_sha256
                                    ),
                                    "action_output_sha256": receipt.action_output_sha256,
                                }
                                for receipt in receipts
                            ],
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
                        posterior_row_visible,
                    )
            finally:
                for name, value in original_config.items():
                    setattr(runtime.model_config, name, value)
                policy.train(policy_was_training)
                graph.train(graph_was_training)
            if len(outputs) != 2 or canonical_bindings is None:
                raise RuntimeError("LTOP G3 evaluation requires two crossed prompts")
            first_score = outputs[0]["score"]
            second_score = outputs[1]["score"]
            if first_score["sample_keys"] != second_score["sample_keys"]:
                raise RuntimeError("LTOP G3 crossed prompts changed the action sample axis")
            if first_score["active_action_counts"] != second_score["active_action_counts"]:
                raise RuntimeError("LTOP G3 crossed prompts changed the executable action surface")
            crossed = [
                float(first) + float(second)
                for first, second in zip(
                    first_score["factual_target_minus_distractor"],
                    second_score["factual_target_minus_distractor"],
                    strict=True,
                )
            ]
            normalized_crossed = [
                0.5 * (float(first) + float(second))
                for first, second in zip(
                    first_score["factual_selectivity_over_all_posterior_block"],
                    second_score["factual_selectivity_over_all_posterior_block"],
                    strict=True,
                )
            ]
            replay = [
                float(value) for output in outputs for value in output["score"]["replay_floor_rms"]
            ]
            prompt_all_block = [
                float(output["score"]["mean_factual_all_posterior_block_effect_rms"])
                for output in outputs
            ]
            return {
                "item_id": scene["item"]["item_id"],
                "sample_key": scene["item"]["sample_key"],
                "prompt_count": len(outputs),
                "target_identities": list(scene["target_identities"]),
                "canonical_bindings": [list(value) for value in canonical_bindings],
                "independent_bindings_by_prompt": independent_bindings_by_prompt,
                "shared_row_gauge": all(
                    tuple(tuple(value) for value in bindings) == canonical_bindings
                    for bindings in independent_bindings_by_prompt
                ),
                "physical_prompt_drift_max_abs": _physical_relation_prompt_drift(
                    relation_predictions[0],
                    relation_predictions[1],
                ),
                "prompts": outputs,
                "score": {
                    "sample_keys": list(first_score["sample_keys"]),
                    "active_action_counts": list(first_score["active_action_counts"]),
                    "blocked_placebo_integrity_verified": all(
                        bool(output["score"]["blocked_placebo_integrity_verified"])
                        for output in outputs
                    ),
                    "replay_floor_rms": replay,
                    "max_replay_floor_rms": max(replay),
                    "prompt_mean_factual_all_posterior_block_effect_rms": prompt_all_block,
                    "minimum_prompt_factual_all_posterior_block_effect_rms": min(prompt_all_block),
                    "crossed_prompt_target_selectivity": crossed,
                    "crossed_prompt_selectivity_over_all_posterior_block": normalized_crossed,
                    "mean_crossed_prompt_target_selectivity": _mean(crossed),
                    "mean_crossed_prompt_selectivity_over_all_posterior_block": _mean(
                        normalized_crossed
                    ),
                    "positive_crossed_prompt_target_selectivity_count": sum(
                        value > 0.0 for value in crossed
                    ),
                    "sample_count": len(crossed),
                },
            }

        def evaluate_retention_scene(scene: dict[str, Any]) -> dict[str, Any]:
            """Re-evaluate physical rows and report direct-action diagnostics."""

            distributions = []
            adoption_masses: list[float] = []
            target_rows: list[int] = []
            bindings_by_prompt = []
            independent_bindings_by_prompt = []
            set_losses: list[float] = []
            set_loss_components: dict[str, list[float]] = {
                name: [] for name in G3_PHYSICAL_SET_LOSS_COMPONENTS
            }
            relation_predictions = []
            canonical_assignment = None
            canonical_bindings = None
            canonical_identity_keys = None
            first_batch = batch_to_device(scene["batches"][0])
            prior = build_prior(first_batch)
            with torch.no_grad():
                for prompt_index, host_batch in enumerate(scene["batches"]):
                    batch = first_batch if prompt_index == 0 else batch_to_device(host_batch)
                    result, posterior_receipts = training_forward(
                        batch,
                        prior,
                        require_grad=False,
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
                    physical_set_loss = physical["set_loss"]
                    set_losses.append(float(physical_set_loss.total.float().item()))
                    for name in G3_PHYSICAL_SET_LOSS_COMPONENTS:
                        set_loss_components[name].append(
                            float(getattr(physical_set_loss, name).float().item())
                        )
                    relation_predictions.append(physical["relation"])
                    final_receipt = posterior_receipts[-1]
                    if final_receipt.layer_index != registered_layer_indices[-1]:
                        raise RuntimeError(
                            "ADR172 retention did not receive the final registered action layer"
                        )
                    if final_receipt.layer_index != final_receipt.layer_count - 1:
                        raise RuntimeError(
                            "ADR172 retention final registered receipt is not the final host layer"
                        )
                    distributions.append(
                        aggregate_action_posterior_distribution(final_receipt.posterior_attention)
                    )
                    adoption_masses.append(
                        float(final_receipt.total_posterior_mass.detach().float().mean().item())
                    )
                    if prompt_index != 0:
                        del batch
                    del final_receipt, posterior_receipts, result
            direct_action_diagnostic = _scene_metrics(
                (distributions[0], distributions[1]),
                (target_rows[0], target_rows[1]),
                task_address_row_coverage=task_address_row_coverage,
                torch_module=torch,
            )
            return {
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
                "physical_set_loss_components": {
                    name: sum(values) / len(values) for name, values in set_loss_components.items()
                },
                "physical_prompt_drift_max_abs": _physical_relation_prompt_drift(
                    relation_predictions[0],
                    relation_predictions[1],
                ),
                "direct_action_diagnostic": {
                    **direct_action_diagnostic,
                    "minimum_adoption_mass": min(adoption_masses),
                    "mean_adoption_mass": sum(adoption_masses) / len(adoption_masses),
                    "query_axis": "one-adoption-weighted-action-aggregate",
                    "scientific_gate": False,
                    "retention_gate": False,
                },
            }

        def evaluate_retention_partition(partition: str) -> dict[str, Any]:
            per_scene = [evaluate_retention_scene(scene) for scene in scenes[partition]]
            prompts = [
                prompt
                for scene in per_scene
                for prompt in scene["direct_action_diagnostic"]["prompts"]
            ]
            return {
                "scene_count": len(per_scene),
                "prompt_count": len(prompts),
                "mean_physical_set_loss": sum(
                    float(scene["mean_physical_set_loss"]) for scene in per_scene
                )
                / len(per_scene),
                "mean_physical_set_loss_components": {
                    name: sum(
                        float(scene["physical_set_loss_components"][name]) for scene in per_scene
                    )
                    / len(per_scene)
                    for name in G3_PHYSICAL_SET_LOSS_COMPONENTS
                },
                "physical_prompt_drift_max_abs": max(
                    float(scene["physical_prompt_drift_max_abs"]) for scene in per_scene
                ),
                "shared_row_gauge": all(bool(scene["shared_row_gauge"]) for scene in per_scene),
                "direct_action_diagnostic": {
                    "mean_margin": sum(float(prompt["margin"]) for prompt in prompts)
                    / len(prompts),
                    "positive_margin_count": sum(float(prompt["margin"]) > 0 for prompt in prompts),
                    "mean_target_nll": sum(
                        float(scene["direct_action_diagnostic"]["mean_target_nll"])
                        for scene in per_scene
                    )
                    / len(per_scene),
                    "minimum_adoption_mass": min(
                        float(scene["direct_action_diagnostic"]["minimum_adoption_mass"])
                        for scene in per_scene
                    ),
                    "mean_adoption_mass": sum(
                        float(scene["direct_action_diagnostic"]["mean_adoption_mass"])
                        for scene in per_scene
                    )
                    / len(per_scene),
                    "metric_self_checks": {
                        "matched_row_permutation_max_abs_error": max(
                            float(
                                scene["direct_action_diagnostic"]["metric_self_checks"][
                                    "matched_row_permutation_max_abs_error"
                                ]
                            )
                            for scene in per_scene
                        )
                    },
                    "query_axis": "one-adoption-weighted-action-aggregate",
                    "scientific_gate": False,
                    "retention_gate": False,
                    "prompts": prompts,
                },
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
                scene_outputs = [evaluate_scene(scene) for scene in partition_scenes]
                replay = [
                    value for scene in scene_outputs for value in scene["score"]["replay_floor_rms"]
                ]
                entry[partition] = {
                    "scene_count": len(scene_outputs),
                    "prompt_count": sum(int(scene["prompt_count"]) for scene in scene_outputs),
                    "scenes": scene_outputs,
                    "max_replay_floor_rms": max(replay, default=math.inf),
                }
            history.append(entry)

        action_losses: list[float] = []
        total_losses: list[float] = []
        physical_losses: list[float] = []
        grounding_losses: list[float] = []
        grounding_history: list[dict[str, Any]] = []
        gradient_metrics_history: list[dict[str, Any]] = []
        direct_route_history: list[dict[str, Any]] = []
        direct_route_counts = {G3_DIRECT_ROUTE: 0}
        action_supervision_history: list[dict[str, Any]] = []
        task_address_supervision_history: list[dict[str, Any]] = []
        all_gradients_finite = True
        direct_schedule: dict[str, Any] | None = None
        if args.mode in {"direct-trial", "smoke"}:
            direct_schedule = build_g3_direct_source_schedule(
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
            route: str,
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
                        "route": route,
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
                direct_schedule["entries"][step - 1] if direct_schedule is not None else None
            )
            if schedule_entry is None:
                scene_index = (step - 1) % len(scenes["validation"])
                direct_route = G3_DIRECT_ROUTE
            else:
                scene_index = int(schedule_entry["scene_index"])
                direct_route = str(schedule_entry["route"])
            scene = scenes["validation"][scene_index]
            if schedule_entry is not None and (
                schedule_entry["scene_key"] != scene["item"]["item_id"]
                or schedule_entry["source_task_key"] != scene["source_task_key"]
                or direct_route != G3_DIRECT_ROUTE
            ):
                raise RuntimeError("ADR172 direct schedule differs from the materialized scene")
            supervision = scene["source_action_supervision"]
            require_factual_action_supervision(supervision)
            trace_training_stage(
                step=step,
                stage="batch-begin",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
                synchronize=True,
            )
            batch = batch_to_device(scene["source_batch"])
            trace_training_stage(
                step=step,
                stage="prior-begin",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
                synchronize=True,
            )
            prior = build_prior(scene["source_batch"])
            trace_training_stage(
                step=step,
                stage="forward-begin",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
                synchronize=True,
            )
            result, posterior_receipts = training_forward(
                batch,
                prior,
                require_grad=True,
            )
            trace_training_stage(
                step=step,
                stage="forward-done",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
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
                route=direct_route,
                synchronize=True,
            )
            target_row = physical["target_row"]
            if batch.routing.batch_size != 1:
                raise RuntimeError("ADR172 source-target runner is preregistered at batch size one")
            action_head_count = posterior_receipts[0].posterior_attention.shape[1]
            if any(
                receipt.posterior_attention.shape[1] != action_head_count
                for receipt in posterior_receipts
            ):
                raise RuntimeError("ADR172 registered action layers disagree on head count")
            grounding_head_indices_tuple = _direct_posterior_head_indices(
                args.direct_posterior_head_scope,
                head_count=action_head_count,
            )
            grounding_head_indices = (
                None
                if grounding_head_indices_tuple is None
                else torch.tensor(
                    grounding_head_indices_tuple,
                    dtype=torch.long,
                    device=device,
                )
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
            if target_row is not None:
                target_row_weights[:, target_row] = 1.0
                target_valid[:] = True
            grounding_results = tuple(
                action_posterior_target_mass_loss(
                    receipt.posterior_attention,
                    target_row_weights=target_row_weights,
                    target_valid=target_valid,
                    head_indices=grounding_head_indices,
                )
                for receipt in posterior_receipts
            )
            grounding_loss = torch.stack(tuple(value.loss for value in grounding_results)).mean()
            loss = (
                args.official_loss_weight * result.official_total_loss
                + args.physical_set_weight * physical["set_loss"].total
                + args.direct_grounding_weight * grounding_loss
            )
            trace_training_stage(
                step=step,
                stage="backward-begin",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
                synchronize=True,
            )
            loss.backward()
            trace_training_stage(
                step=step,
                stage="backward-done",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
                synchronize=True,
            )
            metrics = _distributed_gradient_metrics(
                policy,
                (
                    ("native_graph", "picf_native_graph"),
                    ("shared_host", "qwenvl_with_expert.qwen"),
                    ("shared_q_projection", "q_proj"),
                    ("shared_k_projection", "k_proj"),
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
                route=direct_route,
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
                route=direct_route,
                synchronize=True,
            )
            optimizer.step()
            trace_training_stage(
                step=step,
                stage="optimizer-done",
                scene_key=scene["item"]["item_id"],
                route=direct_route,
                synchronize=True,
            )
            action_loss_value = float(result.official_action_loss.detach().float().item())
            total_loss_value = float(loss.detach().float().item())
            physical_loss_value = float(physical["set_loss"].total.detach().float().item())
            grounding_loss_value = float(grounding_loss.detach().float().item())
            action_losses.append(action_loss_value)
            total_losses.append(total_loss_value)
            physical_losses.append(physical_loss_value)
            grounding_losses.append(grounding_loss_value)
            grounding_record = {
                "global_step": step,
                "target_row": target_row,
                "target_valid": bool(target_valid.item()),
                "registered_layer_indices": list(registered_layer_indices),
                "head_scope": args.direct_posterior_head_scope,
                "head_indices": (
                    None
                    if grounding_head_indices_tuple is None
                    else list(grounding_head_indices_tuple)
                ),
                "layers": [
                    {
                        "layer_index": receipt.layer_index,
                        "target_mass_mean": float(value.target_mass.detach().float().mean().item()),
                        "total_posterior_mass_mean": float(
                            value.total_posterior_mass.detach().float().mean().item()
                        ),
                    }
                    for receipt, value in zip(
                        posterior_receipts,
                        grounding_results,
                        strict=True,
                    )
                ],
            }
            grounding_history.append(grounding_record)
            direct_route_counts[direct_route] += 1
            route_record = {
                "global_step": step,
                "cycle_index": (None if schedule_entry is None else schedule_entry["cycle_index"]),
                "scene_index": scene_index,
                "scene_key": scene["item"]["item_id"],
                "prompt_index": 0,
                "prompt_key": f"source-task/{scene['source_task_key']}",
                "route": direct_route,
            }
            direct_route_history.append(route_record)
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
                        "schema": G3_ROUTE_JOURNAL_SCHEMA,
                        "rank": rank,
                        **route_record,
                        "schedule_sha256": (
                            None if direct_schedule is None else direct_schedule["sha256"]
                        ),
                        "sample_keys": list(batch.routing.episode_keys),
                        "action_loss": action_loss_value,
                        "total_loss": total_loss_value,
                        "physical_set_loss": physical_loss_value,
                        "direct_grounding_loss": grounding_loss_value,
                        "source_task_key": scene["source_task_key"],
                        "source_target_identity": scene["source_target_identity"],
                        "direct_grounding_supervision_enabled": target_row is not None,
                        "direct_grounding_supervision_reason": physical["target_row_reason"],
                        "direct_grounding": grounding_record,
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
            del (
                batch,
                grounding_loss,
                grounding_results,
                loss,
                physical,
                posterior_receipts,
                prior,
                result,
                target_row_weights,
                target_valid,
            )
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
                        "direct_route_counts": direct_route_counts,
                        "schedule_sha256": (
                            None if direct_schedule is None else direct_schedule["sha256"]
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
        training_prepublication_failures: list[str] = []
        if args.phase == "training":
            prepublication_rank_report = {
                "rank": rank,
                "all_gradients_finite": all_gradients_finite,
                "cuda_memory_bytes": {
                    "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                },
                "gradient_metrics_history": gradient_metrics_history,
                "action_losses": action_losses,
                "action_supervision_history": action_supervision_history,
                "direct_grounding_losses": grounding_losses,
                "direct_grounding_history": grounding_history,
                "task_address_supervision_history": task_address_supervision_history,
                "arm_journal": journal_receipt,
            }
            gathered_prepublication: list[dict[str, Any] | None] = [None] * G2_WORLD_SIZE
            dist.all_gather_object(gathered_prepublication, prepublication_rank_report)
            prepublication_outcome: list[object] = [None]
            if rank == 0:
                ordered_prepublication = sorted(
                    (value for value in gathered_prepublication if value is not None),
                    key=lambda value: value["rank"],
                )
                prepublication_outcome[0] = _training_failures(
                    ordered_prepublication,
                    mode=args.mode,
                    head_scope=args.direct_posterior_head_scope,
                )
            dist.broadcast_object_list(prepublication_outcome, src=0)
            if not isinstance(prepublication_outcome[0], list) or any(
                not isinstance(value, str) for value in prepublication_outcome[0]
            ):
                raise RuntimeError("ADR172 training prepublication gate is malformed")
            training_prepublication_failures = list(prepublication_outcome[0])
        checkpoint_report: dict[str, Any] | None = None
        post_checkpoint_save_model_local_state_sha256: str | None = None
        if args.phase == "training" and not training_prepublication_failures:
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
                    "direct_route_counts": dict(direct_route_counts),
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
                        "direct_action_causal_surface": G3_DIRECT_ACTION_CAUSAL_SURFACE,
                        "direct_route": G3_DIRECT_ROUTE,
                        "picf_source_contract": picf_source_contract,
                        "direct_posterior_registered_layer_indices": list(registered_layer_indices),
                        "direct_posterior_head_scope": args.direct_posterior_head_scope,
                        "direct_posterior_head_indices": (
                            None
                            if args.direct_posterior_head_scope == ADR172_ACTION_HEAD_SCOPE
                            else list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES)
                        ),
                        "direct_grounding_weight": args.direct_grounding_weight,
                        "direct_grounding_upstream_contract": (
                            _direct_grounding_upstream_contract(args.direct_posterior_head_scope)
                        ),
                        "training_final_model_local_state_sha256_by_rank": (
                            ordered_training_digests
                        ),
                        "direct_route_schedule_sha256": (
                            None if direct_schedule is None else direct_schedule["sha256"]
                        ),
                        "direct_route_counts_by_rank": [
                            item["direct_route_counts"] for item in ordered_training_states
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
                        "direct_posterior_registered_layer_indices": list(registered_layer_indices),
                        "direct_posterior_head_scope": args.direct_posterior_head_scope,
                        "direct_posterior_head_indices": (
                            None
                            if args.direct_posterior_head_scope == ADR172_ACTION_HEAD_SCOPE
                            else list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES)
                        ),
                        "direct_grounding_weight": args.direct_grounding_weight,
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
            "direct_action_causal_surface": G3_DIRECT_ACTION_CAUSAL_SURFACE,
            "g2_physical_retention_reference": g2_physical_retention_reference[rank],
            "history": history,
            "action_losses": action_losses,
            "total_losses": total_losses,
            "physical_losses": physical_losses,
            "direct_grounding_losses": grounding_losses,
            "direct_grounding_history": grounding_history,
            "direct_route_history": direct_route_history,
            "direct_route_counts": direct_route_counts,
            "action_supervision_history": action_supervision_history,
            "action_supervision_schema": TASK_ACTION_SUPERVISION_SCHEMA,
            "task_address_supervision_history": task_address_supervision_history,
            "direct_route_schedule_sha256": (
                None if direct_schedule is None else direct_schedule["sha256"]
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
                failures = _training_failures(
                    rank_reports,
                    mode=args.mode,
                    head_scope=args.direct_posterior_head_scope,
                )
                if failures != training_prepublication_failures:
                    raise RuntimeError("ADR172 training gate changed after checkpoint decision")
                if bool(failures) == bool(checkpoint_report):
                    raise RuntimeError(
                        "ADR172 checkpoint publication disagrees with the training gate"
                    )
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
                "direct_action_causal_surface": G3_DIRECT_ACTION_CAUSAL_SURFACE,
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
                "evaluation_route": (G3_DIRECT_ROUTE if args.phase == "evaluation" else None),
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
                "trained_picf_source_contract": (
                    trained_picf_source_contract
                    if args.phase in {"evaluation", "retention"}
                    else picf_source_contract
                ),
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
                        "active_action_surface": "joint_mask AND NOT action_is_pad",
                        "arms": "factual/repeat/every-row removal/all-posterior block",
                        "blocked_row_arms": "bitwise execution-integrity placebos only",
                        "labels_opened_after_all_forward_receipts": True,
                        "moe_backend": moe_inference_backend,
                    }
                ),
                "causal_adoption_contract": (
                    {
                        "claim": (
                            "the current posterior contributes additional task-selective causal "
                            "information to executable action while native RGB, prompt, and "
                            "proprioceptive paths remain available"
                        ),
                        "exclusive_visual_path_claim": False,
                        "canonical_object_axis": True,
                        "crossed_prompt_scene_score": (
                            "[effect_A(row_A)-effect_A(row_B)] + [effect_B(row_B)-effect_B(row_A)]"
                        ),
                        "effect": (
                            "masked RMS(factual action, row-intervened action) over executable "
                            "non-padding action elements"
                        ),
                        "all_posterior_block_effect_reported": True,
                        "blocked_row_placebo_gating": "bitwise equality",
                    }
                    if args.phase == "evaluation"
                    else None
                ),
                "representation_retention_contract": (
                    {
                        "purpose": (
                            "verify physical-set posterior retention against the accepted G2b "
                            "endpoint"
                        ),
                        "optimizer_updates": 0,
                        "scenes_per_rank_per_partition": (G3_EVALUATION_SCENES_PER_PARTITION),
                        "crossed_prompts_per_scene": 2,
                        "reference": "accepted G2b paired physical-set scene endpoints",
                        "decision_metrics": (
                            "partition mean and every paired scene physical-set loss, plus "
                            "per-scene components when present in G2b"
                        ),
                        "absolute_tolerance": G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE,
                        "component_baseline_policy": (
                            "gate paired components when G2b publishes them; otherwise publish "
                            "the missing-baseline gap without fabricating component references"
                        ),
                        "action_diagnostic": {
                            "published": True,
                            "gating": False,
                            "scientific_evidence": False,
                        },
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
                        "ambiguous_source_direct_grounding_loss": False,
                        "unobservable_source_target_direct_grounding_loss": False,
                        "unobservable_source_target_policy": (
                            "disable-direct-grounding-only-with-explicit-loss-side-receipt"
                        ),
                    },
                    "direct_posterior_adoption": {
                        "route": "native-task-independent-direct-posterior",
                        "registered_layer_indices": list(registered_layer_indices),
                        "head_scope": args.direct_posterior_head_scope,
                        "head_indices": (
                            None
                            if args.direct_posterior_head_scope == ADR172_ACTION_HEAD_SCOPE
                            else list(ADR172_GUIDEDVLA_OBJECT_HEAD_INDICES)
                        ),
                        "upstream_contract": _direct_grounding_upstream_contract(
                            args.direct_posterior_head_scope
                        ),
                        "single_forward_per_optimizer_step": True,
                        "deploy_time_module_added": False,
                    },
                    "loss_weights": {
                        "official": args.official_loss_weight,
                        "physical_set": args.physical_set_weight,
                        "direct_grounding": args.direct_grounding_weight,
                    },
                },
                "checkpoint": checkpoint_report,
                "thresholds": (
                    {
                        "physical_set_loss_absolute_tolerance": (
                            G3_PHYSICAL_RETENTION_ABSOLUTE_TOLERANCE
                        ),
                        "partition_mean_physical_set_loss_required": True,
                        "paired_scene_physical_set_loss_required": True,
                        "paired_component_physical_set_loss_required_when_present_in_g2": True,
                        "physical_prompt_drift_max_abs": G3_PHYSICAL_PROMPT_DRIFT_MAX_ABS,
                        "shared_row_gauge_required": True,
                        "action_diagnostic_gating": False,
                    }
                    if args.phase == "retention"
                    else {
                        "bitwise_factual_replay": True,
                        "blocked_row_placebo_bitwise_equality": True,
                        "shared_canonical_row_gauge": True,
                        "mean_crossed_prompt_target_selectivity_strictly_positive": True,
                        "mean_normalized_crossed_prompt_selectivity_strictly_positive": True,
                        "mean_all_posterior_block_effect_strictly_positive": True,
                        "joint_positive_scene_requires_normalized_selectivity": True,
                        "joint_positive_scene_fraction_minimum": (G3_COLD_POSITIVE_SCENE_FRACTION),
                    }
                ),
                "rank_reports": rank_reports,
            }
            if args.phase == "evaluation":
                expected_scenes = 1 if args.mode == "smoke" else G3_EVALUATION_SCENES_PER_PARTITION
                report["cold_causal_summary"] = {
                    partition: _cold_causal_partition_evaluation(
                        rank_reports,
                        partition=partition,
                        expected_scenes_per_rank=expected_scenes,
                        apply_scientific_gate=args.mode != "smoke",
                    )
                    for partition in (
                        ("validation",) if args.mode == "smoke" else ("validation", "heldout")
                    )
                }
            if args.phase == "retention":
                report["physical_retention_summary"] = {
                    partition: _physical_retention_summary(
                        rank_reports,
                        partition=partition,
                    )
                    for partition in ("validation", "heldout")
                }
            write_text_durable_exclusive(args.output, _canonical_json(report) + "\n")
            outcome[:] = [report["status"], report["failures"]]
        dist.broadcast_object_list(outcome, src=0)
        if outcome[0] != "PASS":
            raise RuntimeError(f"LTOP G3 failed: {outcome[1]}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Adapt shared LingBot Qwen grounding while keeping the action policy frozen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT, _ROOT / "src"):
    _text = str(_path)
    while _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

from tools.cuda_allocator_bootstrap import (
    CUDA_ALLOCATOR_MODES,
    bootstrap_cuda_allocator,
    configure_cuda_allocator as _configure_cuda_allocator,
)

_BOOTSTRAPPED_CUDA_ALLOCATOR = (
    bootstrap_cuda_allocator(sys.argv[1:]) if __name__ == "__main__" else None
)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.data.calvin_qwen_grounding import (
    CalvinQwenGroundingRecord,
    CalvinQwenSceneGroundingRecord,
)
from picf_next.data.public_native_vl import (
    NativeVLInstructionRecord,
    PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
    PublicNativeVLRetentionManifest,
    load_frozen_public_native_vl_retention_gate,
)
from picf_next.lingbot_native.crossed_bounded_plan import (
    CROSSED_BOUNDED_TOTAL_STEPS,
    CrossedBoundedPlan,
    CrossedBoundedRecord,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.gradient_alignment import (
    WeightedGradientTripleMoments,
    summarize_weighted_qwen_gradient_triple,
)
from picf_next.lingbot_native.gradient_audit_runtime import (
    distributed_pair_rows,
    snapshot_local_gradients,
)
from picf_next.lingbot_native.lattice_feasibility import (
    LATTICE_BASELINE,
    configure_native_processor_area_budget,
    configure_native_processor_lattice,
    validate_native_processor_record_grid,
)
from picf_next.lingbot_native.runtime_provenance import (
    adr127_runtime_python_trees_contract,
)
from tools.bootstrap_lingbot_vla2 import validate_checkpoint, validate_processor
from tools.bootstrap_lingbot_vla2_native import (
    LINGBOT_NATIVE_SOURCE_COMMIT,
    MODEL_SOURCE,
    QWEN_PROCESSOR_REVISION,
)
from tools.bootstrap_lingbot_vla2_native_vl import (
    NATIVE_VL_PATCH_RELATIVE_PATH,
    NATIVE_VL_PATCHED_MODEL_SHA256,
    _validate_native_vl_model,
    detect_native_vl_patch_state,
    verify_native_vl_patch,
)
from tools.lingbot_vla2_runtime_helpers import (
    _merge_qwen_config,
    _resolve_training_config,
    load_lingbot_training_config,
    register_native_fsdp_forward_methods,
    resolve_lingbot_optimizer_contract,
    strip_targetless_alignment_teacher_heads,
)
from tools.probe_lingbot_native_vl_grounding import (
    _validate_optional_qwen_restore,
    _validate_qwen_restore_load_result,
)
from tools.probe_qwen3vl_grounding_baseline import _model_hashes

WORLD_SIZE = 2
OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-grounding-adaptation.v9"
ADR128_OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-crossed-adaptation.v1"
CURRICULUM_OBSERVATION_DUAL_LATTICE = "dual_lattice"
CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE = "official_native_once"
CURRICULUM_OBSERVATION_MODES = (
    CURRICULUM_OBSERVATION_DUAL_LATTICE,
    CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE,
)
PAIR_PLAN_OBSERVATION_SINGLE_LATTICE = "legacy_pair_plan_single_lattice"
CALVIN_FACTOR_TARGET_ONLY = "target_only"
CALVIN_FACTOR_TARGET_REPEAT_CONTROL = "target_repeat_control"
CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE = "counterfactual_scene_candidate"
CALVIN_FACTOR_MODES = (
    CALVIN_FACTOR_TARGET_ONLY,
    CALVIN_FACTOR_TARGET_REPEAT_CONTROL,
    CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE,
)
ADR127_FACTOR_MODES = (
    CALVIN_FACTOR_TARGET_REPEAT_CONTROL,
    CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE,
)
ADR127_MAX_STEPS = 64
ADR127_SCHEDULE_TOTAL_STEPS = 432
ADR127_INITIAL_QWEN_REVISION = "0196dc7bb23f3c742616147c3254d0e4f1207787"
ADR127_GRADIENT_AUDIT_STEPS = (0, 21, 42, 63)
ADR127_GRADIENT_OBJECTIVES = ("target", "scene", "public")
ADR127_GRADIENT_WEIGHTS = (0.5, 0.5, PUBLIC_NATIVE_VL_RETENTION_WEIGHT)
ADR128_INITIAL_QWEN_REVISION = ADR127_INITIAL_QWEN_REVISION
ADR128_MAX_STEPS = CROSSED_BOUNDED_TOTAL_STEPS
ADR128_SMOKE_STEPS = 2
ADR128_SCHEDULE_TOTAL_STEPS = ADR127_SCHEDULE_TOTAL_STEPS
ADR128_SEED = 20260802
ADR128_LEARNING_RATE = 1e-6
ADR128_WEIGHT_DECAY = 0.0
ADR128_ADAM_BETA1 = 0.9
ADR128_ADAM_BETA2 = 0.999
ADR128_ADAM_EPS = 1e-8
ADR128_MAX_GRAD_NORM = 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_git_revision(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError("native VL adaptation PICF revision must be one Git commit")
    return value


def _validate_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"native VL adaptation {name} must be one lowercase SHA-256")
    return value


def _validate_training_horizons(
    *,
    max_steps: int,
    schedule_total_steps: int,
    warmup_steps: int,
) -> tuple[int, int, int]:
    if (
        isinstance(max_steps, bool)
        or not isinstance(max_steps, int)
        or max_steps <= 0
        or isinstance(schedule_total_steps, bool)
        or not isinstance(schedule_total_steps, int)
        or schedule_total_steps < max_steps
        or isinstance(warmup_steps, bool)
        or not isinstance(warmup_steps, int)
        or not 0 <= warmup_steps < max_steps
    ):
        raise ContractError("native VL adaptation training horizons are invalid")
    return max_steps, schedule_total_steps, warmup_steps


def _select_curriculum_microbatches(
    microbatches: tuple[Any, ...],
    *,
    observation_mode: str,
) -> tuple[Any, ...]:
    """Select a preregistered observation measure without changing task pairs."""

    if observation_mode == CURRICULUM_OBSERVATION_DUAL_LATTICE:
        return microbatches
    if observation_mode != CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE:
        raise ContractError("native VL curriculum observation mode is unsupported")
    if len(microbatches) != 2 or tuple(batch[0] for batch in microbatches) != (8, 14):
        raise ContractError("official-native-once requires the signed dual-lattice curriculum")
    return microbatches[:1]


def _require_variant_pair(value: object) -> tuple[Any, Any]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ContractError("native VL materialization requires one two-variant tuple")
    return value[0], value[1]


def _calvin_factor_weights(mode: str) -> tuple[tuple[str, float], ...]:
    """Return the complete explicit CALVIN objective mixture for one mode."""

    if mode == CALVIN_FACTOR_TARGET_ONLY:
        return (("target", 1.0),)
    if mode == CALVIN_FACTOR_TARGET_REPEAT_CONTROL:
        return (("target", 0.5), ("target_repeat", 0.5))
    if mode == CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE:
        return (("target", 0.5), ("scene", 0.5))
    raise ContractError("native VL CALVIN factor mode is unsupported")


def _summarize_gradient_triple_rows(
    *,
    target_scene_names: tuple[str, ...],
    target_scene_rows: list[list[float]],
    target_public_names: tuple[str, ...],
    target_public_rows: list[list[float]],
    scene_public_names: tuple[str, ...],
    scene_public_rows: list[list[float]],
) -> dict[str, object]:
    """Reconstruct one exact three-objective Gram matrix from pair reductions."""

    if (
        not target_scene_names
        or target_scene_names != target_public_names
        or target_scene_names != scene_public_names
        or not (
            len(target_scene_names)
            == len(target_scene_rows)
            == len(target_public_rows)
            == len(scene_public_rows)
        )
    ):
        raise ContractError("native VL ADR-127 gradient parameter rows changed")

    def same(value: float, expected: float) -> bool:
        return math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-12)

    moments = {}
    for name, target_scene, target_public, scene_public in zip(
        target_scene_names,
        target_scene_rows,
        target_public_rows,
        scene_public_rows,
        strict=True,
    ):
        if any(len(row) != 4 for row in (target_scene, target_public, scene_public)):
            raise ContractError("native VL ADR-127 gradient moment row is malformed")
        elements = tuple(int(round(row[3])) for row in (target_scene, target_public, scene_public))
        if (
            len(set(elements)) != 1
            or not same(target_scene[1], target_public[1])
            or not same(target_scene[2], scene_public[1])
            or not same(target_public[2], scene_public[2])
        ):
            raise ContractError("native VL ADR-127 repeated gradient norms changed")
        moments[name] = WeightedGradientTripleMoments(
            first_squared_norm=float(target_scene[1]),
            second_squared_norm=float(target_scene[2]),
            third_squared_norm=float(target_public[2]),
            first_second_dot=float(target_scene[0]),
            first_third_dot=float(target_public[0]),
            second_third_dot=float(scene_public[0]),
            elements=elements[0],
        )
    return summarize_weighted_qwen_gradient_triple(
        moments,
        objective_names=ADR127_GRADIENT_OBJECTIVES,
        weights=ADR127_GRADIENT_WEIGHTS,
    )


def _counterfactual_gradient_audit_status(reports: list[dict[str, Any]]) -> str:
    if [report.get("completed_updates_before_audit") for report in reports] != list(
        ADR127_GRADIENT_AUDIT_STEPS
    ):
        raise ContractError("native VL ADR-127 gradient audit steps changed")
    for report in reports:
        summary = report.get("summary")
        global_summary = summary.get("global") if isinstance(summary, dict) else None
        descends = (
            global_summary.get("mixed_gradient_descends")
            if isinstance(global_summary, dict)
            else None
        )
        if not isinstance(descends, dict) or any(
            not isinstance(descends.get(name), bool) for name in ADR127_GRADIENT_OBJECTIVES
        ):
            raise ContractError("native VL ADR-127 gradient audit summary is malformed")
        if not descends["target"] or not descends["scene"]:
            return "FAIL"
    return "PASS"


def _validate_calvin_factor_mode(
    args: argparse.Namespace,
    *,
    public_retention_enabled: bool,
) -> str:
    """Fail closed unless a causal ADR-127 arm exactly matches its preregistration."""

    mode = args.calvin_factor_mode
    _calvin_factor_weights(mode)
    if mode == CALVIN_FACTOR_TARGET_ONLY:
        if args.adr127_smoke or args.counterfactual_gradient_audit:
            raise ContractError("native VL ADR-127 flags require one ADR-127 factor mode")
        return mode
    if (
        args.curriculum_plan is None
        or args.curriculum_observation_mode != CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE
        or args.schedule_total_steps != ADR127_SCHEDULE_TOTAL_STEPS
        or args.warmup_steps != 0
        or args.initial_qwen_revision != ADR127_INITIAL_QWEN_REVISION
        or not public_retention_enabled
    ):
        raise ContractError("native VL ADR-127 factor mode differs from its frozen experiment")
    if args.adr127_smoke:
        if args.max_steps != 1 or args.counterfactual_gradient_audit:
            raise ContractError("native VL ADR-127 smoke differs from its frozen contract")
    elif args.max_steps != ADR127_MAX_STEPS:
        raise ContractError("native VL ADR-127 factor mode differs from its frozen experiment")
    elif (
        mode == CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE
    ) != args.counterfactual_gradient_audit:
        raise ContractError("native VL ADR-127 gradient audit mode differs from its arm")
    return mode


def _validate_crossed_bounded_mode(
    args: argparse.Namespace,
    *,
    public_retention_enabled: bool,
) -> bool:
    """Fail closed around the sole ADR-128 bounded candidate/control contract."""

    values = (
        args.crossed_bounded_plan,
        args.crossed_bounded_plan_sha256,
        args.crossed_arm,
    )
    if not any(value is not None for value in values):
        if args.adr128_smoke:
            raise ContractError("native VL ADR-128 smoke requires a crossed bounded plan")
        return False
    if any(value is None for value in values):
        raise ContractError("native VL crossed bounded arguments must be all present")
    if (
        args.curriculum_plan is None
        or args.curriculum_observation_mode != CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE
        or args.calvin_factor_mode != CALVIN_FACTOR_TARGET_ONLY
        or args.schedule_total_steps != ADR128_SCHEDULE_TOTAL_STEPS
        or args.warmup_steps != 0
        or args.initial_qwen_revision != ADR128_INITIAL_QWEN_REVISION
        or args.adr127_smoke
        or args.counterfactual_gradient_audit
        or not public_retention_enabled
        or args.seed != ADR128_SEED
    ):
        raise ContractError("native VL ADR-128 mode differs from its frozen experiment")
    expected_steps = ADR128_SMOKE_STEPS if args.adr128_smoke else ADR128_MAX_STEPS
    if args.max_steps != expected_steps:
        raise ContractError("native VL ADR-128 horizon differs from its frozen experiment")
    exact_hyperparameters = (
        (args.learning_rate, ADR128_LEARNING_RATE),
        (args.weight_decay, ADR128_WEIGHT_DECAY),
        (args.adam_beta1, ADR128_ADAM_BETA1),
        (args.adam_beta2, ADR128_ADAM_BETA2),
        (args.adam_eps, ADR128_ADAM_EPS),
        (args.max_grad_norm, ADR128_MAX_GRAD_NORM),
    )
    if any(float(observed) != expected for observed, expected in exact_hyperparameters):
        raise ContractError("native VL ADR-128 optimizer differs from its frozen experiment")
    if not isinstance(args.crossed_bounded_plan_sha256, str):
        raise ContractError("native VL crossed bounded plan SHA-256 is missing")
    _validate_sha256(args.crossed_bounded_plan_sha256, name="crossed bounded plan SHA-256")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--initial-qwen-dir", type=Path, required=True)
    parser.add_argument("--initial-qwen-revision", required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--public-vl-retention-manifest", type=Path)
    parser.add_argument("--public-vl-retention-manifest-sha256")
    parser.add_argument("--public-vl-retention-root", type=Path)
    parser.add_argument("--public-vl-retention-weight", type=float)
    plan = parser.add_mutually_exclusive_group(required=True)
    plan.add_argument("--pair-plan", type=Path)
    plan.add_argument("--curriculum-plan", type=Path)
    parser.add_argument("--curriculum-plan-sha256")
    parser.add_argument("--crossed-bounded-plan", type=Path)
    parser.add_argument("--crossed-bounded-plan-sha256")
    parser.add_argument("--crossed-arm", choices=("candidate", "control"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--schedule-total-steps", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--adam-beta1", type=float, required=True)
    parser.add_argument("--adam-beta2", type=float, required=True)
    parser.add_argument("--adam-eps", type=float, required=True)
    parser.add_argument("--max-grad-norm", type=float, required=True)
    parser.add_argument("--visual-lattice", type=int)
    parser.add_argument(
        "--curriculum-observation-mode",
        choices=CURRICULUM_OBSERVATION_MODES,
        default=CURRICULUM_OBSERVATION_DUAL_LATTICE,
    )
    parser.add_argument(
        "--calvin-factor-mode",
        choices=CALVIN_FACTOR_MODES,
        default=CALVIN_FACTOR_TARGET_ONLY,
    )
    parser.add_argument("--adr127-smoke", action="store_true")
    parser.add_argument("--adr128-smoke", action="store_true")
    parser.add_argument("--counterfactual-gradient-audit", action="store_true")
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_GPU_SHARDED,
    )
    parser.add_argument(
        "--cuda-allocator",
        choices=CUDA_ALLOCATOR_MODES,
        default="native",
        help="Explicit allocator mode configured before any PyTorch import.",
    )
    parser.add_argument("--export-max-shard-size", default="4GB")
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    if args.pair_plan is not None and args.visual_lattice is None:
        args.visual_lattice = LATTICE_BASELINE
    return args


def _validate_public_vl_retention_args(
    args: argparse.Namespace,
) -> PublicNativeVLRetentionManifest | None:
    retention_values = (
        args.public_vl_retention_manifest,
        args.public_vl_retention_manifest_sha256,
        args.public_vl_retention_root,
        args.public_vl_retention_weight,
    )
    if not any(value is not None for value in retention_values):
        return None
    if any(value is None for value in retention_values):
        raise ContractError("native VL public retention arguments must be all present")
    if args.curriculum_observation_mode != CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE:
        raise ContractError("native VL public retention requires official-native-once")
    if args.public_vl_retention_weight != PUBLIC_NATIVE_VL_RETENTION_WEIGHT:
        raise ContractError("native VL public retention weight differs from ADR-125")
    if not isinstance(args.public_vl_retention_manifest, Path):
        raise ContractError("native VL public retention manifest path is missing")
    if not isinstance(args.public_vl_retention_root, Path):
        raise ContractError("native VL public retention root is missing")
    if not isinstance(args.public_vl_retention_manifest_sha256, str):
        raise ContractError("native VL public retention manifest SHA-256 is missing")
    return load_frozen_public_native_vl_retention_gate(
        manifest_path=args.public_vl_retention_manifest,
        manifest_file_sha256=args.public_vl_retention_manifest_sha256,
        artifact_root=args.public_vl_retention_root,
        max_steps=args.max_steps,
    )


def _validate_args(args: argparse.Namespace) -> Path:
    plan_path = args.curriculum_plan if args.curriculum_plan is not None else args.pair_plan
    if plan_path is None:
        raise ContractError("native VL adaptation requires exactly one training plan")
    required_files = [
        args.training_config,
        args.dataset_manifest,
        plan_path,
        args.source_checkout / MODEL_SOURCE,
        _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH,
    ]
    if args.crossed_bounded_plan is not None:
        required_files.append(args.crossed_bounded_plan)
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.initial_qwen_dir,
        args.dataset_split,
        args.physical_sidecar_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    partial = args.output_dir.with_name(f"{args.output_dir.name}.partial")
    for path in (args.output_dir, partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    _validate_training_horizons(
        max_steps=args.max_steps,
        schedule_total_steps=args.schedule_total_steps,
        warmup_steps=args.warmup_steps,
    )
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ContractError("native VL adaptation integer arguments are invalid")
    if args.curriculum_plan is not None:
        if args.curriculum_plan_sha256 is None:
            raise ContractError("native VL adaptation curriculum requires its file SHA-256")
        _validate_sha256(args.curriculum_plan_sha256, name="curriculum file SHA-256")
        if args.visual_lattice is not None:
            raise ContractError("native VL curriculum owns its visual lattices")
    elif (
        args.curriculum_plan_sha256 is not None
        or args.curriculum_observation_mode != CURRICULUM_OBSERVATION_DUAL_LATTICE
        or isinstance(args.visual_lattice, bool)
        or not isinstance(args.visual_lattice, int)
        or args.visual_lattice <= 0
    ):
        raise ContractError("native VL pair-plan lattice arguments are invalid")
    for name, value, lower, upper, lower_inclusive in (
        ("learning rate", args.learning_rate, 0.0, math.inf, False),
        ("weight decay", args.weight_decay, 0.0, math.inf, True),
        ("Adam beta1", args.adam_beta1, 0.0, 1.0, True),
        ("Adam beta2", args.adam_beta2, 0.0, 1.0, True),
        ("Adam epsilon", args.adam_eps, 0.0, math.inf, False),
        ("maximum gradient norm", args.max_grad_norm, 0.0, math.inf, False),
    ):
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ContractError(f"native VL adaptation {name} must be numeric")
        numeric = float(value)
        if not math.isfinite(numeric) or numeric > upper:
            raise ContractError(f"native VL adaptation {name} is outside its valid range")
        if (lower_inclusive and numeric < lower) or (not lower_inclusive and numeric <= lower):
            raise ContractError(f"native VL adaptation {name} is outside its valid range")
    if not (args.adam_beta1 < 1.0 and args.adam_beta2 < 1.0):
        raise ContractError("native VL adaptation Adam betas must be below one")
    if not isinstance(args.export_max_shard_size, str) or not args.export_max_shard_size:
        raise ContractError("native VL adaptation export shard size must be nonempty text")
    _validate_git_revision(args.picf_code_revision)
    _validate_optional_qwen_restore(args.initial_qwen_dir, args.initial_qwen_revision)
    placement = validate_fsdp2_placement(args.fsdp2_placement)
    if args.cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ContractError("native VL adaptation CUDA allocator mode is unsupported")
    if placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
        raise ContractError("native VL tied embeddings cannot use selective embedding offload")
    args.public_vl_retention_manifest_object = _validate_public_vl_retention_args(args)
    _validate_calvin_factor_mode(
        args,
        public_retention_enabled=args.public_vl_retention_manifest_object is not None,
    )
    args.crossed_bounded_enabled = _validate_crossed_bounded_mode(
        args,
        public_retention_enabled=args.public_vl_retention_manifest_object is not None,
    )
    return partial


def _learning_rate_for_step(
    step: int,
    *,
    schedule_total_steps: int,
    warmup_steps: int,
    base_learning_rate: float,
) -> float:
    """Linear warmup followed by cosine decay, evaluated before each update."""

    if not 0 <= step < schedule_total_steps or not 0 <= warmup_steps < schedule_total_steps:
        raise ContractError("native VL adaptation schedule indices are invalid")
    if not math.isfinite(base_learning_rate) or base_learning_rate <= 0.0:
        raise ContractError("native VL adaptation base learning rate is invalid")
    if warmup_steps and step < warmup_steps:
        return base_learning_rate * step / warmup_steps
    decay_steps = schedule_total_steps - warmup_steps
    progress = (step - warmup_steps) / max(1, decay_steps)
    return base_learning_rate * 0.5 * (1.0 + math.cos(math.pi * progress))


def _validate_crossed_materialized_record(
    evidence: CrossedBoundedRecord,
    record: CalvinQwenGroundingRecord,
) -> None:
    """Rebind one materialized record to every frozen model-visible field."""

    if not isinstance(evidence, CrossedBoundedRecord) or not isinstance(
        record, CalvinQwenGroundingRecord
    ):
        raise TypeError("crossed materialization requires typed evidence and record")
    if (
        record.global_index != evidence.global_index
        or record.task_key != evidence.task_key
        or hashlib.sha256(record.instruction.encode("utf-8")).hexdigest()
        != evidence.instruction_sha256
        or record.target_identity_key != evidence.target_identity_key
        or record.camera_name != evidence.camera_name
        or record.source_rgb_sha256 != evidence.source_rgb_sha256
        or record.qwen_bbox_xyxy != evidence.bbox_qwen_xyxy
    ):
        raise ContractError("materialized crossed record differs from its frozen evidence")


def _distributed_gradient_metrics(
    model: Any,
    *,
    device: Any,
    dist: Any,
    torch_module: Any,
    max_grad_norm: float,
) -> dict[str, float | int | bool]:
    squared = torch_module.zeros((), dtype=torch_module.float64, device=device)
    trainable_elements = 0
    frozen_gradient_elements = 0
    finite = torch_module.ones((), dtype=torch_module.int32, device=device)
    gradients = []
    for parameter in model.parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        local = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
        if not parameter.requires_grad:
            frozen_gradient_elements += int(local.numel())
            continue
        finite.mul_(torch_module.isfinite(local).all().to(dtype=torch_module.int32))
        squared.add_(local.detach().float().square().sum().to(dtype=torch_module.float64))
        trainable_elements += int(local.numel())
        gradients.append(gradient)
    packed = torch_module.tensor(
        [float(trainable_elements), float(frozen_gradient_elements)],
        dtype=torch_module.float64,
        device=device,
    )
    dist.all_reduce(squared, op=dist.ReduceOp.SUM)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    norm = math.sqrt(float(squared.item()))
    coefficient = min(1.0, max_grad_norm / (norm + 1e-6))
    if coefficient < 1.0:
        for gradient in gradients:
            gradient.mul_(coefficient)
    return {
        "all_finite": bool(finite.item()),
        "clip_coefficient": coefficient,
        "frozen_gradient_elements": int(packed[1].item()),
        "global_norm_before_clip": norm,
        "trainable_gradient_elements": int(packed[0].item()),
    }


def _export_qwen_candidate(
    policy: Any,
    *,
    candidate_dir: Path,
    max_shard_size: str,
    rank: int,
) -> dict[str, str]:
    from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict

    qwen = policy.model.qwenvl_with_expert.qwenvl
    state = get_model_state_dict(
        policy,
        submodules={qwen},
        options=StateDictOptions(
            full_state_dict=True,
            cpu_offload=True,
            keep_submodule_prefixes=False,
        ),
    )
    if rank != 0:
        if state:
            raise RuntimeError("nonzero rank unexpectedly received the full Qwen state")
        return {}
    if not state or not any(name.startswith("model.language_model.") for name in state):
        raise RuntimeError("Qwen export omitted its language model")
    if any("qwen_expert" in name or "picf_native_graph" in name for name in state):
        raise RuntimeError("Qwen-only export contains a frozen action/PICF tensor")
    candidate_dir.mkdir()
    qwen.save_pretrained(
        candidate_dir,
        state_dict=state,
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    del state
    return _model_hashes(candidate_dir)


def main() -> None:
    args = _parse_args()
    partial = _validate_args(args)
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("CUDA allocator pre-bootstrap differs from parsed arguments")
    patch_report = verify_native_vl_patch(root=_ROOT, checkout=args.source_checkout)
    overlay = _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH
    if detect_native_vl_patch_state(args.source_checkout, overlay) != "applied":
        raise RuntimeError("native VL adaptation source overlay is not applied")
    if _validate_native_vl_model(args.source_checkout / MODEL_SOURCE) != (
        NATIVE_VL_PATCHED_MODEL_SHA256
    ):
        raise RuntimeError("native VL adaptation source digest differs")
    commit = subprocess.run(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("native VL adaptation source commit differs")
    picf_commit = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if picf_commit != args.picf_code_revision:
        raise RuntimeError("native VL adaptation checkout differs from its declared PICF revision")
    runtime_python_trees = adr127_runtime_python_trees_contract(
        repo_root=_ROOT,
        revision=args.picf_code_revision,
        source_checkout=args.source_checkout,
    )
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("native VL adaptation requires exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("native VL adaptation requires two local GPUs")

    sys.path.insert(0, str(args.source_checkout.resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from transformers import AutoConfig
    from transformers.modeling_utils import load_sharded_checkpoint, no_init_weights

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.fixed_observation import FixedObservationPairPlan
    from picf_next.lingbot_native.vl_curriculum import NativeVLGroundingCurriculumPlan
    from picf_next.lingbot_native.vl_cotraining import (
        build_counterfactual_scene_grounding_records,
        build_native_vl_grounding_batch,
        configure_native_vl_grounding_trainable_scope,
        materialize_fixed_observation_native_vl_record,
        materialize_fixed_observation_native_vl_records,
        register_native_vl_fsdp_forward_method,
        retie_and_validate_native_qwen_lm_head,
        run_native_vl_grounding_forward,
        validate_native_vl_optimizer_membership,
        verify_native_vl_grounding_trainable_scope,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError("native VL adaptation sees an unexpected CUDA topology")
        init_parallel_state(
            dp_size=WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )
        manifest = load_dataset_file_manifest(args.dataset_manifest)
        validate_dataset_runtime_binding(
            manifest,
            args.dataset_split,
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            split_name=args.dataset_split.name,
        )
        pair_plan = None
        curriculum_plan = None
        crossed_plan = None
        if args.curriculum_plan is not None:
            curriculum_file_sha256 = _sha256(args.curriculum_plan)
            if curriculum_file_sha256 != args.curriculum_plan_sha256:
                raise ContractError("native VL adaptation curriculum file SHA-256 changed")
            curriculum_plan = NativeVLGroundingCurriculumPlan.load(args.curriculum_plan)
            plan_identity = (
                curriculum_plan.dataset_id,
                curriculum_plan.dataset_revision,
                curriculum_plan.dataset_manifest_sha256,
            )
            available_steps = len(curriculum_plan.steps)
            initial_visual_lattice = curriculum_plan.steps[0].batches[0].visual_lattice
            if args.crossed_bounded_enabled:
                if (
                    not isinstance(args.crossed_bounded_plan, Path)
                    or not isinstance(args.crossed_bounded_plan_sha256, str)
                    or args.crossed_arm not in {"candidate", "control"}
                ):
                    raise RuntimeError("native VL adaptation lost its crossed plan arguments")
                crossed_file_sha256 = _sha256(args.crossed_bounded_plan)
                if crossed_file_sha256 != args.crossed_bounded_plan_sha256:
                    raise ContractError("native VL crossed bounded plan file SHA-256 changed")
                crossed_plan = CrossedBoundedPlan.load(args.crossed_bounded_plan)
                if (
                    crossed_plan.picf_code_revision != args.picf_code_revision
                    or crossed_plan.curriculum_file_sha256 != curriculum_file_sha256
                    or crossed_plan.curriculum_artifact_sha256 != curriculum_plan.artifact_sha256
                    or (
                        crossed_plan.dataset_id,
                        crossed_plan.dataset_revision,
                        crossed_plan.dataset_manifest_sha256,
                    )
                    != plan_identity
                ):
                    raise ContractError(
                        "native VL crossed bounded plan differs from its runtime provenance"
                    )
                available_steps = len(crossed_plan.steps)
                initial_visual_lattice = LATTICE_BASELINE
        else:
            if args.pair_plan is None or args.visual_lattice is None:
                raise ContractError("native VL adaptation pair-plan arguments are incomplete")
            pair_plan = FixedObservationPairPlan.load(args.pair_plan)
            plan_identity = (
                pair_plan.dataset_id,
                pair_plan.dataset_revision,
                pair_plan.dataset_manifest_sha256,
            )
            available_steps = len(pair_plan.pairs)
            initial_visual_lattice = args.visual_lattice
        if plan_identity != (
            manifest.dataset_id,
            manifest.dataset_revision,
            manifest.tree_sha256,
        ):
            raise ContractError("native VL adaptation training plan belongs to another dataset")
        if args.max_steps > available_steps:
            raise ContractError("native VL adaptation requests more steps than its frozen plan")
        index = CalvinDatasetIndex.load(
            args.dataset_split,
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        sidecar = CalvinPhysicalSupervisionSidecar(args.physical_sidecar_root, index)
        retention_manifest = args.public_vl_retention_manifest_object
        if retention_manifest is not None and not isinstance(
            retention_manifest, PublicNativeVLRetentionManifest
        ):
            raise RuntimeError("native VL adaptation lost its typed public retention manifest")

        crossed_materialization_report = None
        if crossed_plan is not None:
            if curriculum_plan is None:
                raise RuntimeError("native VL crossed plan lost its source curriculum")
            planned_records = tuple(
                sorted(
                    {
                        record
                        for step in crossed_plan.steps
                        for records in (step.candidate_records, step.control_records)
                        for record in records
                    },
                    key=lambda record: (
                        record.group_index,
                        record.variant_index,
                        record.camera_name,
                        record.instruction_sha256,
                    ),
                )
            )
            local_evidence = planned_records[rank::WORLD_SIZE]
            local_status: dict[str, object]
            try:
                for evidence in local_evidence:
                    group, variant = crossed_plan.resolve_record(
                        curriculum_plan.groups,
                        evidence,
                    )
                    materialized = materialize_fixed_observation_native_vl_record(
                        index=index,
                        sidecar=sidecar,
                        group=group,
                        variant=variant,
                        expected_camera_name=evidence.camera_name,
                    )
                    _validate_crossed_materialized_record(evidence, materialized)
                materialized_payload = json.dumps(
                    [record.as_dict() for record in local_evidence],
                    allow_nan=False,
                    ensure_ascii=True,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("ascii")
                local_status = {
                    "count": len(local_evidence),
                    "rank": rank,
                    "record_set_sha256": hashlib.sha256(materialized_payload).hexdigest(),
                    "status": "PASS",
                }
            except Exception as error:  # Every rank must reach the collective failure gate.
                local_status = {
                    "error": f"{type(error).__name__}: {error}",
                    "rank": rank,
                    "status": "FAIL",
                }
            materialization_reports: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(materialization_reports, local_status)
            if any(report.get("status") != "PASS" for report in materialization_reports):
                raise ContractError(
                    f"native VL crossed CPU materialization failed: {materialization_reports}"
                )
            if sum(int(report["count"]) for report in materialization_reports) != len(
                planned_records
            ):
                raise ContractError("native VL crossed materialization coverage changed")
            crossed_materialization_report = {
                "rank_reports": materialization_reports,
                "unique_record_count": len(planned_records),
            }

        training = load_lingbot_training_config(args.training_config)
        train_values = training.get("train")
        if not isinstance(train_values, dict):
            raise ContractError("native VL adaptation training config has no train mapping")
        released_lr = train_values.get("lr", 5e-5)
        if isinstance(released_lr, bool) or not isinstance(released_lr, int | float):
            raise ContractError("native VL adaptation released learning rate is invalid")
        runtime_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(released_lr),
        )
        merged, _ = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=args.max_steps,
        )
        merged.update(
            {
                "attention_implementation": "eager",
                "use_cache": False,
                "use_compile": False,
                "use_lm_head": True,
                "vit_attn_implementation": "eager",
            }
        )
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        # QWEN_PROCESSOR_REVISION is an exact commit and this load is local-only.
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())
        config.use_lm_head = True

        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        processor = build_processor(str(args.processor_dir.resolve()))
        image_processor = getattr(processor, "image_processor", None)
        snapshot_size = getattr(image_processor, "size", None)
        if not isinstance(snapshot_size, dict):
            raise RuntimeError("native VL processor snapshot has no resolution mapping")
        processor_snapshot_size = dict(snapshot_size)
        processor_lattice = configure_native_processor_lattice(
            processor,
            initial_visual_lattice,
        )
        configured_size = getattr(image_processor, "size", None)
        if args.curriculum_observation_mode == CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE and (
            not isinstance(configured_size, dict) or processor_snapshot_size != configured_size
        ):
            raise RuntimeError("official-native-once changed the pinned processor snapshot")
        processor_lattices = {str(initial_visual_lattice): processor_lattice}
        retention_processor = None
        retention_processor_contract = None
        if retention_manifest is not None:
            retention_processor = build_processor(str(args.processor_dir.resolve()))
            retention_processor_contract = configure_native_processor_area_budget(
                retention_processor,
                LATTICE_BASELINE,
            )
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        load_started = time.perf_counter()
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        preload_tied_name = retie_and_validate_native_qwen_lm_head(policy)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        loaded_tied_name = retie_and_validate_native_qwen_lm_head(policy)
        if loaded_tied_name != preload_tied_name:
            raise ContractError("native VL tied parameter name changed during released load")
        restore_result = _validate_qwen_restore_load_result(
            load_sharded_checkpoint(
                policy.model.qwenvl_with_expert.qwenvl,
                args.initial_qwen_dir,
                strict=False,
                prefer_safe=True,
            )
        )
        restored_tied_name = retie_and_validate_native_qwen_lm_head(policy)
        if restored_tied_name != loaded_tied_name:
            raise ContractError("native VL tied parameter name changed during Qwen restoration")
        teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        initial_scope = configure_native_vl_grounding_trainable_scope(policy)
        policy.train()
        full_cpu_offload = args.fsdp2_placement == FSDP2_CPU_OFFLOAD
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=runtime_contract.enable_mixed_precision,
            enable_fp32=runtime_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=full_cpu_offload,
            enable_shared_embedding_offload=False,
            fsdp_kwargs={},
            basic_modules=policy._no_split_modules,
            enable_reentrant=False,
            enable_forward_prefetch=False,
            fsdp_llm_blocks=False,
            ignore_norm=False,
            use_depth_align=False,
            split_fused_experts_from_decoder_fsdp=False,
            vlm_fsdp=True,
            use_future_image=False,
        )
        register_native_fsdp_forward_methods(policy)
        register_native_vl_fsdp_forward_method(policy)
        sharded_scope = verify_native_vl_grounding_trainable_scope(
            policy,
            expected=initial_scope,
        )
        trainable_parameters = tuple(
            parameter for parameter in policy.parameters() if parameter.requires_grad
        )
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_eps,
            weight_decay=args.weight_decay,
            foreach=False,
            fused=False,
        )
        optimizer_tied_name = validate_native_vl_optimizer_membership(policy, optimizer)
        load_seconds = time.perf_counter() - load_started

        def run_counterfactual_gradient_audit(
            *,
            step: int,
            source_group: Any,
            microbatches: tuple[Any, ...],
        ) -> dict[str, Any] | None:
            """Audit isolated objective gradients without changing weights or RNG state."""

            if (
                len(microbatches) != 1
                or microbatches[0][0] != LATTICE_BASELINE
                or retention_manifest is None
                or retention_processor is None
                or not isinstance(args.public_vl_retention_root, Path)
            ):
                raise ContractError("native VL ADR-127 gradient audit runtime changed")
            visual_lattice, camera_name, variants = microbatches[0]
            if source_group is None:
                raise ContractError("native VL gradient audit lost its fixed source group")
            configure_native_processor_lattice(processor, visual_lattice)
            target_records = materialize_fixed_observation_native_vl_records(
                index=index,
                sidecar=sidecar,
                group=source_group,
                variants=_require_variant_pair(variants),
                expected_camera_name=camera_name,
            )
            scene_records = build_counterfactual_scene_grounding_records(
                target_records,
                sidecar.source_frame(source_group.source_global_index),
                visual_lattice=visual_lattice,
            )
            retention_descriptor = retention_manifest.training_record_for_rank(
                optimizer_step=step,
                rank=rank,
            )
            retention_record = retention_manifest.materialize_record(
                retention_descriptor,
                artifact_root=args.public_vl_retention_root,
            )
            objective_records = (
                ("target", target_records[rank], processor),
                ("scene", scene_records[rank], processor),
                ("public", retention_record, retention_processor),
            )

            python_rng_state = random.getstate()
            numpy_rng_state = np.random.get_state()
            torch_rng_state = torch.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all()
            target_gradients = None
            scene_gradients = None
            target_scene_names = None
            target_scene_rows = None
            target_public_names = None
            target_public_rows = None
            scene_public_names = None
            scene_public_rows = None
            objective_reports = []
            audit_started = time.perf_counter()
            try:
                for objective_name, record, record_processor in objective_records:
                    audit_seed = args.seed + 30_000_000 + step * WORLD_SIZE + rank
                    random.seed(audit_seed)
                    np.random.seed(audit_seed)
                    torch.manual_seed(audit_seed)
                    torch.cuda.manual_seed(audit_seed)
                    policy.zero_grad(set_to_none=True)
                    batch = build_native_vl_grounding_batch(record, record_processor)
                    image_grid_thw = batch.image_grid_thw.detach().cpu().tolist()
                    if objective_name == "public":
                        grid_contract = validate_native_processor_record_grid(
                            image_grid_thw,
                            image_height=retention_descriptor.height,
                            image_width=retention_descriptor.width,
                            lattice=LATTICE_BASELINE,
                        )
                    else:
                        expected_grid = [[1, visual_lattice * 2, visual_lattice * 2]]
                        if image_grid_thw != expected_grid:
                            raise RuntimeError(
                                "native VL ADR-127 gradient audit image grid changed"
                            )
                        grid_contract = {"image_grid_thw": image_grid_thw}
                    batch = batch.to(device, pixel_dtype=torch.bfloat16)
                    objective_started = time.perf_counter()
                    loss = run_native_vl_grounding_forward(policy, batch)
                    loss.backward()
                    local_report = {
                        "assistant_text_sha256": hashlib.sha256(
                            record.assistant_text.encode("utf-8")
                        ).hexdigest(),
                        "elapsed_seconds": time.perf_counter() - objective_started,
                        "grid_contract": grid_contract,
                        "image_grid_thw": image_grid_thw,
                        "loss": float(loss.detach().float().item()),
                        "objective": objective_name,
                        "rank": rank,
                        "supervised_token_count": batch.supervised_token_count,
                    }
                    if objective_name == "public":
                        if not isinstance(record, NativeVLInstructionRecord):
                            raise ContractError("native VL public gradient record type changed")
                        local_report.update(
                            {
                                "image_rgb_sha256": retention_descriptor.image_rgb_sha256,
                                "record_id": retention_descriptor.record_id,
                                "record_sha256": retention_descriptor.record_sha256,
                                "user_text_sha256": hashlib.sha256(
                                    record.user_text.encode("utf-8")
                                ).hexdigest(),
                            }
                        )
                    else:
                        if not isinstance(
                            record,
                            CalvinQwenGroundingRecord | CalvinQwenSceneGroundingRecord,
                        ):
                            raise ContractError("native VL CALVIN gradient record type changed")
                        local_report.update(
                            {
                                "camera_name": record.camera_name,
                                "global_index": record.global_index,
                                "record_type": objective_name,
                                "source_rgb_sha256": record.source_rgb_sha256,
                                "user_text_sha256": hashlib.sha256(
                                    record.grounding_request.encode("utf-8")
                                ).hexdigest(),
                                "visual_lattice": visual_lattice,
                            }
                        )
                    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
                    dist.all_gather_object(gathered, local_report)
                    if rank == 0:
                        objective_reports.append({"objective": objective_name, "ranks": gathered})

                    if objective_name == "target":
                        target_gradients = snapshot_local_gradients(
                            policy,
                            torch_module=torch,
                        )
                    elif objective_name == "scene":
                        if target_gradients is None:
                            raise RuntimeError("native VL ADR-127 audit omitted target gradients")
                        target_scene_names, target_scene_rows = distributed_pair_rows(
                            policy,
                            first_gradients=target_gradients,
                            device=device,
                            dist=dist,
                            torch_module=torch,
                        )
                        scene_gradients = snapshot_local_gradients(
                            policy,
                            torch_module=torch,
                        )
                    elif objective_name == "public":
                        if target_gradients is None or scene_gradients is None:
                            raise RuntimeError("native VL ADR-127 audit omitted CALVIN gradients")
                        target_public_names, target_public_rows = distributed_pair_rows(
                            policy,
                            first_gradients=target_gradients,
                            device=device,
                            dist=dist,
                            torch_module=torch,
                        )
                        scene_public_names, scene_public_rows = distributed_pair_rows(
                            policy,
                            first_gradients=scene_gradients,
                            device=device,
                            dist=dist,
                            torch_module=torch,
                        )
                    del batch, loss
                if any(
                    value is None
                    for value in (
                        target_scene_names,
                        target_scene_rows,
                        target_public_names,
                        target_public_rows,
                        scene_public_names,
                        scene_public_rows,
                    )
                ):
                    raise RuntimeError("native VL ADR-127 audit omitted gradient pair moments")
                assert target_scene_names is not None
                assert target_scene_rows is not None
                assert target_public_names is not None
                assert target_public_rows is not None
                assert scene_public_names is not None
                assert scene_public_rows is not None
                summary = _summarize_gradient_triple_rows(
                    target_scene_names=target_scene_names,
                    target_scene_rows=target_scene_rows,
                    target_public_names=target_public_names,
                    target_public_rows=target_public_rows,
                    scene_public_names=scene_public_names,
                    scene_public_rows=scene_public_rows,
                )
            finally:
                policy.zero_grad(set_to_none=True)
                random.setstate(python_rng_state)
                np.random.set_state(numpy_rng_state)
                torch.set_rng_state(torch_rng_state)
                torch.cuda.set_rng_state_all(cuda_rng_states)
                del target_gradients, scene_gradients
                gc.collect()
                torch.cuda.empty_cache()
            if rank != 0:
                return None
            global_summary = summary.get("global")
            if not isinstance(global_summary, dict):
                raise RuntimeError("native VL ADR-127 gradient summary has no global surface")
            descends = global_summary.get("mixed_gradient_descends")
            if not isinstance(descends, dict):
                raise RuntimeError("native VL ADR-127 gradient summary has no directions")
            return {
                "completed_updates_before_audit": step,
                "elapsed_seconds": time.perf_counter() - audit_started,
                "objective_rank_reports": objective_reports,
                "status": (
                    "PASS"
                    if descends.get("target") is True and descends.get("scene") is True
                    else "FAIL"
                ),
                "summary": summary,
            }

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        training_started = time.perf_counter()
        step_reports = []
        counterfactual_gradient_reports = []
        for step in range(args.max_steps):
            crossed_step = None
            planned_evidence = None
            if crossed_plan is not None:
                if curriculum_plan is None or args.crossed_arm not in {
                    "candidate",
                    "control",
                }:
                    raise RuntimeError("native VL crossed runtime lost its frozen inputs")
                crossed_step = crossed_plan.steps[step]
                planned_evidence = crossed_step.records_for_arm(args.crossed_arm)
                source_group = None
                microbatches = (
                    (
                        LATTICE_BASELINE,
                        planned_evidence[0].camera_name,
                        None,
                    ),
                )
                step_plan_metadata = {
                    "crossed_arm": args.crossed_arm,
                    "crossed_cell": crossed_step.cell,
                    "crossed_plan_optimizer_step": crossed_step.optimizer_step,
                    "observation_mode": args.curriculum_observation_mode,
                }
            elif curriculum_plan is not None:
                source_group, curriculum_batches = curriculum_plan.resolve_step(step)
                microbatches = _select_curriculum_microbatches(
                    curriculum_batches,
                    observation_mode=args.curriculum_observation_mode,
                )
                step_plan_metadata = {
                    "curriculum_group_index": curriculum_plan.steps[step].group_index,
                    "curriculum_optimizer_step": curriculum_plan.steps[step].optimizer_step,
                    "observation_mode": args.curriculum_observation_mode,
                }
            else:
                if pair_plan is None or args.visual_lattice is None:
                    raise RuntimeError("native VL adaptation lost its loaded pair plan")
                pair = pair_plan.pairs[step]
                source_group = pair.group
                microbatches = ((args.visual_lattice, None, pair.variants),)
                step_plan_metadata = {
                    "observation_mode": PAIR_PLAN_OBSERVATION_SINGLE_LATTICE,
                    "pair_plan_optimizer_step": pair.optimizer_step,
                }
            learning_rate = _learning_rate_for_step(
                step,
                schedule_total_steps=args.schedule_total_steps,
                warmup_steps=args.warmup_steps,
                base_learning_rate=args.learning_rate,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            if args.counterfactual_gradient_audit and step in ADR127_GRADIENT_AUDIT_STEPS:
                gradient_report = run_counterfactual_gradient_audit(
                    step=step,
                    source_group=source_group,
                    microbatches=microbatches,
                )
                if rank == 0:
                    if gradient_report is None:
                        raise RuntimeError("native VL ADR-127 rank zero lost its gradient report")
                    counterfactual_gradient_reports.append(gradient_report)
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            microbatch_reports = []
            for visual_lattice, camera_name, variants in microbatches:
                lattice_contract = configure_native_processor_lattice(
                    processor,
                    visual_lattice,
                )
                lattice_key = str(visual_lattice)
                previous_contract = processor_lattices.get(lattice_key)
                if previous_contract is not None and previous_contract != lattice_contract:
                    raise RuntimeError("native VL processor lattice contract changed in-run")
                processor_lattices[lattice_key] = lattice_contract
                evidence = None
                if crossed_plan is not None:
                    if curriculum_plan is None or planned_evidence is None:
                        raise RuntimeError("native VL crossed step lost its evidence")
                    evidence = planned_evidence[rank]
                    group, variant = crossed_plan.resolve_record(
                        curriculum_plan.groups,
                        evidence,
                    )
                    target_record = materialize_fixed_observation_native_vl_record(
                        index=index,
                        sidecar=sidecar,
                        group=group,
                        variant=variant,
                        expected_camera_name=evidence.camera_name,
                    )
                    _validate_crossed_materialized_record(evidence, target_record)
                    records = None
                else:
                    if source_group is None:
                        raise ContractError("native VL materialization lost its fixed source group")
                    records = materialize_fixed_observation_native_vl_records(
                        index=index,
                        sidecar=sidecar,
                        group=source_group,
                        variants=_require_variant_pair(variants),
                        expected_camera_name=camera_name,
                    )
                    target_record = records[rank]
                if camera_name is not None and target_record.camera_name != camera_name:
                    raise RuntimeError("native VL adaptation materialized an unplanned camera")
                factor_weights = _calvin_factor_weights(args.calvin_factor_mode)
                if args.calvin_factor_mode == CALVIN_FACTOR_COUNTERFACTUAL_SCENE_CANDIDATE:
                    if records is None or source_group is None:
                        raise ContractError("native VL scene factor requires one fixed source pair")
                    scene_records = build_counterfactual_scene_grounding_records(
                        records,
                        sidecar.source_frame(source_group.source_global_index),
                        visual_lattice=visual_lattice,
                    )
                    scene_record = scene_records[rank]
                    factor_records = {
                        "target": target_record,
                        "scene": scene_record,
                    }
                else:
                    scene_records = None
                    factor_records = {
                        "target": target_record,
                        "target_repeat": target_record,
                    }
                factor_reports = []
                for factor_name, factor_weight in factor_weights:
                    factor_record = factor_records[factor_name]
                    batch = build_native_vl_grounding_batch(factor_record, processor).to(
                        device,
                        pixel_dtype=torch.bfloat16,
                    )
                    image_grid_thw = batch.image_grid_thw.detach().cpu().tolist()
                    expected_grid_thw = [[1, visual_lattice * 2, visual_lattice * 2]]
                    if image_grid_thw != expected_grid_thw:
                        raise RuntimeError(
                            "native VL adaptation image grid differs from its declared lattice"
                        )
                    factor_started = time.perf_counter()
                    loss = run_native_vl_grounding_forward(policy, batch)
                    effective_weight = factor_weight / len(microbatches)
                    (effective_weight * loss).backward()
                    local_report = {
                        "assistant_text_sha256": hashlib.sha256(
                            factor_record.assistant_text.encode("utf-8")
                        ).hexdigest(),
                        "camera_name": factor_record.camera_name,
                        "elapsed_seconds": time.perf_counter() - factor_started,
                        "factor_name": factor_name,
                        "global_index": factor_record.global_index,
                        "image_grid_thw": image_grid_thw,
                        "loss": float(loss.detach().float().item()),
                        "loss_weight": effective_weight,
                        "rank": rank,
                        "record_type": ("scene" if factor_name == "scene" else "target"),
                        "source_rgb_sha256": factor_record.source_rgb_sha256,
                        "supervised_token_count": batch.supervised_token_count,
                        "user_text_sha256": hashlib.sha256(
                            factor_record.grounding_request.encode("utf-8")
                        ).hexdigest(),
                        "visual_lattice": visual_lattice,
                    }
                    if factor_name == "scene":
                        if not isinstance(factor_record, CalvinQwenSceneGroundingRecord):
                            raise ContractError("native VL scene factor record type changed")
                        local_report.update(
                            {
                                "absent_identity_keys": list(factor_record.absent_identity_keys),
                                "category_identity_order": list(
                                    factor_record.category_identity_order
                                ),
                                "minimum_projected_target_mass": (
                                    factor_record.minimum_projected_target_mass
                                ),
                                "object_identity_keys": [
                                    item.identity_key for item in factor_record.objects
                                ],
                                "object_projected_target_masses": [
                                    item.projected_target_mass for item in factor_record.objects
                                ],
                                "object_visible_owner_pixels": [
                                    item.visible_owner_pixels for item in factor_record.objects
                                ],
                                "subpatch_visible_identity_keys": list(
                                    factor_record.subpatch_visible_identity_keys
                                ),
                                "subpatch_projected_target_masses": [
                                    item.projected_target_mass
                                    for item in factor_record.subpatch_objects
                                ],
                                "subpatch_visible_owner_pixels": [
                                    item.visible_owner_pixels
                                    for item in factor_record.subpatch_objects
                                ],
                            }
                        )
                    else:
                        if not isinstance(factor_record, CalvinQwenGroundingRecord):
                            raise ContractError("native VL target factor record type changed")
                        local_report.update(
                            {
                                "instruction": factor_record.instruction,
                                "target_identity_key": factor_record.target_identity_key,
                                "task_key": factor_record.task_key,
                            }
                        )
                        if evidence is not None:
                            local_report.update(
                                {
                                    "crossed_bbox_qwen_xyxy": list(evidence.bbox_qwen_xyxy),
                                    "crossed_group_index": evidence.group_index,
                                    "crossed_instruction_sha256": (evidence.instruction_sha256),
                                    "crossed_variant_index": evidence.variant_index,
                                }
                            )
                    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
                    dist.all_gather_object(gathered, local_report)
                    if rank == 0:
                        factor_reports.append(
                            {
                                "factor_name": factor_name,
                                "loss_weight": effective_weight,
                                "ranks": gathered,
                            }
                        )
                    del batch, loss
                if rank == 0:
                    microbatch_reports.append(
                        {
                            "factors": factor_reports,
                            "visual_lattice": visual_lattice,
                        }
                    )
                del factor_records, records, scene_records, target_record
            retention_step_report = None
            if retention_manifest is not None:
                if retention_processor is None or retention_processor_contract is None:
                    raise RuntimeError("native VL adaptation lost its public retention processor")
                retention_record_descriptor = retention_manifest.training_record_for_rank(
                    optimizer_step=step,
                    rank=rank,
                )
                retention_record = retention_manifest.materialize_record(
                    retention_record_descriptor,
                    artifact_root=args.public_vl_retention_root,
                )
                retention_batch = build_native_vl_grounding_batch(
                    retention_record,
                    retention_processor,
                )
                retention_grid_thw = retention_batch.image_grid_thw.detach().cpu().tolist()
                retention_grid_budget = validate_native_processor_record_grid(
                    retention_grid_thw,
                    image_height=retention_record_descriptor.height,
                    image_width=retention_record_descriptor.width,
                    lattice=LATTICE_BASELINE,
                )
                retention_batch = retention_batch.to(
                    device,
                    pixel_dtype=torch.bfloat16,
                )
                retention_started = time.perf_counter()
                retention_loss = run_native_vl_grounding_forward(policy, retention_batch)
                (PUBLIC_NATIVE_VL_RETENTION_WEIGHT * retention_loss).backward()
                local_retention_report = {
                    "assistant_text_sha256": hashlib.sha256(
                        retention_record_descriptor.assistant_text.encode("utf-8")
                    ).hexdigest(),
                    "elapsed_seconds": time.perf_counter() - retention_started,
                    "family": retention_record.family,
                    "grid_budget": retention_grid_budget,
                    "image_height": retention_record_descriptor.height,
                    "image_rgb_sha256": retention_record_descriptor.image_rgb_sha256,
                    "image_grid_thw": retention_grid_thw,
                    "image_width": retention_record_descriptor.width,
                    "loss": float(retention_loss.detach().float().item()),
                    "loss_weight": PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
                    "rank": rank,
                    "record_id": retention_record.record_id,
                    "record_sha256": retention_record_descriptor.record_sha256,
                    "source_row_index": retention_record_descriptor.source_row_index,
                    "source_subindex": retention_record_descriptor.source_subindex,
                    "supervised_token_count": retention_batch.supervised_token_count,
                    "target_answer_sha256": hashlib.sha256(
                        retention_record_descriptor.assistant_text.encode("utf-8")
                    ).hexdigest(),
                    "user_text": retention_record.user_text,
                    "user_text_sha256": hashlib.sha256(
                        retention_record.user_text.encode("utf-8")
                    ).hexdigest(),
                }
                gathered_retention: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(gathered_retention, local_retention_report)
                if rank == 0:
                    retention_step_report = {"ranks": gathered_retention}
                del retention_batch, retention_loss, retention_record
            gradient_metrics = _distributed_gradient_metrics(
                policy,
                device=device,
                dist=dist,
                torch_module=torch,
                max_grad_norm=args.max_grad_norm,
            )
            if not bool(gradient_metrics["all_finite"]):
                raise RuntimeError("native VL adaptation gradients are non-finite")
            if (
                int(gradient_metrics["trainable_gradient_elements"])
                != sharded_scope.trainable_numel
            ):
                raise RuntimeError(
                    "native VL adaptation did not cover its complete trainable scope"
                )
            if int(gradient_metrics["frozen_gradient_elements"]) != 0:
                raise RuntimeError("native VL adaptation produced frozen-host gradients")
            optimizer.step()
            elapsed = time.perf_counter() - step_started
            if rank == 0:
                step_reports.append(
                    {
                        "elapsed_seconds": elapsed,
                        "gradient_metrics": gradient_metrics,
                        "learning_rate": learning_rate,
                        "microbatches": microbatch_reports,
                        "optimizer_step": step,
                        "public_vl_retention": retention_step_report,
                        **step_plan_metadata,
                    }
                )
                print(
                    json.dumps(
                        {
                            "event": "native_vl_grounding_step",
                            "learning_rate": learning_rate,
                            "calvin_factor_losses": [
                                {
                                    factor["factor_name"]: [
                                        item["loss"] for item in factor["ranks"]
                                    ]
                                    for factor in report["factors"]
                                }
                                for report in microbatch_reports
                            ],
                            "public_vl_retention_losses": (
                                None
                                if retention_step_report is None
                                else [item["loss"] for item in retention_step_report["ranks"]]
                            ),
                            "step": step,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        training_seconds = time.perf_counter() - training_started
        optimizer_state_parameter_count = len(optimizer.state)
        if optimizer_state_parameter_count <= 0:
            raise RuntimeError("native VL adaptation optimizer created no update state")
        local_training_memory = {
            "allocated_gib": torch.cuda.memory_allocated(device) / (1024**3),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "rank": rank,
            "reserved_gib": torch.cuda.memory_reserved(device) / (1024**3),
        }
        training_memory: list[Any] = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(training_memory, local_training_memory)

        if rank == 0:
            partial.mkdir()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats(device)
        candidate_hashes = _export_qwen_candidate(
            policy,
            candidate_dir=partial / "qwen_candidate",
            max_shard_size=args.export_max_shard_size,
            rank=rank,
        )
        dist.barrier()
        local_export_memory = {
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "rank": rank,
        }
        export_memory: list[Any] = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(export_memory, local_export_memory)
        if rank == 0:
            if (
                adr127_runtime_python_trees_contract(
                    repo_root=_ROOT,
                    revision=args.picf_code_revision,
                    source_checkout=args.source_checkout,
                )
                != runtime_python_trees
            ):
                raise ContractError("native VL adaptation runtime source changed during execution")
            initial_hashes = _model_hashes(args.initial_qwen_dir)
            if crossed_plan is not None:
                if (
                    curriculum_plan is None
                    or args.curriculum_plan is None
                    or args.crossed_bounded_plan is None
                    or args.crossed_arm not in {"candidate", "control"}
                ):
                    raise RuntimeError("native VL adaptation lost its crossed plan binding")
                training_plan_report = {
                    "arm": args.crossed_arm,
                    "artifact_sha256": crossed_plan.artifact_sha256,
                    "bounded_training_authorized": True,
                    "file_sha256": _sha256(args.crossed_bounded_plan),
                    "long_training_authorized": False,
                    "observation_mode": args.curriculum_observation_mode,
                    "source_curriculum_artifact_sha256": curriculum_plan.artifact_sha256,
                    "source_curriculum_file_sha256": _sha256(args.curriculum_plan),
                    "summary": crossed_plan.summary,
                    "type": "crossed_bounded_candidate_control",
                    "visual_lattices": [LATTICE_BASELINE],
                }
            elif curriculum_plan is not None:
                if args.curriculum_plan is None:
                    raise RuntimeError("native VL adaptation lost its curriculum path")
                training_plan_report = {
                    "artifact_sha256": curriculum_plan.artifact_sha256,
                    "file_sha256": _sha256(args.curriculum_plan),
                    "source_pair_plan_artifact_sha256": (
                        curriculum_plan.source_pair_plan_artifact_sha256
                    ),
                    "observation_mode": args.curriculum_observation_mode,
                    "source_visual_lattices": list(curriculum_plan.visual_lattices),
                    "type": (
                        "official_native_once_curriculum"
                        if args.curriculum_observation_mode
                        == CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE
                        else "exhaustive_dual_lattice_curriculum"
                    ),
                    "visual_lattices": (
                        [LATTICE_BASELINE]
                        if args.curriculum_observation_mode
                        == CURRICULUM_OBSERVATION_OFFICIAL_NATIVE_ONCE
                        else list(curriculum_plan.visual_lattices)
                    ),
                }
            else:
                if pair_plan is None or args.pair_plan is None or args.visual_lattice is None:
                    raise RuntimeError("native VL adaptation lost its pair-plan report binding")
                training_plan_report = {
                    "artifact_sha256": pair_plan.artifact_sha256,
                    "file_sha256": _sha256(args.pair_plan),
                    "observation_mode": PAIR_PLAN_OBSERVATION_SINGLE_LATTICE,
                    "type": "legacy_pair_plan",
                    "visual_lattices": [args.visual_lattice],
                }
            if retention_manifest is None:
                public_retention_report = {"enabled": False}
            else:
                if not isinstance(args.public_vl_retention_manifest, Path) or not isinstance(
                    args.public_vl_retention_root, Path
                ):
                    raise RuntimeError("native VL adaptation lost public retention paths")
                public_retention_report = {
                    "artifact_root": str(args.public_vl_retention_root.resolve()),
                    "artifact_sha256": retention_manifest.artifact_sha256,
                    "enabled": True,
                    "family_partition_counts": retention_manifest.family_partition_counts,
                    "global_loss_factors": {
                        "referring": PUBLIC_NATIVE_VL_RETENTION_WEIGHT / WORLD_SIZE,
                        "vqa": PUBLIC_NATIVE_VL_RETENTION_WEIGHT / WORLD_SIZE,
                    },
                    "manifest_file": str(args.public_vl_retention_manifest.resolve()),
                    "manifest_file_sha256": args.public_vl_retention_manifest_sha256,
                    "quality_exclusions": [
                        item.to_dict() for item in retention_manifest.quality_exclusions
                    ],
                    "rank_loss_weight": PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
                    "rank_streams": {"0": "referring", "1": "vqa"},
                    "processor": retention_processor_contract,
                    "sources": {
                        key: retention_manifest.sources[key].to_dict()
                        for key in sorted(retention_manifest.sources)
                    },
                }
            report = {
                "candidate_model_file_sha256": candidate_hashes,
                "calvin_factor_contract": {
                    "adr127_smoke": args.adr127_smoke,
                    "adr128_smoke": args.adr128_smoke,
                    "crossed_arm": args.crossed_arm,
                    "mode": args.calvin_factor_mode,
                    "rank_factor_weights_before_microbatch_average": {
                        name: weight
                        for name, weight in _calvin_factor_weights(args.calvin_factor_mode)
                    },
                },
                "counterfactual_gradient_audit": (
                    {"enabled": False}
                    if not args.counterfactual_gradient_audit
                    else {
                        "enabled": True,
                        "objective_weights": dict(
                            zip(
                                ADR127_GRADIENT_OBJECTIVES,
                                ADR127_GRADIENT_WEIGHTS,
                                strict=True,
                            )
                        ),
                        "reports": counterfactual_gradient_reports,
                        "status": _counterfactual_gradient_audit_status(
                            counterfactual_gradient_reports
                        ),
                        "step_indices": list(ADR127_GRADIENT_AUDIT_STEPS),
                    }
                ),
                "cuda_allocator": args.cuda_allocator,
                "crossed_cpu_materialization": crossed_materialization_report,
                "dataset_manifest_sha256": manifest.tree_sha256,
                "export_memory_per_rank": export_memory,
                "fsdp2_placement": args.fsdp2_placement,
                "hyperparameters": {
                    "adam_beta1": args.adam_beta1,
                    "adam_beta2": args.adam_beta2,
                    "adam_eps": args.adam_eps,
                    "learning_rate": args.learning_rate,
                    "max_grad_norm": args.max_grad_norm,
                    "max_steps": args.max_steps,
                    "schedule": "linear-warmup-cosine-decay",
                    "schedule_total_steps": args.schedule_total_steps,
                    "warmup_steps": args.warmup_steps,
                    "weight_decay": args.weight_decay,
                },
                "initial_qwen": {
                    "load_result": restore_result,
                    "model_file_sha256": initial_hashes,
                    "revision": args.initial_qwen_revision,
                },
                "load_seconds": load_seconds,
                "native_vl_patch_sha256": patch_report["native_vl_patch_sha256"],
                "observation_mode": training_plan_report["observation_mode"],
                "optimizer": "torch.optim.AdamW",
                "optimizer_state_parameter_count": optimizer_state_parameter_count,
                "optimizer_tied_parameter_name": optimizer_tied_name,
                "physical_sidecar_manifest_sha256": sidecar.manifest_sha256,
                "picf_code_revision": args.picf_code_revision,
                "processor_lattices": processor_lattices,
                "processor_snapshot_size": processor_snapshot_size,
                "public_vl_retention": public_retention_report,
                "runtime_python_trees": runtime_python_trees,
                "schema": (ADR128_OUTPUT_SCHEMA if crossed_plan is not None else OUTPUT_SCHEMA),
                "seed": args.seed,
                "source_commit": commit,
                "status": "PASS",
                "step_reports": step_reports,
                "teacher_prune": teacher_prune,
                "trainable_scope": sharded_scope.as_dict(),
                "training_memory_per_rank": training_memory,
                "training_plan": training_plan_report,
                "training_seconds": training_seconds,
                "world_size": WORLD_SIZE,
            }
            write_text_durable_exclusive(
                partial / "report.json",
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            os.replace(partial, args.output_dir)
            print(
                json.dumps(
                    {
                        "output_dir": str(args.output_dir),
                        "schema": (
                            ADR128_OUTPUT_SCHEMA if crossed_plan is not None else OUTPUT_SCHEMA
                        ),
                        "status": "PASS",
                        "training_seconds": training_seconds,
                    },
                    sort_keys=True,
                )
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

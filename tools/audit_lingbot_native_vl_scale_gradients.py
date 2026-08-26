#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Measure isolated lattice-8/lattice-14 gradients without updating Qwen."""

from __future__ import annotations

import argparse
import hashlib
import json
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
from picf_next.data.public_native_vl import (
    PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
    PublicNativeVLRetentionManifest,
    load_frozen_public_native_vl_retention_gate,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.gradient_alignment import (
    GradientPairMoments,
    WeightedGradientPairMoments,
    summarize_qwen_gradient_alignment,
    summarize_weighted_qwen_gradient_alignment,
)
from picf_next.lingbot_native.gradient_audit_runtime import (
    distributed_pair_rows as _distributed_pair_rows,
    snapshot_local_gradients as _snapshot_local_gradients,
)
from picf_next.lingbot_native.lattice_feasibility import (
    configure_native_processor_area_budget,
    configure_native_processor_lattice,
    validate_native_processor_record_grid,
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
AUDITED_LATTICES = (8, 14)
ADR125_RETENTION_GRADIENT_STEP_INDICES = (0, 21, 42, 63)
OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-scale-gradient-audit.v2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_git_revision(value: str, *, name: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"native VL scale audit {name} must be one Git commit")
    return value


def _validate_sha256(value: str, *, name: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ContractError(f"native VL scale audit {name} must be one lowercase SHA-256")
    return value


def _parse_step_indices(value: str) -> tuple[int, ...]:
    if not isinstance(value, str) or not value:
        raise argparse.ArgumentTypeError("step indices must be comma-separated integers")
    try:
        parsed = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("step indices must be comma-separated integers") from error
    if not parsed or any(index < 0 for index in parsed) or tuple(sorted(set(parsed))) != parsed:
        raise argparse.ArgumentTypeError("step indices must be unique, sorted and non-negative")
    return parsed


def _validate_retention_step_indices(step_indices: tuple[int, ...]) -> None:
    if step_indices != ADR125_RETENTION_GRADIENT_STEP_INDICES:
        raise ContractError("native VL scale audit retention step indices differ from ADR-125")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--qwen-dir", type=Path, required=True)
    parser.add_argument("--qwen-revision", required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--public-vl-retention-manifest", type=Path)
    parser.add_argument("--public-vl-retention-manifest-sha256")
    parser.add_argument("--public-vl-retention-root", type=Path)
    parser.add_argument("--public-vl-retention-weight", type=float)
    parser.add_argument("--curriculum-plan", type=Path, required=True)
    parser.add_argument("--curriculum-plan-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--picf-code-revision", required=True)
    parser.add_argument("--step-indices", type=_parse_step_indices, required=True)
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
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> Path:
    for path in (
        args.training_config,
        args.dataset_manifest,
        args.curriculum_plan,
        args.source_checkout / MODEL_SOURCE,
        _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.qwen_dir,
        args.dataset_split,
        args.physical_sidecar_root,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    partial = args.output_dir.with_name(f"{args.output_dir.name}.partial")
    for path in (args.output_dir, partial):
        if path.exists() or path.is_symlink():
            raise FileExistsError(path)
    _validate_sha256(args.curriculum_plan_sha256, name="curriculum SHA-256")
    _validate_git_revision(args.picf_code_revision, name="PICF revision")
    _validate_optional_qwen_restore(args.qwen_dir, args.qwen_revision)
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ContractError("native VL scale audit seed must be non-negative")
    placement = validate_fsdp2_placement(args.fsdp2_placement)
    if placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
        raise ContractError("native VL tied embeddings cannot use selective embedding offload")
    if args.cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ContractError("native VL scale audit allocator mode is unsupported")
    retention_values = (
        args.public_vl_retention_manifest,
        args.public_vl_retention_manifest_sha256,
        args.public_vl_retention_root,
        args.public_vl_retention_weight,
    )
    if any(value is not None for value in retention_values):
        if any(value is None for value in retention_values):
            raise ContractError("native VL scale audit retention arguments must be all present")
        _validate_retention_step_indices(args.step_indices)
        if args.public_vl_retention_weight != PUBLIC_NATIVE_VL_RETENTION_WEIGHT:
            raise ContractError("native VL scale audit retention weight differs from ADR-125")
        if not isinstance(args.public_vl_retention_manifest, Path):
            raise ContractError("native VL scale audit retention manifest path is missing")
        if not isinstance(args.public_vl_retention_root, Path):
            raise ContractError("native VL scale audit retention root is missing")
        if not isinstance(args.public_vl_retention_manifest_sha256, str):
            raise ContractError("native VL scale audit retention manifest SHA-256 is missing")
        args.public_vl_retention_manifest_object = load_frozen_public_native_vl_retention_gate(
            manifest_path=args.public_vl_retention_manifest,
            manifest_file_sha256=args.public_vl_retention_manifest_sha256,
            artifact_root=args.public_vl_retention_root,
            max_steps=max(args.step_indices) + 1,
        )
    else:
        args.public_vl_retention_manifest_object = None
    return partial


def _distributed_alignment(
    model: Any,
    *,
    lattice8_gradients: dict[str, Any],
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, object]:
    names, reduced = _distributed_pair_rows(
        model,
        first_gradients=lattice8_gradients,
        device=device,
        dist=dist,
        torch_module=torch_module,
    )
    moments = {
        name: GradientPairMoments(
            dot=float(row[0]),
            lattice8_squared_norm=float(row[1]),
            lattice14_squared_norm=float(row[2]),
            elements=int(round(row[3])),
        )
        for name, row in zip(names, reduced, strict=True)
    }
    return summarize_qwen_gradient_alignment(moments)


def _distributed_weighted_alignment(
    model: Any,
    *,
    first_gradients: dict[str, Any],
    first_objective: str,
    first_weight: float,
    second_objective: str,
    second_weight: float,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, object]:
    names, reduced = _distributed_pair_rows(
        model,
        first_gradients=first_gradients,
        device=device,
        dist=dist,
        torch_module=torch_module,
    )
    moments = {
        name: WeightedGradientPairMoments(
            dot=float(row[0]),
            first_squared_norm=float(row[1]),
            second_squared_norm=float(row[2]),
            elements=int(round(row[3])),
        )
        for name, row in zip(names, reduced, strict=True)
    }
    return summarize_weighted_qwen_gradient_alignment(
        moments,
        first_objective=first_objective,
        second_objective=second_objective,
        first_weight=first_weight,
        second_weight=second_weight,
    )


def _step_summary(step_reports: list[dict[str, Any]]) -> dict[str, object]:
    if not step_reports:
        raise RuntimeError("native VL scale audit has no completed steps")
    surface_names = ("global", *sorted(step_reports[0]["alignment"]["groups"]))
    result = {}
    for name in surface_names:
        rows = [
            report["alignment"]["global"]
            if name == "global"
            else report["alignment"]["groups"][name]
            for report in step_reports
        ]
        cosines = [float(row["cosine"]) for row in rows if row["cosine"] is not None]
        result[name] = {
            "audited_step_count": len(rows),
            "cosine_max": max(cosines) if cosines else None,
            "cosine_mean": sum(cosines) / len(cosines) if cosines else None,
            "cosine_min": min(cosines) if cosines else None,
            "lattice8_mean_descent_failure_count": sum(
                not bool(row["mean_gradient_descends_lattice8"]) for row in rows
            ),
            "lattice14_mean_descent_failure_count": sum(
                not bool(row["mean_gradient_descends_lattice14"]) for row in rows
            ),
            "negative_cosine_step_count": sum(cosine < 0.0 for cosine in cosines),
            "parameter_tensor_negative_dot_mass_fraction_mean": sum(
                float(row["parameter_tensor_negative_dot_mass_fraction"]) for row in rows
            )
            / len(rows),
        }
    return result


def _retention_step_summary(step_reports: list[dict[str, Any]]) -> dict[str, object]:
    retention_reports: list[dict[str, Any]] = []
    for step_report in step_reports:
        retention_report = step_report.get("public_vl_retention")
        if not isinstance(retention_report, dict):
            raise RuntimeError("native VL scale audit lacks one retention step report")
        retention_reports.append(retention_report)
    alignments = [report["alignment"] for report in retention_reports]
    surface_names = ("global", *sorted(alignments[0]["groups"]))
    result = {}
    for name in surface_names:
        rows = [
            alignment["global"] if name == "global" else alignment["groups"][name]
            for alignment in alignments
        ]
        cosines = [float(row["cosine"]) for row in rows if row["cosine"] is not None]
        result[name] = {
            "audited_step_count": len(rows),
            "cosine_max": max(cosines) if cosines else None,
            "cosine_mean": sum(cosines) / len(cosines) if cosines else None,
            "cosine_min": min(cosines) if cosines else None,
            "first_objective_descent_failure_count": sum(
                not bool(row["mixed_gradient_descends_first_objective"]) for row in rows
            ),
            "negative_cosine_step_count": sum(cosine < 0.0 for cosine in cosines),
            "parameter_tensor_negative_dot_mass_fraction_mean": sum(
                float(row["parameter_tensor_negative_dot_mass_fraction"]) for row in rows
            )
            / len(rows),
            "second_objective_descent_failure_count": sum(
                not bool(row["mixed_gradient_descends_second_objective"]) for row in rows
            ),
        }
    return result


def _retention_gate_status(summary: dict[str, object]) -> str:
    global_summary = summary.get("global")
    if not isinstance(global_summary, dict):
        raise RuntimeError("native VL scale audit retention summary omits the global surface")
    failure_counts = (
        global_summary.get("first_objective_descent_failure_count"),
        global_summary.get("second_objective_descent_failure_count"),
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in failure_counts):
        raise RuntimeError("native VL scale audit retention failure counts are malformed")
    return "PASS" if failure_counts == (0, 0) else "FAIL"


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
        raise RuntimeError("native VL scale audit source overlay is not applied")
    if _validate_native_vl_model(args.source_checkout / MODEL_SOURCE) != (
        NATIVE_VL_PATCHED_MODEL_SHA256
    ):
        raise RuntimeError("native VL scale audit source digest differs")
    source_commit = subprocess.run(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if source_commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("native VL scale audit source commit differs")
    picf_commit = subprocess.run(
        ["git", "-C", str(_ROOT), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if picf_commit != args.picf_code_revision:
        raise RuntimeError("native VL scale audit checkout differs from its declared revision")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("native VL scale audit requires exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("native VL scale audit requires two local GPUs")

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
    from picf_next.lingbot_native.vl_cotraining import (
        build_native_vl_grounding_batch,
        configure_native_vl_grounding_trainable_scope,
        materialize_fixed_observation_native_vl_records,
        register_native_vl_fsdp_forward_method,
        retie_and_validate_native_qwen_lm_head,
        run_native_vl_grounding_forward,
        verify_native_vl_grounding_trainable_scope,
    )
    from picf_next.lingbot_native.vl_curriculum import NativeVLGroundingCurriculumPlan

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError("native VL scale audit sees an unexpected CUDA topology")
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
        if _sha256(args.curriculum_plan) != args.curriculum_plan_sha256:
            raise ContractError("native VL scale audit curriculum file SHA-256 changed")
        curriculum = NativeVLGroundingCurriculumPlan.load(args.curriculum_plan)
        if args.step_indices[-1] >= len(curriculum.steps):
            raise ContractError("native VL scale audit step lies outside its curriculum")
        if tuple(curriculum.visual_lattices) != AUDITED_LATTICES:
            raise ContractError("native VL scale audit curriculum lattices changed")
        if (
            curriculum.dataset_id,
            curriculum.dataset_revision,
            curriculum.dataset_manifest_sha256,
        ) != (manifest.dataset_id, manifest.dataset_revision, manifest.tree_sha256):
            raise ContractError("native VL scale audit curriculum belongs to another dataset")
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
            retention_manifest,
            PublicNativeVLRetentionManifest,
        ):
            raise RuntimeError("native VL scale audit lost its typed retention manifest")

        training = load_lingbot_training_config(args.training_config)
        train_values = training.get("train")
        if not isinstance(train_values, dict):
            raise ContractError("native VL scale audit training config has no train mapping")
        released_lr = train_values.get("lr", 5e-5)
        if isinstance(released_lr, bool) or not isinstance(released_lr, int | float):
            raise ContractError("native VL scale audit released learning rate is invalid")
        runtime_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(released_lr),
        )
        merged, _ = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=1,
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
        processor_lattices = {
            str(lattice): configure_native_processor_lattice(processor, lattice)
            for lattice in AUDITED_LATTICES
        }
        retention_processor = None
        retention_processor_contract = None
        if retention_manifest is not None:
            retention_processor = build_processor(str(args.processor_dir.resolve()))
            retention_processor_contract = configure_native_processor_area_budget(
                retention_processor,
                AUDITED_LATTICES[0],
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
            raise ContractError("native VL scale audit tied parameter changed during host load")
        restore_result = _validate_qwen_restore_load_result(
            load_sharded_checkpoint(
                policy.model.qwenvl_with_expert.qwenvl,
                args.qwen_dir,
                strict=False,
                prefer_safe=True,
            )
        )
        restored_tied_name = retie_and_validate_native_qwen_lm_head(policy)
        if restored_tied_name != loaded_tied_name:
            raise ContractError("native VL scale audit tied parameter changed during Qwen restore")
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
        load_seconds = time.perf_counter() - load_started

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        audit_started = time.perf_counter()
        step_reports: list[dict[str, Any]] = []
        retention_gate_pass = True
        for step_index in args.step_indices:
            source_group, batches = curriculum.resolve_step(step_index)
            if tuple(batch[0] for batch in batches) != AUDITED_LATTICES:
                raise ContractError("native VL scale audit step changed lattice order")
            lattice8_gradients = None
            alignment = None
            lattice_reports = []
            for lattice, camera_name, variants in batches:
                configure_native_processor_lattice(processor, lattice)
                records = materialize_fixed_observation_native_vl_records(
                    index=index,
                    sidecar=sidecar,
                    group=source_group,
                    variants=variants,
                    expected_camera_name=camera_name,
                )
                record = records[rank]
                batch = build_native_vl_grounding_batch(record, processor).to(
                    device,
                    pixel_dtype=torch.bfloat16,
                )
                expected_grid = [[1, lattice * 2, lattice * 2]]
                if batch.image_grid_thw.detach().cpu().tolist() != expected_grid:
                    raise RuntimeError("native VL scale audit image grid differs from its lattice")
                probe_seed = args.seed + step_index * WORLD_SIZE + rank
                random.seed(probe_seed)
                np.random.seed(probe_seed)
                torch.manual_seed(probe_seed)
                torch.cuda.manual_seed(probe_seed)
                policy.zero_grad(set_to_none=True)
                scale_started = time.perf_counter()
                loss = run_native_vl_grounding_forward(policy, batch)
                loss.backward()
                rank_report = {
                    "camera_name": record.camera_name,
                    "elapsed_seconds": time.perf_counter() - scale_started,
                    "global_index": record.global_index,
                    "instruction": record.instruction,
                    "loss": float(loss.detach().float().item()),
                    "rank": rank,
                    "supervised_token_count": batch.supervised_token_count,
                    "target_identity_key": record.target_identity_key,
                    "task_key": record.task_key,
                }
                gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(gathered, rank_report)
                if rank == 0:
                    lattice_reports.append({"lattice": lattice, "ranks": gathered})
                if lattice == AUDITED_LATTICES[0]:
                    lattice8_gradients = _snapshot_local_gradients(
                        policy,
                        torch_module=torch,
                    )
                elif lattice == AUDITED_LATTICES[1]:
                    if lattice8_gradients is None:
                        raise RuntimeError("native VL scale audit omitted lattice-8 gradients")
                    alignment = _distributed_alignment(
                        policy,
                        lattice8_gradients=lattice8_gradients,
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                    global_alignment = alignment.get("global")
                    if not isinstance(global_alignment, dict):
                        raise RuntimeError("native VL scale audit global summary is malformed")
                    if int(global_alignment["element_count"]) != sharded_scope.trainable_numel:
                        raise RuntimeError("native VL scale audit gradient coverage changed")
                else:
                    raise RuntimeError("native VL scale audit encountered an undeclared lattice")
                policy.zero_grad(set_to_none=True)
                del batch, loss, record, records
                torch.cuda.empty_cache()
            if lattice8_gradients is None or alignment is None:
                raise RuntimeError("native VL scale audit step did not compare both lattices")
            retention_step_report = None
            if retention_manifest is not None:
                if retention_processor is None or retention_processor_contract is None:
                    raise RuntimeError("native VL scale audit lost its retention processor")
                retention_descriptor = retention_manifest.training_record_for_rank(
                    optimizer_step=step_index,
                    rank=rank,
                )
                retention_record = retention_manifest.materialize_record(
                    retention_descriptor,
                    artifact_root=args.public_vl_retention_root,
                )
                retention_batch = build_native_vl_grounding_batch(
                    retention_record,
                    retention_processor,
                )
                retention_grid_thw = retention_batch.image_grid_thw.detach().cpu().tolist()
                retention_grid_budget = validate_native_processor_record_grid(
                    retention_grid_thw,
                    image_height=retention_descriptor.height,
                    image_width=retention_descriptor.width,
                    lattice=AUDITED_LATTICES[0],
                )
                retention_batch = retention_batch.to(
                    device,
                    pixel_dtype=torch.bfloat16,
                )
                retention_seed = args.seed + 10_000_000 + step_index * WORLD_SIZE + rank
                random.seed(retention_seed)
                np.random.seed(retention_seed)
                torch.manual_seed(retention_seed)
                torch.cuda.manual_seed(retention_seed)
                policy.zero_grad(set_to_none=True)
                retention_started = time.perf_counter()
                retention_loss = run_native_vl_grounding_forward(policy, retention_batch)
                retention_loss.backward()
                local_retention_report = {
                    "elapsed_seconds": time.perf_counter() - retention_started,
                    "family": retention_record.family,
                    "grid_budget": retention_grid_budget,
                    "image_height": retention_descriptor.height,
                    "image_rgb_sha256": retention_descriptor.image_rgb_sha256,
                    "image_grid_thw": retention_grid_thw,
                    "image_width": retention_descriptor.width,
                    "loss": float(retention_loss.detach().float().item()),
                    "rank": rank,
                    "record_id": retention_record.record_id,
                    "record_sha256": retention_descriptor.record_sha256,
                    "source_row_index": retention_descriptor.source_row_index,
                    "source_subindex": retention_descriptor.source_subindex,
                    "supervised_token_count": retention_batch.supervised_token_count,
                    "target_answer_sha256": hashlib.sha256(
                        retention_descriptor.assistant_text.encode("utf-8")
                    ).hexdigest(),
                    "user_text": retention_record.user_text,
                    "user_text_sha256": hashlib.sha256(
                        retention_record.user_text.encode("utf-8")
                    ).hexdigest(),
                }
                gathered_retention: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(gathered_retention, local_retention_report)
                retention_alignment = _distributed_weighted_alignment(
                    policy,
                    first_gradients=lattice8_gradients,
                    first_objective="calvin_official_native_once",
                    first_weight=1.0,
                    second_objective="public_native_vl_retention",
                    second_weight=PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
                retention_global = retention_alignment.get("global")
                if not isinstance(retention_global, dict):
                    raise RuntimeError("native VL scale audit retention summary is malformed")
                if int(retention_global["element_count"]) != sharded_scope.trainable_numel:
                    raise RuntimeError("native VL scale audit retention coverage changed")
                retention_gate_pass = (
                    retention_gate_pass
                    and bool(retention_global["mixed_gradient_descends_first_objective"])
                    and bool(retention_global["mixed_gradient_descends_second_objective"])
                )
                if rank == 0:
                    retention_step_report = {
                        "alignment": retention_alignment,
                        "ranks": gathered_retention,
                    }
                policy.zero_grad(set_to_none=True)
                del retention_batch, retention_loss, retention_record
                torch.cuda.empty_cache()
            if rank == 0:
                step_reports.append(
                    {
                        "alignment": alignment,
                        "curriculum_group_index": curriculum.steps[step_index].group_index,
                        "curriculum_optimizer_step": curriculum.steps[step_index].optimizer_step,
                        "lattices": lattice_reports,
                        "public_vl_retention": retention_step_report,
                        "step_index": step_index,
                    }
                )
            del alignment, lattice8_gradients
        audit_seconds = time.perf_counter() - audit_started
        memory = {
            "allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "rank": rank,
            "reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
        }
        gathered_memory: list[Any] = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(gathered_memory, memory)
        if rank == 0:
            partial.mkdir(parents=True)
            retention_summary = (
                None if retention_manifest is None else _retention_step_summary(step_reports)
            )
            decision_status = (
                "PASS" if retention_summary is None else _retention_gate_status(retention_summary)
            )
            if decision_status != ("PASS" if retention_gate_pass else "FAIL"):
                raise RuntimeError("native VL scale audit retention decision changed")
            report = {
                "audit_seconds": audit_seconds,
                "cuda_allocator": args.cuda_allocator,
                "dataset_manifest_sha256": manifest.tree_sha256,
                "execution_status": "PASS",
                "fsdp2_placement": args.fsdp2_placement,
                "load_seconds": load_seconds,
                "memory_per_rank": gathered_memory,
                "native_vl_patch_sha256": patch_report["native_vl_patch_sha256"],
                "picf_code_revision": args.picf_code_revision,
                "processor_lattices": processor_lattices,
                "public_vl_retention": (
                    {"enabled": False}
                    if retention_manifest is None
                    else {
                        "artifact_root": str(args.public_vl_retention_root.resolve()),
                        "artifact_sha256": retention_manifest.artifact_sha256,
                        "enabled": True,
                        "family_partition_counts": (retention_manifest.family_partition_counts),
                        "global_loss_factors": {
                            "referring": PUBLIC_NATIVE_VL_RETENTION_WEIGHT / WORLD_SIZE,
                            "vqa": PUBLIC_NATIVE_VL_RETENTION_WEIGHT / WORLD_SIZE,
                        },
                        "manifest_file": str(args.public_vl_retention_manifest.resolve()),
                        "manifest_file_sha256": args.public_vl_retention_manifest_sha256,
                        "quality_exclusions": [
                            item.to_dict() for item in retention_manifest.quality_exclusions
                        ],
                        "processor": retention_processor_contract,
                        "rank_streams": {"0": "referring", "1": "vqa"},
                        "rank_weight": PUBLIC_NATIVE_VL_RETENTION_WEIGHT,
                        "sources": {
                            key: retention_manifest.sources[key].to_dict()
                            for key in sorted(retention_manifest.sources)
                        },
                    }
                ),
                "qwen": {
                    "load_result": restore_result,
                    "model_file_sha256": _model_hashes(args.qwen_dir),
                    "revision": args.qwen_revision,
                },
                "schema": OUTPUT_SCHEMA,
                "source_commit": source_commit,
                "status": decision_status,
                "step_indices": list(args.step_indices),
                "step_reports": step_reports,
                "summary": _step_summary(step_reports),
                "retention_summary": retention_summary,
                "teacher_prune": teacher_prune,
                "trainable_scope": sharded_scope.as_dict(),
                "world_size": WORLD_SIZE,
                "weight_update_count": 0,
            }
            write_text_durable_exclusive(
                partial / "report.json",
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            os.replace(partial, args.output_dir)
            print(
                json.dumps(
                    {
                        "audit_seconds": audit_seconds,
                        "output_dir": str(args.output_dir),
                        "schema": OUTPUT_SCHEMA,
                        "status": decision_status,
                    },
                    sort_keys=True,
                )
            )
        dist.barrier()
        if retention_manifest is not None and not retention_gate_pass:
            raise SystemExit(2)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Prove ADR218 real-CALVIN FSDP2 update and cold-resume equivalence."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

WORLD_SIZE = 2
SCHEMA = "picf-next.adr218-fsdp-optimizer-checkpoint.v1"
EXTRA_SCHEMA = "picf-next.adr218-fsdp-checkpoint-extra.v1"
CUBLAS_WORKSPACE_CONFIG = ":4096:8"
PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
SELECTIVE_CPU_MODULES = (
    "model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens",
)
FSDP_OFFLOAD_MODE = "shared-embedding-plus-future3d-classes"
BASE_GRADIENT_FRAGMENTS = (
    ("wsa_future", "adr218_wsa_training_runtime.future"),
    ("lingbot_host", "qwenvl.model.language_model.layers"),
    ("lingbot_action", "qwen_expert.model.layers"),
    ("action_output", "action_out_proj"),
)
FULL_MODAL_GRADIENT_FRAGMENTS = (
    *BASE_GRADIENT_FRAGMENTS,
    ("picf_graph", "picf_native_graph"),
    ("anytouch_projection", "modality_projections.anytouch"),
    ("sonata_projection", "modality_projections.sonata"),
    ("vjepa_projection", "modality_projections.vjepa"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("fresh", "resume", "composition", "causality"),
        required=True,
    )
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--robot-config", type=Path, required=True)
    parser.add_argument("--model-checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--wsa-source-root", type=Path, required=True)
    parser.add_argument("--wsa-checkpoint", type=Path, required=True)
    parser.add_argument("--da3-source-root", type=Path, required=True)
    parser.add_argument("--da3-model-dir", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dense-evidence-cache-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--dense-evidence-cache-manifest-sha256",
        action="append",
        default=[],
    )
    parser.add_argument("--composition-first-global-index", type=int, default=96395)
    parser.add_argument("--seed", type=int, default=218)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move_model_inputs(
    model_inputs: Mapping[str, Any],
    *,
    device: Any,
    dtype: Any,
    torch_module: Any,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in model_inputs.items():
        if torch_module.is_tensor(value):
            moved[name] = value.to(
                device=device,
                dtype=dtype if value.is_floating_point() else value.dtype,
                non_blocking=False,
            )
        else:
            moved[name] = value
    return moved


def _distributed_gradient_metrics(
    model: Any,
    *,
    device: Any,
    dist: Any,
    torch_module: Any,
    fragments: tuple[tuple[str, str], ...] = BASE_GRADIENT_FRAGMENTS,
) -> dict[str, float | int | bool]:
    squares: dict[str, Any | None] = {name: None for name, _ in fragments}
    counts = {name: 0 for name, _ in fragments}
    finite = torch_module.ones((), dtype=torch_module.int32, device=device)
    for parameter_name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        local = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
        finite.mul_(
            torch_module.isfinite(local)
            .all()
            .to(device=device, dtype=torch_module.int32)
        )
        for metric_name, fragment in fragments:
            if fragment not in parameter_name:
                continue
            value = local.detach().float().square().sum().to(device=device)
            previous = squares[metric_name]
            squares[metric_name] = value if previous is None else previous + value
            counts[metric_name] += int(local.numel())
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    packed = []
    for metric_name, _ in fragments:
        value = squares[metric_name]
        packed.extend(
            (
                torch_module.zeros((), dtype=torch_module.float64, device=device)
                if value is None
                else value.to(dtype=torch_module.float64),
                torch_module.tensor(
                    float(counts[metric_name]),
                    dtype=torch_module.float64,
                    device=device,
                ),
            )
        )
    reduced = torch_module.stack(packed)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    values = reduced.cpu().tolist()
    result: dict[str, float | int | bool] = {"all_finite": bool(finite.item())}
    for index, (metric_name, _) in enumerate(fragments):
        result[f"{metric_name}_norm"] = math.sqrt(float(values[index * 2]))
        result[f"{metric_name}_elements"] = int(values[index * 2 + 1])
    return result


def _validate_gradient_metrics(
    metrics: Mapping[str, object],
    *,
    fragments: tuple[tuple[str, str], ...] = BASE_GRADIENT_FRAGMENTS,
) -> None:
    if metrics.get("all_finite") is not True:
        raise RuntimeError("ADR218 distributed gradients are non-finite")
    for name, _ in fragments:
        if int(metrics[f"{name}_elements"]) <= 0:
            raise RuntimeError(f"ADR218 produced no {name} gradient elements")
        if float(metrics[f"{name}_norm"]) <= 0.0:
            raise RuntimeError(f"ADR218 produced a zero {name} gradient")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _emit_phase(*, rank: int, phase: str) -> None:
    print(
        json.dumps(
            {
                "schema": "picf-next.adr218-fsdp-phase.v1",
                "rank": rank,
                "phase": phase,
                "unix_time": time.time(),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _configure_exact_resume_determinism(torch_module: Any) -> dict[str, object]:
    """Make an uninterrupted update reproducible after a cold process restart."""

    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured != CUBLAS_WORKSPACE_CONFIG:
        raise RuntimeError(
            "ADR218 exact-resume gate requires CUBLAS_WORKSPACE_CONFIG="
            f"{CUBLAS_WORKSPACE_CONFIG}, got {configured!r}"
        )
    allocator_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if allocator_config != PYTORCH_CUDA_ALLOC_CONF:
        raise RuntimeError(
            "ADR218 exact-resume gate requires PYTORCH_CUDA_ALLOC_CONF="
            f"{PYTORCH_CUDA_ALLOC_CONF}, got {allocator_config!r}"
        )
    torch_module.use_deterministic_algorithms(True)
    torch_module.backends.cudnn.deterministic = True
    torch_module.backends.cudnn.benchmark = False
    torch_module.backends.cuda.matmul.allow_tf32 = False
    torch_module.backends.cudnn.allow_tf32 = False
    torch_module.set_float32_matmul_precision("highest")
    receipt = {
        "cublas_workspace_config": configured,
        "pytorch_cuda_alloc_conf": allocator_config,
        "deterministic_algorithms": bool(
            torch_module.are_deterministic_algorithms_enabled()
        ),
        "deterministic_warn_only": bool(
            torch_module.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_deterministic": bool(torch_module.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch_module.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(
            torch_module.backends.cuda.matmul.allow_tf32
        ),
        "cudnn_allow_tf32": bool(torch_module.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch_module.get_float32_matmul_precision(),
    }
    if receipt != {
        "cublas_workspace_config": CUBLAS_WORKSPACE_CONFIG,
        "pytorch_cuda_alloc_conf": PYTORCH_CUDA_ALLOC_CONF,
        "deterministic_algorithms": True,
        "deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }:
        raise RuntimeError(f"ADR218 deterministic runtime differs: {receipt}")
    return receipt


def _release_checkpoint_allocator_cache(
    *,
    rank: int,
    device: Any,
    torch_module: Any,
) -> None:
    """Release only unoccupied CUDA cache retained by DCP boundary work."""

    gc.collect()
    torch_module.cuda.empty_cache()
    torch_module.cuda.synchronize(device)
    _emit_phase(rank=rank, phase="checkpoint-allocator-cache-released")


def main() -> None:
    args = _parse_args()
    full_modal_phase = args.phase in {"composition", "causality"}
    root = Path(__file__).resolve().parents[1]
    for import_path in (root, root / "src", args.source_checkout):
        sys.path.insert(0, str(import_path.resolve()))
    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("ADR218 FSDP Gate requires exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("ADR218 FSDP Gate requires two local GPUs")

    configured_cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured_cublas not in (None, CUBLAS_WORKSPACE_CONFIG):
        raise RuntimeError(
            "ADR218 exact-resume gate refuses conflicting "
            f"CUBLAS_WORKSPACE_CONFIG={configured_cublas!r}"
        )
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = CUBLAS_WORKSPACE_CONFIG
    configured_allocator = os.environ.get("PYTORCH_CUDA_ALLOC_CONF")
    if configured_allocator not in (None, PYTORCH_CUDA_ALLOC_CONF):
        raise RuntimeError(
            "ADR218 exact-resume gate refuses conflicting "
            f"PYTORCH_CUDA_ALLOC_CONF={configured_allocator!r}"
        )
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = PYTORCH_CUDA_ALLOC_CONF

    import numpy as np
    import torch
    import torch.distributed as dist
    import torch.nn.functional as F  # noqa: N812
    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import (
        LingbotVLAV2Config,
    )
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import (
        LingbotVlaV2Policy,
    )
    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import (
        build_moe_load_balance_hook,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from lingbotvla.optim import build_muon_optimizer
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    from picf_next.artifact_io import write_text_durable_exclusive
    from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
    from picf_next.data.dense_evidence_cache import FrozenDenseEvidenceCacheBank
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.data.lingbot_calvin import map_calvin_transition_to_lingbot
    from picf_next.lingbot_native.calvin import (
        audit_native_calvin_model_inputs,
        build_native_calvin_training_batch,
        collate_native_calvin_training_batch,
        with_native_modalities,
    )
    from picf_next.lingbot_native.action_posterior_collector import (
        RegisteredActionPosteriorReceiptCollector,
    )
    from picf_next.lingbot_native.dense_modalities import (
        NativeDenseModalityBinding,
        native_modalities_from_dense_evidence,
    )
    from picf_next.lingbot_native.host import (
        EXACT_NATIVE_MODALITY_BRIDGE,
        LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        install_lingbot_native_graph,
        native_context_from_persistent_state,
    )
    from picf_next.lingbot_native.modalities import (
        CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
        NO_RELATION_TARGET,
        NativeRelationSurfaceSpec,
    )
    from picf_next.lingbot_native.state import NativeLayerwisePosteriorState
    from picf_next.lingbot_native.official_config import official_lingbot_data_config
    from picf_next.lingbot_native.torch_dcp_compat import (
        install_torch_2_8_sparse_optimizer_state_backport,
    )
    from picf_next.lingbot_native.training import (
        native_persistent_output,
        run_native_policy_diagnostic_forward,
        run_native_policy_observation_diagnostic_forward,
        run_native_policy_training_forward,
    )
    from picf_next.lingbot_native.wsa_future_expert_runtime import WSAFutureExpertRuntime
    from picf_next.lingbot_native.wsa_da3_loss import WSADA3TeacherTargets
    from picf_next.lingbot_native.wsa_lingbot_install import (
        WSA_FSDP_BLOCK_CLASS,
        WSA_FSDP_EXPERT_CLASS,
        WSA_LARGE_OPTIMIZER,
        WSALingBotForwardRole,
        audit_wsa_lingbot_optimizer,
        configure_wsa_lingbot_optimizer_contract,
        install_wsa_lingbot_optimizer,
        install_wsa_lingbot_training_runtime,
        register_wsa_lingbot_fsdp_forward_methods,
        wsa_lingbot_forward_kwargs,
        wsa_lingbot_installation_receipt,
        wsa_lingbot_optimizer_transaction,
    )
    from picf_next.lingbot_native.wsa_lingbot_training_runtime import (
        WSALingBotAttentionIntervention,
        WSALingBotTrainingRuntime,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _merge_qwen_config,
        _resolve_training_config,
        build_lingbot_official_optimizer,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        resolve_lingbot_optimizer_contract,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _capture_rank_rng,
        _checkpoint_boundary,
        _restore_rank_rng,
        _validate_optimizer_state,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    determinism_receipt = _configure_exact_resume_determinism(torch)
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError("ADR218 FSDP Gate sees an unexpected CUDA topology")
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
        dcp_backport = install_torch_2_8_sparse_optimizer_state_backport(torch)

        manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
        norm_payload = json.loads(args.norm_stats.read_text(encoding="utf-8"))
        validate_lingbot_calvin_norm_stats(norm_payload)
        norm_source = norm_payload["source"]
        if (
            norm_source["dataset_id"] != manifest.dataset_id
            or norm_source["dataset_revision"] != manifest.dataset_revision
            or norm_source["dataset_tree_sha256"] != manifest.tree_sha256
            or manifest.split_name != args.dataset_split.name
        ):
            raise RuntimeError("ADR218 FSDP CALVIN manifest and normalization differ")
        dataset_binding = validate_dataset_runtime_binding(
            manifest,
            args.dataset_split.resolve(),
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            split_name=args.dataset_split.name,
        )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        dense_evidence_bank = None
        dense_evidence_bindings: tuple[NativeDenseModalityBinding, ...] = ()
        if full_modal_phase:
            if (
                len(args.dense_evidence_cache_root) != 3
                or len(args.dense_evidence_cache_manifest_sha256) != 3
            ):
                raise ValueError(
                    "ADR218 composition requires exactly three authenticated dense caches"
                )
            dense_evidence_bank = FrozenDenseEvidenceCacheBank.load(
                args.dense_evidence_cache_root,
                manifest_sha256s=args.dense_evidence_cache_manifest_sha256,
                dataset_tree_sha256=manifest.tree_sha256,
                memory_capacity=1,
            )
            if dense_evidence_bank.modalities != ("anytouch", "sonata", "vjepa"):
                raise RuntimeError("ADR218 composition dense modality set changed")
            dense_evidence_bindings = tuple(
                NativeDenseModalityBinding(
                    name=contract.modality,
                    encoder_contract=contract.encoder_contract,
                    token_width=contract.token_width,
                    maximum_tokens=contract.maximum_tokens,
                    geometry_width=contract.geometry_width,
                )
                for contract in dense_evidence_bank.contracts
            )

        training = load_lingbot_training_config(args.training_config)
        train_values = training.get("train")
        if not isinstance(train_values, dict):
            raise RuntimeError("LingBot training config has no train mapping")
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(train_values["lr"]),
        )
        optimizer_contract = configure_wsa_lingbot_optimizer_contract(optimizer_contract)
        merged, data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=args.model_checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=2,
        )
        merged.update(
            {
                "attention_implementation": (
                    "flex_cached" if full_modal_phase else "eager"
                ),
                "train_expert_only": False,
                "use_cache": False,
                "use_compile": False,
                "vit_attn_implementation": (
                    "flash_attention_2" if full_modal_phase else "eager"
                ),
            }
        )
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        qwen_config = AutoConfig.from_pretrained(
            args.processor_dir,
            local_files_only=True,
        )
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())

        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed(args.seed + rank)
        processor = build_processor(str(args.processor_dir.resolve()))
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        load_started = time.perf_counter()
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        load_model_weights(
            policy,
            str(args.model_checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        graph = None
        if full_modal_phase:
            modality_specs = tuple(binding.native_spec for binding in dense_evidence_bindings)
            relation_surface_specs = (
                NativeRelationSurfaceSpec(
                    name="anytouch",
                    geometry_kind="contact_sites",
                    layout="anytouch2.calvin.contact-sites.v1",
                    target_kind=NO_RELATION_TARGET,
                ),
                NativeRelationSurfaceSpec(
                    name="sonata",
                    geometry_kind="world_points",
                    layout="sonata.calvin.world-points.v1",
                    target_kind=NO_RELATION_TARGET,
                ),
                NativeRelationSurfaceSpec(
                    name="vjepa",
                    geometry_kind="image_grid",
                    layout="vjepa21.calvin.static-gripper.24x24.v1",
                    target_kind=CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
                ),
            )
            graph_config = LingBotNativeGraphConfig.from_policy(
                policy,
                capacity=200,
                maximum_control_tokens=8,
                task_query_count=0,
                prediction_address_width=0,
                predictive_target_widths=(),
                modality_specs=modality_specs,
                modality_bridge_identity=EXACT_NATIVE_MODALITY_BRIDGE,
                modality_bridge_query_count=0,
                resampled_modality_names=(),
                direct_action_modality_names=(),
                relation_surface_specs=relation_surface_specs,
                object_query_spatial_specs=(),
                relation_supervision_layers=(),
                architecture_identity=LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
            )
            graph = LingBotNativeGraph(
                graph_config,
                device=device,
                dtype=torch.float32,
            ).train()
            install_lingbot_native_graph(policy, graph)
        future = WSAFutureExpertRuntime.from_adapted_checkpoint(
            source_root=args.wsa_source_root,
            checkpoint=args.wsa_checkpoint,
            device=device,
            dtype=torch.float32,
        )
        install_wsa_lingbot_training_runtime(policy, WSALingBotTrainingRuntime(future))
        policy.train()
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=False,
            enable_shared_embedding_offload=True,
            selective_cpu_module_classes=(
                WSA_FSDP_BLOCK_CLASS,
                WSA_FSDP_EXPERT_CLASS,
            ),
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
        selective_cpu_modules = tuple(
            getattr(policy, "_lingbot_fsdp2_selective_cpu_modules", ())
        )
        if selective_cpu_modules != SELECTIVE_CPU_MODULES:
            raise RuntimeError(
                "ADR218 dual-A100 selective-offload topology differs: "
                f"{selective_cpu_modules}"
            )
        selective_cpu_module_classes = tuple(
            getattr(policy, "_lingbot_fsdp2_selective_cpu_module_classes", ())
        )
        expected_selective_cpu_module_classes = (
            WSA_FSDP_BLOCK_CLASS,
            WSA_FSDP_EXPERT_CLASS,
        )
        if selective_cpu_module_classes != expected_selective_cpu_module_classes:
            raise RuntimeError(
                "ADR218 selective Future3D offload topology differs: "
                f"{selective_cpu_module_classes}"
            )
        register_native_fsdp_forward_methods(policy)
        wsa_fsdp_forward_methods = register_wsa_lingbot_fsdp_forward_methods(policy)
        optimizer = build_lingbot_official_optimizer(
            policy,
            optimizer_contract,
            build_muon_optimizer=build_muon_optimizer,
            build_moe_load_balance_hook=build_moe_load_balance_hook,
        )
        install_wsa_lingbot_optimizer(policy, optimizer)
        optimizer_receipt = audit_wsa_lingbot_optimizer(policy, optimizer)
        checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
        load_seconds = time.perf_counter() - load_started

        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        if full_modal_phase:
            source_globals = (
                args.composition_first_global_index + rank * 2,
                args.composition_first_global_index + rank * 2 + 1,
            )
            sample_index_by_global = {
                locator.global_index: dataset_index
                for dataset_index, locator in enumerate(dataset.locators)
            }
            try:
                sample_indices = tuple(sample_index_by_global[value] for value in source_globals)
            except KeyError as error:
                raise RuntimeError(
                    "ADR218 composition source frame is not an action-bearing CALVIN sample"
                ) from error
        else:
            base_index = (rank * len(dataset)) // WORLD_SIZE
            sample_indices = (base_index, base_index + 1)
            source_globals = tuple(dataset.locators[value].global_index for value in sample_indices)
        samples = tuple(dataset[index_value] for index_value in sample_indices)
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            official_lingbot_data_config(data_mapping),
            config,
            processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )
        base_inputs = []
        collated_batches = []
        mapped_samples = []
        for offset, sample in enumerate(samples):
            mapped_samples.append(map_calvin_transition_to_lingbot(sample))
            raw_batch = build_native_calvin_training_batch(
                (sample,),
                lane_ids=(rank,),
                optimizer_step=offset,
                device=device,
                dtype=torch.bfloat16,
            )
            augmentation_seed = args.seed + rank * 100 + offset
            collated = collate_native_calvin_training_batch(
                raw_batch,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=(augmentation_seed,),
                source_digest=hashlib.sha256(
                    f"{sample.sample_key}\0{augmentation_seed}".encode()
                ).hexdigest(),
            )
            if dense_evidence_bank is not None:
                source_global_index = source_globals[offset]
                records = {
                    record.source_global_index: record
                    for record in dense_evidence_bank.caches[0].records
                }
                try:
                    canonical_key = records[source_global_index].sample_key
                except KeyError as error:
                    raise RuntimeError(
                        "ADR218 composition frame is absent from dense evidence"
                    ) from error
                evidence = dense_evidence_bank.evidence_for(
                    source_global_index=source_global_index,
                    sample_key=canonical_key,
                )
                if any(not row.available or row.token_count <= 0 for row in evidence):
                    raise RuntimeError("ADR218 composition selected missing dense evidence")
                collated = with_native_modalities(
                    collated,
                    native_modalities_from_dense_evidence(
                        (evidence,),
                        dense_evidence_bindings,
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                )
            collated_batches.append(collated)
            base_inputs.append(
                _move_model_inputs(
                    collated.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                )
            )

        if args.phase == "causality":
            prior_inputs = dict(base_inputs[0])
            current_inputs = dict(base_inputs[1])
            for inputs in (prior_inputs, current_inputs):
                actions = inputs["actions"]
                inputs["noise"] = torch.randn(
                    actions.shape,
                    device=device,
                    dtype=torch.bfloat16,
                )
                inputs["time"] = torch.full(
                    (actions.shape[0],),
                    0.5,
                    device=device,
                    dtype=torch.bfloat16,
                )
                audit_native_calvin_model_inputs(inputs, require_randomness=True)
            prior_batch, current_batch = collated_batches
            current_modalities = current_batch.modalities
            if current_modalities is None:
                raise RuntimeError("ADR218 causal gate omitted current dense modalities")

            def tensor_delta(reference: Any, candidate: Any) -> dict[str, float | bool]:
                difference = candidate.detach().float() - reference.detach().float()
                return {
                    "exact": bool(torch.equal(reference, candidate)),
                    "maximum_absolute": float(difference.abs().max()),
                    "rms": float(difference.square().mean().sqrt()),
                }

            def measure_action(
                *,
                state: NativeLayerwisePosteriorState,
                modalities: Any,
                inputs: Mapping[str, Any],
                attention_intervention: WSALingBotAttentionIntervention | None = None,
            ) -> tuple[dict[str, Any], NativeLayerwisePosteriorState]:
                captured_action: list[Any] = []
                captured_future: list[Any] = []

                def capture_joint(output: Any) -> None:
                    action_hidden = output.native_outputs[1]
                    if action_hidden is None or action_hidden.ndim != 3:
                        raise RuntimeError("WSA measurement lost its final action stream")
                    captured_action.append(action_hidden[:, 1:].detach().clone())
                    future_hidden = output.future_runtime.tokens
                    if future_hidden.ndim != 3:
                        raise RuntimeError("WSA measurement lost its final future stream")
                    captured_future.append(future_hidden.detach().clone())

                context = native_context_from_persistent_state(
                    controls=current_batch.controls,
                    persistent_state=state,
                    persistent_state_valid=torch.ones(1, dtype=torch.bool, device=device),
                    modalities=modalities,
                )
                result = run_native_policy_diagnostic_forward(
                    policy,
                    model_inputs=inputs,
                    context=context,
                    wsa_measurement_callback=capture_joint,
                    wsa_attention_intervention=attention_intervention,
                )
                if len(captured_action) != 1 or len(captured_future) != 1:
                    raise RuntimeError("WSA measurement callback did not execute exactly once")
                emitted = native_persistent_output(result.context)
                if not isinstance(emitted, NativeLayerwisePosteriorState):
                    raise RuntimeError("ADR218 causal gate emitted another posterior type")
                return (
                    {
                        "action_loss": float(result.official_action_loss.detach()),
                        "action_hidden": captured_action[0],
                        "future_hidden": captured_future[0],
                    },
                    NativeLayerwisePosteriorState(emitted.layer_rows.detach().clone()),
                )

            prior_context = native_context_from_persistent_state(
                controls=prior_batch.controls,
                persistent_state=None,
                persistent_state_valid=torch.zeros(1, dtype=torch.bool, device=device),
                modalities=prior_batch.modalities,
            )
            prior_context = run_native_policy_observation_diagnostic_forward(
                policy,
                model_inputs=prior_inputs,
                context=prior_context,
            )
            policy.model.qwenvl_with_expert.adr218_wsa_training_runtime.assert_output_consumed()
            previous_state = native_persistent_output(prior_context)
            if not isinstance(previous_state, NativeLayerwisePosteriorState):
                raise RuntimeError("ADR218 causal burn-in emitted another posterior type")
            previous_state = NativeLayerwisePosteriorState(
                previous_state.layer_rows.detach().clone()
            )
            gathered_rows = [
                torch.empty_like(previous_state.layer_rows) for _ in range(WORLD_SIZE)
            ]
            dist.all_gather(gathered_rows, previous_state.layer_rows.contiguous())
            cross_rank_state = NativeLayerwisePosteriorState(gathered_rows[1 - rank])
            zero_state = NativeLayerwisePosteriorState(
                torch.zeros_like(previous_state.layer_rows)
            )
            if torch.equal(cross_rank_state.layer_rows, previous_state.layer_rows):
                raise RuntimeError(
                    "cross-rank posterior intervention is identical to factual state"
                )

            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            factual, factual_state = measure_action(
                state=previous_state,
                modalities=current_modalities,
                inputs=current_inputs,
            )
            repeat, repeat_state = measure_action(
                state=previous_state,
                modalities=current_modalities,
                inputs=current_inputs,
            )
            blocked_future_to_action, blocked_future_to_action_state = measure_action(
                state=previous_state,
                modalities=current_modalities,
                inputs=current_inputs,
                attention_intervention=(
                    WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION
                ),
            )
            zero, zero_emitted = measure_action(
                state=zero_state,
                modalities=current_modalities,
                inputs=current_inputs,
            )
            cross_rank, cross_rank_emitted = measure_action(
                state=cross_rank_state,
                modalities=current_modalities,
                inputs=current_inputs,
            )

            candidate_inputs = dict(current_inputs)
            actions = current_inputs["actions"]
            active = ~current_inputs["action_is_pad"]
            perturbed_actions = torch.where(active.unsqueeze(-1), -actions, actions)
            if torch.equal(perturbed_actions, actions):
                raise RuntimeError("candidate-action intervention did not change active actions")
            candidate_inputs["actions"] = perturbed_actions
            candidate, candidate_state = measure_action(
                state=previous_state,
                modalities=current_modalities,
                inputs=candidate_inputs,
            )

            omission_reports: dict[str, Any] = {}
            for stream in current_modalities.streams:
                omitted, omitted_state = measure_action(
                    state=previous_state,
                    modalities=current_modalities.omit((stream.name,)),
                    inputs=current_inputs,
                )
                omission_reports[stream.name] = {
                    "action_loss": omitted["action_loss"],
                    "action_loss_absolute_delta": abs(
                        omitted["action_loss"] - factual["action_loss"]
                    ),
                    "action_hidden_delta": tensor_delta(
                        factual["action_hidden"], omitted["action_hidden"]
                    ),
                    "future_hidden_delta": tensor_delta(
                        factual["future_hidden"], omitted["future_hidden"]
                    ),
                    "emitted_posterior_delta": tensor_delta(
                        factual_state.layer_rows,
                        omitted_state.layer_rows,
                    ),
                }

            repeat_action_delta = tensor_delta(
                factual["action_hidden"], repeat["action_hidden"]
            )
            repeat_future_delta = tensor_delta(
                factual["future_hidden"], repeat["future_hidden"]
            )
            repeat_state_delta = tensor_delta(
                factual_state.layer_rows,
                repeat_state.layer_rows,
            )
            candidate_state_delta = tensor_delta(
                factual_state.layer_rows,
                candidate_state.layer_rows,
            )
            zero_action_delta = tensor_delta(
                factual["action_hidden"], zero["action_hidden"]
            )
            cross_rank_action_delta = tensor_delta(
                factual["action_hidden"], cross_rank["action_hidden"]
            )
            candidate_future_delta = tensor_delta(
                factual["future_hidden"], candidate["future_hidden"]
            )
            blocked_future_to_action_delta = tensor_delta(
                factual["action_hidden"],
                blocked_future_to_action["action_hidden"],
            )
            blocked_future_to_action_state_delta = tensor_delta(
                factual_state.layer_rows,
                blocked_future_to_action_state.layer_rows,
            )
            minimum_effect = max(1e-6, 10.0 * float(repeat_action_delta["rms"]))
            posterior_reaches_action = max(
                float(zero_action_delta["rms"]),
                float(cross_rank_action_delta["rms"]),
            ) > minimum_effect
            all_modalities_reach_action = all(
                float(value["action_hidden_delta"]["rms"]) > minimum_effect
                for value in omission_reports.values()
            )
            candidate_reaches_action = (
                float(
                    tensor_delta(
                        factual["action_hidden"], candidate["action_hidden"]
                    )["rms"]
                )
                > minimum_effect
            )
            candidate_reaches_future = (
                float(candidate_future_delta["rms"]) > minimum_effect
            )
            future_reaches_action = (
                float(blocked_future_to_action_delta["rms"]) > minimum_effect
            )
            accepted = (
                bool(repeat_action_delta["exact"])
                and bool(repeat_future_delta["exact"])
                and bool(repeat_state_delta["exact"])
                and bool(candidate_state_delta["exact"])
                and bool(blocked_future_to_action_state_delta["exact"])
                and posterior_reaches_action
                and all_modalities_reach_action
                and candidate_reaches_action
                and candidate_reaches_future
                and future_reaches_action
            )
            if not accepted:
                raise RuntimeError("ADR218 fixed-weight causal reach gate failed")
            return_report = {
                "rank": rank,
                "source_global_indices": list(source_globals),
                "sample_keys": [sample.sample_key for sample in samples],
                "task": mapped_samples[1].task,
                "factual_action_loss": factual["action_loss"],
                "repeat_action_loss_absolute_delta": abs(
                    repeat["action_loss"] - factual["action_loss"]
                ),
                "repeat_action_hidden_delta": repeat_action_delta,
                "repeat_future_hidden_delta": repeat_future_delta,
                "repeat_emitted_posterior_delta": repeat_state_delta,
                "zero_posterior": {
                    "action_loss": zero["action_loss"],
                    "action_loss_absolute_delta": abs(
                        zero["action_loss"] - factual["action_loss"]
                    ),
                    "action_hidden_delta": zero_action_delta,
                    "future_hidden_delta": tensor_delta(
                        factual["future_hidden"], zero["future_hidden"]
                    ),
                    "emitted_posterior_delta": tensor_delta(
                        factual_state.layer_rows,
                        zero_emitted.layer_rows,
                    ),
                },
                "cross_rank_posterior": {
                    "action_loss": cross_rank["action_loss"],
                    "action_loss_absolute_delta": abs(
                        cross_rank["action_loss"] - factual["action_loss"]
                    ),
                    "input_posterior_delta": tensor_delta(
                        previous_state.layer_rows,
                        cross_rank_state.layer_rows,
                    ),
                    "action_hidden_delta": cross_rank_action_delta,
                    "future_hidden_delta": tensor_delta(
                        factual["future_hidden"], cross_rank["future_hidden"]
                    ),
                    "emitted_posterior_delta": tensor_delta(
                        factual_state.layer_rows,
                        cross_rank_emitted.layer_rows,
                    ),
                },
                "candidate_action": {
                    "action_loss": candidate["action_loss"],
                    "action_hidden_delta": tensor_delta(
                        factual["action_hidden"], candidate["action_hidden"]
                    ),
                    "future_hidden_delta": candidate_future_delta,
                    "emitted_posterior_delta": candidate_state_delta,
                },
                "blocked_future_to_action": {
                    "action_loss": blocked_future_to_action["action_loss"],
                    "action_loss_absolute_delta": abs(
                        blocked_future_to_action["action_loss"] - factual["action_loss"]
                    ),
                    "action_hidden_delta": blocked_future_to_action_delta,
                    "future_hidden_delta": tensor_delta(
                        factual["future_hidden"],
                        blocked_future_to_action["future_hidden"],
                    ),
                    "emitted_posterior_delta": blocked_future_to_action_state_delta,
                },
                "modality_omissions": omission_reports,
                "minimum_detectable_effect_rms": minimum_effect,
                "posterior_reaches_action": posterior_reaches_action,
                "all_modalities_reach_action": all_modalities_reach_action,
                "candidate_reaches_action": candidate_reaches_action,
                "candidate_reaches_future": candidate_reaches_future,
                "future_reaches_action": future_reaches_action,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
            }
            gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered, return_report)
            if rank == 0:
                if args.run_dir.exists() or args.run_dir.is_symlink():
                    raise FileExistsError(args.run_dir)
                args.run_dir.mkdir(parents=True)
                report = {
                    "schema": "picf-next.adr220-full-modal-fixed-weight-causality.v1",
                    "status": "PASS",
                    "world_size": WORLD_SIZE,
                    "scope": (
                        "fixed-weight exact-graph reachability only; posterior and modality "
                        "interventions, action-to-posterior non-leakage, no advantage claim"
                    ),
                    "rank_reports": gathered,
                    "scientific_advantage_claimed": False,
                }
                write_text_durable_exclusive(
                    args.run_dir / "causality_report.json",
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                print(json.dumps(report, indent=2, sort_keys=True))
            dist.barrier()
            return

        from lerobot.policies.WSA_Base.da3_teacher import DA3BackboneTeacher

        teacher_batches = []
        for mapped in mapped_samples:
            views = []
            for array in (mapped.camera_top, mapped.camera_wrist_left):
                view = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.0
                views.append(
                    F.interpolate(
                        view.unsqueeze(0),
                        size=(504, 504),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                )
            teacher_batches.append(torch.stack(views))
        teacher_images = torch.stack(teacher_batches).to(device)
        teacher = DA3BackboneTeacher(
            str(args.da3_model_dir.resolve()),
            process_res=504,
            dtype=torch.bfloat16,
            teacher_layers=(11, 15, 19, 23),
            code_root=str(args.da3_source_root.resolve()),
        ).to(device)
        with torch.inference_mode():
            all_teacher_layers = tuple(layer.cpu() for layer in teacher(teacher_images))
        teacher_layers_by_step = tuple(
            tuple(layer[offset : offset + 1] for layer in all_teacher_layers) for offset in range(2)
        )
        del teacher, teacher_images, all_teacher_layers
        torch.cuda.empty_cache()

        def run_step(step_index: int) -> dict[str, object]:
            _emit_phase(rank=rank, phase=f"step-{step_index}-start")
            optimizer.zero_grad(set_to_none=True)
            model_inputs = dict(base_inputs[step_index - 1])
            actions = model_inputs["actions"]
            model_inputs["noise"] = torch.randn(
                actions.shape,
                device=device,
                dtype=torch.bfloat16,
            )
            model_inputs["time"] = torch.full(
                (actions.shape[0],),
                0.5,
                device=device,
                dtype=torch.bfloat16,
            )
            audit_native_calvin_model_inputs(model_inputs, require_randomness=True)
            teacher_layers = tuple(
                layer.to(device=device, non_blocking=False)
                for layer in teacher_layers_by_step[step_index - 1]
            )
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            teacher_targets = WSADA3TeacherTargets(
                layers=teacher_layers,
                view_valid=torch.ones(
                    1,
                    2,
                    dtype=torch.bool,
                    device=device,
                ),
            )
            with wsa_lingbot_optimizer_transaction(policy) as wsa_step_ledger:
                outputs = policy(
                    **model_inputs,
                    compute_alignment_losses=False,
                    **wsa_lingbot_forward_kwargs(
                        policy,
                        role=WSALingBotForwardRole.PRIMARY_FACTUAL,
                        teacher_targets=teacher_targets,
                    ),
                )
            if wsa_step_ledger is None:
                raise RuntimeError("ADR218 WSA step ledger was not installed")
            wsa_step_receipt = wsa_step_ledger.receipt()
            _emit_phase(rank=rank, phase=f"step-{step_index}-forward-complete")
            total_loss = outputs[0]
            if not torch.isfinite(total_loss):
                raise RuntimeError("ADR218 FSDP loss is non-finite")
            total_loss_value = float(total_loss.detach())
            action_loss_value = float(outputs[1].detach())
            future_3d_loss_value = float(outputs[6]["loss_future_3d"])
            total_loss.backward()
            _emit_phase(rank=rank, phase=f"step-{step_index}-backward-complete")
            metrics = _distributed_gradient_metrics(
                policy,
                device=device,
                dist=dist,
                torch_module=torch,
            )
            _validate_gradient_metrics(metrics)
            _emit_phase(rank=rank, phase=f"step-{step_index}-gradient-audit-complete")
            preclip_norm = clip_lingbot_distributed_l2_grad_norm_(
                policy.parameters(),
                float(WSA_LARGE_OPTIMIZER["grad_clip_norm"]),
                device=device,
                dist_module=dist,
                torch_module=torch,
                error_if_nonfinite=True,
            )
            del outputs, total_loss, teacher_layers
            _emit_phase(rank=rank, phase=f"step-{step_index}-optimizer-start")
            optimizer.step()
            _emit_phase(rank=rank, phase=f"step-{step_index}-optimizer-complete")
            optimizer.zero_grad(set_to_none=True)
            optimizer_state = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=step_index,
            )
            torch.cuda.synchronize(device)
            return {
                "step": step_index,
                "sample_index": sample_indices[step_index - 1],
                "sample_key": samples[step_index - 1].sample_key,
                "task": mapped_samples[step_index - 1].task,
                "total_loss": total_loss_value,
                "action_loss": action_loss_value,
                "future_3d_loss": future_3d_loss_value,
                "preclip_global_norm": float(preclip_norm),
                "gradient_metrics": metrics,
                "optimizer_state": optimizer_state,
                "wsa_step_ledger": wsa_step_receipt,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "attention_implementation": (
                    policy.model.qwenvl_with_expert.config.attention_implementation
                ),
            }

        def run_full_modal_composition() -> dict[str, object]:
            if graph is None or dense_evidence_bank is None:
                raise RuntimeError("ADR218 full-modal composition graph is incomplete")
            if any(batch.modalities is None for batch in collated_batches):
                raise RuntimeError("ADR218 full-modal composition omitted dense modalities")
            if mapped_samples[0].task != mapped_samples[1].task:
                raise RuntimeError("ADR218 composition pair crossed a CALVIN instruction boundary")
            optimizer.zero_grad(set_to_none=True)
            prior_inputs = dict(base_inputs[0])
            current_inputs = dict(base_inputs[1])
            for inputs in (prior_inputs, current_inputs):
                actions = inputs["actions"]
                inputs["noise"] = torch.randn(
                    actions.shape,
                    device=device,
                    dtype=torch.bfloat16,
                )
                inputs["time"] = torch.full(
                    (actions.shape[0],),
                    0.5,
                    device=device,
                    dtype=torch.bfloat16,
                )
                audit_native_calvin_model_inputs(inputs, require_randomness=True)

            prior_batch, current_batch = collated_batches
            current_modalities = current_batch.modalities
            if current_modalities is None:
                raise RuntimeError("ADR218 current composition frame omitted modalities")
            modality_leaves: dict[str, tuple[Any, Any | None]] = {}
            for stream in current_modalities.streams:
                stream.tokens.requires_grad_(True)
                if stream.metadata is not None:
                    stream.metadata.requires_grad_(True)
                modality_leaves[stream.name] = (stream.tokens, stream.metadata)

            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            previous_valid = torch.zeros(1, dtype=torch.bool, device=device)
            prior_context = native_context_from_persistent_state(
                controls=prior_batch.controls,
                persistent_state=None,
                persistent_state_valid=previous_valid,
                modalities=prior_batch.modalities,
            )
            prior_context = run_native_policy_observation_diagnostic_forward(
                policy,
                model_inputs=prior_inputs,
                context=prior_context,
            )
            policy.model.qwenvl_with_expert.adr218_wsa_training_runtime.assert_output_consumed()
            previous_state = native_persistent_output(prior_context)
            if previous_state.layer_rows.requires_grad:
                raise RuntimeError("ADR218 no-grad burn-in retained an autograd graph")
            current_context = native_context_from_persistent_state(
                controls=current_batch.controls,
                persistent_state=previous_state,
                persistent_state_valid=torch.ones(1, dtype=torch.bool, device=device),
                modalities=current_modalities,
            )
            teacher_targets = WSADA3TeacherTargets(
                layers=tuple(
                    layer.to(device=device, non_blocking=False)
                    for layer in teacher_layers_by_step[1]
                ),
                view_valid=torch.ones(1, 2, dtype=torch.bool, device=device),
            )
            action_attention_collector = RegisteredActionPosteriorReceiptCollector(
                registered_layer_indices=(0, 17, 35),
            )
            with wsa_lingbot_optimizer_transaction(policy) as wsa_step_ledger:
                result = run_native_policy_training_forward(
                    policy,
                    model_inputs=current_inputs,
                    context=current_context,
                    wsa_da3_teacher_targets=teacher_targets,
                    action_attention_callback=action_attention_collector,
                )
            if wsa_step_ledger is None:
                raise RuntimeError("ADR218 full-modal composition omitted its WSA ledger")
            total_loss = result.official_total_loss
            if not torch.isfinite(total_loss):
                raise RuntimeError("ADR218 full-modal composition loss is non-finite")
            total_loss_value = float(total_loss.detach())
            action_loss_value = float(result.official_action_loss.detach())
            future_3d_loss_value = float(result.official_outputs[6]["loss_future_3d"])
            action_attention_receipts = action_attention_collector.finalize()
            action_attention_summary = [
                {
                    "layer_index": receipt.layer_index,
                    "shape": list(receipt.posterior_attention.shape),
                    "mean_total_posterior_mass": float(
                        receipt.total_posterior_mass.detach().float().mean()
                    ),
                    "maximum_total_posterior_mass": float(
                        receipt.total_posterior_mass.detach().float().max()
                    ),
                }
                for receipt in action_attention_receipts
            ]
            if any(
                not math.isfinite(row["mean_total_posterior_mass"])
                or not math.isfinite(row["maximum_total_posterior_mass"])
                for row in action_attention_summary
            ):
                raise RuntimeError("ADR218 synchronous action attention receipt is non-finite")
            total_loss.backward()
            gradient_metrics = _distributed_gradient_metrics(
                policy,
                device=device,
                dist=dist,
                torch_module=torch,
                fragments=FULL_MODAL_GRADIENT_FRAGMENTS,
            )
            _validate_gradient_metrics(
                gradient_metrics,
                fragments=FULL_MODAL_GRADIENT_FRAGMENTS,
            )
            modality_gradient_squares: dict[str, float] = {}
            for name, (tokens, metadata) in modality_leaves.items():
                gradients = [tokens.grad, None if metadata is None else metadata.grad]
                if gradients[0] is None or (metadata is not None and gradients[1] is None):
                    raise RuntimeError(f"ADR218 {name} evidence is detached from the action loss")
                local_square = sum(
                    gradient.detach().float().square().sum()
                    for gradient in gradients
                    if gradient is not None
                )
                dist.all_reduce(local_square, op=dist.ReduceOp.SUM)
                value = float(local_square)
                if not math.isfinite(value) or value <= 0.0:
                    raise RuntimeError(f"ADR218 {name} evidence has no nonzero action gradient")
                modality_gradient_squares[name] = value
            preclip_norm = clip_lingbot_distributed_l2_grad_norm_(
                policy.parameters(),
                float(WSA_LARGE_OPTIMIZER["grad_clip_norm"]),
                device=device,
                dist_module=dist,
                torch_module=torch,
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_state = _validate_optimizer_state(
                optimizer,
                torch,
                expected_step=1,
            )
            torch.cuda.synchronize(device)
            return {
                "rank": rank,
                "source_global_indices": list(source_globals),
                "sample_keys": [sample.sample_key for sample in samples],
                "task": mapped_samples[1].task,
                "modalities": list(dense_evidence_bank.modalities),
                "modality_token_counts": {
                    stream.name: int(stream.valid.sum())
                    for stream in current_modalities.streams
                },
                "observation_only_wsa_executions": 0,
                "total_loss": total_loss_value,
                "action_loss": action_loss_value,
                "future_3d_loss": future_3d_loss_value,
                "wsa_step_ledger": wsa_step_ledger.receipt(),
                "action_posterior_attention": action_attention_summary,
                "gradient_metrics": gradient_metrics,
                "modality_input_gradient_square_sums": modality_gradient_squares,
                "preclip_global_norm": float(preclip_norm),
                "optimizer_state": optimizer_state,
                "elapsed_seconds": time.perf_counter() - started,
                "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                "attention_implementation": (
                    policy.model.qwenvl_with_expert.config.attention_implementation
                ),
            }

        if args.phase == "composition":
            if rank == 0:
                if args.run_dir.exists() or args.run_dir.is_symlink():
                    raise FileExistsError(args.run_dir)
                args.run_dir.mkdir(parents=True)
                _fsync_directory(args.run_dir.parent)
            dist.barrier()
            rank_report = run_full_modal_composition()
            gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered, rank_report)
            if rank == 0:
                report = {
                    "schema": "picf-next.adr220-full-modal-composition.v1",
                    "status": "PASS",
                    "world_size": WORLD_SIZE,
                    "launch_binding": {
                        "argv": sys.argv[1:],
                        "python_executable": str(Path(sys.executable).resolve()),
                        "training_config": {
                            "path": str(args.training_config.resolve()),
                            "sha256": _sha256(args.training_config),
                        },
                        "robot_config": {
                            "path": str(args.robot_config.resolve()),
                            "sha256": _sha256(args.robot_config),
                        },
                        "dense_evidence": [
                            {
                                "root": str(root.resolve()),
                                "manifest_sha256": digest,
                            }
                            for root, digest in zip(
                                args.dense_evidence_cache_root,
                                args.dense_evidence_cache_manifest_sha256,
                                strict=True,
                            )
                        ],
                    },
                    "scope": (
                        "complete LingBot host/action, layerwise persistent PICF graph, exact "
                        "V-JEPA/AnyTouch/Sonata cache ingress, complete WSA Future3D and all "
                        "four official DA3 targets; mechanics and gradient mediation only"
                    ),
                    "dataset_binding": dataset_binding,
                    "wsa_installation": wsa_lingbot_installation_receipt(policy),
                    "wsa_optimizer": optimizer_receipt,
                    "dense_cache_modalities": list(dense_evidence_bank.modalities),
                    "rank_reports": gathered,
                    "scientific_advantage_claimed": False,
                }
                write_text_durable_exclusive(
                    args.run_dir / "composition_report.json",
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                print(json.dumps(report, indent=2, sort_keys=True))
            dist.barrier()
            return

        checkpoint_dir = args.run_dir / "checkpoint_step_1"
        staging_dir = args.run_dir / "checkpoint_step_1.staging"
        fresh_report_path = args.run_dir / "fresh_report.json"
        resume_report_path = args.run_dir / "resume_report.json"
        lane_snapshot = f"adr218-rank-{rank}".encode("ascii")

        if args.phase == "fresh":
            if rank == 0:
                if args.run_dir.exists() or args.run_dir.is_symlink():
                    raise FileExistsError(args.run_dir)
                args.run_dir.mkdir(parents=True)
                _fsync_directory(args.run_dir.parent)
            dist.barrier()
            random.seed(args.seed + rank)
            np.random.seed(args.seed + rank)
            torch.manual_seed(args.seed + rank)
            torch.cuda.manual_seed(args.seed + rank)
            step_one = run_step(1)
            _emit_phase(rank=rank, phase="step-1-complete")
            checkpoint_rng = _capture_rank_rng(torch, np, device=device)
            saved_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=checkpoint_rng,
                torch_module=torch,
            )
            _emit_phase(rank=rank, phase="step-1-boundary-complete")
            extra_state = {
                "schema": EXTRA_SCHEMA,
                "global_step": 1,
                "rank": rank,
                "world_size": WORLD_SIZE,
                "boundary": saved_boundary,
                "rank_rng_state": checkpoint_rng,
            }
            checkpointer.save(
                str(staging_dir),
                {"model": policy, "optimizer": optimizer, "extra_state": extra_state},
                global_steps=None,
            )
            _emit_phase(rank=rank, phase="checkpoint-save-complete")
            dist.barrier()
            if rank == 0:
                staging_dir.rename(checkpoint_dir)
                _emit_phase(rank=rank, phase="checkpoint-rename-complete")
                _fsync_directory(args.run_dir)
                _emit_phase(rank=rank, phase="checkpoint-directory-fsync-complete")
            dist.barrier()
            _release_checkpoint_allocator_cache(
                rank=rank,
                device=device,
                torch_module=torch,
            )
            _restore_rank_rng(checkpoint_rng, torch, np, device=device)
            step_two = run_step(2)
            _emit_phase(rank=rank, phase="step-2-complete")
            expected_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=_capture_rank_rng(torch, np, device=device),
                torch_module=torch,
            )
            _emit_phase(rank=rank, phase="step-2-boundary-complete")
            rank_report = {
                "rank": rank,
                "saved_boundary": saved_boundary,
                "expected_resumed_boundary": expected_boundary,
                "step_one": step_one,
                "step_two": step_two,
            }
            gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered, rank_report)
            if rank == 0:
                report = {
                    "schema": SCHEMA,
                    "phase": "fresh",
                    "status": "PASS",
                    "world_size": WORLD_SIZE,
                    "load_seconds": load_seconds,
                    "dataset_binding": dataset_binding,
                    "dataset_manifest_file_sha256": _sha256(args.dataset_manifest),
                    "normalization_artifact_sha256": norm_payload["artifact_sha256"],
                    "wsa_installation": wsa_lingbot_installation_receipt(policy),
                    "wsa_fsdp_forward_methods": wsa_fsdp_forward_methods,
                    "wsa_optimizer": optimizer_receipt,
                    "selective_cpu_modules": list(selective_cpu_modules),
                    "selective_cpu_module_classes": list(
                        selective_cpu_module_classes
                    ),
                    "fsdp_offload_mode": FSDP_OFFLOAD_MODE,
                    "da3_teacher_cache_residency": "cpu-between-updates",
                    "dcp_backport": dcp_backport,
                    "determinism": determinism_receipt,
                    "checkpoint": str(checkpoint_dir.resolve()),
                    "rank_reports": gathered,
                    "scheduler_status": "open-before-long-training",
                }
                write_text_durable_exclusive(
                    fresh_report_path,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                print(json.dumps(report, indent=2, sort_keys=True))
        else:
            if not checkpoint_dir.is_dir() or not fresh_report_path.is_file():
                raise FileNotFoundError("ADR218 fresh checkpoint/report is incomplete")
            if resume_report_path.exists() or resume_report_path.is_symlink():
                raise FileExistsError(resume_report_path)
            state = {"model": policy, "optimizer": optimizer, "extra_state": {}}
            checkpointer.load(str(checkpoint_dir), state)
            extra_state = state["extra_state"]
            if (
                not isinstance(extra_state, dict)
                or extra_state.get("schema") != EXTRA_SCHEMA
                or extra_state.get("global_step") != 1
                or extra_state.get("rank") != rank
                or extra_state.get("world_size") != WORLD_SIZE
            ):
                raise RuntimeError("ADR218 cold-resume extra state differs")
            _validate_optimizer_state(optimizer, torch, expected_step=1)
            loaded_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=extra_state["rank_rng_state"],
                torch_module=torch,
            )
            if loaded_boundary != extra_state["boundary"]:
                raise RuntimeError("ADR218 cold-resume checkpoint boundary differs")
            _release_checkpoint_allocator_cache(
                rank=rank,
                device=device,
                torch_module=torch,
            )
            _restore_rank_rng(extra_state["rank_rng_state"], torch, np, device=device)
            step_two = run_step(2)
            resumed_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=lane_snapshot,
                rank_rng_state=_capture_rank_rng(torch, np, device=device),
                torch_module=torch,
            )
            fresh_report = json.loads(fresh_report_path.read_text(encoding="utf-8"))
            expected = fresh_report["rank_reports"][rank]
            if expected["rank"] != rank:
                raise RuntimeError("ADR218 fresh rank order differs")
            if resumed_boundary != expected["expected_resumed_boundary"]:
                boundary_differences = {
                    name: {
                        "actual": resumed_boundary[name],
                        "expected": expected["expected_resumed_boundary"][name],
                    }
                    for name in resumed_boundary
                    if resumed_boundary[name] != expected["expected_resumed_boundary"][name]
                }
                print(
                    json.dumps(
                        {
                            "schema": "picf-next.adr218-cold-resume-failure.v1",
                            "rank": rank,
                            "boundary_differences": boundary_differences,
                            "actual_step_two": step_two,
                            "expected_step_two": expected["step_two"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                raise RuntimeError("ADR218 cold-resume trajectory is not bit-exact")
            if step_two["total_loss"] != expected["step_two"]["total_loss"]:
                raise RuntimeError("ADR218 cold-resume loss differs")
            rank_report = {
                "rank": rank,
                "loaded_boundary": loaded_boundary,
                "resumed_boundary": resumed_boundary,
                "step_two": step_two,
            }
            gathered = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered, rank_report)
            if rank == 0:
                report = {
                    "schema": SCHEMA,
                    "phase": "resume",
                    "status": "PASS",
                    "world_size": WORLD_SIZE,
                    "cold_resume_bit_exact": True,
                    "load_seconds": load_seconds,
                    "fresh_report_sha256": _sha256(fresh_report_path),
                    "determinism": determinism_receipt,
                    "wsa_optimizer": optimizer_receipt,
                    "selective_cpu_modules": list(selective_cpu_modules),
                    "selective_cpu_module_classes": list(
                        selective_cpu_module_classes
                    ),
                    "fsdp_offload_mode": FSDP_OFFLOAD_MODE,
                    "da3_teacher_cache_residency": "cpu-between-updates",
                    "rank_reports": gathered,
                    "scheduler_status": "open-before-long-training",
                }
                write_text_durable_exclusive(
                    resume_report_path,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
                print(json.dumps(report, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

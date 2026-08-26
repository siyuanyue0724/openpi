#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Prove shared-Qwen native grounding gradients through production FSDP2."""

from __future__ import annotations

import argparse
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

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.contracts import ContractError
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_GPU_SHARDED,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
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
    build_lingbot_official_optimizer,
    load_lingbot_training_config,
    register_native_fsdp_forward_methods,
    resolve_lingbot_optimizer_contract,
    strip_targetless_alignment_teacher_heads,
)
from tools.probe_qwen3vl_grounding_baseline import (
    INPUT_SCHEMA,
    _load_probe_report,
    _record_from_payload,
)

WORLD_SIZE = 2
OUTPUT_SCHEMA = "picf-next.lingbot-native-vl-grounding-fsdp2-g1.v1"
_GRADIENT_FRAGMENTS = (
    (
        "shared_embedding",
        "model.qwenvl_with_expert.qwenvl.model.language_model.embed_tokens.weight",
    ),
    ("language_layers", "model.qwenvl_with_expert.qwenvl.model.language_model.layers"),
    ("vision_layers", "model.qwenvl_with_expert.qwenvl.model.visual"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--input-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_GPU_SHARDED,
    )
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> None:
    for path in (
        args.training_config,
        args.dataset_manifest,
        args.input_report,
        args.source_checkout / MODEL_SOURCE,
        _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    for path in (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
    ):
        if not path.is_dir():
            raise FileNotFoundError(path)
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(args.output_dir)
    if isinstance(args.seed, bool) or not isinstance(args.seed, int) or args.seed < 0:
        raise ContractError("native VL FSDP2 seed must be non-negative")
    _validate_native_vl_fsdp2_placement(args.fsdp2_placement)


def _validate_native_vl_fsdp2_placement(value: object) -> str:
    """Keep tied input/output embeddings in one FSDP2 parameter group."""

    placement = validate_fsdp2_placement(value)
    if placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
        raise ContractError(
            "native VL tied embeddings cannot use selective embedding offload; "
            "FSDP2 shared parameters must belong to one fully_shard group"
        )
    return placement


def _rank_record_indices(record_count: int, world_size: int = WORLD_SIZE) -> tuple[int, ...]:
    """Choose deterministic, widely separated records for distributed ranks."""

    if record_count < world_size or world_size <= 0:
        raise ContractError("native VL FSDP2 probe has too few records for its ranks")
    indices = tuple((rank * record_count) // world_size for rank in range(world_size))
    if len(set(indices)) != world_size:
        raise ContractError("native VL FSDP2 rank record selection is not unique")
    return indices


def _distributed_gradient_metrics(
    model: Any,
    *,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, float | int | bool]:
    squares: dict[str, Any | None] = {name: None for name, _ in _GRADIENT_FRAGMENTS}
    counts = {name: 0 for name, _ in _GRADIENT_FRAGMENTS}
    finite = torch_module.ones((), dtype=torch_module.int32, device=device)
    for parameter_name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            continue
        local = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
        finite.mul_(torch_module.isfinite(local).all().to(device=device, dtype=torch_module.int32))
        for metric_name, fragment in _GRADIENT_FRAGMENTS:
            if fragment not in parameter_name:
                continue
            value = local.detach().float().square().sum().to(device=device)
            previous = squares[metric_name]
            squares[metric_name] = value if previous is None else previous + value
            counts[metric_name] += int(local.numel())
    dist.all_reduce(finite, op=dist.ReduceOp.MIN)
    packed = []
    for metric_name, _ in _GRADIENT_FRAGMENTS:
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
    for index, (metric_name, _) in enumerate(_GRADIENT_FRAGMENTS):
        result[f"{metric_name}_norm"] = math.sqrt(float(values[index * 2]))
        result[f"{metric_name}_elements"] = int(values[index * 2 + 1])
    return result


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    patch_report = verify_native_vl_patch(root=_ROOT, checkout=args.source_checkout)
    overlay = _ROOT / NATIVE_VL_PATCH_RELATIVE_PATH
    if detect_native_vl_patch_state(args.source_checkout, overlay) != "applied":
        raise RuntimeError("native VL FSDP2 source overlay is not applied")
    if _validate_native_vl_model(args.source_checkout / MODEL_SOURCE) != (
        NATIVE_VL_PATCHED_MODEL_SHA256
    ):
        raise RuntimeError("native VL FSDP2 source digest differs")
    commit = subprocess.run(
        ["git", "-C", str(args.source_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("native VL FSDP2 source commit differs")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("native VL FSDP2 probe requires exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("native VL FSDP2 probe requires two local GPUs")

    sys.path.insert(0, str(args.source_checkout.resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
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

    from picf_next.data.calvin import CalvinDatasetIndex
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.vl_cotraining import (
        build_native_vl_grounding_batch,
        register_native_vl_fsdp_forward_method,
        retie_and_validate_native_qwen_lm_head,
        run_native_vl_grounding_forward,
        validate_native_vl_optimizer_membership,
        validate_tied_qwen_lm_head,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError("native VL FSDP2 probe sees an unexpected CUDA topology")
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
        report = _load_probe_report(args.input_report)
        if report.get("schema") != INPUT_SCHEMA:
            raise ContractError("native VL FSDP2 input schema differs")
        manifest = load_dataset_file_manifest(args.dataset_manifest)
        validate_dataset_runtime_binding(
            manifest,
            args.dataset_split,
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            split_name=args.dataset_split.name,
        )
        if report.get("dataset_manifest_sha256") != manifest.tree_sha256:
            raise ContractError("native VL FSDP2 input belongs to another dataset tree")
        index = CalvinDatasetIndex.load(
            args.dataset_split,
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        raw_records = report.get("records")
        if not isinstance(raw_records, list):
            raise ContractError("native VL FSDP2 input contains no record list")
        record_indices = _rank_record_indices(len(raw_records))
        record = _record_from_payload(index, raw_records[record_indices[rank]])

        training = load_lingbot_training_config(args.training_config)
        train_values = training.get("train")
        if not isinstance(train_values, dict):
            raise ContractError("native VL FSDP2 training config has no train mapping")
        learning_rate = train_values.get("lr", 5e-5)
        if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
            raise ContractError("native VL FSDP2 learning rate is not numeric")
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(learning_rate),
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
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
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
            raise ContractError("native VL tied parameter name changed during loading")
        teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        policy.train()
        full_cpu_offload = args.fsdp2_placement == FSDP2_CPU_OFFLOAD
        selective_offload = args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=full_cpu_offload,
            enable_shared_embedding_offload=selective_offload,
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
        sharded_tied_name = validate_tied_qwen_lm_head(policy)
        if sharded_tied_name != loaded_tied_name:
            raise ContractError("native VL tied parameter name changed during FSDP2 sharding")
        optimizer = build_lingbot_official_optimizer(
            policy,
            optimizer_contract,
            build_muon_optimizer=build_muon_optimizer,
            build_moe_load_balance_hook=build_moe_load_balance_hook,
        )
        optimizer_tied_name = validate_native_vl_optimizer_membership(policy, optimizer)
        batch = build_native_vl_grounding_batch(record, processor).to(
            device,
            pixel_dtype=torch.bfloat16,
        )
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        loss = run_native_vl_grounding_forward(policy, batch)
        loss.backward()
        elapsed = time.perf_counter() - started
        gradient_metrics = _distributed_gradient_metrics(
            policy,
            device=device,
            dist=dist,
            torch_module=torch,
        )
        if not bool(gradient_metrics["all_finite"]):
            raise RuntimeError("native VL FSDP2 grounding gradients are non-finite")
        for name, _ in _GRADIENT_FRAGMENTS:
            if int(gradient_metrics[f"{name}_elements"]) <= 0:
                raise RuntimeError(f"native VL FSDP2 produced no {name} gradient elements")
            if float(gradient_metrics[f"{name}_norm"]) <= 0.0:
                raise RuntimeError(f"native VL FSDP2 produced a zero {name} gradient")
        rank_report = {
            "camera_name": record.camera_name,
            "elapsed_seconds": elapsed,
            "global_index": record.global_index,
            "gradient_metrics": gradient_metrics,
            "instruction": record.instruction,
            "loss": float(loss.detach().item()),
            "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / (1024**3),
            "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / (1024**3),
            "rank": rank,
            "record_index": record_indices[rank],
            "supervised_token_count": batch.supervised_token_count,
            "target_identity_key": record.target_identity_key,
            "task_key": record.task_key,
        }
        gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
        dist.all_gather_object(gathered, rank_report)
        if rank == 0:
            output = {
                "dataset_manifest_sha256": manifest.tree_sha256,
                "fsdp2_placement": args.fsdp2_placement,
                "input_report_sha256": _sha256(args.input_report),
                "native_vl_patch_sha256": patch_report["native_vl_patch_sha256"],
                "optimizer_tied_parameter_name": optimizer_tied_name,
                "rank_reports": gathered,
                "record_indices": list(record_indices),
                "schema": OUTPUT_SCHEMA,
                "source_commit": commit,
                "status": "PASS",
                "teacher_prune": teacher_prune,
                "tied_parameter_name": sharded_tied_name,
                "world_size": WORLD_SIZE,
            }
            args.output_dir.mkdir(parents=True)
            write_text_durable_exclusive(
                args.output_dir / "report.json",
                json.dumps(output, indent=2, sort_keys=True) + "\n",
            )
            print(json.dumps(output, indent=2, sort_keys=True))
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

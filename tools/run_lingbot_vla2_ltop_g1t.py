#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Run strict two-GPU training-ABI parity for the ADR-159 LTOP graph.

G1T compares one released no-PICF LingBot training forward with the
``ObjectReadActionIntervention.BLOCKED`` exact-native-cache training forward.
Every rank reuses one immutable CALVIN batch, explicit flow noise/time, and an
identical process RNG snapshot.  The two paths are executed serially so their
autograd graphs are never resident together.

Exact family-wise gradient cosine is recovered without copying the 6B gradient
vector.  The runner measures ``||g_released||^2``, accumulates one BLOCKED
backward to measure ``||g_released + g_blocked||^2``, then clears gradients and
replays BLOCKED to measure ``||g_blocked||^2``.  The polarization identity gives
the exact distributed dot product.  This adds one bounded BLOCKED replay but no
learned module and no second retained graph.

Accelerator and upstream imports remain inside :func:`main`, keeping argument,
gradient algebra, and report validation locally testable.
"""

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
from typing import Any, cast

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
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from tools.bootstrap_lingbot_vla2_native import (
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _RouteTrace,
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        resolve_lingbot_optimizer_contract,
        select_lingbot_deterministic_moe_backend,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _capture_rank_rng,
        _distributed_rank_local_call,
        _implementation_digest,
        _move_model_inputs,
        _rank_rng_digest,
        _restore_rank_rng,
        _validate_fsdp2_parameter_storage,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_REVISION,
        QWEN_PROCESSOR_REVISION,
        validate_checkpoint,
        validate_processor,
    )
    from bootstrap_lingbot_vla2_native import (  # type: ignore[no-redef]
        CHECKOUT_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        PATCH_RELATIVE_PATH,
        validate_prepared_native_source,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        _RouteTrace,
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        resolve_lingbot_optimizer_contract,
        select_lingbot_deterministic_moe_backend,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _capture_rank_rng,
        _distributed_rank_local_call,
        _implementation_digest,
        _move_model_inputs,
        _rank_rng_digest,
        _restore_rank_rng,
        _validate_fsdp2_parameter_storage,
    )


G1T_WORLD_SIZE = 2
G1T_PHYSICAL_CAPACITY = 16
G1T_TASK_QUERY_COUNT = 4
G1T_ARCHITECTURE = "lingbot_task_query_object_value_read_v1"
G1T_COMPARISON_ID = "lingbot-vla2-ltop-g1t-training-abi-parity"
G1T_SCHEMA = "picf-next.ltop-g1t-training-abi-parity.v1"
G1T_PARALLEL_CONTRACT = {
    "backend": "cpu:gloo,cuda:nccl",
    "dp_size": G1T_WORLD_SIZE,
    "dp_replicate_size": 1,
    "dp_shard_size": G1T_WORLD_SIZE,
    "tp_size": 1,
    "ep_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "ulysses_size": 1,
    "dp_mode": "fsdp2",
}
G1T_TRAINING_CONTRACT = {
    "attention_implementation": "eager",
    "vit_attn_implementation": "eager",
    "use_cache": False,
    "use_compile": False,
    "gradient_checkpointing": True,
    "alignment_losses": False,
    "serial_graphs": True,
    "gradient_dot_product": "polarization_identity",
}
G1T_GRADIENT_FAMILIES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("action_output", ("action_out_proj",)),
    (
        "action_conditioning",
        ("state_proj", "action_in_proj", "action_time_mlp_in", "action_time_mlp_out"),
    ),
    ("action_expert", ("qwenvl_with_expert.qwen_expert.",)),
    ("vision_language", ("qwenvl_with_expert.qwenvl.",)),
    ("picf_graph", ("picf_native_graph.",)),
)
G1T_REQUIRED_PARITY_FAMILIES = (
    "action_output",
    "action_conditioning",
    "action_expert",
    "vision_language",
)
G1T_GRAPH_FAMILY = "picf_graph"
G1T_DEFAULT_THRESHOLDS = {
    "loss_abs_max": 1.0e-6,
    "loss_rel_max": 1.0e-5,
    "velocity_max_abs": 1.0e-5,
    "velocity_mean_abs": 1.0e-6,
    "gradient_cosine_min": 0.9999,
    "gradient_norm_rel_max": 1.0e-3,
    "gradient_residual_rel_max": 2.0e-3,
    "graph_gradient_norm_max": 0.0,
}
_MODEL_INPUT_FIELDS = frozenset(
    {
        "actions",
        "image_grid_thw",
        "images",
        "img_masks",
        "lang_masks",
        "lang_tokens",
        "noise",
        "state",
        "time",
    }
)


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_NATIVE_SOURCE") or (
        root / CHECKOUT_RELATIVE_PATH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=root / "configs/lingbot/calvin_robot.yaml",
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=root / "configs/lingbot/calvin_data.json",
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--capacity", type=int, default=G1T_PHYSICAL_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G1T_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--maximum-peak-reserved-gib", type=float, default=39.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
    parser.add_argument(
        "--cuda-allocator",
        choices=CUDA_ALLOCATOR_MODES,
        default="expandable-segments",
    )
    for name, default in G1T_DEFAULT_THRESHOLDS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=float, default=default)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _thresholds_from_args(args: argparse.Namespace) -> dict[str, float]:
    return {name: float(getattr(args, name)) for name in G1T_DEFAULT_THRESHOLDS}


def _validate_args(args: argparse.Namespace) -> None:
    validate_fsdp2_placement(args.fsdp2_placement)
    if args.cuda_allocator not in CUDA_ALLOCATOR_MODES:
        raise ValueError("G1T CUDA allocator mode is unsupported")
    required = {
        "source checkout": args.source_checkout,
        "patch": args.patch,
        "training config": args.training_config,
        "robot config": args.robot_config,
        "data config": args.data_config,
        "checkpoint": args.checkpoint_dir,
        "processor": args.processor_dir,
        "dataset split": args.dataset_split,
        "dataset manifest": args.dataset_manifest,
        "normalization": args.norm_stats,
    }
    missing = [name for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"G1T required paths are absent: {missing}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if not args.output.parent.is_dir():
        raise FileNotFoundError(args.output.parent)
    for name in ("seed", "capacity", "task_query_count", "maximum_control_tokens"):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"G1T {name} must be a positive integer")
    if args.capacity != G1T_PHYSICAL_CAPACITY:
        raise ValueError("G1T must preserve the 16-row LTOP capacity")
    if args.task_query_count != G1T_TASK_QUERY_COUNT:
        raise ValueError("G1T must preserve four task-query rows")
    if args.seed > 0xFFFFFFFF - (G1T_WORLD_SIZE - 1):
        raise ValueError("G1T rank seeds must fit NumPy's uint32 domain")
    if not math.isfinite(args.maximum_peak_reserved_gib) or args.maximum_peak_reserved_gib <= 0:
        raise ValueError("G1T peak-memory limit must be finite and positive")
    thresholds = _thresholds_from_args(args)
    for name, value in thresholds.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"G1T threshold {name} must be finite and non-negative")
    if thresholds["gradient_cosine_min"] > 1.0:
        raise ValueError("G1T gradient cosine threshold cannot exceed one")


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_manifest(values: Mapping[str, Any]) -> tuple[dict[str, str], str]:
    manifest = {name: _tensor_sha256(values[name]) for name in sorted(values)}
    return manifest, _canonical_json_sha256(manifest)


def _episode_ids(episode_keys: tuple[str, ...], *, torch_module: Any, device: Any) -> Any:
    values = [
        int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    {"comparison_id": G1T_COMPARISON_ID, "episode_key": key},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).digest()[:8],
            "big",
        )
        >> 1
        for key in episode_keys
    ]
    return torch_module.tensor(values, dtype=torch_module.long, device=device)


def _family_for_parameter(name: str) -> str | None:
    matches = [
        family
        for family, fragments in G1T_GRADIENT_FAMILIES
        if any(fragment in name for fragment in fragments)
    ]
    if len(matches) > 1:
        raise ValueError(f"G1T parameter matched multiple gradient families: {name}: {matches}")
    return None if not matches else matches[0]


def _local_tensor(value: Any) -> Any:
    to_local = getattr(value, "to_local", None)
    return to_local() if callable(to_local) else value


def _distributed_gradient_squares(
    model: Any,
    *,
    torch_module: Any,
    dist_module: Any,
    device: Any,
) -> dict[str, dict[str, float | int | bool]]:
    """Return exact global family gradient squares for the current ``.grad`` state."""

    family_names = [name for name, _fragments in G1T_GRADIENT_FAMILIES]
    local = {
        name: {
            "parameter_elements": 0,
            "gradient_elements": 0,
            "gradient_tensors": 0,
            "square": 0.0,
            "finite": True,
        }
        for name in family_names
    }
    device_squares = {
        name: torch_module.zeros((), dtype=torch_module.float64, device=device)
        for name in family_names
    }
    device_counts = {
        name: torch_module.zeros(3, dtype=torch_module.float64, device=device)
        for name in family_names
    }
    device_finite = {
        name: torch_module.ones((), dtype=torch_module.int32, device=device)
        for name in family_names
    }
    for parameter_name, parameter in model.named_parameters():
        family = _family_for_parameter(parameter_name)
        if family is None or not parameter.requires_grad:
            continue
        parameter_local = _local_tensor(parameter)
        device_counts[family][0] += int(parameter_local.numel())
        gradient = parameter.grad
        if gradient is None:
            continue
        gradient_local = _local_tensor(gradient)
        square = gradient_local.detach().float().square().sum().to(
            device=device,
            dtype=torch_module.float64,
        )
        device_squares[family].add_(square)
        device_counts[family][1] += int(gradient_local.numel())
        device_counts[family][2] += 1
        finite = torch_module.isfinite(gradient_local).all().to(
            device=device,
            dtype=torch_module.int32,
        )
        device_finite[family].mul_(finite)

    packed = []
    for family in family_names:
        packed.extend(
            [
                device_squares[family],
                *device_counts[family].unbind(),
                device_finite[family].to(dtype=torch_module.float64),
            ]
        )
    reduced = torch_module.stack(packed)
    dist_module.all_reduce(reduced, op=dist_module.ReduceOp.SUM)
    values = reduced.detach().cpu().tolist()
    width = 5
    for index, family in enumerate(family_names):
        square, parameter_elements, gradient_elements, gradient_tensors, finite_ranks = values[
            width * index : width * (index + 1)
        ]
        local[family] = {
            "parameter_elements": int(parameter_elements),
            "gradient_elements": int(gradient_elements),
            "gradient_tensors": int(gradient_tensors),
            "square": float(square),
            "finite": int(finite_ranks) == dist_module.get_world_size(),
        }
    return local


def _gradient_comparison_from_squares(
    *,
    baseline_square: float,
    accumulated_square: float,
    blocked_square: float,
) -> dict[str, float | bool]:
    """Recover exact cosine and residual from three serial gradient norms."""

    values = (baseline_square, accumulated_square, blocked_square)
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("G1T gradient squares must be finite and non-negative")
    baseline_norm = math.sqrt(baseline_square)
    blocked_norm = math.sqrt(blocked_square)
    dot = 0.5 * (accumulated_square - baseline_square - blocked_square)
    rounding_scale = max(baseline_square + blocked_square + accumulated_square, 1.0)
    residual_square = 2.0 * baseline_square + 2.0 * blocked_square - accumulated_square
    if residual_square < 0 and abs(residual_square) <= 1.0e-10 * rounding_scale:
        residual_square = 0.0
    denominator = baseline_norm * blocked_norm
    cosine = 1.0 if baseline_norm == blocked_norm == 0 else (
        float("nan") if denominator == 0 else dot / denominator
    )
    norm_rel_error = abs(blocked_norm - baseline_norm) / max(baseline_norm, 1.0e-30)
    residual_rel = (
        math.sqrt(max(residual_square, 0.0)) / max(baseline_norm, 1.0e-30)
    )
    return {
        "baseline_square": baseline_square,
        "blocked_square": blocked_square,
        "accumulated_square": accumulated_square,
        "baseline_norm": baseline_norm,
        "blocked_norm": blocked_norm,
        "accumulated_norm": math.sqrt(accumulated_square),
        "dot": dot,
        "cosine": cosine,
        "norm_relative_error": norm_rel_error,
        "residual_square": residual_square,
        "residual_relative_norm": residual_rel,
        "both_zero": baseline_norm == blocked_norm == 0,
    }


def _loss_comparison(baseline: float, blocked: float) -> dict[str, float]:
    absolute = abs(blocked - baseline)
    return {
        "baseline": baseline,
        "blocked": blocked,
        "absolute_error": absolute,
        "relative_error": absolute / max(abs(baseline), 1.0e-30),
    }


def _tensor_summary(tensor: Any, *, torch_module: Any) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu")
    floating = value.float()
    return {
        "sha256": _tensor_sha256(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite": bool(torch_module.isfinite(floating).all().item()),
        "mean": float(floating.mean().item()),
        "std": float(floating.std(unbiased=False).item()),
        "l2_norm": float(floating.square().sum().sqrt().item()),
        "max_abs": float(floating.abs().max().item()),
    }


def _detached_scalar(value: Any, *, torch_module: Any, name: str) -> float:
    if torch_module.is_tensor(value):
        if value.numel() != 1:
            raise RuntimeError(f"G1T {name} must be scalar")
        result = float(value.detach().float().item())
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        raise TypeError(f"G1T {name} must be numeric")
    if not math.isfinite(result):
        raise RuntimeError(f"G1T {name} must be finite")
    return result


def _moe_metric_summary(metrics: Any, *, torch_module: Any) -> dict[str, Any]:
    if not isinstance(metrics, dict):
        raise TypeError("G1T released MoE metrics must be a dictionary")
    scalars: dict[str, float] = {}
    structured: dict[str, str] = {}
    for name in sorted(metrics):
        value = metrics[name]
        if torch_module.is_tensor(value) and value.numel() == 1:
            scalars[name] = float(value.detach().float().item())
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            scalars[name] = float(value)
        else:
            structured[name] = _canonical_json_sha256(str(value))
    return {
        "scalars": scalars,
        "structured_sha256": structured,
        "summary_sha256": _canonical_json_sha256(
            {"scalars": scalars, "structured_sha256": structured}
        ),
    }


class _OutputProjectionTrace:
    """Capture the released ``action_out_proj`` velocity without changing forward."""

    def __init__(self, module: Any) -> None:
        self._outputs: list[Any] = []
        self._handle = module.register_forward_hook(self._capture)

    def _capture(self, _module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
        self._outputs.append(output.detach().to(device="cpu"))

    def finish(self) -> Any:
        self._handle.remove()
        if len(self._outputs) != 1:
            raise RuntimeError(
                f"G1T expected one action_out_proj call, observed {len(self._outputs)}"
            )
        return self._outputs[0]


def _unique_named_module(model: Any, suffix: str) -> Any:
    matches = [module for name, module in model.named_modules() if name.endswith(suffix)]
    if len(matches) != 1:
        raise RuntimeError(f"G1T expected one module ending in {suffix!r}, found {len(matches)}")
    return matches[0]


def _released_action_moe_blocks(policy: Any) -> list[Any]:
    root = getattr(policy, "model", None)
    host = getattr(root, "qwenvl_with_expert", None)
    expert = getattr(host, "qwen_expert", None)
    layers = getattr(getattr(expert, "model", None), "layers", None)
    if layers is None:
        raise RuntimeError("G1T could not locate the released action expert layers")
    blocks = [layer.mlp for layer in layers]
    if not blocks:
        raise RuntimeError("G1T released action expert has no MoE blocks")
    return blocks


def _release_graph(torch_module: Any, device: Any) -> None:
    gc.collect()
    torch_module.cuda.empty_cache()
    torch_module.cuda.synchronize(device)


def _context_cache_audit(context: Any, *, torch_module: Any) -> dict[str, Any]:
    visible = context.expanded_action_cache_visible
    native_valid = context.native_valid
    if visible is None or native_valid is None:
        raise RuntimeError("G1T BLOCKED context omitted action-cache metadata")
    native_width = native_valid.shape[1]
    inserted_visible = visible[:, native_width:]
    return {
        "context_finalized": bool(context._finalized),
        "native_width": int(native_width),
        "expanded_width": int(visible.shape[1]),
        "inserted_rows": int(visible.shape[1] - native_width),
        "all_inserted_action_cache_edges_blocked": bool(
            not inserted_visible.any().item()
        ),
        "expanded_action_cache_visible_sha256": _tensor_sha256(visible),
        "native_valid_sha256": _tensor_sha256(native_valid),
        "finite": bool(torch_module.isfinite(visible.float()).all().item()),
    }


def _computed_failures(report: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    thresholds = report["thresholds"]
    rank_reports = report["rank_reports"]
    for rank_report in rank_reports:
        rank = rank_report["rank"]
        for loss_name in ("action_loss", "total_loss"):
            comparison = rank_report[loss_name]
            if comparison["absolute_error"] > thresholds["loss_abs_max"] and (
                comparison["relative_error"] > thresholds["loss_rel_max"]
            ):
                failures.append(f"rank {rank}: {loss_name} parity failed")
        for loss_name in (
            "official_moe_regularizer",
            "sequence_wise_moe_loss",
            "router_z_loss",
        ):
            comparison = rank_report["moe_auxiliary"][loss_name]
            if comparison["absolute_error"] > thresholds["loss_abs_max"] and (
                comparison["relative_error"] > thresholds["loss_rel_max"]
            ):
                failures.append(f"rank {rank}: {loss_name} parity failed")
        velocity = rank_report["velocity"]
        if velocity["max_abs_error"] > thresholds["velocity_max_abs"]:
            failures.append(f"rank {rank}: velocity max-absolute parity failed")
        if velocity["mean_abs_error"] > thresholds["velocity_mean_abs"]:
            failures.append(f"rank {rank}: velocity mean-absolute parity failed")
        if not rank_report["routes_equal"]:
            failures.append(f"rank {rank}: action MoE routes differ")
        if not rank_report["blocked_repeat_equal"]:
            failures.append(f"rank {rank}: BLOCKED replay is nondeterministic")
        cache = rank_report["blocked_cache"]
        if not cache["context_finalized"]:
            failures.append(f"rank {rank}: BLOCKED context did not finalize")
        if not cache["all_inserted_action_cache_edges_blocked"]:
            failures.append(f"rank {rank}: BLOCKED exposed inserted rows to action")
        if not rank_report["rng_restoration_equal"]:
            failures.append(f"rank {rank}: fixed RNG restoration failed")
        if rank_report["peak_reserved_gib"] > report["maximum_peak_reserved_gib"]:
            failures.append(f"rank {rank}: peak reserved CUDA memory exceeded limit")

    gradients = report["gradient_comparisons"]
    for family in G1T_REQUIRED_PARITY_FAMILIES:
        comparison = gradients[family]
        if comparison["both_zero"]:
            failures.append(f"gradient family {family}: both paths are zero")
            continue
        if not comparison["finite"]:
            failures.append(f"gradient family {family}: non-finite gradient")
        if comparison["cosine"] < thresholds["gradient_cosine_min"]:
            failures.append(f"gradient family {family}: cosine below threshold")
        if comparison["norm_relative_error"] > thresholds["gradient_norm_rel_max"]:
            failures.append(f"gradient family {family}: norm relative error above threshold")
        if comparison["residual_relative_norm"] > thresholds["gradient_residual_rel_max"]:
            failures.append(f"gradient family {family}: residual relative norm above threshold")
    graph = gradients[G1T_GRAPH_FAMILY]
    if graph["parameter_elements"] <= 0:
        failures.append("PICF graph family contains no trainable parameters")
    if not graph["finite"]:
        failures.append("PICF graph BLOCKED action gradient is non-finite")
    if graph["blocked_norm"] > thresholds["graph_gradient_norm_max"]:
        failures.append("PICF graph BLOCKED total-objective gradient is non-zero")
    if not graph["action_only_finite"]:
        failures.append("PICF graph BLOCKED action-only gradient is non-finite")
    if graph["action_only_blocked_norm"] > thresholds["graph_gradient_norm_max"]:
        failures.append("PICF graph BLOCKED action-only gradient is non-zero")
    return failures


def validate_ltop_g1t_report(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("G1T report must be a dictionary")
    report = cast(dict[str, Any], value)
    required = {
        "schema",
        "status",
        "failures",
        "source_commit",
        "patch_sha256",
        "patched_source_sha256",
        "source_diff_sha256",
        "checkpoint_revision",
        "processor_revision",
        "implementation_sha256",
        "architecture_identity",
        "world_size",
        "seed",
        "capacity",
        "task_query_count",
        "training_contract",
        "parallel_contract",
        "fsdp2_placement",
        "cuda_allocator",
        "maximum_peak_reserved_gib",
        "thresholds",
        "dataset_contract",
        "config_sha256",
        "gradient_families",
        "gradient_comparisons",
        "parameter_manifest",
        "alignment_teacher_prune",
        "moe_backend",
        "rank_reports",
    }
    if set(report) != required:
        raise ValueError("G1T report fields differ from the frozen schema")
    if report["schema"] != G1T_SCHEMA:
        raise ValueError("G1T report schema differs")
    if report["architecture_identity"] != G1T_ARCHITECTURE:
        raise ValueError("G1T architecture differs from LTOP")
    if report["world_size"] != G1T_WORLD_SIZE:
        raise ValueError("G1T world size differs from two-GPU FSDP")
    if report["capacity"] != G1T_PHYSICAL_CAPACITY:
        raise ValueError("G1T physical capacity differs")
    if report["task_query_count"] != G1T_TASK_QUERY_COUNT:
        raise ValueError("G1T task-query capacity differs")
    if report["training_contract"] != G1T_TRAINING_CONTRACT:
        raise ValueError("G1T training contract differs")
    if report["parallel_contract"] != G1T_PARALLEL_CONTRACT:
        raise ValueError("G1T parallel contract differs")
    validate_fsdp2_placement(report["fsdp2_placement"])
    if report["cuda_allocator"] not in CUDA_ALLOCATOR_MODES:
        raise ValueError("G1T CUDA allocator differs")
    if report["gradient_families"] != {
        name: list(fragments) for name, fragments in G1T_GRADIENT_FAMILIES
    }:
        raise ValueError("G1T gradient family contract differs")
    thresholds = report["thresholds"]
    if not isinstance(thresholds, dict) or set(thresholds) != set(G1T_DEFAULT_THRESHOLDS):
        raise ValueError("G1T threshold fields differ")
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) < 0
        for value in thresholds.values()
    ):
        raise ValueError("G1T thresholds are malformed")
    rank_reports = report["rank_reports"]
    if not isinstance(rank_reports, list) or len(rank_reports) != G1T_WORLD_SIZE:
        raise ValueError("G1T requires one report per rank")
    if sorted(item["rank"] for item in rank_reports) != list(range(G1T_WORLD_SIZE)):
        raise ValueError("G1T rank reports are incomplete")
    sample_keys = [tuple(item["sample_keys"]) for item in rank_reports]
    if len(set(sample_keys)) != len(sample_keys):
        raise ValueError("G1T reused one CALVIN sample across ranks")
    gradient_comparisons = report["gradient_comparisons"]
    expected_families = {name for name, _fragments in G1T_GRADIENT_FAMILIES}
    if not isinstance(gradient_comparisons, dict) or set(gradient_comparisons) != expected_families:
        raise ValueError("G1T gradient comparisons differ from the family contract")
    for family, comparison in gradient_comparisons.items():
        recomputed = _gradient_comparison_from_squares(
            baseline_square=float(comparison["baseline_square"]),
            accumulated_square=float(comparison["accumulated_square"]),
            blocked_square=float(comparison["blocked_square"]),
        )
        for name, expected in recomputed.items():
            actual = comparison[name]
            if isinstance(expected, float) and math.isnan(expected):
                if not isinstance(actual, float) or not math.isnan(actual):
                    raise ValueError(f"G1T gradient family {family} has forged {name}")
            elif actual != expected:
                raise ValueError(f"G1T gradient family {family} has forged {name}")
    graph_comparison = gradient_comparisons[G1T_GRAPH_FAMILY]
    action_only_square = graph_comparison.get("action_only_blocked_square")
    action_only_norm = graph_comparison.get("action_only_blocked_norm")
    if (
        not isinstance(action_only_square, (int, float))
        or isinstance(action_only_square, bool)
        or not math.isfinite(float(action_only_square))
        or float(action_only_square) < 0
        or action_only_norm != math.sqrt(float(action_only_square))
    ):
        raise ValueError("G1T PICF graph action-only gradient evidence is malformed")
    expected_failures = _computed_failures(report)
    if report["failures"] != expected_failures:
        raise ValueError("G1T failures differ from recomputed evidence")
    expected_status = "PASS" if not expected_failures else "FAIL"
    if report["status"] != expected_status:
        raise ValueError("G1T status differs from recomputed evidence")
    return report


def _build_gradient_comparisons(
    baseline: Mapping[str, Mapping[str, float | int | bool]],
    accumulated: Mapping[str, Mapping[str, float | int | bool]],
    blocked: Mapping[str, Mapping[str, float | int | bool]],
) -> dict[str, dict[str, float | int | bool]]:
    result: dict[str, dict[str, float | int | bool]] = {}
    for family, _fragments in G1T_GRADIENT_FAMILIES:
        algebra = _gradient_comparison_from_squares(
            baseline_square=float(baseline[family]["square"]),
            accumulated_square=float(accumulated[family]["square"]),
            blocked_square=float(blocked[family]["square"]),
        )
        result[family] = {
            **algebra,
            "parameter_elements": int(blocked[family]["parameter_elements"]),
            "baseline_gradient_elements": int(baseline[family]["gradient_elements"]),
            "blocked_gradient_elements": int(blocked[family]["gradient_elements"]),
            "finite": bool(
                baseline[family]["finite"]
                and accumulated[family]["finite"]
                and blocked[family]["finite"]
            ),
        }
    return result


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("G1T CUDA allocator pre-bootstrap differs from parsed arguments")
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    prepared_source = validate_prepared_native_source(
        checkout=args.source_checkout,
        patch_path=args.patch,
    )
    expected_hashes = patch_report.get("patched_source_sha256")
    actual_hashes = prepared_source.get("patched_source_sha256")
    if not isinstance(expected_hashes, dict) or actual_hashes != expected_hashes:
        raise RuntimeError("G1T LingBot source differs from immutable patch replay")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    if os.environ.get("WORLD_SIZE") != str(G1T_WORLD_SIZE):
        raise RuntimeError("G1T requires torchrun with exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(G1T_WORLD_SIZE):
        raise RuntimeError("G1T requires both processes on one two-GPU host")

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
    import torch.distributed as dist

    from picf_next.lingbot_native.torch_dcp_compat import (
        install_torch_2_8_sparse_optimizer_state_backport,
    )

    install_torch_2_8_sparse_optimizer_state_backport(torch)

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
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import (
        apply_lingbot_qwen2_patch,
    )
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import (
        apply_lingbot_qwen3_vl_patch,
    )
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
    from lingbotvla.ops import fused_moe
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

    from picf_next.data.calvin import CalvinDatasetIndex, CalvinStatefulTransitionDataset
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.calvin import (
        CollatedNativeCALVINBatch,
        build_native_calvin_stream_plan,
        build_planned_native_calvin_batch,
        collate_native_calvin_training_batch,
        materialize_native_flow_randomness,
    )
    from picf_next.lingbot_native.host import (
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        ObjectReadActionIntervention,
        install_lingbot_native_graph,
        native_context_from_prior_trace,
    )
    from picf_next.lingbot_native.state import AddressedLayerwisePriorTrace
    from picf_next.lingbot_native.training import (
        run_native_policy_training_forward,
        run_official_policy_training_forward,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != G1T_WORLD_SIZE:
            raise RuntimeError("G1T process sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("G1T requires two A100 devices with at least 39 GiB each")
        conflict = torch.tensor(
            int(args.output.exists() or args.output.is_symlink()),
            dtype=torch.int32,
            device=device,
        )
        dist.all_reduce(conflict, op=dist.ReduceOp.MAX)
        if bool(conflict.item()):
            raise FileExistsError(args.output)

        dataset_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
        norm_stats = json.loads(args.norm_stats.read_text())
        validate_lingbot_calvin_norm_stats(norm_stats)
        source = norm_stats["source"]
        if (
            source["dataset_id"] != dataset_manifest.dataset_id
            or source["dataset_revision"] != dataset_manifest.dataset_revision
            or source["dataset_tree_sha256"] != dataset_manifest.tree_sha256
            or dataset_manifest.split_name != args.dataset_split.name
        ):
            raise ValueError("G1T CALVIN manifest and normalization differ")
        dataset_contract = {
            "status": "PASS",
            "manifest_sha256": _sha256(args.dataset_manifest),
            "normalization_sha256": _sha256(args.norm_stats),
            "validation": validate_dataset_runtime_binding(
                dataset_manifest,
                args.dataset_split,
                dataset_id=source["dataset_id"],
                dataset_revision=source["dataset_revision"],
                split_name=args.dataset_split.name,
            ),
        }

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        init_parallel_state(
            dp_size=G1T_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=G1T_WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )

        training = load_lingbot_training_config(args.training_config)
        train_section = training.get("train")
        if not isinstance(train_section, dict):
            raise ValueError("G1T LingBot training config omits its train mapping")
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(train_section.get("lr", 5.0e-5)),
        )
        merged, _data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=2,
        )
        merged.update(
            {
                "use_cache": False,
                "use_compile": False,
                "attention_implementation": "eager",
                "vit_attn_implementation": "eager",
            }
        )
        config_sha256 = _canonical_json_sha256(merged)
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())

        timings: dict[str, float] = {}
        load_started = time.perf_counter()
        processor = build_processor(str(args.processor_dir.resolve()))
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(config=config, eval=False).to(torch.float32)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        policy.train()
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            task_query_count=args.task_query_count,
            architecture_identity=G1T_ARCHITECTURE,
        )
        graph = LingBotNativeGraph(graph_config, device=device, dtype=torch.float32).train()
        install_lingbot_native_graph(policy, graph)
        full_cpu_offload = args.fsdp2_placement == FSDP2_CPU_OFFLOAD
        selective_embedding_offload = (
            args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
        )
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=optimizer_contract.enable_fp32,
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=full_cpu_offload,
            enable_shared_embedding_offload=selective_embedding_offload,
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
        fsdp2_storage = _validate_fsdp2_parameter_storage(
            policy,
            torch,
            expected_placement=args.fsdp2_placement,
        )
        timings["load_and_shard_model_s"] = time.perf_counter() - load_started

        moe_backend = select_lingbot_deterministic_moe_backend(
            action_expert_module=qwen2_action_expert,
            fused_moe_module=fused_moe,
        )
        action_blocks = _released_action_moe_blocks(policy)
        action_out_proj = _unique_named_module(policy, "action_out_proj")

        rank_seed = args.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=dataset_manifest,
        )
        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        plan = build_native_calvin_stream_plan(
            dataset,
            comparison_id=G1T_COMPARISON_ID,
            seed=args.seed,
            global_batch_size=G1T_WORLD_SIZE,
            total_steps=1,
        )
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            official_lingbot_data_config(json.loads(args.data_config.read_text())),
            config,
            processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )
        planned = build_planned_native_calvin_batch(
            plan,
            dataset,
            optimizer_step=0,
            rank=rank,
            world_size=G1T_WORLD_SIZE,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            device=device,
            dtype=torch.bfloat16,
        )
        collated = collate_native_calvin_training_batch(
            planned.training,
            feature_transform=feature_transform,
            collator=VLADataCollatorWithPacking(),
            augmentation_seeds=planned.augmentation_seeds,
            source_digest=planned.source_digest,
        )
        collated = CollatedNativeCALVINBatch(
            model_inputs=_move_model_inputs(
                collated.model_inputs,
                device=device,
                dtype=torch.bfloat16,
                torch_module=torch,
            ),
            controls=collated.controls,
            routing=collated.routing,
            source_digest=collated.source_digest,
            structural_target_requests=collated.structural_target_requests,
            modalities=(
                None
                if collated.modalities is None
                else collated.modalities.to(device=device, dtype=torch.bfloat16)
            ),
            prior_control_chunks=collated.prior_control_chunks,
        )
        collated = materialize_native_flow_randomness(collated, planned)
        if not _MODEL_INPUT_FIELDS.issubset(collated.model_inputs):
            raise RuntimeError("G1T model inputs omitted explicit action/noise/time fields")
        model_input_tensors, model_input_sha256 = _tensor_manifest(
            {name: collated.model_inputs[name] for name in sorted(_MODEL_INPUT_FIELDS)}
        )

        episode_ids = _episode_ids(
            collated.routing.episode_keys,
            torch_module=torch,
            device=device,
        )
        prior_stepper = LingBotNativePriorStepper(policy, graph)
        prior: Any | None = None
        previous_memory_valid = torch.zeros(
            collated.routing.batch_size,
            dtype=torch.bool,
            device=device,
        )
        prior_started = time.perf_counter()
        with torch.no_grad():
            for controls in collated.effective_prior_control_chunks:
                prior = prior_stepper(
                    prior,
                    controls,
                    previous_memory_valid=previous_memory_valid,
                    episode_ids=episode_ids,
                )
                previous_memory_valid = torch.ones_like(previous_memory_valid)
        if not isinstance(prior, AddressedLayerwisePriorTrace):
            raise RuntimeError("G1T prior rollout omitted its addressed trace")
        timings["detached_prior_rollout_s"] = time.perf_counter() - prior_started

        def blocked_context() -> Any:
            return native_context_from_prior_trace(
                controls=collated.controls,
                prior_trace=prior,
                modalities=collated.modalities,
                object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
            )

        fixed_rng = _capture_rank_rng(torch, np, device=device)
        fixed_rng_sha256 = _rank_rng_digest(fixed_rng)

        def execute_pass(
            *,
            name: str,
            native: bool,
            clear_gradients: bool,
            capture_observables: bool,
            backward_root: str = "total",
        ) -> tuple[dict[str, Any], Any | None, dict[str, Any] | None]:
            if backward_root not in {"total", "action"}:
                raise ValueError("G1T backward root must be total or action")
            if clear_gradients:
                policy.zero_grad(set_to_none=True)
            _restore_rank_rng(fixed_rng, torch, np, device=device)
            restored_sha256 = _rank_rng_digest(_capture_rank_rng(torch, np, device=device))
            if restored_sha256 != fixed_rng_sha256:
                raise RuntimeError(f"G1T {name} failed to restore its fixed RNG")
            context = blocked_context() if native else None
            route_trace = _RouteTrace(torch, action_blocks) if capture_observables else None
            output_trace = _OutputProjectionTrace(action_out_proj) if capture_observables else None
            started = time.perf_counter()
            try:
                result = (
                    run_native_policy_training_forward(
                        policy,
                        model_inputs=collated.model_inputs,
                        context=context,
                    )
                    if native
                    else run_official_policy_training_forward(
                        policy,
                        model_inputs=collated.model_inputs,
                    )
                )
                total_loss = result.official_total_loss
                action_loss = result.official_action_loss
                official_outputs = result.official_outputs
                sequence_wise_moe_loss = _detached_scalar(
                    official_outputs[5],
                    torch_module=torch,
                    name="sequence-wise MoE loss",
                )
                official_metrics = official_outputs[6]
                if (
                    not isinstance(official_metrics, dict)
                    or "router_z_loss" not in official_metrics
                ):
                    raise RuntimeError("G1T official outputs omit router_z_loss")
                router_z_loss = _detached_scalar(
                    official_metrics["router_z_loss"],
                    torch_module=torch,
                    name="router z loss",
                )
                official_moe_regularizer = _detached_scalar(
                    result.official_moe_regularizer,
                    torch_module=torch,
                    name="official MoE regularizer",
                )
                backward_loss = total_loss if backward_root == "total" else action_loss
                backward_loss.backward()
                torch.cuda.synchronize(device)
                gradient_squares = _distributed_gradient_squares(
                    policy,
                    torch_module=torch,
                    dist_module=dist,
                    device=device,
                )
                velocity = None if output_trace is None else output_trace.finish()
                output_trace = None
                routes = None if route_trace is None else route_trace.finish()
                route_trace = None
                cache = None if context is None else _context_cache_audit(
                    context,
                    torch_module=torch,
                )
                record = {
                    "name": name,
                    "backward_root": backward_root,
                    "action_loss": float(action_loss.detach().float().item()),
                    "total_loss": float(total_loss.detach().float().item()),
                    "official_moe_regularizer": official_moe_regularizer,
                    "sequence_wise_moe_loss": sequence_wise_moe_loss,
                    "router_z_loss": router_z_loss,
                    "moe_metrics": _moe_metric_summary(
                        official_metrics,
                        torch_module=torch,
                    ),
                    "rng_sha256": restored_sha256,
                    "duration_s": time.perf_counter() - started,
                    "gradient_squares": gradient_squares,
                }
                return record, velocity, {"routes": routes, "cache": cache}
            finally:
                if output_trace is not None:
                    output_trace._handle.remove()
                if route_trace is not None:
                    route_trace.finish()
                if "result" in locals():
                    del result
                if "total_loss" in locals():
                    del total_loss
                if "action_loss" in locals():
                    del action_loss
                if "backward_loss" in locals():
                    del backward_loss
                if context is not None:
                    del context

        baseline, baseline_velocity, baseline_observables = _distributed_rank_local_call(
            action=lambda: execute_pass(
                name="released",
                native=False,
                clear_gradients=True,
                capture_observables=True,
            ),
            phase="g1t-released-forward-backward",
            rank=rank,
            dist_module=dist,
        )
        _release_graph(torch, device)
        blocked_accumulated, blocked_velocity, blocked_observables = (
            _distributed_rank_local_call(
                action=lambda: execute_pass(
                    name="blocked-accumulated",
                    native=True,
                    clear_gradients=False,
                    capture_observables=True,
                ),
                phase="g1t-blocked-accumulated-forward-backward",
                rank=rank,
                dist_module=dist,
            )
        )
        policy.zero_grad(set_to_none=True)
        _release_graph(torch, device)
        blocked_isolated, blocked_repeat_velocity, blocked_repeat_observables = (
            _distributed_rank_local_call(
                action=lambda: execute_pass(
                    name="blocked-isolated",
                    native=True,
                    clear_gradients=True,
                    capture_observables=True,
                ),
                phase="g1t-blocked-isolated-forward-backward",
                rank=rank,
                dist_module=dist,
            )
        )
        policy.zero_grad(set_to_none=True)
        _release_graph(torch, device)
        blocked_action_only, _unused_velocity, _unused_observables = (
            _distributed_rank_local_call(
                action=lambda: execute_pass(
                    name="blocked-action-only",
                    native=True,
                    clear_gradients=True,
                    capture_observables=False,
                    backward_root="action",
                ),
                phase="g1t-blocked-action-only-backward",
                rank=rank,
                dist_module=dist,
            )
        )
        policy.zero_grad(set_to_none=True)
        _release_graph(torch, device)

        if baseline_velocity is None or blocked_velocity is None or blocked_repeat_velocity is None:
            raise RuntimeError("G1T velocity capture vanished")
        baseline_observables = cast(dict[str, Any], baseline_observables)
        blocked_observables = cast(dict[str, Any], blocked_observables)
        blocked_repeat_observables = cast(dict[str, Any], blocked_repeat_observables)
        velocity_error = (blocked_velocity.float() - baseline_velocity.float()).abs()
        blocked_repeat_error = (
            blocked_repeat_velocity.float() - blocked_velocity.float()
        ).abs()
        gradient_comparisons = _build_gradient_comparisons(
            baseline["gradient_squares"],
            blocked_accumulated["gradient_squares"],
            blocked_isolated["gradient_squares"],
        )
        graph_action_only = blocked_action_only["gradient_squares"][G1T_GRAPH_FAMILY]
        graph_comparison = gradient_comparisons[G1T_GRAPH_FAMILY]
        graph_comparison.update(
            {
                "action_only_blocked_square": float(graph_action_only["square"]),
                "action_only_blocked_norm": math.sqrt(float(graph_action_only["square"])),
                "action_only_gradient_elements": int(
                    graph_action_only["gradient_elements"]
                ),
                "action_only_finite": bool(graph_action_only["finite"]),
            }
        )
        memory = _cuda_memory(torch, device)
        if memory is None:
            raise RuntimeError("G1T CUDA memory accounting vanished")
        peak_reserved_gib = memory["peak_reserved"] / 1024**3
        rank_report = {
            "rank": rank,
            "device_name": properties.name,
            "sample_keys": list(collated.routing.sample_keys),
            "episode_keys": list(collated.routing.episode_keys),
            "episode_ids": episode_ids.detach().to(device="cpu").tolist(),
            "frame_indices": list(collated.routing.frame_indices),
            "source_digest": collated.source_digest,
            "model_input_sha256": model_input_sha256,
            "model_input_tensors": model_input_tensors,
            "noise_sha256": _tensor_sha256(collated.model_inputs["noise"]),
            "time_sha256": _tensor_sha256(collated.model_inputs["time"]),
            "fixed_rng_sha256": fixed_rng_sha256,
            "rng_restoration_equal": bool(
                baseline["rng_sha256"]
                == blocked_accumulated["rng_sha256"]
                == blocked_isolated["rng_sha256"]
                == fixed_rng_sha256
            ),
            "action_loss": _loss_comparison(
                baseline["action_loss"],
                blocked_accumulated["action_loss"],
            ),
            "total_loss": _loss_comparison(
                baseline["total_loss"],
                blocked_accumulated["total_loss"],
            ),
            "moe_auxiliary": {
                name: _loss_comparison(
                    baseline[name],
                    blocked_accumulated[name],
                )
                for name in (
                    "official_moe_regularizer",
                    "sequence_wise_moe_loss",
                    "router_z_loss",
                )
            }
            | {
                "released_metrics": baseline["moe_metrics"],
                "blocked_metrics": blocked_accumulated["moe_metrics"],
                "metrics_equal": bool(
                    baseline["moe_metrics"] == blocked_accumulated["moe_metrics"]
                ),
            },
            "velocity": {
                "released": _tensor_summary(baseline_velocity, torch_module=torch),
                "blocked": _tensor_summary(blocked_velocity, torch_module=torch),
                "max_abs_error": float(velocity_error.max().item()),
                "mean_abs_error": float(velocity_error.mean().item()),
            },
            "released_routes": baseline_observables["routes"],
            "blocked_routes": blocked_observables["routes"],
            "routes_equal": bool(
                baseline_observables["routes"] == blocked_observables["routes"]
            ),
            "blocked_repeat_equal": bool(
                blocked_accumulated["action_loss"] == blocked_isolated["action_loss"]
                and blocked_accumulated["total_loss"] == blocked_isolated["total_loss"]
                and blocked_observables["routes"]
                == blocked_repeat_observables["routes"]
                and float(blocked_repeat_error.max().item()) == 0.0
            ),
            "blocked_repeat_velocity_max_abs_error": float(
                blocked_repeat_error.max().item()
            ),
            "blocked_cache": blocked_observables["cache"],
            "timings": {
                **timings,
                "released_forward_backward_s": baseline["duration_s"],
                "blocked_accumulated_forward_backward_s": blocked_accumulated[
                    "duration_s"
                ],
                "blocked_isolated_forward_backward_s": blocked_isolated["duration_s"],
                "blocked_action_only_forward_backward_s": blocked_action_only[
                    "duration_s"
                ],
            },
            "cuda_memory_bytes": memory,
            "peak_reserved_gib": peak_reserved_gib,
        }
        gathered: list[dict[str, Any] | None] = [None] * G1T_WORLD_SIZE
        dist.all_gather_object(gathered, rank_report)
        outcome: list[object] = [None, None]
        if rank == 0:
            rank_reports = [item for item in gathered if item is not None]
            parameter_manifest = {
                family: {
                    "parameter_elements": gradient_comparisons[family][
                        "parameter_elements"
                    ]
                }
                for family, _fragments in G1T_GRADIENT_FAMILIES
            }
            provisional = {
                "schema": G1T_SCHEMA,
                "status": "PASS",
                "failures": [],
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "patch_sha256": patch_report["patch_sha256"],
                "patched_source_sha256": actual_hashes,
                "source_diff_sha256": hashlib.sha256(
                    _git_output(args.source_checkout, "diff", "--binary").encode()
                ).hexdigest(),
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "processor_revision": QWEN_PROCESSOR_REVISION,
                "implementation_sha256": _implementation_digest(
                    root,
                    entrypoint=Path(__file__),
                ),
                "architecture_identity": G1T_ARCHITECTURE,
                "world_size": G1T_WORLD_SIZE,
                "seed": args.seed,
                "capacity": args.capacity,
                "task_query_count": args.task_query_count,
                "training_contract": G1T_TRAINING_CONTRACT,
                "parallel_contract": G1T_PARALLEL_CONTRACT,
                "fsdp2_placement": args.fsdp2_placement,
                "cuda_allocator": args.cuda_allocator,
                "maximum_peak_reserved_gib": args.maximum_peak_reserved_gib,
                "thresholds": _thresholds_from_args(args),
                "dataset_contract": dataset_contract,
                "config_sha256": config_sha256,
                "gradient_families": {
                    name: list(fragments) for name, fragments in G1T_GRADIENT_FAMILIES
                },
                "gradient_comparisons": gradient_comparisons,
                "parameter_manifest": {
                    "families": parameter_manifest,
                    "fsdp2_storage": fsdp2_storage,
                },
                "alignment_teacher_prune": alignment_teacher_prune,
                "moe_backend": moe_backend,
                "rank_reports": rank_reports,
            }
            provisional["failures"] = _computed_failures(provisional)
            provisional["status"] = "PASS" if not provisional["failures"] else "FAIL"
            try:
                validated = validate_ltop_g1t_report(provisional)
                write_text_durable_exclusive(
                    args.output,
                    json.dumps(validated, indent=2, sort_keys=True) + "\n",
                )
                outcome[0] = validated
            except BaseException as error:
                outcome[1] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(outcome, src=0)
        if outcome[1] is not None:
            raise RuntimeError(f"G1T report publication failed: {outcome[1]}")
        dist.barrier()
        if rank == 0:
            print(json.dumps(outcome[0], indent=2, sort_keys=True))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

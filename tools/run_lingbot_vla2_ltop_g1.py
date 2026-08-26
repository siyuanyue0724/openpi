#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the two-GPU released-action parity gate for the ADR-159 LTOP graph.

Each rank loads one full released LingBot policy, selects one distinct audited
CALVIN sample, and evaluates deterministic fixed-noise calls for:

1. the installed graph with no PICF context (released baseline),
2. an exact repeat of that baseline, and
3. the full LTOP prior/correction path with only OBJECT_READ -> ACTION blocked,
4. an equal-shape blocked path whose prior and control contents are neutralized.

This is a causal evaluation gate, not an optimization run. Accelerator and
upstream imports remain inside ``main`` so CLI and evidence validation are
testable on a CPU-only workstation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from picf_next.artifact_io import write_text_durable_exclusive
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
        PATCHED_SOURCES,
        detect_native_patch_state,
        verify_native_patch,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _RouteTrace,
        _sha256,
        _tensor_sha256,
        load_lingbot_training_config,
        select_lingbot_deterministic_moe_backend,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _implementation_digest,
        _move_model_inputs,
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
        PATCHED_SOURCES,
        detect_native_patch_state,
        verify_native_patch,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        _cuda_memory,
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
        _RouteTrace,
        _sha256,
        _tensor_sha256,
        load_lingbot_training_config,
        select_lingbot_deterministic_moe_backend,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _implementation_digest,
        _move_model_inputs,
    )


G1_WORLD_SIZE = 2
G1_PHYSICAL_CAPACITY = 16
G1_TASK_QUERY_COUNT = 4
G1_DENOISE_STEPS = 10
G1_SCHEMA = "picf-next.ltop-g1-released-action-parity.v1"
G1_COMPARISON_ID = "lingbot-vla2-ltop-g1-released-action-parity"
G1_ARCHITECTURE = "lingbot_task_query_object_value_read_v1"
G1_INFERENCE_CONTRACT = {
    "use_cache": True,
    "use_compile": False,
    "attention_implementation": "eager",
    "vit_attn_implementation": "eager",
}
G1_PARALLEL_CONTRACT = {
    "backend": "cpu:gloo,cuda:nccl",
    "dp_size": G1_WORLD_SIZE,
    "dp_replicate_size": 1,
    "dp_shard_size": G1_WORLD_SIZE,
    "tp_size": 1,
    "ep_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "ulysses_size": 1,
    "dp_mode": "fsdp2",
}
_RUNTIME_MODEL_FIELDS = frozenset(
    {
        "image_grid_thw",
        "images",
        "img_masks",
        "lang_masks",
        "lang_tokens",
        "state",
    }
)
_REPORT_FIELDS = {
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
    "num_steps",
    "inference_contract",
    "parallel_contract",
    "dataset_contract",
    "config_sha256",
    "parameter_manifest",
    "alignment_teacher_prune",
    "moe_inference_backend",
    "rank_reports",
}
_RANK_FIELDS = {
    "rank",
    "device_name",
    "sample_keys",
    "episode_keys",
    "frame_indices",
    "source_digest",
    "model_input_sha256",
    "model_input_tensors",
    "flow_noise_sha256",
    "episode_ids",
    "address_receipt",
    "prior_trace_sha256",
    "baseline_action_sha256",
    "baseline_repeat_action_sha256",
    "blocked_action_sha256",
    "blocked_repeat_action_sha256",
    "neutral_action_sha256",
    "neutral_repeat_action_sha256",
    "blocked_cache_metadata_sha256",
    "neutral_cache_metadata_sha256",
    "baseline_repeat_bitwise_equal",
    "blocked_repeat_bitwise_equal",
    "blocked_vs_baseline_bitwise_equal",
    "neutral_repeat_bitwise_equal",
    "blocked_vs_neutral_bitwise_equal",
    "baseline_repeat_max_abs_error",
    "blocked_repeat_max_abs_error",
    "blocked_vs_baseline_max_abs_error",
    "blocked_vs_baseline_mean_abs_error",
    "neutral_repeat_max_abs_error",
    "blocked_vs_neutral_max_abs_error",
    "blocked_vs_neutral_mean_abs_error",
    "actions_finite",
    "baseline_routes",
    "baseline_repeat_routes",
    "blocked_routes",
    "blocked_repeat_routes",
    "neutral_routes",
    "neutral_repeat_routes",
    "baseline_repeat_routes_equal",
    "blocked_repeat_routes_equal",
    "blocked_vs_baseline_routes_equal",
    "neutral_repeat_routes_equal",
    "blocked_vs_neutral_routes_equal",
    "blocked_neutral_cache_metadata_equal",
    "object_read_action_cache_edge_blocked",
    "all_inserted_action_cache_edges_blocked",
    "context_finalized",
    "timings",
    "cuda_memory_bytes",
}
_REQUIRED_TRUE_RANK_FIELDS = (
    "baseline_repeat_bitwise_equal",
    "blocked_repeat_bitwise_equal",
    "blocked_vs_baseline_bitwise_equal",
    "neutral_repeat_bitwise_equal",
    "blocked_vs_neutral_bitwise_equal",
    "actions_finite",
    "baseline_repeat_routes_equal",
    "blocked_repeat_routes_equal",
    "blocked_vs_baseline_routes_equal",
    "neutral_repeat_routes_equal",
    "blocked_vs_neutral_routes_equal",
    "blocked_neutral_cache_metadata_equal",
    "object_read_action_cache_edge_blocked",
    "all_inserted_action_cache_edges_blocked",
    "context_finalized",
)
_ZERO_ERROR_RANK_FIELDS = (
    "baseline_repeat_max_abs_error",
    "blocked_repeat_max_abs_error",
    "blocked_vs_baseline_max_abs_error",
    "blocked_vs_baseline_mean_abs_error",
    "neutral_repeat_max_abs_error",
    "blocked_vs_neutral_max_abs_error",
    "blocked_vs_neutral_mean_abs_error",
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
    parser.add_argument("--capacity", type=int, default=G1_PHYSICAL_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G1_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=G1_DENOISE_STEPS)
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    return args


def _validate_args(args: argparse.Namespace) -> None:
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
        raise FileNotFoundError(f"LTOP G1 required paths are absent: {missing}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for name in (
        "seed",
        "capacity",
        "task_query_count",
        "maximum_control_tokens",
        "num_steps",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"LTOP G1 {name} must be a positive integer")
    if args.capacity != G1_PHYSICAL_CAPACITY:
        raise ValueError("LTOP G1 must preserve the G0 physical capacity of 16 rows")
    if args.task_query_count != G1_TASK_QUERY_COUNT:
        raise ValueError("LTOP G1 must preserve the G0 relation-read capacity of four")
    if args.num_steps != G1_DENOISE_STEPS:
        raise ValueError("LTOP G1 must preserve the released ten-step action sampler")


def _validated_patched_source_hashes(
    checkout: Path,
    patch_report: dict[str, object],
) -> dict[str, str]:
    expected = patch_report.get("patched_source_sha256")
    accepted_paths = {str(path) for path in PATCHED_SOURCES}
    if not isinstance(expected, dict) or set(expected) != accepted_paths:
        raise RuntimeError("native patch verifier returned the wrong source hash contract")
    actual = {relative: _sha256(checkout / relative) for relative in sorted(accepted_paths)}
    if actual != expected:
        raise RuntimeError("LingBot native source differs from immutable patch replay")
    return actual


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tensor_manifest(values: dict[str, Any]) -> tuple[dict[str, str], str]:
    manifest = {name: _tensor_sha256(values[name]) for name in sorted(values)}
    return manifest, _canonical_json_sha256(manifest)


def _episode_ids(episode_keys: tuple[str, ...], *, torch_module: Any, device: Any) -> Any:
    values = [
        int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    {"comparison_id": G1_COMPARISON_ID, "episode_key": key},
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


def _parameter_manifest(policy: Any) -> dict[str, object]:
    records = []
    total = 0
    trainable = 0
    for name, parameter in policy.named_parameters():
        count = int(parameter.numel())
        total += count
        if parameter.requires_grad:
            trainable += count
        records.append(
            {
                "dtype": str(parameter.dtype),
                "name": name,
                "numel": count,
                "requires_grad": bool(parameter.requires_grad),
                "shape": tuple(parameter.shape),
            }
        )
    return {
        "parameter_count": len(records),
        "total_numel": total,
        "active_trainable_numel": trainable,
        "schema_sha256": _canonical_json_sha256(records),
    }


def _exact_dict(value: object, *, fields: set[str], name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise ValueError(f"{name} differs from the frozen schema")
    return cast(dict[str, Any], value)


def validate_ltop_g1_report(report: object) -> dict[str, Any]:
    """Validate one complete strict G1 report before publication."""

    value = _exact_dict(report, fields=_REPORT_FIELDS, name="LTOP G1 report")
    if value["schema"] != G1_SCHEMA or value["architecture_identity"] != G1_ARCHITECTURE:
        raise ValueError("LTOP G1 report has the wrong architecture identity")
    if value["world_size"] != G1_WORLD_SIZE:
        raise ValueError("LTOP G1 report must contain exactly two ranks")
    if value["num_steps"] != G1_DENOISE_STEPS:
        raise ValueError("LTOP G1 report changed the released denoise schedule")
    if (
        value["capacity"] != G1_PHYSICAL_CAPACITY
        or value["task_query_count"] != G1_TASK_QUERY_COUNT
    ):
        raise ValueError("LTOP G1 report changed the frozen G0 topology")
    if value["inference_contract"] != G1_INFERENCE_CONTRACT:
        raise ValueError("LTOP G1 report changed the released eager inference contract")
    if value["parallel_contract"] != G1_PARALLEL_CONTRACT:
        raise ValueError("LTOP G1 report changed the proven LingBot parallel contract")
    rank_reports = value["rank_reports"]
    if not isinstance(rank_reports, list) or len(rank_reports) != G1_WORLD_SIZE:
        raise ValueError("LTOP G1 report lacks one result per rank")
    observed_ranks: set[int] = set()
    observed_samples: set[str] = set()
    failures: list[str] = []
    for raw in rank_reports:
        rank = _exact_dict(raw, fields=_RANK_FIELDS, name="LTOP G1 rank report")
        rank_index = rank["rank"]
        if (
            isinstance(rank_index, bool)
            or not isinstance(rank_index, int)
            or not 0 <= rank_index < G1_WORLD_SIZE
            or rank_index in observed_ranks
        ):
            raise ValueError("LTOP G1 rank identity is invalid or duplicated")
        observed_ranks.add(rank_index)
        sample_keys = rank["sample_keys"]
        if not isinstance(sample_keys, list) or not sample_keys:
            raise ValueError("LTOP G1 rank report has no sample identity")
        if observed_samples.intersection(sample_keys):
            raise ValueError("LTOP G1 ranks reused one CALVIN sample")
        observed_samples.update(sample_keys)
        for name in (
            "baseline_action_sha256",
            "baseline_repeat_action_sha256",
            "blocked_action_sha256",
            "blocked_repeat_action_sha256",
            "neutral_action_sha256",
            "neutral_repeat_action_sha256",
            "blocked_cache_metadata_sha256",
            "neutral_cache_metadata_sha256",
        ):
            digest = rank[name]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError(f"LTOP G1 rank report has an invalid {name}")
        hash_equalities = {
            "baseline_repeat_bitwise_equal": (
                rank["baseline_action_sha256"]
                == rank["baseline_repeat_action_sha256"]
            ),
            "blocked_repeat_bitwise_equal": (
                rank["blocked_action_sha256"]
                == rank["blocked_repeat_action_sha256"]
            ),
            "blocked_vs_baseline_bitwise_equal": (
                rank["blocked_action_sha256"] == rank["baseline_action_sha256"]
            ),
            "neutral_repeat_bitwise_equal": (
                rank["neutral_action_sha256"]
                == rank["neutral_repeat_action_sha256"]
            ),
            "blocked_vs_neutral_bitwise_equal": (
                rank["blocked_action_sha256"] == rank["neutral_action_sha256"]
            ),
            "blocked_neutral_cache_metadata_equal": (
                rank["blocked_cache_metadata_sha256"]
                == rank["neutral_cache_metadata_sha256"]
            ),
        }
        for name, expected in hash_equalities.items():
            if rank[name] is not expected:
                raise ValueError(f"LTOP G1 rank report contradicts {name}")
        route_equalities = {
            "baseline_repeat_routes_equal": (
                rank["baseline_routes"] == rank["baseline_repeat_routes"]
            ),
            "blocked_repeat_routes_equal": (
                rank["blocked_routes"] == rank["blocked_repeat_routes"]
            ),
            "blocked_vs_baseline_routes_equal": (
                rank["blocked_routes"] == rank["baseline_routes"]
            ),
            "neutral_repeat_routes_equal": (
                rank["neutral_routes"] == rank["neutral_repeat_routes"]
            ),
            "blocked_vs_neutral_routes_equal": (
                rank["blocked_routes"] == rank["neutral_routes"]
            ),
        }
        for name, expected in route_equalities.items():
            if rank[name] is not expected:
                raise ValueError(f"LTOP G1 rank report contradicts {name}")
        failures.extend(
            f"rank {rank_index}: {name} is false"
            for name in _REQUIRED_TRUE_RANK_FIELDS
            if rank[name] is not True
        )
        for name in _ZERO_ERROR_RANK_FIELDS:
            error = rank[name]
            if (
                isinstance(error, bool)
                or not isinstance(error, (int, float))
                or not math.isfinite(error)
                or error < 0.0
            ):
                raise ValueError(f"LTOP G1 rank report has an invalid {name}")
            if error != 0.0:
                failures.append(f"rank {rank_index}: {name}={rank[name]}")
    if observed_ranks != set(range(G1_WORLD_SIZE)):
        raise ValueError("LTOP G1 report omitted a rank")
    manifest = value["parameter_manifest"]
    if not isinstance(manifest, dict) or manifest.get("active_trainable_numel") != 0:
        failures.append("G1 policy had active trainable parameters")
    declared_failures = value["failures"]
    if not isinstance(declared_failures, list) or any(
        not isinstance(item, str) for item in declared_failures
    ):
        raise ValueError("LTOP G1 failures must be a string list")
    if declared_failures != failures:
        raise ValueError("LTOP G1 declared failures differ from recomputed failures")
    expected_status = "PASS" if not failures else "FAIL"
    if value["status"] != expected_status:
        raise ValueError("LTOP G1 status differs from its strict evidence")
    return value


def apply_ltop_g1_inference_contract(config: Any) -> None:
    """Restore the released full-weight action-sampling settings after config init."""

    for name, value in G1_INFERENCE_CONTRACT.items():
        setattr(config, name, value)


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(
        root=root,
        checkout=args.source_checkout,
        check_apply=True,
    )
    if _git_output(args.source_checkout, "rev-parse", "HEAD") != LINGBOT_NATIVE_SOURCE_COMMIT:
        raise RuntimeError("LingBot source checkout differs from the pinned commit")
    if detect_native_patch_state(args.source_checkout, args.patch) != "applied":
        raise RuntimeError("LingBot source patch is not in its exact applied state")
    patched_source_sha256 = _validated_patched_source_hashes(args.source_checkout, patch_report)
    checkpoint_report = validate_checkpoint(args.checkpoint_dir)
    processor_report = validate_processor(args.processor_dir)
    del checkpoint_report, processor_report

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import torch
    import torch.distributed as dist
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
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
    from picf_next.lingbot_native.controls import ExecutedControlBatch
    from picf_next.lingbot_native.graph import NativeRole
    from picf_next.lingbot_native.host import (
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        ObjectReadActionIntervention,
        install_lingbot_native_graph,
        native_context_from_prior_trace,
    )
    from picf_next.lingbot_native.state import AddressedLayerwisePriorTrace

    if int(os.environ.get("WORLD_SIZE", "0")) != G1_WORLD_SIZE:
        raise RuntimeError("LTOP G1 must run under torchrun with exactly two processes")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=G1_PARALLEL_CONTRACT["backend"])
    try:
        if torch.cuda.device_count() != G1_WORLD_SIZE:
            raise RuntimeError("LTOP G1 process sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("LTOP G1 requires two A100 devices with at least 39 GiB each")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
        init_parallel_state(
            **{
                name: value
                for name, value in G1_PARALLEL_CONTRACT.items()
                if name != "backend"
            }
        )

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
            raise ValueError("LTOP G1 CALVIN manifest and normalization differ")
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

        training = load_lingbot_training_config(args.training_config)
        merged, data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=args.num_steps,
        )
        merged.update(G1_INFERENCE_CONTRACT)
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
        apply_ltop_g1_inference_contract(config)
        config.num_steps = args.num_steps

        moe_inference_backend = select_lingbot_deterministic_moe_backend(
            action_expert_module=qwen2_action_expert,
            fused_moe_module=fused_moe,
        )
        timings: dict[str, float] = {}
        started = time.perf_counter()
        processor = build_processor(str(args.processor_dir.resolve()))
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(config=config, eval=True).to(torch.bfloat16)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        policy.eval()
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            task_query_count=args.task_query_count,
            architecture_identity=G1_ARCHITECTURE,
        )
        graph = LingBotNativeGraph(
            graph_config,
            device=device,
            dtype=torch.bfloat16,
        ).eval()
        install_lingbot_native_graph(policy, graph)
        policy.requires_grad_(False)
        parameter_manifest = _parameter_manifest(policy)
        if parameter_manifest["active_trainable_numel"] != 0:
            raise RuntimeError("LTOP G1 failed to freeze every parameter")
        timings["load_model_s"] = time.perf_counter() - started

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
            comparison_id=G1_COMPARISON_ID,
            seed=args.seed,
            global_batch_size=G1_WORLD_SIZE,
            total_steps=1,
        )
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
        planned = build_planned_native_calvin_batch(
            plan,
            dataset,
            optimizer_step=0,
            rank=rank,
            world_size=G1_WORLD_SIZE,
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
            modalities=None,
            prior_control_chunks=collated.prior_control_chunks,
        )
        collated = materialize_native_flow_randomness(collated, planned)
        runtime_inputs = {
            name: collated.model_inputs[name] for name in sorted(_RUNTIME_MODEL_FIELDS)
        }
        model_input_tensors, model_input_sha256 = _tensor_manifest(runtime_inputs)
        flow_noise = collated.model_inputs["noise"]

        action_blocks = [
            layer.mlp for layer in policy.model.qwenvl_with_expert.qwen_expert.model.layers
        ]

        def action_call(context: Any | None) -> tuple[Any, dict[str, Any], float]:
            route_trace = _RouteTrace(torch, action_blocks)
            started_call = time.perf_counter()
            with torch.inference_mode():
                actions = policy.sample_actions(
                    **runtime_inputs,
                    noise=flow_noise.clone(),
                    picf_native_context=context,
                )
            torch.cuda.synchronize(device)
            return actions, route_trace.finish(), time.perf_counter() - started_call

        baseline, baseline_routes, timings["baseline_action_s"] = action_call(None)
        baseline_repeat, baseline_repeat_routes, timings["baseline_repeat_action_s"] = (
            action_call(None)
        )

        episode_ids = _episode_ids(
            collated.routing.episode_keys,
            torch_module=torch,
            device=device,
        )
        prior_stepper = LingBotNativePriorStepper(policy, graph)
        prior: Any | None = None
        prior_valid = torch.zeros(
            collated.routing.batch_size,
            dtype=torch.bool,
            device=device,
        )
        started_prior = time.perf_counter()
        with torch.inference_mode():
            for controls in collated.effective_prior_control_chunks:
                prior = prior_stepper(
                    prior,
                    controls,
                    previous_memory_valid=prior_valid,
                    episode_ids=episode_ids,
                )
                prior_valid = torch.ones_like(prior_valid)
        torch.cuda.synchronize(device)
        timings["prior_rollout_s"] = time.perf_counter() - started_prior
        if not isinstance(prior, AddressedLayerwisePriorTrace):
            raise RuntimeError("LTOP G1 prior stepper omitted its addressed trace")

        def blocked_context() -> Any:
            return native_context_from_prior_trace(
                controls=collated.controls,
                prior_trace=prior,
                modalities=None,
                object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
            )

        neutral_controls = ExecutedControlBatch(
            values=torch.zeros_like(collated.controls.values),
            field_valid=torch.zeros_like(collated.controls.field_valid),
            token_valid=collated.controls.token_valid.clone(),
            delta_time=torch.zeros_like(collated.controls.delta_time),
            reset=torch.zeros_like(collated.controls.reset),
            acknowledged=collated.controls.acknowledged.clone(),
        )
        neutral_prior = AddressedLayerwisePriorTrace(
            layer_rows=torch.zeros_like(prior.layer_rows),
            episode_address_state=prior.episode_address_state,
            architecture_identity=prior.architecture_identity,
        )

        def neutral_context() -> Any:
            return native_context_from_prior_trace(
                controls=neutral_controls,
                prior_trace=neutral_prior,
                modalities=None,
                object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
            )

        blocked_ctx = blocked_context()
        blocked, blocked_routes, timings["blocked_action_s"] = action_call(blocked_ctx)
        blocked_repeat_ctx = blocked_context()
        blocked_repeat, blocked_repeat_routes, timings["blocked_repeat_action_s"] = action_call(
            blocked_repeat_ctx
        )
        neutral_ctx = neutral_context()
        neutral, neutral_routes, timings["neutral_action_s"] = action_call(neutral_ctx)
        neutral_repeat_ctx = neutral_context()
        neutral_repeat, neutral_repeat_routes, timings["neutral_repeat_action_s"] = action_call(
            neutral_repeat_ctx
        )

        def cache_metadata(context: Any) -> tuple[dict[str, str], str]:
            values = {
                "expanded_action_cache_visible": context.expanded_action_cache_visible,
                "expanded_cache_position_ids": context.expanded_cache_position_ids,
                "expanded_cache_valid": context.expanded_cache_valid,
                "native_roles": context.native_roles,
                "native_valid": context.native_valid,
            }
            if any(value is None for value in values.values()):
                raise RuntimeError("LTOP G1 context omitted cache metadata")
            return _tensor_manifest(cast(dict[str, Any], values))

        blocked_cache_tensors, blocked_cache_metadata_sha256 = cache_metadata(blocked_ctx)
        neutral_cache_tensors, neutral_cache_metadata_sha256 = cache_metadata(neutral_ctx)
        blocked_neutral_cache_metadata_equal = (
            blocked_cache_tensors == neutral_cache_tensors
        )

        cache_visible = blocked_ctx.expanded_action_cache_visible
        native_roles = blocked_ctx.native_roles
        if cache_visible is None or native_roles is None:
            raise RuntimeError("LTOP G1 blocked context omitted inference cache metadata")
        language = native_roles == int(NativeRole.LANGUAGE)
        language_indices = torch.nonzero(language[0], as_tuple=False).flatten()
        if not language_indices.numel():
            raise RuntimeError("LTOP G1 blocked context omitted the language span")
        language_span_count = int(language_indices[-1].item() - language_indices[0].item() + 1)
        object_read_start = (
            native_roles.shape[1]
            + language_span_count
            + collated.controls.token_count
            + 2 * args.capacity
            + args.task_query_count
        )
        object_read_slice = slice(
            object_read_start,
            object_read_start + args.task_query_count,
        )
        object_read_action_cache_edge_blocked = bool(
            not cache_visible[:, object_read_slice].any().item()
        )
        all_inserted_action_cache_edges_blocked = bool(
            not cache_visible[:, native_roles.shape[1] :].any().item()
        )

        baseline_error = (baseline_repeat.float() - baseline.float()).abs()
        blocked_repeat_error = (blocked_repeat.float() - blocked.float()).abs()
        parity_error = (blocked.float() - baseline.float()).abs()
        neutral_repeat_error = (neutral_repeat.float() - neutral.float()).abs()
        content_isolation_error = (blocked.float() - neutral.float()).abs()
        actions_finite = bool(
            torch.isfinite(baseline).all()
            and torch.isfinite(baseline_repeat).all()
            and torch.isfinite(blocked).all()
            and torch.isfinite(blocked_repeat).all()
            and torch.isfinite(neutral).all()
            and torch.isfinite(neutral_repeat).all()
        )
        rank_report = {
            "rank": rank,
            "device_name": properties.name,
            "sample_keys": list(collated.routing.sample_keys),
            "episode_keys": list(collated.routing.episode_keys),
            "frame_indices": list(collated.routing.frame_indices),
            "source_digest": collated.source_digest,
            "model_input_sha256": model_input_sha256,
            "model_input_tensors": model_input_tensors,
            "flow_noise_sha256": _tensor_sha256(flow_noise),
            "episode_ids": episode_ids.detach().to(device="cpu").tolist(),
            "address_receipt": prior.address_receipt,
            "prior_trace_sha256": _tensor_sha256(prior.layer_rows),
            "baseline_action_sha256": _tensor_sha256(baseline),
            "baseline_repeat_action_sha256": _tensor_sha256(baseline_repeat),
            "blocked_action_sha256": _tensor_sha256(blocked),
            "blocked_repeat_action_sha256": _tensor_sha256(blocked_repeat),
            "neutral_action_sha256": _tensor_sha256(neutral),
            "neutral_repeat_action_sha256": _tensor_sha256(neutral_repeat),
            "blocked_cache_metadata_sha256": blocked_cache_metadata_sha256,
            "neutral_cache_metadata_sha256": neutral_cache_metadata_sha256,
            "baseline_repeat_bitwise_equal": bool(
                _tensor_sha256(baseline_repeat) == _tensor_sha256(baseline)
            ),
            "blocked_repeat_bitwise_equal": bool(
                _tensor_sha256(blocked_repeat) == _tensor_sha256(blocked)
            ),
            "blocked_vs_baseline_bitwise_equal": bool(
                _tensor_sha256(blocked) == _tensor_sha256(baseline)
            ),
            "neutral_repeat_bitwise_equal": bool(
                _tensor_sha256(neutral_repeat) == _tensor_sha256(neutral)
            ),
            "blocked_vs_neutral_bitwise_equal": bool(
                _tensor_sha256(blocked) == _tensor_sha256(neutral)
            ),
            "baseline_repeat_max_abs_error": float(baseline_error.max().item()),
            "blocked_repeat_max_abs_error": float(blocked_repeat_error.max().item()),
            "blocked_vs_baseline_max_abs_error": float(parity_error.max().item()),
            "blocked_vs_baseline_mean_abs_error": float(parity_error.mean().item()),
            "neutral_repeat_max_abs_error": float(neutral_repeat_error.max().item()),
            "blocked_vs_neutral_max_abs_error": float(content_isolation_error.max().item()),
            "blocked_vs_neutral_mean_abs_error": float(content_isolation_error.mean().item()),
            "actions_finite": actions_finite,
            "baseline_routes": baseline_routes,
            "baseline_repeat_routes": baseline_repeat_routes,
            "blocked_routes": blocked_routes,
            "blocked_repeat_routes": blocked_repeat_routes,
            "neutral_routes": neutral_routes,
            "neutral_repeat_routes": neutral_repeat_routes,
            "baseline_repeat_routes_equal": baseline_repeat_routes == baseline_routes,
            "blocked_repeat_routes_equal": blocked_repeat_routes == blocked_routes,
            "blocked_vs_baseline_routes_equal": blocked_routes == baseline_routes,
            "neutral_repeat_routes_equal": neutral_repeat_routes == neutral_routes,
            "blocked_vs_neutral_routes_equal": blocked_routes == neutral_routes,
            "blocked_neutral_cache_metadata_equal": (
                blocked_neutral_cache_metadata_equal
            ),
            "object_read_action_cache_edge_blocked": object_read_action_cache_edge_blocked,
            "all_inserted_action_cache_edges_blocked": (
                all_inserted_action_cache_edges_blocked
            ),
            "context_finalized": bool(
                blocked_ctx._finalized
                and blocked_repeat_ctx._finalized
                and neutral_ctx._finalized
                and neutral_repeat_ctx._finalized
            ),
            "timings": timings,
            "cuda_memory_bytes": _cuda_memory(torch, device),
        }
        gathered: list[dict[str, Any] | None] = [None] * G1_WORLD_SIZE
        dist.all_gather_object(gathered, rank_report)
        outcome: list[object] = [None, None]
        if rank == 0:
            rank_reports = [item for item in gathered if item is not None]
            provisional = {
                "schema": G1_SCHEMA,
                "status": "PASS",
                "failures": [],
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "patch_sha256": patch_report["patch_sha256"],
                "patched_source_sha256": patched_source_sha256,
                "source_diff_sha256": hashlib.sha256(
                    _git_output(args.source_checkout, "diff", "--binary").encode()
                ).hexdigest(),
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "processor_revision": QWEN_PROCESSOR_REVISION,
                "implementation_sha256": _implementation_digest(
                    root,
                    entrypoint=Path(__file__),
                ),
                "architecture_identity": G1_ARCHITECTURE,
                "world_size": G1_WORLD_SIZE,
                "seed": args.seed,
                "capacity": args.capacity,
                "task_query_count": args.task_query_count,
                "num_steps": config.num_steps,
                "inference_contract": G1_INFERENCE_CONTRACT,
                "parallel_contract": G1_PARALLEL_CONTRACT,
                "dataset_contract": dataset_contract,
                "config_sha256": config_sha256,
                "parameter_manifest": parameter_manifest,
                "alignment_teacher_prune": alignment_teacher_prune,
                "moe_inference_backend": moe_inference_backend,
                "rank_reports": rank_reports,
            }
            recomputed_failures: list[str] = []
            for item in rank_reports:
                for name in _REQUIRED_TRUE_RANK_FIELDS:
                    if item[name] is not True:
                        recomputed_failures.append(f"rank {item['rank']}: {name} is false")
                for name in _ZERO_ERROR_RANK_FIELDS:
                    if item[name] != 0.0:
                        recomputed_failures.append(
                            f"rank {item['rank']}: {name}={item[name]}"
                        )
            if parameter_manifest["active_trainable_numel"] != 0:
                recomputed_failures.append("G1 policy had active trainable parameters")
            provisional["failures"] = recomputed_failures
            provisional["status"] = "PASS" if not recomputed_failures else "FAIL"
            report = validate_ltop_g1_report(provisional)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_text_durable_exclusive(
                args.output,
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            outcome = [report["status"], report["failures"]]
        dist.broadcast_object_list(outcome, src=0)
        dist.barrier()
        if outcome[0] != "PASS":
            raise RuntimeError(f"LTOP G1 rejected: {outcome[1]}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

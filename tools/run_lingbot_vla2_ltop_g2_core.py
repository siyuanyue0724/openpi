#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the two-GPU LTOP G2 task-address learnability gates.

Both gates add no deploy-time module. ``query-only`` freezes the released
LingBot policy and every PICF parameter except the existing TASK_QUERY
embeddings. ``representation`` instead uses the production frozen-action
representation scope under released FSDP2 and optimizer semantics. Each scene
uses one prompt-free Hungarian row gauge per optimizer step; crossed prompts
must reproduce that complete binding before task-address evidence is accepted.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config

try:
    from tools.bootstrap_lingbot_vla2 import (
        LINGBOT_CHECKPOINT_ID,
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
        _merge_qwen_config,
        _resolve_training_config,
        build_lingbot_query_only_optimizer,
        build_lingbot_representation_optimizer,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _distributed_rank_local_call,
        _fsync_tree,
        _model_local_state_digest,
        _move_model_inputs,
    )
except ModuleNotFoundError:  # Direct ``python tools/...`` execution.
    from bootstrap_lingbot_vla2 import (  # type: ignore[no-redef]
        LINGBOT_CHECKPOINT_ID,
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
        _merge_qwen_config,
        _resolve_training_config,
        build_lingbot_query_only_optimizer,
        build_lingbot_representation_optimizer,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _distributed_rank_local_call,
        _fsync_tree,
        _model_local_state_digest,
        _move_model_inputs,
    )


G2_WORLD_SIZE = 2
G2_CAPACITY = 16
G2_TASK_QUERY_COUNT = 4
G2_ARCHITECTURE = "lingbot_task_query_object_value_read_v1"
G2_SCHEMA = "picf-next.ltop-g2-core-query-learnability.v1"
G2_REPRESENTATION_SCHEMA = "picf-next.ltop-g2-shared-representation.v3"
G2_TRAINING_SCOPES = ("query-only", "representation")


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if value is None else Path(value)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-checkout",
        type=Path,
        default=_environment_path("PICF_LINGBOT_NATIVE_SOURCE")
        or root / CHECKOUT_RELATIVE_PATH,
    )
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path, default=None)
    parser.add_argument(
        "--robot-config",
        type=Path,
        default=root / "configs/lingbot/calvin_robot.yaml",
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
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--execution-contract", type=Path, required=True)
    parser.add_argument("--offline-labels", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-checkpoint", type=Path, default=None)
    parser.add_argument("--progress-output", type=Path, default=None)
    parser.add_argument("--progress-every", type=int, default=8)
    parser.add_argument(
        "--training-scope",
        choices=G2_TRAINING_SCOPES,
        default="query-only",
    )
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--capacity", type=int, default=G2_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G2_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--maximum-grad-norm", type=float, default=1.0)
    parser.add_argument("--physical-set-weight", type=float, default=1.0)
    parser.add_argument("--task-address-weight", type=float, default=1.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
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
        "checkpoint": args.checkpoint_dir,
        "processor": args.processor_dir,
        "dataset split": args.dataset_split,
        "dataset manifest": args.dataset_manifest,
        "normalization": args.norm_stats,
        "physical sidecar": args.physical_sidecar_root,
        "physical sidecar manifest": args.physical_sidecar_manifest,
        "execution contract": args.execution_contract,
        "offline labels": args.offline_labels,
    }
    missing = [name for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"LTOP G2 required paths are absent: {missing}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if args.output_checkpoint is not None:
        if args.training_scope != "representation":
            raise ValueError("LTOP G2 checkpoint output requires representation scope")
        staging = args.output_checkpoint.with_name(
            f".{args.output_checkpoint.name}.staging"
        )
        for path in (args.output_checkpoint, staging):
            if path.exists() or path.is_symlink():
                raise FileExistsError(path)
    if args.progress_output is not None and args.progress_output == args.output:
        raise ValueError("LTOP G2 progress and final report paths must differ")
    integer_fields = (
        "steps",
        "eval_every",
        "seed",
        "capacity",
        "task_query_count",
        "maximum_control_tokens",
        "progress_every",
    )
    for name in integer_fields:
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"LTOP G2 {name} must be positive")
    if args.training_scope == "query-only" and (args.steps != 32 or args.eval_every != 8):
        raise ValueError("LTOP G2 deadline gate is fixed to 32 steps and 8-step evaluation")
    if args.steps % args.eval_every:
        raise ValueError("LTOP G2 evaluation interval must divide the training steps")
    if args.capacity != G2_CAPACITY or args.task_query_count != G2_TASK_QUERY_COUNT:
        raise ValueError("LTOP G2 physical capacity or task-query count changed")
    if not 0 < args.maximum_grad_norm <= 100:
        raise ValueError("LTOP G2 maximum gradient norm is invalid")
    if (
        not 0 < args.physical_set_weight <= 100
        or not 0 < args.task_address_weight <= 100
    ):
        raise ValueError("LTOP G2 loss weights must be finite positive values")
    sidecar_sha256 = args.physical_sidecar_manifest_sha256
    if (
        not isinstance(sidecar_sha256, str)
        or len(sidecar_sha256) != 64
        or any(character not in "0123456789abcdef" for character in sidecar_sha256)
    ):
        raise ValueError("LTOP G2 physical sidecar manifest SHA-256 is invalid")


def _recursive_keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {
            key for child in value.values() for key in _recursive_keys(child)
        }
    if isinstance(value, list):
        return {key for child in value for key in _recursive_keys(child)}
    return set()


def _write_json_atomic_replace(path: Path, payload: object) -> None:
    """Publish one replaceable progress record without exposing partial JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _stage_transfer_checkpoint_state(policy: Any) -> dict[str, Any]:
    """Return the model-only G2-to-G3 stage-transfer state."""

    if policy is None:
        raise TypeError("LTOP G2 checkpoint state requires a policy")
    return {"model": policy}


def _validate_representation_execution_provenance(
    execution: dict[str, Any],
    *,
    dataset_manifest_file_sha256: str,
    dataset_tree_sha256: str,
) -> None:
    """Bind the immutable four-rank source schedule to current dataset bytes."""

    if execution.get("world_size") != 4:
        raise ValueError("LTOP G2 representation source schedule must use four ranks")
    provenance = execution.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("LTOP G2 representation execution omits provenance")
    if provenance.get("current_dataset_manifest_file_sha256") != (
        dataset_manifest_file_sha256
    ):
        raise ValueError("LTOP G2 execution belongs to another dataset manifest file")
    if provenance.get("current_dataset_tree_sha256") != dataset_tree_sha256:
        raise ValueError("LTOP G2 execution belongs to another dataset tree")


def _validate_representation_item_source(
    item: dict[str, Any],
    *,
    request: Any,
    canonical_source_global_index: int,
    sidecar_source_state_sha256: str,
) -> None:
    """Bind one scheduled item to the live CALVIN record and sidecar state."""

    if item.get("sample_key") != request.sample_key:
        raise ValueError("LTOP G2 execution sample key differs from the live source")
    if isinstance(canonical_source_global_index, bool) or not isinstance(
        canonical_source_global_index, int
    ):
        raise TypeError("LTOP G2 canonical source index must be an integer")
    declared_source_global_index = item.get("source_global_index")
    if (
        declared_source_global_index is not None
        and declared_source_global_index != canonical_source_global_index
    ):
        raise ValueError("LTOP G2 execution source index differs from its sample key")
    if request.source_global_index != canonical_source_global_index:
        raise ValueError("LTOP G2 execution source index differs from the live source")
    if item.get("source_sensor_sha256") != request.source_sensor_hash_by_field:
        raise ValueError("LTOP G2 execution sensor hashes differ from the live source")
    if item.get("source_state_sha256") != sidecar_source_state_sha256:
        raise ValueError("LTOP G2 execution state hash differs from the live sidecar")


def _physical_relation_prompt_drift(left: Any, right: Any) -> float:
    """Measure prompt drift over the complete tensor-valued physical interface."""

    fields = (
        "support_logits",
        "visible_support",
        "ownership",
        "ownership_log_probability",
        "existence",
        "existence_logits",
        "row_embeddings",
        "relation_temperature",
        "sensor_valid",
        "structural_sensor_valid",
    )
    maximum = 0.0
    for field in fields:
        left_value = getattr(left, field)
        right_value = getattr(right, field)
        if left_value is None or right_value is None:
            if left_value is not None or right_value is not None:
                return math.inf
            continue
        if left_value.shape != right_value.shape:
            return math.inf
        drift = float((left_value.float() - right_value.float()).abs().max().item())
        maximum = max(maximum, drift)
    return maximum


def _scene_level_robustness(
    rank_reports: list[dict[str, Any]],
    *,
    partition: str,
    seed: int,
    bootstrap_samples: int = 100_000,
) -> dict[str, Any]:
    """Summarize scene-level support without treating prompts as independent trials."""

    rows = sorted(
        (
            scene["item_id"],
            float(scene["mean_margin"]),
        )
        for rank in rank_reports
        for scene in rank["history"][-1][partition]["scenes"]
    )
    if len(rows) != 8 or len({item_id for item_id, _margin in rows}) != len(rows):
        raise ValueError(f"LTOP G2 {partition} scene axis is incomplete or duplicated")
    margins = [margin for _item_id, margin in rows]
    mean_margin = sum(margins) / len(margins)
    rng = random.Random(seed)
    bootstrap = sorted(
        sum(margins[rng.randrange(len(margins))] for _ in margins) / len(margins)
        for _ in range(bootstrap_samples)
    )
    lower = bootstrap[int(0.025 * bootstrap_samples)]
    upper = bootstrap[int(0.975 * bootstrap_samples) - 1]
    leave_one_out = [
        sum((*margins[:index], *margins[index + 1 :])) / (len(margins) - 1)
        for index in range(len(margins))
    ]
    positive = sum(margin > 0 for margin in margins)
    sign_p = sum(
        math.comb(len(margins), count)
        for count in range(positive, len(margins) + 1)
    ) / (2 ** len(margins))
    if lower > 0:
        interpretation = "ROBUST_POSITIVE"
    elif mean_margin > 0:
        interpretation = "FRAGILE_POSITIVE"
    else:
        interpretation = "NONPOSITIVE"
    return {
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "bootstrap_95_percent_ci": [lower, upper],
        "interpretation": interpretation,
        "leave_one_scene_out_mean_range": [min(leave_one_out), max(leave_one_out)],
        "mean_margin": mean_margin,
        "positive_scene_count": positive,
        "scene_count": len(margins),
        "scene_margins": [
            {"item_id": item_id, "mean_margin": margin} for item_id, margin in rows
        ],
        "sign_test_one_sided_p": sign_p,
    }


def _load_contracts(
    execution_path: Path,
    labels_path: Path,
    *,
    expected_item_count: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    labels = json.loads(labels_path.read_text(encoding="utf-8"))
    if execution.get("schema") != "picf-next.adr157-g2-label-free-execution/v1":
        raise ValueError("LTOP G2 execution contract has the wrong schema")
    if labels.get("schema") != "picf-next.adr157-g2-offline-labels/v1":
        raise ValueError("LTOP G2 offline labels have the wrong schema")
    forbidden = {"target_identity_key", "target_mass", "target_row"}
    retained = sorted(forbidden & _recursive_keys(execution))
    if retained:
        raise ValueError(f"LTOP G2 execution contract leaks labels: {retained}")
    if labels.get("source_execution_sha256") != hashlib.sha256(
        _canonical_bytes(execution)
    ).hexdigest():
        raise ValueError("LTOP G2 labels belong to another execution contract")
    execution_ids = {item["item_id"] for item in execution["items"]}
    label_ids = {item["item_id"] for item in labels["items"]}
    if execution_ids != label_ids or len(execution_ids) != expected_item_count:
        raise ValueError("LTOP G2 contract item identities differ")
    source_world_size = execution.get("world_size")
    if (
        isinstance(source_world_size, bool)
        or not isinstance(source_world_size, int)
        or source_world_size <= 0
    ):
        raise ValueError("LTOP G2 execution contract has an invalid source world size")
    for item in execution["items"]:
        execution_rank = item.get("execution_rank")
        if (
            isinstance(execution_rank, bool)
            or not isinstance(execution_rank, int)
            or not 0 <= execution_rank < source_world_size
        ):
            raise ValueError("LTOP G2 execution contract has an invalid source rank")
        for prompt in item.get("prompts", ()):
            instruction = prompt.get("instruction")
            instruction_sha256 = prompt.get("instruction_sha256")
            if (
                not isinstance(instruction, str)
                or not instruction
                or instruction_sha256
                != hashlib.sha256(instruction.encode("utf-8")).hexdigest()
            ):
                raise ValueError("LTOP G2 execution prompt digest is invalid")
    return execution, labels


def _local_contract_items(
    execution: dict[str, Any],
    labels: dict[str, Any],
    *,
    rank: int,
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    execution_by_key = {
        (item["partition"], item["ordinal"]): item for item in execution["items"]
    }
    labels_by_id = {item["item_id"]: item for item in labels["items"]}
    selected = []
    for partition in ("validation", "heldout"):
        item = execution_by_key[(partition, rank)]
        label = labels_by_id[item["item_id"]]
        if [prompt["name"] for prompt in item["prompts"]] != [
            prompt["name"] for prompt in label["prompts"]
        ]:
            raise ValueError("LTOP G2 prompt labels differ from execution order")
        selected.append((item, label))
    return tuple(selected)


def _local_representation_contract_items(
    execution: dict[str, Any],
    labels: dict[str, Any],
    *,
    rank: int,
) -> tuple[tuple[tuple[dict[str, Any], dict[str, Any]], ...], dict[str, Any]]:
    if execution.get("world_size") != 4:
        raise ValueError("LTOP G2 representation source schedule must use four ranks")
    labels_by_id = {item["item_id"]: item for item in labels["items"]}
    runtime_records = [
        {
            "item_id": item["item_id"],
            "partition": item["partition"],
            "ordinal": item["ordinal"],
            "source_execution_rank": item["execution_rank"],
            "runtime_execution_rank": item["ordinal"] % G2_WORLD_SIZE,
        }
        for item in sorted(
            execution["items"],
            key=lambda value: (value["partition"], value["ordinal"]),
        )
    ]
    runtime_schedule = {
        "schema": "picf-next.ltop-g2-runtime-rank-rebind.v1",
        "source_world_size": execution["world_size"],
        "runtime_world_size": G2_WORLD_SIZE,
        "selection": "partition-local-ordinal-modulo-runtime-world-size",
        "records": runtime_records,
    }
    runtime_schedule["sha256"] = hashlib.sha256(
        _canonical_bytes(runtime_schedule)
    ).hexdigest()
    selected = []
    for item in execution["items"]:
        runtime_rank = item["ordinal"] % G2_WORLD_SIZE
        if runtime_rank != rank:
            continue
        label = labels_by_id[item["item_id"]]
        if [prompt["name"] for prompt in item["prompts"]] != [
            prompt["name"] for prompt in label["prompts"]
        ]:
            raise ValueError("LTOP G2 prompt labels differ from execution order")
        selected.append((item, label))
    expected = {"validation": 4, "heldout": 4}
    observed = {
        partition: sum(item["partition"] == partition for item, _label in selected)
        for partition in expected
    }
    if observed != expected:
        raise ValueError(f"LTOP G2 representation partition differs: {observed}")
    return (
        tuple(sorted(selected, key=lambda value: (value[0]["partition"], value[0]["ordinal"]))),
        runtime_schedule,
    )


def _prompt_variant(source: Any, prompt: dict[str, Any]) -> Any:
    host_item = copy.deepcopy(source.training.host_items[0])
    host_item["task"] = prompt["instruction"]
    request = source.training.structural_target_requests[0]
    training = replace(
        source.training,
        host_items=(host_item,),
        structural_target_requests=(replace(request, task_key=prompt["task_key"]),),
    )
    return replace(source, training=training)


def _episode_ids(episode_keys: tuple[str, ...], *, torch_module: Any, device: Any) -> Any:
    values = [
        int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big")
        & ((1 << 63) - 1)
        for value in episode_keys
    ]
    return torch_module.tensor(values, dtype=torch_module.long, device=device)


def _scene_metrics(
    distributions: tuple[Any, Any],
    target_rows: tuple[int, int],
    *,
    task_address_row_coverage: Any,
    torch_module: Any,
) -> dict[str, Any]:
    prompts = []
    row_permutation_errors = []
    for index, distribution in enumerate(distributions):
        device = distribution.device
        target = torch_module.tensor([target_rows[index]], dtype=torch_module.long, device=device)
        alternate = torch_module.tensor(
            [target_rows[1 - index]], dtype=torch_module.long, device=device
        )
        target_coverage = float(
            task_address_row_coverage(distribution, target).detach().cpu().item()
        )
        alternate_coverage = float(
            task_address_row_coverage(distribution, alternate).detach().cpu().item()
        )
        permuted = distribution.flip(dims=(-1,))
        permuted_target = torch_module.tensor(
            [distribution.shape[-1] - 1 - target_rows[index]],
            dtype=torch_module.long,
            device=device,
        )
        permuted_target_coverage = float(
            task_address_row_coverage(permuted, permuted_target).detach().cpu().item()
        )
        row_permutation_errors.append(abs(target_coverage - permuted_target_coverage))
        mean_distribution = distribution.mean(dim=1)[0]
        prompts.append(
            {
                "target_row": target_rows[index],
                "alternate_row": target_rows[1 - index],
                "target_coverage": target_coverage,
                "alternate_coverage": alternate_coverage,
                "margin": target_coverage - alternate_coverage,
                "top_row": int(mean_distribution.argmax().item()),
                "mean_row_distribution": [float(value) for value in mean_distribution.cpu()],
            }
        )
    first = distributions[0].float().flatten(1)
    second = distributions[1].float().flatten(1)
    cosine = torch_module.nn.functional.cosine_similarity(first, second, dim=-1)
    prompt_l1 = (distributions[0] - distributions[1]).abs().sum(dim=-1).mean()
    margins = [value["margin"] for value in prompts]
    losses = [
        -float(
            torch_module.log(
                torch_module.tensor(max(value["target_coverage"], 1e-30))
            )
        )
        for value in prompts
    ]
    return {
        "prompts": prompts,
        "mean_margin": sum(margins) / len(margins),
        "positive_margin_count": sum(value > 0 for value in margins),
        "mean_target_nll": sum(losses) / len(losses),
        "prompt_distribution_cosine": float(cosine.mean().cpu().item()),
        "prompt_distribution_mean_l1": float(prompt_l1.cpu().item()),
        "metric_self_checks": {
            "matched_row_permutation_max_abs_error": max(row_permutation_errors),
        },
    }


def _computed_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    ranks = report["rank_reports"]
    for rank_report in ranks:
        history = rank_report["history"]
        initial = history[0]
        final = history[-1]
        validation_initial = initial["validation"]
        validation_final = final["validation"]
        if validation_final["positive_margin_count"] != 2:
            failures.append(
                f"rank {rank_report['rank']}: "
                "validation crossed margins are not both positive"
            )
        if validation_final["mean_margin"] < 0.10:
            failures.append(f"rank {rank_report['rank']}: validation mean margin is below 0.10")
        if validation_final["mean_target_nll"] > 0.70 * validation_initial["mean_target_nll"]:
            failures.append(
                f"rank {rank_report['rank']}: validation NLL did not fall by 30 percent"
            )
        if final["heldout"]["positive_margin_count"] < 1:
            failures.append(
                f"rank {rank_report['rank']}: heldout pair has no positive target margin"
            )
        if not rank_report["all_gradients_finite"]:
            failures.append(f"rank {rank_report['rank']}: task-query gradient became non-finite")
        if min(rank_report["gradient_norms"]) <= 0:
            failures.append(f"rank {rank_report['rank']}: task-query gradient vanished")
    heldout_margins = [
        prompt["margin"]
        for rank_report in ranks
        for prompt in rank_report["history"][-1]["heldout"]["prompts"]
    ]
    if sum(value > 0 for value in heldout_margins) < 3:
        failures.append("heldout target margin is positive for fewer than three of four prompts")
    if sum(heldout_margins) / len(heldout_margins) <= 0:
        failures.append("heldout mean target margin is not positive")
    if len({rank_report["final_task_query_local_sha256"] for rank_report in ranks}) != 1:
        failures.append("data-parallel task-query replicas diverged")
    return failures


def _computed_representation_failures(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    ranks = report["rank_reports"]
    validation_initial = [rank["history"][0]["validation"] for rank in ranks]
    validation_final = [rank["history"][-1]["validation"] for rank in ranks]
    heldout_initial = [rank["history"][0]["heldout"] for rank in ranks]
    heldout_final = [rank["history"][-1]["heldout"] for rank in ranks]

    def aggregate(values: list[dict[str, Any]], field: str) -> float:
        return sum(float(value[field]) for value in values) / len(values)

    validation_prompts = [
        prompt for value in validation_final for prompt in value["prompts"]
    ]
    heldout_prompts = [prompt for value in heldout_final for prompt in value["prompts"]]
    if sum(prompt["margin"] > 0 for prompt in validation_prompts) < 12:
        failures.append("validation target margin is positive for fewer than 12 of 16 prompts")
    if aggregate(validation_final, "mean_margin") < 0.02:
        failures.append("validation mean target margin is below 0.02")
    if aggregate(validation_final, "mean_target_nll") > 0.85 * aggregate(
        validation_initial,
        "mean_target_nll",
    ):
        failures.append("validation target NLL did not fall by 15 percent")
    if sum(prompt["margin"] > 0 for prompt in heldout_prompts) < 10:
        failures.append("heldout target margin is positive for fewer than 10 of 16 prompts")
    if aggregate(heldout_final, "mean_margin") <= 0:
        failures.append("heldout mean target margin is not positive")
    if aggregate(heldout_final, "mean_target_nll") > aggregate(
        heldout_initial,
        "mean_target_nll",
    ):
        failures.append("heldout target NLL regressed")
    if aggregate(validation_final, "mean_physical_set_loss") > 0.80 * aggregate(
        validation_initial,
        "mean_physical_set_loss",
    ):
        failures.append("validation physical set loss did not fall by 20 percent")
    if aggregate(heldout_final, "mean_physical_set_loss") > aggregate(
        heldout_initial,
        "mean_physical_set_loss",
    ):
        failures.append("heldout physical set loss regressed")
    if max(
        float(value["physical_prompt_drift_max_abs"])
        for value in (*validation_final, *heldout_final)
    ) > 1.0e-5:
        failures.append("task text leaked into task-independent physical row predictions")
    if any(
        not partition["shared_row_gauge"]
        for rank in ranks
        for history in rank["history"]
        for partition in (history["validation"], history["heldout"])
    ):
        failures.append("crossed prompts did not preserve one canonical physical row gauge")
    if max(
        float(
            partition["metric_self_checks"][
                "matched_row_permutation_max_abs_error"
            ]
        )
        for rank in ranks
        for history in rank["history"]
        for partition in (history["validation"], history["heldout"])
    ) > 1.0e-6:
        failures.append("task-address metric implementation self-check failed")
    runtime_schedule_digests = {
        rank.get("runtime_schedule_sha256") for rank in ranks
    }
    if None in runtime_schedule_digests or len(runtime_schedule_digests) != 1:
        failures.append("data-parallel ranks used different runtime execution schedules")
    optimizer_schema_digests = {
        rank.get("optimizer_parameter_manifest", {}).get("schema_sha256")
        for rank in ranks
    }
    if None in optimizer_schema_digests or len(optimizer_schema_digests) != 1:
        failures.append("data-parallel ranks optimized different parameter schemas")
    frozen_action_name_axes = {
        tuple(rank.get("frozen_action_state", {}).get("tensor_names", ()))
        for rank in ranks
    }
    if () in frozen_action_name_axes or len(frozen_action_name_axes) != 1:
        failures.append("data-parallel ranks audited different frozen action tensors")
    if report.get("checkpoint", {}).get("requested"):
        model_digests = [rank.get("model_local_state_sha256") for rank in ranks]
        if any(
            not isinstance(value, str) or len(value) != 64 for value in model_digests
        ):
            failures.append("checkpoint candidate omitted a rank-local model digest")
    for rank in ranks:
        if not rank["all_gradients_finite"] or min(rank["gradient_norms"]) <= 0:
            failures.append(f"rank {rank['rank']}: representation gradient is invalid")
        final_gradient = rank["gradient_metrics_history"][-1]
        for surface in ("native_graph", "task_query", "shared_host"):
            if float(final_gradient.get(f"{surface}_norm", 0.0)) <= 0:
                failures.append(f"rank {rank['rank']}: {surface} gradient vanished")
        if float(final_gradient.get("action_output_norm", 0.0)) != 0:
            failures.append(f"rank {rank['rank']}: frozen action output received a gradient")
        frozen_action_state = rank.get("frozen_action_state")
        if not isinstance(frozen_action_state, dict):
            failures.append(f"rank {rank['rank']}: frozen action state was not audited")
        elif (
            frozen_action_state["before_sha256"] != frozen_action_state["after_sha256"]
            or frozen_action_state["changed_tensors"]
        ):
            failures.append(f"rank {rank['rank']}: frozen action state changed")
    return failures


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(root=root, checkout=args.source_checkout, check_apply=True)
    prepared = validate_prepared_native_source(
        checkout=args.source_checkout,
        patch_path=args.patch,
    )
    if prepared.get("patched_source_sha256") != patch_report.get("patched_source_sha256"):
        raise RuntimeError("LTOP G2 LingBot source differs from immutable patch replay")
    validate_checkpoint(args.checkpoint_dir)
    validate_processor(args.processor_dir)
    execution, labels = _load_contracts(
        args.execution_contract,
        args.offline_labels,
        expected_item_count=4 if args.training_scope == "query-only" else 16,
    )

    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(args.source_checkout.resolve()))

    import numpy as np
    import torch
    import torch.distributed as dist
    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.optim import build_muon_optimizer, build_optimizer
    from transformers import AutoConfig
    from transformers.modeling_utils import no_init_weights

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
    from picf_next.lingbot_native.capacity import (
        require_checkpoint_write_capacity,
        require_persistent_run_root,
    )
    from picf_next.lingbot_native.entity_set_objective import (
        match_physical_frame_entities,
        physical_frame_set_loss,
    )
    from picf_next.lingbot_native.host import (
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        ObjectReadActionIntervention,
        install_lingbot_native_graph,
        native_context_from_prior_trace,
    )
    from picf_next.lingbot_native.physical_relations import PhysicalRelationOutput
    from picf_next.lingbot_native.representation_stage import (
        configure_native_representation_parameter_scope,
        native_representation_action_state_changes,
        native_representation_action_state_manifest_sha256,
        native_representation_frozen_action_state_manifest,
        verify_native_representation_parameter_scope,
    )
    from picf_next.lingbot_native.state import AddressedLayerwisePriorTrace
    from picf_next.lingbot_native.task_address_learning import (
        task_address_row_coverage,
        task_address_target_coverage,
    )
    from picf_next.lingbot_native.task_address_receipt import task_address_attention_receipt
    from picf_next.lingbot_native.training import (
        _run_native_observation_training_forward,
        audit_native_optimizer_coverage,
        run_native_policy_observation_diagnostic_forward,
    )
    from tools.run_lingbot_vla2_native_g0 import _distributed_gradient_metrics

    if os.environ.get("WORLD_SIZE") != str(G2_WORLD_SIZE):
        raise RuntimeError("LTOP G2 requires torchrun with exactly two processes")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if torch.cuda.device_count() != G2_WORLD_SIZE:
            raise RuntimeError("LTOP G2 process sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("LTOP G2 requires two A100 devices with at least 39 GiB each")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed + rank)
        np.random.seed(args.seed + rank)
        torch.cuda.reset_peak_memory_stats(device)
        if args.training_scope == "representation":
            init_parallel_state(
                dp_size=G2_WORLD_SIZE,
                dp_replicate_size=1,
                dp_shard_size=G2_WORLD_SIZE,
                tp_size=1,
                ep_size=1,
                pp_size=1,
                cp_size=1,
                ulysses_size=1,
                dp_mode="fsdp2",
            )

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
            raise ValueError("LTOP G2 CALVIN manifest and normalization differ")
        dataset_contract = validate_dataset_runtime_binding(
            manifest,
            args.dataset_split,
            dataset_id=norm_source["dataset_id"],
            dataset_revision=norm_source["dataset_revision"],
            split_name=args.dataset_split.name,
        )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        sidecar = CalvinPhysicalSupervisionSidecar(
            args.physical_sidecar_root,
            index,
            manifest_path=args.physical_sidecar_manifest,
            expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        )
        if args.training_scope == "representation":
            _validate_representation_execution_provenance(
                execution,
                dataset_manifest_file_sha256=_sha256(args.dataset_manifest),
                dataset_tree_sha256=manifest.tree_sha256,
            )

        training = load_lingbot_training_config(args.training_config)
        train_section = training.get("train")
        if not isinstance(train_section, dict):
            raise ValueError("LTOP G2 LingBot training config omits train")
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=float(train_section.get("lr", 5.0e-5)),
        )
        merged, data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=args.steps,
        )
        merged.update(
            {
                "use_cache": False,
                "use_compile": False,
                "attention_implementation": "eager",
                "vit_attn_implementation": "eager",
            }
        )
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

        load_started = time.perf_counter()
        processor = build_processor(str(args.processor_dir.resolve()))
        apply_lingbot_qwen3_vl_patch()
        apply_lingbot_qwen2_patch()
        model_dtype = (
            torch.float32 if args.training_scope == "representation" else torch.bfloat16
        )
        with init_empty_weights(), no_init_weights():
            policy = LingbotVlaV2Policy(
                config=config,
                eval=args.training_scope == "query-only",
            ).to(model_dtype)
        load_model_weights(
            policy,
            str(args.checkpoint_dir.resolve()),
            str(device),
            post_training=True,
            adanorm_time=bool(config.adanorm_time),
        )
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            task_query_count=args.task_query_count,
            architecture_identity=G2_ARCHITECTURE,
        )
        graph = LingBotNativeGraph(graph_config, device=device, dtype=model_dtype)
        install_lingbot_native_graph(policy, graph)
        if graph.task_query_embeddings is None:
            raise RuntimeError("LTOP G2 graph omitted TASK_QUERY embeddings")
        representation_scope = None
        frozen_action_before = None
        if args.training_scope == "query-only":
            policy.requires_grad_(False)
            graph.task_query_embeddings.requires_grad_(True)
            policy.train()
            trainable = [
                (name, value)
                for name, value in policy.named_parameters()
                if value.requires_grad
            ]
            if len(trainable) != 1 or not trainable[0][0].endswith("task_query_embeddings"):
                raise RuntimeError("LTOP G2 query-only parameter scope changed")
            optimizer = build_lingbot_query_only_optimizer(
                policy,
                optimizer_contract,
                build_optimizer=build_optimizer,
            )
        else:
            policy.train()
            representation_scope = configure_native_representation_parameter_scope(policy)
            policy = build_parallelize_model(
                policy,
                enable_full_shard=True,
                enable_mixed_precision=optimizer_contract.enable_mixed_precision,
                enable_fp32=optimizer_contract.enable_fp32,
                enable_gradient_checkpointing=True,
                init_device="cuda",
                enable_fsdp_offload=args.fsdp2_placement == FSDP2_CPU_OFFLOAD,
                enable_shared_embedding_offload=(
                    args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD
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
            register_native_fsdp_forward_methods(policy)
            verify_native_representation_parameter_scope(
                policy,
                expected=representation_scope,
            )
            trainable = [
                (name, value)
                for name, value in policy.named_parameters()
                if value.requires_grad
            ]
            if not trainable or not any(
                name.endswith("task_query_embeddings") for name, _value in trainable
            ):
                raise RuntimeError("LTOP G2 representation scope lost native task queries")
            optimizer = build_lingbot_representation_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
            )
            verify_native_representation_parameter_scope(
                policy,
                expected=representation_scope,
            )
            frozen_action_before = native_representation_frozen_action_state_manifest(
                policy,
                expected=representation_scope,
            )
        parameter_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        load_duration_s = time.perf_counter() - load_started

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
                if "object_read_slice" not in active_capture:
                    context = active_capture["context"]
                    native_valid = context.native_valid
                    native_roles = context.native_roles
                    if native_valid is None or native_roles is None:
                        raise RuntimeError("LTOP G2 attention preceded native metadata binding")
                    original_prefix_count = native_valid.shape[1]
                    language_slice = graph._instruction_span(native_roles)
                    task_text_count = language_slice.stop - language_slice.start
                    prior_start = (
                        original_prefix_count
                        + task_text_count
                        + context.controls.token_count
                    )
                    prior_slice = slice(prior_start, prior_start + args.capacity)
                    posterior_slice = slice(prior_slice.stop, prior_slice.stop + args.capacity)
                    task_query_slice = slice(
                        posterior_slice.stop,
                        posterior_slice.stop + args.task_query_count,
                    )
                    active_capture.update(
                        {
                            "prior_slice": prior_slice,
                            "posterior_slice": posterior_slice,
                            "object_read_slice": slice(
                                task_query_slice.stop,
                                task_query_slice.stop + args.task_query_count,
                            ),
                        }
                    )
                receipt = task_address_attention_receipt(
                    query_states=query_states,
                    key_states=key_states,
                    attention_mask=attention_mask,
                    object_read_slice=active_capture["object_read_slice"],
                    prior_slice=active_capture["prior_slice"],
                    posterior_slice=active_capture["posterior_slice"],
                    capacity=args.capacity,
                )
                active_capture["layer_count"] += 1
                active_capture["final_row_mass"] = receipt.row_mass
            return original_attention_interface(
                query_states,
                key_states,
                value_states,
                attention_mask,
            )

        joint_host.attention_interface = attention_interface_with_receipt

        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        vision_config = config.vision_config
        patch_size = int(vision_config.patch_size)
        merge_size = int(vision_config.spatial_merge_size)
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

        representation_runtime_schedule = None
        if args.training_scope == "query-only":
            local_items = _local_contract_items(execution, labels, rank=rank)
        else:
            local_items, representation_runtime_schedule = (
                _local_representation_contract_items(execution, labels, rank=rank)
            )

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
            prior_valid = torch.zeros(1, dtype=torch.bool, device=device)
            with torch.no_grad():
                for controls in batch.effective_prior_control_chunks:
                    prior = prior_stepper(
                        prior,
                        controls,
                        previous_memory_valid=prior_valid,
                        episode_ids=episode_ids,
                    )
                    prior_valid = torch.ones_like(prior_valid)
            if not isinstance(prior, AddressedLayerwisePriorTrace):
                raise RuntimeError("LTOP G2 prior rollout omitted addressed rows")
            return prior

        def forward_attention(
            batch: CollatedNativeCALVINBatch,
            prior: AddressedLayerwisePriorTrace,
            *,
            require_grad: bool,
        ) -> tuple[Any, Any]:
            nonlocal active_capture
            context = native_context_from_prior_trace(
                controls=batch.controls,
                prior_trace=prior,
                modalities=None,
                object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
            )
            active_capture = {"context": context, "layer_count": 0, "final_row_mass": None}
            try:
                if require_grad:
                    context = _run_native_observation_training_forward(
                        policy,
                        model_inputs=batch.model_inputs,
                        context=context,
                        require_prediction_grad=False,
                        required_relation_grad_fields=(
                            (
                                "support_logits",
                                "ownership_log_probability",
                                "existence_logits",
                            )
                            if args.training_scope == "representation"
                            else ()
                        ),
                    )
                else:
                    context = run_native_policy_observation_diagnostic_forward(
                        policy,
                        model_inputs=batch.model_inputs,
                        context=context,
                    )
            finally:
                captured = active_capture
                active_capture = None
            if captured is None or captured["layer_count"] != graph.config.num_layers:
                raise RuntimeError("LTOP G2 did not capture every shared-host layer")
            row_mass = captured["final_row_mass"]
            if row_mass is None or require_grad != bool(row_mass.requires_grad):
                raise RuntimeError("LTOP G2 final attention attachment differs from the phase")
            return context, row_mass

        scenes: dict[str, list[dict[str, Any]]] = {"validation": [], "heldout": []}
        for (item, label) in local_items:
            source = build_native_calvin_replay_batch(
                dataset,
                sample_key=item["sample_key"],
                lane_id=rank,
                episode_instance_id=f"ltop-g2/{item['item_id']}",
                optimizer_step=0,
                replay_seed=item["replay_seed"],
                device=device,
                dtype=torch.bfloat16,
            )
            if len(source.training.structural_target_requests) != 1:
                raise RuntimeError("LTOP G2 replay did not produce one structural request")
            source_request = source.training.structural_target_requests[0]
            if args.training_scope == "representation":
                _validate_representation_item_source(
                    item,
                    request=source_request,
                    canonical_source_global_index=dataset.source_global_index_by_key(
                        item["sample_key"]
                    ),
                    sidecar_source_state_sha256=sidecar.source_state_sha256(
                        source_request.source_global_index
                    ),
                )
            planned = tuple(_prompt_variant(source, prompt) for prompt in item["prompts"])
            batches = tuple(collate(value) for value in planned)
            scene_fields = ("image_grid_thw", "images", "img_masks")
            if any(
                not torch.equal(batches[0].model_inputs[name], batches[1].model_inputs[name])
                for name in scene_fields
            ):
                raise RuntimeError("LTOP G2 crossed prompts changed the scene tensors")
            target_identities: list[str] = []
            for prompt, prompt_label in zip(item["prompts"], label["prompts"], strict=True):
                identities = calvin_exact_task_loss_identities(prompt["task_key"])
                if identities is None or len(identities) != 1:
                    raise RuntimeError("LTOP G2 requires one exact task identity")
                identity = identities[0]
                if identity != prompt_label["target_identity_key"]:
                    raise RuntimeError("LTOP G2 exact task resolver differs from offline labels")
                target_identities.append(identity)
            scene = {
                "item": item,
                "batches": batches,
                "target_identities": tuple(target_identities),
            }
            if args.training_scope == "query-only":
                prior = build_prior(batches[0])
                with torch.inference_mode():
                    context, _row_mass = forward_attention(batches[0], prior, require_grad=False)
                relation = context.relation_output
                if not isinstance(relation, PhysicalRelationOutput):
                    raise RuntimeError("LTOP G2 observation omitted physical relations")
                target_bundle = build_task_independent_calvin_targets(
                    requests_by_time=(batches[0].structural_target_requests,),
                    model_inputs_by_time=(batches[0].model_inputs,),
                    relations=(relation,),
                    physical_sidecar=sidecar,
                    capacity=args.capacity,
                    patch_size=patch_size,
                    merge_size=merge_size,
                )[0]
                predictions = physical_frame_predictions_from_relation(relation)
                assignment = match_physical_frame_entities(predictions, target_bundle.targets)
                bindings = physical_frame_row_bindings(
                    target_bundle,
                    assignment,
                    capacity=args.capacity,
                )[0]
                binding_map = dict(bindings)
                target_rows = tuple(binding_map[identity] for identity in target_identities)
                if target_rows[0] == target_rows[1]:
                    raise RuntimeError("LTOP G2 crossed target identities share one physical row")
                scene.update(
                    {
                        "prior": prior,
                        "bindings": bindings,
                        "target_rows": target_rows,
                    }
                )
            scenes[item["partition"]].append(scene)

        def physical_supervision(
            *,
            context: Any,
            batch: CollatedNativeCALVINBatch,
            target_identity: str,
            canonical_assignment: Any | None = None,
            canonical_identity_keys: Any | None = None,
        ) -> dict[str, Any]:
            relation = context.relation_output
            if not isinstance(relation, PhysicalRelationOutput):
                raise RuntimeError("LTOP G2 observation omitted physical relations")
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
                    "LTOP G2 crossed prompts changed the task-independent identity axis"
                )
            predictions = physical_frame_predictions_from_relation(relation)
            matched_assignment = match_physical_frame_entities(
                predictions,
                target_bundle.targets,
            )
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
            binding_map = dict(bindings)
            if target_identity not in binding_map:
                raise RuntimeError(
                    f"LTOP G2 target identity is not physically bound: {target_identity}"
                )
            return {
                "relation": relation,
                "set_loss": set_loss,
                "assignment": assignment,
                "matched_assignment": matched_assignment,
                "bindings": bindings,
                "matched_bindings": matched_bindings,
                "identity_keys_by_batch": target_bundle.identity_keys_by_batch,
                "target_row": binding_map[target_identity],
            }

        def evaluate_scene(scene: dict[str, Any]) -> dict[str, Any]:
            distributions = []
            target_rows: list[int] = []
            bindings_by_prompt = []
            set_losses = []
            relation_predictions = []
            independent_bindings_by_prompt = []
            canonical_assignment = None
            canonical_bindings = None
            canonical_identity_keys = None
            prior = scene.get("prior") or build_prior(scene["batches"][0])
            # FSDP2 may retain materialized shards across the following training
            # forward.  Inference tensors cannot subsequently participate in
            # autograd, while no_grad preserves the exact diagnostic semantics.
            with torch.no_grad():
                for prompt_index, batch in enumerate(scene["batches"]):
                    context, row_mass = forward_attention(batch, prior, require_grad=False)
                    physical = physical_supervision(
                        context=context,
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
                        raise RuntimeError("LTOP G2 canonical row gauge was not established")
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
            metrics = _scene_metrics(
                cast(tuple[Any, Any], tuple(distributions)),
                cast(tuple[int, int], tuple(target_rows)),
                task_address_row_coverage=task_address_row_coverage,
                torch_module=torch,
            )
            physical_prompt_drift = _physical_relation_prompt_drift(
                relation_predictions[0],
                relation_predictions[1],
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
                "physical_prompt_drift_max_abs": physical_prompt_drift,
            }

        def evaluate_partition(partition: str) -> dict[str, Any]:
            per_scene = [evaluate_scene(scene) for scene in scenes[partition]]
            prompts = [prompt for scene in per_scene for prompt in scene["prompts"]]
            return {
                "scene_count": len(per_scene),
                "prompt_count": len(prompts),
                "mean_margin": sum(prompt["margin"] for prompt in prompts) / len(prompts),
                "positive_margin_count": sum(prompt["margin"] > 0 for prompt in prompts),
                "mean_target_nll": sum(scene["mean_target_nll"] for scene in per_scene)
                / len(per_scene),
                "mean_physical_set_loss": sum(
                    scene["mean_physical_set_loss"] for scene in per_scene
                )
                / len(per_scene),
                "physical_prompt_drift_max_abs": max(
                    scene["physical_prompt_drift_max_abs"] for scene in per_scene
                ),
                "shared_row_gauge": all(scene["shared_row_gauge"] for scene in per_scene),
                "metric_self_checks": {
                    "matched_row_permutation_max_abs_error": max(
                        scene["metric_self_checks"][
                            "matched_row_permutation_max_abs_error"
                        ]
                        for scene in per_scene
                    ),
                },
                "prompts": prompts,
                "scenes": per_scene,
            }

        history = []
        gradient_norms: list[float] = []
        gradient_metrics_history: list[dict[str, Any]] = []
        all_gradients_finite = True

        def record(step: int) -> None:
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            try:
                history.append(
                    {
                        "step": step,
                        "validation": evaluate_partition("validation"),
                        "heldout": evaluate_partition("heldout"),
                    }
                )
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)

        record(0)
        train_started = time.perf_counter()
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            if args.training_scope == "query-only":
                scene = scenes["validation"][0]
                losses = []
                for prompt_index, batch in enumerate(scene["batches"]):
                    _context, row_mass = forward_attention(
                        batch,
                        scene["prior"],
                        require_grad=True,
                    )
                    result = task_address_target_coverage(
                        row_mass,
                        torch.tensor(
                            [scene["target_rows"][prompt_index]],
                            dtype=torch.long,
                            device=device,
                        ),
                    )
                    losses.append(result.loss)
                loss = torch.stack(losses).mean()
            else:
                scene_index = (step - 1) % len(scenes["validation"])
                prompt_index = ((step - 1) // len(scenes["validation"])) % 2
                scene = scenes["validation"][scene_index]
                batch = scene["batches"][prompt_index]
                prior = build_prior(scene["batches"][0])
                canonical_assignment = None
                canonical_bindings = None
                canonical_identity_keys = None
                if prompt_index != 0:
                    with torch.no_grad():
                        canonical_context, _canonical_row_mass = forward_attention(
                            scene["batches"][0],
                            prior,
                            require_grad=False,
                        )
                        canonical_physical = physical_supervision(
                            context=canonical_context,
                            batch=scene["batches"][0],
                            target_identity=scene["target_identities"][0],
                        )
                    canonical_assignment = canonical_physical["matched_assignment"]
                    canonical_bindings = canonical_physical["matched_bindings"]
                    canonical_identity_keys = canonical_physical["identity_keys_by_batch"]
                context, row_mass = forward_attention(batch, prior, require_grad=True)
                physical = physical_supervision(
                    context=context,
                    batch=batch,
                    target_identity=scene["target_identities"][prompt_index],
                    canonical_assignment=canonical_assignment,
                    canonical_identity_keys=canonical_identity_keys,
                )
                if canonical_bindings is None:
                    canonical_bindings = physical["matched_bindings"]
                if physical["matched_bindings"] != canonical_bindings:
                    raise RuntimeError(
                        "LTOP G2 crossed prompt changed the canonical physical row gauge"
                    )
                address = task_address_target_coverage(
                    row_mass,
                    torch.tensor(
                        [physical["target_row"]],
                        dtype=torch.long,
                        device=device,
                    ),
                )
                loss = (
                    args.physical_set_weight * physical["set_loss"].total
                    + args.task_address_weight * address.loss
                )
            loss.backward()
            if args.training_scope == "query-only":
                gradient = graph.task_query_embeddings.grad
                if gradient is None:
                    raise RuntimeError("LTOP G2 produced no TASK_QUERY gradient")
                finite = bool(torch.isfinite(gradient).all().item())
                all_gradients_finite &= finite
                if not finite:
                    raise FloatingPointError("LTOP G2 produced a non-finite TASK_QUERY gradient")
                dist.all_reduce(gradient, op=dist.ReduceOp.SUM)
                gradient.div_(G2_WORLD_SIZE)
                step_metrics = {
                    "all_finite": finite,
                    "task_query_norm": float(gradient.detach().float().norm().item()),
                    "task_query_elements": gradient.numel(),
                }
                parameters_to_clip = (graph.task_query_embeddings,)
            else:
                step_metrics = _distributed_gradient_metrics(
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
                finite = bool(step_metrics["all_finite"])
                all_gradients_finite &= finite
                if not finite:
                    raise FloatingPointError(
                        "LTOP G2 produced a non-finite representation gradient"
                    )
                if (
                    float(step_metrics.get("native_graph_norm", 0.0)) <= 0
                    or float(step_metrics.get("task_query_norm", 0.0)) <= 0
                    or float(step_metrics.get("shared_host_norm", 0.0)) <= 0
                ):
                    raise RuntimeError("LTOP G2 representation gradient missed a required surface")
                if float(step_metrics.get("action_output_norm", 0.0)) != 0:
                    raise RuntimeError("LTOP G2 representation loss reached frozen action output")
                parameters_to_clip = tuple(policy.parameters())
            gradient_metrics_history.append(step_metrics)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters_to_clip,
                args.maximum_grad_norm,
                error_if_nonfinite=True,
                foreach=False,
            )
            full_tensor = getattr(grad_norm, "full_tensor", None)
            if callable(full_tensor):
                grad_norm = full_tensor()
            gradient_norms.append(float(grad_norm.item()))
            optimizer.step()
            if step % args.eval_every == 0 or step == args.steps:
                record(step)
            if (
                rank == 0
                and args.progress_output is not None
                and (step % args.progress_every == 0 or step == args.steps)
            ):
                elapsed_s = time.perf_counter() - train_started
                mean_step_s = elapsed_s / step
                _write_json_atomic_replace(
                    args.progress_output,
                    {
                        "schema": "picf-next.ltop-g2-progress.v1",
                        "completed_steps": step,
                        "total_steps": args.steps,
                        "elapsed_s": elapsed_s,
                        "mean_elapsed_per_completed_step_s": mean_step_s,
                        "estimated_remaining_s": mean_step_s * (args.steps - step),
                        "training_scope": args.training_scope,
                        "updated_unix_s": time.time(),
                    },
                )
        torch.cuda.synchronize(device)
        train_duration_s = time.perf_counter() - train_started
        joint_host.attention_interface = original_attention_interface

        frozen_action_after = None
        frozen_action_changes: tuple[str, ...] = ()
        if representation_scope is not None:
            verify_native_representation_parameter_scope(
                policy,
                expected=representation_scope,
            )
            final_parameter_manifest = audit_native_optimizer_coverage(
                modules={"policy": policy},
                optimizer=optimizer,
            )
            if final_parameter_manifest != parameter_manifest:
                raise RuntimeError("LTOP G2 optimizer ownership changed during training")
            if frozen_action_before is None:
                raise RuntimeError("LTOP G2 omitted the initial frozen-action manifest")
            frozen_action_after = native_representation_frozen_action_state_manifest(
                policy,
                expected=representation_scope,
            )
            frozen_action_changes = native_representation_action_state_changes(
                frozen_action_before,
                frozen_action_after,
            )

        query_tensor = graph.task_query_embeddings.detach()
        to_local = getattr(query_tensor, "to_local", None)
        if callable(to_local):
            query_tensor = to_local()
        query_bytes = query_tensor.float().cpu().contiguous().numpy().tobytes()
        model_local_state_sha256 = (
            _model_local_state_digest(policy, torch)
            if representation_scope is not None and args.output_checkpoint is not None
            else None
        )
        rank_report = {
            "rank": rank,
            "local_items": {
                partition: [
                    {
                        "item_id": scene["item"]["item_id"],
                        "sample_key": scene["item"]["sample_key"],
                        "target_identities": list(scene["target_identities"]),
                    }
                    for scene in partition_scenes
                ]
                for partition, partition_scenes in scenes.items()
            },
            "history": history,
            "gradient_norms": gradient_norms,
            "gradient_metrics_history": gradient_metrics_history,
            "all_gradients_finite": all_gradients_finite,
            "runtime_schedule_sha256": (
                None
                if representation_runtime_schedule is None
                else representation_runtime_schedule["sha256"]
            ),
            "optimizer_parameter_manifest": asdict(parameter_manifest),
            "frozen_action_state": (
                None
                if frozen_action_before is None or frozen_action_after is None
                else {
                    "before_sha256": native_representation_action_state_manifest_sha256(
                        frozen_action_before
                    ),
                    "after_sha256": native_representation_action_state_manifest_sha256(
                        frozen_action_after
                    ),
                    "changed_tensors": list(frozen_action_changes),
                    "tensor_count": len(frozen_action_before),
                    "parameter_count": sum(
                        value.kind == "parameter" for value in frozen_action_before
                    ),
                    "buffer_count": sum(
                        value.kind == "buffer" for value in frozen_action_before
                    ),
                    "local_numel": sum(value.numel for value in frozen_action_before),
                    "tensor_names": [value.name for value in frozen_action_before],
                }
            ),
            "model_local_state_sha256": model_local_state_sha256,
            "final_task_query_local_sha256": hashlib.sha256(query_bytes).hexdigest(),
            "timings": {
                "load_model_s": load_duration_s,
                "train_and_eval_s": train_duration_s,
                "mean_optimizer_step_s": train_duration_s / args.steps,
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
            representation_mode = args.training_scope == "representation"
            report = {
                "schema": G2_REPRESENTATION_SCHEMA if representation_mode else G2_SCHEMA,
                "status": "PASS",
                "failures": [],
                "architecture_identity": G2_ARCHITECTURE,
                "model_identity": {
                    "checkpoint_id": LINGBOT_CHECKPOINT_ID,
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "native_source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "patched_source_sha256": prepared["patched_source_sha256"],
                },
                "training_scope": args.training_scope,
                "world_size": G2_WORLD_SIZE,
                "execution_schedule": representation_runtime_schedule,
                "steps": args.steps,
                "eval_steps": [entry["step"] for entry in rank_reports[0]["history"]],
                "capacity": args.capacity,
                "task_query_count": args.task_query_count,
                "trainable_scope": {
                    "parameter_names": (
                        [name for name, _value in trainable]
                        if not representation_mode
                        else None
                    ),
                    "parameter_count": len(trainable),
                    "parameter_numel": sum(value.numel() for _name, value in trainable),
                    "released_policy_frozen": not representation_mode,
                    "released_action_only_frozen": representation_mode,
                    "representation_scope": (
                        None if representation_scope is None else representation_scope.as_dict()
                    ),
                    "action_suffix_executed": False,
                    "deploy_time_module_added": False,
                },
                "loss_contract": {
                    "surface": "real post-MRoPE eager-attention OBJECT_READ row mass",
                    "aggregation": "noisy-OR target coverage across four existing reads",
                    "target_row": (
                        "one prompt-independent canonical Hungarian row gauge per scene and "
                        "optimizer step; every crossed prompt must reproduce the full binding"
                    ),
                    "physical_set_weight": args.physical_set_weight,
                    "task_address_weight": args.task_address_weight,
                    "optimizer": {
                        **optimizer_contract.metadata,
                        "gate_builder": (
                            "lingbotvla.optim.build_muon_optimizer"
                            if representation_mode
                            else "lingbotvla.optim.build_optimizer"
                        ),
                        "gate_algorithm": "released_muon_adamw" if representation_mode else "adamw",
                        "gate_reason": (
                            "production frozen-action representation stage"
                            if representation_mode
                            else "released fallback for embedding-only trainable scope"
                        ),
                    },
                },
                "thresholds": (
                    {
                        "validation_positive_prompt_margins_global_minimum": 12,
                        "validation_mean_margin_minimum": 0.02,
                        "validation_nll_ratio_maximum": 0.85,
                        "heldout_positive_prompt_margins_global_minimum": 10,
                        "heldout_mean_margin_strictly_positive": True,
                        "heldout_nll_ratio_maximum": 1.0,
                        "validation_physical_set_loss_ratio_maximum": 0.80,
                        "heldout_physical_set_loss_ratio_maximum": 1.0,
                        "physical_prompt_drift_max_abs": 1.0e-5,
                        "shared_row_gauge_required": True,
                        "frozen_action_state_hash_unchanged": True,
                    }
                    if representation_mode
                    else {
                        "validation_positive_prompt_margins": 2,
                        "validation_mean_margin_minimum": 0.10,
                        "validation_nll_ratio_maximum": 0.70,
                        "heldout_positive_prompt_margins_global_minimum": 3,
                        "heldout_mean_margin_strictly_positive": True,
                    }
                ),
                "engineering_self_checks": {
                    "matched_row_permutation_max_abs_error": 1.0e-6,
                    "scientific_evidence": False,
                    "purpose": "metric implementation consistency only",
                },
                "scene_level_robustness": (
                    {
                        partition: _scene_level_robustness(
                            rank_reports,
                            partition=partition,
                            seed=args.seed + offset,
                        )
                        for offset, partition in enumerate(("validation", "heldout"))
                    }
                    if representation_mode
                    else None
                ),
                "checkpoint": {
                    "requested": args.output_checkpoint is not None,
                    "path": (
                        None
                        if args.output_checkpoint is None
                        else str(args.output_checkpoint.absolute())
                    ),
                    "format": "lingbot-fsdp2-dcp-model-only",
                    "optimizer_saved": False,
                    "extra_state_saved": False,
                    "stage_transfer_not_exact_resume": True,
                    "publication_status": (
                        "PENDING" if args.output_checkpoint is not None else "NOT_REQUESTED"
                    ),
                },
                "input_sha256": {
                    "execution_contract": _sha256(args.execution_contract),
                    "offline_labels": _sha256(args.offline_labels),
                    "dataset_manifest": _sha256(args.dataset_manifest),
                    "normalization": _sha256(args.norm_stats),
                    "physical_sidecar_manifest": sidecar.manifest_sha256,
                },
                "dataset_contract": dataset_contract,
                "patch_sha256": patch_report["patch_sha256"],
                "alignment_teacher_prune": alignment_teacher_prune,
                "rank_reports": rank_reports,
            }
            report["failures"] = (
                _computed_representation_failures(report)
                if representation_mode
                else _computed_failures(report)
            )
            report["status"] = "PASS" if not report["failures"] else "FAIL"
            outcome = [report["status"], report["failures"]]
        dist.broadcast_object_list(outcome, src=0)
        if outcome[0] != "PASS":
            if rank == 0:
                report["checkpoint"]["publication_status"] = "SCIENTIFIC_REJECTED"
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_text_durable_exclusive(
                    args.output,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
            dist.barrier()
            raise RuntimeError(f"LTOP G2 rejected: {outcome[1]}")
        if args.output_checkpoint is None:
            if rank == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_text_durable_exclusive(
                    args.output,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
            dist.barrier()
        else:
            checkpoint_root = args.output_checkpoint.parent
            staging_checkpoint = args.output_checkpoint.with_name(
                f".{args.output_checkpoint.name}.staging"
            )
            precheckpoint_error: list[str | None] = [None]
            if rank == 0:
                try:
                    checkpoint_root.mkdir(parents=True, exist_ok=True)
                    require_persistent_run_root(checkpoint_root)
                    for path in (args.output_checkpoint, staging_checkpoint):
                        if path.exists() or path.is_symlink():
                            raise FileExistsError(path)
                    require_checkpoint_write_capacity(checkpoint_root)
                except BaseException as error:
                    precheckpoint_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(precheckpoint_error, src=0)
            if precheckpoint_error[0] is not None:
                raise RuntimeError(
                    "LTOP G2 checkpoint preflight failed: "
                    f"{precheckpoint_error[0]}"
                )
            checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
            save_started = time.perf_counter()
            save_error: BaseException | None = None
            try:
                _distributed_rank_local_call(
                    action=lambda: checkpointer.save(
                        str(staging_checkpoint),
                        _stage_transfer_checkpoint_state(policy),
                        global_steps=None,
                    ),
                    phase="ltop-g2-representation-model-only-checkpoint-save",
                    rank=rank,
                    dist_module=dist,
                )
            except BaseException as error:
                save_error = error
            if save_error is not None:
                if rank == 0 and staging_checkpoint.is_dir():
                    shutil.rmtree(staging_checkpoint)
                dist.barrier()
                raise save_error
            save_duration_s = time.perf_counter() - save_started
            publish_error: list[str | None] = [None]
            if rank == 0:
                try:
                    report["checkpoint"]["save_duration_s"] = save_duration_s
                    report["checkpoint"]["publication_status"] = "PASS"
                    write_text_durable_exclusive(
                        staging_checkpoint / "ltop_g2_representation_report.json",
                        json.dumps(report, indent=2, sort_keys=True) + "\n",
                    )
                    _fsync_tree(staging_checkpoint)
                    os.replace(staging_checkpoint, args.output_checkpoint)
                    descriptor = os.open(
                        checkpoint_root,
                        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                    )
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                    try:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        write_text_durable_exclusive(
                            args.output,
                            json.dumps(report, indent=2, sort_keys=True) + "\n",
                        )
                    except BaseException:
                        os.replace(args.output_checkpoint, staging_checkpoint)
                        rollback_descriptor = os.open(
                            checkpoint_root,
                            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
                        )
                        try:
                            os.fsync(rollback_descriptor)
                        finally:
                            os.close(rollback_descriptor)
                        raise
                except BaseException as error:
                    publish_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(publish_error, src=0)
            if publish_error[0] is not None:
                raise RuntimeError(
                    "LTOP G2 checkpoint publication failed: "
                    f"{publish_error[0]}"
                )
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

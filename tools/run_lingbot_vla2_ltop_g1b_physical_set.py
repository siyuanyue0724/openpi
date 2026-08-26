#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Run the full-weight LTOP G1b physical-set prerequisite on LingBot/CALVIN.

This is an evaluation-only gate over the released observation host.  Each of
two ranks selects one real CALVIN observation and evaluates four frozen arms:

1. natural prompt and episode-address gauge,
2. an exact replay that establishes the numerical floor,
3. one consistent row/address permutation, and
4. the same observation with a frozen donor prompt.

The gate asks only whether task-free prior/posterior rows are non-collapsed,
whether an address-gauge change only permutes row identity, and whether prompt
content is unable to rewrite the physical rows.  It never samples actions,
optimizes parameters, or introduces a learned module.  Accelerator and
upstream imports stay inside ``main`` so metrics and report validation remain
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
from dataclasses import replace
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
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
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
        _git_output,
        _merge_qwen_config,
        _resolve_training_config,
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


G1B_WORLD_SIZE = 2
G1B_PHYSICAL_CAPACITY = 16
G1B_TASK_QUERY_COUNT = 4
G1B_PLAN_STEPS = 200
G1B_SCHEMA = "picf-next.ltop-g1b-physical-set-prerequisite.v1"
G1B_PROMPT_SWAP_SCHEMA = "picf-next.ltop-g1b-real-prompt-swap.v1"
G1B_COMPARISON_ID = "lingbot-vla2-ltop-g1b-physical-set-prerequisite"
G1B_ARCHITECTURE = "lingbot_task_query_object_value_read_v1"
G1B_EVAL_CONTRACT = {
    "use_compile": False,
    "attention_implementation": "eager",
    "vit_attn_implementation": "eager",
}
G1B_PARALLEL_CONTRACT = {
    "backend": "cpu:gloo,cuda:nccl",
    "dp_size": G1B_WORLD_SIZE,
    "dp_replicate_size": 1,
    "dp_shard_size": G1B_WORLD_SIZE,
    "tp_size": 1,
    "ep_size": 1,
    "pp_size": 1,
    "cp_size": 1,
    "ulysses_size": 1,
    "dp_mode": "fsdp2",
}
_OBSERVATION_FIELDS = (
    "image_grid_thw",
    "images",
    "img_masks",
    "lang_masks",
    "lang_tokens",
)
_SCENE_FIELDS = ("image_grid_thw", "images", "img_masks")


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _require_row_tensor(name: str, value: Any, *, torch_module: Any) -> None:
    if (
        not isinstance(value, torch_module.Tensor)
        or value.ndim != 4
        or min(value.shape) <= 0
        or not value.is_floating_point()
    ):
        raise ValueError(f"{name} must be floating [batch,layers,capacity,width]")
    if not torch_module.isfinite(value).all():
        raise ValueError(f"{name} contains NaN or infinity")


def _same_row_shape(name: str, *values: Any, torch_module: Any) -> None:
    for value in values:
        _require_row_tensor(name, value, torch_module=torch_module)
    if len({tuple(value.shape) for value in values}) != 1:
        raise ValueError(f"{name} tensors differ in shape")


def replay_noncollapse_metrics(
    reference: Any,
    repeat: Any,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Measure row non-collapse using only exact-replay uncertainty.

    If each endpoint can move by at most ``epsilon`` under exact replay, two
    rows are identifiable only when their measured distance is greater than
    ``2 * epsilon``.  This triangle-inequality bound is the gate threshold;
    there is no tuned tolerance.
    """

    _same_row_shape("replay rows", reference, repeat, torch_module=torch_module)
    ref = reference.detach().float()
    rep = repeat.detach().float()
    delta = rep - ref
    row_floor = torch_module.linalg.vector_norm(delta, dim=-1)
    absolute_floor = delta.abs().amax(dim=(-1, -2))
    centered = ref - ref.mean(dim=2, keepdim=True)
    centered_repeat = rep - rep.mean(dim=2, keepdim=True)
    centered_energy = torch_module.linalg.vector_norm(centered, dim=(-2, -1))
    centered_replay_floor = torch_module.linalg.vector_norm(
        centered_repeat - centered,
        dim=(-2, -1),
    )
    pairwise = torch_module.cdist(ref, ref, p=2)
    capacity = ref.shape[2]
    upper = torch_module.triu(
        torch_module.ones(
            (capacity, capacity),
            dtype=torch_module.bool,
            device=ref.device,
        ),
        diagonal=1,
    )
    per_layer: list[dict[str, Any]] = []
    for batch_index in range(ref.shape[0]):
        for layer_index in range(ref.shape[1]):
            replay_row_l2_floor = float(row_floor[batch_index, layer_index].max().item())
            identifiability_threshold = 2.0 * replay_row_l2_floor
            distances = pairwise[batch_index, layer_index][upper]
            stable_pairs = distances > identifiability_threshold
            per_layer.append(
                {
                    "batch_index": batch_index,
                    "layer_index": layer_index,
                    "replay_max_abs_floor": float(absolute_floor[batch_index, layer_index].item()),
                    "replay_max_row_l2_floor": replay_row_l2_floor,
                    "identifiability_threshold_l2": identifiability_threshold,
                    "maximum_pair_l2": float(distances.max().item()),
                    "stable_distinct_pair_count": int(stable_pairs.sum().item()),
                    "pair_count": int(distances.numel()),
                    "centered_frobenius_energy": float(
                        centered_energy[batch_index, layer_index].item()
                    ),
                    "centered_replay_frobenius_floor": float(
                        centered_replay_floor[batch_index, layer_index].item()
                    ),
                    "noncollapsed": bool(stable_pairs.any().item()),
                }
            )
    return {
        "derivation": "pair_l2 > 2 * exact_replay_max_row_l2",
        "shape": list(ref.shape),
        "replay_max_abs_floor": float(delta.abs().max().item()),
        "replay_max_row_l2_floor": float(row_floor.max().item()),
        "per_layer": per_layer,
        "all_layers_noncollapsed": all(item["noncollapsed"] for item in per_layer),
    }


def permutation_equivariance_metrics(
    reference: Any,
    repeat: Any,
    permuted: Any,
    row_permutation: Any,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Compare a consistently re-addressed arm to the known row permutation."""

    _same_row_shape(
        "permutation rows",
        reference,
        repeat,
        permuted,
        torch_module=torch_module,
    )
    capacity = reference.shape[2]
    if (
        not isinstance(row_permutation, torch_module.Tensor)
        or row_permutation.ndim != 1
        or row_permutation.dtype != torch_module.long
        or row_permutation.shape[0] != capacity
    ):
        raise ValueError("row permutation must be long [capacity]")
    expected_indices = torch_module.arange(capacity, device=row_permutation.device)
    if not torch_module.equal(row_permutation.sort().values, expected_indices):
        raise ValueError("row permutation must contain every row exactly once")
    ref = reference.detach().float()
    rep = repeat.detach().float()
    observed = permuted.detach().float()
    indices = row_permutation.to(ref.device)
    expected = ref.index_select(2, indices)
    replay_expected = rep.index_select(2, indices)
    replay_delta = replay_expected - expected
    equivariance_delta = observed - expected
    replay_row_floor = torch_module.linalg.vector_norm(replay_delta, dim=-1)
    observed_row_error = torch_module.linalg.vector_norm(equivariance_delta, dim=-1)

    reference_pairwise = torch_module.cdist(ref, ref, p=2)
    repeat_pairwise = torch_module.cdist(rep, rep, p=2)
    expected_pairwise = reference_pairwise.index_select(2, indices).index_select(3, indices)
    replay_pairwise = repeat_pairwise.index_select(2, indices).index_select(3, indices)
    observed_pairwise = torch_module.cdist(observed, observed, p=2)
    replay_pairwise_floor = (replay_pairwise - expected_pairwise).abs()
    observed_pairwise_error = (observed_pairwise - expected_pairwise).abs()

    per_layer: list[dict[str, Any]] = []
    for batch_index in range(ref.shape[0]):
        for layer_index in range(ref.shape[1]):
            replay_abs = float(replay_delta[batch_index, layer_index].abs().max().item())
            observed_abs = float(equivariance_delta[batch_index, layer_index].abs().max().item())
            replay_l2 = float(replay_row_floor[batch_index, layer_index].max().item())
            observed_l2 = float(observed_row_error[batch_index, layer_index].max().item())
            replay_set = float(replay_pairwise_floor[batch_index, layer_index].max().item())
            observed_set = float(observed_pairwise_error[batch_index, layer_index].max().item())
            per_layer.append(
                {
                    "batch_index": batch_index,
                    "layer_index": layer_index,
                    "replay_max_abs_floor": replay_abs,
                    "equivariance_max_abs_error": observed_abs,
                    "replay_max_row_l2_floor": replay_l2,
                    "equivariance_max_row_l2_error": observed_l2,
                    "replay_pairwise_set_floor": replay_set,
                    "pairwise_set_error": observed_set,
                    "row_identity_permuted_only": (
                        observed_abs <= replay_abs
                        and observed_l2 <= replay_l2
                        and observed_set <= replay_set
                    ),
                }
            )
    return {
        "derivation": "permuted-arm error <= exact-replay error in the same metric",
        "row_permutation": [int(value) for value in row_permutation.tolist()],
        "per_layer": per_layer,
        "all_layers_equivariant": all(item["row_identity_permuted_only"] for item in per_layer),
    }


def normalized_permutation_recovery_metrics(
    reference: Any,
    permuted: Any,
    row_permutation: Any,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Measure row identity in the normalized state actually consumed by LingBot.

    Persistent rows enter each decoder layer through its pre-norm boundary.
    Raw residual magnitude is therefore diagnostic but is not itself the
    behavioral state seen by Q/K/V.  This companion metric keeps the strict raw
    gate unchanged and asks the threshold-free question that matters for row
    identity: after RMS normalization, is every permuted row still nearest to
    its known source row with a positive nearest-neighbour margin?
    """

    _same_row_shape(
        "normalized permutation rows",
        reference,
        permuted,
        torch_module=torch_module,
    )
    capacity = reference.shape[2]
    if (
        not isinstance(row_permutation, torch_module.Tensor)
        or row_permutation.ndim != 1
        or row_permutation.dtype != torch_module.long
        or row_permutation.shape[0] != capacity
    ):
        raise ValueError("row permutation must be long [capacity]")
    expected_indices = torch_module.arange(capacity, device=row_permutation.device)
    if not torch_module.equal(row_permutation.sort().values, expected_indices):
        raise ValueError("row permutation must contain every row exactly once")

    def rms_normalize(rows: Any) -> Any:
        value = rows.detach().float()
        scale = value.square().mean(dim=-1, keepdim=True).clamp_min(1.0e-12).rsqrt()
        return value * scale

    ref = rms_normalize(reference)
    observed = rms_normalize(permuted)
    expected = row_permutation.to(ref.device)
    distances = torch_module.cdist(observed, ref, p=2)
    per_layer: list[dict[str, Any]] = []
    row = torch_module.arange(capacity, device=ref.device)
    for batch_index in range(ref.shape[0]):
        for layer_index in range(ref.shape[1]):
            layer_distances = distances[batch_index, layer_index]
            predicted = layer_distances.argmin(dim=-1)
            expected_distance = layer_distances[row, expected]
            masked = layer_distances.clone()
            masked[row, expected] = float("inf")
            nearest_competitor = masked.min(dim=-1).values
            margin = nearest_competitor - expected_distance
            correct = predicted == expected
            per_layer.append(
                {
                    "batch_index": batch_index,
                    "layer_index": layer_index,
                    "correct_rows": int(correct.sum().item()),
                    "row_count": capacity,
                    "maximum_expected_l2": float(expected_distance.max().item()),
                    "minimum_competitor_margin_l2": float(margin.min().item()),
                    "all_rows_recovered": bool(correct.all().item() and (margin > 0).all().item()),
                }
            )
    return {
        "derivation": (
            "RMS-normalized permuted row has its known source as the unique nearest row"
        ),
        "row_permutation": [int(value) for value in row_permutation.tolist()],
        "per_layer": per_layer,
        "all_layers_recovered": all(item["all_rows_recovered"] for item in per_layer),
    }


def prompt_invariance_metrics(
    reference: Any,
    repeat: Any,
    prompt_swapped: Any,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Require prompt-swap drift to remain inside exact-replay uncertainty."""

    _same_row_shape(
        "prompt rows",
        reference,
        repeat,
        prompt_swapped,
        torch_module=torch_module,
    )
    ref = reference.detach().float()
    rep = repeat.detach().float()
    swapped = prompt_swapped.detach().float()
    replay_delta = rep - ref
    prompt_delta = swapped - ref
    replay_row_floor = torch_module.linalg.vector_norm(replay_delta, dim=-1)
    prompt_row_error = torch_module.linalg.vector_norm(prompt_delta, dim=-1)
    per_layer: list[dict[str, Any]] = []
    for batch_index in range(ref.shape[0]):
        for layer_index in range(ref.shape[1]):
            replay_abs = float(replay_delta[batch_index, layer_index].abs().max().item())
            prompt_abs = float(prompt_delta[batch_index, layer_index].abs().max().item())
            replay_l2 = float(replay_row_floor[batch_index, layer_index].max().item())
            prompt_l2 = float(prompt_row_error[batch_index, layer_index].max().item())
            per_layer.append(
                {
                    "batch_index": batch_index,
                    "layer_index": layer_index,
                    "replay_max_abs_floor": replay_abs,
                    "prompt_max_abs_error": prompt_abs,
                    "replay_max_row_l2_floor": replay_l2,
                    "prompt_max_row_l2_error": prompt_l2,
                    "prompt_invariant": prompt_abs <= replay_abs and prompt_l2 <= replay_l2,
                }
            )
    return {
        "derivation": "prompt-swap error <= exact-replay error in the same metric",
        "per_layer": per_layer,
        "all_layers_prompt_invariant": all(item["prompt_invariant"] for item in per_layer),
    }


def _finite_nonnegative(value: object, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) < 0.0
    ):
        raise ValueError(f"LTOP G1b {name} must be finite and non-negative")
    return float(value)


def _validate_noncollapse_section(section: object, *, name: str) -> bool:
    if not isinstance(section, dict):
        raise ValueError(f"LTOP G1b {name} must be an object")
    if section.get("derivation") != "pair_l2 > 2 * exact_replay_max_row_l2":
        raise ValueError(f"LTOP G1b {name} changed its threshold derivation")
    per_layer = section.get("per_layer")
    if not isinstance(per_layer, list) or not per_layer:
        raise ValueError(f"LTOP G1b {name} has no per-layer evidence")
    recomputed: list[bool] = []
    for index, item in enumerate(per_layer):
        if not isinstance(item, dict):
            raise ValueError(f"LTOP G1b {name} layer {index} is malformed")
        floor = _finite_nonnegative(
            item.get("replay_max_row_l2_floor"),
            name=f"{name}.per_layer[{index}].replay_max_row_l2_floor",
        )
        threshold = _finite_nonnegative(
            item.get("identifiability_threshold_l2"),
            name=f"{name}.per_layer[{index}].identifiability_threshold_l2",
        )
        maximum_pair = _finite_nonnegative(
            item.get("maximum_pair_l2"),
            name=f"{name}.per_layer[{index}].maximum_pair_l2",
        )
        pair_count = item.get("pair_count")
        stable_count = item.get("stable_distinct_pair_count")
        if (
            isinstance(pair_count, bool)
            or not isinstance(pair_count, int)
            or pair_count <= 0
            or isinstance(stable_count, bool)
            or not isinstance(stable_count, int)
            or not 0 <= stable_count <= pair_count
        ):
            raise ValueError(f"LTOP G1b {name} layer {index} has invalid pair counts")
        if threshold != 2.0 * floor:
            raise ValueError(f"LTOP G1b {name} layer {index} threshold is not replay-derived")
        expected = stable_count > 0 and maximum_pair > threshold
        if item.get("noncollapsed") is not expected:
            raise ValueError(f"LTOP G1b {name} layer {index} contradicts noncollapse metrics")
        recomputed.append(expected)
    expected_all = all(recomputed)
    if section.get("all_layers_noncollapsed") is not expected_all:
        raise ValueError(f"LTOP G1b {name} contradicts its per-layer noncollapse evidence")
    return expected_all


def _validate_permutation_section(section: object, *, name: str) -> bool:
    if not isinstance(section, dict):
        raise ValueError(f"LTOP G1b {name} must be an object")
    if section.get("derivation") != ("permuted-arm error <= exact-replay error in the same metric"):
        raise ValueError(f"LTOP G1b {name} changed its threshold derivation")
    permutation = section.get("row_permutation")
    if (
        not isinstance(permutation, list)
        or len(permutation) != G1B_PHYSICAL_CAPACITY
        or any(isinstance(value, bool) or not isinstance(value, int) for value in permutation)
        or sorted(permutation) != list(range(G1B_PHYSICAL_CAPACITY))
    ):
        raise ValueError(f"LTOP G1b {name} has an invalid row permutation")
    per_layer = section.get("per_layer")
    if not isinstance(per_layer, list) or not per_layer:
        raise ValueError(f"LTOP G1b {name} has no per-layer evidence")
    recomputed: list[bool] = []
    for index, item in enumerate(per_layer):
        if not isinstance(item, dict):
            raise ValueError(f"LTOP G1b {name} layer {index} is malformed")
        replay_abs = _finite_nonnegative(
            item.get("replay_max_abs_floor"),
            name=f"{name}.per_layer[{index}].replay_max_abs_floor",
        )
        observed_abs = _finite_nonnegative(
            item.get("equivariance_max_abs_error"),
            name=f"{name}.per_layer[{index}].equivariance_max_abs_error",
        )
        replay_l2 = _finite_nonnegative(
            item.get("replay_max_row_l2_floor"),
            name=f"{name}.per_layer[{index}].replay_max_row_l2_floor",
        )
        observed_l2 = _finite_nonnegative(
            item.get("equivariance_max_row_l2_error"),
            name=f"{name}.per_layer[{index}].equivariance_max_row_l2_error",
        )
        replay_set = _finite_nonnegative(
            item.get("replay_pairwise_set_floor"),
            name=f"{name}.per_layer[{index}].replay_pairwise_set_floor",
        )
        observed_set = _finite_nonnegative(
            item.get("pairwise_set_error"),
            name=f"{name}.per_layer[{index}].pairwise_set_error",
        )
        expected = (
            observed_abs <= replay_abs and observed_l2 <= replay_l2 and observed_set <= replay_set
        )
        if item.get("row_identity_permuted_only") is not expected:
            raise ValueError(f"LTOP G1b {name} layer {index} contradicts permutation metrics")
        recomputed.append(expected)
    expected_all = all(recomputed)
    if section.get("all_layers_equivariant") is not expected_all:
        raise ValueError(f"LTOP G1b {name} contradicts its per-layer permutation evidence")
    return expected_all


def _validate_prompt_section(section: object, *, name: str) -> bool:
    if not isinstance(section, dict):
        raise ValueError(f"LTOP G1b {name} must be an object")
    if section.get("derivation") != ("prompt-swap error <= exact-replay error in the same metric"):
        raise ValueError(f"LTOP G1b {name} changed its threshold derivation")
    per_layer = section.get("per_layer")
    if not isinstance(per_layer, list) or not per_layer:
        raise ValueError(f"LTOP G1b {name} has no per-layer evidence")
    recomputed: list[bool] = []
    for index, item in enumerate(per_layer):
        if not isinstance(item, dict):
            raise ValueError(f"LTOP G1b {name} layer {index} is malformed")
        replay_abs = _finite_nonnegative(
            item.get("replay_max_abs_floor"),
            name=f"{name}.per_layer[{index}].replay_max_abs_floor",
        )
        prompt_abs = _finite_nonnegative(
            item.get("prompt_max_abs_error"),
            name=f"{name}.per_layer[{index}].prompt_max_abs_error",
        )
        replay_l2 = _finite_nonnegative(
            item.get("replay_max_row_l2_floor"),
            name=f"{name}.per_layer[{index}].replay_max_row_l2_floor",
        )
        prompt_l2 = _finite_nonnegative(
            item.get("prompt_max_row_l2_error"),
            name=f"{name}.per_layer[{index}].prompt_max_row_l2_error",
        )
        expected = prompt_abs <= replay_abs and prompt_l2 <= replay_l2
        if item.get("prompt_invariant") is not expected:
            raise ValueError(f"LTOP G1b {name} layer {index} contradicts prompt metrics")
        recomputed.append(expected)
    expected_all = all(recomputed)
    if section.get("all_layers_prompt_invariant") is not expected_all:
        raise ValueError(f"LTOP G1b {name} contradicts its per-layer prompt evidence")
    return expected_all


def _metric_failures(rank_report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    rank = rank_report.get("rank", "?")
    required_true = (
        "same_scene_non_language_exact",
        "prompt_changed",
        "address_permutation_consistent",
        "contexts_finalized",
        "rows_finite",
    )
    failures.extend(
        f"rank {rank}: {name} is false"
        for name in required_true
        if rank_report.get(name) is not True
    )
    for surface in ("prior", "posterior"):
        metrics = rank_report.get(surface)
        if not isinstance(metrics, dict):
            failures.append(f"rank {rank}: missing {surface} metrics")
            continue
        checks = (
            (
                "noncollapse",
                "all_layers_noncollapsed",
                _validate_noncollapse_section,
            ),
            (
                "permutation",
                "all_layers_equivariant",
                _validate_permutation_section,
            ),
            (
                "prompt_invariance",
                "all_layers_prompt_invariant",
                _validate_prompt_section,
            ),
        )
        for section_name, field, validator in checks:
            passed = validator(
                metrics.get(section_name),
                name=f"rank[{rank}].{surface}.{section_name}",
            )
            if not passed:
                failures.append(f"rank {rank}: {surface}.{section_name}.{field} is false")
    return failures


def validate_ltop_g1b_report(report: object) -> dict[str, Any]:
    """Recompute the strict G1b verdict before durable publication."""

    if not isinstance(report, dict):
        raise ValueError("LTOP G1b report must be a JSON object")
    value = cast(dict[str, Any], report)
    if value.get("schema") != G1B_SCHEMA:
        raise ValueError("LTOP G1b report uses another schema")
    if value.get("world_size") != G1B_WORLD_SIZE:
        raise ValueError("LTOP G1b report has the wrong world size")
    if value.get("architecture_identity") != G1B_ARCHITECTURE:
        raise ValueError("LTOP G1b report has the wrong architecture")
    rank_reports = value.get("rank_reports")
    if not isinstance(rank_reports, list) or len(rank_reports) != G1B_WORLD_SIZE:
        raise ValueError("LTOP G1b report must contain exactly two rank reports")
    observed_ranks = {item.get("rank") for item in rank_reports if isinstance(item, dict)}
    if observed_ranks != set(range(G1B_WORLD_SIZE)):
        raise ValueError("LTOP G1b report omitted or duplicated a rank")
    failures: list[str] = []
    for item in sorted(rank_reports, key=lambda entry: entry["rank"]):
        failures.extend(_metric_failures(item))
    manifest = value.get("parameter_manifest")
    if not isinstance(manifest, dict) or manifest.get("active_trainable_numel") != 0:
        failures.append("G1b policy had active trainable parameters")
    declared = value.get("failures")
    if declared != failures:
        raise ValueError("LTOP G1b declared failures differ from recomputed failures")
    expected_status = "PASS" if not failures else "FAIL"
    if value.get("status") != expected_status:
        raise ValueError("LTOP G1b status differs from its strict evidence")
    return value


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
    parser.add_argument("--capacity", type=int, default=G1B_PHYSICAL_CAPACITY)
    parser.add_argument("--task-query-count", type=int, default=G1B_TASK_QUERY_COUNT)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument("--plan-steps", type=int, default=G1B_PLAN_STEPS)
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
    }
    missing = [name for name, path in required.items() if path is None or not path.exists()]
    if missing:
        raise FileNotFoundError(f"LTOP G1b required paths are absent: {missing}")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for name in (
        "seed",
        "capacity",
        "task_query_count",
        "maximum_control_tokens",
        "plan_steps",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"LTOP G1b {name} must be a positive integer")
    if args.capacity != G1B_PHYSICAL_CAPACITY:
        raise ValueError("LTOP G1b requires the 16-row physical contract")
    if args.task_query_count != G1B_TASK_QUERY_COUNT:
        raise ValueError("LTOP G1b requires four task-query/object-read rows")
    if args.plan_steps < G1B_WORLD_SIZE:
        raise ValueError("LTOP G1b prompt matching requires at least two plan steps")


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


def _episode_ids(episode_keys: tuple[str, ...], *, torch_module: Any, device: Any) -> Any:
    values = [
        int.from_bytes(
            hashlib.sha256(
                json.dumps(
                    {"comparison_id": G1B_COMPARISON_ID, "episode_key": key},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("ascii")
            ).digest()[:8],
            "big",
        )
        >> 1
        for key in episode_keys
    ]
    return torch_module.tensor(values, dtype=torch_module.long, device=device)


def _sample_task_instruction(dataset: Any, sample_key: str) -> tuple[str, str]:
    locator = dataset.locator_by_key(sample_key)
    try:
        segment = dataset.index.segments[locator.segment_index]
    except (AttributeError, IndexError) as error:
        raise ValueError("G1b prompt donor has no immutable language segment") from error
    task_key = dataset.task_key_by_key(sample_key)
    instruction = segment.instruction
    if segment.task_key != task_key or not isinstance(instruction, str) or not instruction:
        raise ValueError("G1b prompt donor differs from immutable CALVIN language")
    return task_key, instruction


def _transition_identity(transition: Any) -> tuple[str, str, str]:
    return (
        transition.lane_id,
        transition.episode_instance_id,
        transition.sample.sample_key,
    )


def _build_prompt_swap_plan(stream_plan: Any, dataset: Any) -> dict[str, Any]:
    """Select real, different prompts without imposing G1T target matching."""

    catalog: list[dict[str, Any]] = []
    for optimizer_step in range(stream_plan.total_steps):
        transitions = stream_plan.global_batch(optimizer_step).transitions
        for transition in transitions:
            task_key, instruction = _sample_task_instruction(
                dataset,
                transition.sample.sample_key,
            )
            catalog.append(
                {
                    "optimizer_step": optimizer_step,
                    "lane_id": transition.lane_id,
                    "episode_instance_id": transition.episode_instance_id,
                    "sample_key": transition.sample.sample_key,
                    "task_key": task_key,
                    "instruction": instruction,
                    "instruction_sha256": hashlib.sha256(instruction.encode("utf-8")).hexdigest(),
                }
            )
    if len(catalog) < 2:
        raise ValueError("G1b prompt swap requires at least two frozen stream slots")

    for evaluation_step in range(stream_plan.total_steps):
        recipients = stream_plan.global_batch(evaluation_step).transitions
        if len(recipients) != G1B_WORLD_SIZE:
            continue
        slots: list[dict[str, Any]] = []
        for recipient in recipients:
            natural_task_key, natural_instruction = _sample_task_instruction(
                dataset,
                recipient.sample.sample_key,
            )
            natural_sha256 = hashlib.sha256(natural_instruction.encode("utf-8")).hexdigest()
            candidates = [
                donor for donor in catalog if donor["instruction_sha256"] != natural_sha256
            ]
            if not candidates:
                break
            recipient_identity = _transition_identity(recipient)
            donor = min(
                candidates,
                key=lambda item: hashlib.sha256(
                    json.dumps(
                        {
                            "comparison_id": G1B_COMPARISON_ID,
                            "recipient": recipient_identity,
                            "donor": (
                                item["lane_id"],
                                item["episode_instance_id"],
                                item["sample_key"],
                            ),
                            "stream_plan_sha256": stream_plan.plan_sha256,
                        },
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).digest(),
            )
            slots.append(
                {
                    "recipient_lane_id": recipient.lane_id,
                    "recipient_episode_instance_id": recipient.episode_instance_id,
                    "recipient_sample_key": recipient.sample.sample_key,
                    "recipient_task_key": natural_task_key,
                    "recipient_instruction_sha256": natural_sha256,
                    "donor_optimizer_step": donor["optimizer_step"],
                    "donor_lane_id": donor["lane_id"],
                    "donor_episode_instance_id": donor["episode_instance_id"],
                    "donor_sample_key": donor["sample_key"],
                    "donor_task_key": donor["task_key"],
                    "donor_instruction": donor["instruction"],
                    "donor_instruction_sha256": donor["instruction_sha256"],
                }
            )
        if len(slots) != G1B_WORLD_SIZE:
            continue
        payload: dict[str, Any] = {
            "schema": G1B_PROMPT_SWAP_SCHEMA,
            "selection": "sha256-order over real frozen-stream prompts with unequal language",
            "comparison_id": G1B_COMPARISON_ID,
            "stream_plan_sha256": stream_plan.plan_sha256,
            "evaluation_step": evaluation_step,
            "slots": slots,
        }
        payload["artifact_sha256"] = _canonical_json_sha256(payload)
        return payload
    raise ValueError("frozen CALVIN stream contains no two-rank real prompt swap")


def _apply_prompt_swap(planned: Any, prompt_plan: dict[str, Any], dataset: Any) -> Any:
    """Replace only real prompt language and its loss-side task key."""

    if planned.task_intervention_sha256 is not None:
        raise ValueError("G1b prompt swap may be applied only once")
    if planned.plan_sha256 != prompt_plan["stream_plan_sha256"]:
        raise ValueError("G1b prompt plan and planned batch use different streams")
    if planned.plan_microbatch.optimizer_step != prompt_plan["evaluation_step"]:
        raise ValueError("G1b prompt plan was applied at another optimizer step")
    slots = {
        (
            item["recipient_lane_id"],
            item["recipient_episode_instance_id"],
            item["recipient_sample_key"],
        ): item
        for item in prompt_plan["slots"]
    }
    host_items: list[dict[str, Any]] = []
    requests = []
    for transition, host_item, request in zip(
        planned.plan_microbatch.transitions,
        planned.training.host_items,
        planned.training.structural_target_requests,
        strict=True,
    ):
        slot = slots.get(_transition_identity(transition))
        if slot is None:
            raise ValueError("G1b prompt plan omitted the planned transition")
        natural_task_key, natural_instruction = _sample_task_instruction(
            dataset,
            transition.sample.sample_key,
        )
        if (
            host_item["task"] != natural_instruction
            or request.task_key != natural_task_key
            or natural_task_key != slot["recipient_task_key"]
            or hashlib.sha256(natural_instruction.encode("utf-8")).hexdigest()
            != slot["recipient_instruction_sha256"]
        ):
            raise ValueError("G1b natural prompt differs from its immutable source")
        donor_task_key, donor_instruction = _sample_task_instruction(
            dataset,
            slot["donor_sample_key"],
        )
        if (
            donor_task_key != slot["donor_task_key"]
            or donor_instruction != slot["donor_instruction"]
            or hashlib.sha256(donor_instruction.encode("utf-8")).hexdigest()
            != slot["donor_instruction_sha256"]
            or donor_instruction == natural_instruction
        ):
            raise ValueError("G1b donor prompt differs from its frozen selection")
        replaced_item = dict(host_item)
        replaced_item["task"] = donor_instruction
        host_items.append(replaced_item)
        requests.append(replace(request, task_key=donor_task_key))
    training = replace(
        planned.training,
        host_items=tuple(host_items),
        structural_target_requests=tuple(requests),
    )
    return replace(
        planned,
        training=training,
        task_intervention_sha256=prompt_plan["artifact_sha256"],
    )


def _nonidentity_row_permutation(
    capacity: int,
    rank: int,
    *,
    torch_module: Any,
    device: Any,
) -> Any:
    shift = (rank + 1) % capacity
    if shift == 0:
        shift = 1
    return torch_module.roll(
        torch_module.arange(capacity, dtype=torch_module.long, device=device),
        shifts=shift,
    )


def _tensor_mapping_sha256(values: dict[str, Any]) -> tuple[dict[str, str], str]:
    manifest = {name: _tensor_sha256(values[name]) for name in sorted(values)}
    return manifest, _canonical_json_sha256(manifest)


def task_address_attention_metrics(
    natural: Any,
    repeat: Any,
    permuted: Any,
    prompt: Any,
    row_permutation: Any,
    *,
    torch_module: Any,
) -> dict[str, Any]:
    """Summarize real OBJECT_READ-to-row attention without semantic labels."""

    expected_shape = natural.shape
    if (
        natural.ndim != 4
        or repeat.shape != expected_shape
        or permuted.shape != expected_shape
        or prompt.shape != expected_shape
        or row_permutation.shape != (natural.shape[-1],)
    ):
        raise ValueError("LTOP task-address attention receipts have incompatible shapes")
    if any(
        value.device != natural.device or not value.is_floating_point()
        for value in (repeat, permuted, prompt)
    ):
        raise ValueError("LTOP task-address attention receipts changed device or dtype")
    if row_permutation.dtype != torch_module.long or row_permutation.device != natural.device:
        raise ValueError("LTOP task-address attention permutation is incompatible")
    if any(not torch_module.isfinite(value).all() for value in (natural, repeat, permuted, prompt)):
        raise ValueError("LTOP task-address attention receipt is non-finite")

    def conditional(value: Any) -> Any:
        denominator = value.sum(dim=-1, keepdim=True)
        if not (denominator > 0).all():
            raise ValueError("LTOP OBJECT_READ assigned no mass to physical rows")
        return value / denominator

    natural_distribution = conditional(natural.float())
    repeat_distribution = conditional(repeat.float())
    permuted_distribution = conditional(permuted.float())
    prompt_distribution = conditional(prompt.float())
    expected_permuted = natural_distribution.index_select(-1, row_permutation)
    replay_error = (repeat_distribution - natural_distribution).abs()
    permutation_error = (permuted_distribution - expected_permuted).abs()
    prompt_error = (prompt_distribution - natural_distribution).abs()
    per_layer: list[dict[str, Any]] = []
    for layer_index in range(natural.shape[1]):
        replay_layer = replay_error[:, layer_index]
        permutation_layer = permutation_error[:, layer_index]
        prompt_layer = prompt_error[:, layer_index]
        replay_floor = float(replay_layer.max().item())
        observed_permutation = float(permutation_layer.max().item())
        observed_prompt = float(prompt_layer.max().item())
        per_layer.append(
            {
                "layer_index": layer_index,
                "replay_max_abs_floor": replay_floor,
                "permutation_max_abs_error": observed_permutation,
                "permutation_within_replay": observed_permutation <= replay_floor,
                "prompt_max_abs_shift": observed_prompt,
                "prompt_mean_l1_shift": float(
                    prompt_layer.sum(dim=-1).mean().item()
                ),
                "prompt_exceeds_replay": observed_prompt > replay_floor,
                "natural_top_rows": [
                    [int(row) for row in batch_rows]
                    for batch_rows in natural_distribution[:, layer_index]
                    .argmax(dim=-1)
                    .tolist()
                ],
                "prompt_top_rows": [
                    [int(row) for row in batch_rows]
                    for batch_rows in prompt_distribution[:, layer_index]
                    .argmax(dim=-1)
                    .tolist()
                ],
            }
        )
    final_natural = natural_distribution[:, -1]
    final_prompt = prompt_distribution[:, -1]
    final_cosine = torch_module.nn.functional.cosine_similarity(
        final_natural.flatten(1),
        final_prompt.flatten(1),
        dim=-1,
    )
    return {
        "derivation": (
            "real post-MRoPE eager-attention Q/K mass over paired "
            "memory+PRIOR+POSTERIOR carriers, normalized across physical rows"
        ),
        "shape": list(natural.shape),
        "all_layers_permutation_within_replay": all(
            item["permutation_within_replay"] for item in per_layer
        ),
        "prompt_responsive_layer_count": sum(
            item["prompt_exceeds_replay"] for item in per_layer
        ),
        "final_prompt_cosine": [float(value) for value in final_cosine.tolist()],
        "final_natural_distribution": final_natural.tolist(),
        "final_prompt_distribution": final_prompt.tolist(),
        "per_layer": per_layer,
    }


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
    bootstrap_rank = int(os.environ.get("RANK", "0"))
    asset_validation: dict[str, object] | None = None
    if bootstrap_rank == 0:
        asset_validation = {
            "checkpoint": validate_checkpoint(args.checkpoint_dir),
            "processor": validate_processor(args.processor_dir),
        }

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
    from picf_next.lingbot_native.host import (
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        ObjectReadActionIntervention,
        install_lingbot_native_graph,
        native_context_from_prior_trace,
    )
    from picf_next.lingbot_native.state import (
        AddressedLayerwisePosteriorState,
        AddressedLayerwisePriorTrace,
    )
    from picf_next.lingbot_native.task_address_receipt import (
        task_address_attention_receipt,
    )
    from picf_next.lingbot_native.training import (
        run_native_policy_observation_diagnostic_forward,
    )

    if int(os.environ.get("WORLD_SIZE", "0")) != G1B_WORLD_SIZE:
        raise RuntimeError("LTOP G1b must run under torchrun with exactly two processes")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend=G1B_PARALLEL_CONTRACT["backend"])
    try:
        asset_validation_box: list[object] = [asset_validation]
        dist.broadcast_object_list(asset_validation_box, src=0)
        asset_validation = cast(dict[str, object], asset_validation_box[0])
        if set(asset_validation) != {"checkpoint", "processor"}:
            raise RuntimeError("LTOP G1b asset validation receipt is incomplete")
        if torch.cuda.device_count() != G1B_WORLD_SIZE:
            raise RuntimeError("LTOP G1b process sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("LTOP G1b requires two A100 devices with at least 39 GiB each")
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        torch.backends.cudnn.benchmark = False
        init_parallel_state(
            **{name: value for name, value in G1B_PARALLEL_CONTRACT.items() if name != "backend"}
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
            raise ValueError("LTOP G1b CALVIN manifest and normalization differ")
        dataset_contract = {
            "status": "PASS",
            "manifest_sha256": _sha256(args.dataset_manifest),
            "normalization_sha256": _sha256(args.norm_stats),
            "validation": validate_dataset_runtime_binding(
                manifest,
                args.dataset_split,
                dataset_id=norm_source["dataset_id"],
                dataset_revision=norm_source["dataset_revision"],
                split_name=args.dataset_split.name,
            ),
        }

        training = load_lingbot_training_config(args.training_config)
        merged, data_mapping = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=10,
        )
        merged.update(G1B_EVAL_CONTRACT)
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
        config.num_steps = 10

        moe_backend = select_lingbot_deterministic_moe_backend(
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
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            task_query_count=args.task_query_count,
            architecture_identity=G1B_ARCHITECTURE,
        )
        graph = LingBotNativeGraph(
            graph_config,
            device=device,
            dtype=torch.bfloat16,
        )
        install_lingbot_native_graph(policy, graph)
        policy.train()
        graph.eval()
        policy.requires_grad_(False)
        joint_host = policy.model.qwenvl_with_expert
        original_attention_interface = joint_host.attention_interface
        active_attention_capture: dict[str, Any] | None = None

        def attention_interface_with_receipt(
            query_states: Any,
            key_states: Any,
            value_states: Any,
            attention_mask: Any,
        ) -> Any:
            if active_attention_capture is not None:
                if "object_read_slice" not in active_attention_capture:
                    context = active_attention_capture["context"]
                    native_valid = context.native_valid
                    native_roles = context.native_roles
                    if native_valid is None or native_roles is None:
                        raise RuntimeError(
                            "LTOP attention call preceded native metadata binding"
                        )
                    original_prefix_count = native_valid.shape[1]
                    language_slice = graph._instruction_span(native_roles)
                    task_text_count = language_slice.stop - language_slice.start
                    prior_start = (
                        original_prefix_count
                        + task_text_count
                        + context.controls.token_count
                    )
                    prior_slice = slice(prior_start, prior_start + args.capacity)
                    posterior_slice = slice(
                        prior_slice.stop,
                        prior_slice.stop + args.capacity,
                    )
                    task_query_slice = slice(
                        posterior_slice.stop,
                        posterior_slice.stop + args.task_query_count,
                    )
                    active_attention_capture.update(
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
                    object_read_slice=active_attention_capture["object_read_slice"],
                    prior_slice=active_attention_capture["prior_slice"],
                    posterior_slice=active_attention_capture["posterior_slice"],
                    capacity=args.capacity,
                )
                active_attention_capture["layers"].append(
                    receipt.row_mass.detach().to(device="cpu")
                )
            return original_attention_interface(
                query_states,
                key_states,
                value_states,
                attention_mask,
            )

        joint_host.attention_interface = attention_interface_with_receipt
        parameter_manifest = _parameter_manifest(policy)
        if parameter_manifest["active_trainable_numel"] != 0:
            raise RuntimeError("LTOP G1b failed to freeze every parameter")
        timings["load_model_s"] = time.perf_counter() - started

        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=manifest.dataset_id,
            dataset_revision=manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=manifest,
        )
        dataset = CalvinStatefulTransitionDataset(index, action_horizon=config.chunk_size)
        stream_plan = build_native_calvin_stream_plan(
            dataset,
            comparison_id=G1B_COMPARISON_ID,
            seed=args.seed,
            global_batch_size=G1B_WORLD_SIZE,
            total_steps=args.plan_steps,
        )
        prompt_swap_plan = _build_prompt_swap_plan(stream_plan, dataset)
        evaluation_step = prompt_swap_plan["evaluation_step"]
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
            stream_plan,
            dataset,
            optimizer_step=evaluation_step,
            rank=rank,
            world_size=G1B_WORLD_SIZE,
            gradient_accumulation_steps=1,
            accumulation_index=0,
            device=device,
            dtype=torch.bfloat16,
        )
        prompt_planned = _apply_prompt_swap(
            planned,
            prompt_swap_plan,
            dataset,
        )

        def collate(candidate: Any) -> CollatedNativeCALVINBatch:
            result = collate_native_calvin_training_batch(
                candidate.training,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=candidate.augmentation_seeds,
                source_digest=candidate.source_digest,
            )
            result = CollatedNativeCALVINBatch(
                model_inputs=_move_model_inputs(
                    result.model_inputs,
                    device=device,
                    dtype=torch.bfloat16,
                    torch_module=torch,
                ),
                controls=result.controls,
                routing=result.routing,
                source_digest=result.source_digest,
                structural_target_requests=result.structural_target_requests,
                modalities=None,
                prior_control_chunks=result.prior_control_chunks,
            )
            return materialize_native_flow_randomness(result, candidate)

        natural = collate(planned)
        prompt = collate(prompt_planned)
        natural_observation = {name: natural.model_inputs[name] for name in _OBSERVATION_FIELDS}
        prompt_observation = {name: prompt.model_inputs[name] for name in _OBSERVATION_FIELDS}
        natural_scene_manifest, natural_scene_sha256 = _tensor_mapping_sha256(
            {name: natural_observation[name] for name in _SCENE_FIELDS}
        )
        prompt_scene_manifest, prompt_scene_sha256 = _tensor_mapping_sha256(
            {name: prompt_observation[name] for name in _SCENE_FIELDS}
        )
        natural_language_manifest, natural_language_sha256 = _tensor_mapping_sha256(
            {name: natural_observation[name] for name in ("lang_tokens", "lang_masks")}
        )
        prompt_language_manifest, prompt_language_sha256 = _tensor_mapping_sha256(
            {name: prompt_observation[name] for name in ("lang_tokens", "lang_masks")}
        )
        controls_exact = all(
            torch.equal(getattr(natural.controls, name), getattr(prompt.controls, name))
            for name in (
                "values",
                "field_valid",
                "token_valid",
                "delta_time",
                "reset",
                "acknowledged",
            )
        )
        same_scene_non_language_exact = (
            natural_scene_sha256 == prompt_scene_sha256 and controls_exact
        )
        natural_instruction = planned.training.host_items[0]["task"]
        donor_instruction = prompt_planned.training.host_items[0]["task"]
        prompt_changed = bool(
            isinstance(natural_instruction, str)
            and isinstance(donor_instruction, str)
            and natural_instruction != donor_instruction
            and natural_language_sha256 != prompt_language_sha256
        )

        episode_ids = _episode_ids(
            natural.routing.episode_keys,
            torch_module=torch,
            device=device,
        )
        prior_stepper = LingBotNativePriorStepper(policy, graph)

        def prior_rollout() -> AddressedLayerwisePriorTrace:
            prior_value: Any | None = None
            prior_valid = torch.zeros(
                natural.routing.batch_size,
                dtype=torch.bool,
                device=device,
            )
            with torch.inference_mode():
                for controls in natural.effective_prior_control_chunks:
                    prior_value = prior_stepper(
                        prior_value,
                        controls,
                        previous_memory_valid=prior_valid,
                        episode_ids=episode_ids,
                    )
                    prior_valid = torch.ones_like(prior_valid)
            if not isinstance(prior_value, AddressedLayerwisePriorTrace):
                raise RuntimeError("LTOP G1b prior rollout omitted addressed rows")
            return prior_value

        started_prior = time.perf_counter()
        natural_prior = prior_rollout()
        repeat_prior = prior_rollout()
        prompt_prior = prior_rollout()
        row_permutation = _nonidentity_row_permutation(
            args.capacity,
            rank,
            torch_module=torch,
            device=device,
        )
        permuted_prior = natural_prior.permute_rows(row_permutation)
        torch.cuda.synchronize(device)
        timings["prior_arms_s"] = time.perf_counter() - started_prior

        def observation_context(
            prior_trace: AddressedLayerwisePriorTrace,
            batch: CollatedNativeCALVINBatch,
        ) -> tuple[Any, Any]:
            nonlocal active_attention_capture
            context = native_context_from_prior_trace(
                controls=batch.controls,
                prior_trace=prior_trace,
                modalities=None,
                object_read_action_intervention=ObjectReadActionIntervention.BLOCKED,
            )
            active_attention_capture = {
                "context": context,
                "layers": [],
            }
            started_call = time.perf_counter()
            try:
                run_native_policy_observation_diagnostic_forward(
                    policy,
                    model_inputs=batch.model_inputs,
                    context=context,
                )
            finally:
                captured = active_attention_capture
                active_attention_capture = None
            torch.cuda.synchronize(device)
            timings.setdefault("observation_arm_s", []).append(time.perf_counter() - started_call)
            if not isinstance(context.posterior_memory, AddressedLayerwisePosteriorState):
                raise RuntimeError("LTOP G1b correction omitted addressed posterior rows")
            if captured is None or len(captured["layers"]) != graph.config.num_layers:
                raise RuntimeError("LTOP G1b did not capture every host attention layer")
            return context, torch.stack(captured["layers"], dim=1).to(device=device)

        natural_context, natural_attention = observation_context(natural_prior, natural)
        repeat_context, repeat_attention = observation_context(repeat_prior, natural)
        permuted_context, permuted_attention = observation_context(permuted_prior, natural)
        prompt_context, prompt_attention = observation_context(prompt_prior, prompt)
        joint_host.attention_interface = original_attention_interface

        natural_posterior = natural_context.posterior_memory.layer_rows
        repeat_posterior = repeat_context.posterior_memory.layer_rows
        permuted_posterior = permuted_context.posterior_memory.layer_rows
        prompt_posterior = prompt_context.posterior_memory.layer_rows
        prior_metrics = {
            "noncollapse": replay_noncollapse_metrics(
                natural_prior.layer_rows,
                repeat_prior.layer_rows,
                torch_module=torch,
            ),
            "permutation": permutation_equivariance_metrics(
                natural_prior.layer_rows,
                repeat_prior.layer_rows,
                permuted_prior.layer_rows,
                row_permutation,
                torch_module=torch,
            ),
            "normalized_permutation_recovery": normalized_permutation_recovery_metrics(
                natural_prior.layer_rows,
                permuted_prior.layer_rows,
                row_permutation,
                torch_module=torch,
            ),
            "prompt_invariance": prompt_invariance_metrics(
                natural_prior.layer_rows,
                repeat_prior.layer_rows,
                prompt_prior.layer_rows,
                torch_module=torch,
            ),
        }
        posterior_metrics = {
            "noncollapse": replay_noncollapse_metrics(
                natural_posterior,
                repeat_posterior,
                torch_module=torch,
            ),
            "permutation": permutation_equivariance_metrics(
                natural_posterior,
                repeat_posterior,
                permuted_posterior,
                row_permutation,
                torch_module=torch,
            ),
            "normalized_permutation_recovery": normalized_permutation_recovery_metrics(
                natural_posterior,
                permuted_posterior,
                row_permutation,
                torch_module=torch,
            ),
            "prompt_invariance": prompt_invariance_metrics(
                natural_posterior,
                repeat_posterior,
                prompt_posterior,
                torch_module=torch,
            ),
        }
        task_addressing = task_address_attention_metrics(
            natural_attention,
            repeat_attention,
            permuted_attention,
            prompt_attention,
            row_permutation,
            torch_module=torch,
        )
        expected_address_state = natural_prior.episode_address_state.permute_rows(row_permutation)
        address_permutation_consistent = bool(
            expected_address_state.same_assignment(permuted_prior.episode_address_state)
            and torch.equal(
                permuted_prior.layer_rows,
                natural_prior.layer_rows.index_select(2, row_permutation),
            )
        )
        contexts_finalized = all(
            context._finalized
            for context in (
                natural_context,
                repeat_context,
                permuted_context,
                prompt_context,
            )
        )
        rows_finite = all(
            torch.isfinite(rows).all().item()
            for rows in (
                natural_prior.layer_rows,
                repeat_prior.layer_rows,
                permuted_prior.layer_rows,
                prompt_prior.layer_rows,
                natural_posterior,
                repeat_posterior,
                permuted_posterior,
                prompt_posterior,
            )
        )
        rank_report = {
            "rank": rank,
            "device_name": properties.name,
            "evaluation_step": evaluation_step,
            "sample_keys": list(natural.routing.sample_keys),
            "episode_keys": list(natural.routing.episode_keys),
            "frame_indices": list(natural.routing.frame_indices),
            "natural_source_digest": natural.source_digest,
            "prompt_source_digest": prompt.source_digest,
            "natural_instruction_sha256": hashlib.sha256(
                natural_instruction.encode("utf-8")
            ).hexdigest(),
            "donor_instruction_sha256": hashlib.sha256(
                donor_instruction.encode("utf-8")
            ).hexdigest(),
            "natural_scene_manifest": natural_scene_manifest,
            "prompt_scene_manifest": prompt_scene_manifest,
            "natural_language_manifest": natural_language_manifest,
            "prompt_language_manifest": prompt_language_manifest,
            "same_scene_non_language_exact": same_scene_non_language_exact,
            "prompt_changed": prompt_changed,
            "episode_ids": [int(value) for value in episode_ids.tolist()],
            "natural_address_receipt": natural_prior.address_receipt,
            "repeat_address_receipt": repeat_prior.address_receipt,
            "prompt_address_receipt": prompt_prior.address_receipt,
            "permuted_address_receipt": permuted_prior.address_receipt,
            "row_permutation": [int(value) for value in row_permutation.tolist()],
            "address_permutation_consistent": address_permutation_consistent,
            "prior": prior_metrics,
            "posterior": posterior_metrics,
            "task_addressing": task_addressing,
            "contexts_finalized": contexts_finalized,
            "rows_finite": rows_finite,
            "timings": timings,
            "cuda_memory_bytes": {
                "allocated": int(torch.cuda.memory_allocated(device)),
                "reserved": int(torch.cuda.memory_reserved(device)),
                "peak_allocated": int(torch.cuda.max_memory_allocated(device)),
                "peak_reserved": int(torch.cuda.max_memory_reserved(device)),
            },
        }
        gathered: list[dict[str, Any] | None] = [None] * G1B_WORLD_SIZE
        dist.all_gather_object(gathered, rank_report)
        outcome: list[object] = [None, None]
        if rank == 0:
            rank_reports = [item for item in gathered if item is not None]
            failures: list[str] = []
            for item in sorted(rank_reports, key=lambda entry: entry["rank"]):
                failures.extend(_metric_failures(item))
            provisional = {
                "schema": G1B_SCHEMA,
                "status": "PASS" if not failures else "FAIL",
                "failures": failures,
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "patch_sha256": patch_report["patch_sha256"],
                "patched_source_sha256": patched_source_sha256,
                "source_diff_sha256": hashlib.sha256(
                    _git_output(args.source_checkout, "diff", "--binary").encode("utf-8")
                ).hexdigest(),
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "processor_revision": QWEN_PROCESSOR_REVISION,
                "asset_validation": asset_validation,
                "implementation_sha256": _implementation_digest(
                    root,
                    entrypoint=Path(__file__),
                ),
                "architecture_identity": G1B_ARCHITECTURE,
                "world_size": G1B_WORLD_SIZE,
                "seed": args.seed,
                "capacity": args.capacity,
                "task_query_count": args.task_query_count,
                "plan_steps": args.plan_steps,
                "evaluation_step": evaluation_step,
                "threshold_contract": {
                    "noncollapse": "pair_l2 > 2 * exact_replay_max_row_l2",
                    "permutation": "permuted-arm error <= exact-replay error",
                    "prompt": "prompt-swap error <= exact-replay error",
                },
                "eval_contract": G1B_EVAL_CONTRACT,
                "parallel_contract": G1B_PARALLEL_CONTRACT,
                "dataset_contract": dataset_contract,
                "stream_plan_sha256": stream_plan.plan_sha256,
                "task_intervention_sha256": prompt_swap_plan["artifact_sha256"],
                "config_sha256": config_sha256,
                "parameter_manifest": parameter_manifest,
                "alignment_teacher_prune": alignment_teacher_prune,
                "moe_inference_backend": moe_backend,
                "rank_reports": rank_reports,
            }
            report = validate_ltop_g1b_report(provisional)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            write_text_durable_exclusive(
                args.output,
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            outcome = [report["status"], report["failures"]]
        dist.broadcast_object_list(outcome, src=0)
        dist.barrier()
        if outcome[0] != "PASS":
            raise RuntimeError(f"LTOP G1b rejected: {outcome[1]}")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

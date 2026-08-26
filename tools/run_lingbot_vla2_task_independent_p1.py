#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Run a bounded two-A100 current-frame physical-entity learning probe.

This runner executes the released LingBot observation root, the complete 6B
trainable VLM host, and the task-independent PICF entity set. It deliberately
omits the action suffix by default: P1 asks whether physical entities are
learnable without a prompt-conditioned winner or simulator input leak.  Its
registered joint-action arm executes the released action suffix on the same
current-frame stream to isolate action-gradient competition from recurrent
posterior replay.  The optional staged P2 gate remains action-free and tests
future prediction and exact source-disjoint counterfactual use.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import random
import sys
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass
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

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.lingbot_native.adr175_contract import Adr175BroadSupportContract
from picf_next.lingbot_native.capacity import require_persistent_run_root
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_CPU_OFFLOAD,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from picf_next.lingbot_native.representation_stage import (
    NATIVE_ACTION_ONLY_PARAMETER_PREFIXES,
    configure_native_representation_parameter_scope,
    is_native_action_only_parameter,
    verify_native_representation_parameter_scope,
)
from picf_next.training.run_lease import acquire_distributed_run_lease

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
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
        build_lingbot_official_optimizer,
        build_lingbot_representation_optimizer,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _distributed_gradient_metrics,
        _model_local_state_digest,
        _move_model_inputs,
        _validate_optimizer_state,
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
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        _tensor_sha256,
        build_lingbot_official_optimizer,
        build_lingbot_representation_optimizer,
        clip_lingbot_distributed_l2_grad_norm_,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _distributed_gradient_metrics,
        _model_local_state_digest,
        _move_model_inputs,
        _validate_optimizer_state,
        _validate_fsdp2_parameter_storage,
    )


P1_WORLD_SIZE = 2
P1_COMPARISON_ID = "lingbot-vla2-task-independent-p1"
P1_REPORT_SCHEMA = "picf-next.lingbot-vla2-task-independent-p1.v3"
P1_CURVE_SNAPSHOT_SCHEMA = "picf-next.lingbot-vla2-task-independent-p1-snapshot.v2"
ADR175_EVALUATION_SNAPSHOT_SCHEMA = "picf-next.adr175-evaluation-snapshot.v1"
P2_GATE_REPORT_SCHEMA = "picf-next.lingbot-vla2-task-independent-p2-gate.v1"
P2_UPDATE_GATE_REPORT_SCHEMA = "picf-next.lingbot-vla2-task-independent-p2-update-gate.v1"
STAGED_P2_REPORT_SCHEMA = "picf-next.lingbot-vla2-task-independent-staged-p2.v5"
P2_CAUSAL_REPORT_SCHEMA = "picf-next.lingbot-vla2-p2-future-causal-evidence.v1"
P2_CAUSAL_REPLAY_CLOSURE_SCHEMA = "picf-next.calvin-causal-replay-file-closure/v2"
_MAXIMUM_P1_STEPS = 20
_MAXIMUM_P1_CURVE_STEPS = 200
_MAXIMUM_ADR175_CURVE_STEPS = 2_000
_MAXIMUM_STAGED_P2_STEPS = 200
_MAXIMUM_P2_CAUSAL_PROBE_STEPS = 32
_SUPPORTED_P1_VISUAL_LATTICES = (8, 12)
ADR175_ARMS = ("lbot", "physical-set", "native-attention")
ADR175_MILESTONES = (0, 250, 500, 1_000, 2_000)
ADR175_ENTITY_WEIGHT = 0.08
ADR175_NATIVE_ATTENTION_WEIGHT = 0.001
ADR175_REGISTERED_LAYER_OFFSETS = (-4, -1)
ADR175_GUIDEDVLA_OBJECT_HEAD_INDICES = (0, 1)
ADR175_GRAPH_INIT_SEED = 20260816
_P2_CAUSAL_MINIMUM_SAMPLE_COUNT = 14
_P2_CAUSAL_MARGIN_EPSILON = 1e-6
_P2_CAUSAL_POSITIVE_FRACTION_THRESHOLDS = {
    "absent_source": 0.6,
    "batch_shift_control": 0.5,
    "batch_shift_source": 0.6,
    "row_shift_source": 0.6,
    "wrong_time_source": 0.5,
    "zero_control": 0.5,
    "zero_current_observation": 0.6,
    "zero_source": 0.6,
}
_P2_CAUSAL_NEUTRAL_CONTROLS = ("matched_noise_source",)
_PREDICTIVE_BUILD_REPORT_FIELDS = {
    "cache_manifest_sha256",
    "coverage_sha256",
    "expected_record_count",
    "output_root",
    "pair_keys_sha256",
    "patch_sha256",
    "physical_visual_acceptance_sha256",
    "stream_plan_sha256",
    "teacher_encoder_digest",
    "temporal_estimator_sha256",
}
_P2_CAUSAL_REPLAY_CLOSURE_FIELDS = {
    "action_horizon",
    "artifact_sha256",
    "available_roots",
    "count",
    "dataset_manifest_sha256",
    "horizon",
    "missing_paths",
    "predictive_cache_manifest_sha256",
    "prefix_frames",
    "representation_split_file_sha256",
    "required_paths",
    "schema",
    "selection_domain",
    "selection_seed",
    "selections",
    "training_source_episode_indices_sha256",
}


@dataclass(frozen=True, slots=True)
class _StagedP2Selection:
    record: Any
    segment: Any
    episode: Any
    transition_index: int
    plan_optimizer_step: int
    rank: int


_ACTION_ONLY_GRADIENT_METRICS = tuple(
    (
        name,
        prefix.removeprefix("model"),
    )
    for name, prefix in zip(
        (
            "action_expert",
            "action_state",
            "action_input",
            "action_output",
            "action_time_input",
            "action_time_output",
        ),
        NATIVE_ACTION_ONLY_PARAMETER_PREFIXES,
        strict=True,
    )
)


def _add_action_only_gradient_summary(
    metrics: dict[str, float | int | bool],
) -> dict[str, float | int | bool]:
    """Aggregate the complete released action-only parameter partition."""

    metrics["action_only_elements"] = sum(
        int(metrics[f"{name}_elements"]) for name, _fragment in _ACTION_ONLY_GRADIENT_METRICS
    )
    metrics["action_only_norm"] = math.sqrt(
        sum(
            float(metrics[f"{name}_norm"]) ** 2 for name, _fragment in _ACTION_ONLY_GRADIENT_METRICS
        )
    )
    return metrics


def _require_carried_bindings_preserved(
    prior_by_batch: tuple[Any, ...],
    current_by_batch: tuple[Any, ...],
    *,
    capacity: int,
) -> int:
    """Fail closed if a loss-side physical identity changes row."""

    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("binding audit capacity must be a positive integer")
    if not prior_by_batch or len(prior_by_batch) != len(current_by_batch):
        raise ValueError("binding audit batches must be nonempty and aligned")
    preserved = 0
    for batch_index, (prior, current) in enumerate(
        zip(prior_by_batch, current_by_batch, strict=True)
    ):
        prior_map = dict(prior)
        current_map = dict(current)
        if len(prior_map) != len(prior) or len(current_map) != len(current):
            raise ValueError("binding audit identities must be unique within each batch")
        for identity, row in prior_map.items():
            if (
                not isinstance(identity, str)
                or not identity
                or isinstance(row, bool)
                or not isinstance(row, int)
                or not 0 <= row < capacity
            ):
                raise ValueError("binding audit received an invalid identity-row pair")
            if current_map.get(identity) != row:
                raise RuntimeError(
                    "staged P2 changed a carried physical row: "
                    f"batch={batch_index}, identity={identity!r}, "
                    f"prior={row}, current={current_map.get(identity)!r}"
                )
            preserved += 1
    if preserved <= 0:
        raise RuntimeError("staged P2 binding audit observed no carried identity")
    return preserved


def _select_staged_p2_records_from_plan(
    *,
    plan: Any,
    dataset: Any,
    predictive_cache: Any,
    segments: Any,
    episodes: Any,
    start_optimizer_step: int,
    steps: int,
    world_size: int,
    horizon: int,
    allowed_episode_indices: frozenset[int],
) -> tuple[_StagedP2Selection, ...]:
    """Resolve a label-independent causal P2 subsequence from one frozen plan."""

    for name, value, minimum in (
        ("start optimizer step", start_optimizer_step, 0),
        ("steps", steps, 1),
        ("world size", world_size, 1),
        ("horizon", horizon, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"staged P2 {name} is outside its valid range")
    if not allowed_episode_indices or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in allowed_episode_indices
    ):
        raise ValueError("staged P2 requires nonempty training episode indices")
    total_steps = getattr(plan, "total_steps", None)
    if not isinstance(total_steps, int) or isinstance(total_steps, bool):
        raise TypeError("staged P2 requires a frozen stream plan")

    episode_by_key = {episode.episode_key: episode for episode in episodes}
    segment_by_index = {int(segment.index): segment for segment in segments}
    selected_batches: list[tuple[_StagedP2Selection, ...]] = []
    for plan_step in range(start_optimizer_step, total_steps):
        transitions = tuple(plan.global_batch(plan_step).transitions)
        if len(transitions) != world_size:
            raise RuntimeError("staged P2 frozen plan has the wrong global batch size")
        candidates: list[tuple[Any, Any, int, int]] = []
        complete_global_batch = True
        for transition in transitions:
            episode = episode_by_key.get(transition.episode_key)
            if episode is None:
                raise RuntimeError("staged P2 plan references an absent episode")
            transition_index = int(transition.transition_index)
            if transition_index < 1 or transition_index + horizon >= len(episode.sample_keys):
                complete_global_batch = False
                continue
            if episode.sample_keys[transition_index] != transition.sample.sample_key:
                raise RuntimeError("staged P2 plan and episode sample identity differ")
            segment = segment_by_index.get(int(episode.segment_index))
            if segment is None:
                raise RuntimeError("staged P2 episode references an absent CALVIN segment")
            if int(segment.episode_index) not in allowed_episode_indices:
                complete_global_batch = False
                continue
            source_global_index = int(
                dataset.source_global_index_by_key(transition.sample.sample_key)
            )
            candidates.append((episode, segment, transition_index, source_global_index))
        if not complete_global_batch or len(candidates) != world_size:
            continue

        selected_batch: list[_StagedP2Selection] = []
        for rank, (episode, segment, transition_index, source_global_index) in enumerate(
            candidates
        ):
            record = predictive_cache.record_for(
                source_global_index=source_global_index,
                horizon=horizon,
            )
            if record is None:
                raise RuntimeError("staged P2 cache omits a causally eligible frozen-plan record")
            selected_batch.append(
                _StagedP2Selection(
                    record=record,
                    segment=segment,
                    episode=episode,
                    transition_index=transition_index,
                    plan_optimizer_step=plan_step,
                    rank=rank,
                )
            )
        selected_batches.append(tuple(selected_batch))
        if len(selected_batches) == steps:
            return tuple(item for batch in selected_batches for item in batch)
    raise RuntimeError("frozen stream plan has insufficient causal P2 records")


def _select_p2_causal_records(
    *,
    records: Any,
    segments: Any,
    episodes: Any,
    horizon: int,
    count: int,
    prefix_frames: int = 1,
    allowed_episode_indices: frozenset[int] | None = None,
    require_positive_importance: bool = True,
    selection_seed: int | None = None,
    distinct_source_episodes: bool = False,
) -> tuple[tuple[Any, Any, Any, int], ...]:
    """Select distinct cached sources with an observed prefix and complete future."""

    if (
        isinstance(horizon, bool)
        or not isinstance(horizon, int)
        or horizon < 1
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or isinstance(prefix_frames, bool)
        or not isinstance(prefix_frames, int)
        or prefix_frames < 1
    ):
        raise ValueError("P2 causal selection requires positive horizon and count")
    if not isinstance(require_positive_importance, bool):
        raise TypeError("P2 importance filter must be boolean")
    if not isinstance(distinct_source_episodes, bool):
        raise TypeError("P2 distinct-episode filter must be boolean")
    if selection_seed is not None and (
        isinstance(selection_seed, bool)
        or not isinstance(selection_seed, int)
        or selection_seed < 0
    ):
        raise ValueError("P2 selection seed must be a non-negative integer")
    segment_axis = tuple(segments)
    episode_axis = tuple(episodes)
    if len(segment_axis) != len(episode_axis):
        raise ValueError("P2 dataset segments and episodes differ")

    causal_records: list[tuple[Any, Any, Any, int]] = []
    for record in records:
        if record.horizon != horizon or (
            require_positive_importance and not bool((record.importance > 0).any())
        ):
            continue
        matches: list[tuple[Any, Any, int]] = []
        for segment, episode in zip(segment_axis, episode_axis, strict=True):
            if (
                allowed_episode_indices is not None
                and int(segment.episode_index) not in allowed_episode_indices
            ):
                continue
            transition_index = record.source_global_index - segment.start
            if (
                transition_index >= prefix_frames
                and transition_index + horizon < len(episode.sample_keys)
            ):
                matches.append((segment, episode, transition_index))
        if matches:
            canonical = min(matches, key=lambda value: (value[0].index, value[2]))
            causal_records.append((record, *canonical))
    if selection_seed is not None:
        causal_records.sort(
            key=lambda item: hashlib.sha256(
                f"{selection_seed}\0{int(item[0].source_global_index)}".encode("ascii")
            ).digest()
        )
    if distinct_source_episodes:
        distinct_records: list[tuple[Any, Any, Any, int]] = []
        selected_episode_indices: set[int] = set()
        for item in causal_records:
            episode_index = int(item[1].episode_index)
            if episode_index in selected_episode_indices:
                continue
            selected_episode_indices.add(episode_index)
            distinct_records.append(item)
        causal_records = distinct_records
    if len(causal_records) < count:
        raise RuntimeError("P2 cache has insufficient records with a causal prefix and future")
    selected = tuple(causal_records[:count])
    if len({item[0].source_global_index for item in selected}) != count:
        raise RuntimeError("P2 gate selected duplicate source frames")
    return selected


def _calvin_causal_replay_dependency_closure(
    *,
    source_global_index: int,
    segment_start: int,
    segment_end: int,
    horizon: int,
    prefix_frames: int,
    action_horizon: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return replay frames and every CALVIN frame read while materializing them."""

    values = (
        source_global_index,
        segment_start,
        segment_end,
        horizon,
        prefix_frames,
        action_horizon,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
        raise TypeError("CALVIN causal replay closure requires integer coordinates")
    if (
        min(horizon, prefix_frames, action_horizon) <= 0
        or segment_start < 0
        or segment_end <= segment_start
        or source_global_index - prefix_frames < segment_start
        or source_global_index + horizon >= segment_end
    ):
        raise ValueError("CALVIN causal replay closure crosses its language segment")
    replay_indices = tuple(
        range(
            source_global_index - prefix_frames,
            source_global_index + horizon + 1,
        )
    )
    required: set[int] = set()
    for replay_index in replay_indices:
        if replay_index > segment_start:
            required.add(replay_index - 1)
        required.update(
            range(
                replay_index,
                min(replay_index + action_horizon, segment_end),
            )
        )
    return replay_indices, tuple(sorted(required))


def _distributed_ring_exchange_tensor(
    value: Any,
    *,
    dist_module: Any,
    torch_module: Any,
) -> Any:
    """Return the same-shaped peer tensor for a two-rank negative control."""

    if not torch_module.is_tensor(value) or value.ndim < 1:
        raise TypeError("distributed causal exchange requires a non-scalar tensor")
    world_size = int(dist_module.get_world_size())
    rank = int(dist_module.get_rank())
    if world_size != P1_WORLD_SIZE or not 0 <= rank < world_size:
        raise RuntimeError("causal exchange requires the frozen two-rank topology")
    gathered = [torch_module.empty_like(value) for _ in range(world_size)]
    dist_module.all_gather(gathered, value.detach())
    peer = gathered[(rank + 1) % world_size]
    if peer.shape != value.shape or peer.dtype != value.dtype or peer.device != value.device:
        raise RuntimeError("causal exchange changed tensor metadata")
    return peer


def _summarize_p2_causal_evidence(
    gathered_rank_reports: list[Any],
    *,
    expected_global_steps: int,
) -> dict[str, Any]:
    """Apply one preregistered gate to source-disjoint future counterfactuals."""

    if (
        isinstance(expected_global_steps, bool)
        or not isinstance(expected_global_steps, int)
        or expected_global_steps <= 0
    ):
        raise ValueError("causal summary requires positive global steps")
    if len(gathered_rank_reports) != P1_WORLD_SIZE or {
        int(item.get("rank", -1)) for item in gathered_rank_reports
    } != set(range(P1_WORLD_SIZE)):
        raise ValueError("causal summary rank set changed")
    records: list[Mapping[str, Any]] = []
    for rank_report in gathered_rank_reports:
        steps = rank_report.get("causal_steps")
        if not isinstance(steps, list) or len(steps) != expected_global_steps:
            raise ValueError("causal summary has the wrong per-rank step count")
        records.extend(steps)
    expected_samples = expected_global_steps * P1_WORLD_SIZE
    if (
        len(records) != expected_samples
        or len({int(item["source_global_index"]) for item in records}) != expected_samples
        or len({int(item["source_episode_index"]) for item in records})
        != expected_samples
    ):
        raise ValueError(
            "causal summary source frame or episode identities are incomplete or duplicated"
        )

    by_intervention: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in _P2_CAUSAL_POSITIVE_FRACTION_THRESHOLDS
    }
    by_neutral_control: dict[str, list[Mapping[str, Any]]] = {
        name: [] for name in _P2_CAUSAL_NEUTRAL_CONTROLS
    }
    expected_names = set(by_intervention) | set(by_neutral_control)
    for record in records:
        if record.get("exact_correction_then_prior_route") is not True:
            raise RuntimeError("causal probe differs from the registered factual route")
        if record.get("partition") not in {"validation", "heldout", "causal_audit"}:
            raise ValueError("causal record has an unknown source-disjoint partition")
        diagnostics = record.get("diagnostics")
        if not isinstance(diagnostics, Mapping):
            raise ValueError("causal record omitted diagnostics")
        interventions = diagnostics.get("interventions")
        if not isinstance(interventions, list):
            raise ValueError("causal diagnostics omitted intervention records")
        names = {str(item.get("name")) for item in interventions}
        if names != expected_names:
            raise ValueError("causal intervention set differs from the preregistration")
        for item in interventions:
            name = str(item["name"])
            destination = by_intervention if name in by_intervention else by_neutral_control
            destination[name].append(item)

    summaries: dict[str, dict[str, float | int | bool]] = {}
    all_arms_pass = True
    for name, values in sorted(by_intervention.items()):
        margins = [float(item["loss_margin_over_factual"]) for item in values]
        distances = [float(item["normalized_prediction_l1"]) for item in values]
        if any(not math.isfinite(value) for value in (*margins, *distances)):
            raise ValueError("causal intervention metrics must be finite")
        mean_margin = math.fsum(margins) / len(margins)
        mean_distance = math.fsum(distances) / len(distances)
        positive_fraction = sum(
            value > _P2_CAUSAL_MARGIN_EPSILON for value in margins
        ) / len(margins)
        threshold = _P2_CAUSAL_POSITIVE_FRACTION_THRESHOLDS[name]
        arm_pass = (
            mean_margin > _P2_CAUSAL_MARGIN_EPSILON
            and mean_distance > _P2_CAUSAL_MARGIN_EPSILON
            and positive_fraction >= threshold
        )
        summaries[name] = {
            "sample_count": len(values),
            "mean_loss_margin_over_factual": mean_margin,
            "mean_normalized_prediction_l1": mean_distance,
            "positive_margin_fraction": positive_fraction,
            "required_positive_margin_fraction": threshold,
            "pass": arm_pass,
        }
        all_arms_pass = all_arms_pass and arm_pass

    neutral_controls: dict[str, dict[str, float | int | str]] = {}
    for name, values in sorted(by_neutral_control.items()):
        margins = [float(item["loss_margin_over_factual"]) for item in values]
        distances = [float(item["normalized_prediction_l1"]) for item in values]
        if any(not math.isfinite(value) for value in (*margins, *distances)):
            raise ValueError("causal neutral-control metrics must be finite")
        neutral_controls[name] = {
            "role": "paired_null_control_not_an_acceptance_arm",
            "sample_count": len(values),
            "mean_loss_margin_over_factual": math.fsum(margins) / len(margins),
            "mean_normalized_prediction_l1": math.fsum(distances) / len(distances),
        }

    sample_count = len(records)
    status = (
        "INCONCLUSIVE"
        if sample_count < _P2_CAUSAL_MINIMUM_SAMPLE_COUNT
        else "PASS"
        if all_arms_pass
        else "FAIL"
    )
    return {
        "schema": P2_CAUSAL_REPORT_SCHEMA,
        "status": status,
        "sample_count": sample_count,
        "minimum_sample_count": _P2_CAUSAL_MINIMUM_SAMPLE_COUNT,
        "margin_epsilon": _P2_CAUSAL_MARGIN_EPSILON,
        "interventions": summaries,
        "neutral_controls": neutral_controls,
        "records": records,
    }


def cache_horizons_as_ints(values: object) -> tuple[int, ...]:
    """Published cache horizons, normalised to integers for membership tests."""

    if not isinstance(values, (list, tuple)):
        raise ValueError("cache horizons must be a sequence")
    return tuple(int(value) for value in values)


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
        "--checkpoint-dir", type=Path, default=_environment_path("PICF_CHECKPOINT_DIR")
    )
    parser.add_argument(
        "--processor-dir", type=Path, default=_environment_path("PICF_PROCESSOR_DIR")
    )
    parser.add_argument("--dataset-split", type=Path, default=_environment_path("PICF_DATASET_DIR"))
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=_environment_path("PICF_DATASET_MANIFEST"),
    )
    parser.add_argument(
        "--norm-stats", type=Path, default=_environment_path("PICF_LINGBOT_NORM_STATS")
    )
    parser.add_argument(
        "--physical-sidecar-root",
        type=Path,
        default=_environment_path("PICF_CALVIN_PHYSICAL_SIDECAR"),
    )
    parser.add_argument(
        "--physical-sidecar-manifest",
        type=Path,
        default=_environment_path("PICF_CALVIN_PHYSICAL_SIDECAR_MANIFEST"),
    )
    parser.add_argument(
        "--physical-sidecar-manifest-sha256",
        default=os.environ.get("PICF_CALVIN_PHYSICAL_SIDECAR_SHA256"),
    )
    parser.add_argument("--run-dir", type=Path, default=_environment_path("PICF_RUN_DIR"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stream-plan", type=Path)
    parser.add_argument("--stream-plan-sha256")
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument("--entity-evaluation-plan", type=Path)
    parser.add_argument("--entity-evaluation-plan-sha256")
    parser.add_argument(
        "--adr175-arm",
        choices=ADR175_ARMS,
        help="Execute one preregistered ADR-175 matched arm.",
    )
    parser.add_argument("--adr175-contract", type=Path)
    parser.add_argument("--adr175-contract-sha256")
    parser.add_argument(
        "--evaluation-steps",
        help="Comma-separated post-update steps; curve mode also requires step zero.",
    )
    parser.add_argument("--evaluation-visuals-per-partition", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument(
        "--visual-lattice",
        type=int,
        choices=_SUPPORTED_P1_VISUAL_LATTICES,
        default=8,
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--maximum-peak-reserved-gib", type=float, default=39.0)
    parser.add_argument("--minimum-supervised-fraction", type=float, default=0.0)
    parser.add_argument(
        "--current-frame-action-weight",
        type=float,
        default=0.0,
        help=(
            "Execute the released action suffix with this objective weight while "
            "keeping recurrent input absent. Positive values define the registered "
            "joint-action causal arm."
        ),
    )
    parser.add_argument(
        "--current-frame-entity-weight",
        type=float,
        default=1.0,
        help="Entity-family weight for the current-frame P1 or joint-action arm.",
    )
    parser.add_argument("--p2-predictive-cache-root", type=Path)
    parser.add_argument("--p2-predictive-cache-build-report", type=Path)
    parser.add_argument("--p2-predictive-cache-build-report-sha256")
    parser.add_argument("--p2-causal-probe-cache-root", type=Path)
    parser.add_argument("--p2-causal-probe-cache-build-report", type=Path)
    parser.add_argument("--p2-causal-probe-cache-build-report-sha256")
    parser.add_argument("--p2-causal-replay-closure", type=Path)
    parser.add_argument("--p2-causal-replay-closure-sha256")
    parser.add_argument(
        "--p2-causal-probe-steps",
        type=int,
        default=0,
        help="Source-disjoint two-rank future-counterfactual global steps after staged P2.",
    )
    parser.add_argument("--p2-stream-plan", type=Path)
    parser.add_argument("--p2-stream-plan-sha256")
    parser.add_argument("--p2-representation-split", type=Path)
    parser.add_argument("--p2-representation-split-sha256")
    parser.add_argument("--p2-horizon", type=int, default=1)
    # Defaults to --p2-horizon. Set it higher to ask the causal probe about a target
    # further away than the next frame without disturbing P2 training.
    parser.add_argument("--p2-causal-horizon", type=int, default=None)
    parser.add_argument(
        "--staged-p2-steps",
        type=int,
        default=0,
        help=(
            "Continue from P1 into this many P2 updates in the same process, "
            "model, and optimizer. Requires the frozen P1 split and P2 cache."
        ),
    )
    parser.add_argument(
        "--p2-optimizer-update",
        action="store_true",
        help="Execute and audit one real optimizer update after the bounded P2 backward gate.",
    )
    parser.add_argument("--mask-focal-weight", type=float, default=1.0)
    parser.add_argument("--mask-dice-weight", type=float, default=1.0)
    parser.add_argument("--existence-weight", type=float, default=1.0)
    parser.add_argument("--ownership-weight", type=float, default=1.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
    parser.add_argument("--cuda-allocator", choices=CUDA_ALLOCATOR_MODES, default="native")
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    if args.physical_sidecar_manifest is None and args.physical_sidecar_root is not None:
        args.physical_sidecar_manifest = args.physical_sidecar_root / "manifest.json"
    if args.output is None and args.run_dir is not None:
        filename = (
            f"adr175_{args.adr175_arm}_steps_{args.steps}.json"
            if args.adr175_arm is not None
            else f"task_independent_staged_p1_{args.steps}_p2_{args.staged_p2_steps}.json"
            if args.staged_p2_steps > 0
            else (
                f"task_independent_current_frame_joint_action_steps_{args.steps}.json"
                if args.current_frame_action_weight > 0
                else f"task_independent_p1_steps_{args.steps}.json"
            )
        )
        args.output = args.run_dir / filename
    return args


def _require_sha256(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _evaluation_steps(value: str | None) -> tuple[int, ...]:
    if value is None:
        return ()
    try:
        result = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("P1 evaluation steps must be comma-separated integers") from error
    if not result or result != tuple(sorted(set(result))) or any(item < 0 for item in result):
        raise ValueError("P1 evaluation steps must be unique sorted non-negative integers")
    return result


def _evaluation_visual_sample_keys(
    items: tuple[Any, ...],
    *,
    partitions: tuple[str, ...],
    per_partition: int,
) -> tuple[str, ...]:
    """Choose deterministic visuals from distinct tasks before any replicate."""

    if isinstance(per_partition, bool) or not isinstance(per_partition, int):
        raise ValueError("evaluation visual count must be an integer")
    if per_partition < 0:
        raise ValueError("evaluation visual count must be non-negative")
    selected: list[str] = []
    for partition in partitions:
        seen_tasks: set[str] = set()
        for item in items:
            if item.partition != partition or item.task_key in seen_tasks:
                continue
            selected.append(item.sample_key)
            seen_tasks.add(item.task_key)
            if len(seen_tasks) == per_partition:
                break
        if len(seen_tasks) != per_partition:
            raise ValueError(
                f"evaluation partition {partition!r} lacks {per_partition} distinct tasks"
            )
    return tuple(selected)


def _adr175_registered_layer_indices(layer_count: int) -> tuple[int, ...]:
    """Resolve the preregistered late shared-host layers without guessing depth."""

    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("ADR-175 shared-host layer count must be positive")
    indices = tuple(layer_count + offset for offset in ADR175_REGISTERED_LAYER_OFFSETS)
    if any(not 0 <= index < layer_count for index in indices):
        raise ValueError("ADR-175 registered layer offsets are outside the shared host")
    return indices


def _validate_entity_loss_weights(args: argparse.Namespace) -> None:
    names = (
        "mask_focal_weight",
        "mask_dice_weight",
        "existence_weight",
        "ownership_weight",
    )
    values = tuple(getattr(args, name) for name in names)
    for name, value in zip(names, values, strict=True):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"P1 {name} must be finite and non-negative")
    if not any(value > 0 for value in values):
        raise ValueError("P1 requires at least one active entity loss component")


def _validate_current_frame_objective_weights(args: argparse.Namespace) -> bool:
    """Validate the one-factor current-frame action treatment."""

    for name in ("current_frame_action_weight", "current_frame_entity_weight"):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"P1 {name} must be finite and non-negative")
    if args.current_frame_entity_weight <= 0:
        raise ValueError("the current-frame causal arm requires a positive entity weight")
    return args.current_frame_action_weight > 0


def _validate_visual_lattice_inputs(
    model_inputs: dict[str, Any],
    *,
    visual_lattice: int,
    merge_size: int,
) -> dict[str, int]:
    """Bind each collated CALVIN frame to the declared two-view Qwen lattice."""

    if visual_lattice not in _SUPPORTED_P1_VISUAL_LATTICES:
        raise ValueError("P1 visual lattice is not a registered arm")
    if isinstance(merge_size, bool) or not isinstance(merge_size, int) or merge_size <= 0:
        raise ValueError("P1 visual merge size must be positive")
    grids = model_inputs.get("image_grid_thw")
    image_valid = model_inputs.get("img_masks")
    if (
        grids is None
        or image_valid is None
        or getattr(grids, "ndim", None) != 3
        or getattr(image_valid, "ndim", None) != 2
        or grids.shape[:2] != image_valid.shape
        or grids.shape[-1] != 3
    ):
        raise ValueError("P1 visual lattice audit received malformed Qwen frame inputs")
    valid = image_valid.bool()
    valid_counts = valid.sum(dim=1).detach().cpu().tolist()
    if any(int(count) != 2 for count in valid_counts):
        raise RuntimeError("P1 requires exactly two valid CALVIN camera views")
    expected_grid = [1, visual_lattice * merge_size, visual_lattice * merge_size]
    actual_grids = grids[valid].detach().cpu().tolist()
    if any([int(value) for value in grid] != expected_grid for grid in actual_grids):
        raise RuntimeError("P1 Qwen image grid differs from its declared visual lattice")
    return {
        "valid_views_per_sample": 2,
        "merged_tokens_per_view": visual_lattice**2,
        "merged_visual_tokens_per_sample": 2 * visual_lattice**2,
    }


def _entity_evaluation_replay_seed(plan_sha256: str, sample_key: str) -> int:
    _require_sha256("entity evaluation plan SHA-256", plan_sha256)
    if not isinstance(sample_key, str) or not sample_key:
        raise ValueError("entity evaluation sample key must be a nonempty string")
    return int.from_bytes(
        hashlib.sha256(f"{plan_sha256}\0{sample_key}".encode("ascii")).digest()[:8],
        "big",
    )


def _canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    ).hexdigest()


def _optimizer_hparam_value(value: Any) -> object:
    if value is None or isinstance(value, str | bool | int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("optimizer hyperparameters must be finite")
        return value
    if isinstance(value, tuple | list):
        return [_optimizer_hparam_value(item) for item in value]
    return str(value)


def _shared_optimizer_manifest(
    *,
    policy: Any,
    optimizer: Any,
    expected_update_count: int,
) -> dict[str, Any]:
    """Bind shared LingBot parameter order and optimizer groups across ADR-175 arms."""

    if (
        isinstance(expected_update_count, bool)
        or not isinstance(expected_update_count, int)
        or expected_update_count <= 0
    ):
        raise ValueError("shared optimizer update count must be positive")
    parameter_names = {id(parameter): name for name, parameter in policy.named_parameters()}
    seen: set[str] = set()
    groups: list[dict[str, Any]] = []
    for group_index, group in enumerate(optimizer.param_groups):
        shared_parameters: list[dict[str, Any]] = []
        for group_position, parameter in enumerate(group["params"]):
            name = parameter_names.get(id(parameter))
            if name is None:
                raise RuntimeError("optimizer contains a parameter absent from the policy")
            if "picf_native_graph" in name:
                continue
            if name in seen:
                raise RuntimeError(f"shared optimizer parameter appears twice: {name}")
            seen.add(name)
            shared_parameters.append(
                {
                    "dtype": str(parameter.dtype),
                    "group_position": group_position,
                    "name": name,
                    "requires_grad": bool(parameter.requires_grad),
                    "shape": [int(value) for value in parameter.shape],
                }
            )
        if shared_parameters:
            groups.append(
                {
                    "group_index": group_index,
                    "hyperparameters": {
                        str(name): _optimizer_hparam_value(value)
                        for name, value in sorted(group.items())
                        if name != "params"
                    },
                    "parameters": shared_parameters,
                }
            )
    expected = {
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and "picf_native_graph" not in name
    }
    if seen != expected:
        missing = tuple(sorted(expected - seen))
        extra = tuple(sorted(seen - expected))
        raise RuntimeError(
            f"shared optimizer manifest differs from trainable LingBot parameters: "
            f"missing={missing}, extra={extra}"
        )
    return {
        "expected_update_count": expected_update_count,
        "groups": groups,
        "schema": "picf-next.adr175-shared-optimizer-manifest.v1",
        "shared_parameter_count": len(seen),
    }


def _adr175_rank_step_receipt(
    *,
    planned: Any,
    collated: Any,
) -> dict[str, str]:
    """Hash one rank's exact matched stream, target, flow and prompt inputs."""

    prompt_receipts = tuple(planned.physical_prompt_selection_receipts)
    if len(prompt_receipts) != 1:
        raise RuntimeError(
            "ADR-175 global-batch-two stream requires one raw prompt receipt per rank"
        )
    prompt_sha256 = prompt_receipts[0]
    _require_sha256("ADR-175 prompt-selection receipt", prompt_sha256)
    model_inputs = collated.model_inputs
    required = ("actions", "action_is_pad", "noise", "time")
    missing = tuple(name for name in required if name not in model_inputs)
    if missing:
        raise RuntimeError(f"ADR-175 model inputs omit matched tensors: {missing}")
    actions_sha256 = _tensor_sha256(model_inputs["actions"])
    action_is_pad_sha256 = _tensor_sha256(model_inputs["action_is_pad"])
    return {
        "sample_sha256": _canonical_json_sha256(
            {
                "sample_keys": list(collated.routing.sample_keys),
                "source_digest": collated.source_digest,
            }
        ),
        "action_target_sha256": _canonical_json_sha256(
            {
                "actions": actions_sha256,
                "action_is_pad": action_is_pad_sha256,
            }
        ),
        "noise_sha256": _tensor_sha256(model_inputs["noise"]),
        "time_sha256": _tensor_sha256(model_inputs["time"]),
        "prompt_sha256": prompt_sha256,
    }


def _mean_finite(values: list[float], *, name: str) -> float:
    if not values or any(not math.isfinite(value) for value in values):
        raise RuntimeError(f"ADR-175 {name} requires nonempty finite values")
    return math.fsum(values) / len(values)


def _distributed_phase_error(
    *,
    error: BaseException | None,
    phase: str,
    rank: int,
    dist_module: Any,
) -> None:
    """Exchange rank-local evaluation failures before the next FSDP forward."""

    local = (
        None
        if error is None
        else {
            "rank": rank,
            "phase": phase,
            "type": type(error).__name__,
            "message": str(error)[:4096],
        }
    )
    gathered: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
    dist_module.all_gather_object(gathered, local)
    failures = tuple(item for item in gathered if item is not None)
    if failures:
        rendered = "; ".join(
            f"rank {item['rank']} {item['phase']} {item['type']}: {item['message']}"
            for item in failures
        )
        raise RuntimeError(f"distributed P1 phase failed: {rendered}")


def _validate_args(args: argparse.Namespace) -> None:
    validate_fsdp2_placement(args.fsdp2_placement)
    joint_action_mode = _validate_current_frame_objective_weights(args)
    adr175_mode = args.adr175_arm is not None
    adr175_values = (args.adr175_contract, args.adr175_contract_sha256)
    if adr175_mode and any(value is None for value in adr175_values):
        raise ValueError("ADR-175 requires its broad-support contract and file digest")
    if not adr175_mode and any(value is not None for value in adr175_values):
        raise ValueError("ADR-175 contract inputs require --adr175-arm")
    required = {
        "checkpoint-dir": args.checkpoint_dir,
        "processor-dir": args.processor_dir,
        "dataset-split": args.dataset_split,
        "dataset-manifest": args.dataset_manifest,
        "norm-stats": args.norm_stats,
        "physical-sidecar-root": args.physical_sidecar_root,
        "physical-sidecar-manifest": args.physical_sidecar_manifest,
        "run-dir": args.run_dir,
        "output": args.output,
    }
    absent = sorted(name for name, value in required.items() if value is None)
    if absent:
        raise ValueError(f"P1 paths are absent: {absent}")
    curve_values = (
        args.stream_plan,
        args.stream_plan_sha256,
        args.representation_split,
        args.representation_split_sha256,
        args.entity_evaluation_plan,
        args.entity_evaluation_plan_sha256,
    )
    curve_mode = any(value is not None for value in curve_values)
    p2_values = (
        args.p2_predictive_cache_root,
        args.p2_predictive_cache_build_report,
        args.p2_predictive_cache_build_report_sha256,
    )
    staged_p2_values = (
        args.p2_stream_plan,
        args.p2_stream_plan_sha256,
        args.p2_representation_split,
        args.p2_representation_split_sha256,
    )
    causal_probe_values = (
        args.p2_causal_probe_cache_root,
        args.p2_causal_probe_cache_build_report,
        args.p2_causal_probe_cache_build_report_sha256,
        args.p2_causal_replay_closure,
        args.p2_causal_replay_closure_sha256,
    )
    p2_mode = any(value is not None for value in p2_values)
    if (
        isinstance(args.staged_p2_steps, bool)
        or not isinstance(args.staged_p2_steps, int)
        or not 0 <= args.staged_p2_steps <= _MAXIMUM_STAGED_P2_STEPS
    ):
        raise ValueError(f"staged P2 steps must be an integer in [0,{_MAXIMUM_STAGED_P2_STEPS}]")
    staged_mode = args.staged_p2_steps > 0
    causal_probe_mode = args.p2_causal_probe_steps > 0 or any(
        value is not None for value in causal_probe_values
    )
    if joint_action_mode and (p2_mode or staged_mode or causal_probe_mode):
        raise ValueError(
            "current-frame joint action is exclusive to P1 and cannot activate P2"
        )
    if adr175_mode:
        if not joint_action_mode:
            raise ValueError("ADR-175 requires the released action suffix")
        if not math.isclose(
            args.current_frame_action_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("ADR-175 fixes current-frame action weight at 1.0")
        if not math.isclose(
            args.current_frame_entity_weight,
            ADR175_ENTITY_WEIGHT,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError(
                f"ADR-175 fixes current-frame entity weight at {ADR175_ENTITY_WEIGHT}"
            )
        if args.capacity != 16 or args.maximum_control_tokens != 64:
            raise ValueError("ADR-175 fixes capacity=16 and maximum_control_tokens=64")
        if args.visual_lattice != 8:
            raise ValueError("ADR-175 fixes the shared LingBot visual lattice at 8")
    if (
        isinstance(args.p2_causal_probe_steps, bool)
        or not isinstance(args.p2_causal_probe_steps, int)
        or not 0 <= args.p2_causal_probe_steps <= _MAXIMUM_P2_CAUSAL_PROBE_STEPS
    ):
        raise ValueError(
            "P2 causal probe steps must be an integer in "
            f"[0,{_MAXIMUM_P2_CAUSAL_PROBE_STEPS}]"
        )
    if causal_probe_mode and args.p2_causal_probe_steps == 0:
        raise ValueError("P2 causal probe paths require positive probe steps")
    if args.p2_causal_probe_steps > 0 and any(
        value is None for value in causal_probe_values
    ):
        raise ValueError(
            "P2 causal probe requires its cache, replay closure, and registered digests"
        )
    if causal_probe_mode and not staged_mode:
        raise ValueError("P2 causal probe is defined only after staged P2 training")
    if args.p2_optimizer_update and not p2_mode:
        raise ValueError("P2 optimizer update requires predictive-cache mode")
    if staged_mode and not p2_mode:
        raise ValueError("staged P2 requires predictive-cache mode")
    if staged_mode and not curve_mode:
        raise ValueError("staged P2 requires the frozen P1 stream and split contracts")
    if staged_mode and any(value is None for value in staged_p2_values):
        raise ValueError("staged P2 requires its frozen stream plan and split contracts")
    if not staged_mode and any(value is not None for value in staged_p2_values):
        raise ValueError("P2 stream and split contracts are exclusive to staged mode")
    if staged_mode and args.p2_optimizer_update:
        raise ValueError("staged P2 performs its own updates and rejects the gate-only flag")
    if p2_mode and any(value is None for value in p2_values):
        raise ValueError("P2 gate requires its cache root, build report, and digest")
    if (
        p2_mode
        and not staged_mode
        and (curve_mode or args.steps != 1 or args.evaluation_steps is not None)
    ):
        raise ValueError("P2 gate is one step and cannot execute P1 curve evaluation")
    if p2_mode and (
        isinstance(args.p2_horizon, bool)
        or not isinstance(args.p2_horizon, int)
        or args.p2_horizon < 1
    ):
        raise ValueError("P2 horizon must be a positive integer")
    if args.p2_causal_horizon is None:
        args.p2_causal_horizon = args.p2_horizon
    if p2_mode and (
        isinstance(args.p2_causal_horizon, bool)
        or not isinstance(args.p2_causal_horizon, int)
        or args.p2_causal_horizon < 1
    ):
        raise ValueError("P2 causal horizon must be a positive integer")
    if curve_mode and any(value is None for value in curve_values):
        raise ValueError("P1 curve mode requires all plan, split, and digest arguments")
    evaluation_steps = _evaluation_steps(args.evaluation_steps)
    if curve_mode:
        maximum_curve_steps = (
            _MAXIMUM_ADR175_CURVE_STEPS if adr175_mode else _MAXIMUM_P1_CURVE_STEPS
        )
        if not 1 <= args.steps <= maximum_curve_steps:
            raise ValueError(
                f"P1 curve mode is bounded to 1..{maximum_curve_steps} optimizer steps"
            )
        if not evaluation_steps and not staged_mode:
            raise ValueError("P1 curve evaluation steps are required")
        if evaluation_steps and evaluation_steps[-1] != args.steps:
            raise ValueError("P1 curve evaluation must include the final step")
        if evaluation_steps and not joint_action_mode and evaluation_steps[0] != 0:
            raise ValueError("action-free P1 curve evaluation must include step zero")
        if any(step > args.steps for step in evaluation_steps):
            raise ValueError("P1 curve evaluation exceeds the optimizer-step budget")
        if adr175_mode:
            expected = tuple(step for step in ADR175_MILESTONES if step <= args.steps)
            if expected[-1] != args.steps:
                expected = (*expected, args.steps)
            if evaluation_steps != expected:
                raise ValueError(
                    "ADR-175 evaluation steps must be the registered milestones through "
                    "the requested final prefix"
                )
    else:
        if adr175_mode:
            raise ValueError("ADR-175 requires the frozen broad-support curve contracts")
        if args.evaluation_steps is not None:
            raise ValueError("P1 smoke mode cannot register curve evaluation steps")
        if not 1 <= args.steps <= _MAXIMUM_P1_STEPS:
            raise ValueError(f"P1 is bounded to 1..{_MAXIMUM_P1_STEPS} optimizer steps")
    if (
        isinstance(args.evaluation_visuals_per_partition, bool)
        or not isinstance(args.evaluation_visuals_per_partition, int)
        or args.evaluation_visuals_per_partition < 0
    ):
        raise ValueError("P1 evaluation visual count must be a non-negative integer")

    files = (
        args.patch,
        args.training_config,
        args.robot_config,
        args.data_config,
        args.dataset_manifest,
        args.norm_stats,
        args.physical_sidecar_manifest,
        *((args.adr175_contract,) if args.adr175_contract is not None else ()),
        *((args.stream_plan,) if args.stream_plan is not None else ()),
        *((args.representation_split,) if args.representation_split is not None else ()),
        *((args.entity_evaluation_plan,) if args.entity_evaluation_plan is not None else ()),
        *((args.p2_stream_plan,) if args.p2_stream_plan is not None else ()),
        *((args.p2_representation_split,) if args.p2_representation_split is not None else ()),
        *(
            (args.p2_predictive_cache_build_report,)
            if args.p2_predictive_cache_build_report is not None
            else ()
        ),
        *(
            (args.p2_causal_probe_cache_build_report,)
            if args.p2_causal_probe_cache_build_report is not None
            else ()
        ),
        *((args.p2_causal_replay_closure,) if args.p2_causal_replay_closure is not None else ()),
    )
    directories = (
        args.source_checkout,
        args.checkpoint_dir,
        args.processor_dir,
        args.dataset_split,
        args.physical_sidecar_root,
        args.run_dir,
        *((args.p2_predictive_cache_root,) if args.p2_predictive_cache_root is not None else ()),
        *(
            (args.p2_causal_probe_cache_root,)
            if args.p2_causal_probe_cache_root is not None
            else ()
        ),
    )
    if any(not Path(path).is_file() for path in files):
        raise FileNotFoundError("one or more P1 source/config/data files are absent")
    if any(not Path(path).is_dir() for path in directories):
        raise FileNotFoundError("one or more P1 source/model/data directories are absent")
    if args.output.parent.resolve() != args.run_dir.resolve():
        raise ValueError("P1 output must be a direct child of its persistent run directory")
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    integers = (
        args.steps,
        args.seed,
        args.capacity,
        args.maximum_control_tokens,
    )
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise TypeError("P1 integer controls must be Python integers")
    if args.seed < 0 or min(integers[2:]) <= 0:
        raise ValueError("P1 seed and dimensions are outside their valid ranges")
    if args.seed > 0xFFFFFFFF - (P1_WORLD_SIZE - 1):
        raise ValueError("P1 rank seeds must fit NumPy's uint32 domain")
    for name in (
        "learning_rate",
        "max_grad_norm",
        "maximum_peak_reserved_gib",
    ):
        value = getattr(args, name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"P1 {name} must be finite and positive")
    _validate_entity_loss_weights(args)
    fraction = args.minimum_supervised_fraction
    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, (int, float))
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("P1 minimum supervised fraction must lie in [0,1]")
    _require_sha256(
        "physical sidecar manifest SHA-256",
        args.physical_sidecar_manifest_sha256,
    )
    if curve_mode:
        _require_sha256("stream plan file SHA-256", args.stream_plan_sha256)
        _require_sha256(
            "representation split file SHA-256",
            args.representation_split_sha256,
        )
        _require_sha256(
            "entity evaluation plan file SHA-256",
            args.entity_evaluation_plan_sha256,
        )
    if adr175_mode:
        _require_sha256("ADR-175 contract file SHA-256", args.adr175_contract_sha256)
    if p2_mode:
        _require_sha256(
            "P2 predictive cache build-report SHA-256",
            args.p2_predictive_cache_build_report_sha256,
        )
    if staged_mode:
        _require_sha256("P2 stream plan file SHA-256", args.p2_stream_plan_sha256)
        _require_sha256(
            "P2 representation split file SHA-256",
            args.p2_representation_split_sha256,
        )
    if causal_probe_mode:
        _require_sha256(
            "P2 causal cache build-report SHA-256",
            args.p2_causal_probe_cache_build_report_sha256,
        )
        _require_sha256(
            "P2 causal replay closure SHA-256",
            args.p2_causal_replay_closure_sha256,
        )


def _implementation_provenance(root: Path) -> tuple[dict[str, str], str]:
    relative = (
        "configs/cloud/adr175_matched_arm.sh",
        "configs/lingbot/calvin_data.json",
        "configs/lingbot/calvin_robot.yaml",
        "references/patches/lingbot_vla2_picf_native.patch",
        "src/picf_next/eval/calvin_task_relevance.py",
        "src/picf_next/lingbot_native/adr175_contract.py",
        "src/picf_next/lingbot_native/calvin.py",
        "src/picf_next/lingbot_native/calvin_entity_set.py",
        "src/picf_next/lingbot_native/calvin_entity_training.py",
        "src/picf_next/lingbot_native/controls.py",
        "src/picf_next/lingbot_native/entity_evaluation_plan.py",
        "src/picf_next/lingbot_native/entity_set_objective.py",
        "src/picf_next/lingbot_native/entity_set_evaluation.py",
        "src/picf_next/lingbot_native/entity_training.py",
        "src/picf_next/lingbot_native/fsdp2_placement.py",
        "src/picf_next/lingbot_native/graph.py",
        "src/picf_next/lingbot_native/host.py",
        "src/picf_next/lingbot_native/lattice_feasibility.py",
        "src/picf_next/lingbot_native/modalities.py",
        "src/picf_next/lingbot_native/physical_relations.py",
        "src/picf_next/lingbot_native/physical_sequence.py",
        "src/picf_next/lingbot_native/prediction.py",
        "src/picf_next/lingbot_native/predictive_cache.py",
        "src/picf_next/lingbot_native/predictive_objective.py",
        "src/picf_next/lingbot_native/predictive_probes.py",
        "src/picf_next/lingbot_native/representation_split.py",
        "src/picf_next/lingbot_native/representation_stage.py",
        "src/picf_next/lingbot_native/row_binding.py",
        "src/picf_next/lingbot_native/state.py",
        "src/picf_next/lingbot_native/temporal.py",
        "src/picf_next/lingbot_native/training.py",
        "src/picf_next/lingbot_native/visual_audit.py",
        "src/picf_next/training/control.py",
        "tools/build_adr175_broad_support_contract.py",
        "tools/lingbot_vla2_runtime_helpers.py",
        "tools/run_lingbot_vla2_native_g0.py",
        "tools/run_lingbot_vla2_task_independent_p1.py",
    )
    hashes = {path: _sha256(root / path) for path in relative}
    digest = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return hashes, digest


def _float(value: Any) -> float:
    result = float(value.detach().float().item())
    if not math.isfinite(result):
        raise RuntimeError("P1 report encountered a non-finite scalar")
    return result


def _optimizer_state_family_counts(model: Any, optimizer: Any) -> dict[str, int]:
    """Count optimizer state without activating dormant parameter families."""

    state = getattr(optimizer, "state", None)
    if not isinstance(state, Mapping) or not state:
        raise RuntimeError("representation optimizer has no continuation state")
    fragments = {
        "native_graph": "picf_native_graph",
        "vlm_host": ".qwenvl.",
        "predictive_readout": "predictive_readouts.dino_video",
        "action_expert": ".qwen_expert.",
    }
    counts = {**{family: 0 for family in fragments}, "action_only": 0}
    for name, parameter in model.named_parameters():
        entry = state.get(parameter)
        if entry is None:
            continue
        if not isinstance(entry, Mapping):
            raise RuntimeError("representation optimizer parameter state is not a mapping")
        if not entry:
            continue
        for family, fragment in fragments.items():
            if fragment in name:
                counts[family] += 1
        if is_native_action_only_parameter(name):
            counts["action_only"] += 1
    return counts


def _audit_optimizer_family_state(
    model: Any,
    optimizer: Any,
    torch_module: Any,
    *,
    family: str,
    fragment: str,
    expected_adamw_step: int | None,
    dist_module: Any | None = None,
) -> dict[str, int | float | None]:
    """Validate finite optimizer moments globally across possible empty FSDP shards."""

    if not family or not fragment:
        raise ValueError("optimizer family audit requires nonempty names")
    if expected_adamw_step is not None and (
        isinstance(expected_adamw_step, bool)
        or not isinstance(expected_adamw_step, int)
        or expected_adamw_step <= 0
    ):
        raise ValueError("expected AdamW step must be a positive integer")
    entries = 0
    adamw_entries = 0
    muon_entries = 0
    moment_elements = 0
    nonzero_moment_elements = 0
    adamw_steps: list[float] = []
    local_error: BaseException | None = None
    try:
        state = getattr(optimizer, "state", None)
        if not isinstance(state, Mapping):
            raise RuntimeError("representation optimizer continuation state is not a mapping")
        for name, parameter in model.named_parameters():
            if fragment not in name:
                continue
            entry = state.get(parameter)
            if entry is None or (isinstance(entry, Mapping) and not entry):
                continue
            if not isinstance(entry, Mapping):
                raise RuntimeError(f"{family} optimizer state entry is not a mapping")
            entries += 1
            if "step" in entry:
                step = entry["step"]
                with torch_module.no_grad():
                    local_step = (
                        step.to_local() if callable(getattr(step, "to_local", None)) else step
                    )
                if (
                    not torch_module.is_tensor(local_step)
                    or local_step.numel() != 1
                    or not torch_module.isfinite(local_step).all()
                ):
                    raise RuntimeError(f"{family} AdamW step is invalid")
                measured_step = float(local_step.item())
                if measured_step <= 0 or (
                    expected_adamw_step is not None and measured_step != float(expected_adamw_step)
                ):
                    raise RuntimeError(f"{family} AdamW step differs from its phase clock")
                adamw_steps.append(measured_step)
                fields = ("exp_avg", "exp_avg_sq")
                adamw_entries += 1
            elif "momentum_buffer" in entry:
                fields = ("momentum_buffer",)
                muon_entries += 1
            else:
                raise RuntimeError(f"{family} optimizer entry is neither AdamW nor Muon")
            for field in fields:
                value = entry.get(field)
                if value is None or not torch_module.is_tensor(value):
                    raise RuntimeError(f"{family} optimizer state omits {field}")
                with torch_module.no_grad():
                    local = (
                        value.to_local() if callable(getattr(value, "to_local", None)) else value
                    )
                if not torch_module.isfinite(local).all():
                    raise RuntimeError(f"{family} optimizer moment {field} is non-finite")
                moment_elements += int(local.numel())
                nonzero_moment_elements += int(torch_module.count_nonzero(local).item())
    except BaseException as error:
        local_error = error

    local_payload = {
        "error": (
            None
            if local_error is None
            else {"type": type(local_error).__name__, "message": str(local_error)[:4096]}
        ),
        "entries": entries,
        "adamw_entries": adamw_entries,
        "muon_entries": muon_entries,
        "moment_elements": moment_elements,
        "nonzero_moment_elements": nonzero_moment_elements,
        "adamw_steps": adamw_steps,
    }
    gathered: list[Any] = [local_payload]
    if dist_module is not None:
        world_size = int(dist_module.get_world_size())
        gathered = [None for _ in range(world_size)]
        dist_module.all_gather_object(gathered, local_payload)
    failures = tuple(
        (rank, payload["error"])
        for rank, payload in enumerate(gathered)
        if payload["error"] is not None
    )
    if failures:
        rendered = "; ".join(
            f"rank {rank} {error['type']}: {error['message']}" for rank, error in failures
        )
        raise RuntimeError(f"{family} optimizer-state audit failed: {rendered}")

    global_entries = sum(int(payload["entries"]) for payload in gathered)
    global_moment_elements = sum(int(payload["moment_elements"]) for payload in gathered)
    global_nonzero_moment_elements = sum(
        int(payload["nonzero_moment_elements"]) for payload in gathered
    )
    global_adamw_steps = [float(step) for payload in gathered for step in payload["adamw_steps"]]
    if global_entries <= 0 or global_moment_elements <= 0 or global_nonzero_moment_elements <= 0:
        raise RuntimeError(f"{family} optimizer state has no global nontrivial finite moment")
    return {
        "entries": global_entries,
        "adamw_entries": sum(int(payload["adamw_entries"]) for payload in gathered),
        "muon_entries": sum(int(payload["muon_entries"]) for payload in gathered),
        "local_moment_elements": moment_elements,
        "local_nonzero_moment_elements": nonzero_moment_elements,
        "global_moment_elements": global_moment_elements,
        "global_nonzero_moment_elements": global_nonzero_moment_elements,
        "adamw_step_minimum": min(global_adamw_steps) if global_adamw_steps else None,
        "adamw_step_maximum": max(global_adamw_steps) if global_adamw_steps else None,
    }


def _p2_optimizer_state_families(
    model: Any,
    optimizer: Any,
    *,
    require_predictive: bool = True,
) -> dict[str, int]:
    """Require P2 state while preserving the disabled action family."""

    if not isinstance(require_predictive, bool):
        raise TypeError("P2 predictive-state requirement must be boolean")
    counts = _optimizer_state_family_counts(model, optimizer)
    required = ["native_graph", "vlm_host"]
    if require_predictive:
        required.append("predictive_readout")
    for family in required:
        if counts[family] <= 0:
            raise RuntimeError(f"P2 optimizer state omits the {family} family")
    if counts["action_only"] != 0:
        raise RuntimeError("P2 optimizer created state for the disabled action-only partition")
    return counts


def _publish_p2_update_stage(*, run_dir: Path, rank: int, stage: str) -> None:
    """Publish rank-local progress around the first distributed optimizer update."""

    if not stage or any(character not in "abcdefghijklmnopqrstuvwxyz_" for character in stage):
        raise ValueError("P2 update stage must use lowercase ASCII letters and underscores")
    path = run_dir / f"p2_update_rank_{rank}.stage"
    temporary = path.with_suffix(".stage.tmp")
    temporary.write_text(stage + "\n", encoding="ascii")
    temporary.replace(path)
    print(f"[P2 update][rank={rank}] {stage}", flush=True)


def _synchronize_p2_update_ranks(
    *,
    device: Any,
    dist_module: Any,
    torch_module: Any,
) -> None:
    """Order local CUDA work before the next rank-symmetric update phase."""

    if getattr(device, "type", None) != "cuda" or getattr(device, "index", None) is None:
        raise ValueError("P2 distributed update synchronization requires an indexed CUDA device")
    torch_module.cuda.synchronize(device)
    dist_module.barrier(device_ids=[device.index])


def _load_predictive_build_report(
    *,
    path: Path | None,
    expected_sha256: str | None,
    cache_root: Path | None,
    name: str,
) -> dict[str, Any] | None:
    if path is None:
        return None
    if _sha256(path) != expected_sha256:
        raise ValueError(f"{name} build report differs from its registered digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} build report is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != _PREDICTIVE_BUILD_REPORT_FIELDS:
        raise ValueError(f"{name} build report fields differ from schema")
    for field_name in _PREDICTIVE_BUILD_REPORT_FIELDS - {
        "expected_record_count",
        "output_root",
    }:
        _require_sha256(f"{name} report {field_name}", value[field_name])
    if (
        isinstance(value["expected_record_count"], bool)
        or not isinstance(value["expected_record_count"], int)
        or value["expected_record_count"] <= 0
    ):
        raise ValueError(f"{name} expected record count must be positive")
    if cache_root is None or Path(value["output_root"]).resolve() != cache_root.resolve():
        raise ValueError(f"{name} report and cache root differ")
    return value


def _load_p2_predictive_build_report(args: argparse.Namespace) -> dict[str, Any] | None:
    return _load_predictive_build_report(
        path=args.p2_predictive_cache_build_report,
        expected_sha256=args.p2_predictive_cache_build_report_sha256,
        cache_root=args.p2_predictive_cache_root,
        name="P2 predictive cache",
    )


def _load_p2_causal_build_report(args: argparse.Namespace) -> dict[str, Any] | None:
    return _load_predictive_build_report(
        path=args.p2_causal_probe_cache_build_report,
        expected_sha256=args.p2_causal_probe_cache_build_report_sha256,
        cache_root=args.p2_causal_probe_cache_root,
        name="P2 causal cache",
    )


def _load_p2_causal_replay_closure(args: argparse.Namespace) -> dict[str, Any] | None:
    path = args.p2_causal_replay_closure
    if path is None:
        return None
    if _sha256(path) != args.p2_causal_replay_closure_sha256:
        raise ValueError("P2 causal replay closure differs from its registered digest")
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("P2 causal replay closure is not valid JSON") from error
    if not isinstance(value, dict) or set(value) != _P2_CAUSAL_REPLAY_CLOSURE_FIELDS:
        raise ValueError("P2 causal replay closure fields differ from schema")
    if value["schema"] != P2_CAUSAL_REPLAY_CLOSURE_SCHEMA:
        raise ValueError("P2 causal replay closure schema changed")
    claimed_artifact = _require_sha256(
        "P2 causal replay closure artifact SHA-256",
        value["artifact_sha256"],
    )
    canonical = dict(value)
    del canonical["artifact_sha256"]
    computed_artifact = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if claimed_artifact != computed_artifact:
        raise ValueError("P2 causal replay closure artifact digest is inconsistent")
    for name in (
        "dataset_manifest_sha256",
        "predictive_cache_manifest_sha256",
        "representation_split_file_sha256",
        "training_source_episode_indices_sha256",
    ):
        _require_sha256(f"P2 causal replay closure {name}", value[name])
    if (
        value["dataset_manifest_sha256"] != _sha256(args.dataset_manifest)
        or value["representation_split_file_sha256"] != _sha256(args.representation_split)
        or value["predictive_cache_manifest_sha256"]
        != _sha256(args.p2_causal_probe_cache_root / "manifest.json")
    ):
        raise ValueError("P2 causal replay closure belongs to another immutable input")
    integer_contract = {
        "count": args.p2_causal_probe_steps * P1_WORLD_SIZE,
        # Callers that predate the separate causal horizon pass only --p2-horizon,
        # and for them the two are the same number.
        "horizon": getattr(args, "p2_causal_horizon", None) or args.p2_horizon,
        "prefix_frames": 2,
        "selection_seed": args.seed + 2_000_003,
    }
    if any(value[name] != expected for name, expected in integer_contract.items()):
        raise ValueError("P2 causal replay closure numeric contract changed")
    if value["selection_domain"] != "all-nontraining":
        raise ValueError("P2 causal replay closure is not source-disjoint from training")
    if value["missing_paths"] != []:
        raise ValueError("P2 causal replay closure still reports missing source files")
    required_paths = value["required_paths"]
    if (
        not isinstance(required_paths, list)
        or not required_paths
        or required_paths != sorted(set(required_paths))
        or any(
            not isinstance(relative, str)
            or len(relative) != len("episode_0000000.npz")
            or not relative.startswith("episode_")
            or not relative.endswith(".npz")
            or not relative[8:15].isdigit()
            for relative in required_paths
        )
    ):
        raise ValueError("P2 causal replay closure has invalid required paths")
    selections = value["selections"]
    selection_fields = {
        "replay_global_indices",
        "required_global_indices",
        "segment_index",
        "source_episode_index",
        "source_global_index",
        "target_global_index",
        "transition_index",
    }
    if (
        not isinstance(selections, list)
        or len(selections) != value["count"]
        or any(not isinstance(item, dict) or set(item) != selection_fields for item in selections)
    ):
        raise ValueError("P2 causal replay closure has invalid selections")
    if (
        len({int(item["source_global_index"]) for item in selections}) != len(selections)
        or len({int(item["source_episode_index"]) for item in selections})
        != len(selections)
    ):
        raise ValueError("P2 causal replay closure repeats a source frame or episode")
    closure_paths = sorted(
        {
            f"episode_{int(index):07d}.npz"
            for item in selections
            for index in item["required_global_indices"]
        }
    )
    if closure_paths != required_paths:
        raise ValueError("P2 causal replay closure required-path union is inconsistent")
    roots = value["available_roots"]
    if not isinstance(roots, list) or not roots or any(
        not isinstance(root, str) or not root for root in roots
    ):
        raise ValueError("P2 causal replay closure omits its available roots")
    return value


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    curve_mode = args.stream_plan is not None
    adr175_mode = args.adr175_arm is not None
    adr175_picf_active = args.adr175_arm in {"physical-set", "native-attention"}
    adr175_attention_active = args.adr175_arm == "native-attention"
    p2_mode = args.p2_predictive_cache_root is not None
    staged_mode = args.staged_p2_steps > 0
    p2_gate_mode = p2_mode and not staged_mode
    p2_predictive_report = _load_p2_predictive_build_report(args)
    p2_causal_report = _load_p2_causal_build_report(args)
    p2_causal_replay_closure = _load_p2_causal_replay_closure(args)
    registered_evaluation_steps = _evaluation_steps(args.evaluation_steps)
    require_persistent_run_root(args.run_dir)
    if _BOOTSTRAPPED_CUDA_ALLOCATOR is None:
        _configure_cuda_allocator(args.cuda_allocator)
    elif args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("P1 CUDA allocator pre-bootstrap differs from parsed arguments")

    root = Path(__file__).resolve().parents[1]
    patch_report = verify_native_patch(root=root, checkout=args.source_checkout, check_apply=True)
    prepared_source = validate_prepared_native_source(
        checkout=args.source_checkout,
        patch_path=args.patch,
    )
    expected_hashes = patch_report.get("patched_source_sha256")
    actual_hashes = prepared_source.get("patched_source_sha256")
    if not isinstance(expected_hashes, dict) or actual_hashes != expected_hashes:
        raise RuntimeError("P1 LingBot source differs from immutable patch replay")
    implementation_files, implementation_sha256 = _implementation_provenance(root)
    adr175_contract = None
    if adr175_mode:
        if _sha256(args.adr175_contract) != args.adr175_contract_sha256:
            raise ValueError("ADR-175 broad-support contract file SHA-256 differs")
        adr175_contract = Adr175BroadSupportContract.load(args.adr175_contract)
        if adr175_contract.training_prefix_steps < args.steps:
            raise ValueError("ADR-175 requested prefix exceeds its frozen support contract")

    if os.environ.get("WORLD_SIZE") != str(P1_WORLD_SIZE):
        raise RuntimeError("P1 requires torchrun with exactly two processes")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(P1_WORLD_SIZE):
        raise RuntimeError("P1 requires both processes on one two-GPU host")

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

    from picf_next.data.calvin import (
        CalvinDatasetIndex,
        CalvinPhysicalTransitionDataset,
        CalvinStatefulTransitionDataset,
    )
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.calvin_physical_supervision_schema import (
        CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        DatasetFileManifest,
        load_dataset_file_manifest,
        read_verified_dataset_file,
        validate_dataset_runtime_binding,
    )
    from picf_next.lingbot_native.calvin import (
        CollatedNativeCALVINBatch,
        PlannedNativeCALVINReplayBatch,
        build_native_calvin_physical_episode_domain,
        build_native_calvin_physical_sample_domain,
        build_native_calvin_stream_plan,
        build_native_calvin_replay_batch,
        build_planned_native_calvin_batch,
        collate_native_calvin_training_batch,
        materialize_native_flow_randomness,
    )
    from picf_next.lingbot_native.calvin_entity_set import (
        build_task_independent_calvin_targets,
        physical_frame_predictions_from_relation,
        physical_frame_row_bindings,
    )
    from picf_next.lingbot_native.calvin_entity_training import (
        run_task_independent_calvin_current_frame_diagnostic,
        run_task_independent_calvin_current_frame_objective,
        run_task_independent_calvin_sequence_objective,
    )
    from picf_next.lingbot_native.action_posterior_collector import (
        RegisteredActionPosteriorReceiptCollector,
    )
    from picf_next.lingbot_native.action_posterior_learning import (
        action_posterior_target_mass_loss,
    )
    from picf_next.lingbot_native.entity_training import (
        TaskIndependentEntityObjectiveConfig,
    )
    from picf_next.lingbot_native.controls import ExecutedControlBatch
    from picf_next.lingbot_native.entity_evaluation_plan import (
        ENTITY_EVALUATION_PARTITIONS,
        EntityEvaluationPlan,
        build_entity_evaluation_plan,
    )
    from picf_next.lingbot_native.entity_set_evaluation import (
        evaluate_physical_entity_frame,
        summarize_entity_evaluation_partition,
    )
    from picf_next.lingbot_native.host import (
        TASK_INDEPENDENT_ENTITY_POSTERIOR,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        install_lingbot_native_graph,
    )
    from picf_next.lingbot_native.lattice_feasibility import (
        configure_native_processor_lattice,
    )
    from picf_next.lingbot_native.physical_relations import (
        PhysicalRelationOutput,
        TASK_INDEPENDENT_PHYSICAL_INTERFACE,
    )
    from picf_next.lingbot_native.physical_sequence import (
        match_physical_sequence_entities,
    )
    from picf_next.lingbot_native.prediction import (
        PredictionSource,
        make_native_future_request,
    )
    from picf_next.lingbot_native.predictive_cache import (
        LINGBOT_PREDICTIVE_TARGET_SPACE,
        LingBotPredictiveTargetCache,
        native_predictive_query_schema_digest,
    )
    from picf_next.lingbot_native.predictive_probes import (
        predictive_future_counterfactual_diagnostics,
        run_native_future_counterfactual_forwards,
    )
    from picf_next.lingbot_native.causal_evidence import (
        matched_noise_rows,
        predictive_evidence_mass,
        state_manipulation_strength,
    )
    from picf_next.lingbot_native.state import NativePosteriorState
    from picf_next.lingbot_native.temporal import rollout_native_prior_prediction
    from picf_next.lingbot_native.training import (
        audit_native_optimizer_coverage,
        run_official_policy_diagnostic_forward,
        run_official_policy_training_forward,
    )
    from picf_next.eval.calvin_task_relevance import calvin_task_physical_relevance
    from picf_next.lingbot_native.representation_split import (
        RepresentationTrialSplit,
        verify_representation_trial_split_training_evidence,
    )
    from picf_next.training.control import (
        EpisodeSampleSequence,
        FrozenSamplePlan,
        load_frozen_episode_stream_plan,
    )
    from picf_next.lingbot_native.visual_audit import (
        TASK_INDEPENDENT_ENTITY_VISUAL_SCHEMA,
        render_task_independent_entity_visuals,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    run_lease = None
    try:
        run_lease = acquire_distributed_run_lease(args.run_dir, rank=rank, distributed=dist)
        if torch.cuda.device_count() != P1_WORLD_SIZE:
            raise RuntimeError("P1 process sees a CUDA topology other than two devices")
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("P1 requires two A100 devices with at least 39 GiB each")

        artifact_contract: list[Any] = [None]
        if rank == 0:
            try:
                artifact_contract[0] = {
                    "status": "PASS",
                    "checkpoint": validate_checkpoint(args.checkpoint_dir),
                    "processor": validate_processor(args.processor_dir),
                }
            except BaseException as error:
                artifact_contract[0] = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(artifact_contract, src=0)
        artifact_contract_report = artifact_contract[0]
        if (
            not isinstance(artifact_contract_report, dict)
            or artifact_contract_report.get("status") != "PASS"
        ):
            raise RuntimeError(f"P1 model artifact contract failed: {artifact_contract_report}")
        checkpoint_report = artifact_contract_report["checkpoint"]
        processor_report = artifact_contract_report["processor"]

        dataset_contract: list[Any] = [None]
        rank_zero_manifest: DatasetFileManifest | None = None
        if rank == 0:
            try:
                rank_zero_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
                norm_stats = json.loads(args.norm_stats.read_text())
                validate_lingbot_calvin_norm_stats(norm_stats)
                source = norm_stats["source"]
                if (
                    source["dataset_id"] != rank_zero_manifest.dataset_id
                    or source["dataset_revision"] != rank_zero_manifest.dataset_revision
                    or source["dataset_tree_sha256"] != rank_zero_manifest.tree_sha256
                    or rank_zero_manifest.split_name != args.dataset_split.name
                ):
                    raise ValueError("P1 CALVIN manifest and normalization differ")
                verified_causal_files = 0
                if p2_causal_replay_closure is not None:
                    for relative in p2_causal_replay_closure["required_paths"]:
                        record = rank_zero_manifest.record_for(relative)
                        read_verified_dataset_file(
                            rank_zero_manifest,
                            args.dataset_split,
                            relative,
                            maximum_bytes=max(record.size_bytes, 1),
                        )
                        verified_causal_files += 1
                dataset_contract[0] = {
                    "status": "PASS",
                    "manifest_sha256": _sha256(args.dataset_manifest),
                    "normalization_sha256": _sha256(args.norm_stats),
                    "validation": validate_dataset_runtime_binding(
                        rank_zero_manifest,
                        args.dataset_split,
                        dataset_id=source["dataset_id"],
                        dataset_revision=source["dataset_revision"],
                        split_name=args.dataset_split.name,
                    ),
                    "verified_causal_replay_files": verified_causal_files,
                }
            except BaseException as error:
                dataset_contract[0] = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(dataset_contract, src=0)
        dataset_contract_report = dataset_contract[0]
        if (
            not isinstance(dataset_contract_report, dict)
            or dataset_contract_report.get("status") != "PASS"
        ):
            raise RuntimeError(f"P1 dataset contract failed: {dataset_contract_report}")
        dataset_manifest = (
            rank_zero_manifest
            if rank_zero_manifest is not None
            else load_dataset_file_manifest(args.dataset_manifest.resolve())
        )
        index = CalvinDatasetIndex.load(
            args.dataset_split.resolve(),
            dataset_id=dataset_manifest.dataset_id,
            dataset_revision=dataset_manifest.dataset_revision,
            verify_files=False,
            dataset_manifest=dataset_manifest,
        )
        physical_sidecar = CalvinPhysicalSupervisionSidecar(
            args.physical_sidecar_root,
            index,
            manifest_path=args.physical_sidecar_manifest,
            expected_manifest_sha256=args.physical_sidecar_manifest_sha256,
        )
        if physical_sidecar.coverage != CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES:
            raise RuntimeError("P1 requires all-source physical supervision")

        predictive_cache = None
        causal_predictive_cache = None
        if p2_mode:
            if p2_predictive_report is None or args.p2_predictive_cache_root is None:
                raise RuntimeError("P2 cache mode lost its validated build report")
            cache_manifest = args.p2_predictive_cache_root / "manifest.json"
            if _sha256(cache_manifest) != p2_predictive_report["cache_manifest_sha256"]:
                raise ValueError("P2 predictive cache manifest differs from its build report")
            cache_manifest_value = json.loads(cache_manifest.read_text(encoding="ascii"))
            cache_contract = cache_manifest_value.get("contract")
            if not isinstance(cache_contract, dict):
                raise ValueError("P2 predictive cache manifest omitted its contract")
            cache_horizons = tuple(cache_contract.get("horizons", ()))
            if args.p2_horizon not in cache_horizons_as_ints(cache_horizons):
                raise ValueError(
                    f"the predictive cache publishes horizons {cache_horizons} and cannot "
                    f"serve horizon {args.p2_horizon}"
                )
            query_schema_sha256 = native_predictive_query_schema_digest(
                target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
                route_id=0,
                horizons=cache_horizons,
            )
            predictive_cache = LingBotPredictiveTargetCache.load(
                args.p2_predictive_cache_root,
                manifest_sha256=p2_predictive_report["cache_manifest_sha256"],
                dataset_tree_sha256=dataset_manifest.tree_sha256,
                physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
                encoder_digest=p2_predictive_report["teacher_encoder_digest"],
                query_schema_sha256=query_schema_sha256,
                coverage_sha256=p2_predictive_report["coverage_sha256"],
                memory_capacity=1,
            )
            if (
                predictive_cache.contract.lingbot_source_commit != LINGBOT_NATIVE_SOURCE_COMMIT
                or predictive_cache.contract.lingbot_checkpoint_revision
                != LINGBOT_CHECKPOINT_REVISION
                or predictive_cache.contract.expected_record_count
                != p2_predictive_report["expected_record_count"]
            ):
                raise RuntimeError("P2 predictive cache provenance differs from released LingBot")
        if args.p2_causal_probe_steps > 0:
            if p2_causal_report is None or args.p2_causal_probe_cache_root is None:
                raise RuntimeError("P2 causal mode lost its validated predictive cache")
            causal_manifest = args.p2_causal_probe_cache_root / "manifest.json"
            if _sha256(causal_manifest) != p2_causal_report["cache_manifest_sha256"]:
                raise ValueError("P2 causal cache manifest differs from its build report")
            causal_manifest_value = json.loads(causal_manifest.read_text(encoding="ascii"))
            causal_contract = causal_manifest_value.get("contract")
            if not isinstance(causal_contract, dict):
                raise ValueError("P2 causal cache manifest omitted its contract")
            causal_horizons = tuple(causal_contract.get("horizons", ()))
            if args.p2_causal_horizon not in cache_horizons_as_ints(causal_horizons):
                raise ValueError(
                    f"the causal cache publishes horizons {causal_horizons} and cannot "
                    f"serve horizon {args.p2_causal_horizon}"
                )
            causal_query_schema_sha256 = native_predictive_query_schema_digest(
                target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
                route_id=0,
                horizons=causal_horizons,
            )
            causal_predictive_cache = LingBotPredictiveTargetCache.load(
                args.p2_causal_probe_cache_root,
                manifest_sha256=p2_causal_report["cache_manifest_sha256"],
                dataset_tree_sha256=dataset_manifest.tree_sha256,
                physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
                encoder_digest=p2_causal_report["teacher_encoder_digest"],
                query_schema_sha256=causal_query_schema_sha256,
                coverage_sha256=p2_causal_report["coverage_sha256"],
                memory_capacity=1,
            )
            if (
                causal_predictive_cache.contract.lingbot_source_commit
                != LINGBOT_NATIVE_SOURCE_COMMIT
                or causal_predictive_cache.contract.lingbot_checkpoint_revision
                != LINGBOT_CHECKPOINT_REVISION
                or causal_predictive_cache.contract.expected_record_count
                != p2_causal_report["expected_record_count"]
            ):
                raise RuntimeError("P2 causal cache provenance differs from released LingBot")
            if p2_predictive_report is None or any(
                p2_causal_report[field] != p2_predictive_report[field]
                for field in (
                    "teacher_encoder_digest",
                    "physical_visual_acceptance_sha256",
                )
            ):
                raise RuntimeError("P2 training and causal caches use different frozen teachers")

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)
        init_parallel_state(
            dp_size=P1_WORLD_SIZE,
            dp_replicate_size=1,
            dp_shard_size=P1_WORLD_SIZE,
            tp_size=1,
            ep_size=1,
            pp_size=1,
            cp_size=1,
            ulysses_size=1,
            dp_mode="fsdp2",
        )
        training = load_lingbot_training_config(args.training_config)
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=args.learning_rate,
        )
        require_lingbot_exact_resume_contract(optimizer_contract)
        shared_optimizer_contract_sha256: str | None = None
        shared_optimizer_manifest: dict[str, Any] | None = None
        merged, _ = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=args.steps + args.staged_p2_steps,
        )
        merged["use_cache"] = False
        merged["use_compile"] = False
        merged["attention_implementation"] = "eager"
        merged["vit_attn_implementation"] = "eager"
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        if bool(config.train_expert_only) or bool(config.freeze_vision_encoder):
            raise RuntimeError("P1 requires the complete trainable VLM host")
        qwen_config = AutoConfig.from_pretrained(  # nosec B615
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        patch_size = int(qwen_config.vision_config.patch_size)
        merge_size = int(qwen_config.vision_config.spatial_merge_size)
        if patch_size <= 0 or merge_size <= 0:
            raise RuntimeError("P1 loaded invalid Qwen vision geometry")
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())
        if (
            p2_causal_replay_closure is not None
            and p2_causal_replay_closure["action_horizon"] != int(config.chunk_size)
        ):
            raise ValueError("P2 causal replay closure uses another action horizon")
        evaluation_dataset = CalvinStatefulTransitionDataset(
            index,
            action_horizon=config.chunk_size,
        )
        dataset = (
            CalvinPhysicalTransitionDataset(index, action_horizon=config.chunk_size)
            if adr175_mode
            else evaluation_dataset
        )
        p2_selected_records: tuple[Any, ...] = ()
        if p2_gate_mode:
            if predictive_cache is None:
                raise RuntimeError("P2 gate lost its immutable predictive cache")
            p2_selected_records = _select_p2_causal_records(
                records=predictive_cache.iter_records(),
                segments=index.segments,
                episodes=dataset.episode_manifest,
                horizon=args.p2_horizon,
                count=P1_WORLD_SIZE,
            )

        processor = build_processor(str(args.processor_dir.resolve()))
        processor_lattice = configure_native_processor_lattice(
            processor,
            args.visual_lattice,
        )
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
        shared_initialization_local_sha256 = _model_local_state_digest(policy, torch)
        shared_initialization_digests: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
        dist.all_gather_object(
            shared_initialization_digests,
            shared_initialization_local_sha256,
        )
        if len(set(shared_initialization_digests)) != 1:
            raise RuntimeError("ADR-175 ranks loaded different shared LingBot initialization")
        shared_initialization_sha256 = shared_initialization_digests[0]
        graph_config = None
        graph = None
        picf_graph_sha256: str | None = None
        picf_initialization_sha256: str | None = None
        registered_layer_indices: tuple[int, ...] = ()
        if not adr175_mode or adr175_picf_active:
            graph_context = (
                torch.random.fork_rng(devices=[local_rank])
                if adr175_mode
                else contextlib.nullcontext()
            )
            with graph_context:
                if adr175_mode:
                    torch.manual_seed(ADR175_GRAPH_INIT_SEED)
                    torch.cuda.manual_seed_all(ADR175_GRAPH_INIT_SEED)
                graph_config = LingBotNativeGraphConfig.from_policy(
                    policy,
                    capacity=args.capacity,
                    maximum_control_tokens=args.maximum_control_tokens,
                    predictive_target_widths=(
                        ()
                        if predictive_cache is None
                        else (
                            (
                                LINGBOT_PREDICTIVE_TARGET_SPACE,
                                predictive_cache.contract.hidden_size,
                            ),
                        )
                    ),
                    architecture_identity=TASK_INDEPENDENT_ENTITY_POSTERIOR,
                )
                graph = LingBotNativeGraph(
                    graph_config,
                    device=device,
                    dtype=torch.float32,
                ).train()
                picf_graph_sha256 = _canonical_json_sha256(asdict(graph_config))
                local_picf_initialization_sha256 = _model_local_state_digest(graph, torch)
                picf_initialization_digests: list[Any] = [
                    None for _ in range(P1_WORLD_SIZE)
                ]
                dist.all_gather_object(
                    picf_initialization_digests,
                    local_picf_initialization_sha256,
                )
                if len(set(picf_initialization_digests)) != 1:
                    raise RuntimeError("ADR-175 ranks initialized different PICF graphs")
                picf_initialization_sha256 = picf_initialization_digests[0]
            install_lingbot_native_graph(policy, graph)
            if adr175_mode:
                registered_layer_indices = _adr175_registered_layer_indices(
                    graph_config.num_layers
                )
        joint_action_mode = adr175_mode or args.current_frame_action_weight > 0
        representation_parameter_scope = (
            None
            if joint_action_mode
            else configure_native_representation_parameter_scope(policy)
        )
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
        if graph is not None:
            register_native_fsdp_forward_methods(policy)
        if representation_parameter_scope is not None:
            representation_parameter_scope = verify_native_representation_parameter_scope(
                policy,
                expected=representation_parameter_scope,
            )
        parameter_storage = _validate_fsdp2_parameter_storage(
            policy,
            torch,
            expected_placement=args.fsdp2_placement,
        )
        optimizer = (
            build_lingbot_official_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
                build_moe_load_balance_hook=build_moe_load_balance_hook,
            )
            if joint_action_mode
            else build_lingbot_representation_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
            )
        )
        parameter_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        if adr175_mode:
            shared_optimizer_manifest = _shared_optimizer_manifest(
                policy=policy,
                optimizer=optimizer,
                expected_update_count=args.steps,
            )
            shared_optimizer_contract_sha256 = _canonical_json_sha256(
                shared_optimizer_manifest
            )
            gathered_optimizer_digests: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
            dist.all_gather_object(
                gathered_optimizer_digests,
                shared_optimizer_contract_sha256,
            )
            if len(set(gathered_optimizer_digests)) != 1:
                raise RuntimeError("ADR-175 ranks built different shared optimizer manifests")

        rank_seed = args.seed + rank
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
        representation_split: RepresentationTrialSplit | None = None
        p2_representation_split: RepresentationTrialSplit | None = None
        p2_plan: Any | None = None
        entity_evaluation_plan: EntityEvaluationPlan | None = None
        p2_schedule_sha256: str | None = None
        p2_schedule_file_sha256: str | None = None
        p2_causal_selected_records: tuple[Any, ...] = ()
        p2_causal_schedule_sha256: str | None = None
        p2_causal_schedule_file_sha256: str | None = None
        if curve_mode:
            if (
                _sha256(args.stream_plan) != args.stream_plan_sha256
                or _sha256(args.representation_split) != args.representation_split_sha256
                or _sha256(args.entity_evaluation_plan) != args.entity_evaluation_plan_sha256
            ):
                raise ValueError("P1 curve contract file SHA-256 differs")
            representation_split = RepresentationTrialSplit.load(args.representation_split)
            if adr175_mode:
                stream_payload = json.loads(args.stream_plan.read_text())
                stream_metadata = stream_payload.get("metadata")
                if not isinstance(stream_metadata, dict):
                    raise ValueError("ADR-175 stream plan omits metadata")
                if stream_metadata.get("schema") == "picf-next.frozen-sample-plan.v1":
                    plan = FrozenSamplePlan.from_metadata(
                        args.stream_plan,
                        sample_keys=build_native_calvin_physical_sample_domain(
                            dataset,
                            excluded_source_episode_indices=(
                                representation_split.stream_domain_excluded_source_episode_indices
                            ),
                        ),
                    )
                else:
                    plan = load_frozen_episode_stream_plan(
                        args.stream_plan,
                        episodes=build_native_calvin_physical_episode_domain(
                            dataset,
                            excluded_source_episode_indices=(
                                representation_split.stream_domain_excluded_source_episode_indices
                            ),
                        ),
                    )
            else:
                plan = load_frozen_episode_stream_plan(
                    args.stream_plan,
                    episodes=tuple(
                        EpisodeSampleSequence(
                            episode_key=episode.episode_key,
                            sample_keys=episode.sample_keys,
                        )
                        for episode in dataset.episode_manifest
                    ),
                )
            entity_evaluation_plan = EntityEvaluationPlan.load(args.entity_evaluation_plan)
            if plan.total_steps < args.steps:
                raise ValueError("P1 prefix exceeds the frozen P1 stream plan")
            if plan.global_batch_size != P1_WORLD_SIZE:
                raise ValueError("P1 curve stream plan has the wrong global batch")
            if representation_split.stream_plan_sha256 != plan.plan_sha256:
                raise ValueError("P1 curve split and stream plan differ")
            if representation_split.training_steps != plan.total_steps:
                raise ValueError("P1 curve split does not cover the complete stream plan")
            verify_representation_trial_split_training_evidence(
                representation_split,
                plan,
                dataset,
            )
            if (
                entity_evaluation_plan.representation_split_sha256
                != representation_split.artifact_sha256
            ):
                raise ValueError("P1 entity evaluation plan belongs to another split")
            if (
                build_entity_evaluation_plan(representation_split, evaluation_dataset)
                != entity_evaluation_plan
            ):
                raise ValueError("P1 entity evaluation plan is not reproducible from source")
            evaluation_sources = {
                item.source_episode_index for item in entity_evaluation_plan.items
            }
            if evaluation_sources.intersection(
                representation_split.training_source_episode_indices
            ):
                raise ValueError("P1 curve evaluation overlaps a training source episode")
            rank_evaluation_counts = [
                sum(item.rank == item_rank for item in entity_evaluation_plan.items)
                for item_rank in range(P1_WORLD_SIZE)
            ]
            if len(set(rank_evaluation_counts)) != 1:
                raise ValueError("P1 curve evaluation gives ranks unequal forward counts")
            if adr175_mode:
                if adr175_contract is None:
                    raise AssertionError("validated ADR-175 contract disappeared")
                if (
                    adr175_contract.dataset_id != representation_split.dataset_id
                    or adr175_contract.dataset_revision
                    != representation_split.dataset_revision
                    or adr175_contract.dataset_manifest_sha256
                    != representation_split.dataset_manifest_sha256
                    or adr175_contract.stream_plan_sha256 != plan.plan_sha256
                    or adr175_contract.representation_split_artifact_sha256
                    != representation_split.artifact_sha256
                    or adr175_contract.entity_evaluation_plan_artifact_sha256
                    != entity_evaluation_plan.artifact_sha256
                    or adr175_contract.global_batch_size != P1_WORLD_SIZE
                ):
                    raise ValueError("ADR-175 broad-support contract differs from runtime data")
            if staged_mode:
                if predictive_cache is None:
                    raise RuntimeError("staged P2 lost its immutable predictive cache")
                if (
                    _sha256(args.p2_stream_plan) != args.p2_stream_plan_sha256
                    or _sha256(args.p2_representation_split) != args.p2_representation_split_sha256
                ):
                    raise ValueError("staged P2 contract file SHA-256 differs")
                p2_plan = load_frozen_episode_stream_plan(
                    args.p2_stream_plan,
                    episodes=tuple(
                        EpisodeSampleSequence(
                            episode_key=episode.episode_key,
                            sample_keys=episode.sample_keys,
                        )
                        for episode in dataset.episode_manifest
                    ),
                )
                p2_representation_split = RepresentationTrialSplit.load(
                    args.p2_representation_split
                )
                if p2_plan.total_steps < args.staged_p2_steps:
                    raise ValueError("P2 prefix exceeds the frozen P2 stream plan")
                if p2_plan.global_batch_size != P1_WORLD_SIZE:
                    raise ValueError("P2 stream plan has the wrong global batch")
                if p2_representation_split.stream_plan_sha256 != p2_plan.plan_sha256:
                    raise ValueError("P2 split and stream plan differ")
                if p2_representation_split.training_steps != p2_plan.total_steps:
                    raise ValueError("P2 split does not cover the complete stream plan")
                verify_representation_trial_split_training_evidence(
                    p2_representation_split,
                    p2_plan,
                    dataset,
                )
                p1_dataset_identity = (
                    representation_split.dataset_id,
                    representation_split.dataset_revision,
                    representation_split.dataset_manifest_sha256,
                )
                p2_dataset_identity = (
                    p2_representation_split.dataset_id,
                    p2_representation_split.dataset_revision,
                    p2_representation_split.dataset_manifest_sha256,
                )
                if p2_dataset_identity != p1_dataset_identity:
                    raise ValueError("P1 and P2 dataset identities differ")
                p1_training_sources = frozenset(
                    representation_split.training_source_episode_indices
                )
                p2_training_sources = frozenset(
                    p2_representation_split.training_source_episode_indices
                )
                if not p2_training_sources.issubset(p1_training_sources):
                    raise ValueError("P2 training sources escape the frozen P1 training domain")
                if p2_training_sources.intersection(evaluation_sources):
                    raise ValueError("P2 training overlaps P1 validation or held-out sources")
                if predictive_cache.contract.stream_plan_sha256 != p2_plan.plan_sha256:
                    raise RuntimeError("staged P2 cache belongs to another frozen stream plan")
                p2_selected_records = _select_staged_p2_records_from_plan(
                    plan=p2_plan,
                    dataset=dataset,
                    predictive_cache=predictive_cache,
                    segments=index.segments,
                    episodes=dataset.episode_manifest,
                    start_optimizer_step=0,
                    steps=args.staged_p2_steps,
                    world_size=P1_WORLD_SIZE,
                    horizon=args.p2_horizon,
                    allowed_episode_indices=p2_training_sources,
                )
                p2_schedule_payload = {
                    "schema": "picf-next.task-independent-staged-p2-schedule.v3",
                    "selection_semantics": (
                        "frozen-stream-causal-global-batch-subsequence-without-label-or-"
                        "importance-filter/v2"
                    ),
                    "start_optimizer_step": 0,
                    "horizon": args.p2_horizon,
                    "world_size": P1_WORLD_SIZE,
                    "steps": args.staged_p2_steps,
                    "predictive_cache_manifest_sha256": predictive_cache.manifest_sha256,
                    "p1_stream_plan_sha256": plan.plan_sha256,
                    "p1_representation_split_sha256": representation_split.artifact_sha256,
                    "p2_stream_plan_sha256": p2_plan.plan_sha256,
                    "p2_representation_split_sha256": (p2_representation_split.artifact_sha256),
                    "records": [
                        {
                            "rank": ordinal % P1_WORLD_SIZE,
                            "p2_global_step": ordinal // P1_WORLD_SIZE + 1,
                            "plan_optimizer_step": item.plan_optimizer_step,
                            "source_episode_index": int(item.segment.episode_index),
                            "source_global_index": int(item.record.source_global_index),
                            "target_global_index": int(item.record.target_global_index),
                            "transition_index": int(item.transition_index),
                        }
                        for ordinal, item in enumerate(p2_selected_records)
                    ],
                }
                p2_schedule_sha256 = hashlib.sha256(
                    json.dumps(
                        p2_schedule_payload,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("ascii")
                ).hexdigest()
                schedule_digests: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
                dist.all_gather_object(schedule_digests, p2_schedule_sha256)
                if len(set(schedule_digests)) != 1:
                    raise RuntimeError("staged P2 ranks derived different schedules")
                schedule_publication_error: list[str | None] = [None]
                schedule_path = args.run_dir / "staged_p2_schedule.json"
                if rank == 0:
                    try:
                        write_text_durable_exclusive(
                            schedule_path,
                            json.dumps(
                                {
                                    **p2_schedule_payload,
                                    "artifact_sha256": p2_schedule_sha256,
                                },
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                        )
                    except BaseException as error:
                        schedule_publication_error[0] = f"{type(error).__name__}: {error}"
                dist.broadcast_object_list(schedule_publication_error, src=0)
                if schedule_publication_error[0] is not None:
                    raise RuntimeError(
                        f"staged P2 schedule publication failed: {schedule_publication_error[0]}"
                    )
                dist.barrier()
                p2_schedule_file_sha256 = _sha256(schedule_path)
                if args.p2_causal_probe_steps > 0:
                    if (
                        causal_predictive_cache is None
                        or p2_causal_replay_closure is None
                    ):
                        raise RuntimeError("staged P2 causal cache or replay closure vanished")
                    causal_source_episode_indices = frozenset(
                        int(segment.episode_index)
                        for segment in index.segments
                        if int(segment.episode_index) not in p1_training_sources
                    )
                    causal_pairs = tuple(
                        (segment, episode)
                        for segment, episode in zip(
                            index.segments,
                            dataset.episode_manifest,
                            strict=True,
                        )
                        if int(segment.episode_index) in causal_source_episode_indices
                    )
                    if not causal_pairs:
                        raise RuntimeError("P2 causal audit has no source-disjoint segments")
                    p2_causal_selected_records = _select_p2_causal_records(
                        records=causal_predictive_cache.iter_records(),
                        segments=tuple(item[0] for item in causal_pairs),
                        episodes=tuple(item[1] for item in causal_pairs),
                        horizon=args.p2_causal_horizon,
                        count=args.p2_causal_probe_steps * P1_WORLD_SIZE,
                        prefix_frames=2,
                        allowed_episode_indices=causal_source_episode_indices,
                        selection_seed=args.seed + 2_000_003,
                        distinct_source_episodes=True,
                    )
                    training_sources_sha256 = hashlib.sha256(
                        json.dumps(sorted(p1_training_sources)).encode("ascii")
                    ).hexdigest()
                    if (
                        p2_causal_replay_closure[
                            "training_source_episode_indices_sha256"
                        ]
                        != training_sources_sha256
                    ):
                        raise ValueError("P2 causal replay closure uses another training split")
                    selected_closure = []
                    for record, segment, _episode, transition_index in (
                        p2_causal_selected_records
                    ):
                        replay_indices, required_indices = (
                            _calvin_causal_replay_dependency_closure(
                                source_global_index=int(record.source_global_index),
                                segment_start=int(segment.start),
                                segment_end=int(segment.end),
                                horizon=args.p2_causal_horizon,
                                prefix_frames=2,
                                action_horizon=int(config.chunk_size),
                            )
                        )
                        selected_closure.append(
                            {
                                "source_global_index": int(record.source_global_index),
                                "target_global_index": int(record.target_global_index),
                                "segment_index": int(segment.index),
                                "source_episode_index": int(segment.episode_index),
                                "transition_index": int(transition_index),
                                "replay_global_indices": list(replay_indices),
                                "required_global_indices": list(required_indices),
                            }
                        )
                    if selected_closure != p2_causal_replay_closure["selections"]:
                        raise ValueError(
                            "P2 causal replay closure differs from deterministic selection"
                        )
                    validation_segment_indices = {
                        item.segment_index for item in representation_split.validation_segments
                    }
                    heldout_segment_indices = {
                        item.segment_index for item in representation_split.heldout_segments
                    }
                    causal_schedule_payload = {
                        "schema": "picf-next.task-independent-p2-causal-schedule.v2",
                        "selection_semantics": (
                            "source-disjoint-distinct-episode-positive-target-two-prefix-"
                            "one-future-hash/v2"
                        ),
                        "world_size": P1_WORLD_SIZE,
                        "steps": args.p2_causal_probe_steps,
                        "horizon": args.p2_causal_horizon,
                        "predictive_cache_manifest_sha256": (
                            causal_predictive_cache.manifest_sha256
                        ),
                        "p1_representation_split_sha256": (
                            representation_split.artifact_sha256
                        ),
                        "records": [
                            {
                                "rank": ordinal % P1_WORLD_SIZE,
                                "causal_global_step": ordinal // P1_WORLD_SIZE + 1,
                                "partition": (
                                    "validation"
                                    if int(item[1].index) in validation_segment_indices
                                    else "heldout"
                                    if int(item[1].index) in heldout_segment_indices
                                    else "causal_audit"
                                ),
                                "source_episode_index": int(item[1].episode_index),
                                "source_global_index": int(item[0].source_global_index),
                                "target_global_index": int(item[0].target_global_index),
                                "transition_index": int(item[3]),
                            }
                            for ordinal, item in enumerate(p2_causal_selected_records)
                        ],
                    }
                    p2_causal_schedule_sha256 = hashlib.sha256(
                        json.dumps(
                            causal_schedule_payload,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    causal_schedule_digests: list[Any] = [
                        None for _ in range(P1_WORLD_SIZE)
                    ]
                    dist.all_gather_object(
                        causal_schedule_digests,
                        p2_causal_schedule_sha256,
                    )
                    if len(set(causal_schedule_digests)) != 1:
                        raise RuntimeError("P2 causal ranks derived different schedules")
                    causal_schedule_path = args.run_dir / "p2_causal_schedule.json"
                    causal_schedule_publication_error: list[str | None] = [None]
                    if rank == 0:
                        try:
                            write_text_durable_exclusive(
                                causal_schedule_path,
                                json.dumps(
                                    {
                                        **causal_schedule_payload,
                                        "artifact_sha256": p2_causal_schedule_sha256,
                                    },
                                    indent=2,
                                    sort_keys=True,
                                )
                                + "\n",
                            )
                        except BaseException as error:
                            causal_schedule_publication_error[0] = (
                                f"{type(error).__name__}: {error}"
                            )
                    dist.broadcast_object_list(causal_schedule_publication_error, src=0)
                    if causal_schedule_publication_error[0] is not None:
                        raise RuntimeError(
                            "P2 causal schedule publication failed: "
                            + causal_schedule_publication_error[0]
                        )
                    dist.barrier()
                    p2_causal_schedule_file_sha256 = _sha256(causal_schedule_path)
        else:
            plan = build_native_calvin_stream_plan(
                dataset,
                comparison_id=P1_COMPARISON_ID,
                seed=args.seed,
                global_batch_size=P1_WORLD_SIZE,
                total_steps=args.steps,
            )
        model_family_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "architecture": (
                        "released-lingbot-action-only"
                        if adr175_mode and args.adr175_arm == "lbot"
                        else TASK_INDEPENDENT_ENTITY_POSTERIOR
                    ),
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "graph": None if graph_config is None else asdict(graph_config),
                    "implementation_sha256": implementation_sha256,
                    "p1_plan_sha256": plan.plan_sha256,
                    "p2_plan_sha256": (None if p2_plan is None else p2_plan.plan_sha256),
                    "parameter_scope": (
                        "full_joint_action"
                        if representation_parameter_scope is None
                        else representation_parameter_scope.as_dict()
                    ),
                    "visual_lattice": args.visual_lattice,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
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
        objective_config = TaskIndependentEntityObjectiveConfig(
            action_weight=args.current_frame_action_weight,
            entity_weight=args.current_frame_entity_weight,
            predictive_weight=1.0 if p2_gate_mode else 0.0,
            mask_focal_weight=args.mask_focal_weight,
            mask_dice_weight=args.mask_dice_weight,
            existence_weight=args.existence_weight,
            ownership_weight=args.ownership_weight,
        )
        entity_evaluation_objective_config = TaskIndependentEntityObjectiveConfig(
            action_weight=0.0,
            entity_weight=args.current_frame_entity_weight,
            predictive_weight=0.0,
            mask_focal_weight=args.mask_focal_weight,
            mask_dice_weight=args.mask_dice_weight,
            existence_weight=args.existence_weight,
            ownership_weight=args.ownership_weight,
        )
        evaluation_snapshot_reports: list[dict[str, Any]] = []

        def adr175_attention_targets(
            *,
            bindings_by_batch: tuple[Any, ...],
            batch: CollatedNativeCALVINBatch,
            dtype: Any,
        ) -> tuple[Any, Any, tuple[dict[str, Any], ...]]:
            if len(bindings_by_batch) != batch.routing.batch_size:
                raise ValueError("ADR-175 row bindings differ from the training batch")
            weights = torch.zeros(
                (batch.routing.batch_size, args.capacity),
                dtype=dtype,
                device=device,
            )
            valid = torch.zeros(
                batch.routing.batch_size,
                dtype=torch.bool,
                device=device,
            )
            audit_rows: list[dict[str, Any]] = []
            for batch_index, (bindings, request) in enumerate(
                zip(
                    bindings_by_batch,
                    batch.structural_target_requests,
                    strict=True,
                )
            ):
                relevance = calvin_task_physical_relevance(request.task_key)
                row_by_identity = dict(bindings)
                selected_rows = tuple(
                    row_by_identity[identity]
                    for identity in relevance.action_target_identity_keys
                    if identity in row_by_identity
                )
                target_valid = bool(
                    relevance.exact_action_target
                    and len(selected_rows) == len(relevance.action_target_identity_keys)
                )
                if target_valid:
                    mass = 1.0 / len(selected_rows)
                    for row_index in selected_rows:
                        weights[batch_index, row_index] = mass
                    valid[batch_index] = True
                audit_rows.append(
                    {
                        "task_key": request.task_key,
                        "exact_action_target": relevance.exact_action_target,
                        "target_identity_keys": list(
                            relevance.action_target_identity_keys
                        ),
                        "selected_rows": list(selected_rows),
                        "target_valid": target_valid,
                    }
                )
            return weights, valid, tuple(audit_rows)

        def collate_replay(
            planned: PlannedNativeCALVINReplayBatch,
        ) -> CollatedNativeCALVINBatch:
            collated = collate_native_calvin_training_batch(
                planned.training,
                feature_transform=feature_transform,
                collator=VLADataCollatorWithPacking(),
                augmentation_seeds=planned.augmentation_seeds,
                source_digest=planned.source_digest,
            )
            _validate_visual_lattice_inputs(
                dict(collated.model_inputs),
                visual_lattice=args.visual_lattice,
                merge_size=merge_size,
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
            )
            return materialize_native_flow_randomness(collated, planned)

        if p2_gate_mode:
            if predictive_cache is None or p2_predictive_report is None:
                raise RuntimeError("P2 gate lost its immutable predictive cache")
            horizon = args.p2_horizon
            record, segment, episode, transition_index = p2_selected_records[rank]
            source_sample_key = episode.sample_keys[transition_index]
            prefix_sample_key = episode.sample_keys[transition_index - 1]
            future_sample_keys = episode.sample_keys[
                transition_index + 1 : transition_index + horizon + 1
            ]
            if dataset.source_global_index_by_key(
                source_sample_key
            ) != record.source_global_index or tuple(
                dataset.source_global_index_by_key(key) for key in future_sample_keys
            ) != tuple(
                range(
                    record.source_global_index + 1,
                    record.source_global_index + horizon + 1,
                )
            ):
                raise RuntimeError("P2 source/future sample resolution changed")

            def p2_replay(sample_key: str, *, replay_offset: int) -> PlannedNativeCALVINReplayBatch:
                return build_native_calvin_replay_batch(
                    dataset,
                    sample_key=sample_key,
                    lane_id=rank,
                    episode_instance_id=f"task-independent-p2-gate/rank-{rank}",
                    optimizer_step=0,
                    replay_seed=args.seed + rank * 1009 + replay_offset,
                    device=device,
                    dtype=torch.bfloat16,
                )

            prefix_planned = p2_replay(prefix_sample_key, replay_offset=0)
            source_planned = p2_replay(source_sample_key, replay_offset=1)
            future_planned = tuple(
                p2_replay(sample_key, replay_offset=2 + offset)
                for offset, sample_key in enumerate(future_sample_keys)
            )
            prefix_batch = collate_replay(prefix_planned)
            source_batch = collate_replay(source_planned)
            prefix_config = TaskIndependentEntityObjectiveConfig(
                action_weight=0.0,
                entity_weight=1.0,
                predictive_weight=0.0,
                mask_focal_weight=args.mask_focal_weight,
                mask_dice_weight=args.mask_dice_weight,
                existence_weight=args.existence_weight,
                ownership_weight=args.ownership_weight,
            )
            with torch.no_grad():
                prefix_result = run_task_independent_calvin_current_frame_diagnostic(
                    policy,
                    batch=prefix_batch,
                    physical_sidecar=physical_sidecar,
                    objective_config=prefix_config,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    minimum_supervised_fraction=args.minimum_supervised_fraction,
                    capacity_seeds=prefix_planned.augmentation_seeds,
                )
            prefix_state = prefix_result.context.posterior_state
            if prefix_state is None:
                raise RuntimeError("P2 causal prefix produced no posterior state")
            prefix_row_bindings = physical_frame_row_bindings(
                prefix_result.targets,
                prefix_result.objective.frame_losses[0].assignment,
                capacity=args.capacity,
            )
            if not prefix_row_bindings[0]:
                raise RuntimeError("P2 causal prefix established no physical row gauge")
            previous_state = NativePosteriorState(prefix_state.rows.detach())
            del prefix_result, prefix_state, prefix_batch

            prior_stepper = LingBotNativePriorStepper(policy, graph)
            future_controls = tuple(value.training.controls for value in future_planned)

            def predictive_rollout(state: NativePosteriorState) -> Any:
                request = make_native_future_request(
                    source=PredictionSource.PRIOR,
                    batch_size=state.batch_size,
                    horizon=horizon,
                    valid=torch.ones(state.batch_size, dtype=torch.bool, device=device),
                    device=device,
                    dtype=state.rows.dtype,
                    route_id=predictive_cache.contract.route_id,
                    address_width=graph.config.prediction_address_width,
                )
                return rollout_native_prior_prediction(
                    prior_stepper,
                    state,
                    future_controls,
                    request=request,
                    target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
                )

            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            result = run_task_independent_calvin_sequence_objective(
                policy,
                batches=(source_batch,),
                physical_sidecar=physical_sidecar,
                objective_config=objective_config,
                patch_size=patch_size,
                merge_size=merge_size,
                previous_state=previous_state,
                previous_state_valid=torch.ones(1, dtype=torch.bool, device=device),
                prior_row_bindings_by_batch=prefix_row_bindings,
                predictive_rollout_factory=predictive_rollout,
                predictive_cache=predictive_cache,
                minimum_supervised_fraction=args.minimum_supervised_fraction,
                capacity_seeds=prefix_planned.augmentation_seeds,
            )
            result.objective.objective.total.backward()
            gradient_metrics = _add_action_only_gradient_summary(
                _distributed_gradient_metrics(
                    policy,
                    (
                        ("native_graph", "picf_native_graph"),
                        ("vlm_host", ".qwenvl."),
                        ("predictive_readout", "predictive_readouts.dino_video"),
                        *_ACTION_ONLY_GRADIENT_METRICS,
                    ),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
            )
            if not bool(gradient_metrics["all_finite"]):
                raise RuntimeError("P2 gate produced non-finite gradients")
            for family in ("native_graph", "vlm_host", "predictive_readout"):
                if float(gradient_metrics[f"{family}_norm"]) <= 0:
                    raise RuntimeError(f"P2 gate produced no {family} gradient")
            if int(gradient_metrics["action_only_elements"]) != 0:
                raise RuntimeError("P2 gate unexpectedly trained the action-only partition")
            optimizer_state = None
            if args.p2_optimizer_update:
                _publish_p2_update_stage(
                    run_dir=args.run_dir,
                    rank=rank,
                    stage="pre_clip_sync_started",
                )
                _synchronize_p2_update_ranks(
                    device=device,
                    dist_module=dist,
                    torch_module=torch,
                )
                _publish_p2_update_stage(
                    run_dir=args.run_dir,
                    rank=rank,
                    stage="clip_started",
                )
                clipped = torch.nn.utils.clip_grad_norm_(
                    policy.parameters(),
                    args.max_grad_norm,
                    error_if_nonfinite=True,
                    foreach=False,
                )
                full_tensor = getattr(clipped, "full_tensor", None)
                if callable(full_tensor):
                    clipped = full_tensor()
                gradient_metrics["preclip_global_norm"] = float(clipped.item())
                _publish_p2_update_stage(
                    run_dir=args.run_dir,
                    rank=rank,
                    stage="optimizer_step_started",
                )
                optimizer.step()
                _publish_p2_update_stage(
                    run_dir=args.run_dir,
                    rank=rank,
                    stage="optimizer_state_audit_started",
                )
                optimizer_state = _validate_optimizer_state(
                    optimizer,
                    torch,
                    expected_step=1,
                )
                optimizer_state["family_state_entries"] = _p2_optimizer_state_families(
                    policy,
                    optimizer,
                )
                optimizer.zero_grad(set_to_none=True)
                _publish_p2_update_stage(
                    run_dir=args.run_dir,
                    rank=rank,
                    stage="complete",
                )
            torch.cuda.synchronize(device)
            elapsed_seconds = time.perf_counter() - started
            peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
            if peak_reserved_bytes > int(args.maximum_peak_reserved_gib * 1024**3):
                raise RuntimeError("P2 gate exceeded the registered CUDA reservation budget")
            valid_counts = {
                name: int(value) for name, value in result.objective.objective.valid_counts.items()
            }
            predictive_valid = sum(
                count for name, count in valid_counts.items() if name.startswith("rollout/")
            )
            if predictive_valid <= 0:
                raise RuntimeError("P2 gate has no valid object-indexed future targets")
            local_report = {
                "rank": rank,
                "source_global_index": record.source_global_index,
                "target_global_index": record.target_global_index,
                "horizon": record.horizon,
                "task_key": segment.task_key,
                "instruction": segment.instruction,
                "prefix_sample_key": prefix_sample_key,
                "source_sample_key": source_sample_key,
                "future_sample_keys": list(future_sample_keys),
                "source_rgb_sha256": record.source_rgb_sha256,
                "target_rgb_sha256": record.target_rgb_sha256,
                "cached_identity_keys": list(record.identity_keys),
                "cached_positive_identity_count": int((record.importance > 0).sum()),
                "prefix_row_bindings": [
                    [list(binding) for binding in bindings] for bindings in prefix_row_bindings
                ],
                "objective_total": _float(result.objective.objective.total),
                "family_terms": {
                    name: _float(value)
                    for name, value in result.objective.objective.family_terms.items()
                },
                "valid_counts": valid_counts,
                "gradient_metrics": gradient_metrics,
                "optimizer_state": optimizer_state,
                "elapsed_seconds": elapsed_seconds,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": peak_reserved_bytes,
                "row_bindings": [
                    [list(binding) for binding in bindings]
                    for bindings in result.objective.row_bindings_by_batch
                ],
            }
            gathered: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
            dist.all_gather_object(gathered, local_report)
            publication_error: list[str | None] = [None]
            report: dict[str, Any] | None = None
            if rank == 0:
                try:
                    if len({item["source_global_index"] for item in gathered}) != P1_WORLD_SIZE:
                        raise RuntimeError("P2 ranks consumed overlapping source frames")
                    report = {
                        "schema": (
                            P2_UPDATE_GATE_REPORT_SCHEMA
                            if args.p2_optimizer_update
                            else P2_GATE_REPORT_SCHEMA
                        ),
                        "status": "PASS",
                        "architecture_identity": TASK_INDEPENDENT_ENTITY_POSTERIOR,
                        "task_scorer_present": False,
                        "task_used_by_entity_objective": False,
                        "action_suffix_executed": False,
                        "optimizer_step_executed": args.p2_optimizer_update,
                        "prefix_gradient_mode": "fixed_weight_no_grad_one_frame",
                        "source_gradient_mode": "posterior_correction_plus_shared_host_prior",
                        "prediction_target": LINGBOT_PREDICTIVE_TARGET_SPACE,
                        "prediction_source": "prior",
                        "prediction_horizon": horizon,
                        "predictive_cache_manifest_sha256": predictive_cache.manifest_sha256,
                        "predictive_build_report_sha256": (
                            args.p2_predictive_cache_build_report_sha256
                        ),
                        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                        "implementation_sha256": implementation_sha256,
                        "graph": asdict(graph_config),
                        "objective": asdict(objective_config),
                        "visual_lattice": args.visual_lattice,
                        "fsdp2_placement": args.fsdp2_placement,
                        "maximum_peak_reserved_bytes": max(
                            item["peak_cuda_reserved_bytes"] for item in gathered
                        ),
                        "rank_reports": gathered,
                    }
                    write_text_durable_exclusive(
                        args.output,
                        json.dumps(report, indent=2, sort_keys=True) + "\n",
                    )
                except BaseException as error:
                    publication_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(publication_error, src=0)
            if publication_error[0] is not None:
                raise RuntimeError(f"P2 gate report publication failed: {publication_error[0]}")
            dist.barrier()
            if rank == 0:
                if report is None:
                    raise RuntimeError("rank zero lost the P2 gate report")
                print(json.dumps(report, indent=2, sort_keys=True), flush=True)
            return

        visual_sample_keys: frozenset[str] = frozenset()
        if entity_evaluation_plan is not None:
            visual_sample_keys = frozenset(
                _evaluation_visual_sample_keys(
                    entity_evaluation_plan.items,
                    partitions=ENTITY_EVALUATION_PARTITIONS,
                    per_partition=args.evaluation_visuals_per_partition,
                )
            )

        def snapshot_adr175_runtime_buffers() -> dict[str, Any]:
            suffixes = ("avg_topk_sigmoid_score", "tokens_per_expert")
            all_buffers = tuple(policy.named_buffers())
            mutable = tuple(
                (name, buffer, buffer.detach().clone())
                for name, buffer in all_buffers
                if name.endswith(suffixes)
            )
            if not mutable or not any(
                name.endswith("tokens_per_expert") for name, _buffer, _saved in mutable
            ):
                raise RuntimeError("ADR-175 found no released action-MoE runtime counters")
            return {
                "buffer_names": tuple(name for name, _buffer in all_buffers),
                "mutable": mutable,
                "versions": {
                    name: int(getattr(buffer, "_version", -1))
                    for name, buffer in all_buffers
                },
            }

        def restore_adr175_runtime_buffers(snapshot: dict[str, Any]) -> None:
            current = tuple(policy.named_buffers())
            current_names = tuple(name for name, _buffer in current)
            if current_names != snapshot["buffer_names"]:
                raise RuntimeError("ADR-175 evaluation changed the model buffer inventory")
            mutable_names = {name for name, _buffer, _saved in snapshot["mutable"]}
            unexpected_changes = tuple(
                name
                for name, buffer in current
                if int(getattr(buffer, "_version", -1)) != snapshot["versions"][name]
                and name not in mutable_names
            )
            with torch.no_grad():
                for name, buffer, saved in snapshot["mutable"]:
                    if buffer.shape != saved.shape or buffer.dtype != saved.dtype:
                        raise RuntimeError(
                            f"ADR-175 runtime buffer changed contract: {name}"
                        )
                    buffer.copy_(saved)
            if unexpected_changes:
                raise RuntimeError(
                    "ADR-175 evaluation mutated undeclared runtime buffers: "
                    f"{unexpected_changes}"
                )

        def run_adr175_evaluation(checkpoint_global_step: int) -> dict[str, Any]:
            if (
                not adr175_mode
                or entity_evaluation_plan is None
                or representation_split is None
            ):
                raise RuntimeError("ADR-175 evaluation contract is absent")
            optimizer.zero_grad(set_to_none=True)
            local_items = tuple(item for item in entity_evaluation_plan.items if item.rank == rank)
            python_rng = random.getstate()
            numpy_rng = np.random.get_state()
            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state(device)
            dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "adr175_evaluation_start",
                            "arm": args.adr175_arm,
                            "checkpoint_global_step": checkpoint_global_step,
                            "samples_per_rank": len(local_items),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            local_samples: list[dict[str, Any]] = []
            try:
                with torch.no_grad():
                    for local_index, item in enumerate(local_items):
                        replay_seed: int | None = None
                        planned = None
                        collated = None
                        prepare_error: BaseException | None = None
                        try:
                            replay_seed = _entity_evaluation_replay_seed(
                                entity_evaluation_plan.artifact_sha256,
                                item.sample_key,
                            )
                            planned = build_native_calvin_replay_batch(
                                evaluation_dataset,
                                sample_key=item.sample_key,
                                lane_id=rank,
                                episode_instance_id=(
                                    f"adr175-evaluation/{item.partition}/{item.ordinal}"
                                ),
                                optimizer_step=0,
                                replay_seed=replay_seed,
                                device=device,
                                dtype=torch.bfloat16,
                            )
                            collated = collate_replay(planned)
                        except BaseException as error:
                            prepare_error = error
                        _distributed_phase_error(
                            error=prepare_error,
                            phase=(
                                f"adr175-checkpoint-{checkpoint_global_step}-"
                                f"sample-{local_index}-prepare"
                            ),
                            rank=rank,
                            dist_module=dist,
                        )
                        if replay_seed is None or planned is None or collated is None:
                            raise RuntimeError("ADR-175 evaluation preparation vanished")

                        result = None
                        official_evaluation = None
                        collector = None
                        forward_seconds = 0.0
                        runtime_snapshot: dict[str, Any] | None = None
                        forward_error: BaseException | None = None
                        try:
                            runtime_snapshot = snapshot_adr175_runtime_buffers()
                            torch.cuda.synchronize(device)
                            started = time.perf_counter()
                            with torch.random.fork_rng(devices=[local_rank]):
                                torch.manual_seed(replay_seed)
                                torch.cuda.manual_seed(replay_seed)
                                if args.adr175_arm == "lbot":
                                    official_evaluation = (
                                        run_official_policy_diagnostic_forward(
                                            policy,
                                            model_inputs=collated.model_inputs,
                                        )
                                    )
                                else:
                                    collector = RegisteredActionPosteriorReceiptCollector(
                                        registered_layer_indices=registered_layer_indices
                                    )
                                    result = (
                                        run_task_independent_calvin_current_frame_diagnostic(
                                            policy,
                                            batch=collated,
                                            physical_sidecar=physical_sidecar,
                                            objective_config=(
                                                entity_evaluation_objective_config
                                            ),
                                            patch_size=patch_size,
                                            merge_size=merge_size,
                                            minimum_supervised_fraction=(
                                                args.minimum_supervised_fraction
                                            ),
                                            capacity_seeds=planned.augmentation_seeds,
                                            action_attention_callback=collector,
                                        )
                                    )
                                    official_evaluation = result.policy_forward
                            torch.cuda.synchronize(device)
                            forward_seconds = time.perf_counter() - started
                        except BaseException as error:
                            forward_error = error
                        finally:
                            if runtime_snapshot is not None:
                                try:
                                    restore_adr175_runtime_buffers(runtime_snapshot)
                                except BaseException as error:
                                    if forward_error is None:
                                        forward_error = error
                        _distributed_phase_error(
                            error=forward_error,
                            phase=(
                                f"adr175-checkpoint-{checkpoint_global_step}-"
                                f"sample-{local_index}-forward"
                            ),
                            rank=rank,
                            dist_module=dist,
                        )
                        if official_evaluation is None:
                            raise RuntimeError("ADR-175 evaluation forward vanished")

                        evidence: dict[str, Any] | None = None
                        evidence_error: BaseException | None = None
                        try:
                            evidence = {
                                "schema": "picf-next.adr175-evaluation-sample.v1",
                                "checkpoint_global_step": checkpoint_global_step,
                                "partition": item.partition,
                                "ordinal": item.ordinal,
                                "rank": rank,
                                "task_key": item.task_key,
                                "segment_index": item.segment_index,
                                "source_episode_index": item.source_episode_index,
                                "source_global_index": item.source_global_index,
                                "transition_index": item.transition_index,
                                "sample_key": item.sample_key,
                                "source_digest": collated.source_digest,
                                "official_action_loss": _float(
                                    official_evaluation.official_action_loss
                                ),
                                "posterior_adoption": None,
                                "target_mass": None,
                                "conditional_selectivity": None,
                                "target_valid": False,
                                "entity_evidence": None,
                                "visual_artifacts": [],
                                "forward_seconds": forward_seconds,
                            }
                            if result is not None:
                                frame_loss = result.objective.frame_losses[0]
                                row_bindings = physical_frame_row_bindings(
                                    result.targets,
                                    frame_loss.assignment,
                                    capacity=args.capacity,
                                )
                                if collector is None:
                                    raise RuntimeError(
                                        "ADR-175 treatment lost attention collector"
                                    )
                                receipts = collector.finalize()
                                target_weights, target_valid, target_audit = (
                                    adr175_attention_targets(
                                        bindings_by_batch=row_bindings,
                                        batch=collated,
                                        dtype=receipts[0].posterior_attention.dtype,
                                    )
                                )
                                head_indices = torch.tensor(
                                    ADR175_GUIDEDVLA_OBJECT_HEAD_INDICES,
                                    dtype=torch.long,
                                    device=device,
                                )
                                attention_results = tuple(
                                    action_posterior_target_mass_loss(
                                        receipt.posterior_attention,
                                        target_row_weights=target_weights,
                                        target_valid=target_valid,
                                        head_indices=head_indices,
                                    )
                                    for receipt in receipts
                                )
                                valid_entries = attention_results[0].valid_entries
                                if any(
                                    not torch.equal(candidate.valid_entries, valid_entries)
                                    for candidate in attention_results[1:]
                                ):
                                    raise RuntimeError(
                                        "ADR-175 evaluation layers disagree on target validity"
                                    )
                                adoption = torch.stack(
                                    tuple(
                                        candidate.total_posterior_mass
                                        for candidate in attention_results
                                    )
                                ).mean(dim=0)
                                target_mass = torch.stack(
                                    tuple(
                                        candidate.target_mass
                                        for candidate in attention_results
                                    )
                                ).mean(dim=0)
                                if bool(valid_entries.any()):
                                    evidence["posterior_adoption"] = _float(
                                        adoption.masked_select(valid_entries).mean()
                                    )
                                    evidence["target_mass"] = _float(
                                        target_mass.masked_select(valid_entries).mean()
                                    )
                                    evidence["conditional_selectivity"] = _float(
                                        (
                                            target_mass / adoption.clamp_min(1e-6)
                                        ).masked_select(valid_entries).mean()
                                    )
                                evidence["target_valid"] = bool(valid_entries.any())
                                evidence["target_audit"] = list(target_audit)
                                evidence["entity_evidence"] = (
                                    evaluate_physical_entity_frame(
                                        physical_frame_predictions_from_relation(
                                            result.relation
                                        ),
                                        result.targets.targets,
                                        frame_loss.assignment,
                                        identity_keys=(
                                            result.targets.identity_keys_by_batch[0]
                                        ),
                                    )
                                )
                                if item.sample_key in visual_sample_keys:
                                    evidence["visual_artifacts"] = (
                                        render_task_independent_entity_visuals(
                                            output_root=args.run_dir,
                                            global_step=checkpoint_global_step,
                                            input_weight_global_step=(
                                                checkpoint_global_step
                                            ),
                                            weight_boundary=(
                                                "fixed_checkpoint_evaluation"
                                            ),
                                            rank=rank,
                                            host_items=planned.training.host_items,
                                            model_inputs=collated.model_inputs,
                                            relation=result.relation,
                                            target_bundle=result.targets,
                                            set_loss=frame_loss,
                                            sample_keys=collated.routing.sample_keys,
                                            merge_size=merge_size,
                                        )
                                    )
                        except BaseException as error:
                            evidence_error = error
                        _distributed_phase_error(
                            error=evidence_error,
                            phase=(
                                f"adr175-checkpoint-{checkpoint_global_step}-"
                                f"sample-{local_index}-evidence"
                            ),
                            rank=rank,
                            dist_module=dist,
                        )
                        if evidence is None:
                            raise RuntimeError("ADR-175 evaluation evidence vanished")
                        local_samples.append(evidence)
                        if rank == 0 and (
                            local_index == 0
                            or local_index + 1 == len(local_items)
                            or (local_index + 1) % 5 == 0
                        ):
                            print(
                                json.dumps(
                                    {
                                        "event": "adr175_evaluation_progress",
                                        "arm": args.adr175_arm,
                                        "checkpoint_global_step": checkpoint_global_step,
                                        "completed_per_rank": local_index + 1,
                                        "samples_per_rank": len(local_items),
                                    },
                                    sort_keys=True,
                                ),
                                flush=True,
                            )
            finally:
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, device)

            gathered_samples: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
            dist.all_gather_object(gathered_samples, local_samples)
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    samples = sorted(
                        (sample for rank_samples in gathered_samples for sample in rank_samples),
                        key=lambda sample: int(sample["ordinal"]),
                    )
                    expected_keys = [item.sample_key for item in entity_evaluation_plan.items]
                    if [sample["sample_key"] for sample in samples] != expected_keys:
                        raise RuntimeError("ADR-175 evaluation sample set changed")
                    partition_summaries: dict[str, dict[str, Any]] = {}
                    for partition in ENTITY_EVALUATION_PARTITIONS:
                        selected = [
                            sample for sample in samples if sample["partition"] == partition
                        ]
                        summary: dict[str, Any] = {
                            "partition": partition,
                            "sample_count": len(selected),
                            "action_loss": _mean_finite(
                                [float(sample["official_action_loss"]) for sample in selected],
                                name=f"{partition} action loss",
                            ),
                            "posterior_adoption": None,
                            "conditional_selectivity": None,
                            "entity_set_score": None,
                            "entity_set_summary": None,
                        }
                        if adr175_picf_active:
                            entity_samples = []
                            for sample in selected:
                                entity = sample["entity_evidence"]
                                if not isinstance(entity, dict):
                                    raise RuntimeError(
                                        "ADR-175 treatment omitted entity evidence"
                                    )
                                entity_samples.append(
                                    {
                                        **entity,
                                        "partition": partition,
                                        "task_key": sample["task_key"],
                                        "sample_key": sample["sample_key"],
                                    }
                                )
                            entity_summary = summarize_entity_evaluation_partition(
                                entity_samples,
                                partition=partition,
                            )
                            valid_attention = [
                                sample for sample in selected if sample["target_valid"]
                            ]
                            summary.update(
                                {
                                    "posterior_adoption": _mean_finite(
                                        [
                                            float(sample["posterior_adoption"])
                                            for sample in valid_attention
                                        ],
                                        name=f"{partition} posterior adoption",
                                    ),
                                    "conditional_selectivity": _mean_finite(
                                        [
                                            float(sample["conditional_selectivity"])
                                            for sample in valid_attention
                                        ],
                                        name=f"{partition} conditional selectivity",
                                    ),
                                    "entity_set_score": float(
                                        entity_summary["mean_support_soft_iou_efficiency"]
                                    ),
                                    "entity_set_summary": entity_summary,
                                }
                            )
                        partition_summaries[partition] = summary
                    evaluation_input_sha256 = _canonical_json_sha256(
                        [
                            {
                                "sample_key": sample["sample_key"],
                                "source_digest": sample["source_digest"],
                            }
                            for sample in samples
                        ]
                    )
                    payload = {
                        "schema": ADR175_EVALUATION_SNAPSHOT_SCHEMA,
                        "status": "PASS",
                        "arm": args.adr175_arm,
                        "checkpoint_global_step": checkpoint_global_step,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "stream_plan_sha256": plan.plan_sha256,
                        "representation_split_sha256": (
                            representation_split.artifact_sha256
                        ),
                        "entity_evaluation_plan_sha256": (
                            entity_evaluation_plan.artifact_sha256
                        ),
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "samples": samples,
                        "partition_summaries": partition_summaries,
                    }
                    artifact_sha256 = _canonical_json_sha256(payload)
                    snapshot = {**payload, "artifact_sha256": artifact_sha256}
                    destination = (
                        args.run_dir
                        / f"adr175_evaluation_step_{checkpoint_global_step:06d}.json"
                    )
                    write_text_durable_exclusive(
                        destination,
                        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                    )
                    publication[0] = {
                        "artifact_sha256": artifact_sha256,
                        "file_sha256": _sha256(destination),
                        "path": str(destination),
                        "checkpoint_global_step": checkpoint_global_step,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "partition_summaries": partition_summaries,
                    }
                except BaseException as error:
                    publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(f"ADR-175 evaluation publication failed: {publication[0]}")
            dist.barrier()
            return publication[0]

        def run_entity_evaluation(checkpoint_global_step: int) -> dict[str, Any]:
            if adr175_mode:
                return run_adr175_evaluation(checkpoint_global_step)
            if entity_evaluation_plan is None or representation_split is None:
                raise RuntimeError("P1 curve evaluation contract is absent")
            optimizer.zero_grad(set_to_none=True)
            local_items = tuple(item for item in entity_evaluation_plan.items if item.rank == rank)
            dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "task_independent_p1_evaluation_start",
                            "checkpoint_global_step": checkpoint_global_step,
                            "samples_per_rank": len(local_items),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            local_samples: list[dict[str, Any]] = []
            with torch.no_grad():
                for local_index, item in enumerate(local_items):
                    replay_seed: int | None = None
                    planned = None
                    collated = None
                    prepare_error: BaseException | None = None
                    try:
                        replay_seed = _entity_evaluation_replay_seed(
                            entity_evaluation_plan.artifact_sha256,
                            item.sample_key,
                        )
                        planned = build_native_calvin_replay_batch(
                            evaluation_dataset,
                            sample_key=item.sample_key,
                            lane_id=rank,
                            episode_instance_id=(
                                f"task-independent-p1-evaluation/{item.partition}/{item.ordinal}"
                            ),
                            optimizer_step=0,
                            replay_seed=replay_seed,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        collated = collate_replay(planned)
                    except BaseException as error:
                        prepare_error = error
                    _distributed_phase_error(
                        error=prepare_error,
                        phase=f"checkpoint-{checkpoint_global_step}-sample-{local_index}-prepare",
                        rank=rank,
                        dist_module=dist,
                    )
                    if replay_seed is None or planned is None or collated is None:
                        raise RuntimeError("P1 entity evaluation preparation vanished")

                    result = None
                    forward_seconds = 0.0
                    forward_error: BaseException | None = None
                    try:
                        torch.cuda.synchronize(device)
                        started = time.perf_counter()
                        with torch.random.fork_rng(devices=[local_rank]):
                            torch.manual_seed(replay_seed)
                            torch.cuda.manual_seed(replay_seed)
                            result = run_task_independent_calvin_current_frame_diagnostic(
                                policy,
                                batch=collated,
                                physical_sidecar=physical_sidecar,
                                objective_config=entity_evaluation_objective_config,
                                patch_size=patch_size,
                                merge_size=merge_size,
                                minimum_supervised_fraction=(args.minimum_supervised_fraction),
                                capacity_seeds=planned.augmentation_seeds,
                            )
                        torch.cuda.synchronize(device)
                        forward_seconds = time.perf_counter() - started
                    except BaseException as error:
                        forward_error = error
                    _distributed_phase_error(
                        error=forward_error,
                        phase=f"checkpoint-{checkpoint_global_step}-sample-{local_index}-forward",
                        rank=rank,
                        dist_module=dist,
                    )
                    if result is None:
                        raise RuntimeError("P1 entity evaluation forward vanished")

                    evidence: dict[str, Any] | None = None
                    evidence_error: BaseException | None = None
                    try:
                        frame_loss = result.objective.frame_losses[0]
                        evidence = evaluate_physical_entity_frame(
                            physical_frame_predictions_from_relation(result.relation),
                            result.targets.targets,
                            frame_loss.assignment,
                            identity_keys=result.targets.identity_keys_by_batch[0],
                        )
                        visuals: list[dict[str, Any]] = []
                        if (
                            checkpoint_global_step in {0, args.steps}
                            and item.sample_key in visual_sample_keys
                        ):
                            visuals = render_task_independent_entity_visuals(
                                output_root=args.run_dir,
                                global_step=checkpoint_global_step,
                                input_weight_global_step=checkpoint_global_step,
                                weight_boundary="fixed_checkpoint_evaluation",
                                rank=rank,
                                host_items=planned.training.host_items,
                                model_inputs=collated.model_inputs,
                                relation=result.relation,
                                target_bundle=result.targets,
                                set_loss=frame_loss,
                                sample_keys=collated.routing.sample_keys,
                                merge_size=merge_size,
                            )
                        evidence.update(
                            {
                                "checkpoint_global_step": checkpoint_global_step,
                                "partition": item.partition,
                                "ordinal": item.ordinal,
                                "rank": rank,
                                "task_key": item.task_key,
                                "task_used_by_entity_objective": False,
                                "segment_index": item.segment_index,
                                "source_episode_index": item.source_episode_index,
                                "source_global_index": item.source_global_index,
                                "transition_index": item.transition_index,
                                "sample_key": item.sample_key,
                                "source_digest": collated.source_digest,
                                "objective_total": _float(result.objective.objective.total),
                                "mask_focal": _float(frame_loss.mask_focal),
                                "mask_dice": _float(frame_loss.mask_dice),
                                "existence_focal": _float(frame_loss.existence_focal),
                                "ownership_nll": _float(frame_loss.ownership_nll),
                                "forward_seconds": forward_seconds,
                                "visual_artifacts": visuals,
                            }
                        )
                    except BaseException as error:
                        evidence_error = error
                    _distributed_phase_error(
                        error=evidence_error,
                        phase=f"checkpoint-{checkpoint_global_step}-sample-{local_index}-evidence",
                        rank=rank,
                        dist_module=dist,
                    )
                    if evidence is None:
                        raise RuntimeError("P1 entity evaluation evidence vanished")
                    local_samples.append(evidence)
                    if rank == 0 and (
                        local_index == 0
                        or local_index + 1 == len(local_items)
                        or (local_index + 1) % 5 == 0
                    ):
                        print(
                            json.dumps(
                                {
                                    "event": "task_independent_p1_evaluation_progress",
                                    "checkpoint_global_step": checkpoint_global_step,
                                    "completed_per_rank": local_index + 1,
                                    "samples_per_rank": len(local_items),
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

            gathered_samples: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
            dist.all_gather_object(gathered_samples, local_samples)
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    samples = sorted(
                        (sample for rank_samples in gathered_samples for sample in rank_samples),
                        key=lambda sample: int(sample["ordinal"]),
                    )
                    expected_keys = [item.sample_key for item in entity_evaluation_plan.items]
                    if [sample["sample_key"] for sample in samples] != expected_keys:
                        raise RuntimeError("P1 curve evaluation sample set changed")
                    evaluation_input_sha256 = hashlib.sha256(
                        json.dumps(
                            [
                                {
                                    "sample_key": sample["sample_key"],
                                    "source_digest": sample["source_digest"],
                                }
                                for sample in samples
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    payload = {
                        "schema": P1_CURVE_SNAPSHOT_SCHEMA,
                        "status": "PASS",
                        "checkpoint_global_step": checkpoint_global_step,
                        "architecture_identity": TASK_INDEPENDENT_ENTITY_POSTERIOR,
                        "visual_lattice": args.visual_lattice,
                        "processor_lattice": processor_lattice,
                        "task_scorer_present": False,
                        "task_used_by_entity_objective": False,
                        "action_suffix_executed": False,
                        "posterior_input_mode": "current_frame_discovery_only",
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "stream_plan_sha256": plan.plan_sha256,
                        "representation_split_sha256": representation_split.artifact_sha256,
                        "entity_evaluation_plan_sha256": (entity_evaluation_plan.artifact_sha256),
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "samples": samples,
                        "partition_summaries": {
                            partition: summarize_entity_evaluation_partition(
                                samples,
                                partition=partition,
                            )
                            for partition in ENTITY_EVALUATION_PARTITIONS
                        },
                    }
                    artifact_sha256 = hashlib.sha256(
                        json.dumps(
                            payload,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    snapshot = {**payload, "artifact_sha256": artifact_sha256}
                    destination = (
                        args.run_dir / f"entity_evaluation_step_{checkpoint_global_step:06d}.json"
                    )
                    write_text_durable_exclusive(
                        destination,
                        json.dumps(snapshot, indent=2, sort_keys=True) + "\n",
                    )
                    publication[0] = {
                        "artifact_sha256": artifact_sha256,
                        "file_sha256": _sha256(destination),
                        "path": str(destination),
                        "checkpoint_global_step": checkpoint_global_step,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "partition_summaries": snapshot["partition_summaries"],
                    }
                except BaseException as error:
                    publication[0] = {
                        "error": f"{type(error).__name__}: {error}",
                    }
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(f"P1 curve evaluation publication failed: {publication[0]}")
            dist.barrier()
            return publication[0]

        if 0 in registered_evaluation_steps:
            evaluation_snapshot_reports.append(run_entity_evaluation(0))

        rank_steps: list[dict[str, Any]] = []
        maximum_peak_reserved_bytes = int(args.maximum_peak_reserved_gib * 1024**3)
        for optimizer_step in range(args.steps):
            planned = None
            collated = None
            visual_lattice_contract = None
            adr175_input_receipt: dict[str, str] | None = None
            prepare_error: BaseException | None = None
            try:
                planned = build_planned_native_calvin_batch(
                    plan,
                    dataset,
                    optimizer_step=optimizer_step,
                    rank=rank,
                    world_size=P1_WORLD_SIZE,
                    gradient_accumulation_steps=1,
                    accumulation_index=0,
                    device=device,
                    dtype=torch.bfloat16,
                    maximum_control_tokens=(
                        args.maximum_control_tokens if adr175_mode else None
                    ),
                )
                collated = collate_native_calvin_training_batch(
                    planned.training,
                    feature_transform=feature_transform,
                    collator=VLADataCollatorWithPacking(),
                    augmentation_seeds=planned.augmentation_seeds,
                    source_digest=planned.source_digest,
                )
                visual_lattice_contract = _validate_visual_lattice_inputs(
                    dict(collated.model_inputs),
                    visual_lattice=args.visual_lattice,
                    merge_size=merge_size,
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
                )
                collated = materialize_native_flow_randomness(collated, planned)
                if adr175_mode:
                    adr175_input_receipt = _adr175_rank_step_receipt(
                        planned=planned,
                        collated=collated,
                    )
                optimizer.zero_grad(set_to_none=True)
            except BaseException as error:
                prepare_error = error
            _distributed_phase_error(
                error=prepare_error,
                phase=f"p1-step-{optimizer_step}-prepare",
                rank=rank,
                dist_module=dist,
            )
            if (
                planned is None
                or collated is None
                or visual_lattice_contract is None
                or (adr175_mode and adr175_input_receipt is None)
            ):
                raise RuntimeError("P1 step preparation vanished")

            started = time.perf_counter()
            result = None
            row_bindings = None
            official_result = None
            training_loss = None
            attention_summary: dict[str, Any] | None = None
            target_audit: tuple[dict[str, Any], ...] = ()
            objective_error: BaseException | None = None
            try:
                if adr175_mode and args.adr175_arm == "lbot":
                    official_result = run_official_policy_training_forward(
                        policy,
                        model_inputs=collated.model_inputs,
                    )
                    training_loss = official_result.official_total_loss
                    row_bindings = ()
                else:
                    collector = (
                        RegisteredActionPosteriorReceiptCollector(
                            registered_layer_indices=registered_layer_indices
                        )
                        if adr175_mode
                        else None
                    )
                    result = run_task_independent_calvin_current_frame_objective(
                        policy,
                        batch=collated,
                        physical_sidecar=physical_sidecar,
                        objective_config=objective_config,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        minimum_supervised_fraction=args.minimum_supervised_fraction,
                        capacity_seeds=planned.augmentation_seeds,
                        action_attention_callback=collector,
                    )
                    official_result = result.policy_forward
                    training_loss = result.objective.objective.total
                    if result.context.posterior_state is None:
                        raise RuntimeError("P1 observation root did not produce posterior rows")
                    if result.context.previous_state is not None or bool(
                        result.context.previous_state_valid.any()
                    ):
                        raise RuntimeError("P1 current-frame objective consumed recurrent state")
                    row_bindings = physical_frame_row_bindings(
                        result.targets,
                        result.objective.frame_losses[0].assignment,
                        capacity=args.capacity,
                    )
                    if collector is not None:
                        receipts = collector.finalize()
                        if not receipts:
                            raise RuntimeError("ADR-175 collected no native attention receipts")
                        head_count = receipts[0].posterior_attention.shape[1]
                        if any(
                            receipt.posterior_attention.shape[1] != head_count
                            for receipt in receipts
                        ) or head_count <= max(ADR175_GUIDEDVLA_OBJECT_HEAD_INDICES):
                            raise RuntimeError(
                                "ADR-175 registered layers have incompatible action heads"
                            )
                        target_weights, target_valid, target_audit = (
                            adr175_attention_targets(
                                bindings_by_batch=row_bindings,
                                batch=collated,
                                dtype=receipts[0].posterior_attention.dtype,
                            )
                        )
                        head_indices = torch.tensor(
                            ADR175_GUIDEDVLA_OBJECT_HEAD_INDICES,
                            dtype=torch.long,
                            device=device,
                        )
                        attention_results = tuple(
                            action_posterior_target_mass_loss(
                                receipt.posterior_attention,
                                target_row_weights=target_weights,
                                target_valid=target_valid,
                                head_indices=head_indices,
                            )
                            for receipt in receipts
                        )
                        attention_loss = torch.stack(
                            tuple(item.loss for item in attention_results)
                        ).mean()
                        if adr175_attention_active:
                            training_loss = (
                                training_loss
                                + ADR175_NATIVE_ATTENTION_WEIGHT * attention_loss
                            )
                        valid_entries = attention_results[0].valid_entries
                        if any(
                            not torch.equal(item.valid_entries, valid_entries)
                            for item in attention_results[1:]
                        ):
                            raise RuntimeError(
                                "ADR-175 registered layers disagree on target validity"
                            )
                        target_mass = torch.stack(
                            tuple(item.target_mass for item in attention_results)
                        ).mean(dim=0)
                        adoption = torch.stack(
                            tuple(item.total_posterior_mass for item in attention_results)
                        ).mean(dim=0)
                        if bool(valid_entries.any()):
                            conditional = target_mass / adoption.clamp_min(1e-6)
                            attention_summary = {
                                "loss": _float(attention_loss),
                                "posterior_adoption": _float(
                                    adoption.masked_select(valid_entries).mean()
                                ),
                                "target_mass": _float(
                                    target_mass.masked_select(valid_entries).mean()
                                ),
                                "conditional_selectivity": _float(
                                    conditional.masked_select(valid_entries).mean()
                                ),
                                "valid_entry_count": int(valid_entries.sum().item()),
                                "registered_layer_indices": list(
                                    registered_layer_indices
                                ),
                                "registered_head_indices": list(
                                    ADR175_GUIDEDVLA_OBJECT_HEAD_INDICES
                                ),
                            }
                        else:
                            attention_summary = {
                                "loss": _float(attention_loss),
                                "posterior_adoption": None,
                                "target_mass": None,
                                "conditional_selectivity": None,
                                "valid_entry_count": 0,
                                "registered_layer_indices": list(
                                    registered_layer_indices
                                ),
                                "registered_head_indices": list(
                                    ADR175_GUIDEDVLA_OBJECT_HEAD_INDICES
                                ),
                            }
            except BaseException as error:
                objective_error = error
            _distributed_phase_error(
                error=objective_error,
                phase=f"p1-step-{optimizer_step}-objective",
                rank=rank,
                dist_module=dist,
            )
            if training_loss is None or official_result is None or row_bindings is None:
                raise RuntimeError("P1 objective result vanished")

            backward_error: BaseException | None = None
            try:
                training_loss.backward()
            except BaseException as error:
                optimizer.zero_grad(set_to_none=True)
                backward_error = error
            _distributed_phase_error(
                error=backward_error,
                phase=f"p1-step-{optimizer_step}-backward",
                rank=rank,
                dist_module=dist,
            )

            gradient_metrics = _add_action_only_gradient_summary(
                _distributed_gradient_metrics(
                    policy,
                    (
                        ("native_graph", "picf_native_graph"),
                        ("vlm_host", ".qwenvl."),
                        *_ACTION_ONLY_GRADIENT_METRICS,
                    ),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
            )
            gradient_error: BaseException | None = None
            try:
                if not bool(gradient_metrics["all_finite"]):
                    raise RuntimeError("P1 produced a non-finite gradient")
                if (
                    (not adr175_mode or adr175_picf_active)
                    and float(gradient_metrics.get("native_graph_norm", 0.0)) <= 0
                ):
                    raise RuntimeError("P1 produced no gradient in the entity graph")
                if float(gradient_metrics.get("vlm_host_norm", 0.0)) <= 0:
                    raise RuntimeError("P1 produced no gradient in the shared LingBot VLM")
                action_gradient_elements = int(
                    gradient_metrics.get("action_only_elements", 0)
                )
                action_gradient_norm = float(gradient_metrics.get("action_only_norm", 0.0))
                if joint_action_mode:
                    if action_gradient_elements <= 0 or action_gradient_norm <= 0:
                        raise RuntimeError(
                            "current-frame joint action produced no action-only gradient"
                        )
                elif action_gradient_elements != 0:
                    raise RuntimeError("P1 unexpectedly trained the action-only partition")
            except BaseException as error:
                gradient_error = error
            _distributed_phase_error(
                error=gradient_error,
                phase=f"p1-step-{optimizer_step}-gradient-audit",
                rank=rank,
                dist_module=dist,
            )

            clipped = None
            clip_error: BaseException | None = None
            try:
                clipped = clip_lingbot_distributed_l2_grad_norm_(
                    tuple(policy.parameters()),
                    args.max_grad_norm,
                    device=device,
                    dist_module=dist,
                    torch_module=torch,
                    error_if_nonfinite=True,
                )
                gradient_metrics["preclip_global_norm"] = float(clipped)
            except BaseException as error:
                clip_error = error
            _distributed_phase_error(
                error=clip_error,
                phase=f"p1-step-{optimizer_step}-gradient-clip",
                rank=rank,
                dist_module=dist,
            )
            if clipped is None:
                raise RuntimeError("P1 gradient clipping result vanished")

            update_error: BaseException | None = None
            try:
                optimizer.step()
            except BaseException as error:
                update_error = error
            _distributed_phase_error(
                error=update_error,
                phase=f"p1-step-{optimizer_step}-optimizer-step",
                rank=rank,
                dist_module=dist,
            )

            frame_loss = None
            targets = None
            step_report = None
            post_update_error: BaseException | None = None
            try:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                step_time_s = time.perf_counter() - started
                visual_artifacts: list[dict[str, Any]] = []
                if result is not None:
                    frame_loss = result.objective.frame_losses[0]
                    targets = result.targets.targets
                if result is not None and optimizer_step in {0, args.steps - 1}:
                    visual_artifacts = render_task_independent_entity_visuals(
                        output_root=args.run_dir,
                        global_step=optimizer_step + 1,
                        input_weight_global_step=optimizer_step,
                        rank=rank,
                        host_items=planned.training.host_items,
                        model_inputs=collated.model_inputs,
                        relation=result.relation,
                        target_bundle=result.targets,
                        set_loss=frame_loss,
                        sample_keys=collated.routing.sample_keys,
                        merge_size=merge_size,
                    )
                step_report = {
                    "global_step": optimizer_step + 1,
                    "sample_keys": list(collated.routing.sample_keys),
                    "lane_ids": list(collated.routing.lane_ids),
                    "frame_indices": list(collated.routing.frame_indices),
                    "reset": list(collated.routing.reset),
                    "source_digest": collated.source_digest,
                    "visual_lattice": args.visual_lattice,
                    "visual_lattice_contract": visual_lattice_contract,
                    "previous_state_ages": [0] * collated.routing.batch_size,
                    "optimizer_lags": [0] * collated.routing.batch_size,
                    "previous_state_input_absent": True,
                    "adr175_arm": args.adr175_arm,
                    "picf_graph_active": result is not None,
                    "objective_total": _float(training_loss),
                    "base_objective_total": (
                        None
                        if result is None
                        else _float(result.objective.objective.total)
                    ),
                    "structural_family": (
                        None
                        if result is None
                        else _float(result.objective.objective.family_terms["structural"])
                    ),
                    "action_family": (
                        _float(official_result.official_total_loss)
                        if result is None
                        else _float(result.objective.objective.family_terms["action"])
                    ),
                    "official_action_loss": _float(official_result.official_action_loss),
                    "official_policy_loss": _float(official_result.official_total_loss),
                    "mask_focal": None if frame_loss is None else _float(frame_loss.mask_focal),
                    "mask_dice": None if frame_loss is None else _float(frame_loss.mask_dice),
                    "existence_focal": (
                        None if frame_loss is None else _float(frame_loss.existence_focal)
                    ),
                    "ownership_nll": (
                        None if frame_loss is None else _float(frame_loss.ownership_nll)
                    ),
                    "track_counts": (
                        []
                        if targets is None
                        else [
                            int(value) for value in targets.track_valid.sum(dim=1).tolist()
                        ]
                    ),
                    "capacity_censored_counts": (
                        []
                        if targets is None
                        else [
                            int(value)
                            for value in targets.capacity_censored.sum(dim=1).tolist()
                        ]
                    ),
                    "matched_row_counts": (
                        []
                        if frame_loss is None
                        else [
                            int(value)
                            for value in (
                                frame_loss.assignment.row_to_track >= 0
                            ).sum(dim=1).tolist()
                        ]
                    ),
                    "row_bindings": [[list(item) for item in value] for value in row_bindings],
                    "relation_interface": (
                        None if result is None else result.relation.interface
                    ),
                    "relation_temperature": (
                        None
                        if result is None
                        else _float(result.relation.relation_temperature)
                    ),
                    "policy_forward_absent": False,
                    "attention_summary": attention_summary,
                    "target_audit": list(target_audit),
                    "adr175_input_receipt": adr175_input_receipt,
                    "gradient_metrics": gradient_metrics,
                    "visual_artifacts": visual_artifacts,
                    "step_time_s": step_time_s,
                    "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
                }
                if step_report["peak_cuda_reserved_bytes"] > maximum_peak_reserved_bytes:
                    raise RuntimeError("P1 exceeded the registered CUDA reservation budget")
            except BaseException as error:
                post_update_error = error
            _distributed_phase_error(
                error=post_update_error,
                phase=f"p1-step-{optimizer_step}-post-update-audit",
                rank=rank,
                dist_module=dist,
            )
            if step_report is None or (
                (not adr175_mode or adr175_picf_active)
                and (frame_loss is None or targets is None)
            ):
                raise RuntimeError("P1 post-update report vanished")
            rank_steps.append(step_report)
            step_log_error: BaseException | None = None
            try:
                if rank == 0:
                    current = rank_steps[-1]
                    print(
                        json.dumps(
                            {
                                "event": "task_independent_p1_step",
                                "global_step": current["global_step"],
                                "objective_total": current["objective_total"],
                                "mask_focal": current["mask_focal"],
                                "mask_dice": current["mask_dice"],
                                "existence_focal": current["existence_focal"],
                                "ownership_nll": current["ownership_nll"],
                                "gradient_metrics": current["gradient_metrics"],
                                "step_time_s": current["step_time_s"],
                                "peak_cuda_reserved_bytes": current["peak_cuda_reserved_bytes"],
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            except BaseException as error:
                step_log_error = error
            _distributed_phase_error(
                error=step_log_error,
                phase=f"p1-step-{optimizer_step}-log",
                rank=rank,
                dist_module=dist,
            )
            completed_step = optimizer_step + 1
            if completed_step in registered_evaluation_steps:
                evaluation_snapshot_reports.append(run_entity_evaluation(completed_step))

        gathered: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
        dist.all_gather_object(
            gathered,
            {
                "rank": rank,
                "steps": rank_steps,
            },
        )
        publication_error: list[str | None] = [None]
        report: dict[str, Any] | None = None
        if rank == 0:
            try:
                if [item["checkpoint_global_step"] for item in evaluation_snapshot_reports] != list(
                    registered_evaluation_steps
                ):
                    raise RuntimeError("P1 curve evaluation checkpoints changed")
                if (
                    evaluation_snapshot_reports
                    and len(
                        {item["evaluation_input_sha256"] for item in evaluation_snapshot_reports}
                    )
                    != 1
                ):
                    raise RuntimeError("P1 curve evaluation inputs differ across checkpoints")
                for step_index in range(args.steps):
                    sample_sets = [
                        set(rank_report["steps"][step_index]["sample_keys"])
                        for rank_report in gathered
                    ]
                    if sample_sets[0].intersection(sample_sets[1]):
                        raise RuntimeError("P1 distributed ranks consumed an overlapping sample")
                    for rank_report in gathered:
                        item = rank_report["steps"][step_index]
                        if item["global_step"] != step_index + 1:
                            raise RuntimeError("P1 rank report has a non-contiguous step")
                        expected_relation_interface = (
                            None
                            if adr175_mode and args.adr175_arm == "lbot"
                            else TASK_INDEPENDENT_PHYSICAL_INTERFACE
                        )
                        if item["relation_interface"] != expected_relation_interface:
                            raise RuntimeError("P1 rank exposed the wrong relation interface")
                        if item["policy_forward_absent"] == joint_action_mode:
                            raise RuntimeError(
                                "current-frame action mode and executed policy suffix differ"
                            )
                        expected_visual = (
                            step_index in {0, args.steps - 1}
                            and (not adr175_mode or adr175_picf_active)
                        )
                        if bool(item["visual_artifacts"]) != expected_visual:
                            raise RuntimeError("P1 rank omitted or added an endpoint visual")
                        for artifact in item["visual_artifacts"]:
                            if (
                                artifact.get("schema") != TASK_INDEPENDENT_ENTITY_VISUAL_SCHEMA
                                or artifact.get("task_used_by_entity_objective") is not False
                                or artifact.get("loss_only_labels_visible_to_model") is not False
                            ):
                                raise RuntimeError(
                                    "P1 visual violates the task-free evidence contract"
                                )
                report = {
                    "schema": P1_REPORT_SCHEMA,
                    "status": "PASS",
                    "architecture_identity": (
                        "released-lingbot-action-only"
                        if adr175_mode and args.adr175_arm == "lbot"
                        else TASK_INDEPENDENT_ENTITY_POSTERIOR
                    ),
                    "relation_interface": (
                        None
                        if adr175_mode and args.adr175_arm == "lbot"
                        else TASK_INDEPENDENT_PHYSICAL_INTERFACE
                    ),
                    "task_scorer_present": False,
                    "action_suffix_executed": joint_action_mode,
                    "posterior_input_mode": (
                        "current_frame_joint_action"
                        if joint_action_mode
                        else "current_frame_discovery_only"
                    ),
                    "checkpoint_published": False,
                    "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                    "source_patch_sha256": patch_report["patch_sha256"],
                    "patched_source_sha256": actual_hashes,
                    "implementation_files": implementation_files,
                    "implementation_sha256": implementation_sha256,
                    "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                    "checkpoint_assets": checkpoint_report["checkpoint_assets"],
                    "processor_revision": QWEN_PROCESSOR_REVISION,
                    "processor_assets": processor_report["processor_assets"],
                    "dataset_contract": dataset_contract_report,
                    "physical_sidecar_manifest_sha256": physical_sidecar.manifest_sha256,
                    "plan_sha256": plan.plan_sha256,
                    "curve_mode": curve_mode,
                    "registered_evaluation_steps": list(registered_evaluation_steps),
                    "representation_split_sha256": (
                        None
                        if representation_split is None
                        else representation_split.artifact_sha256
                    ),
                    "entity_evaluation_plan_sha256": (
                        None
                        if entity_evaluation_plan is None
                        else entity_evaluation_plan.artifact_sha256
                    ),
                    "evaluation_snapshots": evaluation_snapshot_reports,
                    "model_family_sha256": model_family_sha256,
                    "adr175": (
                        None
                        if not adr175_mode
                        else {
                            "arm": args.adr175_arm,
                            "contract_artifact_sha256": adr175_contract.artifact_sha256,
                            "contract_file_sha256": args.adr175_contract_sha256,
                            "matched_arm_input_sha256": (
                                adr175_contract.matched_arm_input_sha256
                            ),
                            "shared_initialization_sha256": (
                                shared_initialization_sha256
                            ),
                            "shared_optimizer_contract_sha256": (
                                shared_optimizer_contract_sha256
                            ),
                            "shared_optimizer_manifest": shared_optimizer_manifest,
                            "picf_graph_sha256": picf_graph_sha256,
                            "picf_initialization_sha256": (
                                picf_initialization_sha256
                            ),
                        }
                    ),
                    "world_size": P1_WORLD_SIZE,
                    "steps": args.steps,
                    "seed": args.seed,
                    "graph": None if graph_config is None else asdict(graph_config),
                    "objective": asdict(objective_config),
                    "qwen_vision_geometry": {
                        "patch_size": patch_size,
                        "spatial_merge_size": merge_size,
                        "visual_lattice": args.visual_lattice,
                        "processor_lattice": processor_lattice,
                    },
                    "fsdp2_placement": args.fsdp2_placement,
                    "cuda_allocator": args.cuda_allocator,
                    "gradient_checkpointing": True,
                    "parameter_storage": parameter_storage,
                    "parameter_manifest": {
                        "parameter_count": parameter_manifest.parameter_count,
                        "trainable_numel": parameter_manifest.trainable_numel,
                        "schema_sha256": parameter_manifest.schema_sha256,
                    },
                    "parameter_scope": (
                        "full_joint_action"
                        if representation_parameter_scope is None
                        else "representation_only"
                    ),
                    "representation_parameter_scope": (
                        None
                        if representation_parameter_scope is None
                        else representation_parameter_scope.as_dict()
                    ),
                    "alignment_teacher_prune": alignment_teacher_prune,
                    "maximum_peak_reserved_bytes": maximum_peak_reserved_bytes,
                    "rank_reports": gathered,
                }
                p1_report_path = (
                    args.run_dir / f"p1_boundary_steps_{args.steps}.json"
                    if staged_mode
                    else args.output
                )
                write_text_durable_exclusive(
                    p1_report_path,
                    json.dumps(report, indent=2, sort_keys=True) + "\n",
                )
            except BaseException as error:
                publication_error[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(publication_error, src=0)
        if publication_error[0] is not None:
            raise RuntimeError(f"P1 report publication failed: {publication_error[0]}")
        dist.barrier()
        boundary_log_error: BaseException | None = None
        try:
            if rank == 0:
                if report is None:
                    raise RuntimeError("rank zero lost the P1 report")
                print(json.dumps(report, indent=2, sort_keys=True))
        except BaseException as error:
            boundary_log_error = error
        _distributed_phase_error(
            error=boundary_log_error,
            phase="p1-boundary-log",
            rank=rank,
            dist_module=dist,
        )

        if staged_mode:
            if (
                predictive_cache is None
                or p2_predictive_report is None
                or representation_split is None
                or p2_representation_split is None
                or p2_plan is None
                or p2_schedule_sha256 is None
                or p2_schedule_file_sha256 is None
            ):
                raise RuntimeError("staged P2 lost its validated training contracts")
            if len(p2_selected_records) != P1_WORLD_SIZE * args.staged_p2_steps:
                raise RuntimeError("staged P2 record schedule changed")
            p1_boundary_optimizer_state = None
            p1_boundary_optimizer_audit = None
            p1_boundary_host_optimizer_audit = None
            p1_boundary_local_error: BaseException | None = None
            try:
                p1_boundary_optimizer_state = _optimizer_state_family_counts(policy, optimizer)
                if (
                    p1_boundary_optimizer_state["native_graph"] <= 0
                    or p1_boundary_optimizer_state["vlm_host"] <= 0
                    or p1_boundary_optimizer_state["predictive_readout"] != 0
                    or p1_boundary_optimizer_state["action_only"] != 0
                ):
                    raise RuntimeError("P1 boundary optimizer families violate staged isolation")
                p1_boundary_optimizer_audit = _validate_optimizer_state(
                    optimizer,
                    torch,
                    expected_step=args.steps,
                )
            except BaseException as error:
                p1_boundary_local_error = error
            _distributed_phase_error(
                error=p1_boundary_local_error,
                phase="staged-p1-boundary-local",
                rank=rank,
                dist_module=dist,
            )
            p1_boundary_host_optimizer_audit = _audit_optimizer_family_state(
                policy,
                optimizer,
                torch,
                family="P1 shared VLM host",
                fragment=".qwenvl.",
                expected_adamw_step=args.steps,
                dist_module=dist,
            )
            if (
                p1_boundary_optimizer_state is None
                or p1_boundary_optimizer_audit is None
                or p1_boundary_host_optimizer_audit is None
            ):
                raise RuntimeError("staged P1 boundary audit vanished")
            del result, collated, planned, frame_loss, targets
            torch.cuda.empty_cache()
            dist.barrier()

            staged_objective_config = TaskIndependentEntityObjectiveConfig(
                action_weight=0.0,
                entity_weight=1.0,
                predictive_weight=1.0,
                mask_focal_weight=args.mask_focal_weight,
                mask_dice_weight=args.mask_dice_weight,
                existence_weight=args.existence_weight,
                ownership_weight=args.ownership_weight,
            )
            prior_stepper = LingBotNativePriorStepper(policy, graph)
            p2_rank_steps: list[dict[str, Any]] = []
            p2_predictive_valid_seen = 0
            predictive_activation_step: int | None = None
            for p2_optimizer_step in range(args.staged_p2_steps):
                schedule_index = p2_optimizer_step * P1_WORLD_SIZE + rank
                selection = p2_selected_records[schedule_index]
                record = selection.record
                segment = selection.segment
                episode = selection.episode
                transition_index = selection.transition_index

                def staged_replay(
                    sample_key: str,
                    *,
                    replay_offset: int,
                    staged_step: int = p2_optimizer_step,
                ) -> PlannedNativeCALVINReplayBatch:
                    return build_native_calvin_replay_batch(
                        dataset,
                        sample_key=sample_key,
                        lane_id=rank,
                        episode_instance_id=(
                            f"task-independent-staged-p2/step-{staged_step}/rank-{rank}"
                        ),
                        optimizer_step=args.steps + staged_step,
                        replay_seed=(
                            args.seed
                            + 1_000_003
                            + staged_step * P1_WORLD_SIZE * 17
                            + rank * 1009
                            + replay_offset
                        ),
                        device=device,
                        dtype=torch.bfloat16,
                    )

                prefix_planned = None
                source_planned = None
                future_planned = None
                prefix_batch = None
                source_batch = None
                source_sample_key = ""
                prefix_sample_key = ""
                future_sample_keys: tuple[str, ...] = ()
                prepare_error: BaseException | None = None
                try:
                    if int(segment.episode_index) not in frozenset(
                        p2_representation_split.training_source_episode_indices
                    ):
                        raise RuntimeError("staged P2 consumed a non-training source episode")
                    source_sample_key = episode.sample_keys[transition_index]
                    prefix_sample_key = episode.sample_keys[transition_index - 1]
                    future_sample_keys = tuple(
                        episode.sample_keys[
                            transition_index + 1 : transition_index + args.p2_horizon + 1
                        ]
                    )
                    if dataset.source_global_index_by_key(
                        source_sample_key
                    ) != record.source_global_index or tuple(
                        dataset.source_global_index_by_key(key) for key in future_sample_keys
                    ) != tuple(
                        range(
                            record.source_global_index + 1,
                            record.source_global_index + args.p2_horizon + 1,
                        )
                    ):
                        raise RuntimeError("staged P2 source/future resolution changed")
                    prefix_planned = staged_replay(prefix_sample_key, replay_offset=0)
                    source_planned = staged_replay(source_sample_key, replay_offset=1)
                    future_planned = tuple(
                        staged_replay(sample_key, replay_offset=2 + offset)
                        for offset, sample_key in enumerate(future_sample_keys)
                    )
                    prefix_batch = collate_replay(prefix_planned)
                    source_batch = collate_replay(source_planned)
                except BaseException as error:
                    prepare_error = error
                _distributed_phase_error(
                    error=prepare_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-prepare",
                    rank=rank,
                    dist_module=dist,
                )
                if (
                    prefix_planned is None
                    or source_planned is None
                    or future_planned is None
                    or prefix_batch is None
                    or source_batch is None
                    or not source_sample_key
                    or not prefix_sample_key
                    or len(future_sample_keys) != args.p2_horizon
                ):
                    raise RuntimeError("staged P2 preparation vanished")
                prefix_result = None
                prefix_state = None
                prefix_row_bindings = None
                previous_state = None
                prefix_error: BaseException | None = None
                try:
                    with torch.no_grad():
                        prefix_result = run_task_independent_calvin_current_frame_diagnostic(
                            policy,
                            batch=prefix_batch,
                            physical_sidecar=physical_sidecar,
                            objective_config=objective_config,
                            patch_size=patch_size,
                            merge_size=merge_size,
                            minimum_supervised_fraction=args.minimum_supervised_fraction,
                            capacity_seeds=prefix_planned.augmentation_seeds,
                        )
                    prefix_state = prefix_result.context.posterior_state
                    if prefix_state is None:
                        raise RuntimeError("staged P2 prefix produced no posterior state")
                    prefix_row_bindings = physical_frame_row_bindings(
                        prefix_result.targets,
                        prefix_result.objective.frame_losses[0].assignment,
                        capacity=args.capacity,
                    )
                    if not prefix_row_bindings[0]:
                        raise RuntimeError("staged P2 prefix established no physical row gauge")
                    previous_state = NativePosteriorState(prefix_state.rows.detach())
                except BaseException as error:
                    prefix_error = error
                _distributed_phase_error(
                    error=prefix_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-prefix",
                    rank=rank,
                    dist_module=dist,
                )
                if prefix_row_bindings is None or previous_state is None:
                    raise RuntimeError("staged P2 prefix audit vanished")
                del prefix_result, prefix_state, prefix_batch

                future_controls = tuple(value.training.controls for value in future_planned)

                def staged_predictive_rollout(
                    state: NativePosteriorState,
                    controls: tuple[Any, ...] = future_controls,
                ) -> Any:
                    request = make_native_future_request(
                        source=PredictionSource.PRIOR,
                        batch_size=state.batch_size,
                        horizon=args.p2_horizon,
                        valid=torch.ones(
                            state.batch_size,
                            dtype=torch.bool,
                            device=device,
                        ),
                        device=device,
                        dtype=state.rows.dtype,
                        route_id=predictive_cache.contract.route_id,
                        address_width=graph.config.prediction_address_width,
                    )
                    return rollout_native_prior_prediction(
                        prior_stepper,
                        state,
                        controls,
                        request=request,
                        target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
                    )

                optimizer.zero_grad(set_to_none=True)
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                p2_result = None
                carried_binding_count: int | None = None
                objective_error: BaseException | None = None
                try:
                    p2_result = run_task_independent_calvin_sequence_objective(
                        policy,
                        batches=(source_batch,),
                        physical_sidecar=physical_sidecar,
                        objective_config=staged_objective_config,
                        patch_size=patch_size,
                        merge_size=merge_size,
                        previous_state=previous_state,
                        previous_state_valid=torch.ones(
                            1,
                            dtype=torch.bool,
                            device=device,
                        ),
                        prior_row_bindings_by_batch=prefix_row_bindings,
                        predictive_rollout_factory=staged_predictive_rollout,
                        predictive_cache=predictive_cache,
                        minimum_supervised_fraction=args.minimum_supervised_fraction,
                        capacity_seeds=prefix_planned.augmentation_seeds,
                    )
                    carried_binding_count = _require_carried_bindings_preserved(
                        prefix_row_bindings,
                        p2_result.objective.row_bindings_by_batch,
                        capacity=args.capacity,
                    )
                except BaseException as error:
                    objective_error = error
                _distributed_phase_error(
                    error=objective_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-objective",
                    rank=rank,
                    dist_module=dist,
                )
                if p2_result is None or carried_binding_count is None:
                    raise RuntimeError("staged P2 objective vanished")
                valid_counts = {
                    name: int(value)
                    for name, value in p2_result.objective.objective.valid_counts.items()
                }
                current_predictive_valid = sum(
                    count for name, count in valid_counts.items() if name.startswith("rollout/")
                )
                backward_error: BaseException | None = None
                try:
                    p2_result.objective.objective.total.backward()
                except BaseException as error:
                    optimizer.zero_grad(set_to_none=True)
                    backward_error = error
                _distributed_phase_error(
                    error=backward_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-backward",
                    rank=rank,
                    dist_module=dist,
                )

                gradient_metrics = _add_action_only_gradient_summary(
                    _distributed_gradient_metrics(
                        policy,
                        (
                            ("native_graph", "picf_native_graph"),
                            ("vlm_host", ".qwenvl."),
                            ("predictive_readout", "predictive_readouts.dino_video"),
                            *_ACTION_ONLY_GRADIENT_METRICS,
                        ),
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                )
                gradient_error: BaseException | None = None
                try:
                    if not bool(gradient_metrics["all_finite"]):
                        raise RuntimeError("staged P2 produced non-finite gradients")
                    for family in ("native_graph", "vlm_host"):
                        if float(gradient_metrics[f"{family}_norm"]) <= 0:
                            raise RuntimeError(f"staged P2 produced no {family} gradient")
                    if float(gradient_metrics["predictive_readout_norm"]) <= 0:
                        raise RuntimeError(
                            "staged P2 global step produced no predictive-readout gradient"
                        )
                    if int(gradient_metrics["action_only_elements"]) != 0:
                        raise RuntimeError(
                            "staged P2 unexpectedly trained the action-only partition"
                        )
                except BaseException as error:
                    gradient_error = error
                _distributed_phase_error(
                    error=gradient_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-gradient-audit",
                    rank=rank,
                    dist_module=dist,
                )
                predictive_gradient_active = float(gradient_metrics["predictive_readout_norm"]) > 0

                clipped = None
                clip_error: BaseException | None = None
                try:
                    clipped = torch.nn.utils.clip_grad_norm_(
                        policy.parameters(),
                        args.max_grad_norm,
                        error_if_nonfinite=True,
                        foreach=False,
                    )
                    full_tensor = getattr(clipped, "full_tensor", None)
                    if callable(full_tensor):
                        clipped = full_tensor()
                    gradient_metrics["preclip_global_norm"] = float(clipped.item())
                except BaseException as error:
                    clip_error = error
                _distributed_phase_error(
                    error=clip_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-gradient-clip",
                    rank=rank,
                    dist_module=dist,
                )
                if clipped is None:
                    raise RuntimeError("staged P2 gradient clipping result vanished")

                update_error: BaseException | None = None
                try:
                    optimizer.step()
                except BaseException as error:
                    update_error = error
                _distributed_phase_error(
                    error=update_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-optimizer-step",
                    rank=rank,
                    dist_module=dist,
                )

                family_state_entries = None
                optimizer_family_audits = None
                elapsed_seconds = None
                peak_reserved_bytes = None
                post_update_local_error: BaseException | None = None
                try:
                    p2_predictive_valid_seen += current_predictive_valid
                    if predictive_gradient_active and predictive_activation_step is None:
                        predictive_activation_step = p2_optimizer_step + 1
                    family_state_entries = _p2_optimizer_state_families(
                        policy,
                        optimizer,
                        require_predictive=True,
                    )
                    if (
                        predictive_activation_step is None
                        and family_state_entries["predictive_readout"] != 0
                    ):
                        raise RuntimeError(
                            "staged P2 created predictive optimizer state before a nonzero "
                            "predictive gradient"
                        )
                    optimizer.zero_grad(set_to_none=True)
                    torch.cuda.synchronize(device)
                    elapsed_seconds = time.perf_counter() - started
                    peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
                    if peak_reserved_bytes > maximum_peak_reserved_bytes:
                        raise RuntimeError("staged P2 exceeded the CUDA reservation budget")
                except BaseException as error:
                    post_update_local_error = error
                _distributed_phase_error(
                    error=post_update_local_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-post-update-local",
                    rank=rank,
                    dist_module=dist,
                )
                if (
                    family_state_entries is None
                    or elapsed_seconds is None
                    or peak_reserved_bytes is None
                ):
                    raise RuntimeError("staged P2 post-update audit vanished")
                optimizer_family_audits = {
                    "vlm_host": (
                        _audit_optimizer_family_state(
                            policy,
                            optimizer,
                            torch,
                            family="P2 shared VLM host",
                            fragment=".qwenvl.",
                            expected_adamw_step=args.steps + p2_optimizer_step + 1,
                            dist_module=dist,
                        )
                        if p2_optimizer_step in {0, args.staged_p2_steps - 1}
                        else None
                    ),
                    "predictive_readout": _audit_optimizer_family_state(
                        policy,
                        optimizer,
                        torch,
                        family="P2 predictive readout",
                        fragment="predictive_readouts.dino_video",
                        expected_adamw_step=p2_optimizer_step + 1,
                        dist_module=dist,
                    ),
                }

                step_report = None
                report_error: BaseException | None = None
                try:
                    step_report = {
                        "p2_global_step": p2_optimizer_step + 1,
                        "optimizer_global_step": args.steps + p2_optimizer_step + 1,
                        "rank": rank,
                        "source_episode_index": int(segment.episode_index),
                        "plan_optimizer_step": int(selection.plan_optimizer_step),
                        "source_global_index": int(record.source_global_index),
                        "target_global_index": int(record.target_global_index),
                        "carried_binding_count": carried_binding_count,
                        "task_key": segment.task_key,
                        "instruction": segment.instruction,
                        "prefix_sample_key": prefix_sample_key,
                        "source_sample_key": source_sample_key,
                        "future_sample_keys": list(future_sample_keys),
                        "objective_total": _float(p2_result.objective.objective.total),
                        "family_terms": {
                            name: _float(value)
                            for name, value in p2_result.objective.objective.family_terms.items()
                        },
                        "valid_counts": valid_counts,
                        "predictive_valid_targets": current_predictive_valid,
                        "gradient_metrics": gradient_metrics,
                        "optimizer_family_state_entries": family_state_entries,
                        "optimizer_family_audits": optimizer_family_audits,
                        "predictive_valid_seen": p2_predictive_valid_seen,
                        "predictive_activation_step": predictive_activation_step,
                        "prefix_row_bindings": [
                            [list(binding) for binding in bindings]
                            for bindings in prefix_row_bindings
                        ],
                        "row_bindings": [
                            [list(binding) for binding in bindings]
                            for bindings in p2_result.objective.row_bindings_by_batch
                        ],
                        "elapsed_seconds": elapsed_seconds,
                        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                        "peak_cuda_reserved_bytes": peak_reserved_bytes,
                    }
                except BaseException as error:
                    report_error = error
                _distributed_phase_error(
                    error=report_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-report",
                    rank=rank,
                    dist_module=dist,
                )
                if step_report is None:
                    raise RuntimeError("staged P2 step report vanished")
                p2_rank_steps.append(step_report)
                p2_log_error: BaseException | None = None
                try:
                    if rank == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "task_independent_staged_p2_step",
                                    **{
                                        key: step_report[key]
                                        for key in (
                                            "p2_global_step",
                                            "optimizer_global_step",
                                            "objective_total",
                                            "family_terms",
                                            "gradient_metrics",
                                            "elapsed_seconds",
                                            "peak_cuda_reserved_bytes",
                                        )
                                    },
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                except BaseException as error:
                    p2_log_error = error
                _distributed_phase_error(
                    error=p2_log_error,
                    phase=f"staged-p2-step-{p2_optimizer_step}-log",
                    rank=rank,
                    dist_module=dist,
                )
                del p2_result, source_batch, previous_state

            p2_causal_rank_steps: list[dict[str, Any]] = []
            if args.p2_causal_probe_steps > 0:
                if (
                    causal_predictive_cache is None
                    or p2_causal_report is None
                    or len(p2_causal_selected_records)
                    != args.p2_causal_probe_steps * P1_WORLD_SIZE
                    or p2_causal_schedule_sha256 is None
                    or p2_causal_schedule_file_sha256 is None
                ):
                    raise RuntimeError("staged P2 causal contracts vanished")
                validation_segment_indices = {
                    item.segment_index for item in representation_split.validation_segments
                }
                heldout_segment_indices = {
                    item.segment_index for item in representation_split.heldout_segments
                }
                for causal_step in range(args.p2_causal_probe_steps):
                    selection_index = causal_step * P1_WORLD_SIZE + rank
                    record, segment, episode, transition_index = (
                        p2_causal_selected_records[selection_index]
                    )

                    def causal_replay(
                        sample_key: str,
                        *,
                        replay_offset: int,
                        step: int = causal_step,
                    ) -> PlannedNativeCALVINReplayBatch:
                        return build_native_calvin_replay_batch(
                            dataset,
                            sample_key=sample_key,
                            lane_id=rank,
                            episode_instance_id=(
                                f"task-independent-p2-causal/step-{step}/rank-{rank}"
                            ),
                            optimizer_step=args.steps + args.staged_p2_steps,
                            replay_seed=(
                                args.seed
                                + 3_000_017
                                + step * P1_WORLD_SIZE * 19
                                + rank * 1013
                                + replay_offset
                            ),
                            device=device,
                            dtype=torch.bfloat16,
                        )

                    older_planned = None
                    prefix_planned = None
                    source_planned = None
                    future_planned = None
                    older_batch = None
                    prefix_batch = None
                    source_batch = None
                    sample_keys: tuple[str, ...] = ()
                    prepare_error: BaseException | None = None
                    try:
                        sample_keys = tuple(
                            episode.sample_keys[
                                transition_index - 2 : transition_index + args.p2_causal_horizon + 1
                            ]
                        )
                        if len(sample_keys) != 3 + args.p2_causal_horizon:
                            raise RuntimeError("P2 causal replay lacks its complete local window")
                        if dataset.source_global_index_by_key(sample_keys[2]) != int(
                            record.source_global_index
                        ) or tuple(
                            dataset.source_global_index_by_key(key)
                            for key in sample_keys[3:]
                        ) != tuple(
                            range(
                                int(record.source_global_index) + 1,
                                int(record.source_global_index) + args.p2_causal_horizon + 1,
                            )
                        ):
                            raise RuntimeError("P2 causal source/future resolution changed")
                        older_planned = causal_replay(sample_keys[0], replay_offset=0)
                        prefix_planned = causal_replay(sample_keys[1], replay_offset=1)
                        source_planned = causal_replay(sample_keys[2], replay_offset=2)
                        future_planned = tuple(
                            causal_replay(key, replay_offset=3 + offset)
                            for offset, key in enumerate(sample_keys[3:])
                        )
                        older_batch = collate_replay(older_planned)
                        prefix_batch = collate_replay(prefix_planned)
                        source_batch = collate_replay(source_planned)
                    except BaseException as error:
                        prepare_error = error
                    _distributed_phase_error(
                        error=prepare_error,
                        phase=f"p2-causal-step-{causal_step}-prepare",
                        rank=rank,
                        dist_module=dist,
                    )
                    if (
                        older_planned is None
                        or prefix_planned is None
                        or source_planned is None
                        or future_planned is None
                        or older_batch is None
                        or prefix_batch is None
                        or source_batch is None
                    ):
                        raise RuntimeError("P2 causal preparation vanished")

                    previous_state = None
                    previous_state_valid = torch.ones(1, dtype=torch.bool, device=device)
                    wrong_time_state = None
                    neutral_state = None
                    prefix_row_bindings = None
                    prefix_error: BaseException | None = None
                    try:
                        older_result = run_task_independent_calvin_current_frame_diagnostic(
                            policy,
                            batch=older_batch,
                            physical_sidecar=physical_sidecar,
                            objective_config=objective_config,
                            patch_size=patch_size,
                            merge_size=merge_size,
                            minimum_supervised_fraction=args.minimum_supervised_fraction,
                            capacity_seeds=older_planned.augmentation_seeds,
                        )
                        prefix_result = run_task_independent_calvin_current_frame_diagnostic(
                            policy,
                            batch=prefix_batch,
                            physical_sidecar=physical_sidecar,
                            objective_config=objective_config,
                            patch_size=patch_size,
                            merge_size=merge_size,
                            minimum_supervised_fraction=args.minimum_supervised_fraction,
                            capacity_seeds=prefix_planned.augmentation_seeds,
                        )
                        older_state = older_result.context.posterior_state
                        prefix_state = prefix_result.context.posterior_state
                        if older_state is None or prefix_state is None:
                            raise RuntimeError("P2 causal prefix omitted posterior state")
                        wrong_time_state = NativePosteriorState(older_state.rows.detach())
                        previous_state = NativePosteriorState(prefix_state.rows.detach())
                        # Displace the posterior by exactly as much as the wrong-time
                        # substitution does, using noise that carries no temporal or
                        # episode information. Only an arm's excess over this control
                        # is evidence that the model used what the arm destroyed.
                        neutral_generator = torch.Generator(device=device)
                        neutral_generator.manual_seed(args.seed + 7_000_019 + causal_step)
                        neutral_state = NativePosteriorState(
                            matched_noise_rows(
                                previous_state.rows,
                                target_distance=state_manipulation_strength(
                                    previous_state.rows, wrong_time_state.rows
                                ),
                                generator=neutral_generator,
                            )
                        )
                        prefix_row_bindings = physical_frame_row_bindings(
                            prefix_result.targets,
                            prefix_result.objective.frame_losses[0].assignment,
                            capacity=args.capacity,
                        )
                        if not prefix_row_bindings[0]:
                            raise RuntimeError("P2 causal prefix established no physical row gauge")
                    except BaseException as error:
                        prefix_error = error
                    _distributed_phase_error(
                        error=prefix_error,
                        phase=f"p2-causal-step-{causal_step}-prefix",
                        rank=rank,
                        dist_module=dist,
                    )
                    if (
                        previous_state is None
                        or wrong_time_state is None
                        or neutral_state is None
                        or prefix_row_bindings is None
                    ):
                        raise RuntimeError("P2 causal prefix evidence vanished")
                    del older_result, prefix_result, older_batch, prefix_batch

                    future_controls = tuple(
                        value.training.controls for value in future_planned
                    )
                    peer_state = NativePosteriorState(
                        _distributed_ring_exchange_tensor(
                            previous_state.rows,
                            dist_module=dist,
                            torch_module=torch,
                        )
                    )
                    peer_state_valid = _distributed_ring_exchange_tensor(
                        previous_state_valid,
                        dist_module=dist,
                        torch_module=torch,
                    )

                    def peer_control(control: ExecutedControlBatch) -> ExecutedControlBatch:
                        return ExecutedControlBatch(
                            values=_distributed_ring_exchange_tensor(
                                control.values,
                                dist_module=dist,
                                torch_module=torch,
                            ),
                            field_valid=_distributed_ring_exchange_tensor(
                                control.field_valid,
                                dist_module=dist,
                                torch_module=torch,
                            ),
                            token_valid=_distributed_ring_exchange_tensor(
                                control.token_valid,
                                dist_module=dist,
                                torch_module=torch,
                            ),
                            delta_time=_distributed_ring_exchange_tensor(
                                control.delta_time,
                                dist_module=dist,
                                torch_module=torch,
                            ),
                            reset=_distributed_ring_exchange_tensor(
                                control.reset,
                                dist_module=dist,
                                torch_module=torch,
                            ),
                            acknowledged=_distributed_ring_exchange_tensor(
                                control.acknowledged,
                                dist_module=dist,
                                torch_module=torch,
                            ),
                        )

                    peer_future_controls = tuple(
                        peer_control(control) for control in future_controls
                    )
                    request = make_native_future_request(
                        source=PredictionSource.PRIOR,
                        batch_size=1,
                        horizon=args.p2_causal_horizon,
                        valid=previous_state_valid,
                        device=device,
                        dtype=previous_state.rows.dtype,
                        route_id=causal_predictive_cache.contract.route_id,
                        address_width=graph.config.prediction_address_width,
                    )
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    causal_started = time.perf_counter()
                    causal_step_report = None
                    causal_error: BaseException | None = None
                    try:
                        counterfactual = run_native_future_counterfactual_forwards(
                            policy,
                            stepper=prior_stepper,
                            model_inputs=source_batch.model_inputs,
                            controls=source_batch.controls,
                            future_controls=future_controls,
                            previous_state=previous_state,
                            previous_state_valid=previous_state_valid,
                            request=request,
                            modalities=source_batch.modalities,
                            wrong_batch_state=peer_state,
                            wrong_batch_state_valid=peer_state_valid,
                            wrong_time_state=wrong_time_state,
                            wrong_time_state_valid=previous_state_valid,
                            neutral_state=neutral_state,
                            neutral_state_valid=previous_state_valid,
                            wrong_future_controls=peer_future_controls,
                        )
                        relation = counterfactual.factual_context.relation_output
                        if not isinstance(relation, PhysicalRelationOutput):
                            raise TypeError("P2 causal factual route omitted physical relations")
                        target_bundle = build_task_independent_calvin_targets(
                            requests_by_time=(source_batch.structural_target_requests,),
                            model_inputs_by_time=(source_batch.model_inputs,),
                            relations=(relation,),
                            physical_sidecar=physical_sidecar,
                            capacity=args.capacity,
                            patch_size=patch_size,
                            merge_size=merge_size,
                            minimum_supervised_fraction=args.minimum_supervised_fraction,
                            capacity_seeds=prefix_planned.augmentation_seeds,
                        )[0]
                        assignment = match_physical_sequence_entities(
                            (physical_frame_predictions_from_relation(relation),),
                            (target_bundle.targets,),
                            identity_keys_by_batch=target_bundle.identity_keys_by_batch,
                            prior_bindings_by_batch=prefix_row_bindings,
                        )
                        target = causal_predictive_cache.target_for(
                            source_global_indices=(int(record.source_global_index),),
                            source_rgb_sha256=(
                                source_batch.structural_target_requests[
                                    0
                                ].source_sensor_hash_by_field["rgb_static"],
                            ),
                            track_identity_keys=target_bundle.identity_keys_by_batch,
                            request=request,
                            device=device,
                        )
                        causal_row_binding_valid = assignment.binding_start_phase <= 1
                        diagnostics = predictive_future_counterfactual_diagnostics(
                            counterfactual,
                            target=target,
                            assignment=assignment,
                            row_binding_valid=causal_row_binding_valid,
                        )
                        # The loss above is the training objective, which multiplies
                        # each deviation by the target's importance and then divides by
                        # the count of valid entries. Publishing the mass lets the gate
                        # recover the importance-weighted mean instead of a quantity
                        # that scales with object area.
                        causal_evidence_mass = predictive_evidence_mass(
                            request=request,
                            target=target,
                            assignment=assignment,
                            row_binding_valid=causal_row_binding_valid,
                        )
                        causal_manipulation = {
                            "wrong_time_source": state_manipulation_strength(
                                previous_state.rows, wrong_time_state.rows
                            ),
                            "batch_shift_source": state_manipulation_strength(
                                previous_state.rows, peer_state.rows
                            ),
                            "matched_noise_source": state_manipulation_strength(
                                previous_state.rows, neutral_state.rows
                            ),
                            "zero_source": state_manipulation_strength(
                                previous_state.rows,
                                torch.zeros_like(previous_state.rows),
                            ),
                        }
                        torch.cuda.synchronize(device)
                        causal_step_report = {
                            "causal_global_step": causal_step + 1,
                            "rank": rank,
                            "partition": (
                                "validation"
                                if int(segment.index) in validation_segment_indices
                                else "heldout"
                                if int(segment.index) in heldout_segment_indices
                                else "causal_audit"
                            ),
                            "source_episode_index": int(segment.episode_index),
                            "source_global_index": int(record.source_global_index),
                            "target_global_index": int(record.target_global_index),
                            "task_key": segment.task_key,
                            "instruction": segment.instruction,
                            "sample_keys": list(sample_keys),
                            "exact_correction_then_prior_route": True,
                            "task_scorer_present": False,
                            "task_prompt_entered_shared_host": True,
                            "loss_side_label_or_mask_entered_probe_forward": False,
                            "diagnostics": diagnostics.as_dict(),
                            "evidence_mass": causal_evidence_mass,
                            "source_manipulation": causal_manipulation,
                            "horizon": int(args.p2_causal_horizon),
                            "elapsed_seconds": time.perf_counter() - causal_started,
                            "peak_cuda_allocated_bytes": int(
                                torch.cuda.max_memory_allocated(device)
                            ),
                            "peak_cuda_reserved_bytes": int(
                                torch.cuda.max_memory_reserved(device)
                            ),
                        }
                    except BaseException as error:
                        causal_error = error
                    _distributed_phase_error(
                        error=causal_error,
                        phase=f"p2-causal-step-{causal_step}-score",
                        rank=rank,
                        dist_module=dist,
                    )
                    if causal_step_report is None:
                        raise RuntimeError("P2 causal step report vanished")
                    p2_causal_rank_steps.append(causal_step_report)
                    journal_error: BaseException | None = None
                    try:
                        journal_root = args.run_dir / "p2_causal_rank_journal" / f"rank_{rank}"
                        journal_root.mkdir(parents=True, exist_ok=True)
                        write_text_durable_exclusive(
                            journal_root / f"step_{causal_step + 1:06d}.json",
                            json.dumps(causal_step_report, indent=2, sort_keys=True) + "\n",
                        )
                    except BaseException as error:
                        journal_error = error
                    _distributed_phase_error(
                        error=journal_error,
                        phase=f"p2-causal-step-{causal_step}-journal",
                        rank=rank,
                        dist_module=dist,
                    )
                    if rank == 0:
                        print(
                            json.dumps(
                                {
                                    "event": "task_independent_p2_causal_step",
                                    "causal_global_step": causal_step + 1,
                                    "source_global_index": int(record.source_global_index),
                                    "diagnostics": causal_step_report["diagnostics"],
                                    "elapsed_seconds": causal_step_report["elapsed_seconds"],
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                    del (
                        counterfactual,
                        diagnostics,
                        peer_state,
                        previous_state,
                        source_batch,
                        target,
                        target_bundle,
                        wrong_time_state,
                    )

            gathered_p2: list[Any] = [None for _ in range(P1_WORLD_SIZE)]
            dist.all_gather_object(
                gathered_p2,
                {
                    "rank": rank,
                    "p1_boundary_optimizer_state": p1_boundary_optimizer_state,
                    "p1_boundary_optimizer_audit": p1_boundary_optimizer_audit,
                    "p1_boundary_host_optimizer_audit": (p1_boundary_host_optimizer_audit),
                    "steps": p2_rank_steps,
                    "causal_steps": p2_causal_rank_steps,
                },
            )
            publication_error = [None]
            staged_report: dict[str, Any] | None = None
            if rank == 0:
                try:
                    all_sources: list[int] = []
                    predictive_gradient_steps = 0
                    predictive_gradient_global_steps = 0
                    predictive_valid_targets = 0
                    predictive_valid_global_steps = 0
                    if {int(item["rank"]) for item in gathered_p2} != set(range(P1_WORLD_SIZE)):
                        raise RuntimeError("staged P2 report rank set changed")
                    for rank_report in gathered_p2:
                        if len(rank_report["steps"]) != args.staged_p2_steps:
                            raise RuntimeError("staged P2 report has the wrong per-rank length")
                        for step_index, item in enumerate(rank_report["steps"]):
                            if item["p2_global_step"] != step_index + 1:
                                raise RuntimeError("staged P2 report has non-contiguous steps")
                            if item["source_episode_index"] not in set(
                                p2_representation_split.training_source_episode_indices
                            ):
                                raise RuntimeError("staged P2 report escaped the training split")
                            if item["gradient_metrics"]["action_only_elements"] != 0:
                                raise RuntimeError("staged P2 report contains action gradients")
                            if item["gradient_metrics"]["predictive_readout_norm"] > 0:
                                predictive_gradient_steps += 1
                            predictive_valid_targets += sum(
                                count
                                for name, count in item["valid_counts"].items()
                                if name.startswith("rollout/")
                            )
                            all_sources.append(item["source_global_index"])
                    for step_index in range(args.staged_p2_steps):
                        global_step_items = tuple(
                            rank_report["steps"][step_index] for rank_report in gathered_p2
                        )
                        if (
                            len({int(item["plan_optimizer_step"]) for item in global_step_items})
                            != 1
                        ):
                            raise RuntimeError(
                                "staged P2 combined different frozen-plan optimizer steps"
                            )
                        if {int(item["rank"]) for item in global_step_items} != set(
                            range(P1_WORLD_SIZE)
                        ):
                            raise RuntimeError("staged P2 global step rank set changed")
                        global_valid = sum(
                            int(item["predictive_valid_targets"]) for item in global_step_items
                        )
                        if global_valid <= 0:
                            raise RuntimeError(
                                "staged P2 global step has no predictive supervision"
                            )
                        predictive_valid_global_steps += 1
                        if any(
                            float(item["gradient_metrics"]["predictive_readout_norm"]) <= 0
                            for item in global_step_items
                        ):
                            raise RuntimeError(
                                "staged P2 global step has no predictive-readout gradient"
                            )
                        predictive_gradient_global_steps += 1
                    if (
                        predictive_gradient_global_steps != args.staged_p2_steps
                        or predictive_valid_global_steps != args.staged_p2_steps
                    ):
                        raise RuntimeError("staged P2 predictive step coverage is incomplete")
                    causal_evidence = (
                        None
                        if args.p2_causal_probe_steps == 0
                        else _summarize_p2_causal_evidence(
                            gathered_p2,
                            expected_global_steps=args.p2_causal_probe_steps,
                        )
                    )
                    p1_boundary_path = args.run_dir / f"p1_boundary_steps_{args.steps}.json"
                    staged_report = {
                        "schema": STAGED_P2_REPORT_SCHEMA,
                        "status": (
                            "PASS" if causal_evidence is None else causal_evidence["status"]
                        ),
                        "mechanical_status": "PASS",
                        "scientific_status": (
                            "NOT_RUN" if causal_evidence is None else causal_evidence["status"]
                        ),
                        "architecture_identity": TASK_INDEPENDENT_ENTITY_POSTERIOR,
                        "task_scorer_present": False,
                        "task_used_by_entity_objective": False,
                        "action_suffix_executed": False,
                        "same_process_model_optimizer": True,
                        "p1_steps": args.steps,
                        "p2_steps": args.staged_p2_steps,
                        "p1_boundary_report": str(p1_boundary_path.resolve()),
                        "p1_boundary_report_sha256": _sha256(p1_boundary_path),
                        "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                        "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "p1_stream_plan_sha256": plan.plan_sha256,
                        "p1_representation_split_sha256": (representation_split.artifact_sha256),
                        "p2_stream_plan_sha256": p2_plan.plan_sha256,
                        "p2_representation_split_sha256": (p2_representation_split.artifact_sha256),
                        "p2_schedule_sha256": p2_schedule_sha256,
                        "p2_schedule_file_sha256": p2_schedule_file_sha256,
                        "p2_causal_schedule_sha256": p2_causal_schedule_sha256,
                        "p2_causal_schedule_file_sha256": (
                            p2_causal_schedule_file_sha256
                        ),
                        "representation_parameter_scope": (
                            representation_parameter_scope.as_dict()
                        ),
                        "predictive_cache_manifest_sha256": predictive_cache.manifest_sha256,
                        "predictive_build_report_sha256": (
                            args.p2_predictive_cache_build_report_sha256
                        ),
                        "causal_predictive_cache_manifest_sha256": (
                            None
                            if causal_predictive_cache is None
                            else causal_predictive_cache.manifest_sha256
                        ),
                        "causal_predictive_build_report_sha256": (
                            args.p2_causal_probe_cache_build_report_sha256
                        ),
                        "p2_causal_replay_closure_file_sha256": (
                            args.p2_causal_replay_closure_sha256
                        ),
                        "p2_causal_replay_closure_artifact_sha256": (
                            None
                            if p2_causal_replay_closure is None
                            else p2_causal_replay_closure["artifact_sha256"]
                        ),
                        "p2_causal_replay_required_file_count": (
                            0
                            if p2_causal_replay_closure is None
                            else len(p2_causal_replay_closure["required_paths"])
                        ),
                        "graph": asdict(graph_config),
                        "p1_objective": asdict(objective_config),
                        "p2_objective": asdict(staged_objective_config),
                        "prefix_gradient_mode": "current_weight_no_grad_one_frame",
                        "source_gradient_mode": ("posterior_correction_plus_shared_host_prior"),
                        "fsdp2_placement": args.fsdp2_placement,
                        "maximum_peak_reserved_bytes": max(
                            [
                                item["peak_cuda_reserved_bytes"]
                                for rank_report in gathered_p2
                                for item in rank_report["steps"]
                            ]
                            + [
                                item["peak_cuda_reserved_bytes"]
                                for rank_report in gathered_p2
                                for item in rank_report["causal_steps"]
                            ]
                        ),
                        "p2_causal_probe_global_steps": args.p2_causal_probe_steps,
                        "p2_causal_horizon": int(args.p2_causal_horizon),
                        "p2_horizon": int(args.p2_horizon),
                        "p2_causal_probe_samples": (
                            0 if causal_evidence is None else causal_evidence["sample_count"]
                        ),
                        "causal_evidence": causal_evidence,
                        "predictive_gradient_rank_steps": predictive_gradient_steps,
                        "predictive_gradient_global_steps": (predictive_gradient_global_steps),
                        "predictive_valid_global_steps": predictive_valid_global_steps,
                        "predictive_supervision_global_step_coverage": (
                            predictive_valid_global_steps / args.staged_p2_steps
                        ),
                        "predictive_valid_targets": predictive_valid_targets,
                        "p2_source_occurrences": len(all_sources),
                        "p2_unique_source_count": len(set(all_sources)),
                        "rank_reports": gathered_p2,
                    }
                    write_text_durable_exclusive(
                        args.output,
                        json.dumps(staged_report, indent=2, sort_keys=True) + "\n",
                    )
                except BaseException as error:
                    publication_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(publication_error, src=0)
            if publication_error[0] is not None:
                raise RuntimeError(f"staged P2 report publication failed: {publication_error[0]}")
            dist.barrier()
            if rank == 0:
                if staged_report is None:
                    raise RuntimeError("rank zero lost the staged P2 report")
                print(json.dumps(staged_report, indent=2, sort_keys=True), flush=True)
    finally:
        if run_lease is not None:
            run_lease.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

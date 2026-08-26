#!/usr/bin/env python3
# pyright: reportMissingImports=false, reportMissingModuleSource=false
# ruff: noqa: E402, I001
"""Train the complete task-independent PICF graph on two or four A100 40GB GPUs.

The declared experiment is always a 30k stream. ``--stop-after-step`` is an
early-stop boundary for smoke tests or scientific termination; it does not
change data order, cache coverage, or the optimizer schedule.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from collections.abc import Mapping
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
for _path in (_REPOSITORY_ROOT, _REPOSITORY_ROOT / "src"):
    _text = str(_path)
    while _text in sys.path:
        sys.path.remove(_text)
    sys.path.insert(0, _text)

from tools.cuda_allocator_bootstrap import (
    CUDA_ALLOCATOR_MODES,
    bootstrap_cuda_allocator,
)

_BOOTSTRAPPED_CUDA_ALLOCATOR = (
    bootstrap_cuda_allocator(sys.argv[1:]) if __name__ == "__main__" else None
)

from picf_next.artifact_io import write_text_durable_exclusive
from picf_next.data.calvin_physical_visual_acceptance import (
    load_calvin_physical_visual_acceptance,
)
from picf_next.data.lingbot_calvin_projection import LINGBOT_CALVIN_CAMERA_SLOTS
from picf_next.eval.calvin_task_relevance import calvin_task_physical_relevance
from picf_next.lingbot_native.capacity import (
    require_checkpoint_write_capacity,
    require_persistent_run_root,
)
from picf_next.lingbot_native.fsdp2_placement import (
    FSDP2_BACKWARD_PREFETCH_DEFAULT,
    FSDP2_BACKWARD_PREFETCH_MODES,
    FSDP2_CPU_OFFLOAD,
    FSDP2_FACTUAL_GRADIENT_CPU,
    FSDP2_FACTUAL_GRADIENT_GPU,
    FSDP2_FACTUAL_GRADIENT_STORAGE_MODES,
    FSDP2_PLACEMENTS,
    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
    FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
    configure_fsdp2_backward_prefetch,
    fsdp2_parameter_layout_manifest,
    fsdp2_present_gradient_manifest,
    merge_fsdp2_factual_gradients_from_cpu,
    spill_fsdp2_factual_gradients_to_cpu,
    validate_fsdp2_placement,
)
from picf_next.lingbot_native.host import (
    EXACT_NATIVE_MODALITY_BRIDGE,
    LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE,
    NATIVE_MODALITY_BRIDGES,
    NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR,
    NATIVE_VIDEOMT_QUERY_POSTERIOR,
)
from picf_next.lingbot_native.modalities import (
    CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
    NO_RELATION_TARGET,
)
from picf_next.lingbot_native.official_config import official_lingbot_data_config
from picf_next.lingbot_native.trainable_scope import (
    TRAINABLE_SCOPES,
    TRAINABLE_SCOPE_FROZEN_VISION_HOST,
    TRAINABLE_SCOPE_FULL_HOST,
    lingbot_trainable_scope_receipt as _trainable_scope_receipt,
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
        FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
        MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH,
        PATCH_RELATIVE_PATH,
        SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
        SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH,
        SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH,
        SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
        VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
        validate_prepared_native_source,
        validate_prepared_native_source_with_muon_collective_hotfix,
        validate_prepared_native_source_with_selective_class_cpu_offload,
        validate_prepared_native_source_with_selective_frozen_vision_offload,
        validate_prepared_native_source_with_selective_trainable_vision_offload,
        validate_prepared_native_source_with_trainable_vision_and_selective_class_offload,
        validate_prepared_native_source_with_trainable_vision_and_vlm_selective_class_offload,
        verify_muon_collective_hotfix,
        verify_native_patch,
        verify_selective_class_cpu_offload,
        verify_selective_frozen_vision_offload,
        verify_selective_trainable_vision_offload,
        verify_selective_trainable_vision_with_selective_class_cpu_offload,
        verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload,
    )
    from tools.bootstrap_lingbot_vla2_flare import (
        validate_prepared_flare_overlay,
    )
    from tools.lingbot_vla2_runtime_helpers import (
        LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        build_lingbot_official_optimizer,
        build_lingbot_base_family_identity,
        clip_lingbot_distributed_l2_grad_norm_,
        configure_picf_optimizer_learning_rates,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        require_lingbot_released_action_sampling_steps,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from tools.run_lingbot_vla2_native_full import (
        _cache_producer_patch_sha256,
        _collated_to_device,
        _distributed_pre_backward_failures,
        _distributed_raise_if_local_probe_error,
        _optimizer_attempt,
        _posterior_bank_digest,
        _validate_lingbot_calvin_projection_batch,
        _validate_lingbot_projection_processor,
        _validate_training_supervision_policy,
        _validate_action_fsdp2_topology,
        _validate_vlm_fsdp2_topology,
        load_current_grid_build_report,
        load_predictive_build_report,
    )
    from tools.run_lingbot_vla2_native_g0 import (
        _capture_rank_rng,
        _checkpoint_boundary,
        _fsync_tree,
        _local_import_modules,
        _resolve_local_module,
        _restore_rank_rng,
        _validate_fsdp2_parameter_storage,
        _validate_optimizer_state,
        _write_text_durable,
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
        FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH,
        LINGBOT_NATIVE_SOURCE_COMMIT,
        MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
        MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH,
        PATCH_RELATIVE_PATH,
        SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
        SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH,
        SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH,
        SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
        VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH,
        validate_prepared_native_source,
        validate_prepared_native_source_with_muon_collective_hotfix,
        validate_prepared_native_source_with_selective_class_cpu_offload,
        validate_prepared_native_source_with_selective_frozen_vision_offload,
        validate_prepared_native_source_with_selective_trainable_vision_offload,
        validate_prepared_native_source_with_trainable_vision_and_selective_class_offload,
        validate_prepared_native_source_with_trainable_vision_and_vlm_selective_class_offload,
        verify_muon_collective_hotfix,
        verify_native_patch,
        verify_selective_class_cpu_offload,
        verify_selective_frozen_vision_offload,
        verify_selective_trainable_vision_offload,
        verify_selective_trainable_vision_with_selective_class_cpu_offload,
        verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload,
    )
    from bootstrap_lingbot_vla2_flare import (  # type: ignore[no-redef]
        validate_prepared_flare_overlay,
    )
    from lingbot_vla2_runtime_helpers import (  # type: ignore[no-redef]
        LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
        _merge_qwen_config,
        _resolve_training_config,
        _sha256,
        build_lingbot_official_optimizer,
        build_lingbot_base_family_identity,
        clip_lingbot_distributed_l2_grad_norm_,
        configure_picf_optimizer_learning_rates,
        load_lingbot_training_config,
        register_native_fsdp_forward_methods,
        require_lingbot_exact_resume_contract,
        require_lingbot_released_action_sampling_steps,
        resolve_lingbot_optimizer_contract,
        strip_targetless_alignment_teacher_heads,
    )
    from run_lingbot_vla2_native_full import (  # type: ignore[no-redef]
        _cache_producer_patch_sha256,
        _collated_to_device,
        _distributed_pre_backward_failures,
        _distributed_raise_if_local_probe_error,
        _optimizer_attempt,
        _posterior_bank_digest,
        _validate_lingbot_calvin_projection_batch,
        _validate_lingbot_projection_processor,
        _validate_training_supervision_policy,
        _validate_action_fsdp2_topology,
        _validate_vlm_fsdp2_topology,
        load_current_grid_build_report,
        load_predictive_build_report,
    )
    from run_lingbot_vla2_native_g0 import (  # type: ignore[no-redef]
        _capture_rank_rng,
        _checkpoint_boundary,
        _fsync_tree,
        _local_import_modules,
        _resolve_local_module,
        _restore_rank_rng,
        _validate_fsdp2_parameter_storage,
        _validate_optimizer_state,
        _write_text_durable,
    )


SUPPORTED_WORLD_SIZES = (2, 4)
V3_DISTRIBUTED_PRIOR_SCHEDULE = "rank_max_left_pad_masked_identity_v1"


def _runtime_world_size(environment: Mapping[str, str] | None = None) -> int:
    """Resolve the single-host FSDP world size before building frozen contracts."""

    values = os.environ if environment is None else environment
    raw = values.get("WORLD_SIZE", "2")
    try:
        world_size = int(raw)
    except ValueError as error:
        raise RuntimeError("WORLD_SIZE must be a canonical integer") from error
    if raw != str(world_size) or world_size not in SUPPORTED_WORLD_SIZES:
        raise RuntimeError("task-independent training supports exactly 2 or 4 ranks")
    return world_size


def _distributed_prior_host_step_schedule(
    local_counts: tuple[int, ...],
    *,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> tuple[int, ...]:
    """Resolve one rank-symmetric FSDP host-call count for every prior phase."""

    if not local_counts or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in local_counts
    ):
        raise ValueError("local prior host counts must be a non-empty tuple of positive integers")
    counts = torch_module.tensor(local_counts, dtype=torch_module.int64, device=device)
    dist.all_reduce(counts, op=dist.ReduceOp.MAX)
    resolved = tuple(int(value) for value in counts.tolist())
    if len(resolved) != len(local_counts) or any(
        aligned < local for local, aligned in zip(local_counts, resolved, strict=True)
    ):
        raise RuntimeError("distributed prior host schedule shortened a local control chain")
    return resolved


WORLD_SIZE = _runtime_world_size()
TOTAL_STEPS = 30_000
METRICS_EVERY = 100
VISUAL_EVERY = 250
CHECKPOINT_EVERY = 2_000
ACCEPTANCE_MODES = (
    "none",
    "action-adoption-presence",
    "action-adoption-interventions",
    "posterior-adoption-route",
    "posterior-adoption-dose",
    "dcp-uninterrupted",
    "dcp-restored",
)
PRODUCTION_LOCAL_BPTT_PROBABILITY = 0.0
COMPARISON_ID = "lingbot-vla2-native-picf-full"
RUNNER_SCHEMA = "picf-next.task-independent-full-runner/v19"
CHECKPOINT_SCHEMA = "picf-next.task-independent-full-checkpoint/v8"
METRIC_WINDOW_SCHEMA = "picf-next.task-independent-full-metrics/v1"
PROGRESS_SCHEMA = "picf-next.task-independent-full-progress/v1"
POSTERIOR_ARCHITECTURES = ("legacy_v1", "layerwise_v2", "two_pass_v3")
ACTION_BACKENDS = ("lingbot_released", "wla_complete")
WLA_COMPLETE_ACTION_BACKEND = ACTION_BACKENDS[1]
WLA_HOST_EVIDENCE_ARMS = ("picf_full", "wla_lbot_masked")
WLA_SOURCE_ACTION_HORIZON = 8


@dataclass(frozen=True)
class ActionBackendRuntimeBufferSnapshot:
    action_backend: str
    values: tuple[tuple[str, Any, Any], ...]


def _snapshot_action_backend_runtime_buffers(
    policy: Any,
    *,
    action_backend: str,
) -> ActionBackendRuntimeBufferSnapshot:
    if action_backend not in ACTION_BACKENDS:
        raise ValueError("unknown action backend")
    suffixes = ("avg_topk_sigmoid_score", "tokens_per_expert")
    values = tuple(
        (name, buffer, buffer.detach().to(device="cpu", copy=True))
        for name, buffer in policy.named_buffers()
        if name.endswith(suffixes)
    )
    has_token_counts = any(
        name.endswith("tokens_per_expert") for name, _buffer, _saved in values
    )
    if action_backend == WLA_COMPLETE_ACTION_BACKEND:
        if values:
            raise RuntimeError(
                "complete WLA unexpectedly retained released LingBot action-MoE buffers"
            )
    elif not values or not has_token_counts:
        raise RuntimeError("causal diagnostic found no official action-MoE buffers")
    return ActionBackendRuntimeBufferSnapshot(
        action_backend=action_backend,
        values=values,
    )


def _restore_action_backend_runtime_buffers(
    snapshot: ActionBackendRuntimeBufferSnapshot,
    *,
    action_backend: str,
    torch_module: Any,
) -> None:
    if action_backend not in ACTION_BACKENDS:
        raise ValueError("unknown action backend")
    if snapshot.action_backend != action_backend:
        raise RuntimeError("action runtime-buffer snapshot belongs to a different backend")
    with torch_module.no_grad():
        for name, buffer, saved in snapshot.values:
            if buffer.shape != saved.shape or buffer.dtype != saved.dtype:
                raise RuntimeError(
                    f"causal diagnostic runtime buffer changed contract: {name}"
                )
            buffer.copy_(saved)


PICF_ARCHITECTURE_PROFILES = (
    "legacy",
    "adr177_task_addressed_full_modal_v1",
    "adr178_direct_action_posterior_full_modal_v1",
    "adr193_implicit_multimodal_anchor_v1",
    "adr204_full_source_final_only_v1",
    "adr205_released_query_propagation_v1",
    "adr207_native_videomt_query_posterior_v1",
    "adr209_native_videomt_query_control_t16_v1",
    "adr209_native_videomt_flare_v1",
    "adr221_native_videomt_wsa_full_modal_v1",
    "adr222_native_videomt_world_token_wsa_v1",
    "adr225_pretrained_native_object_memory_v1",
)
ADR177_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[1]
ADR178_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[2]
ADR193_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[3]
ADR204_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[4]
ADR205_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[5]
ADR207_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[6]
ADR209_CONTROL_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[7]
ADR209_FLARE_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[8]
ADR221_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[9]
ADR222_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[10]
ADR225_ARCHITECTURE_PROFILE = PICF_ARCHITECTURE_PROFILES[11]
ADR177_TASK_QUERY_COUNT = 4
ADR177_RELATION_SUPERVISION_LAYERS = (8, 17, 26)
ADR177_PICF_LEARNING_RATE_MULTIPLIER = 2.0
ADR177_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER = 0.5
ADR178_RELATION_SUPERVISION_LAYERS = ADR177_RELATION_SUPERVISION_LAYERS
ADR178_PICF_LEARNING_RATE_MULTIPLIER = ADR177_PICF_LEARNING_RATE_MULTIPLIER
ADR178_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER = ADR177_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER
ADR178_NATIVE_ATTENTION_WEIGHT = 0.001
ADR178_REGISTERED_LAYER_OFFSETS = (-4, -1)
ADR178_REGISTERED_ACTION_HEAD_INDICES = (0, 1)
ADR178_ACCEPTANCE_STOP_STEP = 250
ADR193_ANCHOR_CHECK_STEPS = (50, 100)
ADR207_ANCHOR_CHECK_STEPS = (50, 100, 200)
ADR207_REQUIRED_FUTURE_SOURCE_FRAMES = 4
ADR221_WSA_COMMIT = "bfee742c585d5ee85722e658978111934c926ca3"
ADR221_WSA_TEACHER_SOURCE_SHA256 = (
    "04910cb5025dcd3dfae81e20553393428d7dcf136ecb49a4f8ef180fbefcd765"
)
ADR221_WSA_ADAPTED_CHECKPOINT_SHA256 = (
    "29d789d9a97459e33ed95aa85fb6e0ec0661879789db090bb8cabc1edf6a9130"
)
ADR221_DA3_SOURCE_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
ADR221_DA3_MODEL_SHA256 = (
    "739905c423cf0d6ccaf9e61a8401d82ba1ac32d7f4d3ee6dca8f92b377633f64"
)
ADR221_DA3_CONFIG_SHA256 = (
    "744dcaf53859490ed92fc6cb98d68d3daf624b8c54533aaf604bdb53f06321f5"
)
ADR221_WSA_EDGE_INTERVENTION_SCHEMA = (
    "picf-next.adr221-wsa-future-to-action-intervention/v1"
)
FLARE_REQUIRED_FUTURE_SOURCE_FRAMES = 16
CAUSAL_ABLATION_SCHEMA = "picf-next.adr146-recurrence-causal-ablation/v1"
CAUSAL_ABLATION_MODES = (
    "none",
    "current_frame_branch",
    "zero_state",
    "recurrent_state",
)
CAUSAL_BRANCH_STEP = 200
CAUSAL_ARM_STOP_STEP = 300
CAUSAL_DIAGNOSTIC_STEPS = (250, 300)
CAUSAL_ENTITY_WEIGHT = 0.08
LAYERWISE_CAUSAL_DIAGNOSTIC_SCHEMA = "picf-next.layerwise-posterior-causal-diagnostic/v2"
LAYERWISE_CAUSAL_DIAGNOSTIC_STEPS = (250, 500, 1_000, 2_000)
TWO_PASS_FILTER_DIAGNOSTIC_SCHEMA = "picf-next.adr149-two-pass-filter-diagnostic/v1"
TWO_PASS_FILTER_DIAGNOSTIC_STEPS = (250, 500, 1_000, 2_000)
TWO_PASS_ACTION_EVALUATION_SCHEMA = "picf-next.adr149-cold-action-snapshot/v2"
ADR207_ACTION_EVALUATION_SCHEMA = "picf-next.adr207-cold-native-query-action-snapshot/v1"
ADR207_ANCHOR_EVALUATION_SCHEMA = "picf-next.adr207-heldout-native-query-anchor/v1"
ADR210_CAUSAL_WARM_ACTION_EVALUATION_SCHEMA = (
    "picf-next.adr210-causal-warm-native-query-action-snapshot/v1"
)
ADR210_CAUSAL_WARM_ANCHOR_EVALUATION_SCHEMA = (
    "picf-next.adr210-causal-warm-native-query-anchor/v1"
)
ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS = 4
ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS = (100,)
ADR207_MODALITY_INTERVENTION_SCHEMA = (
    "picf-next.adr207-heldout-native-query-modality-intervention/v1"
)
ADR207_MODALITY_INTERVENTION_STEPS = (200, 2_000)
ADR207_MODALITY_INTERVENTIONS = (
    "value_zero",
    "metadata_zero",
    "value_permutation",
    "joint_permutation",
)
ADR207_ACTION_STABILITY_MAX_ABS_DRIFT = 1e-6
ADR207_ACTION_EFFECT_MIN_ABS_DRIFT = 1e-5
TWO_PASS_ACTION_EVALUATION_STEPS = (0, 20, 100, 200, 500, 1_000, 1_500, 2_000)
POSTERIOR_ADOPTION_STOP_STEP = 500
POSTERIOR_ADOPTION_DOSE_STOP_STEP = 200
POSTERIOR_ADOPTION_DOSE_SOURCE_MASK_PROBABILITY = 1.0
POSTERIOR_ADOPTION_DOSE_ACTION_EVALUATION_STEPS = (0, 20, 100, 200)
ADR148_ENTITY_WEIGHT = 0.08
ADR148_PREDICTIVE_WEIGHT = 0.004
ADR148_SOURCE_MASK_PROBABILITY = 0.10
ADR149_ENTITY_WEIGHT = 0.08
ADR149_FILTER_WEIGHT = 0.004
ADR149_OMITTED_STATIC_PROBABILITY = 0.10
DENSE_EVIDENCE_MODES = ("none", "calvin_full_v1")
CALVIN_FULL_DENSE_MODALITIES = ("anytouch", "sonata", "vjepa")
VIDEOMT_STAGE_PQ_DISABLED = "disabled"
VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5 = "frozen-released-eval-causal-c5"
VIDEOMT_STAGE_PQM_FROZEN_RELEASED_EVAL_C5 = "frozen-released-eval-causal-c5-pqm"
VIDEOMT_STAGE_PQM_FROZEN_ADAPTED_EVAL_C5 = "frozen-adapted-eval-causal-c5-pqm"
VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5 = "frozen-adapted-eval-causal-c5-pqmr"
VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5 = "frozen-adapted-eval-causal-c5-pqrf"
VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5 = (
    "trainable-adapted-native-query-causal-c5"
)
VIDEOMT_SOURCE_UPDATE_JOINT = "joint"
VIDEOMT_SOURCE_UPDATE_FROZEN_COORDINATE_CONTROL = "frozen-coordinate-control"
VIDEOMT_SOURCE_UPDATE_ARMS = (
    VIDEOMT_SOURCE_UPDATE_JOINT,
    VIDEOMT_SOURCE_UPDATE_FROZEN_COORDINATE_CONTROL,
)
VIDEOMT_STAGE_PQ_MODES = (
    VIDEOMT_STAGE_PQ_DISABLED,
    VIDEOMT_STAGE_PQ_FROZEN_RELEASED_EVAL_C5,
    VIDEOMT_STAGE_PQM_FROZEN_RELEASED_EVAL_C5,
    VIDEOMT_STAGE_PQM_FROZEN_ADAPTED_EVAL_C5,
    VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5,
    VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5,
    VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5,
)
VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT = "cuda-resident"
VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS = "cpu-between-forwards"
VIDEOMT_IDLE_PLACEMENTS = (
    VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
    VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS,
)
VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED = "cuda-sharded"
VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD = "cpu-offload"
VIDEOMT_FSDP2_PLACEMENTS = (
    VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
    VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD,
)
LINGBOT_ATTENTION_EAGER = "eager"
LINGBOT_ATTENTION_FLEX_CACHED = "flex_cached"
LINGBOT_ATTENTION_IMPLEMENTATIONS = (
    LINGBOT_ATTENTION_EAGER,
    LINGBOT_ATTENTION_FLEX_CACHED,
)
LINGBOT_COMPILE_DISABLED = "disabled"
LINGBOT_COMPILE_UPSTREAM_DEFAULT = "upstream-default"
LINGBOT_COMPILE_MODES = (
    LINGBOT_COMPILE_DISABLED,
    LINGBOT_COMPILE_UPSTREAM_DEFAULT,
)
VIDEOMT_RELEASED_CHECKPOINT_BYTES = 1_264_120_741
VIDEOMT_RELEASED_CHECKPOINT_SHA256 = (
    "2cfa7a2df68e6f21f29bea3be571b1f63d0f94c90b7b528a67267eb84317c04f"
)
VIDEOMT_DINOV3_CONFIG_BYTES = 742
VIDEOMT_DINOV3_CONFIG_SHA256 = (
    "43304b7ad2d5d2d72d0872ff6092d9aeb722c7e446e755a1ce76a04635760881"
)
OMITTED_STATIC_REMATERIALIZATION_NONE = "none"
OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT = "complete-checkpoint"
OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU = "save-on-cpu"
OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD = "sequential-backward"
OMITTED_STATIC_REMATERIALIZATION_MODES = (
    OMITTED_STATIC_REMATERIALIZATION_NONE,
    OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT,
    OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU,
    OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD,
)
_PREDICTIVE_CACHE_ARGUMENTS = (
    "predictive_cache_root",
    "predictive_cache_build_report",
    "predictive_cache_build_report_sha256",
)
_CURRENT_CACHE_ARGUMENTS = (
    "current_grid_cache_root",
    "current_grid_cache_build_report",
    "current_grid_cache_build_report_sha256",
)
_CURRENT_CACHE_OPTIONAL_ARGUMENTS = ("current_grid_cache_shard_root",)
_FUTURE_LATENT_CACHE_ARGUMENTS = (
    "future_latent_cache_root",
    "future_latent_cache_manifest_sha256",
    "future_latent_cache_build_report",
    "future_latent_cache_build_report_sha256",
)
_AUXILIARY_CACHE_ARGUMENTS = (
    _PREDICTIVE_CACHE_ARGUMENTS + _CURRENT_CACHE_ARGUMENTS + _CURRENT_CACHE_OPTIONAL_ARGUMENTS
)
IMPLEMENTATION_FILES = (
    "adr225/freeze_source.sh",
    "adr225/run_pretrained_object_memory_2gpu.sh",
    "docs/225_PRETRAINED_NATIVE_OBJECT_MEMORY_20260826.md",
    "references/adr225_pretrained_object_memory_sources.json",
    "adr209/build_dense_cache_4gpu_contract.sh",
    "adr209/build_flare_cache_4gpu_contract.sh",
    "adr209/launch_gate_cache_builds_4gpu.sh",
    "adr209/prepare_contracts_4gpu.sh",
    "adr209/run_flare_native_videomt_4gpu.sh",
    "docs/209_LINGBOT_NATIVE_FLARE_ARM_20260823.md",
    "references/patches/lingbot_vla2_flare_generic_target.patch",
    "tools/bootstrap_lingbot_vla2_flare.py",
    "tools/build_flare_future_target_cache.py",
    "adr207/build_dense_cache_2gpu_contract.sh",
    "adr207/freeze_source.sh",
    "adr207/prepare_contracts_2gpu.sh",
    "adr207/prepare_dense_caches_2gpu.sh",
    "adr207/run_matched_lingbot_2gpu.sh",
    "adr207/run_native_videomt_query_posterior_2gpu.sh",
    "adr177/run_upgraded_full_modal.sh",
    "adr178/run_direct_action_posterior_full_modal.sh",
    "adr193/run_implicit_multimodal_anchor_2gpu.sh",
    "adr202/run_adapted_full_pqm_2gpu.sh",
    "adr199/run_full_transplant_stage_pq_2gpu.sh",
    "adr147/restore_four_gpu_runtime.sh",
    "adr152/run_posterior_adoption_route_4gpu.sh",
    "adr149/freeze_inputs.sh",
    "adr149/launch_four_gpu_initial_2k.sh",
    "adr149/launch_four_gpu_30k.sh",
    "adr149/run_full_picf.sh",
    "adr149/run_matched_lbot_4gpu.sh",
    "adr150/freeze_inputs.sh",
    "adr150/launch_four_gpu_30k.sh",
    "adr150/launch_four_gpu_initial_2k.sh",
    "adr150/merge_dense_cache_partitions.sh",
    "adr150/run_dense_cache_partition.sh",
    "adr150/run_full_modal_acceptance_4gpu.sh",
    "adr150/run_full_modal_acceptance_suite_4gpu.sh",
    "adr150/run_full_picf.sh",
    "adr150/run_matched_lbot_4gpu.sh",
    "configs/lingbot/calvin_data.json",
    "configs/lingbot/calvin_robot.yaml",
    "docs/199_FULL_TRANSPLANT_UNIFIED_PICF_ARCHITECTURE_20260821.md",
    "references/full_transplant_sources.json",
    "references/patches/lingbot_vla2_picf_native.patch",
    "src/picf_next/artifact_io.py",
    "src/picf_next/content_addressing.py",
    "src/picf_next/contracts.py",
    "src/picf_next/data/calvin.py",
    "src/picf_next/data/calvin_dense_evidence_audit.py",
    "src/picf_next/data/calvin_dense_evidence_source_audit.py",
    "src/picf_next/data/calvin_frozen_evidence.py",
    "src/picf_next/data/calvin_multimodal.py",
    "src/picf_next/data/calvin_normalization.py",
    "src/picf_next/data/calvin_official_source.py",
    "src/picf_next/data/calvin_physical_supervision_sidecar.py",
    "src/picf_next/data/calvin_physical_supervision_schema.py",
    "src/picf_next/data/calvin_physical_visual_acceptance.py",
    "src/picf_next/data/calvin_target_request.py",
    "src/picf_next/data/calvin_pointcloud.py",
    "src/picf_next/data/calvin_tactile.py",
    "src/picf_next/data/calvin_tactile_calibration.py",
    "src/picf_next/data/dataset_manifest.py",
    "src/picf_next/data/dense_evidence_cache.py",
    "src/picf_next/data/dense_evidence_coverage.py",
    "src/picf_next/data/lingbot_calvin.py",
    "src/picf_next/data/lingbot_calvin_projection.py",
    "src/picf_next/encoders/anytouch2.py",
    "src/picf_next/encoders/spatiallm_sonata.py",
    "src/picf_next/encoders/vjepa21.py",
    "src/picf_next/encoders/vendor/__init__.py",
    "src/picf_next/encoders/vendor/anytouch2/__init__.py",
    "src/picf_next/encoders/vendor/anytouch2/tactile_mae.py",
    "src/picf_next/encoders/vendor/anytouch2/util/__init__.py",
    "src/picf_next/encoders/vendor/anytouch2/util/pos_embed.py",
    "src/picf_next/encoders/vendor/spatiallm_sonata/__init__.py",
    "src/picf_next/encoders/vendor/spatiallm_sonata/model.py",
    "src/picf_next/encoders/vendor/spatiallm_sonata/serialization/__init__.py",
    "src/picf_next/encoders/vendor/spatiallm_sonata/serialization/default.py",
    "src/picf_next/encoders/vendor/spatiallm_sonata/serialization/hilbert.py",
    "src/picf_next/encoders/vendor/spatiallm_sonata/serialization/z_order.py",
    "src/picf_next/encoders/vendor/vjepa21/__init__.py",
    "src/picf_next/encoders/vendor/vjepa21/masks_utils.py",
    "src/picf_next/encoders/vendor/vjepa21/modules.py",
    "src/picf_next/encoders/vendor/vjepa21/patch_embed.py",
    "src/picf_next/encoders/vendor/vjepa21/tensors.py",
    "src/picf_next/encoders/vendor/vjepa21/vision_transformer.py",
    "src/picf_next/full_modal_assets.py",
    "src/picf_next/lingbot_native/calvin.py",
    "src/picf_next/lingbot_native/adr150_lbot_validation.py",
    "src/picf_next/lingbot_native/full_modal_adoption.py",
    "src/picf_next/lingbot_native/calvin_entity_set.py",
    "src/picf_next/lingbot_native/calvin_entity_training.py",
    "src/picf_next/lingbot_native/capacity.py",
    "src/picf_next/lingbot_native/controls.py",
    "src/picf_next/lingbot_native/current_grid_cache.py",
    "src/picf_next/lingbot_native/dense_modalities.py",
    "src/picf_next/lingbot_native/entity_set_objective.py",
    "src/picf_next/lingbot_native/entity_evaluation_plan.py",
    "src/picf_next/lingbot_native/entity_set_evaluation.py",
    "src/picf_next/lingbot_native/entity_training.py",
    "src/picf_next/lingbot_native/fsdp2_placement.py",
    "src/picf_next/lingbot_native/full_training.py",
    "src/picf_next/lingbot_native/graph.py",
    "src/picf_next/lingbot_native/host.py",
    "src/picf_next/lingbot_native/__init__.py",
    "src/picf_next/lingbot_native/modalities.py",
    "src/picf_next/lingbot_native/objective.py",
    "src/picf_next/lingbot_native/official_config.py",
    "src/picf_next/lingbot_native/physical_relations.py",
    "src/picf_next/lingbot_native/physical_sequence.py",
    "src/picf_next/lingbot_native/pretrained_object_memory.py",
    "src/picf_next/lingbot_native/prediction.py",
    "src/picf_next/lingbot_native/predictive_cache.py",
    "src/picf_next/lingbot_native/predictive_objective.py",
    "src/picf_next/lingbot_native/predictive_plan.py",
    "src/picf_next/lingbot_native/predictive_probes.py",
    "src/picf_next/lingbot_native/representation_split.py",
    "src/picf_next/lingbot_native/row_binding.py",
    "src/picf_next/lingbot_native/runtime.py",
    "src/picf_next/lingbot_native/session.py",
    "src/picf_next/lingbot_native/source_mask.py",
    "src/picf_next/lingbot_native/state.py",
    "src/picf_next/lingbot_native/temporal.py",
    "src/picf_next/lingbot_native/torch_dcp_compat.py",
    "src/picf_next/lingbot_native/training.py",
    "src/picf_next/lingbot_native/visual_audit.py",
    "src/picf_next/lingbot_wla_calvin.py",
    "src/picf_next/lingbot_wla_install.py",
    "src/picf_next/lingbot_wla_shared.py",
    "src/picf_next/lingbot_wla_world.py",
    "src/picf_next/wla_upstream.py",
    "src/picf_next/videomt_exact/__init__.py",
    "src/picf_next/videomt_exact/calvin_dataset.py",
    "src/picf_next/videomt_exact/calvin_full_dataset.py",
    "src/picf_next/videomt_exact/calvin_stage_p.py",
    "src/picf_next/videomt_exact/calvin_targets.py",
    "src/picf_next/videomt_exact/checkpoint.py",
    "src/picf_next/videomt_exact/class_agnostic.py",
    "src/picf_next/videomt_exact/distributed_training.py",
    "src/picf_next/videomt_exact/evaluation.py",
    "src/picf_next/videomt_exact/fsdp2.py",
    "src/picf_next/videomt_exact/gradient_diagnostics.py",
    "src/picf_next/videomt_exact/joint_data.py",
    "src/picf_next/videomt_exact/joint_training.py",
    "src/picf_next/videomt_exact/joint_visual.py",
    "src/picf_next/videomt_exact/lingbot_joint.py",
    "src/picf_next/videomt_exact/observations.py",
    "src/picf_next/videomt_exact/optimizer.py",
    "src/picf_next/videomt_exact/paired_training.py",
    "src/picf_next/videomt_exact/partial_supervision.py",
    "src/picf_next/videomt_exact/posterior_refiner.py",
    "src/picf_next/videomt_exact/preprocessing.py",
    "src/picf_next/videomt_exact/runtime.py",
    "src/picf_next/videomt_exact/stage_p.py",
    "src/picf_next/videomt_exact/training.py",
    "src/picf_next/training/control.py",
    "src/picf_next/training/run_lease.py",
    "tools/bootstrap_lingbot_vla2.py",
    "tools/bootstrap_lingbot_vla2_native.py",
    "tools/audit_calvin_dense_evidence_semantics.py",
    "tools/audit_calvin_dense_evidence_source_inputs.py",
    "tools/build_calvin_dense_evidence_coverage_plan.py",
    "tools/build_calvin_frozen_evidence_cache.py",
    "tools/build_lingbot_calvin_current_grid_cache.py",
    "tools/build_lingbot_representation_split.py",
    "tools/compose_adr150_action_adoption_core.py",
    "tools/compose_adr150_full_modal_action_adoption.py",
    "tools/cuda_allocator_bootstrap.py",
    "tools/lingbot_vla2_runtime_helpers.py",
    "tools/merge_dense_evidence_cache_partitions.py",
    "tools/probe_calvin_full_modal_encoders.py",
    "tools/republish_calvin_frozen_evidence_cache.py",
    "tools/run_lingbot_vla2_native_full.py",
    "tools/run_lingbot_vla2_native_g0.py",
    "tools/run_lingbot_vla2_official_lbot.py",
    "tools/run_lingbot_vla2_task_independent_full.py",
)


@dataclass(frozen=True, slots=True)
class ProductionCadence:
    total_steps: int = TOTAL_STEPS
    metrics_every: int = METRICS_EVERY
    visual_every: int = VISUAL_EVERY
    checkpoint_every: int = CHECKPOINT_EVERY

    def __post_init__(self) -> None:
        if (
            self.total_steps,
            self.metrics_every,
            self.visual_every,
            self.checkpoint_every,
        ) != (TOTAL_STEPS, METRICS_EVERY, VISUAL_EVERY, CHECKPOINT_EVERY):
            raise ValueError("the production cadence is frozen at 30k/100/250/2000")

    def metrics_due(self, step: int) -> bool:
        return step > 0 and step % self.metrics_every == 0

    def visual_due(self, step: int) -> bool:
        return step > 0 and step % self.visual_every == 0

    def checkpoint_due(self, step: int) -> bool:
        return step > 0 and step % self.checkpoint_every == 0


def _scientific_terminal_checkpoint_due(*, stop_after_step: int, global_step: int) -> bool:
    """Persist terminal state only at the frozen 2k production cadence."""

    for name, value in (("stop_after_step", stop_after_step), ("global_step", global_step)):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if not 0 < stop_after_step <= TOTAL_STEPS:
        raise ValueError("scientific stop boundary lies outside the declared 30k stream")
    if not 0 <= global_step <= TOTAL_STEPS:
        raise ValueError("global step lies outside the declared 30k stream")
    return (
        global_step == stop_after_step
        and global_step % CHECKPOINT_EVERY == 0
    )


def _evaluation_visual_sample_keys(
    items: tuple[Any, ...],
    *,
    partitions: tuple[str, ...],
    per_partition: int,
) -> tuple[str, ...]:
    """Choose fixed visual examples from distinct tasks before any replicate."""

    if isinstance(per_partition, bool) or not isinstance(per_partition, int):
        raise TypeError("evaluation visual count must be an integer")
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


def _summarize_native_videomt_anchor_partition(
    samples: list[dict[str, Any]],
    *,
    partition: str,
) -> dict[str, object]:
    """Aggregate object-weighted source-query metrics without task selection."""

    selected = [sample for sample in samples if sample.get("partition") == partition]
    if not selected:
        raise ValueError(f"native VidEoMT anchor partition {partition!r} is empty")
    soft = [float(value) for sample in selected for value in sample["soft_ious"]]
    binary = [float(value) for sample in selected for value in sample["binary_ious"]]
    foreground = [
        float(value)
        for sample in selected
        for value in sample["foreground_probabilities"]
    ]
    if not soft or len(soft) != len(binary) or len(soft) != len(foreground):
        raise RuntimeError("native VidEoMT anchor sample metrics have incompatible axes")
    ranked_samples: list[dict[int, Mapping[str, object]]] = []
    expected_top_ks = (10, 25, 50, 100, 200)
    for sample in selected:
        proposals = sample.get("ranked_proposals")
        if not isinstance(proposals, list):
            raise TypeError("native VidEoMT ranked proposals must retain source list ABI")
        by_top_k: dict[int, Mapping[str, object]] = {}
        for proposal in proposals:
            if not isinstance(proposal, Mapping):
                raise TypeError("native VidEoMT ranked proposal must be one mapping")
            top_k = proposal.get("top_k")
            if isinstance(top_k, bool) or not isinstance(top_k, int):
                raise TypeError("native VidEoMT ranked proposal top_k must be an integer")
            if top_k in by_top_k:
                raise RuntimeError("native VidEoMT ranked proposal top_k was duplicated")
            by_top_k[top_k] = proposal
        ranked_samples.append(by_top_k)
    top_ks = tuple(ranked_samples[0])
    if top_ks != expected_top_ks or any(
        tuple(sample) != top_ks for sample in ranked_samples
    ):
        raise RuntimeError("native VidEoMT ranked-proposal inventory changed")
    ranked: dict[str, dict[str, float]] = {}
    for top_k in top_ks:
        values = [sample[top_k] for sample in ranked_samples]
        ranked[str(top_k)] = {
            name: sum(float(value[name]) for value in values) / len(values)
            for name in ("mean_soft_iou", "mean_binary_iou", "recall_at_50")
        }
    return {
        "sample_count": len(selected),
        "object_observation_count": len(soft),
        "mean_soft_iou": sum(soft) / len(soft),
        "mean_binary_iou": sum(binary) / len(binary),
        "recall_at_50": sum(value >= 0.5 for value in binary) / len(binary),
        "mean_foreground_probability": sum(foreground) / len(foreground),
        "ranked_proposals": ranked,
    }


def _validate_production_temporal_estimator(local_bptt_probability: float) -> None:
    if local_bptt_probability != PRODUCTION_LOCAL_BPTT_PROBABILITY:
        raise ValueError(
            "production full training requires local_bptt_probability=0: "
            "deployment-aged detached posterior lanes and shared-host multi-horizon "
            "prediction provide temporal credit without repeated full-image FSDP2 "
            "host invocations in one backward"
        )


def _environment_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return None if not value else Path(value)


def _canonical_digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    ).hexdigest()


def _module_state_digest(module: object) -> str:
    """Hash a dense module state independent of device placement."""

    import torch

    state_dict = getattr(module, "state_dict", None)
    if not callable(state_dict):
        raise TypeError("module-state digest requires a torch module")
    digest = hashlib.sha256()
    for name, value in sorted(state_dict().items()):
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise TypeError("module-state digest supports dense tensors only")
        tensor = value.detach().cpu().contiguous()
        metadata = json.dumps(
            {"dtype": str(tensor.dtype), "name": name, "shape": list(tensor.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(metadata).to_bytes(8, "little"))
        digest.update(metadata)
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _disabled_auxiliary_digest(name: str) -> str:
    """Return one stable provenance value for an intentionally absent cache."""

    reason_by_name = {
        "current_grid": "disabled_until_layerwise_causal_acceptance",
        "dense_evidence": "disabled_by_explicit_dense_evidence_mode_none",
        "future_latent_cache": "disabled_by_absent_flare_cache_abi",
        "predictive": "disabled_until_layerwise_causal_acceptance",
    }
    if name not in reason_by_name:
        raise ValueError("unknown auxiliary cache family")
    return _canonical_digest(
        {
            "architecture": "task_independent_entity_posterior_v2",
            "cache": name,
            "reason": reason_by_name[name],
        }
    )


def _predictive_assets_required(args: argparse.Namespace) -> bool:
    """Return whether controlled future-rollout assets are required."""

    architecture = getattr(args, "posterior_architecture", "legacy_v1")
    if architecture not in POSTERIOR_ARCHITECTURES:
        raise ValueError("unknown posterior architecture")
    return architecture == "legacy_v1"


def _layerwise_predictive_correction_active(args: argparse.Namespace) -> bool:
    architecture = getattr(args, "posterior_architecture", "legacy_v1")
    if architecture not in POSTERIOR_ARCHITECTURES:
        raise ValueError("unknown posterior architecture")
    return architecture == "layerwise_v2" and args.predictive_weight > 0


def _two_pass_filter_active(args: argparse.Namespace) -> bool:
    architecture = getattr(args, "posterior_architecture", "legacy_v1")
    if architecture not in POSTERIOR_ARCHITECTURES:
        raise ValueError("unknown posterior architecture")
    return architecture == "two_pass_v3"


def _current_correction_assets_required(args: argparse.Namespace) -> bool:
    if _adr207_native_videomt_query_posterior_active(args):
        return False
    return (
        _predictive_assets_required(args)
        or _layerwise_predictive_correction_active(args)
        or _two_pass_filter_active(args)
    )


def _objective_profile(args: argparse.Namespace) -> str:
    architecture = getattr(args, "posterior_architecture", "legacy_v1")
    if architecture == "two_pass_v3":
        if _adr207_native_videomt_query_posterior_active(args):
            if _adr225_pretrained_object_memory_active(args):
                return "adr225_pretrained_native_object_memory_joint_action"
            if _adr209_flare_active(args):
                return "adr209_complete_source_native_query_flare_joint_action"
            if _adr209_native_videomt_active(args):
                return "adr209_complete_source_native_query_t16_joint_action_control"
            return "adr207_complete_source_native_query_joint_action"
        if _adr177_task_addressed_full_modal_active(args):
            return "adr177_task_addressed_full_modal_two_pass_filter"
        if getattr(args, "dense_evidence_mode", "none") == "calvin_full_v1":
            return "adr150_full_modal_action_visible_two_pass_filter"
        return "adr149_action_visible_two_pass_filter"
    if architecture == "layerwise_v2":
        return (
            "adr148_prior_current_correction"
            if _layerwise_predictive_correction_active(args)
            else "adr147_recurrent_core"
        )
    return "legacy_full_or_causal"


def _object_action_information_contract(args: argparse.Namespace) -> str:
    if _adr225_pretrained_object_memory_active(args):
        return "videomt_mask_pooled_native_qwen3_object_memory_to_official_action"
    if _adr207_native_videomt_query_posterior_active(args):
        return "native_source_query_i_as_shared_host_posterior_row_i_to_official_action"
    if _adr177_task_addressed_full_modal_active(args):
        return "task_query_QK_to_object_value_read_then_additive_action_residual"
    if _adr178_direct_action_posterior_active(args):
        return "native_action_QK_direct_to_physical_posterior_rows"
    if _adr193_implicit_multimodal_anchor_active(args):
        return "implicit_multimodal_physical_rows_no_private_action_selector"
    return "legacy"


def _validate_auxiliary_cache_args(args: argparse.Namespace) -> None:
    """Keep legacy cache replay exact and v2 core free of retired assets."""

    supplied = {
        name for name in _AUXILIARY_CACHE_ARGUMENTS if getattr(args, name, None) is not None
    }
    if _adr207_native_videomt_query_posterior_active(args):
        if supplied:
            raise ValueError(
                "ADR-207 forbids retired predictive/current-grid cache arguments: "
                + ", ".join(sorted(supplied))
            )
        return
    if _predictive_assets_required(args):
        required = _PREDICTIVE_CACHE_ARGUMENTS + _CURRENT_CACHE_ARGUMENTS
        missing = tuple(name for name in required if name not in supplied)
        if missing:
            raise ValueError(
                "legacy_v1 requires every predictive/current-grid cache argument: "
                + ", ".join(missing)
            )
        return
    if _layerwise_predictive_correction_active(args):
        forbidden = tuple(name for name in _PREDICTIVE_CACHE_ARGUMENTS if name in supplied)
        if forbidden:
            raise ValueError(
                "ADR-148 layerwise correction forbids future predictive-cache arguments: "
                + ", ".join(forbidden)
            )
        missing = tuple(name for name in _CURRENT_CACHE_ARGUMENTS if name not in supplied)
        if missing:
            raise ValueError(
                "ADR-148 layerwise correction requires every current-cache argument: "
                + ", ".join(missing)
            )
        return
    if _two_pass_filter_active(args):
        forbidden = tuple(name for name in _PREDICTIVE_CACHE_ARGUMENTS if name in supplied)
        if forbidden:
            raise ValueError(
                "ADR-149 two-pass filter forbids legacy future-cache arguments: "
                + ", ".join(forbidden)
            )
        missing = tuple(name for name in _CURRENT_CACHE_ARGUMENTS if name not in supplied)
        if missing:
            raise ValueError(
                "ADR-149 two-pass filter requires every current-cache argument: "
                + ", ".join(missing)
            )
        return
    if supplied:
        raise ValueError(
            "layerwise_v2 core forbids predictive/current-grid cache arguments: "
            + ", ".join(sorted(supplied))
        )


def _future_latent_alignment_active(args: argparse.Namespace) -> bool:
    supplied = tuple(
        name for name in _FUTURE_LATENT_CACHE_ARGUMENTS if getattr(args, name, None) is not None
    )
    if supplied and len(supplied) != len(_FUTURE_LATENT_CACHE_ARGUMENTS):
        missing = tuple(name for name in _FUTURE_LATENT_CACHE_ARGUMENTS if name not in supplied)
        raise ValueError(
            "FLARE requires its cache root, manifest digest, build report, and report digest: "
            + ", ".join(missing)
        )
    return bool(supplied)


def _wla_complete_active(args: argparse.Namespace) -> bool:
    backend = getattr(args, "action_backend", "lingbot_released")
    if backend not in ACTION_BACKENDS:
        raise ValueError("unknown action backend")
    return backend == WLA_COMPLETE_ACTION_BACKEND


def _validate_wla_complete_args(args: argparse.Namespace) -> None:
    supplied = tuple(
        name
        for name in ("wla_source_root", "wla_pretrained_root")
        if getattr(args, name, None) is not None
    )
    evidence_arm = getattr(args, "wla_host_evidence_arm", WLA_HOST_EVIDENCE_ARMS[0])
    if evidence_arm not in WLA_HOST_EVIDENCE_ARMS:
        raise ValueError("unknown WLA host-evidence arm")
    if not _wla_complete_active(args):
        if supplied:
            raise ValueError("WLA source assets require --action-backend wla_complete")
        if evidence_arm != WLA_HOST_EVIDENCE_ARMS[0]:
            raise ValueError("WLA host-evidence interventions require wla_complete")
        if getattr(args, "videomt_shared_query_gradient_diagnostic", False):
            raise ValueError("shared-query gradient diagnosis requires complete WLA")
        return
    if len(supplied) != 2:
        raise ValueError("complete WLA requires both source and pretrained roots")
    if args.posterior_architecture != "two_pass_v3":
        raise ValueError("complete WLA PICF training requires the two-pass posterior")
    if args.picf_architecture_profile != ADR207_ARCHITECTURE_PROFILE:
        raise ValueError("complete WLA is pinned to the full native VidEoMT query profile")
    if args.dense_evidence_mode != "calvin_full_v1":
        raise ValueError("complete WLA requires AnyTouch, Sonata and V-JEPA dense evidence")
    if _future_latent_alignment_active(args) or _adr221_full_source_wsa_active(args):
        raise ValueError("complete WLA world training replaces legacy FLARE and WSA objectives")
    if _adr178_direct_action_posterior_active(args):
        raise ValueError("complete WLA forbids the released-action attention callback")
    if args.trainable_scope != TRAINABLE_SCOPE_FULL_HOST:
        raise ValueError("upstream WLA trains the complete language/vision host")
    if args.lingbot_compile_mode != LINGBOT_COMPILE_DISABLED:
        raise ValueError(
            "complete WLA preserves the upstream uncompiled BF16 execution contract"
        )
    if (
        args.learning_rate != 5.0e-5
        or args.picf_learning_rate_multiplier != 1.0
        or args.modality_bridge_learning_rate_multiplier != 1.0
        or args.max_grad_norm != 1.0
    ):
        raise ValueError("complete WLA requires its exact 5e-5/1.0 optimizer surface")
    if getattr(args, "videomt_shared_query_gradient_diagnostic", False) and not (
        _videomt_native_trainable_active(args)
    ):
        raise ValueError("shared-query gradient diagnosis requires trainable native VidEoMT")


def _validate_future_latent_cache_args(args: argparse.Namespace) -> None:
    flare_supplied = _future_latent_alignment_active(args)
    if _adr209_native_videomt_active(args) and not flare_supplied:
        raise ValueError("ADR-209 profiles require the complete future-latent cache ABI")
    if not flare_supplied:
        return
    if not _adr209_native_videomt_active(args):
        raise ValueError("FLARE assets require a frozen ADR-209 profile")
    if getattr(args, "dense_evidence_mode", "none") != "calvin_full_v1":
        raise ValueError("ADR-209 FLARE requires the complete AnyTouch/Sonata/V-JEPA evidence set")
    if getattr(args, "lingbot_compile_mode", LINGBOT_COMPILE_DISABLED) != (
        LINGBOT_COMPILE_UPSTREAM_DEFAULT
    ):
        raise ValueError(
            "ADR-209 requires LingBot's released FSDP2-then-whole-model compile path"
        )


def _required_future_source_frames(args: argparse.Namespace) -> int:
    flare_active = _future_latent_alignment_active(args)
    adr207_active = _adr207_native_videomt_query_posterior_active(args)
    intrinsic = (
        WLA_SOURCE_ACTION_HORIZON
        if _wla_complete_active(args)
        else
        FLARE_REQUIRED_FUTURE_SOURCE_FRAMES
        if flare_active or _adr209_native_videomt_active(args)
        else ADR207_REQUIRED_FUTURE_SOURCE_FRAMES
        if adr207_active
        else 0
    )
    requested = getattr(args, "minimum_future_source_frames", None)
    if requested is None:
        return intrinsic
    if not adr207_active:
        raise ValueError("an explicit future-frame domain is restricted to ADR-207/209")
    if requested < intrinsic:
        raise ValueError("the explicit future-frame domain is shorter than the active model")
    if flare_active and requested != FLARE_REQUIRED_FUTURE_SOURCE_FRAMES:
        raise ValueError("FLARE fixes the future-frame domain to exactly t+16")
    return requested


def _implementation_paths(root: Path) -> tuple[Path, ...]:
    """Resolve every fixed artifact and transitive local Python dependency."""

    root = root.resolve()
    resolved: set[Path] = set()
    pending: list[Path] = []
    for relative in IMPLEMENTATION_FILES:
        candidate = root / relative
        if candidate.is_symlink() or not candidate.is_file():
            raise FileNotFoundError(candidate)
        path = candidate.resolve()
        path.relative_to(root)
        path_relative = path.relative_to(root)
        if path.suffix == ".py" and path_relative.parts[0] in {"src", "tools"}:
            pending.append(path)
        else:
            resolved.add(path)

    while pending:
        path = pending.pop()
        if path in resolved:
            continue
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
        resolved.add(path)
        for module in _local_import_modules(root, path):
            for imported in _resolve_local_module(root, module):
                if imported.is_symlink() or not imported.is_file():
                    raise FileNotFoundError(imported)
                imported = imported.resolve()
                imported.relative_to(root)
                if imported not in resolved:
                    pending.append(imported)
    return tuple(sorted(resolved))


def _implementation_digest(root: Path) -> str:
    root = root.resolve()
    values = {str(path.relative_to(root)): _sha256(path) for path in _implementation_paths(root)}
    return _canonical_digest(values)


def _positive_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and greater than zero")
    return parsed


def _nonnegative_finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be finite and non-negative")
    return parsed


def _parse_nonnegative_layer_set(value: str) -> tuple[int, ...]:
    try:
        layers = tuple(int(part) for part in value.split(",") if part)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "relation supervision layers must be comma-separated integers"
        ) from error
    if (
        any(layer < 0 for layer in layers)
        or tuple(sorted(set(layers))) != layers
        or (value and not layers)
    ):
        raise argparse.ArgumentTypeError(
            "relation supervision layers must be sorted unique non-negative integers"
        )
    return layers


def _adr177_task_addressed_full_modal_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR177_ARCHITECTURE_PROFILE


def _adr178_direct_action_posterior_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR178_ARCHITECTURE_PROFILE


def _adr193_implicit_multimodal_anchor_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile in {
        ADR193_ARCHITECTURE_PROFILE,
        ADR204_ARCHITECTURE_PROFILE,
        ADR205_ARCHITECTURE_PROFILE,
    }


def _adr204_full_source_final_only_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR204_ARCHITECTURE_PROFILE


def _adr205_released_query_propagation_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR205_ARCHITECTURE_PROFILE


def _adr207_native_videomt_query_posterior_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile in {
        ADR207_ARCHITECTURE_PROFILE,
        ADR209_CONTROL_ARCHITECTURE_PROFILE,
        ADR209_FLARE_ARCHITECTURE_PROFILE,
        ADR221_ARCHITECTURE_PROFILE,
        ADR222_ARCHITECTURE_PROFILE,
        ADR225_ARCHITECTURE_PROFILE,
    }


def _adr225_pretrained_object_memory_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR225_ARCHITECTURE_PROFILE


def _pretrained_object_memory_step_report(
    context: Any,
    *,
    capacity: int,
    torch_module: Any,
) -> dict[str, Any]:
    """Fail closed and summarize the action-visible ADR-225 memory path."""

    support = getattr(context, "object_memory_support_mass", None)
    valid = getattr(context, "object_memory_query_valid", None)
    generation = getattr(context, "object_memory_capture_generation", None)
    if not isinstance(support, torch_module.Tensor) or not isinstance(
        valid,
        torch_module.Tensor,
    ):
        raise RuntimeError("ADR-225 action context omitted object-memory diagnostics")
    if (
        support.ndim != 2
        or support.shape != valid.shape
        or support.shape[1] != capacity
        or not support.is_floating_point()
        or valid.dtype != torch_module.bool
        or support.device != valid.device
    ):
        raise ValueError("ADR-225 object-memory diagnostic tensors changed ABI")
    if (
        not torch_module.isfinite(support).all()
        or (support < 0).any()
        or (support > 1).any()
    ):
        raise ValueError("ADR-225 object-memory support mass is not a probability")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation <= 0:
        raise ValueError("ADR-225 object-memory capture generation is invalid")

    valid_support = support.masked_select(valid).detach().float()
    valid_count = int(valid.sum().item())
    return {
        "schema": "picf-next.pretrained-object-memory-step/v1",
        "active": True,
        "capture_generation": generation,
        "query_capacity": capacity,
        "valid_query_count": valid_count,
        "valid_query_count_by_batch": [
            int(value) for value in valid.sum(dim=1).detach().cpu().tolist()
        ],
        "mean_valid_support_mass": (
            0.0 if valid_count == 0 else float(valid_support.mean().item())
        ),
        "maximum_valid_support_mass": (
            0.0 if valid_count == 0 else float(valid_support.max().item())
        ),
        "zero_support_valid_query_count": int(
            ((support <= 0) & valid).sum().detach().cpu().item()
        ),
    }


def _adr209_native_videomt_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile in {
        ADR209_CONTROL_ARCHITECTURE_PROFILE,
        ADR209_FLARE_ARCHITECTURE_PROFILE,
    }


def _adr209_flare_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR209_FLARE_ARCHITECTURE_PROFILE


def _adr221_full_source_wsa_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile in {
        ADR221_ARCHITECTURE_PROFILE,
        ADR222_ARCHITECTURE_PROFILE,
    }


def _adr222_world_token_adoption_active(args: argparse.Namespace) -> bool:
    profile = getattr(args, "picf_architecture_profile", "legacy")
    if profile not in PICF_ARCHITECTURE_PROFILES:
        raise ValueError("unknown PICF architecture profile")
    return profile == ADR222_ARCHITECTURE_PROFILE


def _adr221_wsa_edge_diagnostic_active(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "adr221_wsa_edge_diagnostic", False))


def _registered_anchor_evaluation_steps(args: argparse.Namespace) -> tuple[int, ...]:
    if _adr193_implicit_multimodal_anchor_active(args):
        return ADR193_ANCHOR_CHECK_STEPS
    if _adr207_native_videomt_query_posterior_active(args):
        return ADR207_ANCHOR_CHECK_STEPS
    return ()


def _anchor_evaluation_due(*, args: argparse.Namespace, global_step: int) -> bool:
    return global_step in _registered_anchor_evaluation_steps(args)


def _picf_optimizer_learning_rate_stratification_active(
    args: argparse.Namespace,
) -> bool:
    """Apply the frozen PICF optimizer groups to every upgraded profile."""

    return (
        _adr177_task_addressed_full_modal_active(args)
        or _adr178_direct_action_posterior_active(args)
        or _adr193_implicit_multimodal_anchor_active(args)
    )


def _adr178_registered_layer_indices(layer_count: int) -> tuple[int, ...]:
    """Resolve late shared-host layers from the released model depth."""

    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
        raise ValueError("ADR-178 shared-host layer count must be positive")
    indices = tuple(layer_count + offset for offset in ADR178_REGISTERED_LAYER_OFFSETS)
    if any(not 0 <= index < layer_count for index in indices):
        raise ValueError("ADR-178 registered layer offsets are outside the shared host")
    return indices


def _direct_action_posterior_targets(
    *,
    bindings_by_batch: tuple[Any, ...],
    structural_target_requests: tuple[Any, ...],
    capacity: int,
    dtype: Any,
    device: Any,
    torch_module: Any,
) -> tuple[Any, Any, tuple[dict[str, Any], ...]]:
    """Map exact CALVIN action objects onto Hungarian posterior-row bindings."""

    if len(bindings_by_batch) != len(structural_target_requests):
        raise ValueError("action-posterior bindings differ from the training batch")
    weights = torch_module.zeros(
        (len(bindings_by_batch), capacity),
        dtype=dtype,
        device=device,
    )
    valid = torch_module.zeros(
        len(bindings_by_batch),
        dtype=torch_module.bool,
        device=device,
    )
    audit_rows: list[dict[str, Any]] = []
    for batch_index, (bindings, request) in enumerate(
        zip(bindings_by_batch, structural_target_requests, strict=True)
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
            and selected_rows
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
                "target_identity_keys": list(relevance.action_target_identity_keys),
                "selected_rows": list(selected_rows),
                "target_valid": target_valid,
            }
        )
    return weights, valid, tuple(audit_rows)


def _validate_picf_architecture_profile(args: argparse.Namespace) -> None:
    """Freeze the upgraded information contract as one indivisible recipe."""

    adr177_active = _adr177_task_addressed_full_modal_active(args)
    adr178_active = _adr178_direct_action_posterior_active(args)
    adr193_active = _adr193_implicit_multimodal_anchor_active(args)
    adr204_active = _adr204_full_source_final_only_active(args)
    adr205_active = _adr205_released_query_propagation_active(args)
    adr207_active = _adr207_native_videomt_query_posterior_active(args)
    if adr207_active:
        wla_complete = _wla_complete_active(args)
        adr225_active = _adr225_pretrained_object_memory_active(args)
        expected = {
            "posterior_architecture": "two_pass_v3",
            "dense_evidence_mode": "calvin_full_v1",
            "dense_token_bridge": EXACT_NATIVE_MODALITY_BRIDGE,
            "videomt_stage_pq_mode": VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5,
            "videomt_idle_placement": VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
            "trainable_scope": TRAINABLE_SCOPE_FULL_HOST,
            "capacity": 200,
            "task_query_count": 0,
            "relation_supervision_layers": (),
            "learning_rate": 5e-5 if wla_complete else 1e-4,
            "picf_learning_rate_multiplier": 1.0,
            "modality_bridge_learning_rate_multiplier": 1.0,
            "entity_weight": 0.0,
            "predictive_weight": 0.0,
            "local_bptt_probability": 0.0,
            "overshoot_probability": 0.0,
            "source_mask_probability": 0.0,
            "lingbot_compile_mode": (
                LINGBOT_COMPILE_DISABLED
                if wla_complete or adr225_active
                else LINGBOT_COMPILE_UPSTREAM_DEFAULT
            ),
        }
        if _adr209_native_videomt_active(args):
            expected["future_latent_objective_scale"] = (
                1.0 if _adr209_flare_active(args) else 0.0
            )
        mismatched = tuple(
            name for name, value in expected.items() if getattr(args, name) != value
        )
        if mismatched:
            raise ValueError(
                f"{args.picf_architecture_profile} changed frozen fields: "
                + ", ".join(mismatched)
            )
        return
    if not adr177_active and not adr178_active and not adr193_active:
        if args.task_query_count or args.relation_supervision_layers:
            raise ValueError(
                "task queries and deep object supervision require a frozen PICF profile"
            )
        return
    expected = {
        "posterior_architecture": "two_pass_v3",
        "dense_evidence_mode": "calvin_full_v1",
        "dense_token_bridge": LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE,
        "task_query_count": ADR177_TASK_QUERY_COUNT if adr177_active else 0,
        "relation_supervision_layers": (
            ()
            if adr204_active or adr205_active
            else (
                ADR177_RELATION_SUPERVISION_LAYERS
                if adr177_active
                else ADR178_RELATION_SUPERVISION_LAYERS
            )
        ),
        "learning_rate": 1e-4,
        "picf_learning_rate_multiplier": (
            ADR177_PICF_LEARNING_RATE_MULTIPLIER
            if adr177_active
            else ADR178_PICF_LEARNING_RATE_MULTIPLIER
        ),
        "modality_bridge_learning_rate_multiplier": (
            ADR177_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER
            if adr177_active
            else ADR178_MODALITY_BRIDGE_LEARNING_RATE_MULTIPLIER
        ),
        "local_bptt_probability": 0.0,
        "overshoot_probability": 0.0,
    }
    mismatched = tuple(name for name, value in expected.items() if getattr(args, name) != value)
    if mismatched:
        raise ValueError(
            f"{args.picf_architecture_profile} changed frozen fields: " + ", ".join(mismatched)
        )


def _require_sha256_value(name: str, value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be one lowercase SHA-256")
    return value


def _validate_frozen_stream_args(
    args: argparse.Namespace,
    *,
    world_size: int = WORLD_SIZE,
) -> bool:
    names = (
        "stream_plan",
        "stream_plan_sha256",
        "representation_split",
        "representation_split_sha256",
        "evaluation_plan",
        "evaluation_plan_sha256",
    )
    supplied = tuple(name for name in names if getattr(args, name, None) is not None)
    if supplied and len(supplied) != len(names):
        raise ValueError(
            "frozen stream mode requires plan, split, evaluation plan, and all file digests"
        )
    if world_size == 4 and not supplied:
        raise ValueError("four-rank training requires a frozen source-disjoint stream")
    if world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError("frozen stream validation received an unsupported world size")
    if not supplied:
        return False
    _require_sha256_value("stream plan file SHA-256", args.stream_plan_sha256)
    _require_sha256_value(
        "representation split file SHA-256",
        args.representation_split_sha256,
    )
    _require_sha256_value(
        "evaluation plan file SHA-256",
        args.evaluation_plan_sha256,
    )
    return True


def _validate_dense_evidence_args(args: argparse.Namespace) -> None:
    mode = getattr(args, "dense_evidence_mode", "none")
    bridge = getattr(args, "dense_token_bridge", EXACT_NATIVE_MODALITY_BRIDGE)
    roots = tuple(getattr(args, "dense_evidence_cache_root", ()) or ())
    manifests = tuple(getattr(args, "dense_evidence_cache_manifest_sha256", ()) or ())
    supplement_roots = tuple(getattr(args, "dense_evidence_supplement_cache_root", ()) or ())
    supplement_manifests = tuple(
        getattr(args, "dense_evidence_supplement_cache_manifest_sha256", ()) or ()
    )
    coverage_plan = getattr(args, "dense_evidence_coverage_plan", None)
    coverage_file_sha256 = getattr(
        args,
        "dense_evidence_coverage_plan_sha256",
        None,
    )
    subset_view = bool(getattr(args, "dense_evidence_subset_view", False))
    if mode not in DENSE_EVIDENCE_MODES:
        raise ValueError("unknown dense evidence mode")
    if bridge not in NATIVE_MODALITY_BRIDGES:
        raise ValueError("unknown dense evidence token bridge")
    if mode == "none":
        if (
            roots
            or manifests
            or supplement_roots
            or supplement_manifests
            or coverage_plan is not None
            or coverage_file_sha256 is not None
            or subset_view
        ):
            raise ValueError(
                "disabled dense evidence forbids cache roots, manifests, or coverage plan"
            )
        if bridge != EXACT_NATIVE_MODALITY_BRIDGE:
            raise ValueError("disabled dense evidence requires the exact-token bridge")
        return
    if getattr(args, "posterior_architecture", None) != "two_pass_v3":
        raise ValueError("full dense evidence requires the two_pass_v3 shared-host architecture")
    if len(roots) != len(CALVIN_FULL_DENSE_MODALITIES) or len(manifests) != len(roots):
        raise ValueError("calvin_full_v1 requires exactly three cache roots and manifest hashes")
    if bool(supplement_roots) != bool(supplement_manifests) or (
        supplement_roots
        and (
            len(supplement_roots) != len(CALVIN_FULL_DENSE_MODALITIES)
            or len(supplement_manifests) != len(supplement_roots)
        )
    ):
        raise ValueError(
            "dense evidence supplements require exactly three cache roots and manifest hashes"
        )
    all_roots = roots + supplement_roots
    all_manifests = manifests + supplement_manifests
    if len({Path(root).resolve() for root in all_roots}) != len(all_roots):
        raise ValueError("dense evidence cache roots must be unique")
    if len(set(all_manifests)) != len(all_manifests):
        raise ValueError("dense evidence cache manifest hashes must be unique")
    if coverage_plan is None or coverage_file_sha256 is None:
        raise ValueError("calvin_full_v1 requires one authenticated coverage plan")
    if any(
        getattr(args, name, None) is None
        for name in (
            "stream_plan",
            "stream_plan_sha256",
            "representation_split",
            "representation_split_sha256",
            "evaluation_plan",
            "evaluation_plan_sha256",
        )
    ):
        raise ValueError("calvin_full_v1 requires the frozen stream and evaluation contract")
    _require_sha256_value(
        "dense evidence coverage plan file SHA-256",
        coverage_file_sha256,
    )
    for digest in all_manifests:
        _require_sha256_value("dense evidence cache manifest SHA-256", digest)


def _videomt_stage_pq_active(args: argparse.Namespace) -> bool:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    return mode != VIDEOMT_STAGE_PQ_DISABLED


def _videomt_stage_pqm_active(args: argparse.Namespace) -> bool:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    return mode in {
        VIDEOMT_STAGE_PQM_FROZEN_RELEASED_EVAL_C5,
        VIDEOMT_STAGE_PQM_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5,
    }


def _videomt_stage_pqmr_active(args: argparse.Namespace) -> bool:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    return mode in {
        VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5,
    }


def _videomt_stage_pqrf_active(args: argparse.Namespace) -> bool:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    return mode == VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5


def _videomt_stage_pq_adapted_active(args: argparse.Namespace) -> bool:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    return mode in {
        VIDEOMT_STAGE_PQM_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_STAGE_PQMR_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_STAGE_PQRF_FROZEN_ADAPTED_EVAL_C5,
        VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5,
    }


def _videomt_native_trainable_active(args: argparse.Namespace) -> bool:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    return mode == VIDEOMT_NATIVE_TRAINABLE_ADAPTED_CAUSAL_C5


def _validate_videomt_stage_pq_args(args: argparse.Namespace) -> None:
    mode = getattr(args, "videomt_stage_pq_mode", VIDEOMT_STAGE_PQ_DISABLED)
    idle_placement = getattr(
        args,
        "videomt_idle_placement",
        VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
    )
    fsdp2_placement = getattr(
        args,
        "videomt_fsdp2_placement",
        VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
    )
    checkpoint = getattr(args, "videomt_checkpoint", None)
    bundle = getattr(args, "videomt_dinov3_bundle", None)
    adapted_checkpoint = getattr(args, "videomt_adapted_checkpoint", None)
    adapted_checkpoint_sha256 = getattr(args, "videomt_adapted_checkpoint_sha256", None)
    source_update_arm = getattr(
        args,
        "videomt_source_update_arm",
        VIDEOMT_SOURCE_UPDATE_JOINT,
    )
    if source_update_arm not in VIDEOMT_SOURCE_UPDATE_ARMS:
        raise ValueError("unknown VidEoMT source-update arm")
    if mode not in VIDEOMT_STAGE_PQ_MODES:
        raise ValueError("unknown VidEoMT Stage-PQ mode")
    if idle_placement not in VIDEOMT_IDLE_PLACEMENTS:
        raise ValueError("unknown VidEoMT idle placement")
    if fsdp2_placement not in VIDEOMT_FSDP2_PLACEMENTS:
        raise ValueError("unknown VidEoMT FSDP2 placement")
    if mode == VIDEOMT_STAGE_PQ_DISABLED:
        if source_update_arm != VIDEOMT_SOURCE_UPDATE_JOINT:
            raise ValueError("disabled VidEoMT forbids a source-update intervention")
        if (
            checkpoint is not None
            or bundle is not None
            or adapted_checkpoint is not None
            or adapted_checkpoint_sha256 is not None
        ):
            raise ValueError("disabled VidEoMT Stage-PQ forbids donor asset arguments")
        if idle_placement != VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT:
            raise ValueError("disabled VidEoMT Stage-PQ forbids a donor idle placement")
        if fsdp2_placement != VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED:
            raise ValueError("disabled VidEoMT Stage-PQ forbids a donor FSDP2 placement")
        return
    if checkpoint is None or bundle is None:
        raise ValueError("enabled VidEoMT Stage-PQ requires checkpoint and DINOv3 bundle")
    if _videomt_stage_pq_adapted_active(args):
        if adapted_checkpoint is None or adapted_checkpoint_sha256 is None:
            raise ValueError(
                "adapted VidEoMT Stage-PQM requires an adapted checkpoint and SHA-256"
            )
        _require_sha256_value(
            "adapted VidEoMT checkpoint SHA-256",
            adapted_checkpoint_sha256,
        )
    elif adapted_checkpoint is not None or adapted_checkpoint_sha256 is not None:
        raise ValueError("released VidEoMT modes forbid adapted checkpoint arguments")
    if getattr(args, "posterior_architecture", None) != "two_pass_v3":
        raise ValueError("VidEoMT Stage-PQ requires the two_pass_v3 shared-host architecture")
    if _videomt_stage_pqm_active(args) and getattr(args, "dense_evidence_mode", "none") != (
        "calvin_full_v1"
    ):
        raise ValueError("full Stage-PQM requires V-JEPA, AnyTouch and Sonata evidence")
    if _videomt_native_trainable_active(args):
        if not _adr207_native_videomt_query_posterior_active(args):
            raise ValueError("trainable native VidEoMT is exclusive to the ADR-207 profile")
        if idle_placement != VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT:
            raise ValueError("trainable native VidEoMT cannot move between devices per forward")
        if getattr(args, "capacity", None) != 200:
            raise ValueError("trainable native VidEoMT requires all 200 paired queries")
        if getattr(args, "dense_token_bridge", None) != EXACT_NATIVE_MODALITY_BRIDGE:
            raise ValueError("trainable native VidEoMT forbids token resampling")
    else:
        if source_update_arm != VIDEOMT_SOURCE_UPDATE_JOINT:
            raise ValueError(
                "source-update intervention requires the native trainable VidEoMT graph"
            )
        if fsdp2_placement != VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED:
            raise ValueError("frozen VidEoMT modes forbid a trainable FSDP2 placement")


def _videomt_stage_pq_asset_receipt(args: argparse.Namespace) -> dict[str, object]:
    """Authenticate assets needed before importing or allocating the donor graph."""

    if not _videomt_stage_pq_active(args):
        return {
            "schema": "picf-next.videomt-stage-pq-assets/v2",
            "mode": VIDEOMT_STAGE_PQ_DISABLED,
            "active": False,
        }
    checkpoint = Path(args.videomt_checkpoint).expanduser().resolve()
    bundle = Path(args.videomt_dinov3_bundle).expanduser().resolve()
    config_path = bundle / "config.json"
    weights_path = bundle / "model.safetensors"
    conversion_path = bundle / "conversion_receipt.json"
    for path in (checkpoint, config_path, weights_path, conversion_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    if checkpoint.stat().st_size != VIDEOMT_RELEASED_CHECKPOINT_BYTES:
        raise ValueError("VidEoMT checkpoint byte count differs from the released artifact")
    checkpoint_sha256 = _sha256(checkpoint)
    if checkpoint_sha256 != VIDEOMT_RELEASED_CHECKPOINT_SHA256:
        raise ValueError("VidEoMT checkpoint SHA-256 differs from the released artifact")
    if config_path.stat().st_size != VIDEOMT_DINOV3_CONFIG_BYTES or (
        _sha256(config_path) != VIDEOMT_DINOV3_CONFIG_SHA256
    ):
        raise ValueError("DINOv3 constructor config differs from the exact conversion")
    conversion = json.loads(conversion_path.read_text(encoding="utf-8"))
    if not isinstance(conversion, dict):
        raise TypeError("DINOv3 conversion receipt must be a mapping")
    published = conversion.get("published_checkpoint")
    if not isinstance(published, dict) or (
        published.get("sha256") != VIDEOMT_RELEASED_CHECKPOINT_SHA256
    ):
        raise ValueError("DINOv3 conversion receipt names another source checkpoint")
    converted_tensor_count = conversion.get("converted_tensor_count")
    if (
        isinstance(converted_tensor_count, bool)
        or not isinstance(converted_tensor_count, int)
        or converted_tensor_count <= 0
    ):
        raise ValueError("DINOv3 conversion receipt has no converted tensor inventory")
    adapted: dict[str, object] | None = None
    if _videomt_stage_pq_adapted_active(args):
        adapted_candidate = Path(args.videomt_adapted_checkpoint).expanduser()
        if adapted_candidate.is_symlink():
            raise ValueError("adapted VidEoMT checkpoint must not be a symlink")
        adapted_checkpoint = adapted_candidate.resolve()
        if not adapted_checkpoint.is_file():
            raise FileNotFoundError(adapted_checkpoint)
        adapted_sha256 = _sha256(adapted_checkpoint)
        if adapted_sha256 != args.videomt_adapted_checkpoint_sha256:
            raise ValueError("adapted VidEoMT checkpoint SHA-256 differs")
        adapted = {
            "checkpoint_path": str(adapted_checkpoint),
            "checkpoint_bytes": adapted_checkpoint.stat().st_size,
            "checkpoint_sha256": adapted_sha256,
        }
    return {
        "schema": "picf-next.videomt-stage-pq-assets/v2",
        "mode": args.videomt_stage_pq_mode,
        "active": True,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": checkpoint_sha256,
        "dinov3_config_bytes": config_path.stat().st_size,
        "dinov3_config_sha256": _sha256(config_path),
        "dinov3_model_bytes": weights_path.stat().st_size,
        "conversion_receipt_sha256": _sha256(conversion_path),
        "converted_tensor_count": converted_tensor_count,
        "adapted": adapted,
    }


_ADR221_ASSET_ARGUMENTS = (
    "wsa_source_root",
    "wsa_adapted_checkpoint",
    "wsa_adapted_checkpoint_sha256",
    "da3_source_root",
    "da3_model_dir",
    "da3_model_checkpoint_sha256",
    "da3_model_config_sha256",
)


def _validate_adr221_asset_args(args: argparse.Namespace) -> None:
    supplied = tuple(
        name for name in _ADR221_ASSET_ARGUMENTS if getattr(args, name, None) is not None
    )
    if not _adr221_full_source_wsa_active(args):
        if supplied:
            raise ValueError("WSA/DA3 assets are exclusive to registered WSA profiles")
        return
    if len(supplied) != len(_ADR221_ASSET_ARGUMENTS):
        missing = tuple(name for name in _ADR221_ASSET_ARGUMENTS if name not in supplied)
        raise ValueError("registered WSA profiles require every WSA/DA3 asset: " + ", ".join(missing))
    for name in (
        "wsa_adapted_checkpoint_sha256",
        "da3_model_checkpoint_sha256",
        "da3_model_config_sha256",
    ):
        _require_sha256_value(name.replace("_", " "), getattr(args, name))
    expected = {
        "wsa_adapted_checkpoint_sha256": ADR221_WSA_ADAPTED_CHECKPOINT_SHA256,
        "da3_model_checkpoint_sha256": ADR221_DA3_MODEL_SHA256,
        "da3_model_config_sha256": ADR221_DA3_CONFIG_SHA256,
    }
    mismatched = tuple(
        name for name, value in expected.items() if getattr(args, name) != value
    )
    if mismatched:
        raise ValueError("ADR-221 asset digests differ from the frozen recipe: " + ", ".join(mismatched))


def _validate_adr221_wsa_edge_diagnostic_args(args: argparse.Namespace) -> None:
    """Keep the WSA edge intervention outside every optimization claim."""

    if not _adr221_wsa_edge_diagnostic_active(args):
        return
    measured = (
        getattr(args, "picf_architecture_profile", "legacy"),
        args.phase,
        args.load_global_step,
        args.stop_after_step,
        getattr(args, "acceptance_mode", "none"),
        args.causal_ablation_mode,
        getattr(args, "engineering_force_omitted_static_step", 0),
        getattr(args, "engineering_force_causal_diagnostic_step", 0),
    )
    expected = (
        ADR221_ARCHITECTURE_PROFILE,
        "fresh",
        0,
        1,
        "none",
        "none",
        0,
        0,
    )
    if measured != expected:
        raise ValueError(
            "ADR-221 WSA edge diagnostic requires profile/fresh/load/stop/"
            "acceptance/causal/forced=(adr221,0,1,none,none,0,0)"
        )
    if "diagnostics" not in args.run_dir.parts:
        raise ValueError("ADR-221 WSA edge diagnostic output must be under diagnostics/")


def _clean_git_revision(repo_root: Path, *, expected_revision: str) -> str:
    resolved = repo_root.expanduser().resolve()
    if resolved.is_symlink() or not resolved.is_dir():
        raise FileNotFoundError(resolved)

    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(resolved), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    if Path(run("rev-parse", "--show-toplevel")).resolve() != resolved:
        raise ValueError("DA3 source path is not its Git repository root")
    revision = run("rev-parse", "HEAD")
    if revision != expected_revision:
        raise ValueError("DA3 source revision differs from the frozen ADR-221 recipe")
    changed = run("status", "--porcelain", "--", "src")
    if changed:
        raise ValueError("DA3 runtime source tree contains local changes")
    return revision


def _adr221_asset_receipt(args: argparse.Namespace) -> dict[str, object]:
    if not _adr221_full_source_wsa_active(args):
        return {
            "schema": "picf-next.adr221-wsa-da3-assets/v1",
            "active": False,
        }
    wsa_source_root = Path(args.wsa_source_root).expanduser().resolve()
    wsa_teacher_source = (
        wsa_source_root
        / "src"
        / "lerobot"
        / "policies"
        / "WSA_Base"
        / "da3_teacher.py"
    )
    wsa_checkpoint = Path(args.wsa_adapted_checkpoint).expanduser().resolve()
    wsa_receipt_path = wsa_checkpoint.with_suffix(".receipt.json")
    da3_source_root = Path(args.da3_source_root).expanduser().resolve()
    da3_model_dir = Path(args.da3_model_dir).expanduser().resolve()
    da3_model = da3_model_dir / "model.safetensors"
    da3_config = da3_model_dir / "config.json"
    for path in (wsa_teacher_source, wsa_checkpoint, wsa_receipt_path, da3_model, da3_config):
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(path)
    measured = {
        "wsa_teacher_source_sha256": _sha256(wsa_teacher_source),
        "wsa_adapted_checkpoint_sha256": _sha256(wsa_checkpoint),
        "da3_model_checkpoint_sha256": _sha256(da3_model),
        "da3_model_config_sha256": _sha256(da3_config),
    }
    expected = {
        "wsa_teacher_source_sha256": ADR221_WSA_TEACHER_SOURCE_SHA256,
        "wsa_adapted_checkpoint_sha256": args.wsa_adapted_checkpoint_sha256,
        "da3_model_checkpoint_sha256": args.da3_model_checkpoint_sha256,
        "da3_model_config_sha256": args.da3_model_config_sha256,
    }
    mismatched = tuple(name for name, value in measured.items() if value != expected[name])
    if mismatched:
        raise ValueError("ADR-221 measured asset digests differ: " + ", ".join(mismatched))
    wsa_receipt = json.loads(wsa_receipt_path.read_text(encoding="utf-8"))
    if not isinstance(wsa_receipt, dict) or (
        wsa_receipt.get("schema")
        != "picf-next.adr218-wsa-full-depth-expert-receipt.v1"
        or wsa_receipt.get("wsa_commit") != ADR221_WSA_COMMIT
        or wsa_receipt.get("output_sha256") != measured["wsa_adapted_checkpoint_sha256"]
        or wsa_receipt.get("target_depth") != 36
        or wsa_receipt.get("target_heads") != 32
        or wsa_receipt.get("future_slots") != 432
        or wsa_receipt.get("unused_source_keys") != []
    ):
        raise ValueError("ADR-221 WSA adaptation receipt differs from the full donor contract")
    da3_revision = _clean_git_revision(
        da3_source_root,
        expected_revision=ADR221_DA3_SOURCE_COMMIT,
    )
    return {
        "schema": "picf-next.adr221-wsa-da3-assets/v1",
        "active": True,
        "wsa_commit": ADR221_WSA_COMMIT,
        "wsa_source_root": str(wsa_source_root),
        "wsa_teacher_source_sha256": measured["wsa_teacher_source_sha256"],
        "wsa_adapted_checkpoint": str(wsa_checkpoint),
        "wsa_adapted_checkpoint_sha256": measured["wsa_adapted_checkpoint_sha256"],
        "wsa_adaptation_receipt_sha256": _sha256(wsa_receipt_path),
        "wsa_future_layers": 36,
        "wsa_future_heads": 32,
        "wsa_future_slots": 432,
        "da3_source_root": str(da3_source_root),
        "da3_source_commit": da3_revision,
        "da3_model_dir": str(da3_model_dir),
        "da3_model_checkpoint_sha256": measured["da3_model_checkpoint_sha256"],
        "da3_model_config_sha256": measured["da3_model_config_sha256"],
        "teacher_target_frame": "last_frame_of_exact_causal_t_through_t_plus_4_source_clip",
    }


def _dense_evidence_training_step_prefix(
    coverage: Any,
    stream_plan: Any,
    *,
    stop_after_step: int,
) -> int:
    """Validate that one cache is an exact usable prefix of the frozen run."""

    visits = coverage.training_visit_count
    global_batch_size = stream_plan.global_batch_size
    if (
        isinstance(visits, bool)
        or not isinstance(visits, int)
        or isinstance(global_batch_size, bool)
        or not isinstance(global_batch_size, int)
        or global_batch_size <= 0
        or visits <= 0
        or visits % global_batch_size
    ):
        raise ValueError("dense evidence training visits do not form complete global steps")
    prefix_steps = visits // global_batch_size
    if prefix_steps > stream_plan.total_steps:
        raise ValueError("dense evidence prefix exceeds the frozen stream")
    if stop_after_step > prefix_steps:
        raise ValueError("dense evidence prefix does not cover this invocation")
    return prefix_steps


def _causal_ablation_active(mode: str) -> bool:
    if mode not in CAUSAL_ABLATION_MODES:
        raise ValueError(f"unknown causal ablation mode: {mode}")
    return mode != "none"


def _validate_causal_ablation_args(args: argparse.Namespace) -> None:
    mode = args.causal_ablation_mode
    active = _causal_ablation_active(mode)
    posterior_architecture = getattr(args, "posterior_architecture", "legacy_v1")
    if _adr207_native_videomt_query_posterior_active(args):
        if active:
            raise ValueError("ADR-207 forbids legacy causal-ablation modes")
        if args.phase == "fresh" and args.load_global_step != 0:
            raise ValueError("fresh ADR-207 training must start from released step zero")
        if args.phase == "resume" and (
            args.load_global_step <= 0 or args.load_global_step % CHECKPOINT_EVERY != 0
        ):
            raise ValueError("ADR-207 resume requires a positive 2k checkpoint boundary")
        return
    if posterior_architecture == "two_pass_v3":
        if active:
            raise ValueError(
                "ADR-149 uses its registered filter interventions, not legacy ablation"
            )
        posterior_adoption_dose = (
            getattr(args, "acceptance_mode", "none") == "posterior-adoption-dose"
        )
        measured = (
            args.entity_weight,
            args.predictive_weight,
            args.local_bptt_probability,
            args.overshoot_probability,
            args.source_mask_probability,
            args.source_prediction_mode,
        )
        expected = (
            ADR149_ENTITY_WEIGHT,
            ADR149_FILTER_WEIGHT,
            0.0,
            0.0,
            (
                POSTERIOR_ADOPTION_DOSE_SOURCE_MASK_PROBABILITY
                if posterior_adoption_dose
                else ADR149_OMITTED_STATIC_PROBABILITY
            ),
            "omitted_static",
        )
        if measured != expected:
            profile = (
                "posterior-adoption-dose requires entity/filter/local/overshoot/omission/mode="
                "(0.08,0.004,0,0,1.0,omitted_static)"
                if posterior_adoption_dose
                else "ADR-149 requires entity/filter/local/overshoot/omission/mode="
                "(0.08,0.004,0,0,0.10,omitted_static)"
            )
            raise ValueError(profile)
        if args.phase == "fresh" and args.load_global_step != 0:
            raise ValueError("fresh two_pass_v3 training must start from released step zero")
        dcp_step_one_restore = (
            getattr(args, "acceptance_mode", "none") == "dcp-restored"
            and args.load_global_step == 1
        )
        if (
            args.phase == "resume"
            and not dcp_step_one_restore
            and (args.load_global_step <= 0 or args.load_global_step % CHECKPOINT_EVERY != 0)
        ):
            raise ValueError("two_pass_v3 resume requires a positive 2k checkpoint boundary")
        return
    if posterior_architecture == "layerwise_v2":
        if active:
            raise ValueError("ADR-146 legacy branches cannot load the layerwise v2 state ABI")
        correction_active = _layerwise_predictive_correction_active(args)
        if correction_active:
            measured = (
                args.entity_weight,
                args.predictive_weight,
                args.local_bptt_probability,
                args.overshoot_probability,
                args.source_mask_probability,
                args.source_prediction_mode,
            )
            expected = (
                ADR148_ENTITY_WEIGHT,
                ADR148_PREDICTIVE_WEIGHT,
                0.0,
                0.0,
                ADR148_SOURCE_MASK_PROBABILITY,
                "omitted_static",
            )
            if measured != expected:
                raise ValueError(
                    "ADR-148 layerwise correction requires entity/predictive/local/"
                    "overshoot/source/mode=(0.08,0.004,0,0,0.10,omitted_static)"
                )
        elif any(
            value != 0
            for value in (
                args.local_bptt_probability,
                args.overshoot_probability,
                args.source_mask_probability,
            )
        ):
            raise ValueError(
                "layerwise_v2 core keeps local/source auxiliaries disabled when "
                "predictive correction is inactive"
            )
        if args.phase == "fresh" and args.load_global_step != 0:
            raise ValueError("fresh layerwise_v2 training must start from the official step zero")
        if args.phase == "resume" and (
            args.load_global_step <= 0 or args.load_global_step % CHECKPOINT_EVERY != 0
        ):
            raise ValueError("layerwise_v2 resume requires a positive 2k checkpoint boundary")
        return
    if posterior_architecture != "legacy_v1":
        raise ValueError("unknown posterior architecture")
    if not active:
        if args.predictive_weight <= 0:
            raise ValueError("production full training requires a positive predictive weight")
        if args.phase == "fresh" and args.load_global_step != 0:
            raise ValueError("fresh training must load global step zero")
        if args.phase == "resume" and (
            args.load_global_step <= 0 or args.load_global_step % CHECKPOINT_EVERY != 0
        ):
            raise ValueError("resume requires a positive 2k checkpoint boundary")
        return

    if args.entity_weight != CAUSAL_ENTITY_WEIGHT or args.predictive_weight != 0:
        raise ValueError("ADR-146 requires entity/predictive weights 0.08/0.0")
    if any(
        value != 0
        for value in (
            args.local_bptt_probability,
            args.overshoot_probability,
            args.source_mask_probability,
        )
    ):
        raise ValueError("ADR-146 disables local BPTT, overshoot, and source masking")
    if mode == "current_frame_branch":
        expected = ("fresh", 0, CAUSAL_BRANCH_STEP)
    else:
        expected = ("resume", CAUSAL_BRANCH_STEP, CAUSAL_ARM_STOP_STEP)
    measured = (args.phase, args.load_global_step, args.stop_after_step)
    if measured != expected:
        raise ValueError(f"ADR-146 {mode} requires phase/load/stop={expected}")


def _validate_acceptance_args(args: argparse.Namespace) -> None:
    acceptance_mode = getattr(args, "acceptance_mode", "none")
    if acceptance_mode not in ACCEPTANCE_MODES:
        raise ValueError("unknown ADR-150 acceptance mode")
    if acceptance_mode == "none":
        return
    expected_phase = {
        "action-adoption-presence": ("fresh", 0, 1),
        "action-adoption-interventions": ("fresh", 0, 1),
        "posterior-adoption-route": ("fresh", 0, POSTERIOR_ADOPTION_STOP_STEP),
        "posterior-adoption-dose": ("fresh", 0, POSTERIOR_ADOPTION_DOSE_STOP_STEP),
        "dcp-uninterrupted": ("fresh", 0, 2),
        "dcp-restored": ("resume", 1, 2),
    }.get(acceptance_mode)
    measured_phase = (args.phase, args.load_global_step, args.stop_after_step)
    if expected_phase is None or measured_phase != expected_phase:
        raise ValueError("ADR-150 acceptance phase/load/stop differs from its registered contract")
    if (
        args.posterior_architecture,
        args.dense_evidence_mode,
        args.causal_ablation_mode,
    ) != ("two_pass_v3", "calvin_full_v1", "none"):
        raise ValueError(
            "ADR-150 acceptance requires two_pass_v3, calvin_full_v1 and no causal ablation"
        )


def _validate_engineering_smoke_args(args: argparse.Namespace) -> None:
    """Keep the bounded forced omitted-static reproduction outside scientific runs."""

    forced_omitted_step = getattr(args, "engineering_force_omitted_static_step", 0)
    forced_causal_step = getattr(args, "engineering_force_causal_diagnostic_step", 0)
    if forced_omitted_step and forced_causal_step:
        raise ValueError("engineering smokes cannot force two independent branches")
    forced_step = forced_omitted_step or forced_causal_step
    if forced_step == 0:
        return
    if (
        isinstance(forced_step, bool)
        or not isinstance(forced_step, int)
        or not 1 <= forced_step <= 32
    ):
        raise ValueError("forced omitted-static smoke step must be an integer in [1, 32]")
    followup_steps = args.stop_after_step - forced_step
    measured = (
        args.phase,
        args.acceptance_mode,
        args.load_global_step,
        0 <= followup_steps <= 2,
        args.posterior_architecture,
        args.dense_evidence_mode,
        args.source_mask_probability,
        args.source_prediction_mode,
    )
    expected = (
        "fresh",
        "none",
        0,
        True,
        "two_pass_v3",
        "calvin_full_v1",
        ADR149_OMITTED_STATIC_PROBABILITY,
        "omitted_static",
    )
    if measured != expected:
        raise ValueError(
            "forced engineering smoke requires fresh/none/0/stop in "
            "[forced-step,forced-step+2]/two_pass_v3/calvin_full_v1/"
            "0.10/omitted_static"
        )
    if forced_causal_step and forced_causal_step < 3:
        raise ValueError("forced causal diagnostic smoke requires step in [3, 32]")
    if "diagnostics" not in args.run_dir.parts:
        raise ValueError("forced engineering smoke output must be under diagnostics/")


def _posterior_adoption_route_active(mode: str) -> bool:
    if mode not in ACCEPTANCE_MODES:
        raise ValueError("unknown ADR-150 acceptance mode")
    return mode in {"posterior-adoption-route", "posterior-adoption-dose"}


def _posterior_adoption_route_active_for_args(args: argparse.Namespace) -> bool:
    return _posterior_adoption_route_active(
        getattr(args, "acceptance_mode", "none")
    ) or _adr222_world_token_adoption_active(args)


def _registered_action_evaluation_steps(args: argparse.Namespace) -> tuple[int, ...]:
    if _adr221_wsa_edge_diagnostic_active(args):
        return (0,)
    if getattr(args, "acceptance_mode", "none") == "posterior-adoption-dose":
        return POSTERIOR_ADOPTION_DOSE_ACTION_EVALUATION_STEPS
    registered = TWO_PASS_ACTION_EVALUATION_STEPS
    stop_after_step = int(args.stop_after_step)
    combined_adr178_acceptance = bool(
        _adr178_direct_action_posterior_active(args)
        and stop_after_step == ADR178_ACCEPTANCE_STOP_STEP
    )
    combined_adr193_anchor_gate = bool(
        _adr193_implicit_multimodal_anchor_active(args)
        and stop_after_step == ADR178_ACCEPTANCE_STOP_STEP
    )
    if args.phase == "fresh" and (
        stop_after_step in registered
        or combined_adr178_acceptance
        or combined_adr193_anchor_gate
    ):
        return tuple(step for step in registered if step <= stop_after_step)
    return registered


def _summarize_adr207_modality_interventions(
    reports: tuple[Mapping[str, object], ...],
    *,
    checkpoint_global_step: int,
    expected_world_size: int,
) -> dict[str, object]:
    """Fail closed on the mature matched-input full-modal action intervention gate."""

    if checkpoint_global_step not in ADR207_MODALITY_INTERVENTION_STEPS:
        raise ValueError("ADR-207 modality intervention ran at an unregistered step")
    if len(reports) != expected_world_size or expected_world_size not in SUPPORTED_WORLD_SIZES:
        raise ValueError("ADR-207 modality intervention reports differ from world size")
    sample_keys = tuple(report.get("sample_key") for report in reports)
    if any(not isinstance(key, str) or not key for key in sample_keys) or len(
        set(sample_keys)
    ) != len(sample_keys):
        raise ValueError("ADR-207 modality intervention sample keys are invalid")

    factual_repeat_max_abs = 0.0
    for report in reports:
        factual_repeat = report.get("factual_repeat")
        if not isinstance(factual_repeat, Mapping):
            raise ValueError("ADR-207 modality intervention lacks factual repeat evidence")
        drift = float(factual_repeat.get("max_abs", math.inf))
        if not math.isfinite(drift):
            raise ValueError("ADR-207 factual-repeat drift is not finite")
        factual_repeat_max_abs = max(factual_repeat_max_abs, drift)
    if factual_repeat_max_abs > ADR207_ACTION_STABILITY_MAX_ABS_DRIFT:
        raise ValueError("ADR-207 factual action replay is not stable")

    summaries: dict[str, object] = {}
    for modality in CALVIN_FULL_DENSE_MODALITIES:
        intervention_summary: dict[str, object] = {}
        for intervention in ADR207_MODALITY_INTERVENTIONS:
            rows: list[Mapping[str, object]] = []
            for report in reports:
                modalities = report.get("modalities")
                if not isinstance(modalities, Mapping):
                    raise ValueError("ADR-207 modality report is malformed")
                modality_rows = modalities.get(modality)
                if not isinstance(modality_rows, Mapping):
                    raise ValueError(f"ADR-207 report omitted modality {modality}")
                row = modality_rows.get(intervention)
                if not isinstance(row, Mapping):
                    raise ValueError(
                        f"ADR-207 report omitted {modality}:{intervention}"
                    )
                rows.append(row)
            changed_elements = sum(int(row.get("changed_elements", -1)) for row in rows)
            if changed_elements <= 0:
                raise ValueError(
                    f"ADR-207 {modality}:{intervention} changed no held-out evidence"
                )
            max_abs = max(float(row.get("max_abs", math.nan)) for row in rows)
            rms = max(float(row.get("rms", math.nan)) for row in rows)
            if not math.isfinite(max_abs) or not math.isfinite(rms):
                raise ValueError(
                    f"ADR-207 {modality}:{intervention} action drift is not finite"
                )
            if intervention == "joint_permutation":
                if max_abs > ADR207_ACTION_STABILITY_MAX_ABS_DRIFT:
                    raise ValueError(
                        f"ADR-207 {modality} joint set permutation changed action"
                    )
                gate = {
                    "comparison": "maximum_allowed",
                    "threshold": ADR207_ACTION_STABILITY_MAX_ABS_DRIFT,
                }
            else:
                if max_abs < ADR207_ACTION_EFFECT_MIN_ABS_DRIFT:
                    raise ValueError(
                        f"ADR-207 {modality}:{intervention} did not affect action"
                    )
                gate = {
                    "comparison": "minimum_required",
                    "threshold": ADR207_ACTION_EFFECT_MIN_ABS_DRIFT,
                }
            intervention_summary[intervention] = {
                "global_changed_elements": changed_elements,
                "maximum_rank_max_abs_action_drift": max_abs,
                "maximum_rank_rms_action_drift": rms,
                **gate,
            }
        summaries[modality] = intervention_summary

    return {
        "schema": ADR207_MODALITY_INTERVENTION_SCHEMA,
        "status": "PASS",
        "checkpoint_global_step": checkpoint_global_step,
        "architecture_identity": NATIVE_VIDEOMT_QUERY_POSTERIOR,
        "state_mode": "cold_reset_fixed_source_queries",
        "source_query_count": 200,
        "sample_keys": list(sample_keys),
        "factual_repeat_max_abs_action_drift": factual_repeat_max_abs,
        "factual_repeat_maximum_allowed": ADR207_ACTION_STABILITY_MAX_ABS_DRIFT,
        "modalities": summaries,
    }


def _summarize_adr221_wsa_edge_intervention(
    samples: list[Mapping[str, object]],
    *,
    partition: str,
) -> dict[str, object]:
    """Summarize one exact paired WSA future-to-action edge intervention."""

    selected = tuple(sample for sample in samples if sample.get("partition") == partition)
    if not selected:
        raise ValueError(f"ADR-221 WSA edge diagnostic has no {partition} samples")
    factual_losses: list[float] = []
    blocked_losses: list[float] = []
    standard_losses: list[float] = []
    deltas: list[float] = []
    for sample in selected:
        intervention = sample.get("wsa_future_to_action_intervention")
        if not isinstance(intervention, Mapping):
            raise ValueError("ADR-221 action sample omitted its WSA edge intervention")
        if intervention.get("intervention") != "block_future_to_action":
            raise ValueError("ADR-221 action sample used another WSA intervention")
        if intervention.get("source_host_batch_reused_by_identity") is not True:
            raise ValueError("ADR-221 WSA intervention did not reuse one source host batch")
        if intervention.get("posterior_exact_equal") is not True:
            raise ValueError("ADR-221 WSA intervention changed the emitted posterior")
        factual = float(intervention.get("factual_action_loss", math.nan))
        blocked = float(intervention.get("blocked_action_loss", math.nan))
        delta = float(intervention.get("blocked_minus_factual_action_loss", math.nan))
        standard = float(sample.get("action_loss", math.nan))
        if not all(math.isfinite(value) for value in (factual, blocked, delta, standard)):
            raise ValueError("ADR-221 WSA edge diagnostic contains a non-finite loss")
        if not math.isclose(
            delta,
            blocked - factual,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError("ADR-221 WSA edge paired action losses are inconsistent")
        factual_losses.append(factual)
        blocked_losses.append(blocked)
        standard_losses.append(standard)
        deltas.append(delta)
    standard_mean = sum(standard_losses) / len(standard_losses)
    factual_mean = sum(factual_losses) / len(factual_losses)
    blocked_mean = sum(blocked_losses) / len(blocked_losses)
    mean_delta = sum(deltas) / len(deltas)
    return {
        "schema": ADR221_WSA_EDGE_INTERVENTION_SCHEMA,
        "partition": partition,
        "sample_count": len(selected),
        "standard_action_loss_mean": standard_mean,
        "factual_action_loss_mean": factual_mean,
        "paired_factual_minus_standard_action_loss_mean": factual_mean - standard_mean,
        "blocked_action_loss_mean": blocked_mean,
        "blocked_minus_factual_action_loss_mean": mean_delta,
        "relative_mean_delta": mean_delta / max(abs(factual_mean), 1e-12),
        "blocked_improved_fraction": sum(delta < 0.0 for delta in deltas) / len(deltas),
        "source_host_batch_reused_by_identity_fraction": 1.0,
        "posterior_exact_equal_fraction": 1.0,
    }


def _acceptance_terminal_evidence_due(*, mode: str, global_step: int) -> bool:
    if mode not in ACCEPTANCE_MODES:
        raise ValueError("unknown ADR-150 acceptance mode")
    return mode == "posterior-adoption-dose" and global_step == (POSTERIOR_ADOPTION_DOSE_STOP_STEP)


def _validate_posterior_adoption_dose_step(
    *,
    mode: str,
    source_masked_branch: bool,
    omitted_static_view: object | None,
    posterior_adoption_route: object | None,
    expected_batch_size: int,
    result: object,
) -> None:
    """Fail closed unless the high-dose arm executes both action branches."""

    if mode != "posterior-adoption-dose":
        return
    if source_masked_branch is not True or omitted_static_view is None:
        raise RuntimeError("posterior-adoption-dose omitted its every-step static-view branch")
    route_shape = getattr(posterior_adoption_route, "shape", None)
    route_all = getattr(posterior_adoption_route, "all", None)
    if route_shape != (expected_batch_size,) or not callable(route_all):
        raise RuntimeError("posterior-adoption-dose route mask has the wrong batch contract")
    if not bool(route_all().item()):
        raise RuntimeError("posterior-adoption-dose route mask did not select every sample")
    if getattr(result, "primary", None) is None:
        raise RuntimeError("posterior-adoption-dose omitted its factual action branch")
    if (
        getattr(result, "omitted_static_branch", None) is None
        or getattr(
            result,
            "omitted_static_policy",
            None,
        )
        is None
    ):
        raise RuntimeError("posterior-adoption-dose omitted its routed action branch")


def _acceptance_checkpoint_due(*, mode: str, global_step: int) -> bool:
    if mode == "dcp-uninterrupted":
        return global_step == 1
    if mode == "posterior-adoption-route":
        return global_step == POSTERIOR_ADOPTION_STOP_STEP
    if mode == "posterior-adoption-dose":
        return global_step == POSTERIOR_ADOPTION_DOSE_STOP_STEP
    if mode in {
        "none",
        "action-adoption-presence",
        "action-adoption-interventions",
        "dcp-restored",
    }:
        return False
    raise ValueError("unknown ADR-150 acceptance mode")


def _action_evaluation_active(args: argparse.Namespace) -> bool:
    registered_steps = _registered_action_evaluation_steps(args)
    return _two_pass_filter_active(args) and (
        args.stop_after_step >= registered_steps[-1]
        or _posterior_adoption_route_active_for_args(args)
    )


def _execution_contract(args: argparse.Namespace) -> dict[str, object]:
    causal = _causal_ablation_active(args.causal_ablation_mode)
    physical_event_stream = _two_pass_filter_active(args)
    raw_acceptance_mode = getattr(args, "acceptance_mode", "none")
    posterior_adoption_route = _posterior_adoption_route_active_for_args(args)
    posterior_adoption_dose = raw_acceptance_mode == "posterior-adoption-dose"
    acceptance_mode = raw_acceptance_mode
    if acceptance_mode in {"dcp-uninterrupted", "dcp-restored"}:
        acceptance_mode = "dcp-cold-restore"
    return {
        "schema": (
            CAUSAL_ABLATION_SCHEMA
            if causal
            else "picf-next.task-independent-full-execution-contract/v9"
        ),
        "experiment": "adr146-recurrence-only" if causal else "production-full",
        "acceptance_mode": acceptance_mode,
        "engineering_force_omitted_static_step": int(
            getattr(args, "engineering_force_omitted_static_step", 0)
        ),
        "engineering_force_causal_diagnostic_step": int(
            getattr(args, "engineering_force_causal_diagnostic_step", 0)
        ),
        "allowed_modes": CAUSAL_ABLATION_MODES[1:] if causal else ("none",),
        "branch_step": CAUSAL_BRANCH_STEP if causal else None,
        "arm_stop_step": CAUSAL_ARM_STOP_STEP if causal else None,
        "world_size": WORLD_SIZE,
        "global_batch_size": WORLD_SIZE,
        "gradient_accumulation_steps": 1,
        "trainable_scope": getattr(
            args,
            "trainable_scope",
            TRAINABLE_SCOPE_FULL_HOST,
        ),
        "physical_event_stream": physical_event_stream,
        "prompt_overlay": (
            "deterministic_plan_episode_sample_candidates_v1" if physical_event_stream else None
        ),
        "control_receipt": (
            "exact_raw_actions_chunked_without_semantic_compression_v1"
            if physical_event_stream
            else None
        ),
        "distributed_prior_host_schedule": (
            V3_DISTRIBUTED_PRIOR_SCHEDULE if physical_event_stream else None
        ),
        "stream_plan_file_sha256": getattr(args, "stream_plan_sha256", None),
        "representation_split_file_sha256": getattr(
            args,
            "representation_split_sha256",
            None,
        ),
        "evaluation_plan_file_sha256": getattr(
            args,
            "evaluation_plan_sha256",
            None,
        ),
        "future_latent_alignment": (
            "flare-generic-siglip2-large-patch16-256-complete-v1"
            if _future_latent_alignment_active(args)
            else None
        ),
        "future_latent_objective_scale": getattr(
            args,
            "future_latent_objective_scale",
            1.0,
        ),
        "future_latent_cache_manifest_sha256": getattr(
            args,
            "future_latent_cache_manifest_sha256",
            None,
        ),
        "future_latent_cache_build_report_sha256": getattr(
            args,
            "future_latent_cache_build_report_sha256",
            None,
        ),
        "minimum_future_source_frames": _required_future_source_frames(args),
        "action_evaluation_steps": (
            _registered_action_evaluation_steps(args) if physical_event_stream else ()
        ),
        "native_query_modality_intervention_steps": (
            list(ADR207_MODALITY_INTERVENTION_STEPS)
            if _adr207_native_videomt_query_posterior_active(args)
            else []
        ),
        "native_query_modality_interventions": (
            list(ADR207_MODALITY_INTERVENTIONS)
            if _adr207_native_videomt_query_posterior_active(args)
            else []
        ),
        "native_query_modality_intervention_scope": (
            "fixed_source_queries_frozen_training_stream_wiring"
            if _adr207_native_videomt_query_posterior_active(args)
            else None
        ),
        "anchor_evaluation_steps": list(_registered_anchor_evaluation_steps(args)),
        "seed": args.seed,
        "capacity": args.capacity,
        "attention_implementation": getattr(
            args,
            "attention_implementation",
            LINGBOT_ATTENTION_EAGER,
        ),
        "lingbot_compile_mode": getattr(
            args,
            "lingbot_compile_mode",
            LINGBOT_COMPILE_DISABLED,
        ),
        "posterior_architecture": getattr(
            args,
            "posterior_architecture",
            "legacy_v1",
        ),
        "picf_architecture_profile": getattr(
            args,
            "picf_architecture_profile",
            "legacy",
        ),
        "action_backend": getattr(args, "action_backend", "lingbot_released"),
        "wla_host_evidence_arm": getattr(
            args,
            "wla_host_evidence_arm",
            WLA_HOST_EVIDENCE_ARMS[0],
        ),
        "wla_source_root": (
            str(args.wla_source_root.resolve()) if _wla_complete_active(args) else None
        ),
        "wla_pretrained_root": (
            str(args.wla_pretrained_root.resolve()) if _wla_complete_active(args) else None
        ),
        "wsa_future_3d_active": _adr221_full_source_wsa_active(args),
        "wsa_future_3d_topology": (
            "released_36_layer_32_head_432_slot_synchronous_mot"
            if _adr221_full_source_wsa_active(args)
            else None
        ),
        "wsa_da3_teacher_target": (
            "two_camera_t_plus_4_last_exact_causal_source_frame"
            if _adr221_full_source_wsa_active(args)
            else None
        ),
        "wsa_action_coupling": (
            "auxiliary_world_decoder_no_future_action_keys"
            if _adr222_world_token_adoption_active(args)
            else (
                "direct_future_keys"
                if _adr221_full_source_wsa_active(args)
                else None
            )
        ),
        "action_scene_context": (
            "task_language_plus_proprioception_plus_200_shared_object_rows"
            if _adr222_world_token_adoption_active(args)
            else "native_lingbot_visibility"
        ),
        "wsa_future_to_action_edge_diagnostic": (
            {
                "active": True,
                "intervention": "block_future_to_action",
                "scope": "measurement_only_fixed_source_host_replay",
                "source_host_batch_reused_by_identity": True,
                "paired_host_rng_replayed_exactly": True,
                "posterior_must_be_exact_equal": True,
                "optimization_graph_changed": False,
            }
            if _adr221_wsa_edge_diagnostic_active(args)
            else {"active": False}
        ),
        "task_query_count": getattr(args, "task_query_count", 0),
        "relation_supervision_layers": list(getattr(args, "relation_supervision_layers", ())),
        "direct_action_posterior_attention": _adr178_direct_action_posterior_active(args),
        "direct_action_posterior_attention_weight": (
            float(ADR178_NATIVE_ATTENTION_WEIGHT).hex()
            if _adr178_direct_action_posterior_active(args)
            else None
        ),
        "direct_action_posterior_layer_offsets": (
            list(ADR178_REGISTERED_LAYER_OFFSETS)
            if _adr178_direct_action_posterior_active(args)
            else []
        ),
        "direct_action_posterior_head_indices": (
            list(ADR178_REGISTERED_ACTION_HEAD_INDICES)
            if _adr178_direct_action_posterior_active(args)
            else []
        ),
        "objective_profile": _objective_profile(args),
        "object_action_information_contract": _object_action_information_contract(args),
        "maximum_control_tokens": args.maximum_control_tokens,
        "prior_gradient_control_tokens": getattr(
            args,
            "prior_gradient_control_tokens",
            args.maximum_control_tokens,
        ),
        "prior_gradient_schedule": "ordered_no_grad_burnin_plus_attached_suffix_v1",
        "maximum_optimizer_lag": args.maximum_optimizer_lag,
        "learning_rate": float(args.learning_rate).hex(),
        "picf_learning_rate_multiplier": float(
            getattr(args, "picf_learning_rate_multiplier", 1.0)
        ).hex(),
        "modality_bridge_learning_rate_multiplier": float(
            getattr(args, "modality_bridge_learning_rate_multiplier", 1.0)
        ).hex(),
        "max_grad_norm": float(args.max_grad_norm).hex(),
        "maximum_peak_reserved_gib": float(args.maximum_peak_reserved_gib).hex(),
        "entity_weight": float(args.entity_weight).hex(),
        "predictive_weight": float(args.predictive_weight).hex(),
        "mask_focal_weight": float(args.mask_focal_weight).hex(),
        "mask_dice_weight": float(args.mask_dice_weight).hex(),
        "existence_weight": float(args.existence_weight).hex(),
        "ownership_weight": float(args.ownership_weight).hex(),
        "predictive_loss_power": float(args.predictive_loss_power).hex(),
        "local_bptt_probability": float(args.local_bptt_probability).hex(),
        "overshoot_probability": float(args.overshoot_probability).hex(),
        "source_mask_probability": float(args.source_mask_probability).hex(),
        "source_mask_token_fraction": float(args.source_mask_token_fraction).hex(),
        "source_prediction_mode": args.source_prediction_mode,
        "minimum_supervised_fraction": float(args.minimum_supervised_fraction).hex(),
        "fsdp2_placement": args.fsdp2_placement,
        "fsdp2_backward_prefetch": getattr(
            args,
            "fsdp2_backward_prefetch",
            FSDP2_BACKWARD_PREFETCH_DEFAULT,
        ),
        "sequential_factual_gradient_storage": getattr(
            args,
            "sequential_factual_gradient_storage",
            FSDP2_FACTUAL_GRADIENT_GPU,
        ),
        "omitted_static_rematerialization": getattr(
            args,
            "omitted_static_rematerialization",
            OMITTED_STATIC_REMATERIALIZATION_NONE,
        ),
        "cuda_allocator": args.cuda_allocator,
        "dense_evidence_mode": getattr(args, "dense_evidence_mode", "none"),
        "native_relation_surfaces": (
            [
                {
                    "name": "anytouch",
                    "geometry_kind": "contact_sites",
                    "layout": "anytouch2.calvin.contact-sites.v1",
                    "target_kind": NO_RELATION_TARGET,
                },
                {
                    "name": "sonata",
                    "geometry_kind": "world_points",
                    "layout": "sonata.calvin.world-points.v1",
                    "target_kind": NO_RELATION_TARGET,
                },
                {
                    "name": "vjepa",
                    "geometry_kind": "image_grid",
                    "layout": "vjepa21.calvin.static-gripper.24x24.v1",
                    "target_kind": CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
                },
            ]
            if getattr(args, "dense_evidence_mode", "none") == "calvin_full_v1"
            else []
        ),
        "posterior_adoption_route": posterior_adoption_route,
        "posterior_adoption_dose": posterior_adoption_dose,
        "posterior_adoption_factual_branch": posterior_adoption_route,
        "posterior_adoption_routed_action_every_step": posterior_adoption_dose,
        "dense_token_bridge": getattr(
            args,
            "dense_token_bridge",
            EXACT_NATIVE_MODALITY_BRIDGE,
        ),
        "dense_evidence_cache_manifest_sha256": tuple(
            sorted(getattr(args, "dense_evidence_cache_manifest_sha256", ()) or ())
        ),
        "dense_evidence_supplement_cache_manifest_sha256": tuple(
            sorted(
                getattr(
                    args,
                    "dense_evidence_supplement_cache_manifest_sha256",
                    (),
                )
                or ()
            )
        ),
        "dense_evidence_subset_view": bool(
            getattr(args, "dense_evidence_subset_view", False)
        ),
        "dense_evidence_coverage_plan_file_sha256": getattr(
            args,
            "dense_evidence_coverage_plan_sha256",
            None,
        ),
        "videomt_stage_pq": {
            "mode": getattr(
                args,
                "videomt_stage_pq_mode",
                VIDEOMT_STAGE_PQ_DISABLED,
            ),
            "active": _videomt_stage_pq_active(args),
            **(
                {
                    "source_update_arm": getattr(
                        args,
                        "videomt_source_update_arm",
                        VIDEOMT_SOURCE_UPDATE_JOINT,
                    ),
                    "source_forward_and_backward_graph": "unchanged_complete_joint_graph",
                }
                if _videomt_native_trainable_active(args)
                else {}
            ),
            "donor_execution_mode": (
                (
                    "complete_calvin_adapted_train_graph_fsdp2_joint"
                    if _videomt_native_trainable_active(args)
                    else (
                        "complete_calvin_adapted_eval_graph_fp32_frozen"
                        if _videomt_stage_pq_adapted_active(args)
                        else "complete_released_eval_graph_fp32_frozen"
                    )
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "temporal_adapter": (
                (
                    "five_real_raw_episode_frames_t_through_t_plus_4_no_padding"
                    if _videomt_native_trainable_active(args)
                    else "five_real_raw_episode_frames_t_minus_4_through_t_no_padding"
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "host_boundary": (
                (
                    "all_200_source_masks_pool_same_index_pretrained_qwen3_visual_cells_"
                    "through_exact_copied_native_merger_mlp_as_shared_host_posterior_"
                    "rows_with_original_qwen3_prefix_retained_no_selection"
                    if _adr225_pretrained_object_memory_active(args)
                    else "all_200_native_source_queries_and_source_masks_same_index_as_"
                    "shared_host_posterior_rows_no_assignment_or_reverse_projection"
                )
                if _videomt_native_trainable_active(args)
                else (
                    (
                        "latest_source_query_bank_with_first_16_rows_replaced_by_released_"
                        "query_propagation_plus_prefixes_patches_rope_and_complete_frozen_"
                        "blocks20_23_prediction_stack"
                        if _adr205_released_query_propagation_active(args)
                        else "latest_all_200_queries_prefixes_patches_rope_plus_complete_"
                        "frozen_blocks20_23_prediction_stack_for_posterior_rows"
                    )
                    if _videomt_stage_pqrf_active(args)
                    else (
                        "latest_all_200_queries_plus_mask_embeddings_and_dense_mask_features_"
                        "tied_posterior_row_decoder_no_selection_or_local_semantic_head"
                        if _videomt_stage_pqmr_active(args)
                        else (
                            "latest_all_200_queries_plus_complete_class_mask_relation_"
                            "no_selection_pooling_or_local_decoder"
                            if _videomt_stage_pqm_active(args)
                            else (
                                "latest_all_200_queries_no_selection_pooling_resampling_or_"
                                "second_norm"
                            )
                        )
                    )
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "idle_placement": (
                getattr(
                    args,
                    "videomt_idle_placement",
                    VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "idle_placement_changes_tensor_semantics": False,
            "fsdp2_placement": (
                getattr(
                    args,
                    "videomt_fsdp2_placement",
                    VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "fsdp2_placement_changes_model_semantics": False,
            "short_prefix_policy": (
                (
                    "stream_domain_requires_four_future_frames"
                    if _videomt_native_trainable_active(args)
                    else "empty_absent_stream_counted"
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "query_count": 200 if _videomt_stage_pq_active(args) else 0,
            "query_width": 1024 if _videomt_stage_pq_active(args) else 0,
            **(
                {
                    "mask_grid": [120, 120],
                    "pixel_row_composition": (
                        (
                            "bilinear_resize(source_mask_logit_i,to_qwen3_merged_grid);"
                            "weight_i=sigmoid(logit_i)*pixel_valid;cell_i=normalized_weighted_"
                            "mean(exact_native_post_norm_pre_mlp_qwen3_cells);posterior_row_i="
                            "exact_copied_qwen3_merger_mlp(cell_i)"
                            if _adr225_pretrained_object_memory_active(args)
                            else "source_mask(query_i,pixel);posterior_row_i=same_native_query_"
                            "address;no_host_to_source_mask_decode"
                        )
                        if _videomt_native_trainable_active(args)
                        else (
                            (
                                "conditioned_row=source_query_prior+source_query_updater("
                                "tied_projected_posterior_row);replace_first_16_source_queries;"
                                "run_frozen_released_blocks20_23_with_rope;row_mask_logit=dot("
                                "released_mask_head(refined_row),released_upscale(refined_patches));"
                                "ownership=softmax(rows_plus_context)"
                                if _adr205_released_query_propagation_active(args)
                                else "prepend(tied_projected_posterior_rows,"
                                "complete_source_stream);"
                                "run_frozen_released_blocks20_23_with_rope;row_mask_logit=dot("
                                "released_mask_head(refined_row),released_upscale(refined_patches));"
                                "ownership=softmax(rows_plus_context)"
                            )
                            if _videomt_stage_pqrf_active(args)
                            else (
                                "row_mask_logit=dot(released_mask_head("
                                "transpose(semantic_query_projection)(full_host_posterior_row)),"
                                "donor_dense_mask_feature);ownership=softmax(rows_plus_context)"
                                if _videomt_stage_pqmr_active(args)
                                else (
                                    "p(row|pixel)=sum_query "
                                    "p(row|query,full_host)*p(query|pixel,donor)"
                                )
                            )
                        )
                    ),
                    "semantic_owner": (
                        "complete_videomt_dinov3_l_and_complete_lingbot_host"
                    ),
                    "local_object_selector_decoder_or_lifecycle_head": False,
                    "posterior_query_integration": (
                        "same_index_pretrained_qwen3_object_memory_no_random_source_query_"
                        "projection"
                        if _adr225_pretrained_object_memory_active(args)
                        else (
                            "same_index_source_to_host_width_projection"
                            if _videomt_native_trainable_active(args)
                            else (
                                "replace_with_released_propagation"
                                if _adr205_released_query_propagation_active(args)
                                else "prepend_projected_rows"
                            )
                        )
                    ),
                    **(
                        {
                            "object_memory_source_primitive": (
                                "unipixel_cache_native_merger_input_mask_mean_then_native_"
                                "merger_mlp"
                            ),
                            "object_memory_source_exact_parts": [
                                "native_qwen3_post_norm_pre_mlp_visual_cells",
                                "deep_copied_linear_fc1_gelu_linear_fc2",
                                "mask_conditioned_cell_mean",
                                "language_width_memory_token_in_shared_vlm",
                            ],
                            "object_memory_picf_adaptations": [
                                "videomt_posterior_mask_logits",
                                "bilinear_logit_resize_before_sigmoid",
                                "soft_normalized_posterior_mean",
                                "all_200_same_index_queries_without_winner_selection",
                            ],
                            "object_memory_required_ablation": (
                                "source_faithful_binary_mask_mean_vs_soft_posterior_mean"
                            ),
                            "object_memory_external_inference_model": None,
                        }
                        if _adr225_pretrained_object_memory_active(args)
                        else {}
                    ),
                }
                if _videomt_stage_pqm_active(args)
                else {}
            ),
            "checkpoint_sha256": (
                (
                    args.videomt_adapted_checkpoint_sha256
                    if _videomt_stage_pq_adapted_active(args)
                    else VIDEOMT_RELEASED_CHECKPOINT_SHA256
                )
                if _videomt_stage_pq_active(args)
                else None
            ),
            "released_checkpoint_sha256": (
                VIDEOMT_RELEASED_CHECKPOINT_SHA256
                if _videomt_stage_pq_active(args)
                else None
            ),
        },
    }


def _physical_step_observability(
    *,
    active: bool,
    planned: Any,
    primary_batch: Any,
    sequence_batch_count: int,
    egress_batch: Any | None,
    prior_host_steps_by_batch: tuple[int, ...] | None,
    prior_gradient_suffix_steps_by_batch: tuple[int, ...] | None,
    egress_prior_host_steps: int | None,
    result: Any,
) -> dict[str, object]:
    """Fail closed on, then serialize, the ADR-149 physical/two-pass receipt."""

    prompt_digest = planned.physical_prompt_selection_sha256
    control_digests = tuple(planned.training.physical_control_span_sha256)
    segment_indices = tuple(planned.training.selected_segment_indices)
    prior_chunks = tuple(primary_batch.prior_control_chunks)
    if not active:
        if prompt_digest is not None or control_digests or segment_indices or prior_chunks:
            raise RuntimeError("non-physical training unexpectedly produced physical receipts")
        return {"active": False}
    if prompt_digest is None or not control_digests or not segment_indices or not prior_chunks:
        raise RuntimeError("ADR-149 physical training omitted a source receipt")
    if len(control_digests) != len(segment_indices):
        raise RuntimeError("ADR-149 physical source receipts have different batch axes")
    if sequence_batch_count <= 0:
        raise RuntimeError("ADR-149 two-pass sequence must contain a factual frame")
    if (
        prior_host_steps_by_batch is None
        or len(prior_host_steps_by_batch) != sequence_batch_count
        or prior_host_steps_by_batch[0] < len(primary_batch.effective_prior_control_chunks)
    ):
        raise RuntimeError("ADR-149 omitted its rank-symmetric prior host schedule")
    if (
        prior_gradient_suffix_steps_by_batch is None
        or len(prior_gradient_suffix_steps_by_batch) != sequence_batch_count
        or not 1 <= prior_gradient_suffix_steps_by_batch[0] <= prior_host_steps_by_batch[0]
    ):
        raise RuntimeError("ADR-176 omitted its bounded prior-gradient schedule")
    if (egress_prior_host_steps is None) != (egress_batch is None):
        raise RuntimeError("ADR-149 egress and prior host schedules disagree")
    if egress_batch is not None and egress_prior_host_steps < len(
        egress_batch.effective_prior_control_chunks
    ):
        raise RuntimeError("ADR-149 egress host schedule is shorter than its control chain")

    prior_trace_count = len(result.v3_prior_traces)
    attached_egress_result = result.attached_egress is not None
    expected_filter_phase_count = 2 * sequence_batch_count + int(egress_batch is not None)
    if prior_trace_count != sequence_batch_count:
        raise RuntimeError("ADR-149 did not produce one prior trace per factual sequence frame")
    if attached_egress_result != (egress_batch is not None):
        raise RuntimeError("ADR-149 attached-egress request and result disagree")
    if len(result.filter_phase_branches) != expected_filter_phase_count:
        raise RuntimeError("ADR-149 did not expose every prior/posterior/egress filter phase")

    return {
        "active": True,
        "physical_prompt_selection_sha256": prompt_digest,
        "physical_control_span_sha256": list(control_digests),
        "selected_segment_indices": list(segment_indices),
        "prior_control_chunk_count": len(prior_chunks),
        "prior_control_chunk_token_counts": [chunk.token_count for chunk in prior_chunks],
        "prior_host_steps_by_batch": list(prior_host_steps_by_batch),
        "prior_gradient_suffix_steps_by_batch": list(prior_gradient_suffix_steps_by_batch),
        "sequence_batch_count": sequence_batch_count,
        "v3_prior_trace_count": prior_trace_count,
        "filter_phase_branch_count": len(result.filter_phase_branches),
        "expected_filter_phase_branch_count": expected_filter_phase_count,
        "attached_egress_result": attached_egress_result,
        "egress_source_digest": (None if egress_batch is None else egress_batch.source_digest),
        "egress_prior_host_steps": egress_prior_host_steps,
    }


def _native_videomt_step_observability(
    *,
    active: bool,
    planned: Any,
    primary_batch: Any,
    source_batch: Any,
    sequence_batch_count: int,
    egress_batch: Any | None,
    prior_host_steps_by_batch: tuple[int, ...] | None,
    prior_gradient_suffix_steps_by_batch: tuple[int, ...] | None,
    egress_prior_host_steps: int | None,
    result: Any,
    native_relation_type: type,
) -> dict[str, object]:
    """Validate the ADR-207 source-complete/current-host information boundary."""

    if not active:
        raise RuntimeError("ADR-207 requires the physical CALVIN event stream")
    prompt_digest = planned.physical_prompt_selection_sha256
    control_digests = tuple(planned.training.physical_control_span_sha256)
    segment_indices = tuple(planned.training.selected_segment_indices)
    prior_chunks = tuple(primary_batch.prior_control_chunks)
    if prompt_digest is None or not control_digests or not segment_indices or not prior_chunks:
        raise RuntimeError("ADR-207 physical training omitted a source receipt")
    if len(control_digests) != len(segment_indices):
        raise RuntimeError("ADR-207 physical source receipts have different batch axes")
    if sequence_batch_count != 1 or egress_batch is not None or egress_prior_host_steps is not None:
        raise RuntimeError("ADR-207 host may observe exactly one current frame and no egress frame")
    if (
        prior_host_steps_by_batch is None
        or len(prior_host_steps_by_batch) != 1
        or prior_host_steps_by_batch[0] < len(primary_batch.effective_prior_control_chunks)
    ):
        raise RuntimeError("ADR-207 omitted its rank-symmetric prior host schedule")
    if (
        prior_gradient_suffix_steps_by_batch is None
        or len(prior_gradient_suffix_steps_by_batch) != 1
        or not 1
        <= prior_gradient_suffix_steps_by_batch[0]
        <= prior_host_steps_by_batch[0]
    ):
        raise RuntimeError("ADR-207 omitted its bounded prior-gradient schedule")
    if source_batch is None or tuple(source_batch.sample_keys) != tuple(
        primary_batch.routing.sample_keys
    ):
        raise RuntimeError("ADR-207 source and host samples differ")
    if any(
        len(indices) != 5
        or any(right != left + 1 for left, right in zip(indices, indices[1:], strict=False))
        for indices in source_batch.global_indices
    ):
        raise RuntimeError("ADR-207 source supervision is not current plus four raw frames")
    if (
        result.source.sequence.merged.class_logits.shape[1] != 5
        or result.source.current_output.class_logits.shape[1] != 1
        or result.next_state.source_queries is not result.source.current_propagated_queries
    ):
        raise RuntimeError("ADR-207 source transaction crossed its current-frame boundary")
    relation = result.policy.context.relation_output
    if not isinstance(relation, native_relation_type):
        raise TypeError("ADR-207 host did not expose the native object-query relation")
    query_count = relation.relation.query_count
    if query_count != 200 or relation.posterior_rows.shape[1] != query_count:
        raise RuntimeError("ADR-207 host changed the complete 200-query source bank")

    return {
        "active": True,
        "architecture_interface": relation.interface,
        "physical_prompt_selection_sha256": prompt_digest,
        "physical_control_span_sha256": list(control_digests),
        "selected_segment_indices": list(segment_indices),
        "prior_control_chunk_count": len(prior_chunks),
        "prior_control_chunk_token_counts": [chunk.token_count for chunk in prior_chunks],
        "prior_host_steps_by_batch": list(prior_host_steps_by_batch),
        "prior_gradient_suffix_steps_by_batch": list(
            prior_gradient_suffix_steps_by_batch
        ),
        "host_visible_frame_count": 1,
        "source_supervision_frame_count": 5,
        "future_source_frames_visible_to_host": False,
        "source_global_indices": [list(indices) for indices in source_batch.global_indices],
        "source_query_count": query_count,
        "source_auxiliary_read_count": len(
            result.source.sequence.merged.auxiliary_outputs
        ),
        "host_modality_names": [
            stream.name for stream in result.host_batch.modalities.streams
        ],
    }


def _causal_checkpoint_due(*, mode: str, global_step: int) -> bool:
    if mode == "current_frame_branch":
        return global_step == CAUSAL_BRANCH_STEP
    if mode in {"zero_state", "recurrent_state"}:
        return global_step == CAUSAL_ARM_STOP_STEP
    if mode == "none":
        return False
    raise ValueError(f"unknown causal ablation mode: {mode}")


def _objective_posterior_inputs(
    *,
    mode: str,
    prepared: Any,
    torch_module: Any,
) -> tuple[Any | None, Any, tuple[tuple[tuple[str, int], ...], ...]]:
    """Select the sole treatment input without mutating the causal lane bank."""

    if mode in {"current_frame_branch", "zero_state"}:
        return (
            None,
            torch_module.zeros_like(prepared.previous_state_valid),
            tuple(() for _ in prepared.previous_row_bindings),
        )
    if mode in {"none", "recurrent_state"}:
        return (
            prepared.previous_state,
            prepared.previous_state_valid,
            prepared.previous_row_bindings,
        )
    raise ValueError(f"unknown causal ablation mode: {mode}")


def _staged_row_bindings(
    *,
    mode: str,
    observed: tuple[tuple[tuple[str, int], ...], ...],
) -> tuple[tuple[tuple[str, int], ...], ...]:
    if mode in {"current_frame_branch", "zero_state"}:
        return tuple(() for _ in observed)
    if mode in {"none", "recurrent_state"}:
        return observed
    raise ValueError(f"unknown causal ablation mode: {mode}")


def _external_stop_requested(*, run_dir: Path, checkpoint_due: bool) -> bool:
    """Honor an immediate evidence gate or a checkpoint-boundary stop request."""

    return (run_dir / "STOP").is_file() or (
        checkpoint_due and (run_dir / "STOP_AFTER_CHECKPOINT").is_file()
    )


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    source_default = _environment_path("PICF_LINGBOT_NATIVE_SOURCE") or (
        root / CHECKOUT_RELATIVE_PATH
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("fresh", "resume"), required=True)
    parser.add_argument("--source-checkout", type=Path, default=source_default)
    parser.add_argument("--patch", type=Path, default=root / PATCH_RELATIVE_PATH)
    parser.add_argument("--training-config", type=Path)
    parser.add_argument(
        "--robot-config", type=Path, default=root / "configs/lingbot/calvin_robot.yaml"
    )
    parser.add_argument(
        "--data-config", type=Path, default=root / "configs/lingbot/calvin_data.json"
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--dataset-split", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--norm-stats", type=Path, required=True)
    parser.add_argument("--stream-plan", type=Path)
    parser.add_argument("--stream-plan-sha256")
    parser.add_argument("--minimum-future-source-frames", type=int, choices=(4, 8, 16))
    parser.add_argument("--representation-split", type=Path)
    parser.add_argument("--representation-split-sha256")
    parser.add_argument("--evaluation-plan", type=Path)
    parser.add_argument("--evaluation-plan-sha256")
    parser.add_argument("--physical-sidecar-root", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest", type=Path, required=True)
    parser.add_argument("--physical-sidecar-manifest-sha256", required=True)
    parser.add_argument("--physical-visual-acceptance", type=Path, required=True)
    parser.add_argument("--physical-visual-acceptance-sha256", required=True)
    parser.add_argument("--predictive-cache-root", type=Path)
    parser.add_argument("--predictive-cache-build-report", type=Path)
    parser.add_argument("--predictive-cache-build-report-sha256")
    parser.add_argument("--current-grid-cache-root", type=Path)
    parser.add_argument("--current-grid-cache-shard-root", type=Path)
    parser.add_argument("--current-grid-cache-build-report", type=Path)
    parser.add_argument("--current-grid-cache-build-report-sha256")
    parser.add_argument("--future-latent-cache-root", type=Path)
    parser.add_argument("--future-latent-cache-manifest-sha256")
    parser.add_argument("--future-latent-cache-build-report", type=Path)
    parser.add_argument("--future-latent-cache-build-report-sha256")
    parser.add_argument(
        "--future-latent-objective-scale",
        type=float,
        choices=(0.0, 1.0),
        default=1.0,
    )
    parser.add_argument(
        "--dense-evidence-mode",
        choices=DENSE_EVIDENCE_MODES,
        default="none",
    )
    parser.add_argument(
        "--dense-token-bridge",
        choices=NATIVE_MODALITY_BRIDGES,
        default=EXACT_NATIVE_MODALITY_BRIDGE,
    )
    parser.add_argument("--dense-evidence-cache-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--dense-evidence-cache-manifest-sha256",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--dense-evidence-supplement-cache-root",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument(
        "--dense-evidence-supplement-cache-manifest-sha256",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--dense-evidence-subset-view",
        action="store_true",
        help=(
            "Create an authenticated zero-copy exact-record view over the primary "
            "cache when a new stream contract selects the same cached identities."
        ),
    )
    parser.add_argument("--dense-evidence-coverage-plan", type=Path)
    parser.add_argument("--dense-evidence-coverage-plan-sha256")
    parser.add_argument(
        "--videomt-stage-pq-mode",
        choices=VIDEOMT_STAGE_PQ_MODES,
        default=VIDEOMT_STAGE_PQ_DISABLED,
        help=(
            "Run the complete VidEoMT graph on five real causal CALVIN frames. "
            "PQM modes retain all 200 queries and the complete class/mask relation "
            "for full LingBot-host probabilistic composition."
        ),
    )
    parser.add_argument(
        "--videomt-source-update-arm",
        choices=VIDEOMT_SOURCE_UPDATE_ARMS,
        default=VIDEOMT_SOURCE_UPDATE_JOINT,
        help=(
            "Keep the complete native source forward/backward graph fixed while either "
            "applying its optimizer update or discarding that update as a causal control."
        ),
    )
    parser.add_argument("--videomt-checkpoint", type=Path)
    parser.add_argument("--videomt-dinov3-bundle", type=Path)
    parser.add_argument("--videomt-adapted-checkpoint", type=Path)
    parser.add_argument("--videomt-adapted-checkpoint-sha256")
    parser.add_argument(
        "--videomt-idle-placement",
        choices=VIDEOMT_IDLE_PLACEMENTS,
        default=VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
        help=(
            "Keep the frozen donor resident on CUDA, or return it to CPU after each "
            "detached query forward so LingBot optimizer state owns the freed memory."
        ),
    )
    parser.add_argument(
        "--videomt-fsdp2-placement",
        choices=VIDEOMT_FSDP2_PLACEMENTS,
        default=VIDEOMT_FSDP2_PLACEMENT_CUDA_SHARDED,
        help=(
            "Keep trainable VidEoMT FSDP2 parameter, gradient and optimizer shards "
            "on CUDA, or use PyTorch CPUOffloadPolicy without changing the donor graph."
        ),
    )
    parser.add_argument("--wsa-source-root", type=Path)
    parser.add_argument("--wsa-adapted-checkpoint", type=Path)
    parser.add_argument("--wsa-adapted-checkpoint-sha256")
    parser.add_argument("--da3-source-root", type=Path)
    parser.add_argument("--da3-model-dir", type=Path)
    parser.add_argument("--da3-model-checkpoint-sha256")
    parser.add_argument("--da3-model-config-sha256")
    parser.add_argument("--action-backend", choices=ACTION_BACKENDS, default=ACTION_BACKENDS[0])
    parser.add_argument(
        "--wla-host-evidence-arm",
        choices=WLA_HOST_EVIDENCE_ARMS,
        default=WLA_HOST_EVIDENCE_ARMS[0],
    )
    parser.add_argument("--wla-source-root", type=Path)
    parser.add_argument("--wla-pretrained-root", type=Path)
    parser.add_argument(
        "--videomt-shared-query-gradient-diagnostic",
        action="store_true",
        help=(
            "At the first executed step, measure source/action/world gradient Gram "
            "moments at VidEoMT's unchanged released prediction-query input."
        ),
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--load-global-step", type=int, default=0)
    parser.add_argument("--stop-after-step", type=int, default=TOTAL_STEPS)
    parser.add_argument(
        "--engineering-force-omitted-static-step",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--engineering-force-causal-diagnostic-step",
        type=int,
        default=0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--adr221-wsa-edge-diagnostic",
        action="store_true",
        help=(
            "Run the registered measurement-only paired intervention that blocks "
            "WSA future keys from official action queries."
        ),
    )
    parser.add_argument("--acceptance-mode", choices=ACCEPTANCE_MODES, default="none")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument(
        "--attention-implementation",
        choices=LINGBOT_ATTENTION_IMPLEMENTATIONS,
        default=LINGBOT_ATTENTION_EAGER,
        help=(
            "Select the native LingBot joint-attention backend. flex_cached is the "
            "released training backend and avoids materializing the full attention "
            "probability tensor; eager remains the historical comparison default."
        ),
    )
    parser.add_argument(
        "--lingbot-compile-mode",
        choices=LINGBOT_COMPILE_MODES,
        default=LINGBOT_COMPILE_DISABLED,
        help=(
            "Use the released LingBot FSDP-then-whole-model torch.compile path, or "
            "retain the historical uncompiled comparison path."
        ),
    )
    parser.add_argument(
        "--trainable-scope",
        choices=TRAINABLE_SCOPES,
        default=TRAINABLE_SCOPE_FULL_HOST,
        help=(
            "Select the released LingBot trainable scope. frozen-vision-host keeps "
            "the complete visual forward but omits visual-tower gradients and "
            "optimizer state."
        ),
    )
    parser.add_argument("--capacity", type=int, default=16)
    parser.add_argument("--maximum-control-tokens", type=int, default=8)
    parser.add_argument(
        "--prior-gradient-control-tokens",
        type=int,
        default=8,
        help=(
            "Retain autograd only for this trailing control suffix; every earlier "
            "control remains an ordered shared-host burn-in input."
        ),
    )
    parser.add_argument(
        "--posterior-architecture",
        choices=POSTERIOR_ARCHITECTURES,
        default="layerwise_v2",
    )
    parser.add_argument(
        "--picf-architecture-profile",
        choices=PICF_ARCHITECTURE_PROFILES,
        default="legacy",
    )
    parser.add_argument("--task-query-count", type=int, default=0)
    parser.add_argument(
        "--relation-supervision-layers",
        type=_parse_nonnegative_layer_set,
        default=(),
    )
    parser.add_argument("--learning-rate", type=_positive_finite_float, default=1e-4)
    parser.add_argument(
        "--picf-learning-rate-multiplier",
        type=_positive_finite_float,
        default=1.0,
    )
    parser.add_argument(
        "--modality-bridge-learning-rate-multiplier",
        type=_positive_finite_float,
        default=1.0,
    )
    parser.add_argument("--max-grad-norm", type=_positive_finite_float, default=1.0)
    parser.add_argument(
        "--maximum-peak-reserved-gib",
        type=_positive_finite_float,
        default=39.0,
    )
    parser.add_argument(
        "--local-bptt-probability",
        type=float,
        default=PRODUCTION_LOCAL_BPTT_PROBABILITY,
    )
    parser.add_argument("--overshoot-probability", type=float, default=0.05)
    parser.add_argument("--source-mask-probability", type=float, default=0.10)
    parser.add_argument("--maximum-optimizer-lag", type=int, default=8)
    parser.add_argument("--source-mask-token-fraction", type=float, default=0.0625)
    parser.add_argument(
        "--source-prediction-mode",
        choices=("omitted_static", "current_grid"),
        default="omitted_static",
    )
    parser.add_argument(
        "--causal-ablation-mode",
        choices=CAUSAL_ABLATION_MODES,
        default="none",
    )
    parser.add_argument("--entity-weight", type=_nonnegative_finite_float, required=True)
    parser.add_argument("--predictive-weight", type=_nonnegative_finite_float, required=True)
    parser.add_argument("--mask-focal-weight", type=float, default=1.0)
    parser.add_argument("--mask-dice-weight", type=float, default=1.0)
    parser.add_argument("--existence-weight", type=float, default=1.0)
    parser.add_argument("--ownership-weight", type=float, default=1.0)
    parser.add_argument("--predictive-loss-power", type=float, default=1.0)
    parser.add_argument("--minimum-supervised-fraction", type=float, default=0.0)
    parser.add_argument(
        "--fsdp2-placement",
        choices=FSDP2_PLACEMENTS,
        default=FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
    )
    parser.add_argument(
        "--fsdp2-backward-prefetch",
        choices=FSDP2_BACKWARD_PREFETCH_MODES,
        default=FSDP2_BACKWARD_PREFETCH_DEFAULT,
    )
    parser.add_argument(
        "--omitted-static-rematerialization",
        choices=OMITTED_STATIC_REMATERIALIZATION_MODES,
        default=OMITTED_STATIC_REMATERIALIZATION_NONE,
    )
    parser.add_argument(
        "--sequential-factual-gradient-storage",
        choices=FSDP2_FACTUAL_GRADIENT_STORAGE_MODES,
        default=FSDP2_FACTUAL_GRADIENT_GPU,
    )
    parser.add_argument("--cuda-allocator", choices=CUDA_ALLOCATOR_MODES, default="native")
    args = parser.parse_args()
    if args.training_config is None:
        args.training_config = args.source_checkout / "configs/vla/robotwin/robotwin.yaml"
    if not args.load_global_step < args.stop_after_step <= TOTAL_STEPS:
        parser.error("stop-after-step must lie after the input boundary and at most 30000")
    if (
        args.prior_gradient_control_tokens <= 0
        or args.prior_gradient_control_tokens > args.maximum_control_tokens
    ):
        parser.error(
            "prior-gradient-control-tokens must be positive and no larger than "
            "maximum-control-tokens"
        )
    if (
        args.sequential_factual_gradient_storage == FSDP2_FACTUAL_GRADIENT_CPU
        and args.omitted_static_rematerialization
        != OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD
    ):
        parser.error("CPU factual gradient storage requires sequential omitted-static backward")
    try:
        _validate_acceptance_args(args)
        _validate_engineering_smoke_args(args)
        _validate_causal_ablation_args(args)
        _validate_auxiliary_cache_args(args)
        _validate_future_latent_cache_args(args)
        _validate_dense_evidence_args(args)
        _validate_videomt_stage_pq_args(args)
        _validate_adr221_asset_args(args)
        _validate_adr221_wsa_edge_diagnostic_args(args)
        _validate_frozen_stream_args(args)
        _validate_picf_architecture_profile(args)
        _validate_wla_complete_args(args)
    except ValueError as error:
        parser.error(str(error))
    return args


def _cache_manifest(
    root: Path,
    *,
    require_complete_field: bool,
) -> tuple[dict[str, Any], str]:
    path = root / "manifest.json"
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not {"schema", "contract", "shards"}.issubset(value):
        raise ValueError(f"cache manifest is malformed: {root}")
    if require_complete_field and value.get("complete") is not True:
        raise ValueError(f"cache is incomplete: {root}")
    return value, _sha256(path)


def _validate_current_cache_build_binding(
    *,
    report: Mapping[str, Any],
    contract: Any,
    manifest_sha256: str,
    output_root: Path,
) -> None:
    """Bind the reviewed producer recipe to the exact consumed current cache.

    The immutable build report identifies the code that produced the cache.
    It must not be coupled to the patch consumed by the training runner: an
    action-only or optimizer-only consumer change cannot alter precomputed
    DINO targets.  When both current and predictive banks are enabled, their
    producer patches are checked together by ``_cache_producer_patch_sha256``.
    """

    if Path(report.get("output_root", "")).resolve() != output_root.resolve():
        raise ValueError("current-cache build report and consumed root differ")
    expected = {
        "cache_manifest_sha256": manifest_sha256,
        "coverage_sha256": contract.coverage_sha256,
        "expected_record_count": contract.expected_record_count,
        "source_keys_sha256": contract.source_keys_sha256,
        "stream_plan_sha256": contract.stream_plan_sha256,
        "teacher_encoder_digest": contract.encoder_digest,
        "temporal_estimator_sha256": contract.temporal_estimator_sha256,
    }
    mismatched = tuple(
        name for name, expected_value in expected.items() if report.get(name) != expected_value
    )
    if mismatched:
        raise ValueError(
            "current-cache build report is not bound to the consumed manifest/contract: "
            + ", ".join(mismatched)
        )


def _resolve_current_grid_cache_coverage(
    *,
    acceptance_mode: str,
    contract: Any,
    expected: Any,
    temporal_config: Any,
) -> tuple[str, dict[str, Any]]:
    """Resolve an exact cache load identity without conflating route dose with content.

    Current-grid targets are deterministic functions of source RGB and the
    frozen DINO teacher. The posterior-adoption dose arm changes only how often
    an already covered frame is routed through the omitted-static action path.
    It may reuse the registered 10% cache only when the complete source set and
    every content-bearing contract field are identical.
    """

    exact = (
        contract.dataset_tree_sha256 == expected.dataset_tree_sha256
        and contract.stream_plan_sha256 == expected.stream_plan_sha256
        and contract.temporal_estimator_sha256 == temporal_config.digest
        and contract.coverage_sha256 == expected.coverage_sha256
        and contract.source_keys_sha256 == expected.source_keys_sha256
        and contract.expected_record_count == len(expected.source_global_indices)
    )
    if exact:
        return contract.coverage_sha256, {
            "mode": "exact_temporal_and_source_coverage",
            "cache_temporal_estimator_sha256": contract.temporal_estimator_sha256,
            "run_temporal_estimator_sha256": temporal_config.digest,
            "source_keys_sha256": contract.source_keys_sha256,
            "record_count": contract.expected_record_count,
        }

    if acceptance_mode != "posterior-adoption-dose":
        raise RuntimeError("current-grid cache does not cover the exact 30k plan")
    donor_temporal = replace(
        temporal_config,
        source_mask_probability=ADR149_OMITTED_STATIC_PROBABILITY,
    )
    source_set_equivalent = (
        temporal_config.source_mask_probability == POSTERIOR_ADOPTION_DOSE_SOURCE_MASK_PROBABILITY
        and contract.dataset_tree_sha256 == expected.dataset_tree_sha256
        and contract.stream_plan_sha256 == expected.stream_plan_sha256
        and contract.temporal_estimator_sha256 == donor_temporal.digest
        and contract.source_keys_sha256 == expected.source_keys_sha256
        and contract.expected_record_count == len(expected.source_global_indices)
    )
    if not source_set_equivalent:
        raise RuntimeError("posterior-adoption-dose current-grid cache changed content coverage")
    return contract.coverage_sha256, {
        "mode": "exact_source_set_reuse_for_route_dose",
        "cache_temporal_estimator_sha256": contract.temporal_estimator_sha256,
        "run_temporal_estimator_sha256": temporal_config.digest,
        "source_keys_sha256": contract.source_keys_sha256,
        "record_count": contract.expected_record_count,
        "content_invariance": (
            "same source RGB set and frozen teacher; only route sampling probability differs"
        ),
    }


def _rank_rng_digest(value: dict[str, bytes]) -> str:
    return _canonical_digest(
        {name: hashlib.sha256(payload).hexdigest() for name, payload in value.items()}
    )


def _emit_progress(
    event: str,
    *,
    rank: int,
    global_step: int,
    details: dict[str, Any] | None = None,
) -> None:
    """Emit rank-local phase evidence around potentially blocking collectives."""

    allowed = {
        "step_started",
        "batch_ready",
        "objective_started",
        "objective_ready",
        "backward_started",
        "factual_backward_completed",
        "factual_gradients_spilled",
        "factual_gradients_merged",
        "backward_completed",
        "optimizer_completed",
        "step_completed",
        "step_graph_released",
        "causal_diagnostic_variant_started",
        "causal_diagnostic_variant_completed",
        "causal_diagnostic_variant_released",
    }
    if event not in allowed:
        raise ValueError(f"unsupported task-independent progress event: {event}")
    payload: dict[str, Any] = {
        "schema": PROGRESS_SCHEMA,
        "event": event,
        "rank": rank,
        "global_step": global_step,
    }
    if details is not None:
        reserved = set(payload).intersection(details)
        if reserved:
            raise ValueError(
                "task-independent progress details override reserved fields: "
                + ", ".join(sorted(reserved))
            )
        payload.update(details)
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")), flush=True)


def _append_rank_metric(handle: Any, value: dict[str, Any], *, durable: bool) -> None:
    handle.write(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    if durable:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _prepare_rank_metric_journal(
    path: Path,
    *,
    phase: str,
    load_global_step: int,
) -> Any:
    path.parent.mkdir(parents=True, exist_ok=True)
    if phase == "fresh":
        return path.open("x", encoding="utf-8")
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"resume metric journal is absent: {path}")
    retained: list[str] = []
    previous_step = 0
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            payload = json.loads(line)
            step = payload["global_step"]
        except (KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(f"metric journal line {line_number} is malformed") from error
        if isinstance(step, bool) or not isinstance(step, int) or step <= previous_step:
            raise ValueError("metric journal steps must be strictly increasing")
        previous_step = step
        if step <= load_global_step:
            retained.append(line)
    if not retained or json.loads(retained[-1])["global_step"] != load_global_step:
        raise ValueError("metric journal does not reach the restored checkpoint boundary")
    staging = path.with_name(f".{path.name}.resume-{os.getpid()}.tmp")
    if staging.exists() or staging.is_symlink():
        raise FileExistsError(staging)
    _write_text_durable(staging, "\n".join(retained) + "\n")
    os.replace(staging, path)
    _fsync_directory(path.parent)
    return path.open("a", encoding="utf-8")


def _prune_resume_publications(run_dir: Path, *, load_global_step: int) -> None:
    metrics = run_dir / "metrics"
    if metrics.is_dir():
        for path in sorted((*metrics.glob("steps_*.json"), *metrics.glob("tail_*.json"))):
            if path.is_symlink() or not path.is_file():
                raise ValueError("resume metric publication must be a regular file")
            payload = json.loads(path.read_text(encoding="utf-8"))
            end_step = payload.get("end_global_step")
            if isinstance(end_step, bool) or not isinstance(end_step, int):
                raise ValueError("resume metric publication has no integer end step")
            if end_step > load_global_step:
                path.unlink()
        _fsync_directory(metrics)
    for visual_directory_name in ("entity_visuals", "native_videomt_query_visuals"):
        visuals = run_dir / visual_directory_name
        if visuals.is_dir():
            for path in sorted(visuals.glob("step_*")):
                if path.is_symlink() or not path.is_dir():
                    raise ValueError("resume visual publication must be a real directory")
                try:
                    step = int(path.name.removeprefix("step_"))
                except ValueError as error:
                    raise ValueError("resume visual step directory is malformed") from error
                if step > load_global_step:
                    shutil.rmtree(path)
            _fsync_directory(visuals)
    for path in sorted(
        (
            *run_dir.glob("run_summary_step_*.json"),
            *run_dir.glob("action_evaluation_curve_step_*.json"),
        )
    ):
        if path.is_symlink() or not path.is_file():
            raise ValueError("resume summary publication must be a regular file")
        try:
            prefix = (
                "run_summary_step_"
                if path.name.startswith("run_summary_step_")
                else "action_evaluation_curve_step_"
            )
            step = int(path.stem.removeprefix(prefix))
        except ValueError as error:
            raise ValueError("resume summary publication has a malformed step") from error
        if step > load_global_step:
            path.unlink()
    for directory_name in (
        "causal_diagnostics",
        "layerwise_causal_diagnostics",
        "two_pass_filter_diagnostics",
        "action_evaluations",
        "heldout_entity_evaluation",
        "heldout_entity_evaluations",
        "heldout_native_videomt_anchor_evaluation",
        "heldout_native_videomt_anchor_evaluations",
        "native_videomt_modality_interventions",
    ):
        root = run_dir / directory_name
        if not root.is_dir():
            continue
        for path in sorted(root.glob("step_*")):
            if path.is_symlink() or not path.is_dir():
                raise ValueError("resume diagnostic publication must be a real directory")
            try:
                step = int(path.name.removeprefix("step_"))
            except ValueError as error:
                raise ValueError("resume diagnostic step directory is malformed") from error
            if step > load_global_step:
                shutil.rmtree(path)
        _fsync_directory(root)
    _fsync_directory(run_dir)


def _publish_metric_window(
    *,
    run_dir: Path,
    global_step: int,
    rank_window: list[dict[str, Any]],
    rank: int,
    dist: Any,
) -> None:
    if len(rank_window) != METRICS_EVERY:
        raise RuntimeError("a metric window must contain exactly 100 local steps")
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, {"rank": rank, "steps": rank_window})
    publish_error: list[str | None] = [None]
    if rank == 0:
        try:
            start = global_step - METRICS_EVERY + 1
            payload = {
                "schema": METRIC_WINDOW_SCHEMA,
                "start_global_step": start,
                "end_global_step": global_step,
                "rank_reports": gathered,
            }
            output = run_dir / "metrics" / f"steps_{start:08d}_{global_step:08d}.json"
            output.parent.mkdir(parents=True, exist_ok=True)
            write_text_durable_exclusive(
                output,
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
            )
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"100-step metric publication failed: {publish_error[0]}")
    dist.barrier()


def _distributed_fsdp2_parameter_layout_contract(
    module: Any,
    *,
    rank: int,
    dist: Any,
) -> dict[str, int | str]:
    """Require one identical global FSDP2 parameter layout on every rank."""

    local_manifest = None
    local_error = None
    try:
        local_manifest = fsdp2_parameter_layout_manifest(module)
    except BaseException as error:
        local_error = error
    failures = _distributed_pre_backward_failures(
        local_error,
        rank=rank,
        expected_world_size=WORLD_SIZE,
        dist=dist,
    )
    if failures:
        if local_error is not None:
            raise local_error
        raise RuntimeError(f"peer FSDP2 parameter manifest failed: {failures}")
    if local_manifest is None:
        raise RuntimeError("FSDP2 parameter manifest returned no result")
    gathered: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered, {"rank": rank, **local_manifest})
    expected_fields = {
        "rank",
        "manifest_sha256",
        "parameter_count",
        "trainable_parameter_count",
    }
    reference = None
    for expected_rank, item in enumerate(gathered):
        if (
            not isinstance(item, dict)
            or set(item) != expected_fields
            or item["rank"] != expected_rank
        ):
            raise RuntimeError("FSDP2 parameter manifest exchange was malformed")
        rank_contract = {key: value for key, value in item.items() if key != "rank"}
        if reference is None:
            reference = rank_contract
        elif rank_contract != reference:
            raise RuntimeError("distributed ranks disagree on the FSDP2 parameter layout")
    if reference is None:
        raise RuntimeError("FSDP2 parameter manifest exchange returned no ranks")
    return reference


def _distributed_complete_source_gradient_metrics(
    source_model: Any,
    *,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Audit every complete-source gradient before either optimizer may step."""

    missing: list[str] = []
    nonfinite: list[str] = []
    local_square_sum = 0.0
    local_maximum = 0.0
    trainable_tensors = 0
    gradient_tensors = 0
    for name, parameter in source_model.named_parameters():
        if not parameter.requires_grad:
            continue
        trainable_tensors += 1
        gradient = parameter.grad
        if gradient is None:
            missing.append(name)
            continue
        local = gradient.to_local() if callable(getattr(gradient, "to_local", None)) else gradient
        if not torch_module.is_tensor(local):
            raise TypeError("complete source gradient local shard is not a tensor")
        gradient_tensors += 1
        value = local.detach().float()
        if not bool(torch_module.isfinite(value).all()):
            nonfinite.append(name)
            continue
        local_square_sum += float(value.square().sum().item())
        if value.numel():
            local_maximum = max(local_maximum, float(value.abs().max().item()))
    local_failure = bool(missing or nonfinite or gradient_tensors != trainable_tensors)
    failure = torch_module.tensor(int(local_failure), device=device, dtype=torch_module.int64)
    square_sum = torch_module.tensor(local_square_sum, device=device, dtype=torch_module.float64)
    maximum = torch_module.tensor(local_maximum, device=device, dtype=torch_module.float64)
    dist.all_reduce(failure, op=dist.ReduceOp.MAX)
    dist.all_reduce(square_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    gathered_failures: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(
        gathered_failures,
        {
            "missing": missing[:16],
            "nonfinite": nonfinite[:16],
            "trainable_tensors": trainable_tensors,
            "gradient_tensors": gradient_tensors,
        },
    )
    return {
        "all_finite_and_present": not bool(failure.item()),
        "global_l2_norm": float(square_sum.sqrt().item()),
        "global_max_abs": float(maximum.item()),
        "rank_failures": gathered_failures,
    }


def _distributed_shared_query_gradient_report(
    moments: Any,
    *,
    dist: Any,
    torch_module: Any,
) -> dict[str, Any]:
    """Reduce exact shared-surface Gram moments without moving a gradient vector."""

    squared_names = tuple(sorted(moments.squared_norms))
    pair_names = tuple(sorted(moments.pairwise_dots))
    if squared_names != ("action", "host", "source", "world"):
        raise RuntimeError("shared-query gradient squared-norm inventory changed")
    expected_pairs = tuple(
        f"{left}__{right}"
        for index, left in enumerate(squared_names)
        for right in squared_names[index + 1 :]
    )
    if pair_names != expected_pairs:
        raise RuntimeError("shared-query gradient dot-product inventory changed")
    values = [
        *(moments.squared_norms[name] for name in squared_names),
        *(moments.pairwise_dots[name] for name in pair_names),
        next(iter(moments.squared_norms.values())).new_tensor(float(moments.elements)),
    ]
    packed = torch_module.stack(values)
    dist.all_reduce(packed, op=dist.ReduceOp.SUM)
    reduced = packed.detach().cpu().tolist()
    squared = dict(zip(squared_names, reduced[: len(squared_names)], strict=True))
    dots = dict(
        zip(
            pair_names,
            reduced[len(squared_names) : len(squared_names) + len(pair_names)],
            strict=True,
        )
    )
    norms = {name: math.sqrt(max(float(value), 0.0)) for name, value in squared.items()}
    cosines: dict[str, float | None] = {}
    for name, dot in dots.items():
        left, right = name.split("__", 1)
        denominator = norms[left] * norms[right]
        cosines[name] = (
            None
            if denominator == 0.0
            else max(-1.0, min(1.0, float(dot) / denominator))
        )
    return {
        "schema": "picf-next.videomt-shared-query-gradient/v1",
        "surface": "released_videomt_final_prediction_query_input_current_frame",
        "objective_names": list(squared_names),
        "global_elements": int(reduced[-1]),
        "global_squared_norms": {name: float(squared[name]) for name in squared_names},
        "global_norms": norms,
        "global_pairwise_dots": {name: float(dots[name]) for name in pair_names},
        "global_pairwise_cosines": cosines,
        "source_to_action_norm_ratio": norms["source"] / max(norms["action"], 1.0e-30),
        "source_to_host_norm_ratio": norms["source"] / max(norms["host"], 1.0e-30),
    }


def _joint_host_source_optimizer_attempt(
    *,
    policy: Any,
    host_optimizer: Any,
    source_model: Any,
    source_optimizer: Any,
    source_scheduler: Any,
    host_auxiliary_scheduler: Any | None,
    source_update_arm: str,
    global_step: int,
    max_grad_norm: float,
    device: Any,
    dist: Any,
    torch_module: Any,
) -> tuple[int | None, dict[str, Any]]:
    """Apply the registered host/source update intervention after one shared backward."""

    if source_update_arm not in VIDEOMT_SOURCE_UPDATE_ARMS:
        raise ValueError("unknown VidEoMT source-update arm")

    source_metrics = _distributed_complete_source_gradient_metrics(
        source_model,
        device=device,
        dist=dist,
        torch_module=torch_module,
    )
    metrics: dict[str, Any] = {f"source_{key}": value for key, value in source_metrics.items()}
    if not source_metrics["all_finite_and_present"]:
        host_optimizer.zero_grad(set_to_none=True)
        source_optimizer.zero_grad(set_to_none=True)
        return None, metrics
    source_preclip = clip_lingbot_distributed_l2_grad_norm_(
        source_model.parameters(),
        max_grad_norm,
        device=device,
        dist_module=dist,
        torch_module=torch_module,
        error_if_nonfinite=True,
    )
    metrics["source_preclip_global_norm"] = float(source_preclip)
    successful, host_metrics = _optimizer_attempt(
        policy=policy,
        optimizer=host_optimizer,
        global_step=global_step,
        max_grad_norm=max_grad_norm,
        device=device,
        dist=dist,
        torch_module=torch_module,
    )
    metrics.update({f"host_{key}": value for key, value in host_metrics.items()})
    if successful is None:
        source_optimizer.zero_grad(set_to_none=True)
        return None, metrics
    try:
        if source_update_arm == VIDEOMT_SOURCE_UPDATE_JOINT:
            source_optimizer.step()
            source_scheduler.step()
        if host_auxiliary_scheduler is not None:
            host_auxiliary_scheduler.step()
    finally:
        source_optimizer.zero_grad(set_to_none=True)
    metrics["source_update_arm"] = source_update_arm
    metrics["source_update_applied"] = (
        source_update_arm == VIDEOMT_SOURCE_UPDATE_JOINT
    )
    metrics["source_learning_rate_min"] = min(
        float(group["lr"]) for group in source_optimizer.param_groups
    )
    metrics["source_learning_rate_max"] = max(
        float(group["lr"]) for group in source_optimizer.param_groups
    )
    metrics["source_scheduler_step"] = int(source_scheduler.last_epoch)
    if host_auxiliary_scheduler is not None:
        metrics["host_auxiliary_scheduler_step"] = int(
            host_auxiliary_scheduler.last_epoch
        )
        metrics["host_auxiliary_learning_rate_min"] = min(
            float(value) for value in host_auxiliary_scheduler.get_last_lr()
        )
        metrics["host_auxiliary_learning_rate_max"] = max(
            float(value) for value in host_auxiliary_scheduler.get_last_lr()
        )
    return successful, metrics


def _checkpoint_extra(
    *,
    policy: Any,
    optimizer: Any,
    bank: Any,
    global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    stream_plan_sha256: str,
    temporal_sha256: str,
    predictive_manifest_sha256: str,
    current_manifest_sha256: str,
    evidence_sha256: str,
    execution_contract_sha256: str,
    torch_module: Any,
    numpy_module: Any,
    device: Any,
    rank: int,
    source_model: Any | None = None,
    source_optimizer: Any | None = None,
    source_scheduler: Any | None = None,
    wsa_scheduler: Any | None = None,
    wla_scheduler: Any | None = None,
) -> tuple[dict[str, Any], dict[str, str]]:
    source_items = (source_model, source_optimizer, source_scheduler)
    if any(item is not None for item in source_items) and not all(
        item is not None for item in source_items
    ):
        raise ValueError("joint checkpoint requires source model, optimizer and scheduler together")
    rng = _capture_rank_rng(torch_module, numpy_module, device=device)
    lane_snapshot = bank.serialize()
    boundary = _checkpoint_boundary(
        model=policy,
        optimizer=optimizer,
        lane_snapshot=lane_snapshot,
        rank_rng_state=rng,
        torch_module=torch_module,
    )
    source_optimizer_summary = None
    if source_model is not None:
        source_boundary = _checkpoint_boundary(
            model=source_model,
            optimizer=source_optimizer,
            lane_snapshot=lane_snapshot,
            rank_rng_state=rng,
            torch_module=torch_module,
        )
        boundary = {
            **boundary,
            "source_model_local_state_sha256": source_boundary["model_local_state_sha256"],
            "source_optimizer_local_state_sha256": source_boundary[
                "optimizer_local_state_sha256"
            ],
        }
        source_optimizer_summary = _validate_optimizer_state(
            source_optimizer,
            torch_module,
            expected_step=global_step,
        )
        if int(getattr(source_scheduler, "last_epoch", -1)) != global_step:
            raise ValueError("source scheduler and joint optimizer step differ")
    optimizer_summary = _validate_optimizer_state(
        optimizer,
        torch_module,
        expected_step=global_step,
    )
    if (
        wsa_scheduler is not None
        and int(getattr(wsa_scheduler, "last_epoch", -1)) != global_step
    ):
        raise ValueError("WSA scheduler and joint optimizer step differ")
    if (
        wla_scheduler is not None
        and int(getattr(wla_scheduler, "last_epoch", -1)) != global_step
    ):
        raise ValueError("WLA scheduler and optimizer step differ")
    extra = {
        "schema": CHECKPOINT_SCHEMA,
        "rank": rank,
        "world_size": WORLD_SIZE,
        "global_step": global_step,
        "next_optimizer_step": global_step,
        "implementation_sha256": implementation_sha256,
        "model_family_sha256": model_family_sha256,
        "stream_plan_sha256": stream_plan_sha256,
        "temporal_estimator_sha256": temporal_sha256,
        "predictive_cache_manifest_sha256": predictive_manifest_sha256,
        "current_grid_cache_manifest_sha256": current_manifest_sha256,
        "evidence_sha256": evidence_sha256,
        "execution_contract_sha256": execution_contract_sha256,
        "lane_snapshot": lane_snapshot,
        "rank_rng_state": rng,
        "rank_rng_sha256": _rank_rng_digest(rng),
        "boundary_sha256": boundary,
        "joint_source_optimizer_summary": source_optimizer_summary,
        "joint_source_scheduler_step": (
            None if source_scheduler is None else int(source_scheduler.last_epoch)
        ),
        "wsa_scheduler_step": (
            None if wsa_scheduler is None else int(wsa_scheduler.last_epoch)
        ),
        "wla_scheduler_step": (
            None if wla_scheduler is None else int(wla_scheduler.last_epoch)
        ),
        **optimizer_summary,
    }
    return extra, boundary


def _validate_resume_extra(
    value: object,
    *,
    rank: int,
    global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    stream_plan_sha256: str,
    temporal_sha256: str,
    predictive_manifest_sha256: str,
    current_manifest_sha256: str,
    evidence_sha256: str,
    execution_contract_sha256: str,
    joint_source_active: bool = False,
    wsa_active: bool = False,
    wla_active: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("task-independent checkpoint extra state differs")
    expected = {
        "rank": rank,
        "world_size": WORLD_SIZE,
        "global_step": global_step,
        "next_optimizer_step": global_step,
        "implementation_sha256": implementation_sha256,
        "model_family_sha256": model_family_sha256,
        "stream_plan_sha256": stream_plan_sha256,
        "temporal_estimator_sha256": temporal_sha256,
        "predictive_cache_manifest_sha256": predictive_manifest_sha256,
        "current_grid_cache_manifest_sha256": current_manifest_sha256,
        "evidence_sha256": evidence_sha256,
        "execution_contract_sha256": execution_contract_sha256,
    }
    if any(value.get(name) != measured for name, measured in expected.items()):
        raise ValueError("task-independent checkpoint provenance differs")
    for name in ("lane_snapshot", "rank_rng_state", "boundary_sha256"):
        if name not in value:
            raise ValueError(f"task-independent checkpoint omits {name}")
    if joint_source_active:
        if value.get("joint_source_optimizer_summary") is None:
            raise ValueError("joint checkpoint omits the source optimizer summary")
        if value.get("joint_source_scheduler_step") != global_step:
            raise ValueError("joint checkpoint source scheduler continuation differs")
        boundary = value["boundary_sha256"]
        if not isinstance(boundary, dict) or not {
            "source_model_local_state_sha256",
            "source_optimizer_local_state_sha256",
        }.issubset(boundary):
            raise ValueError("joint checkpoint boundary omits source state")
    elif value.get("joint_source_optimizer_summary") is not None:
        raise ValueError("non-joint checkpoint unexpectedly contains source optimizer state")
    if wsa_active:
        if value.get("wsa_scheduler_step") != global_step:
            raise ValueError("joint checkpoint WSA scheduler continuation differs")
    elif value.get("wsa_scheduler_step") is not None:
        raise ValueError("non-WSA checkpoint unexpectedly contains WSA scheduler state")
    if wla_active:
        if value.get("wla_scheduler_step") != global_step:
            raise ValueError("checkpoint WLA scheduler continuation differs")
    elif value.get("wla_scheduler_step") is not None:
        raise ValueError("non-WLA checkpoint unexpectedly contains WLA scheduler state")
    return value


def _publish_checkpoint(
    *,
    args: argparse.Namespace,
    checkpointer: Any,
    policy: Any,
    optimizer: Any,
    bank: Any,
    global_step: int,
    implementation_sha256: str,
    model_family_sha256: str,
    stream_plan_sha256: str,
    temporal_sha256: str,
    predictive_manifest_sha256: str,
    current_manifest_sha256: str,
    evidence_sha256: str,
    execution_contract_sha256: str,
    torch_module: Any,
    numpy_module: Any,
    device: Any,
    rank: int,
    dist: Any,
    source_model: Any | None = None,
    source_optimizer: Any | None = None,
    source_scheduler: Any | None = None,
    wsa_scheduler: Any | None = None,
    wla_scheduler: Any | None = None,
) -> None:
    extra, boundary = _checkpoint_extra(
        policy=policy,
        optimizer=optimizer,
        bank=bank,
        global_step=global_step,
        implementation_sha256=implementation_sha256,
        model_family_sha256=model_family_sha256,
        stream_plan_sha256=stream_plan_sha256,
        temporal_sha256=temporal_sha256,
        predictive_manifest_sha256=predictive_manifest_sha256,
        current_manifest_sha256=current_manifest_sha256,
        evidence_sha256=evidence_sha256,
        execution_contract_sha256=execution_contract_sha256,
        torch_module=torch_module,
        numpy_module=numpy_module,
        device=device,
        rank=rank,
        source_model=source_model,
        source_optimizer=source_optimizer,
        source_scheduler=source_scheduler,
        wsa_scheduler=wsa_scheduler,
        wla_scheduler=wla_scheduler,
    )
    gathered_boundaries: list[Any] = [None for _ in range(WORLD_SIZE)]
    dist.all_gather_object(gathered_boundaries, {"rank": rank, "boundary": boundary})
    root = args.run_dir / "checkpoints"
    output = root / f"global_step_{global_step}"
    staging = root / f".global_step_{global_step}.incomplete"
    preflight_error: list[str | None] = [None]
    if rank == 0:
        try:
            root.mkdir(parents=True, exist_ok=True)
            require_checkpoint_write_capacity(root)
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            if staging.is_symlink():
                raise ValueError("checkpoint staging path cannot be a symbolic link")
            if staging.exists():
                if not staging.is_dir():
                    raise ValueError("checkpoint staging path is not a directory")
                shutil.rmtree(staging)
                _fsync_directory(root)
        except BaseException as error:
            preflight_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(preflight_error, src=0)
    if preflight_error[0] is not None:
        raise RuntimeError(f"checkpoint preflight failed: {preflight_error[0]}")
    checkpoint_state = {"model": policy, "optimizer": optimizer, "extra_state": extra}
    if source_model is not None:
        checkpoint_state.update(
            {
                "videomt_model": source_model,
                "videomt_optimizer": source_optimizer,
                "videomt_scheduler": source_scheduler,
            }
        )
    if wsa_scheduler is not None:
        checkpoint_state["wsa_scheduler"] = wsa_scheduler
    if wla_scheduler is not None:
        checkpoint_state["wla_scheduler"] = wla_scheduler
    checkpointer.save(str(staging), checkpoint_state, global_steps=None)
    dist.barrier()
    publish_error: list[str | None] = [None]
    if rank == 0:
        try:
            report = {
                "schema": CHECKPOINT_SCHEMA,
                "global_step": global_step,
                "status": "PASS",
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "stream_plan_sha256": stream_plan_sha256,
                "temporal_estimator_sha256": temporal_sha256,
                "evidence_sha256": evidence_sha256,
                "execution_contract_sha256": execution_contract_sha256,
                "rank_boundaries": gathered_boundaries,
                "joint_source_active": source_model is not None,
                "wsa_active": wsa_scheduler is not None,
                "wla_active": wla_scheduler is not None,
            }
            _write_text_durable(
                staging / "task_independent_checkpoint.json",
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            _fsync_tree(staging)
            os.replace(staging, output)
            _fsync_directory(root)
            expired_step = global_step - 2 * CHECKPOINT_EVERY
            expired = root / f"global_step_{expired_step}"
            if expired.is_symlink():
                raise ValueError("expired checkpoint cannot be a symbolic link")
            if expired_step != CHECKPOINT_EVERY and expired.is_dir() and expired.parent == root:
                shutil.rmtree(expired)
                _fsync_directory(root)
        except BaseException as error:
            publish_error[0] = f"{type(error).__name__}: {error}"
    dist.broadcast_object_list(publish_error, src=0)
    if publish_error[0] is not None:
        raise RuntimeError(f"checkpoint publication failed: {publish_error[0]}")
    dist.barrier()


def main() -> None:
    args = _parse_args()
    _validate_production_temporal_estimator(args.local_bptt_probability)
    cadence = ProductionCadence()
    root = Path(__file__).resolve().parents[1]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.run_dir = require_persistent_run_root(args.run_dir)
    require_checkpoint_write_capacity(args.run_dir)
    if args.cuda_allocator != _BOOTSTRAPPED_CUDA_ALLOCATOR:
        raise RuntimeError("CUDA allocator bootstrap differs from the requested mode")
    validate_fsdp2_placement(args.fsdp2_placement)
    if _future_latent_alignment_active(args):
        frozen_vision_offload = (
            args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD
        )
        trainable_vision_offload = (
            args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD
        )
        if trainable_vision_offload:
            patch_report = verify_selective_trainable_vision_offload(
                root=root,
                checkout=args.source_checkout,
                check_apply=True,
            )
        elif frozen_vision_offload:
            patch_report = verify_selective_frozen_vision_offload(
                root=root,
                checkout=args.source_checkout,
                check_apply=True,
            )
        else:
            patch_report = verify_muon_collective_hotfix(
                root=root,
                checkout=args.source_checkout,
                check_apply=True,
            )
        flare_report = validate_prepared_flare_overlay(
            root=root,
            checkout=args.source_checkout,
            require_muon_collective_hotfix=True,
            require_frozen_vision_offload=frozen_vision_offload,
            require_trainable_vision_offload=trainable_vision_offload,
        )
        source_overlay_sha256 = _canonical_digest(
            {
                "native_patch_sha256": patch_report["native_patch_sha256"],
                "runtime_hotfix_sha256": patch_report["runtime_hotfix_sha256"],
                "muon_mixed_device_megabatch_sha256": patch_report.get(
                    "muon_mixed_device_megabatch_sha256"
                ),
                "flare_patch_sha256": flare_report["flare_patch_sha256"],
                "flare_model_sha256": flare_report["model_sha256"],
                "selective_frozen_vision_offload_sha256": patch_report.get(
                    "selective_frozen_vision_offload_sha256"
                ),
                "frozen_visual_root_offload_sha256": patch_report.get(
                    "frozen_visual_root_offload_sha256"
                ),
                "selective_trainable_vision_offload_sha256": patch_report.get(
                    "selective_trainable_vision_offload_sha256"
                ),
            }
        )
    elif _videomt_stage_pq_active(args):
        if _adr221_full_source_wsa_active(args) or _wla_complete_active(args):
            if (
                args.fsdp2_placement
                == FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD
            ):
                if _wla_complete_active(args):
                    patch_report = verify_selective_trainable_vision_with_vlm_selective_class_cpu_offload(
                        root=root,
                        checkout=args.source_checkout,
                        check_apply=True,
                    )
                    validate_prepared_native_source_with_trainable_vision_and_vlm_selective_class_offload(
                        checkout=args.source_checkout,
                        patch_path=args.patch,
                        hotfix_path=root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
                        frozen_offload_patch_path=(
                            root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                        visual_root_patch_path=(
                            root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
                        ),
                        trainable_offload_patch_path=(
                            root / SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                        mixed_device_muon_patch_path=(
                            root / MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH
                        ),
                        selective_class_patch_path=(
                            root
                            / SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                        vlm_selective_class_patch_path=(
                            root
                            / VLM_SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                    )
                else:
                    patch_report = verify_selective_trainable_vision_with_selective_class_cpu_offload(
                        root=root,
                        checkout=args.source_checkout,
                        check_apply=True,
                    )
                    validate_prepared_native_source_with_trainable_vision_and_selective_class_offload(
                        checkout=args.source_checkout,
                        patch_path=args.patch,
                        hotfix_path=root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
                        frozen_offload_patch_path=(
                            root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                        visual_root_patch_path=(
                            root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH
                        ),
                        trainable_offload_patch_path=(
                            root / SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                        mixed_device_muon_patch_path=(
                            root / MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH
                        ),
                        selective_class_patch_path=(
                            root
                            / SELECTIVE_CLASS_AFTER_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
                        ),
                    )
                source_overlay_sha256 = _canonical_digest(
                    {
                        "native_patch_sha256": patch_report["native_patch_sha256"],
                        "runtime_hotfix_sha256": patch_report[
                            "runtime_hotfix_sha256"
                        ],
                        "muon_mixed_device_megabatch_sha256": patch_report[
                            "muon_mixed_device_megabatch_sha256"
                        ],
                        "selective_frozen_vision_offload_sha256": patch_report[
                            "selective_frozen_vision_offload_sha256"
                        ],
                        "frozen_visual_root_offload_sha256": patch_report[
                            "frozen_visual_root_offload_sha256"
                        ],
                        "selective_trainable_vision_offload_sha256": patch_report[
                            "selective_trainable_vision_offload_sha256"
                        ],
                        "selective_class_after_trainable_vision_offload_sha256": (
                            patch_report[
                                "selective_class_after_trainable_vision_offload_sha256"
                            ]
                        ),
                        **(
                            {
                                "vlm_selective_class_after_trainable_vision_offload_sha256": (
                                    patch_report[
                                        "vlm_selective_class_after_trainable_vision_offload_sha256"
                                    ]
                                )
                            }
                            if _wla_complete_active(args)
                            else {}
                        ),
                    }
                )
            elif args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_OFFLOAD:
                patch_report = verify_selective_class_cpu_offload(
                    root=root,
                    checkout=args.source_checkout,
                    check_apply=True,
                )
                validate_prepared_native_source_with_selective_class_cpu_offload(
                    checkout=args.source_checkout,
                    patch_path=args.patch,
                    hotfix_path=root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
                    offload_patch_path=(
                        root / SELECTIVE_CLASS_CPU_OFFLOAD_RELATIVE_PATH
                    ),
                )
                source_overlay_sha256 = _canonical_digest(
                    {
                        "native_patch_sha256": patch_report["native_patch_sha256"],
                        "runtime_hotfix_sha256": patch_report[
                            "runtime_hotfix_sha256"
                        ],
                        "selective_class_cpu_offload_sha256": patch_report[
                            "selective_class_cpu_offload_sha256"
                        ],
                    }
                )
            else:
                raise ValueError(
                    "full-source WSA/WLA requires selective embedding or "
                    "trainable-vision plus selective-class CPU offload"
                )
        elif (
            args.fsdp2_placement
            == FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD
        ):
            patch_report = verify_selective_trainable_vision_offload(
                root=root,
                checkout=args.source_checkout,
                check_apply=True,
            )
            validate_prepared_native_source_with_selective_trainable_vision_offload(
                checkout=args.source_checkout,
                patch_path=args.patch,
                hotfix_path=root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
                frozen_offload_patch_path=(
                    root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH
                ),
                visual_root_patch_path=root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH,
                trainable_offload_patch_path=(
                    root / SELECTIVE_TRAINABLE_VISION_OFFLOAD_RELATIVE_PATH
                ),
                mixed_device_muon_patch_path=(
                    root / MUON_MIXED_DEVICE_MEGABATCH_RELATIVE_PATH
                ),
            )
            source_overlay_sha256 = _canonical_digest(
                {
                    "native_patch_sha256": patch_report["native_patch_sha256"],
                    "runtime_hotfix_sha256": patch_report["runtime_hotfix_sha256"],
                    "muon_mixed_device_megabatch_sha256": patch_report[
                        "muon_mixed_device_megabatch_sha256"
                    ],
                    "selective_frozen_vision_offload_sha256": patch_report[
                        "selective_frozen_vision_offload_sha256"
                    ],
                    "frozen_visual_root_offload_sha256": patch_report[
                        "frozen_visual_root_offload_sha256"
                    ],
                    "selective_trainable_vision_offload_sha256": patch_report[
                        "selective_trainable_vision_offload_sha256"
                    ],
                }
            )
        elif args.fsdp2_placement == FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD:
            patch_report = verify_selective_frozen_vision_offload(
                root=root,
                checkout=args.source_checkout,
                check_apply=True,
            )
            validate_prepared_native_source_with_selective_frozen_vision_offload(
                checkout=args.source_checkout,
                patch_path=args.patch,
                hotfix_path=root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
                offload_patch_path=root / SELECTIVE_FROZEN_VISION_OFFLOAD_RELATIVE_PATH,
                visual_root_patch_path=root / FROZEN_VISUAL_ROOT_OFFLOAD_RELATIVE_PATH,
            )
            source_overlay_sha256 = _canonical_digest(
                {
                    "native_patch_sha256": patch_report["native_patch_sha256"],
                    "runtime_hotfix_sha256": patch_report["runtime_hotfix_sha256"],
                    "selective_frozen_vision_offload_sha256": patch_report[
                        "selective_frozen_vision_offload_sha256"
                    ],
                    "frozen_visual_root_offload_sha256": patch_report[
                        "frozen_visual_root_offload_sha256"
                    ],
                }
            )
        else:
            patch_report = verify_muon_collective_hotfix(
                root=root,
                checkout=args.source_checkout,
                check_apply=True,
            )
            validate_prepared_native_source_with_muon_collective_hotfix(
                checkout=args.source_checkout,
                patch_path=args.patch,
                hotfix_path=root / MUON_COLLECTIVE_HOTFIX_RELATIVE_PATH,
            )
            source_overlay_sha256 = _canonical_digest(
                {
                    "native_patch_sha256": patch_report["native_patch_sha256"],
                    "runtime_hotfix_sha256": patch_report["runtime_hotfix_sha256"],
                }
            )
    else:
        patch_report = verify_native_patch(
            root=root,
            checkout=args.source_checkout,
            check_apply=True,
        )
        validate_prepared_native_source(checkout=args.source_checkout, patch_path=args.patch)
        source_overlay_sha256 = patch_report["patch_sha256"]
    physical_visual_acceptance = load_calvin_physical_visual_acceptance(
        args.physical_visual_acceptance,
        expected_sha256=args.physical_visual_acceptance_sha256,
        expected_dataset_manifest_sha256=_sha256(args.dataset_manifest),
        expected_sidecar_manifest_sha256=args.physical_sidecar_manifest_sha256,
    )
    _validate_training_supervision_policy(
        physical_visual_acceptance=physical_visual_acceptance,
        minimum_supervised_fraction=args.minimum_supervised_fraction,
    )
    predictive_build_report: dict[str, Any] | None = None
    current_grid_build_report: dict[str, Any] | None = None
    if _predictive_assets_required(args):
        predictive_build_report = load_predictive_build_report(
            args.predictive_cache_build_report,
            expected_sha256=args.predictive_cache_build_report_sha256,
        )
        if Path(predictive_build_report["output_root"]).resolve() != (
            args.predictive_cache_root.resolve()
        ):
            raise ValueError("predictive build report and cache root differ")
        if predictive_build_report["physical_visual_acceptance_sha256"] != (
            args.physical_visual_acceptance_sha256
        ):
            raise ValueError("predictive cache predates the accepted physical visual review")
        predictive_build_report_sha256 = args.predictive_cache_build_report_sha256
    else:
        predictive_build_report_sha256 = _disabled_auxiliary_digest("predictive")
    if _current_correction_assets_required(args):
        current_grid_build_report = load_current_grid_build_report(
            args.current_grid_cache_build_report,
            expected_sha256=args.current_grid_cache_build_report_sha256,
        )
        if Path(current_grid_build_report["output_root"]).resolve() != (
            args.current_grid_cache_root.resolve()
        ):
            raise ValueError("current-grid build report and cache root differ")
        if current_grid_build_report["physical_visual_acceptance_sha256"] != (
            args.physical_visual_acceptance_sha256
        ):
            raise ValueError("current cache predates the accepted physical visual review")
        current_grid_build_report_sha256 = args.current_grid_cache_build_report_sha256
    else:
        current_grid_build_report_sha256 = _disabled_auxiliary_digest("current_grid")
    future_latent_build_report: dict[str, Any] | None = None
    if _future_latent_alignment_active(args):
        if (
            args.future_latent_cache_build_report.is_symlink()
            or not args.future_latent_cache_build_report.is_file()
            or _sha256(args.future_latent_cache_build_report)
            != args.future_latent_cache_build_report_sha256
        ):
            raise ValueError("FLARE build report is absent, linked, or has the wrong digest")
        payload = json.loads(args.future_latent_cache_build_report.read_text("ascii"))
        if not isinstance(payload, dict) or payload.get("schema") != (
            "picf-next.flare-future-target-cache-build.v1"
        ):
            raise ValueError("FLARE build report schema differs")
        future_latent_build_report = payload
        if Path(payload.get("output_root", "")).resolve() != (
            args.future_latent_cache_root.resolve()
        ):
            raise ValueError("FLARE build report and cache root differ")
        if payload.get("cache_manifest_sha256") != (
            args.future_latent_cache_manifest_sha256
        ):
            raise ValueError("FLARE build report and cache manifest differ")
        future_latent_build_report_sha256 = args.future_latent_cache_build_report_sha256
    else:
        future_latent_build_report_sha256 = _disabled_auxiliary_digest(
            "future_latent_cache"
        )
    if predictive_build_report is not None and current_grid_build_report is not None:
        _cache_producer_patch_sha256(
            predictive_build_report,
            current_grid_build_report,
        )
    frozen_stream_enabled = _validate_frozen_stream_args(args)
    if frozen_stream_enabled:
        for path, expected in (
            (args.stream_plan, args.stream_plan_sha256),
            (args.representation_split, args.representation_split_sha256),
            (args.evaluation_plan, args.evaluation_plan_sha256),
        ):
            if path.is_symlink() or not path.is_file():
                raise FileNotFoundError(path)
            if _sha256(path) != expected:
                raise ValueError("frozen stream or representation split file SHA-256 differs")
    if args.dense_evidence_mode == "calvin_full_v1":
        if (
            args.dense_evidence_coverage_plan.is_symlink()
            or not args.dense_evidence_coverage_plan.is_file()
        ):
            raise FileNotFoundError(args.dense_evidence_coverage_plan)
        if _sha256(args.dense_evidence_coverage_plan) != (args.dense_evidence_coverage_plan_sha256):
            raise ValueError("dense evidence coverage plan file SHA-256 differs")
    videomt_stage_pq_asset_receipt = _videomt_stage_pq_asset_receipt(args)
    adr221_asset_receipt = _adr221_asset_receipt(args)
    implementation_sha256 = _implementation_digest(root)
    evidence_sha256 = _canonical_digest(
        {
            "dataset_manifest_sha256": _sha256(args.dataset_manifest),
            "norm_stats_sha256": _sha256(args.norm_stats),
            "training_config_sha256": _sha256(args.training_config),
            "robot_config_sha256": _sha256(args.robot_config),
            "data_config_sha256": _sha256(args.data_config),
            "physical_sidecar_manifest_sha256": args.physical_sidecar_manifest_sha256,
            "physical_visual_acceptance_sha256": args.physical_visual_acceptance_sha256,
            "predictive_cache_build_report_sha256": predictive_build_report_sha256,
            "current_grid_cache_build_report_sha256": current_grid_build_report_sha256,
            "future_latent_cache_build_report_sha256": (
                future_latent_build_report_sha256
            ),
            "future_latent_cache_manifest_sha256": (
                args.future_latent_cache_manifest_sha256
            ),
            "dense_evidence_mode": args.dense_evidence_mode,
            "dense_token_bridge": args.dense_token_bridge,
            "dense_evidence_cache_manifest_sha256": sorted(
                args.dense_evidence_cache_manifest_sha256
            ),
            "dense_evidence_supplement_cache_manifest_sha256": sorted(
                args.dense_evidence_supplement_cache_manifest_sha256
            ),
            "dense_evidence_subset_view": args.dense_evidence_subset_view,
            "dense_evidence_coverage_plan_file_sha256": (args.dense_evidence_coverage_plan_sha256),
            "stream_plan_file_sha256": args.stream_plan_sha256,
            "representation_split_file_sha256": args.representation_split_sha256,
            "evaluation_plan_file_sha256": args.evaluation_plan_sha256,
            "videomt_stage_pq_assets": videomt_stage_pq_asset_receipt,
            "adr221_wsa_da3_assets": adr221_asset_receipt,
        }
    )
    execution_contract = _execution_contract(args)
    execution_contract_sha256 = _canonical_digest(execution_contract)

    if os.environ.get("WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("task-independent full training world-size contract differs")
    if os.environ.get("LOCAL_WORLD_SIZE") != str(WORLD_SIZE):
        raise RuntimeError("all task-independent ranks must run on one host")

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

    from lingbotvla.checkpoint import build_checkpointer
    from lingbotvla.data import VLADataCollatorWithPacking
    from lingbotvla.data.vla_data.utils import FeatureTransform
    from lingbotvla.distributed.parallel_state import init_parallel_state
    from lingbotvla.distributed.torch_parallelize import build_parallelize_model
    from lingbotvla.models import build_processor
    from lingbotvla.models.module_utils import init_empty_weights, load_model_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.moe_load_balance import build_moe_load_balance_hook
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.optim import build_muon_optimizer
    from transformers import AutoConfig, __version__ as transformers_version
    from transformers.modeling_utils import no_init_weights

    from picf_next.data.calvin import (
        CALVIN_HOST_IMAGE_KEYS,
        CalvinDatasetIndex,
        CalvinPhysicalTransitionDataset,
        CalvinStatefulTransitionDataset,
    )
    from picf_next.data.calvin_multimodal import validate_calvin_evidence_timestamps
    from picf_next.data.calvin_normalization import validate_lingbot_calvin_norm_stats
    from picf_next.data.calvin_physical_supervision_schema import (
        CALVIN_PHYSICAL_COVERAGE_ALL_SOURCE_FRAMES,
    )
    from picf_next.data.calvin_physical_supervision_sidecar import (
        CalvinPhysicalSupervisionSidecar,
    )
    from picf_next.data.dataset_manifest import (
        load_dataset_file_manifest,
        validate_dataset_runtime_binding,
    )
    from picf_next.data.dense_evidence_cache import (
        FrozenDenseEvidenceCacheBank,
        FrozenDenseEvidenceCacheView,
        compose_dense_evidence_cache_banks,
    )
    from picf_next.data.dense_evidence_coverage import (
        DenseEvidenceCoveragePlan,
        build_calvin_dense_evidence_coverage_plan,
    )
    from picf_next.lingbot_native.calvin import (
        build_native_calvin_episode_domain,
        build_native_calvin_continuation_batch,
        build_native_calvin_physical_episode_domain,
        build_native_calvin_physical_stream_plan,
        build_native_calvin_replay_batch,
        build_native_calvin_training_stream_plan,
        build_planned_native_calvin_batch,
        collate_native_calvin_training_batch,
        materialize_native_flow_randomness,
        with_native_modalities,
        with_official_proprioception_modality,
        with_wla_world_target,
    )
    from picf_next.lingbot_wla_calvin import (
        build_wla_calvin_target_batch_from_source_indices,
    )
    from picf_next.lingbot_wla_install import (
        WLA_ACTION_BLOCK_CLASS,
        WLA_ACTION_FSDP_PARAMETER_PREFIX,
        WLA_HOST_TEXT_BLOCK_CLASS,
        WLA_HOST_TEXT_FSDP_PARAMETER_PREFIX,
        WLA_WORLD_BLOCK_CLASS,
        WLA_WORLD_FSDP_PARAMETER_PREFIX,
        audit_lingbot_wla_fsdp_topology,
        audit_lingbot_wla_optimizer,
        build_lingbot_wla_optimizer,
        build_lingbot_wla_scheduler,
        install_lingbot_wla_backend,
        register_lingbot_wla_fsdp_forward,
    )
    from picf_next.lingbot_wla_world import build_wla_target_transform
    from picf_next.lingbot_native.dense_modalities import (
        NativeDenseModalityBinding,
        dense_modality_bindings_sha256,
        native_modalities_from_dense_evidence,
    )
    from picf_next.lingbot_native.calvin_entity_training import (
        OMITTED_STATIC_REMATERIALIZATION_MODES as CORE_OMITTED_STATIC_REMATERIALIZATION_MODES,
        TaskIndependentPredictiveRolloutFactory,
        finalize_task_independent_calvin_sequential_omitted_result,
        run_task_independent_calvin_joint_sequence_objective,
        run_task_independent_calvin_recurrent_frame_diagnostic,
        run_task_independent_calvin_sequential_omitted_static_objective,
    )
    from picf_next.lingbot_native.calvin_entity_set import (
        build_task_independent_calvin_targets,
        physical_frame_predictions_from_relation,
    )
    from picf_next.lingbot_native.action_posterior_collector import (
        RegisteredActionPosteriorReceiptCollector,
    )
    from picf_next.lingbot_native.action_posterior_learning import (
        action_posterior_target_mass_loss,
    )
    from picf_next.lingbot_native.current_grid_cache import (
        CurrentGridCacheContract,
        LingBotCurrentGridTargetCache,
    )
    from picf_next.lingbot_native.future_latent_alignment import (
        FutureLatentAlignmentConfig,
        LingBotFutureLatentAlignment,
        install_lingbot_future_latent_alignment,
    )
    from picf_next.lingbot_native.future_latent_cache import (
        FutureLatentTargetCache,
        future_latent_source_keys_digest,
    )
    from picf_next.lingbot_native.entity_training import (
        TaskIndependentEntityObjectiveConfig,
        compose_task_independent_entity_objective,
    )
    from picf_next.lingbot_native.entity_set_evaluation import (
        evaluate_physical_entity_frame,
        summarize_entity_evaluation_partition,
    )
    from picf_next.lingbot_native.entity_evaluation_plan import (
        ENTITY_EVALUATION_PARTITIONS,
        EntityEvaluationPlan,
        build_distributed_causal_warm_evaluation_schedule,
        build_distributed_entity_evaluation_schedule,
        build_entity_evaluation_plan,
    )
    from picf_next.lingbot_native.host import (
        LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR,
        LINGBOT_TASK_QUERY_OBJECT_VALUE_READ,
        TASK_INDEPENDENT_ENTITY_POSTERIOR,
        UNIFIED_LAYERWISE_PREDICT_CORRECT,
        LingBotNativeGraph,
        LingBotNativeGraphConfig,
        LingBotNativePriorStepper,
        install_lingbot_native_graph,
        native_context_from_prior_trace,
    )
    from picf_next.lingbot_native.addresses import address_codebook_sha256
    from picf_next.lingbot_native.modalities import (
        CALVIN_VIDEOMT_MASK_LAYOUT,
        CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
        CALVIN_VJEPA21_VISIBLE_OWNER_TARGET,
        NO_RELATION_TARGET,
        NativeModalitySpec,
        NativeObjectQuerySpatialSpec,
        NativeRelationSurfaceSpec,
    )
    from picf_next.lingbot_native.physical_relations import (
        NativeObjectQueryPosteriorOutput,
        PhysicalRelationOutput,
    )
    from picf_next.lingbot_native.predictive_cache import (
        LINGBOT_PREDICTIVE_TARGET_SPACE,
        LingBotPredictiveTargetCache,
        PredictiveCacheContract,
        native_predictive_query_schema_digest,
    )
    from picf_next.lingbot_native.predictive_plan import (
        build_native_current_grid_coverage_plan,
        build_native_predictive_coverage_plan,
    )
    from picf_next.videomt_exact.calvin_stage_p import (
        VIDEOMT_STAGE_PQ_C5_INTERFACE,
        VIDEOMT_STAGE_PQMR_HOST_OUTPUT,
        VIDEOMT_STAGE_PQM_HOST_OUTPUT,
        VIDEOMT_STAGE_PQRF_HOST_OUTPUT,
        InsufficientCausalPrefixError,
        empty_videomt_query_modality_batch,
        make_videomt_stage_pq_execution_receipt,
        prepare_calvin_stage_pq_c5,
    )
    from picf_next.videomt_exact.posterior_refiner import (
        PREPEND_PROJECTED_ROWS,
        REPLACE_WITH_RELEASED_PROPAGATION,
        FrozenVidEoMTPosteriorRowRefiner,
    )
    from picf_next.videomt_exact.fsdp2 import parallelize_exact_videomt_fsdp2
    from picf_next.videomt_exact.joint_data import (
        audit_native_videomt_source_eligibility,
        prepare_native_videomt_current_frame,
        prepare_native_videomt_source_batch,
    )
    from picf_next.videomt_exact.joint_training import CompleteCalvinVidEoMTObjective
    from picf_next.videomt_exact.gradient_diagnostics import (
        shared_query_gradient_moments,
    )
    from picf_next.videomt_exact.joint_visual import render_native_videomt_query_visuals
    from picf_next.videomt_exact.lingbot_joint import (
        run_causal_warm_native_videomt_lingbot_evaluation,
        run_cold_native_videomt_lingbot_evaluation,
        run_complete_native_videomt_lingbot_step,
        run_native_videomt_host_diagnostic,
    )
    from picf_next.videomt_exact.calvin_targets import prepare_calvin_videomt_clip
    from picf_next.videomt_exact.evaluation import (
        evaluate_videomt_anchors,
        render_videomt_anchor_panel,
    )
    from picf_next.videomt_exact.optimizer import (
        VIDEOMT_ADAPTATION_BUDGET_STEPS,
        build_exact_videomt_optimizer,
        build_exact_videomt_scheduler,
    )
    from picf_next.videomt_exact.runtime import ExactVidEoMTConfig, load_exact_videomt
    from picf_next.videomt_exact.stage_p import (
        VidEoMTStageP,
        VidEoMTStagePQMR,
        VidEoMTStagePQM,
        with_videomt_query_modality_spec,
        with_videomt_row_mask_query_modality_spec,
    )
    from picf_next.lingbot_native.predictive_probes import zero_executed_control
    from picf_next.lingbot_native.prediction import PredictionSource
    from picf_next.lingbot_native.representation_split import (
        RepresentationTrialSplit,
        verify_representation_trial_split_training_evidence,
    )
    from picf_next.lingbot_native.full_training import make_native_future_request
    from picf_next.lingbot_native.full_modal_adoption import (
        ACTION_ADOPTION_CORE_SCHEMA,
        ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT,
        ACTION_ADOPTION_NONZERO_GRADIENT_MIN_NORM,
        ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT,
        DENSE_MODALITIES,
        DENSE_PRESENCE_SUBSETS,
        MODALITY_INTERVENTIONS,
        action_projection_drift_report,
        aggregate_rank_action_outputs,
        aggregate_rank_state_digests,
        captured_action_outputs_sha256,
        capture_action_projection_output,
        current_process_start_ticks,
        dense_presence_code,
        directory_tree_sha256,
        distributed_action_adoption_gradients,
        distributed_maximum_action_drift,
        intervene_modality,
        make_action_adoption_interventions_report,
        make_action_adoption_presence_report,
        make_action_dcp_phase_report,
        process_set_sha256,
        resolve_action_adoption_parameter_groups,
        single_captured_action_output,
        validate_action_dcp_phase_report,
        with_dense_presence,
    )
    from picf_next.lingbot_native.source_mask import (
        sample_qwen_packed_patch_mask,
        sample_qwen_whole_view_omission,
    )
    from picf_next.lingbot_native.addresses import EpisodeAddressState
    from picf_next.lingbot_native.state import (
        AddressedLayerwisePosteriorState,
        AddressedLayerwisePriorTrace,
        NativeLayerwisePosteriorState,
        NativeLayerwisePriorTrace,
        NativePersistentState,
        NativePosteriorState,
        NativeVidEoMTPairedPosteriorState,
        layerwise_prior_trace_with_tensor,
        persistent_state_tensor,
        persistent_state_with_tensor,
    )
    from picf_next.lingbot_native.temporal import (
        NativeLaneConfig,
        NativeTrainingLaneBank,
        TemporalEstimatorConfig,
        native_temporal_batch_seed,
        rollout_native_prior_prediction,
        sample_temporal_batch_plan,
    )
    from picf_next.lingbot_native.training import (
        NativeTrainingLaneCoordinator,
        audit_native_optimizer_coverage,
        native_cold_state_for_episode_keys,
        run_native_policy_diagnostic_forward,
        run_native_v3_prior_chain,
    )
    from picf_next.lingbot_native.wsa_da3_teacher_runtime import (
        OnlineWSADA3TeacherRuntime,
    )
    from picf_next.lingbot_native.wsa_future_expert_runtime import (
        WSAFutureExpertRuntime,
    )
    from picf_next.lingbot_native.wsa_lingbot_install import (
        WSA_FSDP_BLOCK_CLASS,
        WSA_FSDP_EXPERT_CLASS,
        WSA_FSDP_PARAMETER_PREFIX,
        audit_wsa_lingbot_optimizer,
        audit_wsa_lingbot_scheduler,
        build_wsa_lingbot_scheduler,
        configure_wsa_lingbot_optimizer_contract,
        install_wsa_lingbot_optimizer,
        install_wsa_lingbot_training_runtime,
        register_wsa_lingbot_fsdp_forward_methods,
        wsa_lingbot_installation_receipt,
        wsa_lingbot_optimizer_transaction,
    )
    from picf_next.lingbot_native.wsa_lingbot_training_runtime import (
        WSALingBotActionCoupling,
        WSALingBotAttentionIntervention,
        WSALingBotTrainingRuntime,
    )
    from picf_next.lingbot_native.visual_audit import (
        render_task_independent_entity_visuals,
    )
    from picf_next.training.control import derive_subseed, load_frozen_episode_stream_plan
    from tools.run_lingbot_vla2_official_lbot import (
        _evaluation_replay_seed,
        _summarize_action_partition,
    )

    if (
        MODALITY_INTERVENTIONS != ADR207_MODALITY_INTERVENTIONS
        or ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT
        != ADR207_ACTION_STABILITY_MAX_ABS_DRIFT
        or ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT
        != ADR207_ACTION_EFFECT_MIN_ABS_DRIFT
    ):
        raise RuntimeError("ADR-207 modality gate drifted from the mature ADR-150 primitive")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    run_lease = None
    metric_handle = None
    try:
        run_lease = acquire_distributed_run_lease(args.run_dir, rank=rank, distributed=dist)
        run_preflight: list[str | None] = [None]
        if rank == 0:
            try:
                if args.phase == "fresh":
                    blockers = sorted(
                        path.name
                        for path in args.run_dir.iterdir()
                        if path.name != ".picf-single-writer.lock"
                    )
                    if blockers:
                        raise FileExistsError(
                            "fresh run root is not empty: " + ", ".join(blockers[:8])
                        )
                elif not (args.run_dir / "run_manifest.json").is_file():
                    raise FileNotFoundError("resume run root has no run manifest")
                elif _causal_ablation_active(args.causal_ablation_mode):
                    branch_manifest = json.loads(
                        (args.run_dir / "run_manifest.json").read_text(encoding="utf-8")
                    )
                    if (
                        branch_manifest.get("causal_ablation_mode") != "current_frame_branch"
                        or branch_manifest.get("execution_contract_sha256")
                        != execution_contract_sha256
                    ):
                        raise ValueError("ADR-146 arm does not resume its registered branch")
            except BaseException as error:
                run_preflight[0] = f"{type(error).__name__}: {error}"
        dist.broadcast_object_list(run_preflight, src=0)
        if run_preflight[0] is not None:
            raise RuntimeError(f"run-root preflight failed: {run_preflight[0]}")
        if torch.cuda.device_count() != WORLD_SIZE:
            raise RuntimeError(
                f"the process must see exactly {WORLD_SIZE} GPUs for this execution contract"
            )
        properties = torch.cuda.get_device_properties(device)
        if "A100" not in properties.name or properties.total_memory < 39 * 1024**3:
            raise RuntimeError("training requires A100 GPUs with at least 39 GiB each")

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
            raise RuntimeError(
                f"task-independent model artifact contract failed: {artifact_contract_report}"
            )
        checkpoint_report = artifact_contract_report["checkpoint"]
        processor_report = artifact_contract_report["processor"]
        if not isinstance(checkpoint_report, dict) or not isinstance(processor_report, dict):
            raise RuntimeError("model artifact validators returned non-mapping reports")

        dataset_contract: list[Any] = [None]
        rank_zero_manifest = None
        if rank == 0:
            try:
                rank_zero_manifest = load_dataset_file_manifest(args.dataset_manifest.resolve())
                norm_stats = json.loads(args.norm_stats.read_text(encoding="utf-8"))
                validate_lingbot_calvin_norm_stats(norm_stats)
                source = norm_stats["source"]
                if (
                    source["dataset_id"] != rank_zero_manifest.dataset_id
                    or source["dataset_revision"] != rank_zero_manifest.dataset_revision
                    or source["dataset_tree_sha256"] != rank_zero_manifest.tree_sha256
                    or rank_zero_manifest.split_name != args.dataset_split.name
                ):
                    raise ValueError("CALVIN manifest and normalization differ")
                dataset_contract[0] = {
                    "status": "PASS",
                    "validation": validate_dataset_runtime_binding(
                        rank_zero_manifest,
                        args.dataset_split,
                        dataset_id=source["dataset_id"],
                        dataset_revision=source["dataset_revision"],
                        split_name=args.dataset_split.name,
                    ),
                }
            except BaseException as error:
                dataset_contract[0] = {
                    "status": "FAIL",
                    "error": f"{type(error).__name__}: {error}",
                }
        dist.broadcast_object_list(dataset_contract, src=0)
        if not isinstance(dataset_contract[0], dict) or (
            dataset_contract[0].get("status") != "PASS"
        ):
            raise RuntimeError(f"dataset contract failed: {dataset_contract[0]}")
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
            raise RuntimeError("full training requires all-source physical supervision")

        temporal_config = TemporalEstimatorConfig(
            local_bptt_probability=args.local_bptt_probability,
            overshoot_probability=args.overshoot_probability,
            source_mask_probability=args.source_mask_probability,
            maximum_optimizer_lag=args.maximum_optimizer_lag,
        )
        training = load_lingbot_training_config(args.training_config)
        # The released LingBot contract remains part of the immutable base-family
        # identity. Complete WLA replaces its optimizer below with the donor's
        # exact AdamW/scheduler, so validate the YAML against its own LR here.
        released_lingbot_learning_rate = float(training["train"]["lr"])
        optimizer_contract = resolve_lingbot_optimizer_contract(
            training,
            requested_learning_rate=(
                released_lingbot_learning_rate
                if _wla_complete_active(args)
                else args.learning_rate
            ),
        )
        if _adr221_full_source_wsa_active(args):
            optimizer_contract = configure_wsa_lingbot_optimizer_contract(
                optimizer_contract
            )
        require_lingbot_exact_resume_contract(optimizer_contract)
        native_model_patch_sha256 = patch_report.get(
            "native_patch_sha256", patch_report.get("patch_sha256")
        )
        if not isinstance(native_model_patch_sha256, str):
            raise RuntimeError("LingBot native model patch identity is absent")
        lingbot_base_family = build_lingbot_base_family_identity(
            source_commit=LINGBOT_NATIVE_SOURCE_COMMIT,
            native_patch_sha256=native_model_patch_sha256,
            checkpoint_revision=LINGBOT_CHECKPOINT_REVISION,
            checkpoint_report=checkpoint_report,
            processor_revision=QWEN_PROCESSOR_REVISION,
            processor_report=processor_report,
            attention_implementation=args.attention_implementation,
            trainable_scope=args.trainable_scope,
            optimizer_contract=asdict(optimizer_contract),
            maximum_control_tokens=args.maximum_control_tokens,
        )
        lingbot_base_family_sha256 = lingbot_base_family["artifact_sha256"]
        merged, _ = _resolve_training_config(
            training,
            checkpoint_dir=args.checkpoint_dir,
            processor_dir=args.processor_dir,
            num_steps=LINGBOT_RELEASED_ACTION_SAMPLING_STEPS,
        )
        merged["use_cache"] = False
        merged["use_compile"] = (
            args.lingbot_compile_mode == LINGBOT_COMPILE_UPSTREAM_DEFAULT
        )
        merged["attention_implementation"] = args.attention_implementation
        merged["vit_attn_implementation"] = "eager"
        merged["freeze_vision_encoder"] = args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST
        merged["train_expert_only"] = False
        if _wla_complete_active(args):
            # WLA's world target and action sequence are both t+chunk_size.
            # Preserve the pinned LIBERO donor's chunk_size=8 instead of
            # conflating it with repeated_diffusion_steps=16.
            merged["chunk_size"] = WLA_SOURCE_ACTION_HORIZON
        config = LingbotVLAV2Config(**merged)
        for key, value in merged.items():
            if not hasattr(config, key):
                setattr(config, key, value)
        require_lingbot_released_action_sampling_steps(config)
        if bool(config.train_expert_only):
            raise RuntimeError("task-independent full training forbids expert-only scope")
        expected_frozen_vision = args.trainable_scope == TRAINABLE_SCOPE_FROZEN_VISION_HOST
        if bool(config.freeze_vision_encoder) != expected_frozen_vision:
            raise RuntimeError("LingBot visual trainable scope differs from the CLI contract")
        qwen_config = AutoConfig.from_pretrained(
            args.processor_dir,
            revision=QWEN_PROCESSOR_REVISION,
            local_files_only=True,
        )
        vision_config = qwen_config.vision_config
        training_projection = _validate_lingbot_projection_processor(
            physical_visual_acceptance=physical_visual_acceptance,
            processor_report=processor_report,
            vision_config=vision_config,
            dataset_tree_sha256=dataset_manifest.tree_sha256,
            transformers_version=transformers_version,
        )
        patch_size = int(vision_config.patch_size)
        merge_size = int(vision_config.spatial_merge_size)
        _merge_qwen_config(config, qwen_config)
        config.tokenizer_path = str(args.processor_dir.resolve())
        physical_event_stream = _two_pass_filter_active(args)
        posterior_adoption_route_active = _posterior_adoption_route_active_for_args(args)
        action_evaluation_steps = _registered_action_evaluation_steps(args)
        evaluation_dataset = CalvinStatefulTransitionDataset(
            index,
            action_horizon=config.chunk_size,
        )
        physical_dataset = CalvinPhysicalTransitionDataset(
            index,
            action_horizon=config.chunk_size,
        )
        dataset = physical_dataset if physical_event_stream else evaluation_dataset
        dense_evidence_bank = None
        dense_evidence_coverage = None
        dense_evidence_composition: dict[str, Any] | None = None
        dense_evidence_bindings: tuple[NativeDenseModalityBinding, ...] = ()
        canonical_dense_key_by_source_index: dict[int, str] = {}
        if args.dense_evidence_mode == "calvin_full_v1":
            dense_evidence_coverage = DenseEvidenceCoveragePlan.load(
                args.dense_evidence_coverage_plan
            )
            if (
                dense_evidence_coverage.dataset_id,
                dense_evidence_coverage.dataset_revision,
                dense_evidence_coverage.dataset_tree_sha256,
            ) != (
                dataset_manifest.dataset_id,
                dataset_manifest.dataset_revision,
                dataset_manifest.tree_sha256,
            ):
                raise RuntimeError("dense evidence coverage belongs to another dataset")
            primary_dense_evidence_bank = FrozenDenseEvidenceCacheBank.load(
                args.dense_evidence_cache_root,
                manifest_sha256s=args.dense_evidence_cache_manifest_sha256,
                dataset_tree_sha256=dataset_manifest.tree_sha256,
            )
            if primary_dense_evidence_bank.modalities != CALVIN_FULL_DENSE_MODALITIES:
                raise RuntimeError(
                    "calvin_full_v1 requires exact anytouch/sonata/vjepa evidence caches"
                )
            expected_dense_records = dense_evidence_coverage.record_identities
            supplement_roots = tuple(args.dense_evidence_supplement_cache_root)
            if supplement_roots or args.dense_evidence_subset_view:
                source_banks = [primary_dense_evidence_bank]
                if supplement_roots:
                    source_banks.append(
                        FrozenDenseEvidenceCacheBank.load(
                            supplement_roots,
                            manifest_sha256s=(
                                args.dense_evidence_supplement_cache_manifest_sha256
                            ),
                            dataset_tree_sha256=dataset_manifest.tree_sha256,
                        )
                    )
                dense_evidence_bank = compose_dense_evidence_cache_banks(
                    tuple(source_banks),
                    record_identities=expected_dense_records,
                    coverage_plan_sha256=dense_evidence_coverage.artifact_sha256,
                )
                dense_evidence_composition = {
                    "mode": (
                        "authenticated_union_view"
                        if supplement_roots
                        else "authenticated_single_source_subset_view"
                    ),
                    "source_coverage_plan_sha256": [
                        bank.coverage_plan_sha256 for bank in source_banks
                    ],
                    "selected_record_count_by_modality_and_source": {
                        cache.contract.modality: list(cache.source_record_counts)
                        for cache in dense_evidence_bank.caches
                        if isinstance(cache, FrozenDenseEvidenceCacheView)
                    },
                }
            else:
                dense_evidence_bank = primary_dense_evidence_bank
                if dense_evidence_bank.coverage_plan_sha256 != (
                    dense_evidence_coverage.artifact_sha256
                ):
                    raise RuntimeError("dense evidence caches bind another coverage plan")
                observed_dense_records = tuple(
                    (record.source_global_index, record.sample_key)
                    for record in dense_evidence_bank.caches[0].records
                )
                if observed_dense_records != expected_dense_records:
                    raise RuntimeError(
                        "dense evidence caches do not exactly cover the frozen training/eval plan"
                    )
                dense_evidence_composition = {
                    "mode": "exact_single_bank",
                    "source_coverage_plan_sha256": [dense_evidence_bank.coverage_plan_sha256],
                    "selected_record_count_by_modality_and_source": {
                        modality: [dense_evidence_bank.record_count]
                        for modality in dense_evidence_bank.modalities
                    },
                }
            canonical_dense_key_by_source_index = dict(expected_dense_records)
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
            dense_evidence_bindings_sha256 = dense_modality_bindings_sha256(dense_evidence_bindings)
        else:
            dense_evidence_bindings_sha256 = _disabled_auxiliary_digest("dense_evidence")
        representation_split = None
        evaluation_plan = None
        if frozen_stream_enabled:
            representation_split = RepresentationTrialSplit.load(args.representation_split)
            stream_plan = load_frozen_episode_stream_plan(
                args.stream_plan,
                episodes=(
                    build_native_calvin_physical_episode_domain(
                        dataset,
                        excluded_source_episode_indices=(
                            representation_split.stream_domain_excluded_source_episode_indices
                        ),
                        minimum_future_source_frames=_required_future_source_frames(args),
                    )
                    if physical_event_stream
                    else build_native_calvin_episode_domain(
                        dataset,
                        excluded_source_episode_indices=(
                            representation_split.stream_domain_excluded_source_episode_indices
                        ),
                    )
                ),
            )
            if (
                stream_plan.comparison_id != COMPARISON_ID
                or stream_plan.seed != args.seed
                or stream_plan.global_batch_size != WORLD_SIZE
                or stream_plan.total_steps != TOTAL_STEPS
                or stream_plan.lane_interleave_factor != 1
            ):
                raise ValueError("frozen four-rank stream differs from the execution contract")
            verify_representation_trial_split_training_evidence(
                representation_split,
                stream_plan,
                dataset,
            )
            if representation_split.training_steps != TOTAL_STEPS:
                raise ValueError("representation split does not cover the complete 30k stream")
            evaluation_plan = EntityEvaluationPlan.load(args.evaluation_plan)
            if evaluation_plan.representation_split_sha256 != representation_split.artifact_sha256:
                raise ValueError("action evaluation plan belongs to another split")
            if (
                build_entity_evaluation_plan(
                    representation_split,
                    evaluation_dataset,
                    world_size=WORLD_SIZE,
                )
                != evaluation_plan
            ):
                raise ValueError("action evaluation plan is not reproducible from source")
            evaluation_sources = {item.source_episode_index for item in evaluation_plan.items}
            if evaluation_sources.intersection(
                representation_split.training_source_episode_indices
            ):
                raise ValueError("action evaluation overlaps a training source episode")
            rank_forward_counts = [
                len(build_distributed_entity_evaluation_schedule(evaluation_plan, rank=item_rank))
                for item_rank in range(WORLD_SIZE)
            ]
            if len(set(rank_forward_counts)) != 1:
                raise RuntimeError("action evaluation padding failed to align collectives")
            if dense_evidence_coverage is not None and (
                dense_evidence_coverage.stream_plan_sha256 != stream_plan.plan_sha256
                or dense_evidence_coverage.representation_split_sha256
                != representation_split.artifact_sha256
                or dense_evidence_coverage.evaluation_plan_sha256 != evaluation_plan.artifact_sha256
            ):
                raise ValueError(
                    "dense evidence coverage differs from the frozen stream/split/evaluation"
                )
            if dense_evidence_coverage is not None:
                dense_prefix_steps = _dense_evidence_training_step_prefix(
                    dense_evidence_coverage,
                    stream_plan,
                    stop_after_step=args.stop_after_step,
                )
                reproduced_dense_coverage = build_calvin_dense_evidence_coverage_plan(
                    stream_plan=stream_plan,
                    representation_split=representation_split,
                    evaluation_plan=evaluation_plan,
                    physical_dataset=physical_dataset,
                    evaluation_dataset=evaluation_dataset,
                    training_step_prefix=dense_prefix_steps,
                    evaluation_history_transitions=(
                        dense_evidence_coverage.evaluation_history_transition_count
                    ),
                    schema=dense_evidence_coverage.schema,
                )
                if reproduced_dense_coverage != dense_evidence_coverage:
                    raise ValueError(
                        "dense evidence coverage is not the reproducible frozen-stream prefix"
                    )
        else:
            stream_plan = (
                build_native_calvin_physical_stream_plan(
                    dataset,
                    comparison_id=COMPARISON_ID,
                    seed=args.seed,
                    global_batch_size=WORLD_SIZE,
                    total_steps=TOTAL_STEPS,
                    lane_interleave_factor=1,
                    minimum_future_source_frames=_required_future_source_frames(args),
                )
                if physical_event_stream
                else build_native_calvin_training_stream_plan(
                    dataset,
                    comparison_id=COMPARISON_ID,
                    seed=args.seed,
                    global_batch_size=WORLD_SIZE,
                    total_steps=TOTAL_STEPS,
                    lane_interleave_factor=1,
                )
            )
        future_latent_cache = None
        future_latent_config = None
        future_latent_cache_manifest_sha256 = _disabled_auxiliary_digest(
            "future_latent_cache"
        )
        if _future_latent_alignment_active(args):
            if representation_split is None or not frozen_stream_enabled:
                raise RuntimeError("FLARE requires one frozen representation stream")
            cache_error: str | None = None
            try:
                future_latent_cache = FutureLatentTargetCache(
                    args.future_latent_cache_root,
                    verify_shards=rank == 0,
                )
                future_latent_config = FutureLatentAlignmentConfig()
                future_latent_config.assert_adr209_complete()
                contract = future_latent_cache.contract
                expected_identity = (
                    dataset_manifest.dataset_id,
                    dataset_manifest.dataset_revision,
                    args.dataset_split.resolve().name,
                    dataset_manifest.tree_sha256,
                    stream_plan.plan_sha256,
                    args.stream_plan_sha256,
                    args.representation_split_sha256,
                    future_latent_config.digest,
                )
                observed_identity = (
                    contract.dataset_id,
                    contract.dataset_revision,
                    contract.split_name,
                    contract.dataset_tree_sha256,
                    contract.stream_plan_sha256,
                    contract.stream_plan_file_sha256,
                    contract.representation_split_sha256,
                    contract.alignment_config_digest,
                )
                if observed_identity != expected_identity:
                    raise RuntimeError("FLARE cache belongs to another model/data stream")
                if future_latent_cache.manifest_sha256 != (
                    args.future_latent_cache_manifest_sha256
                ):
                    raise RuntimeError("FLARE cache manifest digest differs from the CLI")
                if not args.load_global_step <= args.stop_after_step <= (
                    contract.training_prefix_steps
                ):
                    raise RuntimeError(
                        "FLARE cache does not cover the requested training boundary"
                    )
                identities_by_key: dict[str, tuple[str, int, int]] = {}
                for optimizer_step in range(contract.training_prefix_steps):
                    for transition in stream_plan.global_batch(optimizer_step).transitions:
                        sample_key = transition.sample.sample_key
                        source_index = physical_dataset.source_global_index_by_key(sample_key)
                        future_index = physical_dataset.future_source_global_indices_by_key(
                            sample_key,
                            count=FLARE_REQUIRED_FUTURE_SOURCE_FRAMES,
                        )[-1]
                        identities_by_key[sample_key] = (
                            sample_key,
                            source_index,
                            future_index,
                        )
                identities = tuple(sorted(identities_by_key.values()))
                if (
                    len(identities) != contract.expected_record_count
                    or future_latent_source_keys_digest(identities)
                    != contract.source_keys_sha256
                ):
                    raise RuntimeError("FLARE cache is not the exact frozen-stream prefix")
                if future_latent_build_report is None or (
                    future_latent_build_report.get("contract") != contract.to_dict()
                ):
                    raise RuntimeError("FLARE cache and authenticated build report differ")
                future_latent_cache_manifest_sha256 = (
                    future_latent_cache.manifest_sha256
                )
            except BaseException as error:
                cache_error = f"{type(error).__name__}: {error}"
            gathered_cache_errors: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_cache_errors, cache_error)
            failures = tuple(
                f"rank {item_rank}: {error}"
                for item_rank, error in enumerate(gathered_cache_errors)
                if error is not None
            )
            if failures:
                raise RuntimeError("FLARE cache validation failed: " + "; ".join(failures))
            if future_latent_cache is None or future_latent_config is None:
                raise RuntimeError("FLARE cache validation returned no runtime cache")

        def future_latent_target_for_batch(batch: Any) -> Any | None:
            if future_latent_cache is None:
                return None
            return future_latent_cache.target_for(
                sample_keys=batch.routing.sample_keys,
                source_global_indices=tuple(
                    request.source_global_index
                    for request in batch.structural_target_requests
                ),
                device=device,
            )

        videomt_source_eligibility = (
            audit_native_videomt_source_eligibility(physical_dataset, stream_plan)
            if _adr207_native_videomt_query_posterior_active(args)
            else None
        )
        evaluation_visual_sample_keys: frozenset[str] = frozenset()
        if evaluation_plan is not None:
            evaluation_visual_sample_keys = frozenset(
                _evaluation_visual_sample_keys(
                    evaluation_plan.items,
                    partitions=ENTITY_EVALUATION_PARTITIONS,
                    per_partition=1,
                )
            )
        predictive_contract = None
        current_contract = None
        predictive_manifest_sha256 = _disabled_auxiliary_digest("predictive")
        current_manifest_sha256 = _disabled_auxiliary_digest("current_grid")
        predictive_cache = None
        current_grid_cache = None
        current_grid_cache_binding = None
        if _predictive_assets_required(args):
            predictive_manifest, predictive_manifest_sha256 = _cache_manifest(
                args.predictive_cache_root,
                require_complete_field=True,
            )
            predictive_contract_payload = dict(predictive_manifest["contract"])
            predictive_contract_payload["horizons"] = tuple(predictive_contract_payload["horizons"])
            predictive_contract = PredictiveCacheContract(**predictive_contract_payload)
        if _current_correction_assets_required(args):
            current_manifest, current_manifest_sha256 = _cache_manifest(
                args.current_grid_cache_root,
                require_complete_field=False,
            )
            current_contract = CurrentGridCacheContract.from_mapping(current_manifest["contract"])
            if current_grid_build_report is None:
                raise RuntimeError("current cache lacks its validated build report")
            _validate_current_cache_build_binding(
                report=current_grid_build_report,
                contract=current_contract,
                manifest_sha256=current_manifest_sha256,
                output_root=args.current_grid_cache_root,
            )
        if args.predictive_weight > 0:
            if current_contract is None:
                raise RuntimeError("predictive correction lacks its validated current cache")
            expected_current = build_native_current_grid_coverage_plan(
                stream_plan,
                temporal_config,
                source_global_index_for_sample=dataset.source_global_index_by_key,
                required_future_offsets=((1,) if _two_pass_filter_active(args) else ()),
            )
            load_coverage_sha256, current_grid_cache_binding = _resolve_current_grid_cache_coverage(
                acceptance_mode=args.acceptance_mode,
                contract=current_contract,
                expected=expected_current,
                temporal_config=temporal_config,
            )
            current_grid_cache = LingBotCurrentGridTargetCache.load(
                args.current_grid_cache_root,
                shard_root=args.current_grid_cache_shard_root,
                manifest_sha256=current_manifest_sha256,
                dataset_tree_sha256=dataset_manifest.tree_sha256,
                physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
                encoder_digest=current_contract.encoder_digest,
                coverage_sha256=load_coverage_sha256,
                memory_capacity=2,
            )
            if _predictive_assets_required(args):
                if predictive_contract is None:
                    raise RuntimeError("controlled rollout lacks its predictive cache contract")
                expected_predictive = build_native_predictive_coverage_plan(
                    stream_plan,
                    temporal_config,
                    source_global_index_for_sample=dataset.source_global_index_by_key,
                )
                if (
                    predictive_contract.stream_plan_sha256 != stream_plan.plan_sha256
                    or predictive_contract.temporal_estimator_sha256 != temporal_config.digest
                    or predictive_contract.coverage_sha256 != expected_predictive.coverage_sha256
                    or predictive_contract.pair_keys_sha256 != expected_predictive.pair_keys_sha256
                    or predictive_contract.expected_record_count != len(expected_predictive.pairs)
                ):
                    raise RuntimeError("predictive cache does not cover the exact 30k plan")
                query_schema = native_predictive_query_schema_digest(
                    target_space=LINGBOT_PREDICTIVE_TARGET_SPACE,
                    route_id=0,
                    horizons=temporal_config.overshoot_horizons,
                )
                predictive_cache = LingBotPredictiveTargetCache.load(
                    args.predictive_cache_root,
                    manifest_sha256=predictive_manifest_sha256,
                    dataset_tree_sha256=dataset_manifest.tree_sha256,
                    physical_sidecar_manifest_sha256=physical_sidecar.manifest_sha256,
                    encoder_digest=predictive_contract.encoder_digest,
                    query_schema_sha256=query_schema,
                    coverage_sha256=expected_predictive.coverage_sha256,
                    memory_capacity=2,
                )

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
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
        modality_bridge_projector = None
        modality_bridge_queries = None
        modality_bridge_receipt: dict[str, Any] = {
            "schema": "picf-next.lingbot-modality-bridge-receipt/v1",
            "identity": args.dense_token_bridge,
            "source_component": None,
            "query_count": 0,
            "projector_parameter_count": 0,
        }
        if args.dense_token_bridge == LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE:
            flow = getattr(policy, "model", None)
            source_head = getattr(flow, "current_video_align_head", None)
            source_projector = getattr(source_head, "projector", None)
            source_queries = getattr(flow, "current_video_align_embs", None)
            if not isinstance(source_projector, torch.nn.Module) or not isinstance(
                source_queries,
                torch.nn.Parameter,
            ):
                raise RuntimeError(
                    "released LingBot current-video resampler is absent from the checkpoint"
                )
            if source_queries.ndim != 2 or tuple(source_queries.shape) != (256, 2560):
                raise RuntimeError("released LingBot current-video query contract differs")
            modality_bridge_projector = deepcopy(source_projector)
            modality_bridge_queries = source_queries.detach()
            source_queries.requires_grad_(False)
            modality_bridge_receipt = {
                "schema": "picf-next.lingbot-modality-bridge-receipt/v1",
                "identity": args.dense_token_bridge,
                "source_component": "model.current_video_align_head.projector",
                "query_component": "model.current_video_align_embs",
                "source_query_frozen_after_copy": True,
                "query_count": int(source_queries.shape[0]),
                "projector_parameter_count": sum(
                    parameter.numel() for parameter in source_projector.parameters()
                ),
            }
        alignment_teacher_prune = strip_targetless_alignment_teacher_heads(policy)
        policy.train()
        videomt_runtime = None
        source_mask_head = None
        source_mask_head_sha256 = None
        source_mask_refiner = None
        source_mask_refiner_sha256 = None
        source_mask_refiner_parameter_count = 0
        if _videomt_stage_pqmr_active(args):
            videomt_preload_device = (
                torch.device("cpu")
                if args.videomt_idle_placement
                == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                else device
            )
            videomt_runtime = load_exact_videomt(
                ExactVidEoMTConfig(
                    checkpoint_path=args.videomt_checkpoint.expanduser().resolve(),
                    local_dinov3_bundle=args.videomt_dinov3_bundle.expanduser().resolve(),
                    adapted_checkpoint_path=args.videomt_adapted_checkpoint.expanduser().resolve(),
                    adapted_checkpoint_sha256=args.videomt_adapted_checkpoint_sha256,
                    num_frames=5,
                ),
                device=videomt_preload_device,
                dtype=torch.float32,
            )
            videomt_runtime.requires_grad_(False).eval()
            if _videomt_stage_pqrf_active(args):
                source_mask_refiner = FrozenVidEoMTPosteriorRowRefiner(
                    videomt_runtime.model,
                    query_integration=(
                        REPLACE_WITH_RELEASED_PROPAGATION
                        if _adr205_released_query_propagation_active(args)
                        else PREPEND_PROJECTED_ROWS
                    ),
                )
                source_mask_refiner_sha256 = _module_state_digest(source_mask_refiner)
                source_mask_refiner_parameter_count = sum(
                    parameter.numel() for parameter in source_mask_refiner.parameters()
                )
            else:
                source_mask_head_sha256 = _module_state_digest(videomt_runtime.model.mask_head)
                source_mask_head = deepcopy(videomt_runtime.model.mask_head)
                source_mask_head.requires_grad_(False).eval()
        if args.posterior_architecture in {"layerwise_v2", "two_pass_v3"}:
            prediction_address_width = 0
            predictive_target_widths: tuple[tuple[str, int], ...] = (
                ()
                if current_contract is None
                else ((LINGBOT_PREDICTIVE_TARGET_SPACE, current_contract.hidden_size),)
            )
        else:
            if predictive_contract is None:
                raise RuntimeError("legacy posterior requires its predictive cache contract")
            prediction_address_width = 2
            predictive_target_widths = (
                (LINGBOT_PREDICTIVE_TARGET_SPACE, predictive_contract.hidden_size),
            )
        modality_specs = tuple(
            sorted(
                (
                    *(binding.native_spec for binding in dense_evidence_bindings),
                    *(
                        (
                            NativeModalitySpec(
                                name="proprioception",
                                input_width=55,
                                maximum_tokens=1,
                            ),
                        )
                        if _two_pass_filter_active(args)
                        else ()
                    ),
                ),
                key=lambda spec: spec.name,
            )
        )
        if _videomt_stage_pq_active(args):
            modality_specs = (
                with_videomt_row_mask_query_modality_spec(modality_specs)
                if _videomt_stage_pqmr_active(args)
                else with_videomt_query_modality_spec(modality_specs)
            )
        relation_surface_specs = tuple(
            spec
            for spec in (
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
            if spec.name in {modality.name for modality in modality_specs}
        )
        object_query_spatial_specs: tuple[NativeObjectQuerySpatialSpec, ...] = (
            (
                NativeObjectQuerySpatialSpec(
                    name="videomt_masks",
                    query_modality="videomt_queries",
                    geometry_kind="image_grid",
                    layout=CALVIN_VIDEOMT_MASK_LAYOUT,
                    target_kind=CALVIN_VIDEOMT_VISIBLE_OWNER_TARGET,
                ),
            )
            if _videomt_stage_pqm_active(args)
            else ()
        )
        adr177_task_addressed = _adr177_task_addressed_full_modal_active(args)
        adr178_direct_action_posterior = _adr178_direct_action_posterior_active(args)
        adr207_native_query = _adr207_native_videomt_query_posterior_active(args)
        adr225_object_memory = _adr225_pretrained_object_memory_active(args)
        adr221_wsa_edge_diagnostic = _adr221_wsa_edge_diagnostic_active(args)
        graph_config = LingBotNativeGraphConfig.from_policy(
            policy,
            capacity=args.capacity,
            maximum_control_tokens=args.maximum_control_tokens,
            task_query_count=(args.task_query_count if adr177_task_addressed else 0),
            prediction_address_width=prediction_address_width,
            predictive_target_widths=predictive_target_widths,
            modality_specs=modality_specs,
            modality_bridge_identity=args.dense_token_bridge,
            modality_bridge_query_count=(
                256 if args.dense_token_bridge == LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE else 0
            ),
            resampled_modality_names=(
                CALVIN_FULL_DENSE_MODALITIES
                if args.dense_token_bridge == LINGBOT_TASK_TOKEN_RESAMPLER_BRIDGE
                else ()
            ),
            direct_action_modality_names=(
                ("proprioception",)
                if (
                    posterior_adoption_route_active
                    or adr177_task_addressed
                    or adr178_direct_action_posterior
                )
                else ()
            ),
            relation_surface_specs=relation_surface_specs,
            object_query_spatial_specs=object_query_spatial_specs,
            relation_supervision_layers=args.relation_supervision_layers,
            architecture_identity=(
                NATIVE_VIDEOMT_PRETRAINED_OBJECT_MEMORY_POSTERIOR
                if adr225_object_memory
                else (
                    NATIVE_VIDEOMT_QUERY_POSTERIOR
                    if adr207_native_query
                    else (
                        LINGBOT_TASK_QUERY_OBJECT_VALUE_READ
                        if adr177_task_addressed
                        else (
                            UNIFIED_LAYERWISE_PREDICT_CORRECT
                            if _two_pass_filter_active(args)
                            else (
                                LAYERWISE_TASK_INDEPENDENT_ENTITY_POSTERIOR
                                if args.posterior_architecture == "layerwise_v2"
                                else TASK_INDEPENDENT_ENTITY_POSTERIOR
                            )
                        )
                    )
                )
            ),
        )
        graph = LingBotNativeGraph(
            graph_config,
            device=device,
            dtype=torch.float32,
            modality_bridge_projector=modality_bridge_projector,
            modality_bridge_queries=modality_bridge_queries,
            source_mask_head=source_mask_head,
            source_mask_refiner=source_mask_refiner,
        ).train()
        if source_mask_head is not None:
            if graph.relation_readout.source_mask_head is None or any(
                parameter.requires_grad
                for parameter in graph.relation_readout.source_mask_head.parameters()
            ):
                raise RuntimeError("source-faithful row-mask head was not frozen in the graph")
            if _module_state_digest(graph.relation_readout.source_mask_head) != (
                source_mask_head_sha256
            ):
                raise RuntimeError("copied row-mask head differs from the authenticated donor")
        if source_mask_refiner is not None:
            if graph.relation_readout.source_mask_refiner is None or any(
                parameter.requires_grad
                for parameter in graph.relation_readout.source_mask_refiner.parameters()
            ):
                raise RuntimeError("complete posterior row refiner was not frozen in the graph")
            if _module_state_digest(graph.relation_readout.source_mask_refiner) != (
                source_mask_refiner_sha256
            ):
                raise RuntimeError("copied posterior row refiner differs from the donor")
        registered_action_posterior_layer_indices = (
            _adr178_registered_layer_indices(graph_config.num_layers)
            if adr178_direct_action_posterior
            else ()
        )
        del (
            modality_bridge_projector,
            modality_bridge_queries,
            source_mask_head,
            source_mask_refiner,
        )
        install_lingbot_native_graph(policy, graph)
        object_memory_installation: dict[str, Any] = {
            "schema": "picf-next.pretrained-qwen3-object-memory-installation.v1",
            "active": False,
        }
        if adr225_object_memory:
            bridge = graph.pretrained_object_memory
            if bridge is None:
                raise RuntimeError("ADR-225 graph omitted its pretrained object memory")
            object_memory_installation = {
                "active": True,
                **bridge.installation_receipt(),
            }
        wla_installation: dict[str, Any] = {
            "schema": "picf-next.adr224-lingbot-wla-installation.v1",
            "active": False,
        }
        wla_target_transform = None
        if _wla_complete_active(args):
            if config.chunk_size != WLA_SOURCE_ACTION_HORIZON:
                raise RuntimeError(
                    "complete WLA action/world horizon differs from its pinned 8-step source"
                )
            wla_installation = {
                "active": True,
                **install_lingbot_wla_backend(
                    policy,
                    source_root=args.wla_source_root.expanduser().resolve(),
                    pretrained_root=args.wla_pretrained_root.expanduser().resolve(),
                    tokenizer=processor,
                    chunk_size=config.chunk_size,
                    device=device,
                    dtype=torch.bfloat16,
                ),
            }
            wla_target_transform = build_wla_target_transform(
                args.wla_source_root.expanduser().resolve()
            )
        wsa_training_runtime = None
        wsa_installation = {
            "schema": "picf-next.adr218-wsa-lingbot-installation.v1",
            "active": False,
        }
        if _adr221_full_source_wsa_active(args):
            wsa_future = WSAFutureExpertRuntime.from_adapted_checkpoint(
                source_root=args.wsa_source_root.expanduser().resolve(),
                checkpoint=args.wsa_adapted_checkpoint.expanduser().resolve(),
                device=device,
                dtype=torch.float32,
            )
            wsa_training_runtime = WSALingBotTrainingRuntime(
                wsa_future,
                action_coupling=(
                    WSALingBotActionCoupling.AUXILIARY_WORLD_DECODER
                    if _adr222_world_token_adoption_active(args)
                    else WSALingBotActionCoupling.DIRECT_FUTURE_KEYS
                ),
            )
            install_wsa_lingbot_training_runtime(policy, wsa_training_runtime)
            wsa_installation = {
                "active": True,
                **wsa_lingbot_installation_receipt(policy),
            }
        future_latent_alignment = None
        if future_latent_config is not None:
            future_latent_alignment = LingBotFutureLatentAlignment(
                future_latent_config
            ).to(device=device, dtype=torch.float32).train()
            install_lingbot_future_latent_alignment(
                policy,
                future_latent_alignment,
                require_adr209_complete=True,
            )
        trainable_scope_receipt = _trainable_scope_receipt(
            policy,
            scope=args.trainable_scope,
        )
        model_family_sha256 = _canonical_digest(
            {
                "checkpoint_revision": LINGBOT_CHECKPOINT_REVISION,
                "source_commit": LINGBOT_NATIVE_SOURCE_COMMIT,
                "graph_config": asdict(graph_config),
                "dense_evidence_bindings_sha256": dense_evidence_bindings_sha256,
                "videomt_stage_pq_assets": videomt_stage_pq_asset_receipt,
                "patch_sha256": source_overlay_sha256,
                "lingbot_compile_mode": args.lingbot_compile_mode,
                "future_latent_alignment_config": (
                    None if future_latent_config is None else asdict(future_latent_config)
                ),
                "future_latent_cache_manifest_sha256": (
                    future_latent_cache_manifest_sha256
                ),
                "future_latent_cache_build_report_sha256": (
                    future_latent_build_report_sha256
                ),
                "adr221_wsa_da3_assets": adr221_asset_receipt,
                "adr221_wsa_installation": wsa_installation,
                "adr224_wla_installation": wla_installation,
                "adr225_object_memory_installation": object_memory_installation,
            }
        )
        selective_class_offload_active = args.fsdp2_placement in {
            FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
            FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
        }
        selective_cpu_module_classes = tuple(
            dict.fromkeys(
                (
                    *(
                        (WSA_FSDP_BLOCK_CLASS, WSA_FSDP_EXPERT_CLASS)
                        if _adr221_full_source_wsa_active(args)
                        and selective_class_offload_active
                        else ()
                    ),
                    *(
                        (
                            WLA_HOST_TEXT_BLOCK_CLASS,
                            WLA_ACTION_BLOCK_CLASS,
                            WLA_WORLD_BLOCK_CLASS,
                        )
                        if _wla_complete_active(args)
                        and selective_class_offload_active
                        else ()
                    ),
                )
            )
        )
        selective_cpu_parameter_prefixes = tuple(
            dict.fromkeys(
                (
                    *(
                        (WSA_FSDP_PARAMETER_PREFIX,)
                        if _adr221_full_source_wsa_active(args)
                        and selective_class_offload_active
                        else ()
                    ),
                    *(
                        (
                            WLA_HOST_TEXT_FSDP_PARAMETER_PREFIX,
                            WLA_ACTION_FSDP_PARAMETER_PREFIX,
                            WLA_WORLD_FSDP_PARAMETER_PREFIX,
                        )
                        if _wla_complete_active(args)
                        and selective_class_offload_active
                        else ()
                    ),
                )
            )
        )
        policy = build_parallelize_model(
            policy,
            enable_full_shard=True,
            enable_mixed_precision=optimizer_contract.enable_mixed_precision,
            enable_fp32=(
                False if _wla_complete_active(args) else optimizer_contract.enable_fp32
            ),
            enable_gradient_checkpointing=True,
            init_device="cuda",
            enable_fsdp_offload=(args.fsdp2_placement == FSDP2_CPU_OFFLOAD),
            enable_shared_embedding_offload=(
                args.fsdp2_placement
                in {
                    FSDP2_SELECTIVE_EMBEDDING_OFFLOAD,
                    FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD,
                    FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD,
                }
            ),
            selective_cpu_module_classes=selective_cpu_module_classes,
            enable_frozen_vision_offload=(
                args.fsdp2_placement
                == FSDP2_SELECTIVE_EMBEDDING_FROZEN_VISION_OFFLOAD
            ),
            enable_trainable_vision_offload=(
                args.fsdp2_placement
                == FSDP2_SELECTIVE_EMBEDDING_TRAINABLE_VISION_OFFLOAD
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
        wla_fsdp_forward = None
        if _wla_complete_active(args):
            wla_fsdp_forward = register_lingbot_wla_fsdp_forward(policy)
        wsa_fsdp_forward_methods = None
        if _adr221_full_source_wsa_active(args):
            wsa_fsdp_forward_methods = register_wsa_lingbot_fsdp_forward_methods(
                policy
            )
        backward_prefetch = configure_fsdp2_backward_prefetch(
            policy,
            mode=args.fsdp2_backward_prefetch,
        )
        parameter_storage = _validate_fsdp2_parameter_storage(
            policy,
            torch,
            expected_placement=args.fsdp2_placement,
            expected_selective_cpu_module_classes=selective_cpu_module_classes,
            expected_selective_cpu_parameter_prefixes=(
                selective_cpu_parameter_prefixes
            ),
        )
        fsdp2_parameter_layout = _distributed_fsdp2_parameter_layout_contract(
            policy,
            rank=rank,
            dist=dist,
        )
        action_topology = (
            audit_lingbot_wla_fsdp_topology(policy)
            if _wla_complete_active(args)
            else _validate_action_fsdp2_topology(policy)
        )
        vlm_topology = _validate_vlm_fsdp2_topology(policy)
        if args.lingbot_compile_mode == LINGBOT_COMPILE_UPSTREAM_DEFAULT:
            policy = torch.compile(policy)
        lingbot_compile_receipt = {
            "mode": args.lingbot_compile_mode,
            "enabled": args.lingbot_compile_mode == LINGBOT_COMPILE_UPSTREAM_DEFAULT,
            "ordering": "fsdp2_then_whole_model_compile_then_optimizer",
            "backend": "torch_compile_upstream_default",
        }
        optimizer = (
            build_lingbot_wla_optimizer(policy)
            if _wla_complete_active(args)
            else build_lingbot_official_optimizer(
                policy,
                optimizer_contract,
                build_muon_optimizer=build_muon_optimizer,
                build_moe_load_balance_hook=build_moe_load_balance_hook,
            )
        )
        wla_scheduler = None
        wla_optimizer_receipt = None
        if _wla_complete_active(args):
            wla_scheduler = build_lingbot_wla_scheduler(optimizer)
            wla_optimizer_receipt = audit_lingbot_wla_optimizer(
                policy,
                optimizer,
                wla_scheduler,
            )
        wsa_donor_optimizer = None
        wsa_scheduler = None
        wsa_optimizer_receipt = None
        wsa_scheduler_receipt = None
        wsa_teacher_runtime = None
        if _adr221_full_source_wsa_active(args):
            wsa_donor_optimizer = install_wsa_lingbot_optimizer(policy, optimizer)
            # LinearLR mutates the live optimizer LR during construction. Audit
            # the donor's static AdamW contract before attaching that scheduler.
            wsa_optimizer_receipt = audit_wsa_lingbot_optimizer(policy, optimizer)
            wsa_scheduler = build_wsa_lingbot_scheduler(
                wsa_donor_optimizer,
                total_train_steps=TOTAL_STEPS,
            )
            wsa_scheduler_receipt = audit_wsa_lingbot_scheduler(
                wsa_scheduler,
                wsa_donor_optimizer,
                total_train_steps=TOTAL_STEPS,
            )
            wsa_teacher_runtime = OnlineWSADA3TeacherRuntime.from_official_source(
                wsa_source_root=args.wsa_source_root.expanduser().resolve(),
                da3_model_dir=args.da3_model_dir.expanduser().resolve(),
                da3_source_root=args.da3_source_root.expanduser().resolve(),
            )
        learning_rate_stratification = (
            configure_picf_optimizer_learning_rates(
                optimizer,
                graph,
                picf_multiplier=args.picf_learning_rate_multiplier,
                modality_bridge_multiplier=args.modality_bridge_learning_rate_multiplier,
            )
            if (
                _picf_optimizer_learning_rate_stratification_active(args)
                and not _wla_complete_active(args)
            )
            else {
                "schema": "picf-next.optimizer-learning-rate-stratification/v1",
                "enabled": False,
                "multipliers": {
                    "lingbot_host": 1.0,
                    "picf_graph": 1.0,
                    "pretrained_modality_bridge": 1.0,
                },
            }
        )
        parameter_manifest = audit_native_optimizer_coverage(
            modules={"policy": policy},
            optimizer=optimizer,
        )
        checkpointer = build_checkpointer(dist_backend="fsdp2", ckpt_manager="dcp")
        videomt_stage_p: VidEoMTStageP | VidEoMTStagePQM | VidEoMTStagePQMR | None = None
        videomt_optimizer = None
        videomt_scheduler = None
        videomt_optimizer_receipt = None
        videomt_fsdp2_receipt = None
        videomt_source_objective = None
        videomt_stage_pq_runtime_receipt: dict[str, object] = {
            "schema": "picf-next.videomt-stage-pq-runtime/v1",
            "mode": VIDEOMT_STAGE_PQ_DISABLED,
            "active": False,
        }
        if _videomt_stage_pq_active(args):
            videomt_initial_device = (
                torch.device("cpu")
                if args.videomt_idle_placement
                == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                else device
            )
            if videomt_runtime is None:
                videomt_runtime = load_exact_videomt(
                    ExactVidEoMTConfig(
                        checkpoint_path=args.videomt_checkpoint.expanduser().resolve(),
                        local_dinov3_bundle=args.videomt_dinov3_bundle.expanduser().resolve(),
                        adapted_checkpoint_path=(
                            args.videomt_adapted_checkpoint.expanduser().resolve()
                            if _videomt_stage_pq_adapted_active(args)
                            else None
                        ),
                        adapted_checkpoint_sha256=(
                            args.videomt_adapted_checkpoint_sha256
                            if _videomt_stage_pq_adapted_active(args)
                            else None
                        ),
                        num_frames=5,
                    ),
                    device=videomt_initial_device,
                    dtype=torch.float32,
                )
            if _videomt_native_trainable_active(args):
                videomt_runtime.requires_grad_(True).train()
                parallelized_source, videomt_fsdp2_receipt = parallelize_exact_videomt_fsdp2(
                    videomt_runtime.model,
                    parameter_dtype=torch.bfloat16,
                    reduction_dtype=torch.float32,
                    output_dtype=torch.bfloat16,
                    cpu_offload=(
                        args.videomt_fsdp2_placement
                        == VIDEOMT_FSDP2_PLACEMENT_CPU_OFFLOAD
                    ),
                )
                if parallelized_source is not videomt_runtime.model:
                    raise RuntimeError("VidEoMT FSDP2 replaced the authenticated source instance")
                videomt_optimizer, videomt_optimizer_receipt = build_exact_videomt_optimizer(
                    videomt_runtime.model
                )
                videomt_scheduler = build_exact_videomt_scheduler(
                    videomt_optimizer,
                    videomt_optimizer_receipt,
                    total_steps=VIDEOMT_ADAPTATION_BUDGET_STEPS,
                )
                videomt_source_objective = CompleteCalvinVidEoMTObjective().to(device).train()
                videomt_parameters = tuple(videomt_runtime.model.parameters())
                if (
                    not videomt_parameters
                    or any(not parameter.requires_grad for parameter in videomt_parameters)
                    or not videomt_runtime.training
                    or not videomt_runtime.model.training
                ):
                    raise RuntimeError("ADR-207 requires the complete trainable VidEoMT source")
            else:
                videomt_runtime.requires_grad_(False)
                videomt_runtime.eval()
                if _videomt_stage_pqmr_active(args):
                    videomt_stage_p = VidEoMTStagePQMR(videomt_runtime).eval()
                elif _videomt_stage_pqm_active(args):
                    videomt_stage_p = VidEoMTStagePQM(videomt_runtime).eval()
                else:
                    videomt_stage_p = VidEoMTStageP(videomt_runtime).eval()
                videomt_parameters = tuple(videomt_stage_p.parameters())
                if not videomt_parameters or any(
                    parameter.requires_grad for parameter in videomt_parameters
                ):
                    raise RuntimeError("complete VidEoMT donor is not frozen")
                if any(parameter.dtype != torch.float32 for parameter in videomt_parameters):
                    raise RuntimeError("complete VidEoMT donor is not running in FP32")
                if videomt_stage_p.training or videomt_stage_p.runtime.model.training:
                    raise RuntimeError("complete VidEoMT donor is not in released evaluation mode")
            videomt_stage_pq_runtime_receipt = {
                "schema": "picf-next.videomt-stage-pq-runtime/v1",
                "mode": args.videomt_stage_pq_mode,
                "active": True,
                "interface_identity": VIDEOMT_STAGE_PQ_C5_INTERFACE,
                "parameter_tensor_count": len(videomt_parameters),
                "parameter_numel": sum(parameter.numel() for parameter in videomt_parameters),
                "parameter_dtype": (
                    "FSDP2_mixed_bfloat16_parameter_float32_reduction"
                    if _videomt_native_trainable_active(args)
                    else "torch.float32"
                ),
                "training": _videomt_native_trainable_active(args),
                "requires_grad": _videomt_native_trainable_active(args),
                "optimizer_membership": _videomt_native_trainable_active(args),
                "fsdp2": (
                    asdict(videomt_fsdp2_receipt)
                    if videomt_fsdp2_receipt is not None
                    else None
                ),
                "optimizer": (
                    asdict(videomt_optimizer_receipt)
                    if videomt_optimizer_receipt is not None
                    else None
                ),
                "released_final_outputs_executed": [
                    "class_logits",
                    "mask_logits",
                    "query_embeddings",
                    "propagated_queries",
                    *(
                        ["mask_embeddings", "dense_mask_features"]
                        if _videomt_stage_pqmr_active(args)
                        else []
                    ),
                    *(
                        [
                            "segmenter_input_tokens",
                            "position_cos",
                            "position_sin",
                            "patch_grid_shape",
                        ]
                        if _videomt_stage_pqrf_active(args)
                        else []
                    ),
                ],
                "released_training_only_auxiliary_outputs_active": (
                    _videomt_native_trainable_active(args)
                ),
                "posterior_row_mask_decoder": (
                    {
                        "identity": (
                            "deepcopy_of_complete_released_blocks20_23_norm_class_mask_upscale"
                            if _videomt_stage_pqrf_active(args)
                            else "deepcopy_of_complete_released_mask_head"
                        ),
                        "state_sha256": (
                            source_mask_refiner_sha256
                            if _videomt_stage_pqrf_active(args)
                            else source_mask_head_sha256
                        ),
                        "parameter_numel": (
                            source_mask_refiner_parameter_count
                            if _videomt_stage_pqrf_active(args)
                            else sum(
                                parameter.numel()
                                for parameter in (
                                    graph.relation_readout.source_mask_head.parameters()
                                )
                            )
                        ),
                        "requires_grad": False,
                        "training": False,
                        "semantic_query_projection_tied_transpose": True,
                        "retains_complete_source_query_bank_prefix_and_patch_stream": (
                            _videomt_stage_pqrf_active(args)
                        ),
                    }
                    if _videomt_stage_pqmr_active(args)
                    else None
                ),
                "weight_origin": (
                    "complete_calvin_adapted_checkpoint"
                    if _videomt_stage_pq_adapted_active(args)
                    else "published_youtube_vis_checkpoint"
                ),
                "adapted_checkpoint": (
                    asdict(videomt_runtime.adapted_checkpoint_receipt)
                    if videomt_runtime.adapted_checkpoint_receipt is not None
                    else None
                ),
                "host_injected_output": (
                    "qwen3_native_merger_mask_pooled_object_memory.latest_all_200"
                    if adr225_object_memory
                    else (
                        VIDEOMT_STAGE_PQRF_HOST_OUTPUT
                        if _videomt_stage_pqrf_active(args)
                        else (
                            VIDEOMT_STAGE_PQMR_HOST_OUTPUT
                            if _videomt_stage_pqmr_active(args)
                            else (
                                VIDEOMT_STAGE_PQM_HOST_OUTPUT
                                if _videomt_stage_pqm_active(args)
                                else "query_embeddings.latest_all_200"
                            )
                        )
                    )
                ),
                "host_dtype": "torch.bfloat16",
                "host_selection_pooling_resampling_or_second_norm": (
                    adr225_object_memory
                ),
                "host_selection": False,
                "host_pooling": (
                    "unipixel_native_merger_input_mask_mean"
                    if adr225_object_memory
                    else None
                ),
                "host_resampling": False,
                "host_second_norm": False,
                "pretrained_object_memory": object_memory_installation,
                "execution_device": str(device),
                "initial_device": str(videomt_initial_device),
                "idle_placement": args.videomt_idle_placement,
                "fsdp2_placement": args.videomt_fsdp2_placement,
                "idle_device": (
                    "cpu"
                    if args.videomt_idle_placement
                    == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                    else str(device)
                ),
                "placement_changes_tensor_semantics": False,
                "cpu_idle_clears_released_recurrent_state": (
                    args.videomt_idle_placement
                    == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                ),
                "cpu_idle_trims_cuda_allocator_cache": (
                    args.videomt_idle_placement
                    == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                ),
            }
        prior_stepper = (
            None
            if args.posterior_architecture in {"layerwise_v2", "two_pass_v3"}
            else LingBotNativePriorStepper(policy, graph)
        )
        addressed_codebook_sha256 = None
        if adr177_task_addressed:
            if graph.episode_address_codebook is None:
                raise RuntimeError("ADR-177 graph omitted its immutable address codebook")
            addressed_codebook_sha256 = address_codebook_sha256(graph.episode_address_codebook)
        lane_config = NativeLaneConfig(
            model_digest=model_family_sha256,
            schema_digest=stream_plan.plan_sha256,
            capacity=args.capacity,
            host_width=graph_config.host_width,
            maximum_optimizer_lag=args.maximum_optimizer_lag,
            num_layers=(
                graph_config.num_layers
                if args.posterior_architecture in {"layerwise_v2", "two_pass_v3"}
                else None
            ),
            addressed_architecture_identity=(
                graph_config.architecture_identity if adr177_task_addressed else None
            ),
            episode_address_codebook_sha256=addressed_codebook_sha256,
            paired_source_width=(1024 if adr207_native_query else None),
            paired_architecture_identity=(
                graph_config.architecture_identity if adr207_native_query else None
            ),
            paired_source_dtype=(torch.bfloat16 if adr207_native_query else torch.float32),
            device=str(device),
            dtype=torch.bfloat16,
        )

        global_step = 0
        latest_checkpoint_global_step = 0
        loaded_boundary = None
        if args.phase == "fresh":
            bank = NativeTrainingLaneBank(lane_config)
        else:
            checkpoint_dir = args.run_dir / "checkpoints" / f"global_step_{args.load_global_step}"
            if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
                raise FileNotFoundError(checkpoint_dir)
            state = {"model": policy, "optimizer": optimizer, "extra_state": {}}
            if adr207_native_query:
                if (
                    videomt_runtime is None
                    or videomt_optimizer is None
                    or videomt_scheduler is None
                ):
                    raise RuntimeError("ADR-207 resume lacks complete source training state")
                state.update(
                    {
                        "videomt_model": videomt_runtime.model,
                        "videomt_optimizer": videomt_optimizer,
                        "videomt_scheduler": videomt_scheduler,
                    }
                )
            if _adr221_full_source_wsa_active(args):
                if wsa_scheduler is None:
                    raise RuntimeError("ADR-221 resume lacks the WSA scheduler")
                state["wsa_scheduler"] = wsa_scheduler
            if _wla_complete_active(args):
                if wla_scheduler is None:
                    raise RuntimeError("complete WLA resume lacks its scheduler")
                state["wla_scheduler"] = wla_scheduler
            checkpointer.load(str(checkpoint_dir), state)
            extra = _validate_resume_extra(
                state["extra_state"],
                rank=rank,
                global_step=args.load_global_step,
                implementation_sha256=implementation_sha256,
                model_family_sha256=model_family_sha256,
                stream_plan_sha256=stream_plan.plan_sha256,
                temporal_sha256=temporal_config.digest,
                predictive_manifest_sha256=predictive_manifest_sha256,
                current_manifest_sha256=current_manifest_sha256,
                evidence_sha256=evidence_sha256,
                execution_contract_sha256=execution_contract_sha256,
                joint_source_active=adr207_native_query,
                wsa_active=_adr221_full_source_wsa_active(args),
                wla_active=_wla_complete_active(args),
            )
            bank = NativeTrainingLaneBank.deserialize(lane_config, extra["lane_snapshot"])
            _restore_rank_rng(extra["rank_rng_state"], torch, np, device=device)
            loaded_boundary = _checkpoint_boundary(
                model=policy,
                optimizer=optimizer,
                lane_snapshot=bank.serialize(),
                rank_rng_state=extra["rank_rng_state"],
                torch_module=torch,
            )
            if adr207_native_query:
                source_boundary = _checkpoint_boundary(
                    model=videomt_runtime.model,
                    optimizer=videomt_optimizer,
                    lane_snapshot=bank.serialize(),
                    rank_rng_state=extra["rank_rng_state"],
                    torch_module=torch,
                )
                loaded_boundary = {
                    **loaded_boundary,
                    "source_model_local_state_sha256": source_boundary[
                        "model_local_state_sha256"
                    ],
                    "source_optimizer_local_state_sha256": source_boundary[
                        "optimizer_local_state_sha256"
                    ],
                }
            if loaded_boundary != extra["boundary_sha256"]:
                raise RuntimeError("restored checkpoint boundary differs")
            _validate_optimizer_state(optimizer, torch, expected_step=args.load_global_step)
            if adr207_native_query:
                _validate_optimizer_state(
                    videomt_optimizer,
                    torch,
                    expected_step=args.load_global_step,
                )
                if int(videomt_scheduler.last_epoch) != args.load_global_step:
                    raise RuntimeError("restored source scheduler step differs")
            if _adr221_full_source_wsa_active(args):
                if (
                    wsa_scheduler is None
                    or int(wsa_scheduler.last_epoch) != args.load_global_step
                ):
                    raise RuntimeError("restored WSA scheduler step differs")
            if _wla_complete_active(args):
                if (
                    wla_scheduler is None
                    or int(wla_scheduler.last_epoch) != args.load_global_step
                ):
                    raise RuntimeError("restored WLA scheduler step differs")
            global_step = args.load_global_step
            latest_checkpoint_global_step = args.load_global_step

        if args.phase == "fresh":
            rank_seed = args.seed + rank
            random.seed(rank_seed)
            np.random.seed(rank_seed)
            torch.manual_seed(rank_seed)
            torch.cuda.manual_seed(rank_seed)
        else:
            resume_publication_error: list[str | None] = [None]
            if rank == 0:
                try:
                    _prune_resume_publications(
                        args.run_dir,
                        load_global_step=args.load_global_step,
                    )
                except BaseException as error:
                    resume_publication_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(resume_publication_error, src=0)
            if resume_publication_error[0] is not None:
                raise RuntimeError(
                    f"resume publication pruning failed: {resume_publication_error[0]}"
                )

        data_config_payload = json.loads(args.data_config.read_text(encoding="utf-8"))
        if data_config_payload.get("cameras") != [
            slot.runtime_camera_name for slot in LINGBOT_CALVIN_CAMERA_SLOTS
        ]:
            raise RuntimeError("LingBot CALVIN camera-slot configuration changed")
        feature_transform = FeatureTransform(
            str(args.robot_config.resolve()),
            official_lingbot_data_config(data_config_payload),
            config,
            processor,
            chunk_size=config.chunk_size,
            norm_stats_path=str(args.norm_stats.resolve()),
            use_depth_align=False,
            image_augment=False,
            use_future_image=False,
        )
        collator = VLADataCollatorWithPacking()
        videomt_stage_pq_eligible_collations = 0
        videomt_stage_pq_ineligible_short_prefix_collations = 0
        videomt_stage_pq_cuda_loads = 0
        videomt_stage_pq_cpu_offloads = 0
        videomt_stage_pq_placement_seconds = 0.0
        videomt_stage_pq_current_device = (
            "cpu"
            if _videomt_stage_pq_active(args)
            and args.videomt_idle_placement
            == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
            else str(device)
        )
        videomt_stage_pq_execution_receipt_path = (
            args.run_dir / "receipts" / f"videomt_stage_pq_rank_{rank}.json"
        )
        videomt_stage_pq_first_execution_receipt_written = (
            videomt_stage_pq_execution_receipt_path.is_file()
        )

        def source_global_index_by_sample_key(sample_key: str) -> int:
            try:
                return evaluation_dataset.source_global_index_by_key(sample_key)
            except KeyError:
                return dataset.source_global_index_by_key(sample_key)

        def dense_source_identity(sample_key: str) -> tuple[int, str]:
            source_global_index = source_global_index_by_sample_key(sample_key)
            try:
                canonical_key = canonical_dense_key_by_source_index[source_global_index]
            except KeyError as error:
                raise RuntimeError(
                    "runtime CALVIN sample has no canonical dense-evidence identity"
                ) from error
            if source_global_index_by_sample_key(canonical_key) != source_global_index:
                raise RuntimeError(
                    "canonical dense-evidence key resolves to another CALVIN source frame"
                )
            return source_global_index, canonical_key

        def source_timestamp_s(source_global_index: int) -> float:
            source_episode = index.source_episode(source_global_index)
            return (source_global_index - source_episode.start) / float(index.control_hz)

        adr207_intervention_optimizer_step: int | None = None
        if adr207_native_query and any(
            step <= args.stop_after_step for step in ADR207_MODALITY_INTERVENTION_STEPS
        ):
            if dense_evidence_bank is None:
                raise RuntimeError("ADR-207 intervention gate lacks dense evidence")
            records = {
                cache.contract.modality: {
                    record.source_global_index: record for record in cache.records
                }
                for cache in dense_evidence_bank.caches
            }
            if dense_evidence_coverage is None:
                raise RuntimeError("ADR-207 intervention gate lacks dense coverage")
            covered_steps = (
                dense_evidence_coverage.training_visit_count
                // stream_plan.global_batch_size
            )
            for candidate_step in range(covered_steps):
                transitions = stream_plan.global_batch(candidate_step).transitions
                source_indices = tuple(
                    dense_source_identity(transition.sample.sample_key)[0]
                    for transition in transitions
                )
                if all(
                    all(
                        (record := records[modality].get(source_index)) is not None
                        and record.available
                        and record.token_count >= 2
                        for modality in CALVIN_FULL_DENSE_MODALITIES
                    )
                    for source_index in source_indices
                ):
                    adr207_intervention_optimizer_step = candidate_step
                    break
            if adr207_intervention_optimizer_step is None:
                raise RuntimeError(
                    "ADR-207 frozen stream has no global batch with full dense evidence"
                )

        def collate_planned(planned: Any) -> Any:
            nonlocal videomt_stage_pq_eligible_collations
            nonlocal videomt_stage_pq_ineligible_short_prefix_collations
            nonlocal videomt_stage_pq_first_execution_receipt_written
            nonlocal videomt_stage_pq_cuda_loads
            nonlocal videomt_stage_pq_cpu_offloads
            nonlocal videomt_stage_pq_placement_seconds
            nonlocal videomt_stage_pq_current_device
            collated = collate_native_calvin_training_batch(
                planned.training,
                feature_transform=feature_transform,
                collator=collator,
                augmentation_seeds=planned.augmentation_seeds,
                source_digest=planned.source_digest,
            )
            _validate_lingbot_calvin_projection_batch(
                collated.model_inputs,
                projection=training_projection,
            )
            collated = materialize_native_flow_randomness(
                _collated_to_device(collated, device=device, torch_module=torch),
                planned,
            )
            if dense_evidence_bank is not None:
                evidence_rows = []
                for sample_key in collated.routing.sample_keys:
                    source_global_index, canonical_key = dense_source_identity(sample_key)
                    evidence_row = dense_evidence_bank.evidence_for(
                        source_global_index=source_global_index,
                        sample_key=canonical_key,
                    )
                    validate_calvin_evidence_timestamps(
                        evidence_row,
                        observation_timestamp_s=source_timestamp_s(source_global_index),
                    )
                    evidence_rows.append(evidence_row)
                collated = with_native_modalities(
                    collated,
                    native_modalities_from_dense_evidence(
                        tuple(evidence_rows),
                        dense_evidence_bindings,
                        device=device,
                        dtype=torch.bfloat16,
                    ),
                )
            if _videomt_stage_pq_active(args) and not _videomt_native_trainable_active(args):
                if videomt_stage_p is None:
                    raise RuntimeError("enabled VidEoMT Stage-PQ runtime is absent")
                if len(collated.routing.sample_keys) != 1:
                    raise RuntimeError(
                        "VidEoMT Stage-PQ requires the frozen one-sample-per-rank contract"
                    )
                sample_key = collated.routing.sample_keys[0]
                source_global_index = source_global_index_by_sample_key(sample_key)
                try:
                    prepared_videomt = prepare_calvin_stage_pq_c5(
                        physical_dataset.index,
                        source_global_index,
                    )
                except InsufficientCausalPrefixError:
                    videomt_modalities = empty_videomt_query_modality_batch(
                        batch_size=1,
                        device=device,
                        dtype=torch.bfloat16,
                        include_mask_metadata=_videomt_stage_pqmr_active(args),
                    )
                    videomt_stage_pq_ineligible_short_prefix_collations += 1
                else:
                    if (
                        args.videomt_idle_placement
                        == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                    ):
                        placement_started = time.monotonic()
                        videomt_stage_p.to(device=device)
                        videomt_stage_p.eval()
                        videomt_stage_pq_placement_seconds += (
                            time.monotonic() - placement_started
                        )
                        videomt_stage_pq_cuda_loads += 1
                        videomt_stage_pq_current_device = str(device)
                        if any(
                            parameter.device != device
                            or parameter.dtype != torch.float32
                            or parameter.requires_grad
                            for parameter in videomt_stage_p.parameters()
                        ):
                            raise RuntimeError(
                                "VidEoMT donor CUDA placement changed its frozen FP32 contract"
                            )
                    videomt_input = prepared_videomt.frames.model_input.to(
                        device=device,
                        dtype=torch.float32,
                    )
                    with torch.no_grad():
                        videomt_result = videomt_stage_p(
                            videomt_input,
                            host_dtype=torch.bfloat16,
                            resume=False,
                        )
                    videomt_modalities = videomt_result.modalities
                    videomt_stage_pq_eligible_collations += 1
                    if not videomt_stage_pq_first_execution_receipt_written:
                        execution_receipt = make_videomt_stage_pq_execution_receipt(
                            prepared_videomt,
                            videomt_result.upstream,
                            host_dtype=torch.bfloat16,
                            host_injected_output=(
                                VIDEOMT_STAGE_PQRF_HOST_OUTPUT
                                if _videomt_stage_pqrf_active(args)
                                else (
                                    VIDEOMT_STAGE_PQMR_HOST_OUTPUT
                                    if _videomt_stage_pqmr_active(args)
                                    else (
                                        VIDEOMT_STAGE_PQM_HOST_OUTPUT
                                        if _videomt_stage_pqm_active(args)
                                        else "query_embeddings.latest_all_200"
                                    )
                                )
                            ),
                        )
                        receipt_payload = {
                            "schema": "picf-next.videomt-stage-pq-first-execution/v1",
                            "rank": rank,
                            "sample_key": sample_key,
                            "assets": videomt_stage_pq_asset_receipt,
                            "runtime": videomt_stage_pq_runtime_receipt,
                            "execution": asdict(execution_receipt),
                        }
                        write_text_durable_exclusive(
                            videomt_stage_pq_execution_receipt_path,
                            json.dumps(receipt_payload, indent=2, sort_keys=True) + "\n",
                        )
                        videomt_stage_pq_first_execution_receipt_written = True
                    del videomt_input, videomt_result
                    if (
                        args.videomt_idle_placement
                        == VIDEOMT_IDLE_PLACEMENT_CPU_BETWEEN_FORWARDS
                    ):
                        videomt_stage_p.runtime.reset_state()
                        placement_started = time.monotonic()
                        videomt_stage_p.to(device=torch.device("cpu"))
                        videomt_stage_p.eval()
                        videomt_stage_pq_placement_seconds += (
                            time.monotonic() - placement_started
                        )
                        videomt_stage_pq_cpu_offloads += 1
                        videomt_stage_pq_current_device = "cpu"
                        if any(
                            parameter.device.type != "cpu"
                            or parameter.dtype != torch.float32
                            or parameter.requires_grad
                            for parameter in videomt_stage_p.parameters()
                        ):
                            raise RuntimeError(
                                "VidEoMT donor CPU idle placement changed its frozen FP32 contract"
                            )
                        torch.cuda.empty_cache()
                collated = with_native_modalities(collated, videomt_modalities)
            if _two_pass_filter_active(args):
                collated = with_official_proprioception_modality(collated)
            if _wla_complete_active(args):
                if wla_target_transform is None:
                    raise RuntimeError("complete WLA target transform is absent")
                collated = with_wla_world_target(
                    collated,
                    build_wla_calvin_target_batch_from_source_indices(
                        index,
                        tuple(
                            source_global_index_by_sample_key(sample_key)
                            for sample_key in collated.routing.sample_keys
                        ),
                        action_horizon=config.chunk_size,
                        target_transform=wla_target_transform,
                    ).to(device=device, dtype=torch.bfloat16),
                )
            return collated

        objective_config = TaskIndependentEntityObjectiveConfig(
            action_weight=1.0,
            entity_weight=args.entity_weight,
            predictive_weight=args.predictive_weight,
            mask_focal_weight=args.mask_focal_weight,
            mask_dice_weight=args.mask_dice_weight,
            existence_weight=args.existence_weight,
            ownership_weight=args.ownership_weight,
        )
        entity_evaluation_objective_config = TaskIndependentEntityObjectiveConfig(
            action_weight=0.0,
            entity_weight=1.0,
            predictive_weight=0.0,
            mask_focal_weight=args.mask_focal_weight,
            mask_dice_weight=args.mask_dice_weight,
            existence_weight=args.existence_weight,
            ownership_weight=args.ownership_weight,
        )

        def snapshot_official_runtime_buffers() -> ActionBackendRuntimeBufferSnapshot:
            return _snapshot_action_backend_runtime_buffers(
                policy,
                action_backend=args.action_backend,
            )

        def restore_official_runtime_buffers(
            snapshot: ActionBackendRuntimeBufferSnapshot,
        ) -> None:
            _restore_action_backend_runtime_buffers(
                snapshot,
                action_backend=args.action_backend,
                torch_module=torch,
            )

        def capture_rank_execution_state() -> tuple[
            dict[str, Any],
            ActionBackendRuntimeBufferSnapshot,
        ]:
            return (
                _capture_rank_rng(torch, np, device=device),
                snapshot_official_runtime_buffers(),
            )

        def restore_rank_execution_state(
            state: tuple[dict[str, Any], ActionBackendRuntimeBufferSnapshot],
        ) -> None:
            rank_rng, runtime_buffers = state
            _restore_rank_rng(rank_rng, torch, np, device=device)
            restore_official_runtime_buffers(runtime_buffers)

        @contextmanager
        def preserve_official_runtime_buffers():
            snapshot = snapshot_official_runtime_buffers()
            try:
                yield
            finally:
                restore_official_runtime_buffers(snapshot)

        def omitted_static_checkpoint_contexts() -> tuple[Any, Any]:
            return nullcontext(), preserve_official_runtime_buffers()

        @contextmanager
        def suspend_official_gradient_checkpointing():
            enabled_modules = [
                module
                for module in policy.modules()
                if getattr(module, "gradient_checkpointing", False) is True
            ]
            if not enabled_modules:
                raise RuntimeError(
                    "save-on-cpu omitted-static offload found no enabled gradient checkpoints"
                )
            for module in enabled_modules:
                module.gradient_checkpointing = False
            try:
                yield
            finally:
                for module in enabled_modules:
                    module.gradient_checkpointing = True

        if tuple(CORE_OMITTED_STATIC_REMATERIALIZATION_MODES) != (
            OMITTED_STATIC_REMATERIALIZATION_MODES
        ):
            raise RuntimeError("runner and core omitted-static rematerialization modes differ")

        action_evaluation_snapshot_reports: list[dict[str, Any]] = []
        causal_warm_action_evaluation_snapshot_reports: list[dict[str, Any]] = []

        def cold_previous_state(collated_batch: Any) -> NativePersistentState | None:
            if not adr177_task_addressed:
                return None
            return native_cold_state_for_episode_keys(
                lane_config,
                episode_keys=collated_batch.routing.episode_keys,
            )

        def model_inputs_sha256(model_inputs: Mapping[str, Any]) -> str:
            digest = hashlib.sha256()
            for name in sorted(model_inputs):
                value = model_inputs[name]
                if not isinstance(value, torch.Tensor):
                    raise TypeError(f"action-evaluation model input is not a tensor: {name}")
                local = value.detach().to(device="cpu").contiguous()
                digest.update(name.encode("ascii"))
                digest.update(str(local.dtype).encode("ascii"))
                digest.update(json.dumps(list(local.shape), separators=(",", ":")).encode())
                digest.update(local.view(torch.uint8).numpy().tobytes())
            return digest.hexdigest()

        def finite_scalar(value: Any, *, name: str) -> float:
            measured = float(value.detach().float().item())
            if not math.isfinite(measured):
                raise RuntimeError(f"action evaluation produced non-finite {name}")
            return measured

        def run_cold_picf_action_forward(
            collated_batch: Any,
            *,
            prior_host_steps: int,
        ) -> tuple[Any, Any]:
            if graph is None or not graph.unified_predict_correct:
                raise RuntimeError("cold action evaluation requires the ADR-149 graph")
            source_valid = torch.zeros(
                collated_batch.routing.batch_size,
                dtype=torch.bool,
                device=collated_batch.controls.values.device,
            )
            source, _prediction = run_native_v3_prior_chain(
                policy,
                graph=graph,
                previous_memory=cold_previous_state(collated_batch),
                previous_memory_valid=source_valid,
                control_chunks=collated_batch.effective_prior_control_chunks,
                filter_prediction=None,
                require_attached_memory=False,
                host_step_count=prior_host_steps,
                require_grad=False,
            )
            result = run_native_policy_diagnostic_forward(
                policy,
                model_inputs=collated_batch.model_inputs,
                context=native_context_from_prior_trace(
                    controls=collated_batch.controls,
                    prior_trace=source,
                    modalities=collated_batch.modalities,
                ),
            )
            return result, source

        def run_action_evaluation(checkpoint_global_step: int) -> dict[str, Any]:
            """Evaluate matched held-out inputs through the profile's cold state."""

            if evaluation_plan is None or representation_split is None:
                raise RuntimeError("matched action evaluation contract is absent")
            if checkpoint_global_step not in action_evaluation_steps:
                raise ValueError("matched action evaluation step is not registered")
            optimizer.zero_grad(set_to_none=True)
            if adr207_native_query:
                if videomt_optimizer is None:
                    raise RuntimeError("ADR-207 action evaluation lacks its source optimizer")
                videomt_optimizer.zero_grad(set_to_none=True)
            local_schedule = build_distributed_entity_evaluation_schedule(
                evaluation_plan,
                rank=rank,
            )
            dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": (
                                "adr207_native_query_action_evaluation_start"
                                if adr207_native_query
                                else "adr149_action_evaluation_start"
                            ),
                            "checkpoint_global_step": checkpoint_global_step,
                            "samples_per_rank": len(local_schedule),
                            "scientific_sample_count": len(evaluation_plan.items),
                            "padding_forward_count": (
                                len(local_schedule) * WORLD_SIZE - len(evaluation_plan.items)
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            local_samples: list[dict[str, Any]] = []
            local_entity_samples: list[dict[str, Any]] = []
            for local_index, work_item in enumerate(local_schedule):
                item = work_item.item
                replay_seed = _evaluation_replay_seed(
                    evaluation_plan.artifact_sha256,
                    item.sample_key,
                )
                planned = None
                collated = None
                preparation_error = None
                try:
                    planned = build_native_calvin_replay_batch(
                        evaluation_dataset,
                        sample_key=item.sample_key,
                        lane_id=rank,
                        episode_instance_id=(
                            f"adr149-cold-evaluation/{item.partition}/{item.ordinal}"
                        ),
                        optimizer_step=0,
                        replay_seed=replay_seed,
                        device=device,
                        dtype=torch.bfloat16,
                    )
                    collated = collate_planned(planned)
                except BaseException as error:
                    preparation_error = error
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    stage=(
                        f"ADR-149 action evaluation step {checkpoint_global_step} "
                        f"sample {local_index} preparation"
                    ),
                    local_error=preparation_error,
                )
                if planned is None or collated is None:
                    raise RuntimeError("ADR-149 action evaluation preparation vanished")
                prior_host_steps = _distributed_prior_host_step_schedule(
                    (len(collated.effective_prior_control_chunks),),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )[0]

                result = None
                prior_trace = None
                native_evaluation = None
                factual_host_diagnostic = None
                blocked_host_diagnostic = None
                prepared_current = None
                forward_seconds = 0.0
                blocked_forward_seconds = 0.0
                forward_error = None
                runtime_snapshot = None
                try:
                    runtime_snapshot = snapshot_official_runtime_buffers()
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                        torch.manual_seed(replay_seed)
                        torch.cuda.manual_seed(replay_seed)
                        if adr207_native_query:
                            if videomt_runtime is None or graph is None:
                                raise RuntimeError(
                                    "ADR-207 action evaluation lacks source or host runtime"
                                )
                            prepared_current = prepare_native_videomt_current_frame(
                                index,
                                item.source_global_index,
                            )
                            native_evaluation = (
                                run_cold_native_videomt_lingbot_evaluation(
                                    policy,
                                    videomt_runtime,
                                    graph=graph,
                                    batch=collated,
                                    normalized_current_rgb=(
                                        prepared_current.frames.model_input.to(
                                            device=device,
                                            dtype=torch.float32,
                                        )
                                    ),
                                    relation_spec=graph_config.object_query_spatial_specs[0],
                                    host_dtype=torch.bfloat16,
                                    prior_host_steps=prior_host_steps,
                                    posterior_adoption_route=(
                                        torch.ones(
                                            collated.routing.batch_size,
                                            dtype=torch.bool,
                                            device=device,
                                        )
                                        if _adr222_world_token_adoption_active(args)
                                        else None
                                    ),
                                    wla_host_evidence_arm=args.wla_host_evidence_arm,
                                )
                            )
                            result = native_evaluation.policy
                            prior_trace = native_evaluation.prior_trace
                        else:
                            result, prior_trace = run_cold_picf_action_forward(
                                collated,
                                prior_host_steps=prior_host_steps,
                            )
                    torch.cuda.synchronize(device)
                    forward_seconds = time.perf_counter() - started
                    if adr221_wsa_edge_diagnostic:
                        if (
                            not adr207_native_query
                            or native_evaluation is None
                            or graph is None
                        ):
                            raise RuntimeError(
                                "ADR-221 WSA edge diagnostic lacks its native source graph"
                            )
                        restore_official_runtime_buffers(runtime_snapshot)
                        torch.cuda.synchronize(device)
                        diagnostic_seed = derive_subseed(
                            args.seed,
                            ADR221_WSA_EDGE_INTERVENTION_SCHEMA,
                            str(checkpoint_global_step),
                            item.sample_key,
                            "fixed-host",
                        )
                        with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                            torch.manual_seed(diagnostic_seed)
                            torch.cuda.manual_seed(diagnostic_seed)
                            factual_host_diagnostic = run_native_videomt_host_diagnostic(
                                policy,
                                graph=graph,
                                host_batch=native_evaluation.host_batch,
                                prior_host_steps=prior_host_steps,
                            )
                        restore_official_runtime_buffers(runtime_snapshot)
                        torch.cuda.synchronize(device)
                        blocked_started = time.perf_counter()
                        with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                            torch.manual_seed(diagnostic_seed)
                            torch.cuda.manual_seed(diagnostic_seed)
                            blocked_host_diagnostic = (
                                run_native_videomt_host_diagnostic(
                                    policy,
                                    graph=graph,
                                    host_batch=native_evaluation.host_batch,
                                    prior_host_steps=prior_host_steps,
                                    wsa_attention_intervention=(
                                        WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION
                                    ),
                                )
                            )
                        torch.cuda.synchronize(device)
                        blocked_forward_seconds = time.perf_counter() - blocked_started
                except BaseException as error:
                    forward_error = error
                finally:
                    if runtime_snapshot is not None:
                        restore_official_runtime_buffers(runtime_snapshot)
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    stage=(
                        f"ADR-149 action evaluation step {checkpoint_global_step} "
                        f"sample {local_index} forward"
                    ),
                    local_error=forward_error,
                )
                if result is None or not isinstance(prior_trace, NativeLayerwisePriorTrace):
                    raise RuntimeError("matched action evaluation forward vanished")
                if _wla_complete_active(args) and (
                    result.action_backend != WLA_COMPLETE_ACTION_BACKEND
                ):
                    raise RuntimeError(
                        "matched WLA action evaluation executed another action backend"
                    )
                if adr207_native_query and (
                    native_evaluation is None or prepared_current is None
                ):
                    raise RuntimeError("ADR-207 action evaluation source forward vanished")

                evidence = None
                entity_evidence = None
                evidence_error = None
                try:
                    posterior = result.context.posterior_memory
                    if not isinstance(posterior, NativeLayerwisePosteriorState):
                        raise RuntimeError("matched action evaluation omitted posterior memory")
                    wsa_edge_evidence = None
                    if adr221_wsa_edge_diagnostic:
                        if (
                            native_evaluation is None
                            or factual_host_diagnostic is None
                            or blocked_host_diagnostic is None
                        ):
                            raise RuntimeError(
                                "ADR-221 WSA edge paired host evaluations vanished"
                            )
                        factual_posterior = (
                            factual_host_diagnostic.policy.context.posterior_memory
                        )
                        blocked_posterior = blocked_host_diagnostic.policy.context.posterior_memory
                        if not isinstance(
                            factual_posterior,
                            NativeLayerwisePosteriorState,
                        ) or not isinstance(blocked_posterior, NativeLayerwisePosteriorState):
                            raise RuntimeError(
                                "ADR-221 paired action replay omitted posterior memory"
                            )
                        source_host_batch_reused_by_identity = (
                            factual_host_diagnostic.host_batch
                            is native_evaluation.host_batch
                            and
                            blocked_host_diagnostic.host_batch is native_evaluation.host_batch
                        )
                        prior_trace_exact_equal = torch.equal(
                            factual_host_diagnostic.prior_trace.layer_rows,
                            blocked_host_diagnostic.prior_trace.layer_rows,
                        )
                        posterior_exact_equal = torch.equal(
                            factual_posterior.layer_rows,
                            blocked_posterior.layer_rows,
                        )
                        if not (
                            source_host_batch_reused_by_identity
                            and prior_trace_exact_equal
                            and posterior_exact_equal
                        ):
                            raise RuntimeError(
                                "ADR-221 WSA edge intervention changed source identity, prior, "
                                "or posterior: "
                                + json.dumps(
                                    {
                                        "source_host_batch_reused_by_identity": (
                                            source_host_batch_reused_by_identity
                                        ),
                                        "prior_trace_exact_equal": prior_trace_exact_equal,
                                        "posterior_exact_equal": posterior_exact_equal,
                                    },
                                    sort_keys=True,
                                )
                            )
                        factual_action_loss = finite_scalar(
                            factual_host_diagnostic.policy.official_action_loss,
                            name="ADR-221 factual action loss",
                        )
                        blocked_action_loss = finite_scalar(
                            blocked_host_diagnostic.policy.official_action_loss,
                            name="ADR-221 blocked action loss",
                        )
                        wsa_edge_evidence = {
                            "schema": ADR221_WSA_EDGE_INTERVENTION_SCHEMA,
                            "intervention": (
                                WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION.value
                            ),
                            "scope": "measurement_only_fixed_source_host_replay",
                            "source_host_batch_reused_by_identity": (
                                source_host_batch_reused_by_identity
                            ),
                            "prior_trace_exact_equal": prior_trace_exact_equal,
                            "posterior_exact_equal": posterior_exact_equal,
                            "standard_action_loss": finite_scalar(
                                result.official_action_loss,
                                name="ADR-221 standard action loss",
                            ),
                            "factual_action_loss": factual_action_loss,
                            "blocked_action_loss": blocked_action_loss,
                            "blocked_minus_factual_action_loss": (
                                blocked_action_loss - factual_action_loss
                            ),
                            "blocked_forward_seconds": blocked_forward_seconds,
                        }
                    relation = result.context.relation_output
                    if adr207_native_query:
                        if not isinstance(relation, NativeObjectQueryPosteriorOutput):
                            raise RuntimeError(
                                "ADR-207 action evaluation omitted native source queries"
                            )
                        if native_evaluation is None or prepared_current is None:
                            raise RuntimeError("ADR-207 target boundary lost its source output")
                        target_frame = physical_sidecar.source_frame(item.source_global_index)
                        target_clip = prepare_calvin_videomt_clip(
                            (prepared_current.source_rgb,),
                            (target_frame,),
                        )
                        if not torch.equal(
                            target_clip.frames.model_input,
                            prepared_current.frames.model_input,
                        ):
                            raise RuntimeError(
                                "ADR-207 post-forward target geometry differs from source input"
                            )
                        anchor_evaluation = evaluate_videomt_anchors(
                            native_evaluation.source_output,
                            target_clip,
                        )
                        visuals = []
                        if (
                            not work_item.is_padding
                            and item.sample_key in evaluation_visual_sample_keys
                        ):
                            visual_root = (
                                args.run_dir
                                / "heldout_native_videomt_anchor_evaluation"
                                / f"step_{checkpoint_global_step:08d}"
                            )
                            oracle_path = visual_root / (
                                f"rank_{rank:02d}_ordinal_{item.ordinal:04d}_oracle.png"
                            )
                            render_videomt_anchor_panel(
                                native_evaluation.source_output,
                                target_clip,
                                anchor_evaluation,
                                output_path=oracle_path,
                            )
                            top_10 = next(
                                proposal
                                for proposal in anchor_evaluation.ranked_proposals
                                if proposal.top_k == 10
                            )
                            ranked_path = oracle_path.with_name(
                                oracle_path.name.replace("_oracle.png", "_top10.png")
                            )
                            render_videomt_anchor_panel(
                                native_evaluation.source_output,
                                target_clip,
                                anchor_evaluation,
                                output_path=ranked_path,
                                ranked_proposal=top_10,
                            )
                            visuals = [
                                {
                                    "kind": kind,
                                    "path": str(path),
                                    "sha256": _sha256(path),
                                }
                                for kind, path in (
                                    ("oracle_all_200", oracle_path),
                                    ("foreground_ranked_top10", ranked_path),
                                )
                            ]
                        entity_evidence = anchor_evaluation.to_dict()
                        entity_evidence.update(
                            {
                                "evaluation_kind": "native_videomt_source_queries",
                                "source_rgb_sha256": prepared_current.source_rgb_sha256,
                                "source_query_count": relation.posterior_rows.shape[1],
                                "task_used_by_entity_objective": False,
                                "physical_sidecar_read_after_model_forward": True,
                                "visual_artifacts": visuals,
                            }
                        )
                    else:
                        if not isinstance(relation, PhysicalRelationOutput):
                            raise RuntimeError(
                                "ADR-149 action evaluation omitted physical relations"
                            )
                        target_bundle = build_task_independent_calvin_targets(
                            requests_by_time=(collated.structural_target_requests,),
                            model_inputs_by_time=(collated.model_inputs,),
                            relations=(relation,),
                            physical_sidecar=physical_sidecar,
                            capacity=relation.support_logits.shape[-1],
                            patch_size=patch_size,
                            merge_size=merge_size,
                            minimum_supervised_fraction=args.minimum_supervised_fraction,
                            capacity_seeds=planned.augmentation_seeds,
                        )[0]
                        entity_objective = compose_task_independent_entity_objective(
                            official_policy_loss=None,
                            relations=(relation,),
                            targets=(target_bundle.targets,),
                            config=entity_evaluation_objective_config,
                        )
                        frame_loss = entity_objective.frame_losses[0]
                        entity_evidence = evaluate_physical_entity_frame(
                            physical_frame_predictions_from_relation(relation),
                            target_bundle.targets,
                            frame_loss.assignment,
                            identity_keys=target_bundle.identity_keys_by_batch[0],
                        )
                        visuals = []
                        if item.sample_key in evaluation_visual_sample_keys:
                            visuals = render_task_independent_entity_visuals(
                                output_root=args.run_dir / "heldout_entity_evaluation",
                                global_step=checkpoint_global_step,
                                input_weight_global_step=checkpoint_global_step,
                                weight_boundary="fixed_heldout_action_forward",
                                rank=rank,
                                host_items=planned.training.host_items,
                                model_inputs=collated.model_inputs,
                                relation=relation,
                                target_bundle=target_bundle,
                                set_loss=frame_loss,
                                sample_keys=collated.routing.sample_keys,
                                merge_size=merge_size,
                            )
                        entity_evidence.update(
                            {
                                "objective_total": finite_scalar(
                                    entity_objective.objective.total,
                                    name="heldout entity objective",
                                ),
                                "mask_focal": finite_scalar(
                                    frame_loss.mask_focal,
                                    name="heldout mask focal",
                                ),
                                "mask_dice": finite_scalar(
                                    frame_loss.mask_dice,
                                    name="heldout mask dice",
                                ),
                                "existence_focal": finite_scalar(
                                    frame_loss.existence_focal,
                                    name="heldout existence focal",
                                ),
                                "ownership_nll": finite_scalar(
                                    frame_loss.ownership_nll,
                                    name="heldout ownership NLL",
                                ),
                                "visual_artifacts": visuals,
                            }
                        )
                    entity_evidence.update(
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
                        }
                    )
                    evidence = {
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
                        "model_inputs_sha256": model_inputs_sha256(collated.model_inputs),
                        "total_loss": finite_scalar(
                            result.official_total_loss,
                            name="total loss",
                        ),
                        "action_loss": finite_scalar(
                            result.official_action_loss,
                            name="action loss",
                        ),
                        "action_backend": result.action_backend,
                        "moe_regularizer": finite_scalar(
                            result.official_moe_regularizer,
                            name="MoE regularizer",
                        ),
                        "official_output_arity": len(result.official_outputs),
                        "prior_control_chunk_count": len(collated.effective_prior_control_chunks),
                        "prior_trace_finite": bool(
                            torch.isfinite(prior_trace.layer_rows).all().item()
                        ),
                        "posterior_finite": bool(torch.isfinite(posterior.layer_rows).all().item()),
                        "native_source_rgb_sha256": (
                            prepared_current.source_rgb_sha256
                            if prepared_current is not None
                            else None
                        ),
                        "native_source_query_count": (
                            native_evaluation.source_output.class_logits.shape[2]
                            if native_evaluation is not None
                            else None
                        ),
                        "forward_seconds": forward_seconds,
                        "wsa_future_to_action_intervention": wsa_edge_evidence,
                    }
                except BaseException as error:
                    evidence_error = error
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    stage=(
                        f"ADR-149 action evaluation step {checkpoint_global_step} "
                        f"sample {local_index} evidence"
                    ),
                    local_error=evidence_error,
                )
                if evidence is None:
                    raise RuntimeError("ADR-149 action evaluation evidence vanished")
                if entity_evidence is None:
                    raise RuntimeError("ADR-149 heldout entity evidence vanished")
                if not work_item.is_padding:
                    local_samples.append(evidence)
                    local_entity_samples.append(entity_evidence)

            local_modality_intervention: dict[str, Any] | None = None
            if (
                adr207_native_query
                and checkpoint_global_step in ADR207_MODALITY_INTERVENTION_STEPS
            ):
                if (
                    adr207_intervention_optimizer_step is None
                    or videomt_runtime is None
                    or graph is None
                ):
                    raise RuntimeError("ADR-207 modality intervention contract is incomplete")
                intervention_planned = build_planned_native_calvin_batch(
                    stream_plan,
                    dataset,
                    optimizer_step=adr207_intervention_optimizer_step,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    gradient_accumulation_steps=1,
                    accumulation_index=0,
                    device=device,
                    dtype=torch.bfloat16,
                    maximum_control_tokens=args.maximum_control_tokens,
                    gradient_suffix_control_tokens=args.prior_gradient_control_tokens,
                )
                intervention_batch = collate_planned(intervention_planned)
                if intervention_batch.modalities is None:
                    raise RuntimeError("ADR-207 intervention batch omitted dense modalities")
                intervention_source_index = source_global_index_by_sample_key(
                    intervention_batch.routing.sample_keys[0]
                )
                intervention_current = prepare_native_videomt_current_frame(
                    index,
                    intervention_source_index,
                )
                intervention_prior_host_steps = _distributed_prior_host_step_schedule(
                    (len(intervention_batch.effective_prior_control_chunks),),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )[0]
                source_runtime_snapshot = snapshot_official_runtime_buffers()
                intervention_source = None
                try:
                    with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                        source_seed = derive_subseed(
                            args.seed,
                            ADR207_MODALITY_INTERVENTION_SCHEMA,
                            str(checkpoint_global_step),
                            "source",
                        )
                        torch.manual_seed(source_seed)
                        torch.cuda.manual_seed(source_seed)
                        intervention_source = run_cold_native_videomt_lingbot_evaluation(
                            policy,
                            videomt_runtime,
                            graph=graph,
                            batch=intervention_batch,
                            normalized_current_rgb=(
                                intervention_current.frames.model_input.to(
                                    device=device,
                                    dtype=torch.float32,
                                )
                            ),
                            relation_spec=graph_config.object_query_spatial_specs[0],
                            host_dtype=torch.bfloat16,
                            prior_host_steps=intervention_prior_host_steps,
                            posterior_adoption_route=(
                                torch.ones(
                                    intervention_batch.routing.batch_size,
                                    dtype=torch.bool,
                                    device=device,
                                )
                                if _adr222_world_token_adoption_active(args)
                                else None
                            ),
                        )
                finally:
                    restore_official_runtime_buffers(source_runtime_snapshot)
                if intervention_source is None:
                    raise RuntimeError("ADR-207 intervention source forward vanished")
                fixed_host_batch = intervention_source.host_batch
                stream_by_name = {
                    stream.name: stream for stream in fixed_host_batch.modalities.streams
                }
                if any(
                    modality not in stream_by_name
                    or int(stream_by_name[modality].valid.sum().item()) < 2
                    for modality in CALVIN_FULL_DENSE_MODALITIES
                ):
                    raise RuntimeError(
                        "ADR-207 selected intervention batch is not fully populated"
                    )
                diagnostic_seed = derive_subseed(
                    args.seed,
                    ADR207_MODALITY_INTERVENTION_SCHEMA,
                    str(checkpoint_global_step),
                    "fixed-host",
                )

                def fixed_source_action_output(
                    batch: Any,
                    *,
                    label: str,
                ) -> tuple[Any, float]:
                    runtime_snapshot = snapshot_official_runtime_buffers()
                    diagnostic = None
                    captured: list[Any] = []
                    try:
                        with torch.random.fork_rng(devices=[local_rank]):
                            torch.manual_seed(diagnostic_seed)
                            torch.cuda.manual_seed(diagnostic_seed)
                            with capture_action_projection_output(policy) as captured:
                                diagnostic = run_native_videomt_host_diagnostic(
                                    policy,
                                    graph=graph,
                                    host_batch=batch,
                                    prior_host_steps=intervention_prior_host_steps,
                                    posterior_adoption_route=(
                                        torch.ones(
                                            batch.routing.batch_size,
                                            dtype=torch.bool,
                                            device=device,
                                        )
                                        if _adr222_world_token_adoption_active(args)
                                        else None
                                    ),
                                )
                    finally:
                        restore_official_runtime_buffers(runtime_snapshot)
                    if diagnostic is None:
                        raise RuntimeError(f"ADR-207 {label} host diagnostic vanished")
                    return (
                        single_captured_action_output(captured),
                        finite_scalar(
                            diagnostic.policy.official_action_loss,
                            name=f"ADR-207 {label} action loss",
                        ),
                    )

                factual_output, factual_action_loss = fixed_source_action_output(
                    fixed_host_batch,
                    label="factual",
                )
                repeated_output, repeated_action_loss = fixed_source_action_output(
                    fixed_host_batch,
                    label="factual-repeat",
                )
                modality_reports: dict[str, Any] = {}
                for modality in CALVIN_FULL_DENSE_MODALITIES:
                    rows: dict[str, Any] = {}
                    for intervention in ADR207_MODALITY_INTERVENTIONS:
                        variant = intervene_modality(
                            fixed_host_batch.modalities,
                            modality=modality,
                            intervention=intervention,
                        )
                        variant_output, variant_action_loss = fixed_source_action_output(
                            replace(fixed_host_batch, modalities=variant.batch),
                            label=f"{modality}:{intervention}",
                        )
                        rows[intervention] = {
                            "changed_elements": variant.changed_elements,
                            "action_loss": variant_action_loss,
                            "action_loss_delta": variant_action_loss - factual_action_loss,
                            **action_projection_drift_report(
                                factual_output,
                                variant_output,
                            ),
                        }
                    modality_reports[modality] = rows
                local_modality_intervention = {
                    "rank": rank,
                    "sample_key": intervention_batch.routing.sample_keys[0],
                    "probe_optimizer_step": adr207_intervention_optimizer_step,
                    "source_global_index": intervention_source_index,
                    "source_rgb_sha256": intervention_current.source_rgb_sha256,
                    "factual_action_loss": factual_action_loss,
                    "factual_repeat_action_loss": repeated_action_loss,
                    "factual_repeat": action_projection_drift_report(
                        factual_output,
                        repeated_output,
                    ),
                    "valid_token_count_by_modality": {
                        modality: int(stream_by_name[modality].valid.sum().item())
                        for modality in CALVIN_FULL_DENSE_MODALITIES
                    },
                    "modalities": modality_reports,
                }

            gathered_samples: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_samples, local_samples)
            gathered_entity_samples: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_entity_samples, local_entity_samples)
            gathered_modality_interventions: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(
                gathered_modality_interventions,
                local_modality_intervention,
            )
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    samples = sorted(
                        (sample for rank_samples in gathered_samples for sample in rank_samples),
                        key=lambda sample: int(sample["ordinal"]),
                    )
                    expected_keys = [item.sample_key for item in evaluation_plan.items]
                    if [sample["sample_key"] for sample in samples] != expected_keys:
                        raise RuntimeError("ADR-149 action evaluation sample set changed")
                    entity_samples = sorted(
                        (
                            sample
                            for rank_samples in gathered_entity_samples
                            for sample in rank_samples
                        ),
                        key=lambda sample: int(sample["ordinal"]),
                    )
                    if [sample["sample_key"] for sample in entity_samples] != expected_keys:
                        raise RuntimeError("ADR-202 heldout entity evaluation sample set changed")
                    if [sample["source_digest"] for sample in entity_samples] != [
                        sample["source_digest"] for sample in samples
                    ]:
                        raise RuntimeError(
                            "ADR-202 action and heldout entity evaluations used different inputs"
                        )
                    evaluation_input_sha256 = hashlib.sha256(
                        json.dumps(
                            [
                                {
                                    "sample_key": sample["sample_key"],
                                    "source_digest": sample["source_digest"],
                                    "model_inputs_sha256": sample["model_inputs_sha256"],
                                    "source_rgb_sha256": (
                                        entity_samples[index].get("source_rgb_sha256")
                                        if adr207_native_query
                                        else None
                                    ),
                                }
                                for index, sample in enumerate(samples)
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    wsa_edge_intervention_receipt = None
                    if adr221_wsa_edge_diagnostic:
                        wsa_edge_intervention_receipt = {
                            "schema": ADR221_WSA_EDGE_INTERVENTION_SCHEMA,
                            "status": "MEASURED",
                            "checkpoint_global_step": checkpoint_global_step,
                            "intervention": (
                                WSALingBotAttentionIntervention.BLOCK_FUTURE_TO_ACTION.value
                            ),
                            "scope": "measurement_only_fixed_source_host_replay",
                            "optimization_graph_changed": False,
                            "partition_summaries": {
                                partition: _summarize_adr221_wsa_edge_intervention(
                                    samples,
                                    partition=partition,
                                )
                                for partition in ENTITY_EVALUATION_PARTITIONS
                            },
                        }
                    elif any(
                        sample.get("wsa_future_to_action_intervention") is not None
                        for sample in samples
                    ):
                        raise RuntimeError(
                            "ADR-221 WSA edge intervention ran outside diagnostic mode"
                        )
                    modality_intervention_receipt = None
                    intervention_due = (
                        adr207_native_query
                        and checkpoint_global_step
                        in ADR207_MODALITY_INTERVENTION_STEPS
                    )
                    if intervention_due:
                        if any(
                            not isinstance(report, Mapping)
                            for report in gathered_modality_interventions
                        ):
                            raise RuntimeError(
                                "ADR-207 modality intervention lost a rank report"
                            )
                        intervention_summary = _summarize_adr207_modality_interventions(
                            tuple(gathered_modality_interventions),
                            checkpoint_global_step=checkpoint_global_step,
                            expected_world_size=WORLD_SIZE,
                        )
                        intervention_payload = {
                            **intervention_summary,
                            "implementation_sha256": implementation_sha256,
                            "model_family_sha256": model_family_sha256,
                            "lingbot_base_family_sha256": lingbot_base_family_sha256,
                            "execution_contract_sha256": execution_contract_sha256,
                            "stream_plan_sha256": stream_plan.plan_sha256,
                            "representation_split_sha256": (
                                representation_split.artifact_sha256
                            ),
                            "probe_scope": (
                                "frozen_training_stream_wiring_intervention; "
                                "held-out generalization is measured separately by the "
                                "action and anchor snapshots"
                            ),
                            "rank_reports": gathered_modality_interventions,
                        }
                        intervention_artifact_sha256 = hashlib.sha256(
                            json.dumps(
                                intervention_payload,
                                allow_nan=False,
                                sort_keys=True,
                                separators=(",", ":"),
                            ).encode("ascii")
                        ).hexdigest()
                        intervention_snapshot = {
                            **intervention_payload,
                            "artifact_sha256": intervention_artifact_sha256,
                        }
                        intervention_destination = (
                            args.run_dir
                            / "native_videomt_modality_interventions"
                            / f"step_{checkpoint_global_step:08d}"
                            / "distributed.json"
                        )
                        intervention_destination.parent.mkdir(
                            parents=True,
                            exist_ok=False,
                        )
                        write_text_durable_exclusive(
                            intervention_destination,
                            json.dumps(
                                intervention_snapshot,
                                indent=2,
                                sort_keys=True,
                            )
                            + "\n",
                        )
                        modality_intervention_receipt = {
                            "artifact_sha256": intervention_artifact_sha256,
                            "file_sha256": _sha256(intervention_destination),
                            "path": str(intervention_destination),
                            "status": "PASS",
                        }
                    elif any(
                        report is not None for report in gathered_modality_interventions
                    ):
                        raise RuntimeError(
                            "ADR-207 modality intervention ran outside its registered steps"
                        )
                    entity_payload = {
                        "schema": (
                            ADR207_ANCHOR_EVALUATION_SCHEMA
                            if adr207_native_query
                            else "picf-next.adr202-heldout-entity-snapshot/v1"
                        ),
                        "status": "PASS",
                        "checkpoint_global_step": checkpoint_global_step,
                        "architecture_identity": (
                            NATIVE_VIDEOMT_QUERY_POSTERIOR
                            if adr207_native_query
                            else UNIFIED_LAYERWISE_PREDICT_CORRECT
                        ),
                        "state_mode": "cold_reset",
                        "task_scorer_present": False,
                        "task_used_by_entity_objective": False,
                        "source_query_count": 200 if adr207_native_query else None,
                        "source_target_materialized_only_after_forward": adr207_native_query,
                        "physical_sidecar_read_during_model_forward": False,
                        "physical_sidecar_read_after_model_forward_for_metrics": True,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "lingbot_base_family_sha256": lingbot_base_family_sha256,
                        "execution_contract_sha256": execution_contract_sha256,
                        "stream_plan_sha256": stream_plan.plan_sha256,
                        "representation_split_sha256": representation_split.artifact_sha256,
                        "evaluation_plan_sha256": evaluation_plan.artifact_sha256,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "samples": entity_samples,
                        "partition_summaries": {
                            partition: (
                                _summarize_native_videomt_anchor_partition(
                                    entity_samples,
                                    partition=partition,
                                )
                                if adr207_native_query
                                else summarize_entity_evaluation_partition(
                                    entity_samples,
                                    partition=partition,
                                )
                            )
                            for partition in ENTITY_EVALUATION_PARTITIONS
                        },
                    }
                    entity_artifact_sha256 = hashlib.sha256(
                        json.dumps(
                            entity_payload,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    entity_snapshot = {
                        **entity_payload,
                        "artifact_sha256": entity_artifact_sha256,
                    }
                    entity_destination = (
                        args.run_dir
                        / (
                            "heldout_native_videomt_anchor_evaluations"
                            if adr207_native_query
                            else "heldout_entity_evaluations"
                        )
                        / f"step_{checkpoint_global_step:08d}"
                        / "distributed.json"
                    )
                    entity_destination.parent.mkdir(parents=True, exist_ok=False)
                    write_text_durable_exclusive(
                        entity_destination,
                        json.dumps(entity_snapshot, indent=2, sort_keys=True) + "\n",
                    )
                    auxiliary_evaluation_key = (
                        "heldout_anchor_evaluation"
                        if adr207_native_query
                        else "heldout_entity_evaluation"
                    )
                    payload = {
                        "schema": (
                            ADR207_ACTION_EVALUATION_SCHEMA
                            if adr207_native_query
                            else TWO_PASS_ACTION_EVALUATION_SCHEMA
                        ),
                        "status": "PASS",
                        "checkpoint_global_step": checkpoint_global_step,
                        "architecture_identity": (
                            NATIVE_VIDEOMT_QUERY_POSTERIOR
                            if adr207_native_query
                            else UNIFIED_LAYERWISE_PREDICT_CORRECT
                        ),
                        "picf_graph_installed": True,
                        "physical_sidecar_read_during_model_forward": False,
                        "physical_sidecar_read_after_model_forward_for_metrics": True,
                        "task_scorer_present": False,
                        "action_suffix_executed": True,
                        "state_mode": "cold_reset",
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "lingbot_base_family_sha256": lingbot_base_family_sha256,
                        "execution_contract_sha256": execution_contract_sha256,
                        "stream_plan_sha256": stream_plan.plan_sha256,
                        "representation_split_sha256": (representation_split.artifact_sha256),
                        "evaluation_plan_sha256": evaluation_plan.artifact_sha256,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "full_modal_action_intervention": modality_intervention_receipt,
                        "wsa_future_to_action_intervention": (
                            wsa_edge_intervention_receipt
                        ),
                        auxiliary_evaluation_key: {
                            "artifact_sha256": entity_artifact_sha256,
                            "file_sha256": _sha256(entity_destination),
                            "path": str(entity_destination),
                            "partition_summaries": entity_snapshot["partition_summaries"],
                        },
                        "samples": samples,
                        "partition_summaries": {
                            partition: _summarize_action_partition(
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
                        args.run_dir
                        / "action_evaluations"
                        / f"step_{checkpoint_global_step:08d}"
                        / "distributed.json"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=False)
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
                        "full_modal_action_intervention": modality_intervention_receipt,
                        "wsa_future_to_action_intervention": (
                            wsa_edge_intervention_receipt
                        ),
                        auxiliary_evaluation_key: snapshot[auxiliary_evaluation_key],
                    }
                except BaseException as error:
                    publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(
                    f"ADR-149 action evaluation publication failed: {publication[0]}"
                )
            dist.barrier()
            return publication[0]

        def run_causal_warm_action_evaluation(
            checkpoint_global_step: int,
            *,
            cold_publication: Mapping[str, Any],
        ) -> dict[str, Any]:
            """Evaluate the exact paired posterior after four real past frames."""

            if not adr207_native_query:
                raise RuntimeError("causal-warm action evaluation requires ADR-207")
            if checkpoint_global_step not in ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS:
                raise ValueError("causal-warm action evaluation step is not registered")
            if (
                evaluation_plan is None
                or representation_split is None
                or videomt_runtime is None
                or graph is None
            ):
                raise RuntimeError("causal-warm action evaluation contract is incomplete")
            optimizer.zero_grad(set_to_none=True)
            if videomt_optimizer is None:
                raise RuntimeError("causal-warm action evaluation lacks source optimizer")
            videomt_optimizer.zero_grad(set_to_none=True)
            local_schedule = build_distributed_causal_warm_evaluation_schedule(
                evaluation_plan,
                rank=rank,
                history_transitions=ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
            )
            eligible_items = tuple(
                item
                for item in evaluation_plan.items
                if item.transition_index >= ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS
            )
            excluded_items = tuple(
                item
                for item in evaluation_plan.items
                if item.transition_index < ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS
            )
            dist.barrier()
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "adr210_causal_warm_action_evaluation_start",
                            "checkpoint_global_step": checkpoint_global_step,
                            "history_transitions": ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
                            "eligible_sample_count": len(eligible_items),
                            "excluded_sample_count": len(excluded_items),
                            "samples_per_rank": len(local_schedule),
                            "padding_forward_count": (
                                len(local_schedule) * WORLD_SIZE - len(eligible_items)
                            ),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

            local_samples: list[dict[str, Any]] = []
            local_anchor_samples: list[dict[str, Any]] = []
            for local_index, work_item in enumerate(local_schedule):
                item = work_item.item
                replay_seed = _evaluation_replay_seed(
                    evaluation_plan.artifact_sha256,
                    item.sample_key,
                )
                history_keys = evaluation_dataset.history_sample_keys(item.sample_key)[
                    -ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS:
                ]
                if len(history_keys) != ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS:
                    raise RuntimeError("causal-warm schedule admitted an incomplete history")
                sample_keys = (*history_keys, item.sample_key)
                source_indices = tuple(
                    evaluation_dataset.source_global_index_by_key(sample_key)
                    for sample_key in sample_keys
                )
                if source_indices != tuple(
                    range(
                        item.source_global_index - ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
                        item.source_global_index + 1,
                    )
                ):
                    raise RuntimeError("causal-warm source history is not consecutive")

                planned_batches = None
                collated_batches = None
                prepared_frames = None
                preparation_error = None
                try:
                    episode_instance_id = (
                        f"adr210-causal-warm-evaluation/{item.partition}/{item.ordinal}"
                    )
                    planned_batches = tuple(
                        build_native_calvin_replay_batch(
                            evaluation_dataset,
                            sample_key=sample_key,
                            lane_id=rank,
                            episode_instance_id=episode_instance_id,
                            optimizer_step=0,
                            replay_seed=replay_seed,
                            device=device,
                            dtype=torch.bfloat16,
                        )
                        for sample_key in sample_keys
                    )
                    if len(
                        {
                            planned.training.host_items[0]["task"]
                            for planned in planned_batches
                        }
                    ) != 1:
                        raise RuntimeError("causal-warm replay changed its natural instruction")
                    collated_batches = tuple(
                        collate_planned(planned) for planned in planned_batches
                    )
                    prepared_frames = tuple(
                        prepare_native_videomt_current_frame(index, source_index)
                        for source_index in source_indices
                    )
                except BaseException as error:
                    preparation_error = error
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    stage=(
                        f"ADR-210 causal-warm evaluation step {checkpoint_global_step} "
                        f"sample {local_index} preparation"
                    ),
                    local_error=preparation_error,
                )
                if (
                    planned_batches is None
                    or collated_batches is None
                    or prepared_frames is None
                ):
                    raise RuntimeError("causal-warm preparation vanished")
                aligned_host_steps = _distributed_prior_host_step_schedule(
                    tuple(
                        len(batch.effective_prior_control_chunks)
                        for batch in collated_batches
                    ),
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )

                warm_evaluation = None
                forward_seconds = 0.0
                forward_error = None
                runtime_snapshot = None
                try:
                    runtime_snapshot = snapshot_official_runtime_buffers()
                    normalized_rgb = torch.cat(
                        tuple(frame.frames.model_input for frame in prepared_frames),
                        dim=0,
                    ).to(device=device, dtype=torch.float32)
                    torch.cuda.synchronize(device)
                    started = time.perf_counter()
                    with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                        torch.manual_seed(replay_seed)
                        torch.cuda.manual_seed(replay_seed)
                        warm_evaluation = (
                            run_causal_warm_native_videomt_lingbot_evaluation(
                                policy,
                                videomt_runtime,
                                graph=graph,
                                history_batches=collated_batches[:-1],
                                current_batch=collated_batches[-1],
                                normalized_rgb_sequence=normalized_rgb,
                                relation_spec=graph_config.object_query_spatial_specs[0],
                                host_dtype=torch.bfloat16,
                                prior_host_steps=aligned_host_steps,
                                posterior_adoption_route=(
                                    torch.ones(
                                        collated_batches[-1].routing.batch_size,
                                        dtype=torch.bool,
                                        device=device,
                                    )
                                    if _adr222_world_token_adoption_active(args)
                                    else None
                                ),
                                wla_host_evidence_arm=args.wla_host_evidence_arm,
                            )
                        )
                    torch.cuda.synchronize(device)
                    forward_seconds = time.perf_counter() - started
                except BaseException as error:
                    forward_error = error
                finally:
                    if runtime_snapshot is not None:
                        restore_official_runtime_buffers(runtime_snapshot)
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    stage=(
                        f"ADR-210 causal-warm evaluation step {checkpoint_global_step} "
                        f"sample {local_index} forward"
                    ),
                    local_error=forward_error,
                )
                if warm_evaluation is None:
                    raise RuntimeError("causal-warm forward vanished")

                evidence = None
                anchor_evidence = None
                evidence_error = None
                try:
                    result = warm_evaluation.current.policy
                    posterior = result.context.posterior_memory
                    relation = result.context.relation_output
                    if not isinstance(posterior, NativeLayerwisePosteriorState):
                        raise RuntimeError("causal-warm action omitted posterior memory")
                    if not isinstance(relation, NativeObjectQueryPosteriorOutput):
                        raise RuntimeError("causal-warm action omitted source-query relation")
                    current_prepared = prepared_frames[-1]
                    target_frame = physical_sidecar.source_frame(item.source_global_index)
                    target_clip = prepare_calvin_videomt_clip(
                        (current_prepared.source_rgb,),
                        (target_frame,),
                    )
                    if not torch.equal(
                        target_clip.frames.model_input,
                        current_prepared.frames.model_input,
                    ):
                        raise RuntimeError(
                            "causal-warm post-forward target geometry differs from source input"
                        )
                    current_source = warm_evaluation.source_sequence.per_frame[-1]
                    anchor_evaluation = evaluate_videomt_anchors(
                        current_source,
                        target_clip,
                    )
                    visuals = []
                    if (
                        not work_item.is_padding
                        and item.sample_key in evaluation_visual_sample_keys
                    ):
                        visual_root = (
                            args.run_dir
                            / "heldout_native_videomt_causal_warm_anchor_evaluation"
                            / f"step_{checkpoint_global_step:08d}"
                        )
                        oracle_path = visual_root / (
                            f"rank_{rank:02d}_ordinal_{item.ordinal:04d}_oracle.png"
                        )
                        render_videomt_anchor_panel(
                            current_source,
                            target_clip,
                            anchor_evaluation,
                            output_path=oracle_path,
                        )
                        top_10 = next(
                            proposal
                            for proposal in anchor_evaluation.ranked_proposals
                            if proposal.top_k == 10
                        )
                        ranked_path = oracle_path.with_name(
                            oracle_path.name.replace("_oracle.png", "_top10.png")
                        )
                        render_videomt_anchor_panel(
                            current_source,
                            target_clip,
                            anchor_evaluation,
                            output_path=ranked_path,
                            ranked_proposal=top_10,
                        )
                        visuals = [
                            {
                                "kind": kind,
                                "path": str(path),
                                "sha256": _sha256(path),
                            }
                            for kind, path in (
                                ("oracle_all_200", oracle_path),
                                ("foreground_ranked_top10", ranked_path),
                            )
                        ]
                    anchor_evidence = anchor_evaluation.to_dict()
                    anchor_evidence.update(
                        {
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
                            "source_digest": collated_batches[-1].source_digest,
                            "source_rgb_sha256": current_prepared.source_rgb_sha256,
                            "source_query_count": relation.posterior_rows.shape[1],
                            "history_transitions": ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
                            "task_used_by_entity_objective": False,
                            "physical_sidecar_read_after_model_forward": True,
                            "visual_artifacts": visuals,
                        }
                    )
                    modality_valid_by_frame = []
                    for batch in collated_batches:
                        if batch.modalities is None:
                            raise RuntimeError("causal-warm replay omitted dense modalities")
                        modality_valid_by_frame.append(
                            {
                                stream.name: int(stream.valid.sum().item())
                                for stream in batch.modalities.streams
                            }
                        )
                    evidence = {
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
                        "source_digest": collated_batches[-1].source_digest,
                        "model_inputs_sha256": model_inputs_sha256(
                            collated_batches[-1].model_inputs
                        ),
                        "total_loss": finite_scalar(
                            result.official_total_loss,
                            name="causal-warm total loss",
                        ),
                        "action_loss": finite_scalar(
                            result.official_action_loss,
                            name="causal-warm action loss",
                        ),
                        "moe_regularizer": finite_scalar(
                            result.official_moe_regularizer,
                            name="causal-warm MoE regularizer",
                        ),
                        "official_output_arity": len(result.official_outputs),
                        "prior_control_chunk_count": len(
                            collated_batches[-1].effective_prior_control_chunks
                        ),
                        "prior_trace_finite": bool(
                            torch.isfinite(
                                warm_evaluation.current.prior_trace.layer_rows
                            ).all().item()
                        ),
                        "posterior_finite": bool(
                            torch.isfinite(posterior.layer_rows).all().item()
                        ),
                        "native_source_rgb_sha256": current_prepared.source_rgb_sha256,
                        "native_source_query_count": (
                            current_source.class_logits.shape[2]
                        ),
                        "history_transitions": ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
                        "history_sample_keys": list(history_keys),
                        "history_source_global_indices": list(source_indices[:-1]),
                        "history_source_digests": [
                            batch.source_digest for batch in collated_batches[:-1]
                        ],
                        "history_model_inputs_sha256": [
                            model_inputs_sha256(batch.model_inputs)
                            for batch in collated_batches[:-1]
                        ],
                        "history_source_rgb_sha256": [
                            frame.source_rgb_sha256 for frame in prepared_frames[:-1]
                        ],
                        "modality_valid_token_count_by_frame": modality_valid_by_frame,
                        "forward_seconds": forward_seconds,
                    }
                except BaseException as error:
                    evidence_error = error
                _distributed_raise_if_local_probe_error(
                    dist=dist,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    stage=(
                        f"ADR-210 causal-warm evaluation step {checkpoint_global_step} "
                        f"sample {local_index} evidence"
                    ),
                    local_error=evidence_error,
                )
                if evidence is None or anchor_evidence is None:
                    raise RuntimeError("causal-warm evaluation evidence vanished")
                if not work_item.is_padding:
                    local_samples.append(evidence)
                    local_anchor_samples.append(anchor_evidence)

            gathered_samples: list[Any] = [None for _ in range(WORLD_SIZE)]
            gathered_anchor_samples: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_samples, local_samples)
            dist.all_gather_object(gathered_anchor_samples, local_anchor_samples)
            publication: list[Any] = [None]
            if rank == 0:
                try:
                    samples = sorted(
                        (sample for rank_samples in gathered_samples for sample in rank_samples),
                        key=lambda sample: int(sample["ordinal"]),
                    )
                    anchor_samples = sorted(
                        (
                            sample
                            for rank_samples in gathered_anchor_samples
                            for sample in rank_samples
                        ),
                        key=lambda sample: int(sample["ordinal"]),
                    )
                    expected_keys = [item.sample_key for item in eligible_items]
                    if [sample["sample_key"] for sample in samples] != expected_keys:
                        raise RuntimeError("causal-warm action sample set changed")
                    if [sample["sample_key"] for sample in anchor_samples] != expected_keys:
                        raise RuntimeError("causal-warm anchor sample set changed")
                    cold_path = Path(str(cold_publication["path"]))
                    cold_snapshot = json.loads(cold_path.read_text(encoding="ascii"))
                    if cold_snapshot.get("checkpoint_global_step") != checkpoint_global_step:
                        raise RuntimeError("causal-warm and cold checkpoints differ")
                    cold_by_key = {
                        sample["sample_key"]: sample for sample in cold_snapshot["samples"]
                    }
                    for sample in samples:
                        cold = cold_by_key.get(sample["sample_key"])
                        if cold is None:
                            raise RuntimeError("causal-warm sample is absent from cold snapshot")
                        for name in (
                            "source_digest",
                            "model_inputs_sha256",
                            "native_source_rgb_sha256",
                        ):
                            if sample[name] != cold[name]:
                                raise RuntimeError(
                                    f"causal-warm current input differs from cold field {name}"
                                )

                    def paired_summary(partition: str) -> dict[str, Any]:
                        rows = [sample for sample in samples if sample["partition"] == partition]
                        cold_values = [
                            float(cold_by_key[sample["sample_key"]]["action_loss"])
                            for sample in rows
                        ]
                        warm_values = [float(sample["action_loss"]) for sample in rows]
                        deltas = [
                            warm - cold
                            for cold, warm in zip(cold_values, warm_values, strict=True)
                        ]
                        cold_mean = sum(cold_values) / len(cold_values)
                        warm_mean = sum(warm_values) / len(warm_values)
                        return {
                            "sample_count": len(rows),
                            "cold_mean_action_loss": cold_mean,
                            "causal_warm_mean_action_loss": warm_mean,
                            "warm_minus_cold_mean_action_loss": warm_mean - cold_mean,
                            "relative_action_loss_reduction": (
                                (cold_mean - warm_mean) / cold_mean
                                if cold_mean != 0.0
                                else None
                            ),
                            "warm_win_fraction": (
                                sum(delta < 0.0 for delta in deltas) / len(deltas)
                            ),
                        }

                    evaluation_input_sha256 = hashlib.sha256(
                        json.dumps(
                            [
                                {
                                    "sample_key": sample["sample_key"],
                                    "source_digest": sample["source_digest"],
                                    "model_inputs_sha256": sample["model_inputs_sha256"],
                                    "native_source_rgb_sha256": sample[
                                        "native_source_rgb_sha256"
                                    ],
                                    "history_sample_keys": sample["history_sample_keys"],
                                    "history_source_digests": sample[
                                        "history_source_digests"
                                    ],
                                    "history_model_inputs_sha256": sample[
                                        "history_model_inputs_sha256"
                                    ],
                                    "history_source_rgb_sha256": sample[
                                        "history_source_rgb_sha256"
                                    ],
                                }
                                for sample in samples
                            ],
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    anchor_payload = {
                        "schema": ADR210_CAUSAL_WARM_ANCHOR_EVALUATION_SCHEMA,
                        "status": "PASS",
                        "checkpoint_global_step": checkpoint_global_step,
                        "architecture_identity": NATIVE_VIDEOMT_QUERY_POSTERIOR,
                        "state_mode": "causal_warm_four_past_frames",
                        "history_transitions": ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
                        "eligible_sample_count": len(eligible_items),
                        "excluded_samples": [item.as_dict() for item in excluded_items],
                        "task_scorer_present": False,
                        "task_used_by_entity_objective": False,
                        "source_query_count": 200,
                        "source_target_materialized_only_after_forward": True,
                        "physical_sidecar_read_during_model_forward": False,
                        "physical_sidecar_read_after_model_forward_for_metrics": True,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "lingbot_base_family_sha256": lingbot_base_family_sha256,
                        "execution_contract_sha256": execution_contract_sha256,
                        "stream_plan_sha256": stream_plan.plan_sha256,
                        "representation_split_sha256": (
                            representation_split.artifact_sha256
                        ),
                        "evaluation_plan_sha256": evaluation_plan.artifact_sha256,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "samples": anchor_samples,
                        "partition_summaries": {
                            partition: _summarize_native_videomt_anchor_partition(
                                anchor_samples,
                                partition=partition,
                            )
                            for partition in ENTITY_EVALUATION_PARTITIONS
                        },
                    }
                    anchor_artifact_sha256 = hashlib.sha256(
                        json.dumps(
                            anchor_payload,
                            allow_nan=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("ascii")
                    ).hexdigest()
                    anchor_snapshot = {
                        **anchor_payload,
                        "artifact_sha256": anchor_artifact_sha256,
                    }
                    anchor_destination = (
                        args.run_dir
                        / "causal_warm_native_videomt_anchor_evaluations"
                        / f"step_{checkpoint_global_step:08d}"
                        / "distributed.json"
                    )
                    anchor_destination.parent.mkdir(parents=True, exist_ok=False)
                    write_text_durable_exclusive(
                        anchor_destination,
                        json.dumps(anchor_snapshot, indent=2, sort_keys=True) + "\n",
                    )
                    payload = {
                        "schema": ADR210_CAUSAL_WARM_ACTION_EVALUATION_SCHEMA,
                        "status": "PASS",
                        "checkpoint_global_step": checkpoint_global_step,
                        "architecture_identity": NATIVE_VIDEOMT_QUERY_POSTERIOR,
                        "picf_graph_installed": True,
                        "action_suffix_executed_only_at_current": True,
                        "state_mode": "causal_warm_four_past_frames",
                        "history_transitions": ADR210_CAUSAL_WARM_HISTORY_TRANSITIONS,
                        "eligible_sample_count": len(eligible_items),
                        "excluded_samples": [item.as_dict() for item in excluded_items],
                        "task_scorer_present": False,
                        "physical_sidecar_read_during_model_forward": False,
                        "physical_sidecar_read_after_model_forward_for_metrics": True,
                        "implementation_sha256": implementation_sha256,
                        "model_family_sha256": model_family_sha256,
                        "lingbot_base_family_sha256": lingbot_base_family_sha256,
                        "execution_contract_sha256": execution_contract_sha256,
                        "stream_plan_sha256": stream_plan.plan_sha256,
                        "representation_split_sha256": (
                            representation_split.artifact_sha256
                        ),
                        "evaluation_plan_sha256": evaluation_plan.artifact_sha256,
                        "evaluation_input_sha256": evaluation_input_sha256,
                        "cold_action_evaluation": {
                            "artifact_sha256": cold_publication["artifact_sha256"],
                            "file_sha256": cold_publication["file_sha256"],
                            "path": str(cold_path),
                        },
                        "causal_warm_anchor_evaluation": {
                            "artifact_sha256": anchor_artifact_sha256,
                            "file_sha256": _sha256(anchor_destination),
                            "path": str(anchor_destination),
                            "partition_summaries": anchor_snapshot[
                                "partition_summaries"
                            ],
                        },
                        "samples": samples,
                        "partition_summaries": {
                            partition: _summarize_action_partition(
                                samples,
                                partition=partition,
                            )
                            for partition in ENTITY_EVALUATION_PARTITIONS
                        },
                        "paired_cold_comparisons": {
                            partition: paired_summary(partition)
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
                        args.run_dir
                        / "causal_warm_action_evaluations"
                        / f"step_{checkpoint_global_step:08d}"
                        / "distributed.json"
                    )
                    destination.parent.mkdir(parents=True, exist_ok=False)
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
                        "paired_cold_comparisons": snapshot[
                            "paired_cold_comparisons"
                        ],
                        "causal_warm_anchor_evaluation": snapshot[
                            "causal_warm_anchor_evaluation"
                        ],
                    }
                except BaseException as error:
                    publication[0] = {"error": f"{type(error).__name__}: {error}"}
            dist.broadcast_object_list(publication, src=0)
            if not isinstance(publication[0], dict) or "error" in publication[0]:
                raise RuntimeError(
                    f"ADR-210 causal-warm publication failed: {publication[0]}"
                )
            dist.barrier()
            return publication[0]

        def run_registered_action_evaluations(checkpoint_global_step: int) -> None:
            cold_publication = run_action_evaluation(checkpoint_global_step)
            action_evaluation_snapshot_reports.append(cold_publication)
            if (
                adr207_native_query
                and checkpoint_global_step
                in ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS
            ):
                causal_warm_action_evaluation_snapshot_reports.append(
                    run_causal_warm_action_evaluation(
                        checkpoint_global_step,
                        cold_publication=cold_publication,
                    )
                )

        def run_v3_control_intervention(
            *,
            previous_state: NativeLayerwisePosteriorState,
            previous_state_valid: Any,
            collated_batch: Any,
            diagnostic_seed: int,
            prior_host_steps: int,
        ) -> dict[str, Any]:
            """Isolate executed-control influence through Pass A and official action."""

            if graph is None or not graph.unified_predict_correct:
                raise RuntimeError("v3 control intervention requires the installed two-pass graph")
            factual_chunks = collated_batch.effective_prior_control_chunks
            zero_chunks = tuple(zero_executed_control(chunk) for chunk in factual_chunks)

            def prior_from(chunks: tuple[Any, ...]) -> NativeLayerwisePriorTrace:
                source, _prediction = run_native_v3_prior_chain(
                    policy,
                    graph=graph,
                    previous_memory=previous_state,
                    previous_memory_valid=previous_state_valid,
                    control_chunks=chunks,
                    filter_prediction=None,
                    require_attached_memory=False,
                    host_step_count=prior_host_steps,
                    require_grad=False,
                )
                return source

            def corrected_action(
                prior: NativeLayerwisePriorTrace,
            ) -> tuple[Any, torch.Tensor]:
                with capture_action_projection_output(policy) as captured:
                    action = run_native_policy_diagnostic_forward(
                        policy,
                        model_inputs=collated_batch.model_inputs,
                        context=native_context_from_prior_trace(
                            controls=collated_batch.controls,
                            prior_trace=prior,
                            modalities=collated_batch.modalities,
                        ),
                    )
                return action, single_captured_action_output(captured)

            def run_arm(
                chunks: tuple[Any, ...],
            ) -> tuple[NativeLayerwisePriorTrace, Any, torch.Tensor]:
                runtime_snapshot = snapshot_official_runtime_buffers()
                try:
                    with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                        torch.manual_seed(diagnostic_seed)
                        torch.cuda.manual_seed(diagnostic_seed)
                        prior = prior_from(chunks)
                        action, action_projection = corrected_action(prior)
                finally:
                    restore_official_runtime_buffers(runtime_snapshot)
                return prior, action, action_projection

            def run_corrected_action_arm(
                prior: NativeLayerwisePriorTrace,
            ) -> tuple[Any, torch.Tensor]:
                runtime_snapshot = snapshot_official_runtime_buffers()
                try:
                    with torch.no_grad(), torch.random.fork_rng(devices=[local_rank]):
                        torch.manual_seed(diagnostic_seed)
                        torch.cuda.manual_seed(diagnostic_seed)
                        return corrected_action(prior)
                finally:
                    restore_official_runtime_buffers(runtime_snapshot)

            factual_prior, factual_action, factual_action_projection = run_arm(factual_chunks)
            zero_prior, zero_action, zero_action_projection = run_arm(zero_chunks)
            factual_rows = factual_prior.layer_rows.float()
            zero_rows = zero_prior.layer_rows.float()
            denominator = factual_rows.norm().clamp_min(torch.finfo(torch.float32).eps)
            prior_relative_l2 = (zero_rows - factual_rows).norm() / denominator
            factual_action_loss = factual_action.official_action_loss
            zero_action_loss = zero_action.official_action_loss
            if any(
                not isinstance(value, torch.Tensor) or value.ndim != 0 or not torch.isfinite(value)
                for value in (factual_action_loss, zero_action_loss)
            ):
                raise RuntimeError("v3 control intervention omitted finite official action loss")

            gathered_prior_rows = [
                torch.empty_like(factual_prior.layer_rows) for _ in range(WORLD_SIZE)
            ]
            dist.all_gather(gathered_prior_rows, factual_prior.layer_rows)
            gathered_prior_addresses = None
            if isinstance(factual_prior, AddressedLayerwisePriorTrace):
                gathered_prior_addresses = [
                    torch.empty_like(factual_prior.episode_address_state.permutation)
                    for _ in range(WORLD_SIZE)
                ]
                dist.all_gather(
                    gathered_prior_addresses,
                    factual_prior.episode_address_state.permutation,
                )
            row_permutation = torch.roll(
                torch.arange(
                    factual_prior.capacity,
                    dtype=torch.long,
                    device=factual_prior.layer_rows.device,
                ),
                shifts=1,
            )
            peer_rank = (rank + 1) % WORLD_SIZE
            peer_prior_address = (
                None
                if gathered_prior_addresses is None
                else EpisodeAddressState(
                    permutation=gathered_prior_addresses[peer_rank],
                    codebook_sha256=factual_prior.episode_address_state.codebook_sha256,
                )
            )
            direct_prior_arms = {
                "zero": layerwise_prior_trace_with_tensor(
                    factual_prior,
                    torch.zeros_like(factual_prior.layer_rows),
                ),
                "wrong_row": layerwise_prior_trace_with_tensor(
                    factual_prior,
                    factual_prior.layer_rows.index_select(2, row_permutation),
                ),
                "cross_batch": layerwise_prior_trace_with_tensor(
                    factual_prior,
                    gathered_prior_rows[peer_rank],
                    episode_address_state=peer_prior_address,
                ),
            }
            direct_prior_reports: dict[str, dict[str, Any]] = {
                "factual": {
                    "prior_relative_l2": 0.0,
                    "official_action_loss": float(factual_action_loss.float().item()),
                    "official_action_loss_delta": 0.0,
                    "action_projection_drift": action_projection_drift_report(
                        factual_action_projection,
                        factual_action_projection,
                    ),
                }
            }
            for name, candidate_prior in direct_prior_arms.items():
                candidate_action, candidate_action_projection = run_corrected_action_arm(
                    candidate_prior
                )
                candidate_action_loss = candidate_action.official_action_loss
                if (
                    not isinstance(candidate_action_loss, torch.Tensor)
                    or candidate_action_loss.ndim != 0
                    or not torch.isfinite(candidate_action_loss)
                ):
                    raise RuntimeError(
                        "v3 direct-prior intervention omitted finite official action loss"
                    )
                relative_l2 = (
                    candidate_prior.layer_rows.float() - factual_rows
                ).norm() / denominator
                direct_prior_reports[name] = {
                    "prior_relative_l2": float(relative_l2.item()),
                    "official_action_loss": float(candidate_action_loss.float().item()),
                    "official_action_loss_delta": float(
                        (candidate_action_loss - factual_action_loss).float().item()
                    ),
                    "action_projection_drift": action_projection_drift_report(
                        factual_action_projection,
                        candidate_action_projection,
                    ),
                }
            changed_value_count = sum(
                int(
                    (
                        (factual.values != zero.values)
                        & factual.field_valid
                        & factual.token_valid.unsqueeze(-1)
                    )
                    .sum()
                    .item()
                )
                for factual, zero in zip(factual_chunks, zero_chunks, strict=True)
            )
            return {
                "intervention": "zero_executed_control_values",
                "isolation": (
                    "Pass A controls differ; Pass B observation, controls, action target, "
                    "and RNG are factual and matched"
                ),
                "control_chunk_count": len(factual_chunks),
                "changed_control_value_count": changed_value_count,
                "prior_relative_l2": float(prior_relative_l2.item()),
                "factual_official_action_loss": float(factual_action_loss.float().item()),
                "zero_control_prior_official_action_loss": float(zero_action_loss.float().item()),
                "official_action_loss_delta": float(
                    (zero_action_loss - factual_action_loss).float().item()
                ),
                "action_projection_drift": action_projection_drift_report(
                    factual_action_projection,
                    zero_action_projection,
                ),
                "direct_prior_intervention": {
                    "isolation": (
                        "Pass A is held factual; only the prior trace supplied to Pass B "
                        "changes, while observation, controls, action target, and RNG remain "
                        "factual and matched"
                    ),
                    "arms": direct_prior_reports,
                },
            }

        def run_causal_diagnostic(
            *,
            diagnostic_step: int,
            prepared_batch: Any,
            collated_batch: Any,
            planned_batch: Any,
            prior_host_steps: int | None,
        ) -> dict[str, Any] | None:
            if _adr207_native_videomt_query_posterior_active(args):
                return None
            legacy_diagnostic = (
                args.posterior_architecture == "legacy_v1"
                and args.causal_ablation_mode in {"zero_state", "recurrent_state"}
                and diagnostic_step in CAUSAL_DIAGNOSTIC_STEPS
            )
            layerwise_diagnostic = (
                args.posterior_architecture == "layerwise_v2"
                and args.causal_ablation_mode == "none"
                and diagnostic_step in LAYERWISE_CAUSAL_DIAGNOSTIC_STEPS
            )
            two_pass_diagnostic = (
                args.posterior_architecture == "two_pass_v3"
                and args.causal_ablation_mode == "none"
                and (
                    diagnostic_step in TWO_PASS_FILTER_DIAGNOSTIC_STEPS
                    or diagnostic_step
                    == getattr(args, "engineering_force_causal_diagnostic_step", 0)
                    or _acceptance_terminal_evidence_due(
                        mode=args.acceptance_mode,
                        global_step=diagnostic_step,
                    )
                )
            )
            if not (legacy_diagnostic or layerwise_diagnostic or two_pass_diagnostic):
                return None
            diagnostic_schema = (
                TWO_PASS_FILTER_DIAGNOSTIC_SCHEMA
                if two_pass_diagnostic
                else (
                    LAYERWISE_CAUSAL_DIAGNOSTIC_SCHEMA
                    if layerwise_diagnostic
                    else CAUSAL_ABLATION_SCHEMA
                )
            )
            local_eligible = bool(
                prepared_batch.previous_state_valid.all().item()
                and prepared_batch.wrong_time_state_valid.all().item()
            )
            eligibility = torch.tensor(
                int(local_eligible),
                dtype=torch.int64,
                device=device,
            )
            dist.all_reduce(eligibility, op=dist.ReduceOp.MIN)
            globally_eligible = bool(eligibility.item())
            report: dict[str, Any] = {
                "schema": diagnostic_schema,
                "mode": (
                    "two_pass_predict_correct"
                    if two_pass_diagnostic
                    else (
                        "layerwise_recurrent_state"
                        if layerwise_diagnostic
                        else args.causal_ablation_mode
                    )
                ),
                "global_step": diagnostic_step,
                "input_weight_global_step": diagnostic_step,
                "rank": rank,
                "sample_keys": list(collated_batch.routing.sample_keys),
                "eligible": globally_eligible,
                "task_scorer_present": False,
                "loss_side_target_entered_host_forward": False,
                "variants": {},
            }
            if globally_eligible:
                factual_state = prepared_batch.previous_state
                factual_tensor = persistent_state_tensor(factual_state)
                gathered_rows = [torch.empty_like(factual_tensor) for _ in range(WORLD_SIZE)]
                dist.all_gather(gathered_rows, factual_tensor)
                gathered_state_addresses = None
                if isinstance(factual_state, AddressedLayerwisePosteriorState):
                    gathered_state_addresses = [
                        torch.empty_like(factual_state.episode_address_state.permutation)
                        for _ in range(WORLD_SIZE)
                    ]
                    dist.all_gather(
                        gathered_state_addresses,
                        factual_state.episode_address_state.permutation,
                    )
                peer_rank = (rank + 1) % WORLD_SIZE
                peer_state_address = (
                    None
                    if gathered_state_addresses is None
                    else EpisodeAddressState(
                        permutation=gathered_state_addresses[peer_rank],
                        codebook_sha256=factual_state.episode_address_state.codebook_sha256,
                    )
                )
                peer_state = persistent_state_with_tensor(
                    factual_state,
                    gathered_rows[peer_rank],
                    episode_address_state=peer_state_address,
                )
                variants: tuple[tuple[str, NativePersistentState], ...] = (
                    ("factual", factual_state),
                    (
                        "zero",
                        persistent_state_with_tensor(
                            factual_state,
                            torch.zeros_like(factual_tensor),
                        ),
                    ),
                    ("wrong_time", prepared_batch.wrong_time_state),
                    ("cross_batch", peer_state),
                )
                if isinstance(factual_state, NativeLayerwisePosteriorState):
                    permutation = torch.roll(
                        torch.arange(
                            factual_state.capacity,
                            dtype=torch.long,
                            device=factual_tensor.device,
                        ),
                        shifts=1,
                    )
                    variants += (
                        (
                            "wrong_row",
                            persistent_state_with_tensor(
                                factual_state,
                                factual_tensor.index_select(2, permutation),
                            ),
                        ),
                    )
                correction_diagnostic = bool(
                    (layerwise_diagnostic or two_pass_diagnostic)
                    and graph is not None
                    and current_grid_cache is not None
                )
                diagnostic_omission = None
                if two_pass_diagnostic:
                    diagnostic_omission = sample_qwen_whole_view_omission(
                        images=collated_batch.model_inputs["images"],
                        image_valid=collated_batch.model_inputs["img_masks"],
                        image_grid_thw=collated_batch.model_inputs["image_grid_thw"],
                        seed=derive_subseed(
                            args.seed,
                            diagnostic_schema,
                            str(diagnostic_step),
                            str(rank),
                            "whole-static-view-omission",
                        ),
                        eligible_view_indices=(0,),
                    )
                diagnostic_config = TaskIndependentEntityObjectiveConfig(
                    action_weight=(1.0 if correction_diagnostic else 0.0),
                    entity_weight=1.0,
                    predictive_weight=(1.0 if correction_diagnostic else 0.0),
                    mask_focal_weight=args.mask_focal_weight,
                    mask_dice_weight=args.mask_dice_weight,
                    existence_weight=args.existence_weight,
                    ownership_weight=args.ownership_weight,
                )
                variant_reports: dict[str, Any] = {}
                diagnostic_seed = derive_subseed(
                    args.seed,
                    diagnostic_schema,
                    str(diagnostic_step),
                    str(rank),
                )
                for name, candidate_state in variants:
                    _emit_progress(
                        "causal_diagnostic_variant_started",
                        rank=rank,
                        global_step=diagnostic_step,
                        details={"variant": name},
                    )
                    dist.barrier()
                    runtime_snapshot = snapshot_official_runtime_buffers()
                    try:
                        with torch.random.fork_rng(devices=[local_rank]):
                            torch.manual_seed(diagnostic_seed)
                            torch.cuda.manual_seed(diagnostic_seed)
                            if correction_diagnostic:
                                if prior_host_steps is None:
                                    raise RuntimeError(
                                        "two-pass diagnostic omitted its distributed prior schedule"
                                    )
                                diagnostic = run_task_independent_calvin_joint_sequence_objective(
                                    policy,
                                    batches=(collated_batch,),
                                    physical_sidecar=physical_sidecar,
                                    objective_config=diagnostic_config,
                                    patch_size=patch_size,
                                    merge_size=merge_size,
                                    previous_state=candidate_state,
                                    previous_state_valid=prepared_batch.previous_state_valid,
                                    prior_row_bindings_by_batch=(
                                        prepared_batch.previous_row_bindings
                                    ),
                                    graph=graph,
                                    current_grid_cache=current_grid_cache,
                                    omitted_static_view=diagnostic_omission,
                                    posterior_adoption_route=(
                                        torch.ones(
                                            collated_batch.routing.batch_size,
                                            dtype=torch.bool,
                                            device=device,
                                        )
                                        if posterior_adoption_route_active
                                        else None
                                    ),
                                    minimum_supervised_fraction=(args.minimum_supervised_fraction),
                                    capacity_seeds=planned_batch.augmentation_seeds,
                                    prior_host_steps_by_batch=(prior_host_steps,),
                                    future_latent_target=(
                                        future_latent_target_for_batch(collated_batch)
                                    ),
                                )
                            else:
                                diagnostic = run_task_independent_calvin_recurrent_frame_diagnostic(
                                    policy,
                                    batch=collated_batch,
                                    physical_sidecar=physical_sidecar,
                                    objective_config=diagnostic_config,
                                    patch_size=patch_size,
                                    merge_size=merge_size,
                                    previous_state=candidate_state,
                                    previous_state_valid=(prepared_batch.previous_state_valid),
                                    prior_row_bindings_by_batch=(
                                        prepared_batch.previous_row_bindings
                                    ),
                                    minimum_supervised_fraction=(args.minimum_supervised_fraction),
                                    capacity_seeds=planned_batch.augmentation_seeds,
                                )
                    finally:
                        restore_official_runtime_buffers(runtime_snapshot)
                    frame_loss = diagnostic.objective.frame_losses[0]
                    factual_action_loss = (
                        diagnostic.primary.official_action_loss
                        if correction_diagnostic
                        else diagnostic.diagnostic_action_loss
                    )
                    omitted_policy = getattr(diagnostic, "omitted_static_policy", None)
                    route_diagnostic = correction_diagnostic and posterior_adoption_route_active
                    if route_diagnostic:
                        if omitted_policy is None:
                            raise RuntimeError(
                                "posterior-adoption diagnostic omitted its routed action branch"
                            )
                        action_loss = omitted_policy.official_action_loss
                        action_loss_route = "routed_omitted_static"
                    else:
                        action_loss = factual_action_loss
                        action_loss_route = "factual"
                    if (
                        not isinstance(action_loss, torch.Tensor)
                        or action_loss.ndim != 0
                        or not torch.isfinite(action_loss)
                    ):
                        raise RuntimeError(
                            "recurrent causal diagnostic omitted its official action loss"
                        )
                    evidence = evaluate_physical_entity_frame(
                        physical_frame_predictions_from_relation(diagnostic.relations[0]),
                        diagnostic.targets[0].targets,
                        frame_loss.assignment,
                        identity_keys=diagnostic.targets[0].identity_keys_by_batch[0],
                    )
                    candidate_tensor = persistent_state_tensor(candidate_state)
                    denominator = (
                        factual_tensor.float().norm().clamp_min(torch.finfo(torch.float32).eps)
                    )
                    manipulation = (
                        candidate_tensor.float() - factual_tensor.float()
                    ).norm() / denominator
                    variant_reports[name] = {
                        "official_action_loss": float(action_loss.float().item()),
                        "action_loss_route": action_loss_route,
                        "factual_official_action_loss": float(factual_action_loss.float().item()),
                        "omitted_static_action_loss": (
                            None
                            if omitted_policy is None
                            else float(omitted_policy.official_action_loss.detach().float().item())
                        ),
                        "entity_loss": float(frame_loss.total.float().item()),
                        "predictive_family_loss": (
                            float(
                                diagnostic.objective.objective.family_terms["predictive"]
                                .detach()
                                .float()
                                .item()
                            )
                            if correction_diagnostic
                            else None
                        ),
                        "correction_terms": (
                            {
                                term_name: float(term_value.detach().float().item())
                                for term_name, term_value in (
                                    diagnostic.objective.objective.normalized_terms.items()
                                )
                                if term_name.startswith(
                                    ("correction/", "filter_prior/", "filter_posterior/")
                                )
                            }
                            if correction_diagnostic
                            else {}
                        ),
                        "correction_valid_counts": (
                            {
                                term_name: count
                                for term_name, count in (
                                    diagnostic.objective.objective.valid_counts.items()
                                )
                                if term_name.startswith(
                                    ("correction/", "filter_prior/", "filter_posterior/")
                                )
                            }
                            if correction_diagnostic
                            else {}
                        ),
                        "mask_focal": float(frame_loss.mask_focal.float().item()),
                        "mask_dice": float(frame_loss.mask_dice.float().item()),
                        "existence_focal": float(frame_loss.existence_focal.float().item()),
                        "ownership_nll": float(frame_loss.ownership_nll.float().item()),
                        "relative_state_manipulation": float(manipulation.item()),
                        "entity_evaluation": evidence,
                    }
                    _emit_progress(
                        "causal_diagnostic_variant_completed",
                        rank=rank,
                        global_step=diagnostic_step,
                        details={"variant": name},
                    )
                    del (
                        diagnostic,
                        frame_loss,
                        evidence,
                        factual_action_loss,
                        omitted_policy,
                        action_loss,
                        candidate_tensor,
                        manipulation,
                        runtime_snapshot,
                    )
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize(device)
                    _emit_progress(
                        "causal_diagnostic_variant_released",
                        rank=rank,
                        global_step=diagnostic_step,
                        details={
                            "variant": name,
                            "allocated_gib": round(
                                torch.cuda.memory_allocated(device) / 2**30,
                                3,
                            ),
                            "reserved_gib": round(
                                torch.cuda.memory_reserved(device) / 2**30,
                                3,
                            ),
                        },
                    )
                report["variants"] = variant_reports
                report["correction_diagnostic_active"] = correction_diagnostic
                report["factual_row_bindings"] = [
                    [list(pair) for pair in bindings]
                    for bindings in prepared_batch.previous_row_bindings
                ]
                if two_pass_diagnostic:
                    if not isinstance(factual_state, NativeLayerwisePosteriorState):
                        raise RuntimeError(
                            "two-pass control intervention requires layerwise posterior state"
                        )
                    report["control_intervention"] = run_v3_control_intervention(
                        previous_state=factual_state,
                        previous_state_valid=prepared_batch.previous_state_valid,
                        collated_batch=collated_batch,
                        diagnostic_seed=derive_subseed(
                            args.seed,
                            diagnostic_schema,
                            str(diagnostic_step),
                            str(rank),
                            "control-intervention",
                        ),
                        prior_host_steps=prior_host_steps,
                    )
            gathered_reports: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_reports, report)
            diagnostic_directory = (
                "two_pass_filter_diagnostics"
                if two_pass_diagnostic
                else (
                    "layerwise_causal_diagnostics" if layerwise_diagnostic else "causal_diagnostics"
                )
            )
            diagnostic_root = args.run_dir / diagnostic_directory / f"step_{diagnostic_step:08d}"
            diagnostic_root.mkdir(parents=True, exist_ok=True)
            write_text_durable_exclusive(
                diagnostic_root / f"rank_{rank}.json",
                json.dumps(report, indent=2, sort_keys=True) + "\n",
            )
            if rank == 0:
                write_text_durable_exclusive(
                    diagnostic_root / "distributed.json",
                    json.dumps(
                        {
                            "schema": diagnostic_schema,
                            "mode": report["mode"],
                            "global_step": diagnostic_step,
                            "rank_reports": gathered_reports,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            dist.barrier()
            return report

        def run_full_modal_action_adoption_phase() -> None:
            """Prove one fresh-process half of released full-modal action adoption."""

            if (
                dense_evidence_bank is None
                or dense_evidence_bank.modalities != DENSE_MODALITIES
                or graph is None
                or not graph.unified_predict_correct
                or current_grid_cache is None
            ):
                raise RuntimeError(
                    "action-adoption acceptance requires the complete ADR-150 graph and caches"
                )

            adoption_started_at = time.monotonic()
            torch.cuda.reset_peak_memory_stats(device)

            def adoption_progress(stage: str, **details: Any) -> None:
                payload = {
                    "schema": "picf-next.full-modal-adoption-progress/v1",
                    "stage": stage,
                    "rank": rank,
                    "elapsed_seconds": round(time.monotonic() - adoption_started_at, 3),
                    "allocated_gib": round(torch.cuda.memory_allocated(device) / 2**30, 3),
                    "reserved_gib": round(torch.cuda.memory_reserved(device) / 2**30, 3),
                    "peak_reserved_gib": round(
                        torch.cuda.max_memory_reserved(device) / 2**30,
                        3,
                    ),
                    **details,
                }
                print(json.dumps(payload, sort_keys=True), flush=True)

            adoption_progress("selection_start")

            selected_step: list[int | None] = [None]
            if rank == 0:
                records = {
                    cache.contract.modality: {
                        record.source_global_index: record for record in cache.records
                    }
                    for cache in dense_evidence_bank.caches
                }
                for candidate_step in range(stream_plan.total_steps):
                    transitions = stream_plan.global_batch(candidate_step).transitions
                    source_indices = tuple(
                        dense_source_identity(transition.sample.sample_key)[0]
                        for transition in transitions
                    )
                    if all(
                        any(
                            records[modality][source_index].available
                            and records[modality][source_index].token_count >= 2
                            for source_index in source_indices
                        )
                        for modality in DENSE_MODALITIES
                    ):
                        selected_step[0] = candidate_step
                        break
            dist.broadcast_object_list(selected_step, src=0)
            adoption_step = selected_step[0]
            if isinstance(adoption_step, bool) or not isinstance(adoption_step, int):
                raise RuntimeError("no frozen global batch supports every full-modal intervention")
            adoption_progress("selection_complete", optimizer_step=adoption_step)

            planned = build_planned_native_calvin_batch(
                stream_plan,
                dataset,
                optimizer_step=adoption_step,
                rank=rank,
                world_size=WORLD_SIZE,
                gradient_accumulation_steps=1,
                accumulation_index=0,
                device=device,
                dtype=torch.bfloat16,
                maximum_control_tokens=args.maximum_control_tokens,
                gradient_suffix_control_tokens=args.prior_gradient_control_tokens,
            )
            factual_batch = collate_planned(planned)
            if factual_batch.modalities is None:
                raise RuntimeError("action-adoption batch omitted typed modalities")
            local_prior_steps = len(factual_batch.effective_prior_control_chunks)
            prior_host_steps = _distributed_prior_host_step_schedule(
                (local_prior_steps,),
                device=device,
                dist=dist,
                torch_module=torch,
            )[0]
            previous_state_valid = torch.zeros(
                factual_batch.routing.batch_size,
                dtype=torch.bool,
                device=device,
            )
            empty_bindings = tuple(() for _key in factual_batch.routing.sample_keys)
            capacity_seeds = tuple(
                derive_subseed(args.seed, "adr150-action-adoption-capacity", key)
                for key in factual_batch.routing.sample_keys
            )

            gathered_keys: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_keys, list(factual_batch.routing.sample_keys))
            sample_keys = sorted(key for values in gathered_keys for key in values)
            if len(sample_keys) != len(set(sample_keys)):
                raise RuntimeError("action-adoption global batch contains duplicate samples")

            anytouch_stream = next(
                stream for stream in factual_batch.modalities.streams if stream.name == "anytouch"
            )
            local_active_anytouch = [
                key
                for key, valid in zip(
                    factual_batch.routing.sample_keys,
                    anytouch_stream.valid,
                    strict=True,
                )
                if bool(valid.any().item())
            ]
            gathered_active: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_active, local_active_anytouch)
            active_anytouch_keys = sorted(key for values in gathered_active for key in values)
            if not active_anytouch_keys:
                raise RuntimeError("action-adoption batch has no active AnyTouch measurement")

            parameter_groups = resolve_action_adoption_parameter_groups(
                tuple(policy.named_parameters())
            )
            adoption_progress(
                "batch_ready",
                optimizer_step=adoption_step,
                local_sample_keys=list(factual_batch.routing.sample_keys),
                local_prior_host_steps=prior_host_steps,
            )

            def publish_report(payload: Mapping[str, Any], *, filename: str) -> None:
                adoption_progress("publication_start", filename=filename)
                publication_error: list[str | None] = [None]
                if rank == 0:
                    try:
                        write_text_durable_exclusive(
                            args.run_dir / filename,
                            json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        )
                    except BaseException as error:
                        publication_error[0] = f"{type(error).__name__}: {error}"
                dist.broadcast_object_list(publication_error, src=0)
                if publication_error[0] is not None:
                    raise RuntimeError(
                        f"action-adoption phase publication failed: {publication_error[0]}"
                    )
                dist.barrier()
                adoption_progress("complete", filename=filename)

            def action_objective(batch: Any) -> Any:
                return run_task_independent_calvin_joint_sequence_objective(
                    policy,
                    batches=(batch,),
                    physical_sidecar=physical_sidecar,
                    objective_config=objective_config,
                    patch_size=patch_size,
                    merge_size=merge_size,
                    previous_state=cold_previous_state(batch),
                    previous_state_valid=previous_state_valid,
                    prior_row_bindings_by_batch=empty_bindings,
                    graph=graph,
                    current_grid_cache=current_grid_cache,
                    predictive_minimum_visible_fraction=0.0,
                    predictive_loss_power=args.predictive_loss_power,
                    minimum_supervised_fraction=args.minimum_supervised_fraction,
                    capacity_seeds=capacity_seeds,
                    prior_host_steps_by_batch=(prior_host_steps,),
                    prior_gradient_suffix_steps_by_batch=(1,),
                    future_latent_target=future_latent_target_for_batch(batch),
                )

            presence_reports: list[dict[str, Any]] = []
            presence_subsets = (
                DENSE_PRESENCE_SUBSETS if args.acceptance_mode == "action-adoption-presence" else ()
            )
            for subset_index, present in enumerate(presence_subsets):
                subset_name = dense_presence_code(present)
                subset_batch = replace(
                    factual_batch,
                    modalities=with_dense_presence(factual_batch.modalities, present),
                )
                optimizer.zero_grad(set_to_none=True)
                runtime_snapshot = snapshot_official_runtime_buffers()
                result = None
                try:
                    with torch.random.fork_rng(devices=[local_rank]):
                        arm_seed = derive_subseed(
                            args.seed,
                            ACTION_ADOPTION_CORE_SCHEMA,
                            str(adoption_step),
                            str(subset_index),
                        )
                        torch.manual_seed(arm_seed)
                        torch.cuda.manual_seed(arm_seed)
                        adoption_progress(
                            "presence_forward_start",
                            subset_index=subset_index,
                            subset=subset_name,
                        )
                        result = action_objective(subset_batch)
                        torch.cuda.synchronize(device)
                        adoption_progress(
                            "presence_forward_complete",
                            subset_index=subset_index,
                            subset=subset_name,
                        )
                        adoption_progress(
                            "presence_backward_start",
                            subset_index=subset_index,
                            subset=subset_name,
                        )
                        result.primary.official_action_loss.backward()
                        torch.cuda.synchronize(device)
                        adoption_progress(
                            "presence_backward_complete",
                            subset_index=subset_index,
                            subset=subset_name,
                        )
                finally:
                    restore_official_runtime_buffers(runtime_snapshot)
                if result is None:
                    raise RuntimeError("action-adoption objective returned no result")
                gradients = distributed_action_adoption_gradients(
                    parameter_groups,
                    device=device,
                    dist=dist,
                )

                def required_gradient(
                    name: str,
                    *,
                    required: bool,
                    observed: Any = gradients,
                ) -> dict[str, Any]:
                    measured = dict(observed[name])
                    norm = measured["norm"]
                    if required:
                        if norm is None or float(norm) < ACTION_ADOPTION_NONZERO_GRADIENT_MIN_NORM:
                            raise RuntimeError(
                                f"action-only gradient did not reach required path {name}"
                            )
                    elif norm not in (None, 0.0):
                        raise RuntimeError(f"absent modality leaked action gradient through {name}")
                    return measured

                adapters = {
                    modality: {
                        branch: required_gradient(
                            f"{modality}_{branch}_adapter",
                            required=modality in present,
                        )
                        for branch in ("value", "metadata")
                    }
                    for modality in DENSE_MODALITIES
                }
                presence_reports.append(
                    {
                        "name": subset_name,
                        "present_modalities": list(present),
                        "sample_keys": sample_keys,
                        "adapter_action_only_gradients": adapters,
                        "host_action_only_gradients": {
                            str(layer): required_gradient(f"host_layer_{layer}", required=True)
                            for layer in (0, 18, 35)
                        },
                        "qwen_action_expert_action_only_gradient": required_gradient(
                            "action_expert", required=True
                        ),
                        "action_out_proj_action_only_gradient": required_gradient(
                            "action_output", required=True
                        ),
                    }
                )
                optimizer.zero_grad(set_to_none=True)
                del result
                gc.collect()
                adoption_progress(
                    "presence_complete",
                    subset_index=subset_index,
                    subset=subset_name,
                )

            if args.acceptance_mode == "action-adoption-presence":
                presence_report = make_action_adoption_presence_report(
                    probe_optimizer_step=adoption_step,
                    nonzero_gradient_min_norm=ACTION_ADOPTION_NONZERO_GRADIENT_MIN_NORM,
                    presence_subsets=presence_reports,
                    active_anytouch_sample_keys=active_anytouch_keys,
                    parameter_groups={
                        group.name: group.parameter_names for group in parameter_groups
                    },
                )
                publish_report(
                    presence_report,
                    filename="full_modal_action_adoption_presence.json",
                )
                return

            diagnostic_seed = derive_subseed(
                args.seed,
                ACTION_ADOPTION_CORE_SCHEMA,
                str(adoption_step),
                "interventions",
            )

            def diagnostic_action_output(batch: Any, *, label: str) -> Any:
                optimizer.zero_grad(set_to_none=True)
                runtime_snapshot = snapshot_official_runtime_buffers()
                captured: list[Any]
                result = None
                try:
                    with torch.random.fork_rng(devices=[local_rank]):
                        torch.manual_seed(diagnostic_seed)
                        torch.cuda.manual_seed(diagnostic_seed)
                        adoption_progress("intervention_forward_start", intervention=label)
                        with capture_action_projection_output(policy) as captured:
                            result = action_objective(batch)
                        torch.cuda.synchronize(device)
                        adoption_progress("intervention_forward_complete", intervention=label)
                        adoption_progress("intervention_backward_start", intervention=label)
                        result.primary.official_action_loss.backward()
                        torch.cuda.synchronize(device)
                        adoption_progress("intervention_backward_complete", intervention=label)
                finally:
                    adoption_progress("intervention_runtime_restore_start", intervention=label)
                    restore_official_runtime_buffers(runtime_snapshot)
                    adoption_progress("intervention_runtime_restore_complete", intervention=label)
                if result is None:
                    raise RuntimeError("action intervention objective returned no result")
                output = single_captured_action_output(captured)
                adoption_progress(
                    "intervention_output_materialized",
                    intervention=label,
                    output_device=str(output.device),
                    output_dtype=str(output.dtype),
                    output_shape=list(output.shape),
                )
                optimizer.zero_grad(set_to_none=True)
                del result
                adoption_progress("intervention_cleanup_complete", intervention=label)
                return output

            def maximum_action_drift(left: Any, right: Any, *, label: str) -> float:
                adoption_progress("intervention_drift_start", intervention=label)
                measured = distributed_maximum_action_drift(
                    left,
                    right,
                    dist=dist,
                )
                adoption_progress(
                    "intervention_drift_complete",
                    intervention=label,
                    measured_max_abs_action_drift=measured,
                )
                return measured

            factual_output = diagnostic_action_output(factual_batch, label="factual")
            repeated_output = diagnostic_action_output(factual_batch, label="factual_repeat")
            factual_repeat_drift = maximum_action_drift(
                factual_output,
                repeated_output,
                label="factual_repeat",
            )
            if factual_repeat_drift > ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT:
                raise RuntimeError("identical full-modal action forwards are not stable")

            intervention_reports: list[dict[str, Any]] = []
            for modality in DENSE_MODALITIES:
                global_changed: dict[str, int] = {}
                drifts: dict[str, float] = {}
                local_token_evidence: list[dict[str, Any]] | None = None
                for name in MODALITY_INTERVENTIONS:
                    label = f"{modality}:{name}"
                    adoption_progress("intervention_variant_start", intervention=label)
                    variant = intervene_modality(
                        factual_batch.modalities,
                        modality=modality,
                        intervention=name,
                        require_change=False,
                    )
                    adoption_progress(
                        "intervention_variant_complete",
                        intervention=label,
                        local_changed_elements=variant.changed_elements,
                    )
                    gathered_changed: list[Any] = [None for _ in range(WORLD_SIZE)]
                    adoption_progress(
                        "intervention_changed_count_gather_start",
                        intervention=label,
                    )
                    dist.all_gather_object(gathered_changed, variant.changed_elements)
                    if any(
                        isinstance(value, bool) or not isinstance(value, int) or value < 0
                        for value in gathered_changed
                    ):
                        raise RuntimeError(f"{modality} {name} changed-count is invalid")
                    global_changed[name] = sum(gathered_changed)
                    adoption_progress(
                        "intervention_changed_count_gather_complete",
                        intervention=label,
                        global_changed_elements=global_changed[name],
                    )
                    if global_changed[name] <= 0:
                        raise RuntimeError(f"{modality} {name} changed no global evidence")
                    output = diagnostic_action_output(
                        replace(factual_batch, modalities=variant.batch),
                        label=label,
                    )
                    drifts[name] = maximum_action_drift(
                        factual_output,
                        output,
                        label=label,
                    )
                    if name == "value_permutation":
                        local_token_evidence = [
                            {
                                "sample_key": key,
                                "token_permutation": list(permutation),
                                "valid_before": list(before),
                                "valid_after": list(after),
                            }
                            for key, permutation, before, after in zip(
                                factual_batch.routing.sample_keys,
                                variant.token_permutations,
                                variant.valid_before,
                                variant.valid_after,
                                strict=True,
                            )
                        ]
                    del output, variant
                    adoption_progress("intervention_variant_released", intervention=label)
                for name in ("value_zero", "metadata_zero", "value_permutation"):
                    if drifts[name] < ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT:
                        raise RuntimeError(f"{modality} {name} did not affect released action")
                if drifts["joint_permutation"] > ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT:
                    raise RuntimeError(
                        f"{modality} joint token permutation changed released action"
                    )
                if local_token_evidence is None:
                    raise RuntimeError(f"{modality} value permutation evidence is absent")
                gathered_token_evidence: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(gathered_token_evidence, local_token_evidence)
                token_evidence = sorted(
                    (item for values in gathered_token_evidence for item in values),
                    key=lambda item: item["sample_key"],
                )
                intervention_reports.append(
                    {
                        "modality": modality,
                        "sample_keys": [item["sample_key"] for item in token_evidence],
                        "token_permutations": [
                            item["token_permutation"] for item in token_evidence
                        ],
                        "valid_before": [item["valid_before"] for item in token_evidence],
                        "valid_after": [item["valid_after"] for item in token_evidence],
                        "factual_repeat": {
                            "measured_max_abs_action_drift": factual_repeat_drift,
                            "maximum_allowed": ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT,
                        },
                        "value_zero": {
                            "measured_max_abs_action_drift": drifts["value_zero"],
                            "minimum_required": ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT,
                        },
                        "metadata_zero": {
                            "measured_max_abs_action_drift": drifts["metadata_zero"],
                            "minimum_required": ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT,
                        },
                        "value_only_permutation": {
                            "measured_max_abs_action_drift": drifts["value_permutation"],
                            "minimum_required": ACTION_ADOPTION_EFFECT_MIN_ABS_DRIFT,
                        },
                        "joint_value_metadata_valid_permutation": {
                            "measured_max_abs_action_drift": drifts["joint_permutation"],
                            "maximum_allowed": ACTION_ADOPTION_STABILITY_MAX_ABS_DRIFT,
                        },
                    }
                )

            intervention_report = make_action_adoption_interventions_report(
                probe_optimizer_step=adoption_step,
                modality_interventions=intervention_reports,
                active_anytouch_sample_keys=active_anytouch_keys,
            )
            publish_report(
                intervention_report,
                filename="full_modal_action_adoption_interventions.json",
            )

        if args.acceptance_mode in {
            "action-adoption-presence",
            "action-adoption-interventions",
        }:
            run_full_modal_action_adoption_phase()
            return

        dcp_acceptance = args.acceptance_mode in {"dcp-uninterrupted", "dcp-restored"}
        dcp_continuation: dict[str, Any] | None = None
        coordinator = NativeTrainingLaneCoordinator(bank)
        metric_dir = args.run_dir / "metrics" / "rank_journal"
        metric_error = None
        try:
            metric_handle = _prepare_rank_metric_journal(
                metric_dir / f"rank_{rank}.jsonl",
                phase=args.phase,
                load_global_step=args.load_global_step,
            )
        except BaseException as error:
            metric_error = error
        _distributed_raise_if_local_probe_error(
            dist=dist,
            rank=rank,
            world_size=WORLD_SIZE,
            stage="task-independent metric journal preparation",
            local_error=metric_error,
        )
        if metric_handle is None:
            raise RuntimeError("metric journal preparation returned no handle")
        rank_window: list[dict[str, Any]] = []

        if rank == 0 and args.phase == "fresh":
            manifest = {
                "schema": RUNNER_SCHEMA,
                "status": "DECLARED",
                "declared_total_steps": cadence.total_steps,
                "early_stop_step": args.stop_after_step,
                "metrics_every": cadence.metrics_every,
                "visual_every": cadence.visual_every,
                "checkpoint_every": cadence.checkpoint_every,
                "world_size": WORLD_SIZE,
                "global_batch_size": stream_plan.global_batch_size,
                "gradient_accumulation_steps": 1,
                "stream_plan_sha256": stream_plan.plan_sha256,
                "stream_plan_file_sha256": args.stream_plan_sha256,
                "representation_split_file_sha256": args.representation_split_sha256,
                "representation_split_artifact_sha256": (
                    None if representation_split is None else representation_split.artifact_sha256
                ),
                "evaluation_plan_file_sha256": args.evaluation_plan_sha256,
                "evaluation_plan_artifact_sha256": (
                    None if evaluation_plan is None else evaluation_plan.artifact_sha256
                ),
                "action_evaluation": {
                    "registered_steps": list(action_evaluation_steps),
                    "state_mode": "cold_reset",
                    "matched_input_contract": "same_frozen_LBOT_evaluation_plan_and_replay_seed",
                },
                "evaluation_source_episode_indices": (
                    []
                    if representation_split is None
                    else list(representation_split.evaluation_source_episode_indices)
                ),
                "temporal_estimator_sha256": temporal_config.digest,
                "implementation_sha256": implementation_sha256,
                "model_family_sha256": model_family_sha256,
                "lingbot_base_family": lingbot_base_family,
                "lingbot_base_family_sha256": lingbot_base_family_sha256,
                "predictive_cache_manifest_sha256": predictive_manifest_sha256,
                "current_grid_cache_manifest_sha256": current_manifest_sha256,
                "current_grid_cache_binding": current_grid_cache_binding,
                "dense_evidence": {
                    "mode": args.dense_evidence_mode,
                    "modalities": list(
                        () if dense_evidence_bank is None else dense_evidence_bank.modalities
                    ),
                    "cache_manifest_sha256": sorted(args.dense_evidence_cache_manifest_sha256),
                    "supplement_cache_manifest_sha256": sorted(
                        args.dense_evidence_supplement_cache_manifest_sha256
                    ),
                    "composition": dense_evidence_composition,
                    "bindings_sha256": dense_evidence_bindings_sha256,
                    "coverage_plan_file_sha256": (args.dense_evidence_coverage_plan_sha256),
                    "coverage_plan_artifact_sha256": (
                        None
                        if dense_evidence_coverage is None
                        else dense_evidence_coverage.artifact_sha256
                    ),
                    "coverage_records_sha256": (
                        None
                        if dense_evidence_coverage is None
                        else dense_evidence_coverage.records_sha256
                    ),
                    "record_count": (
                        0 if dense_evidence_bank is None else dense_evidence_bank.record_count
                    ),
                    "semantic_owner": "shared_lingbot_host_and_posterior_rows",
                },
                "videomt_stage_pq": {
                    "assets": videomt_stage_pq_asset_receipt,
                    "runtime": videomt_stage_pq_runtime_receipt,
                    "first_execution_receipt_by_rank": (
                        [
                            f"receipts/videomt_stage_pq_rank_{receipt_rank}.json"
                            for receipt_rank in range(WORLD_SIZE)
                        ]
                        if _videomt_stage_pq_active(args)
                        else []
                    ),
                },
                "pretrained_object_memory": object_memory_installation,
                "auxiliary_caches_enabled": {
                    "future": _predictive_assets_required(args),
                    "current_filter_target": _current_correction_assets_required(args),
                    "dense_observation": dense_evidence_bank is not None,
                    "videomt_stage_pq": _videomt_stage_pq_active(args),
                },
                "physical_stream_semantics": {
                    "active": physical_event_stream,
                    "event_identity": (
                        "unique_raw_labelled_physical_event"
                        if physical_event_stream
                        else "language_segment_event"
                    ),
                    "prompt_overlay": (
                        "deterministic_plan_episode_sample_candidates_v1"
                        if physical_event_stream
                        else None
                    ),
                    "state_reset_boundary": (
                        "raw_source_episode_only" if physical_event_stream else "segment"
                    ),
                    "executed_control_receipt": (
                        "all_raw_actions_since_previous_labelled_event"
                        if physical_event_stream
                        else None
                    ),
                    "maximum_control_tokens_per_prior_pass": (
                        args.maximum_control_tokens if physical_event_stream else None
                    ),
                    "prior_gradient_control_tokens": (
                        args.prior_gradient_control_tokens if physical_event_stream else None
                    ),
                    "prior_gradient_schedule": (
                        "ordered_no_grad_burnin_plus_attached_suffix_v1"
                        if physical_event_stream
                        else None
                    ),
                    "native_videomt_source_eligibility": (
                        None
                        if videomt_source_eligibility is None
                        else videomt_source_eligibility.to_dict()
                    ),
                },
                "evidence_sha256": evidence_sha256,
                "execution_contract": execution_contract,
                "execution_contract_sha256": execution_contract_sha256,
                "causal_ablation_mode": args.causal_ablation_mode,
                "parameter_storage": parameter_storage,
                "fsdp2_backward_prefetch": backward_prefetch,
                "lingbot_compile": lingbot_compile_receipt,
                "fsdp2_parameter_layout": fsdp2_parameter_layout,
                "sequential_factual_gradient_storage": (args.sequential_factual_gradient_storage),
                "action_fsdp2_topology": action_topology,
                "vlm_fsdp2_topology": vlm_topology,
                "alignment_teacher_prune": alignment_teacher_prune,
                "modality_bridge": modality_bridge_receipt,
                "learning_rate_stratification": learning_rate_stratification,
                "trainable_scope": trainable_scope_receipt,
                "parameter_manifest": {
                    "parameter_count": parameter_manifest.parameter_count,
                    "trainable_numel": parameter_manifest.trainable_numel,
                    "schema_sha256": parameter_manifest.schema_sha256,
                },
                "objective": asdict(objective_config),
                "source_prediction_mode": args.source_prediction_mode,
                "temporal_credit": {
                    "full_host_local_bptt": False,
                    "posterior_gradient_boundary": "per_optimizer_step",
                    "posterior_state_distribution": "deployment_aged_lane_replay",
                    "persistent_state": (
                        "same_layer_output_memory_BxLxKxd"
                        if args.posterior_architecture in {"layerwise_v2", "two_pass_v3"}
                        else "final_output_rows_BxKxd"
                    ),
                    "long_horizon_objective": (
                        "action_visible_two_pass_filter_with_attached_one_step_egress"
                        if _two_pass_filter_active(args)
                        else (
                            "prior_current_shared_host_correction"
                            if _layerwise_predictive_correction_active(args)
                            else (
                                "disabled_until_layerwise_causal_acceptance"
                                if args.posterior_architecture == "layerwise_v2"
                                else "shared_host_row_prior"
                            )
                        )
                    ),
                },
                "loss_only_supervision_visible_to_model": False,
                "task_conditioned_winner_or_scorer": False,
                "object_action_information_contract": (
                    execution_contract["object_action_information_contract"]
                ),
                "direct_action_posterior_attention": {
                    "active": adr178_direct_action_posterior,
                    "training_only_target_labels": adr178_direct_action_posterior,
                    "deployment_selector": None,
                    "loss_identity": (
                        "GuidedVLA_mean_head_full_key_target_mass"
                        if adr178_direct_action_posterior
                        else None
                    ),
                    "weight": (
                        ADR178_NATIVE_ATTENTION_WEIGHT if adr178_direct_action_posterior else None
                    ),
                    "registered_layer_indices": list(registered_action_posterior_layer_indices),
                    "registered_head_indices": (
                        list(ADR178_REGISTERED_ACTION_HEAD_INDICES)
                        if adr178_direct_action_posterior
                        else []
                    ),
                },
            }
            write_text_durable_exclusive(
                args.run_dir / "run_manifest.json",
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            )
        elif rank == 0 and _causal_ablation_active(args.causal_ablation_mode):
            arm_manifest = {
                "schema": CAUSAL_ABLATION_SCHEMA,
                "mode": args.causal_ablation_mode,
                "branch_global_step": args.load_global_step,
                "stop_after_step": args.stop_after_step,
                "execution_contract_sha256": execution_contract_sha256,
            }
            write_text_durable_exclusive(
                args.run_dir / "causal_arm_manifest.json",
                json.dumps(arm_manifest, indent=2, sort_keys=True) + "\n",
            )
        dist.barrier()

        if args.phase == "fresh" and _action_evaluation_active(args):
            run_registered_action_evaluations(0)

        def make_prior_rollout_factory(
            *,
            controls: tuple[Any, ...],
            horizon: int,
        ) -> TaskIndependentPredictiveRolloutFactory:
            stepper = prior_stepper
            cache = predictive_cache
            if stepper is None or cache is None:
                raise RuntimeError("non-legacy architectures cannot enter the row-only rollout")

            def run_rollout(state: NativePosteriorState) -> Any:
                request = make_native_future_request(
                    source=PredictionSource.PRIOR,
                    batch_size=state.batch_size,
                    horizon=horizon,
                    valid=torch.ones(state.batch_size, dtype=torch.bool, device=device),
                    device=device,
                    dtype=state.rows.dtype,
                    route_id=cache.contract.route_id,
                    address_width=graph.config.prediction_address_width,
                )
                return rollout_native_prior_prediction(
                    stepper,
                    state,
                    controls,
                    request=request,
                    target_name=LINGBOT_PREDICTIVE_TARGET_SPACE,
                )

            return run_rollout

        maximum_peak_reserved_bytes = int(args.maximum_peak_reserved_gib * 1024**3)
        while global_step < args.stop_after_step:
            started = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)
            _emit_progress(
                "step_started",
                rank=rank,
                global_step=global_step + 1,
                details={"input_global_step": global_step},
            )
            planned = None
            primary_batch = None
            videomt_source_batch = None
            wsa_future_observations = None
            materialization_error = None
            try:
                planned = build_planned_native_calvin_batch(
                    stream_plan,
                    dataset,
                    optimizer_step=global_step,
                    rank=rank,
                    world_size=WORLD_SIZE,
                    gradient_accumulation_steps=1,
                    accumulation_index=0,
                    device=device,
                    dtype=torch.bfloat16,
                    maximum_control_tokens=(
                        args.maximum_control_tokens if physical_event_stream else None
                    ),
                    gradient_suffix_control_tokens=(
                        args.prior_gradient_control_tokens if physical_event_stream else None
                    ),
                )
                primary_batch = collate_planned(planned)
                if adr207_native_query:
                    videomt_source_batch = prepare_native_videomt_source_batch(
                        physical_dataset,
                        physical_dataset.index,
                        physical_sidecar,
                        sample_keys=primary_batch.routing.sample_keys,
                        augmentation_seeds=tuple(
                            derive_subseed(
                                seed,
                                "adr207-complete-videomt-current-future4",
                                key,
                            )
                            for seed, key in zip(
                                planned.augmentation_seeds,
                                primary_batch.routing.sample_keys,
                                strict=True,
                            )
                        ),
                        device=device,
                    )
                    if _adr221_full_source_wsa_active(args):
                        wsa_future_observations = tuple(
                            physical_dataset.index.molmoact2_source_observation(
                                indices[-1]
                            ).images
                            for indices in videomt_source_batch.global_indices
                        )
            except BaseException as error:
                materialization_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=WORLD_SIZE,
                stage="task-independent batch materialization",
                local_error=materialization_error,
            )
            if planned is None or primary_batch is None:
                raise RuntimeError("batch materialization returned no batch")
            if adr207_native_query and videomt_source_batch is None:
                raise RuntimeError("ADR-207 materialization returned no complete source batch")
            wsa_da3_teacher_targets = None
            wsa_da3_teacher_receipt = None
            wsa_teacher_error = None
            if _adr221_full_source_wsa_active(args):
                try:
                    if (
                        wsa_teacher_runtime is None
                        or videomt_source_batch is None
                        or wsa_future_observations is None
                    ):
                        raise RuntimeError("ADR-221 WSA teacher transaction is incomplete")
                    wsa_da3_teacher_targets, wsa_da3_teacher_receipt = (
                        wsa_teacher_runtime.build_targets(
                            future_observations=wsa_future_observations,
                            future_source_global_indices=tuple(
                                indices[-1]
                                for indices in videomt_source_batch.global_indices
                            ),
                            device=device,
                            camera_keys=CALVIN_HOST_IMAGE_KEYS,
                        )
                    )
                except BaseException as error:
                    wsa_teacher_error = error
            _distributed_raise_if_local_probe_error(
                dist=dist,
                rank=rank,
                world_size=WORLD_SIZE,
                stage="ADR-221 official t+4 DA3 teacher",
                local_error=wsa_teacher_error,
            )
            attempt = coordinator.begin(optimizer_step=global_step, source_weight_version=0)
            prepared = attempt.prepare(primary_batch.routing)
            (
                objective_previous_state,
                objective_previous_state_valid,
                objective_prior_row_bindings,
            ) = _objective_posterior_inputs(
                mode=args.causal_ablation_mode,
                prepared=prepared,
                torch_module=torch,
            )
            videomt_transaction = None
            if adr207_native_query:
                source_forward_error = None
                try:
                    if (
                        videomt_runtime is None
                        or videomt_source_objective is None
                        or videomt_optimizer is None
                        or videomt_source_batch is None
                        or len(object_query_spatial_specs) != 1
                    ):
                        raise RuntimeError("ADR-207 complete source transaction is incomplete")
                    if objective_previous_state is not None and not isinstance(
                        objective_previous_state,
                        NativeVidEoMTPairedPosteriorState,
                    ):
                        raise TypeError("ADR-207 previous state does not use the paired ABI")
                    videomt_optimizer.zero_grad(set_to_none=True)
                except BaseException as error:
                    source_forward_error = error
            else:
                source_forward_error = None
            source_forward_failures = _distributed_pre_backward_failures(
                source_forward_error,
                rank=rank,
                expected_world_size=WORLD_SIZE,
                dist=dist,
            )
            if source_forward_failures:
                attempt.abort()
                optimizer.zero_grad(set_to_none=True)
                if videomt_optimizer is not None:
                    videomt_optimizer.zero_grad(set_to_none=True)
                if videomt_runtime is not None:
                    videomt_runtime.reset_state()
                if source_forward_error is not None:
                    raise source_forward_error
                raise RuntimeError(
                    f"peer complete source forward failed: {source_forward_failures}"
                )
            metadata = {
                "sample_keys": primary_batch.routing.sample_keys,
                "state_ages": prepared.next_state_ages,
                "available_future": tuple(
                    dataset.available_future_transitions_by_key(key)
                    for key in primary_batch.routing.sample_keys
                ),
                "optimizer_lags": prepared.optimizer_lags,
            }
            gathered_metadata: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_metadata, metadata)
            global_sample_keys = tuple(
                key for value in gathered_metadata for key in value["sample_keys"]
            )
            attached_egress_active = bool(
                _two_pass_filter_active(args)
                and not adr207_native_query
                and all(
                    available > 0
                    for item in gathered_metadata
                    for available in item["available_future"]
                )
            )
            temporal_seed = native_temporal_batch_seed(
                parent_seed=args.seed,
                comparison_id=COMPARISON_ID,
                optimizer_step=global_step,
                sample_keys=global_sample_keys,
            )
            temporal = sample_temporal_batch_plan(
                temporal_config,
                seed=temporal_seed,
                state_ages=tuple(
                    value for item in gathered_metadata for value in item["state_ages"]
                ),
                available_future_steps=tuple(
                    value for item in gathered_metadata for value in item["available_future"]
                ),
                optimizer_lags=tuple(
                    value for item in gathered_metadata for value in item["optimizer_lags"]
                ),
            )
            if global_step + 1 == args.engineering_force_omitted_static_step:
                temporal = replace(temporal, source_masked_branch=True)
            local_count = temporal.local_bptt_steps or 1
            _emit_progress(
                "batch_ready",
                rank=rank,
                global_step=global_step + 1,
                details={
                    "local_bptt_steps": local_count,
                    "overshoot_horizon": temporal.overshoot_horizon or 0,
                    "source_masked_branch": temporal.source_masked_branch,
                    "attached_egress": attached_egress_active,
                    "temporal_plan_sha256": temporal.digest,
                },
            )
            maximum_offset = max(
                local_count - 1,
                temporal.overshoot_horizon or 0,
                int(attached_egress_active),
            )
            continuation_batches = {
                offset: collate_planned(
                    build_native_calvin_continuation_batch(
                        planned,
                        dataset,
                        offset=offset,
                        device=device,
                        dtype=torch.bfloat16,
                        maximum_control_tokens=(
                            args.maximum_control_tokens if physical_event_stream else None
                        ),
                        gradient_suffix_control_tokens=(
                            args.prior_gradient_control_tokens if physical_event_stream else None
                        ),
                    )
                )
                for offset in range(1, maximum_offset + 1)
            }
            batches = (primary_batch,) + tuple(
                continuation_batches[offset] for offset in range(1, local_count)
            )
            egress_batch = continuation_batches.get(1) if attached_egress_active else None
            prior_host_steps_by_batch = None
            prior_gradient_suffix_steps_by_batch = None
            egress_prior_host_steps = None
            if _two_pass_filter_active(args):
                local_prior_counts = tuple(
                    len(batch.effective_prior_control_chunks) for batch in batches
                )
                schedule_input = (
                    *local_prior_counts,
                    *(
                        ()
                        if egress_batch is None
                        else (len(egress_batch.effective_prior_control_chunks),)
                    ),
                )
                aligned_prior_counts = _distributed_prior_host_step_schedule(
                    schedule_input,
                    device=device,
                    dist=dist,
                    torch_module=torch,
                )
                prior_host_steps_by_batch = aligned_prior_counts[: len(batches)]
                prior_gradient_suffix_steps_by_batch = (
                    (1, *prior_host_steps_by_batch[1:])
                    if physical_event_stream
                    else prior_host_steps_by_batch
                )
                egress_prior_host_steps = None if egress_batch is None else aligned_prior_counts[-1]
            rollout_factory: TaskIndependentPredictiveRolloutFactory | None = None
            if temporal.overshoot_horizon is not None:
                horizon = temporal.overshoot_horizon
                controls = tuple(
                    continuation_batches[offset].controls for offset in range(1, horizon + 1)
                )
                rollout_factory = make_prior_rollout_factory(
                    controls=controls,
                    horizon=horizon,
                )
            source_mask = None
            omitted_static = None
            if temporal.source_masked_branch:
                source_seed = derive_subseed(
                    temporal_seed,
                    f"{args.source_prediction_mode}-source-omission",
                    primary_batch.routing.sample_keys[0],
                )
                if args.source_prediction_mode == "current_grid":
                    source_mask = sample_qwen_packed_patch_mask(
                        images=primary_batch.model_inputs["images"],
                        image_valid=primary_batch.model_inputs["img_masks"],
                        image_grid_thw=primary_batch.model_inputs["image_grid_thw"],
                        spatial_merge_size=merge_size,
                        probability=args.source_mask_token_fraction,
                        seed=source_seed,
                        eligible_view_indices=(0,),
                    )
                else:
                    omitted_static = sample_qwen_whole_view_omission(
                        images=primary_batch.model_inputs["images"],
                        image_valid=primary_batch.model_inputs["img_masks"],
                        image_grid_thw=primary_batch.model_inputs["image_grid_thw"],
                        seed=source_seed,
                        eligible_view_indices=(0,),
                    )
            capacity_seeds = tuple(
                derive_subseed(temporal_seed, "capacity-censor", key)
                for key in primary_batch.routing.sample_keys
            )
            posterior_adoption_route = (
                torch.ones(
                    primary_batch.routing.batch_size,
                    dtype=torch.bool,
                    device=device,
                )
                if posterior_adoption_route_active
                and (
                    omitted_static is not None
                    or _adr222_world_token_adoption_active(args)
                )
                else None
            )
            optimizer.zero_grad(set_to_none=True)
            sequential_omitted_active = (
                omitted_static is not None
                and args.omitted_static_rematerialization
                == OMITTED_STATIC_REMATERIALIZATION_SEQUENTIAL_BACKWARD
            )
            prior_entry_execution_state = (
                capture_rank_execution_state() if sequential_omitted_active else None
            )
            omitted_entry_execution_state = None
            sequential_plan = None
            replayed_prior = None
            omitted_result = None
            factual_gradient_spill = None
            result = None
            native_joint_result = None
            shared_query_gradient_moments_local = None
            shared_query_gradient_report = None
            action_posterior_collector = (
                RegisteredActionPosteriorReceiptCollector(
                    registered_layer_indices=registered_action_posterior_layer_indices
                )
                if adr178_direct_action_posterior
                else None
            )
            direct_action_posterior_loss = None
            direct_action_posterior_summary: dict[str, Any] | None = None
            direct_action_posterior_target_audit: tuple[dict[str, Any], ...] = ()
            dcp_step_action_outputs = None
            wsa_step_ledger_receipt = None
            objective_error = None
            _emit_progress(
                "objective_started",
                rank=rank,
                global_step=global_step + 1,
            )
            try:
                capture_step_two = dcp_acceptance and global_step + 1 == 2
                capture_context = (
                    capture_action_projection_output(policy)
                    if capture_step_two
                    else nullcontext(None)
                )
                wsa_transaction_context = (
                    wsa_lingbot_optimizer_transaction(policy)
                    if _adr221_full_source_wsa_active(args)
                    else nullcontext(None)
                )
                with (
                    wsa_transaction_context as wsa_step_ledger,
                    capture_context as captured,
                ):
                    if adr207_native_query:
                        if (
                            videomt_runtime is None
                            or videomt_source_objective is None
                            or videomt_source_batch is None
                            or len(object_query_spatial_specs) != 1
                        ):
                            raise RuntimeError("ADR-207 objective lost its complete source inputs")
                        native_joint_result = run_complete_native_videomt_lingbot_step(
                            policy,
                            videomt_runtime,
                            videomt_source_objective,
                            batch=primary_batch,
                            normalized_padded_rgb=(
                                videomt_source_batch.normalized_padded_rgb
                            ),
                            clip_targets=videomt_source_batch.clip_targets,
                            relation_spec=object_query_spatial_specs[0],
                            previous_state=objective_previous_state,
                            previous_state_valid=objective_previous_state_valid,
                            host_dtype=torch.bfloat16,
                            prior_host_steps=(
                                None
                                if prior_host_steps_by_batch is None
                                else prior_host_steps_by_batch[0]
                            ),
                            prior_gradient_suffix_steps=(
                                None
                                if prior_gradient_suffix_steps_by_batch is None
                                else prior_gradient_suffix_steps_by_batch[0]
                            ),
                            posterior_adoption_route=posterior_adoption_route,
                            wla_world_target=primary_batch.wla_world_target,
                            future_latent_target=(
                                future_latent_target_for_batch(primary_batch)
                            ),
                            future_latent_objective_scale=(
                                args.future_latent_objective_scale
                            ),
                            wsa_da3_teacher_targets=wsa_da3_teacher_targets,
                            wla_host_evidence_arm=args.wla_host_evidence_arm,
                            architecture_identity=graph.config.architecture_identity,
                        )
                        videomt_transaction = native_joint_result.source
                        primary_batch = native_joint_result.host_batch
                        batches = (primary_batch,)
                        if (
                            args.videomt_shared_query_gradient_diagnostic
                            and global_step == args.load_global_step
                        ):
                            shared_query_gradient_moments_local = (
                                shared_query_gradient_moments(native_joint_result)
                            )
                    else:
                        result = run_task_independent_calvin_joint_sequence_objective(
                            policy,
                            batches=batches,
                            physical_sidecar=physical_sidecar,
                            objective_config=objective_config,
                            patch_size=patch_size,
                            merge_size=merge_size,
                            previous_state=objective_previous_state,
                            previous_state_valid=objective_previous_state_valid,
                            prior_row_bindings_by_batch=objective_prior_row_bindings,
                            graph=(graph if args.predictive_weight > 0 else None),
                            current_grid_cache=current_grid_cache,
                            source_mask=source_mask,
                            omitted_static_view=omitted_static,
                            posterior_adoption_route=posterior_adoption_route,
                            factual_action_attention_callback=action_posterior_collector,
                            future_latent_target=(
                                future_latent_target_for_batch(primary_batch)
                            ),
                            future_latent_objective_scale=(
                                args.future_latent_objective_scale
                            ),
                            egress_batch=egress_batch,
                            predictive_rollout_factory=rollout_factory,
                            predictive_cache=(
                                predictive_cache if rollout_factory is not None else None
                            ),
                            predictive_minimum_visible_fraction=(
                                0.0
                                if predictive_cache is None
                                else predictive_cache.contract.minimum_visible_fraction
                            ),
                            predictive_loss_power=args.predictive_loss_power,
                            minimum_supervised_fraction=args.minimum_supervised_fraction,
                            capacity_seeds=capacity_seeds,
                            prior_host_steps_by_batch=prior_host_steps_by_batch,
                            prior_gradient_suffix_steps_by_batch=(
                                prior_gradient_suffix_steps_by_batch
                            ),
                            egress_prior_host_steps=egress_prior_host_steps,
                            omitted_static_rematerialization=(
                                args.omitted_static_rematerialization
                            ),
                            omitted_static_checkpoint_context_fn=(
                                omitted_static_checkpoint_contexts
                                if args.omitted_static_rematerialization
                                == OMITTED_STATIC_REMATERIALIZATION_COMPLETE_CHECKPOINT
                                else None
                            ),
                            omitted_static_forward_context_fn=(
                                suspend_official_gradient_checkpointing
                                if args.omitted_static_rematerialization
                                == OMITTED_STATIC_REMATERIALIZATION_SAVE_ON_CPU
                                else None
                            ),
                        )
                if wsa_step_ledger is not None:
                    wsa_step_ledger_receipt = wsa_step_ledger.receipt()
                if action_posterior_collector is not None:
                    if result is None:
                        raise RuntimeError("legacy action-posterior collection lost its objective")
                    receipts = action_posterior_collector.finalize()
                    if not receipts:
                        raise RuntimeError("ADR-178 collected no native action-posterior receipts")
                    head_count = receipts[0].posterior_attention.shape[1]
                    if any(
                        receipt.posterior_attention.shape[1] != head_count for receipt in receipts
                    ) or head_count <= max(ADR178_REGISTERED_ACTION_HEAD_INDICES):
                        raise RuntimeError(
                            "ADR-178 registered layers have incompatible action heads"
                        )
                    (
                        target_weights,
                        target_valid,
                        direct_action_posterior_target_audit,
                    ) = _direct_action_posterior_targets(
                        bindings_by_batch=result.objective.row_bindings_by_batch,
                        structural_target_requests=(primary_batch.structural_target_requests),
                        capacity=args.capacity,
                        dtype=receipts[0].posterior_attention.dtype,
                        device=device,
                        torch_module=torch,
                    )
                    head_indices = torch.tensor(
                        ADR178_REGISTERED_ACTION_HEAD_INDICES,
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
                    direct_action_posterior_loss = torch.stack(
                        tuple(item.loss for item in attention_results)
                    ).mean()
                    valid_entries = attention_results[0].valid_entries
                    if any(
                        not torch.equal(item.valid_entries, valid_entries)
                        for item in attention_results[1:]
                    ):
                        raise RuntimeError("ADR-178 registered layers disagree on target validity")
                    target_mass = torch.stack(
                        tuple(item.target_mass for item in attention_results)
                    ).mean(dim=0)
                    adoption = torch.stack(
                        tuple(item.total_posterior_mass for item in attention_results)
                    ).mean(dim=0)
                    if bool(valid_entries.any()):
                        conditional = target_mass / adoption.clamp_min(1e-6)
                        direct_action_posterior_summary = {
                            "loss": float(direct_action_posterior_loss.detach().float().item()),
                            "posterior_adoption": float(
                                adoption.masked_select(valid_entries).mean().detach().float().item()
                            ),
                            "target_mass": float(
                                target_mass.masked_select(valid_entries)
                                .mean()
                                .detach()
                                .float()
                                .item()
                            ),
                            "conditional_selectivity": float(
                                conditional.masked_select(valid_entries)
                                .mean()
                                .detach()
                                .float()
                                .item()
                            ),
                            "valid_entry_count": int(valid_entries.sum().item()),
                            "registered_layer_indices": list(
                                registered_action_posterior_layer_indices
                            ),
                            "registered_head_indices": list(ADR178_REGISTERED_ACTION_HEAD_INDICES),
                        }
                    else:
                        direct_action_posterior_summary = {
                            "loss": float(direct_action_posterior_loss.detach().float().item()),
                            "posterior_adoption": None,
                            "target_mass": None,
                            "conditional_selectivity": None,
                            "valid_entry_count": 0,
                            "registered_layer_indices": list(
                                registered_action_posterior_layer_indices
                            ),
                            "registered_head_indices": list(ADR178_REGISTERED_ACTION_HEAD_INDICES),
                        }
                if sequential_omitted_active:
                    if result is None:
                        raise RuntimeError("sequential omitted path lost its legacy objective")
                    if result.sequential_omitted_static is None:
                        raise RuntimeError(
                            "sequential omitted forward produced no deferred branch plan"
                        )
                    omitted_entry_execution_state = capture_rank_execution_state()
                if capture_step_two:
                    dcp_step_action_outputs = captured
            except BaseException as error:
                objective_error = error
            failures = _distributed_pre_backward_failures(
                objective_error,
                rank=rank,
                expected_world_size=WORLD_SIZE,
                dist=dist,
            )
            if failures:
                if prior_entry_execution_state is not None:
                    restore_rank_execution_state(prior_entry_execution_state)
                attempt.abort()
                optimizer.zero_grad(set_to_none=True)
                if videomt_optimizer is not None:
                    videomt_optimizer.zero_grad(set_to_none=True)
                if videomt_runtime is not None and adr207_native_query:
                    videomt_runtime.reset_state()
                if objective_error is not None:
                    raise objective_error
                raise RuntimeError(f"peer objective failed: {failures}")
            if shared_query_gradient_moments_local is not None:
                shared_query_gradient_report = _distributed_shared_query_gradient_report(
                    shared_query_gradient_moments_local,
                    dist=dist,
                    torch_module=torch,
                )
            if adr207_native_query:
                if native_joint_result is None:
                    raise RuntimeError("ADR-207 objective returned no native joint result")
                physical_observability = _native_videomt_step_observability(
                    active=physical_event_stream,
                    planned=planned,
                    primary_batch=primary_batch,
                    source_batch=videomt_source_batch,
                    sequence_batch_count=len(batches),
                    egress_batch=egress_batch,
                    prior_host_steps_by_batch=prior_host_steps_by_batch,
                    prior_gradient_suffix_steps_by_batch=(
                        prior_gradient_suffix_steps_by_batch
                    ),
                    egress_prior_host_steps=egress_prior_host_steps,
                    result=native_joint_result,
                    native_relation_type=NativeObjectQueryPosteriorOutput,
                )
            else:
                if result is None:
                    raise RuntimeError("legacy objective returned no result")
                physical_observability = _physical_step_observability(
                    active=physical_event_stream,
                    planned=planned,
                    primary_batch=primary_batch,
                    sequence_batch_count=len(batches),
                    egress_batch=egress_batch,
                    prior_host_steps_by_batch=prior_host_steps_by_batch,
                    prior_gradient_suffix_steps_by_batch=(
                        prior_gradient_suffix_steps_by_batch
                    ),
                    egress_prior_host_steps=egress_prior_host_steps,
                    result=result,
                )
            _emit_progress(
                "objective_ready",
                rank=rank,
                global_step=global_step + 1,
            )
            try:
                _emit_progress(
                    "backward_started",
                    rank=rank,
                    global_step=global_step + 1,
                    details={
                        "allocated_gib": round(
                            torch.cuda.memory_allocated(device) / 2**30,
                            3,
                        ),
                        "reserved_gib": round(
                            torch.cuda.memory_reserved(device) / 2**30,
                            3,
                        ),
                        "sequential_factual_objective": (
                            "union_normalized_single_root" if sequential_omitted_active else None
                        ),
                    },
                )
                if adr207_native_query:
                    if native_joint_result is None or videomt_transaction is None:
                        raise RuntimeError("ADR-207 backward lost its native joint objective")
                    native_joint_result.total.backward()
                elif sequential_omitted_active:
                    if result is None:
                        raise RuntimeError("sequential backward lost its legacy objective")
                    sequential_plan = result.sequential_omitted_static
                    if (
                        sequential_plan is None
                        or prior_entry_execution_state is None
                        or omitted_entry_execution_state is None
                        or graph is None
                        or objective_previous_state_valid is None
                        or not result.v3_filter_specs
                    ):
                        raise RuntimeError("sequential omitted backward lost its replay contract")
                    # Each branch is a complete FSDP2 backward. Let FSDP finish
                    # and reshard the factual branch before rematerializing the
                    # omitted branch; its reduced sharded gradients remain on
                    # the parameters and the second backward accumulates into
                    # them. This is exact gradient accumulation with one later
                    # optimizer transaction, without keeping FSDP collectives
                    # pending across rank-dependent graphs.
                    factual_backward_error = None
                    try:
                        factual_backward_loss = sequential_plan.factual_backward_loss
                        if direct_action_posterior_loss is not None:
                            factual_backward_loss = (
                                factual_backward_loss
                                + ADR178_NATIVE_ATTENTION_WEIGHT * direct_action_posterior_loss
                            )
                        factual_backward_loss.backward()
                    except BaseException as error:
                        factual_backward_error = error
                    factual_backward_failures = _distributed_pre_backward_failures(
                        factual_backward_error,
                        rank=rank,
                        expected_world_size=WORLD_SIZE,
                        dist=dist,
                    )
                    if factual_backward_failures:
                        if factual_backward_error is not None:
                            raise factual_backward_error
                        raise RuntimeError(
                            f"peer factual backward failed: {factual_backward_failures}"
                        )
                    _emit_progress(
                        "factual_backward_completed",
                        rank=rank,
                        global_step=global_step + 1,
                    )
                    if args.sequential_factual_gradient_storage == FSDP2_FACTUAL_GRADIENT_CPU:
                        factual_gradient_manifest = None
                        allocated_before_spill = 0
                        spill_error = None
                        try:
                            factual_gradient_manifest = fsdp2_present_gradient_manifest(policy)
                            allocated_before_spill = int(torch.cuda.memory_allocated(device))
                            factual_gradient_spill = spill_fsdp2_factual_gradients_to_cpu(policy)
                            if (
                                factual_gradient_manifest["manifest_sha256"]
                                != factual_gradient_spill.manifest_sha256
                                or factual_gradient_manifest["shard_count"]
                                != len(factual_gradient_spill.shards)
                            ):
                                raise RuntimeError("factual gradient manifest changed during spill")
                        except BaseException as error:
                            spill_error = error
                        spill_failures = _distributed_pre_backward_failures(
                            spill_error,
                            rank=rank,
                            expected_world_size=WORLD_SIZE,
                            dist=dist,
                        )
                        if spill_failures:
                            if spill_error is not None:
                                raise spill_error
                            raise RuntimeError(
                                f"peer factual gradient spill failed: {spill_failures}"
                            )
                        if factual_gradient_spill is None:
                            raise RuntimeError("factual gradient spill returned no result")
                        if factual_gradient_manifest is None:
                            raise RuntimeError("factual gradient manifest returned no result")
                        gc.collect()
                        torch.cuda.empty_cache()
                        allocated_after_spill = int(torch.cuda.memory_allocated(device))
                        _emit_progress(
                            "factual_gradients_spilled",
                            rank=rank,
                            global_step=global_step + 1,
                            details={
                                "shard_count": len(factual_gradient_spill.shards),
                                "total_bytes": factual_gradient_spill.total_bytes,
                                "cuda_source_bytes": (factual_gradient_spill.cuda_source_bytes),
                                "cuda_allocated_bytes_before": allocated_before_spill,
                                "cuda_allocated_bytes_after": allocated_after_spill,
                                "cuda_allocated_bytes_released": max(
                                    0,
                                    allocated_before_spill - allocated_after_spill,
                                ),
                                "local_present_gradient_manifest": (factual_gradient_manifest),
                                "spill_manifest_sha256": (factual_gradient_spill.manifest_sha256),
                            },
                        )
                    else:
                        gc.collect()
                        torch.cuda.empty_cache()
                    replayed_prior = None
                    replay_error = None
                    try:
                        restore_rank_execution_state(prior_entry_execution_state)
                        (
                            replayed_prior,
                            replayed_prior_prediction,
                        ) = run_native_v3_prior_chain(
                            policy,
                            graph=graph,
                            previous_memory=objective_previous_state,
                            previous_memory_valid=objective_previous_state_valid,
                            control_chunks=batches[0].effective_prior_control_chunks,
                            filter_prediction=result.v3_filter_specs[0],
                            require_attached_memory=False,
                            host_step_count=(
                                None
                                if prior_host_steps_by_batch is None
                                else prior_host_steps_by_batch[0]
                            ),
                            gradient_suffix_steps=(
                                None
                                if prior_gradient_suffix_steps_by_batch is None
                                else prior_gradient_suffix_steps_by_batch[0]
                            ),
                        )
                        del replayed_prior_prediction
                    except BaseException as error:
                        replay_error = error
                    replay_failures = _distributed_pre_backward_failures(
                        replay_error,
                        rank=rank,
                        expected_world_size=WORLD_SIZE,
                        dist=dist,
                    )
                    if replay_failures:
                        if replay_error is not None:
                            raise replay_error
                        raise RuntimeError(
                            f"peer sequential prior replay failed: {replay_failures}"
                        )
                    if replayed_prior is None:
                        raise RuntimeError("sequential prior replay returned no result")
                    restore_rank_execution_state(omitted_entry_execution_state)
                    omitted_result = None
                    omitted_error = None
                    try:
                        omitted_result = (
                            run_task_independent_calvin_sequential_omitted_static_objective(
                                policy,
                                plan=sequential_plan,
                                prior_trace=replayed_prior,
                            )
                        )
                    except BaseException as error:
                        omitted_error = error
                    omitted_failures = _distributed_pre_backward_failures(
                        omitted_error,
                        rank=rank,
                        expected_world_size=WORLD_SIZE,
                        dist=dist,
                    )
                    if omitted_failures:
                        if omitted_error is not None:
                            raise omitted_error
                        raise RuntimeError(
                            f"peer sequential omitted forward failed: {omitted_failures}"
                        )
                    if omitted_result is None:
                        raise RuntimeError("sequential omitted forward returned no result")
                    omitted_backward_error = None
                    try:
                        omitted_result.backward_loss.backward()
                    except BaseException as error:
                        omitted_backward_error = error
                    omitted_backward_failures = _distributed_pre_backward_failures(
                        omitted_backward_error,
                        rank=rank,
                        expected_world_size=WORLD_SIZE,
                        dist=dist,
                    )
                    if omitted_backward_failures:
                        if omitted_backward_error is not None:
                            raise omitted_backward_error
                        raise RuntimeError(
                            f"peer sequential omitted backward failed: {omitted_backward_failures}"
                        )
                    if factual_gradient_spill is not None:
                        gradient_merge = None
                        merge_error = None
                        try:
                            gradient_merge = merge_fsdp2_factual_gradients_from_cpu(
                                policy,
                                factual_gradient_spill,
                            )
                        except BaseException as error:
                            merge_error = error
                        merge_failures = _distributed_pre_backward_failures(
                            merge_error,
                            rank=rank,
                            expected_world_size=WORLD_SIZE,
                            dist=dist,
                        )
                        if merge_failures:
                            if merge_error is not None:
                                raise merge_error
                            raise RuntimeError(
                                f"peer factual gradient merge failed: {merge_failures}"
                            )
                        if gradient_merge is None:
                            raise RuntimeError("factual gradient merge returned no result")
                        _emit_progress(
                            "factual_gradients_merged",
                            rank=rank,
                            global_step=global_step + 1,
                            details=gradient_merge,
                        )
                        factual_gradient_spill = None
                    result = finalize_task_independent_calvin_sequential_omitted_result(
                        result,
                        omitted_result,
                    )
                else:
                    if result is None:
                        raise RuntimeError("legacy backward lost its objective")
                    training_total = result.objective.objective.total
                    if direct_action_posterior_loss is not None:
                        training_total = (
                            training_total
                            + ADR178_NATIVE_ATTENTION_WEIGHT * direct_action_posterior_loss
                        )
                    training_total.backward()
                _emit_progress(
                    "backward_completed",
                    rank=rank,
                    global_step=global_step + 1,
                )
                if adr207_native_query:
                    if native_joint_result is None:
                        raise RuntimeError("ADR-207 cannot stage an absent paired posterior")
                    posterior = native_joint_result.next_state
                    staged_bindings = tuple(
                        () for _ in range(primary_batch.routing.batch_size)
                    )
                else:
                    if result is None:
                        raise RuntimeError("legacy objective disappeared before lane staging")
                    _validate_posterior_adoption_dose_step(
                        mode=args.acceptance_mode,
                        source_masked_branch=temporal.source_masked_branch,
                        omitted_static_view=omitted_static,
                        posterior_adoption_route=posterior_adoption_route,
                        expected_batch_size=primary_batch.routing.batch_size,
                        result=result,
                    )
                    posterior = (
                        result.committable_context.posterior_memory
                        if args.posterior_architecture in {"layerwise_v2", "two_pass_v3"}
                        else result.committable_context.posterior_state
                    )
                    if posterior is None:
                        raise RuntimeError(
                            "joint objective omitted its architecture-owned recurrent state"
                        )
                    staged_bindings = _staged_row_bindings(
                        mode=args.causal_ablation_mode,
                        observed=result.objective.row_bindings_by_batch,
                    )
                attempt.stage(
                    prepared,
                    posterior,
                    row_bindings_by_batch=staged_bindings,
                )
            except BaseException as error:
                backward_error = error
            else:
                backward_error = None
            backward_failures = _distributed_pre_backward_failures(
                backward_error,
                rank=rank,
                expected_world_size=WORLD_SIZE,
                dist=dist,
            )
            if backward_failures:
                if sequential_omitted_active and prior_entry_execution_state is not None:
                    restore_rank_execution_state(prior_entry_execution_state)
                attempt.abort()
                optimizer.zero_grad(set_to_none=True)
                if videomt_optimizer is not None:
                    videomt_optimizer.zero_grad(set_to_none=True)
                if videomt_runtime is not None and adr207_native_query:
                    videomt_runtime.reset_state()
                if backward_error is not None:
                    raise backward_error
                raise RuntimeError(f"peer backward transaction failed: {backward_failures}")
            gradient_metrics: dict[str, float | int | bool] = {}

            def optimizer_attempt(
                *,
                step: int = global_step,
                metrics: dict[str, float | int | bool] = gradient_metrics,
            ) -> int | None:
                if adr207_native_query:
                    if (
                        videomt_runtime is None
                        or videomt_optimizer is None
                        or videomt_scheduler is None
                    ):
                        raise RuntimeError("ADR-207 optimizer transaction is incomplete")
                    successful, measured = _joint_host_source_optimizer_attempt(
                        policy=policy,
                        host_optimizer=optimizer,
                        source_model=videomt_runtime.model,
                        source_optimizer=videomt_optimizer,
                        source_scheduler=videomt_scheduler,
                        host_auxiliary_scheduler=(
                            wla_scheduler
                            if _wla_complete_active(args)
                            else (
                                wsa_scheduler
                                if _adr221_full_source_wsa_active(args)
                                else None
                            )
                        ),
                        source_update_arm=args.videomt_source_update_arm,
                        global_step=step,
                        max_grad_norm=args.max_grad_norm,
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                else:
                    successful, measured = _optimizer_attempt(
                        policy=policy,
                        optimizer=optimizer,
                        global_step=step,
                        max_grad_norm=args.max_grad_norm,
                        device=device,
                        dist=dist,
                        torch_module=torch,
                    )
                    if successful is not None and _wla_complete_active(args):
                        if wla_scheduler is None:
                            raise RuntimeError("complete WLA optimizer lost its scheduler")
                        wla_scheduler.step()
                        measured["host_auxiliary_scheduler_step"] = int(
                            wla_scheduler.last_epoch
                        )
                        measured["host_auxiliary_learning_rate_min"] = min(
                            float(value) for value in wla_scheduler.get_last_lr()
                        )
                        measured["host_auxiliary_learning_rate_max"] = max(
                            float(value) for value in wla_scheduler.get_last_lr()
                        )
                metrics.update(measured)
                return successful

            if not attempt.finish(optimizer_attempt):
                raise RuntimeError("optimizer update overflowed or was skipped")
            _emit_progress(
                "optimizer_completed",
                rank=rank,
                global_step=global_step + 1,
            )
            dist.barrier()
            global_step += 1
            torch.cuda.synchronize(device)
            if adr207_native_query:
                if native_joint_result is None:
                    raise RuntimeError("ADR-207 optimizer completed without a joint result")
                primary_policy = native_joint_result.policy
            else:
                if result is None:
                    raise RuntimeError("legacy optimizer completed without an objective result")
                primary_policy = result.primary
            object_memory_report = (
                _pretrained_object_memory_step_report(
                    primary_policy.context,
                    capacity=args.capacity,
                    torch_module=torch,
                )
                if adr225_object_memory
                else None
            )
            if dcp_acceptance and global_step == 2:
                if not dcp_step_action_outputs:
                    raise RuntimeError("DCP continuation captured no released action output")
                continuation_rng = _capture_rank_rng(torch, np, device=device)
                local_continuation_boundary = _checkpoint_boundary(
                    model=policy,
                    optimizer=optimizer,
                    lane_snapshot=bank.serialize(),
                    rank_rng_state=continuation_rng,
                    torch_module=torch,
                )
                gathered_boundaries: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(
                    gathered_boundaries,
                    {"rank": rank, "boundary": local_continuation_boundary},
                )
                local_action_output_sha256 = captured_action_outputs_sha256(dcp_step_action_outputs)
                gathered_action_outputs: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(
                    gathered_action_outputs,
                    {"rank": rank, "action_output_sha256": local_action_output_sha256},
                )
                action_loss = primary_policy.official_action_loss.detach().to(
                    device=device,
                    dtype=torch.float64,
                )
                dist.all_reduce(action_loss, op=dist.ReduceOp.SUM)
                action_loss.div_(WORLD_SIZE)
                dcp_continuation = {
                    "global_step": 2,
                    **aggregate_rank_state_digests(gathered_boundaries),
                    "action_output_sha256": aggregate_rank_action_outputs(gathered_action_outputs),
                    "action_loss": float(action_loss.item()),
                }
            step_time = time.perf_counter() - started
            peak_reserved = int(torch.cuda.max_memory_reserved(device))
            if peak_reserved > maximum_peak_reserved_bytes:
                raise RuntimeError("training exceeded the CUDA reservation budget")
            if adr207_native_query:
                if videomt_transaction is None or native_joint_result is None:
                    raise RuntimeError("ADR-207 reporting lost its complete source transaction")
                objective = None
                frame_losses = ()
                omitted_policy = None
                effective_action_loss = primary_policy.official_action_loss
                effective_policy_loss = primary_policy.official_total_loss
                report_objective_total = native_joint_result.total
                report_training_total = native_joint_result.total
                report_family_terms: dict[str, float] = {}
                report_normalized_terms: dict[str, float] = {}
                report_valid_counts: dict[str, int] = {}
                report_row_bindings: list[list[list[object]]] = []
                report_correction_branch_count = 0
                report_current_grid_branch = False
                report_omitted_static_branch = False
                report_omitted_static_action_branch = False
                report_omitted_static_rematerialization = None
                source_objective = videomt_transaction.source_objective
                native_relation = primary_policy.context.relation_output
                if not isinstance(native_relation, NativeObjectQueryPosteriorOutput):
                    raise TypeError("ADR-207 report lost the native object-query relation")
                report_videomt_source_objective: dict[str, Any] | None = {
                    "frame_count": 5,
                    "host_visible_frame_index": 0,
                    "future_frames_visible_to_host": False,
                    "query_count": native_relation.relation.query_count,
                    "target_count": source_objective.target_count,
                    "total": float(source_objective.total.detach().float().item()),
                    "raw_losses": {
                        name: float(value.detach().float().item())
                        for name, value in sorted(source_objective.raw_losses.items())
                    },
                    "weighted_losses": {
                        name: float(value.detach().float().item())
                        for name, value in sorted(source_objective.weighted_losses.items())
                    },
                    "global_indices": [
                        list(indices) for indices in videomt_source_batch.global_indices
                    ],
                    "mean_object_probability": float(
                        native_relation.object_probability.detach().float().mean().item()
                    ),
                    "mean_mask_probability": float(
                        native_relation.support_probability.detach().float().mean().item()
                    ),
                }
            else:
                if result is None:
                    raise RuntimeError("legacy objective result disappeared before reporting")
                objective = result.objective.objective
                frame_losses = result.objective.frame_losses
                omitted_policy = result.omitted_static_policy
                effective_action_loss = primary_policy.official_action_loss
                effective_policy_loss = primary_policy.official_total_loss
                report_objective_total = objective.total
                report_training_total = objective.total + (
                    0.0
                    if direct_action_posterior_loss is None
                    else ADR178_NATIVE_ATTENTION_WEIGHT
                    * direct_action_posterior_loss.detach()
                )
                report_family_terms = {
                    name: float(value.detach().float().item())
                    for name, value in objective.family_terms.items()
                }
                report_normalized_terms = {
                    name: float(value.detach().float().item())
                    for name, value in objective.normalized_terms.items()
                }
                report_valid_counts = objective.valid_counts
                report_row_bindings = [
                    [list(pair) for pair in bindings]
                    for bindings in result.objective.row_bindings_by_batch
                ]
                report_correction_branch_count = len(result.correction_branches)
                report_current_grid_branch = result.current_grid_branch is not None
                report_omitted_static_branch = result.omitted_static_branch is not None
                report_omitted_static_action_branch = result.omitted_static_policy is not None
                report_omitted_static_rematerialization = result.omitted_static_rematerialization
                report_videomt_source_objective = None
            future_alignment_result = primary_policy.future_latent_alignment
            if _future_latent_alignment_active(args) != (
                future_alignment_result is not None
            ):
                raise RuntimeError("FLARE result presence differs from the execution contract")
            future_alignment_report = (
                None
                if future_alignment_result is None
                else {
                    "raw_loss": float(
                        future_alignment_result.raw_loss.detach().float().item()
                    ),
                    "weighted_loss": float(
                        future_alignment_result.weighted_loss.detach().float().item()
                    ),
                    "objective_scale": args.future_latent_objective_scale,
                    "objective_contribution": float(
                        (
                            future_alignment_result.weighted_loss
                            * args.future_latent_objective_scale
                        )
                        .detach()
                        .float()
                        .item()
                    ),
                    "mean_cosine": float(
                        future_alignment_result.mean_cosine.detach().float().item()
                    ),
                    "target_manifest_sha256": (
                        future_alignment_result.target_manifest_sha256
                    ),
                    "capture_layer_index": (
                        future_alignment_result.capture_layer_index
                    ),
                    "action_layer_count": future_alignment_result.action_layer_count,
                    "future_token_count": future_alignment_result.future_token_count,
                }
            )
            wla_report = None
            if _wla_complete_active(args):
                if (
                    primary_policy.action_backend != WLA_COMPLETE_ACTION_BACKEND
                    or primary_policy.backend_metrics is None
                    or primary_batch.wla_world_target is None
                    or wla_scheduler is None
                    or wla_optimizer_receipt is None
                ):
                    raise RuntimeError("complete WLA reporting lost its factual transaction")
                wla_metric_names = frozenset(primary_policy.backend_metrics)
                if wla_metric_names != {"loss_action", "loss_world"}:
                    raise RuntimeError("complete WLA objective ledger changed")
                wla_metrics = {
                    name: float(value.detach().float().item())
                    for name, value in sorted(primary_policy.backend_metrics.items())
                }
                if not all(math.isfinite(value) and value >= 0.0 for value in wla_metrics.values()):
                    raise RuntimeError("complete WLA emitted invalid objective metrics")
                wla_report = {
                    "backend_identity": WLA_COMPLETE_ACTION_BACKEND,
                    "metrics": wla_metrics,
                    "world_loss_weight": 0.1,
                    "target_source_global_indices": list(
                        primary_batch.wla_world_target.source_global_indices
                    ),
                    "target_source_rgb_sha256": list(
                        primary_batch.wla_world_target.source_rgb_sha256
                    ),
                    "optimizer_contract": wla_optimizer_receipt,
                    "scheduler_step": int(wla_scheduler.last_epoch),
                    "learning_rates": [
                        float(value) for value in wla_scheduler.get_last_lr()
                    ],
                }
            wsa_report = None
            if _adr221_full_source_wsa_active(args):
                if (
                    wsa_da3_teacher_receipt is None
                    or wsa_step_ledger_receipt is None
                    or wsa_donor_optimizer is None
                    or wsa_scheduler is None
                ):
                    raise RuntimeError("ADR-221 reporting lost its factual WSA transaction")
                wsa_metrics = primary_policy.official_outputs[6]
                if not isinstance(wsa_metrics, dict):
                    raise RuntimeError("ADR-221 official metrics are not a dictionary")
                wsa_metric_values = {}
                for metric_name in (
                    "loss_future_3d",
                    "loss_future_3d_weighted",
                    "loss_future_3d_objective",
                ):
                    metric = wsa_metrics.get(metric_name)
                    if metric is None:
                        raise RuntimeError(f"ADR-221 omitted {metric_name}")
                    measured = float(torch.as_tensor(metric).detach().float().item())
                    if not math.isfinite(measured):
                        raise RuntimeError(f"ADR-221 produced non-finite {metric_name}")
                    wsa_metric_values[metric_name] = measured
                wsa_report = {
                    "teacher": asdict(wsa_da3_teacher_receipt),
                    "step_ledger": wsa_step_ledger_receipt,
                    "metrics": wsa_metric_values,
                    "optimizer_contract": wsa_optimizer_receipt,
                    "scheduler_contract": wsa_scheduler_receipt,
                    "scheduler_step": int(wsa_scheduler.last_epoch),
                    "learning_rates": [
                        float(value) for value in wsa_scheduler.get_last_lr()
                    ],
                }
            if omitted_policy is not None:
                effective_action_loss = torch.stack(
                    (
                        primary_policy.official_action_loss,
                        omitted_policy.official_action_loss,
                    )
                ).mean()
                effective_policy_loss = torch.stack(
                    (
                        primary_policy.official_total_loss,
                        omitted_policy.official_total_loss,
                    )
                ).mean()
            visual_artifacts: list[dict[str, Any]] = []
            step_report = {
                "global_step": global_step,
                "sample_keys": list(primary_batch.routing.sample_keys),
                "lane_ids": list(primary_batch.routing.lane_ids),
                "frame_indices": list(primary_batch.routing.frame_indices),
                "reset": list(primary_batch.routing.reset),
                "source_digest": primary_batch.source_digest,
                "physical_observability": physical_observability,
                "augmentation_seeds": list(planned.augmentation_seeds),
                "flow_noise_seeds": list(planned.flow_noise_seeds),
                "flow_timestep_seeds": list(planned.flow_timestep_seeds),
                "state_ages": list(prepared.next_state_ages),
                "optimizer_lags": list(prepared.optimizer_lags),
                "causal_ablation_mode": args.causal_ablation_mode,
                "posterior_input_mode": (
                    "withheld"
                    if args.causal_ablation_mode in {"current_frame_branch", "zero_state"}
                    else "causal_lane"
                ),
                "available_previous_state_count": int(prepared.previous_state_valid.sum().item()),
                "consumed_previous_state_count": int(objective_previous_state_valid.sum().item()),
                "temporal_plan_sha256": temporal.digest,
                "local_bptt_steps": local_count,
                "overshoot_horizon": temporal.overshoot_horizon or 0,
                "source_masked_branch": temporal.source_masked_branch,
                "attached_egress": attached_egress_active,
                "objective_interface": (
                    (
                        "native_videomt_complete_source_plus_complete_wla_action_world_v1"
                        if _wla_complete_active(args)
                        else "native_videomt_complete_source_plus_official_lingbot_action_v1"
                    )
                    if adr207_native_query
                    else "task_independent_physical_entity_objective_v1"
                ),
                "objective_total": float(report_objective_total.detach().float().item()),
                "training_objective_total": float(
                    report_training_total.detach().float().item()
                ),
                "direct_action_posterior": direct_action_posterior_summary,
                "direct_action_posterior_target_audit": list(direct_action_posterior_target_audit),
                "pretrained_object_memory": object_memory_report,
                "videomt_source_objective": report_videomt_source_objective,
                "future_latent_alignment": future_alignment_report,
                "wla_action_world": wla_report,
                "wsa_future_3d": wsa_report,
                "videomt_stage_pq": {
                    "active": _videomt_stage_pq_active(args),
                    "idle_placement": getattr(
                        args,
                        "videomt_idle_placement",
                        VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
                    ),
                    "current_device": videomt_stage_pq_current_device,
                    "cuda_loads_cumulative": videomt_stage_pq_cuda_loads,
                    "cpu_offloads_cumulative": videomt_stage_pq_cpu_offloads,
                    "placement_seconds_cumulative": videomt_stage_pq_placement_seconds,
                    "eligible_collations_cumulative": videomt_stage_pq_eligible_collations,
                    "ineligible_short_prefix_collations_cumulative": (
                        videomt_stage_pq_ineligible_short_prefix_collations
                    ),
                    "first_execution_receipt_written": (
                        videomt_stage_pq_first_execution_receipt_written
                    ),
                },
                "family_terms": report_family_terms,
                "normalized_terms": report_normalized_terms,
                "valid_counts": report_valid_counts,
                "official_action_loss": float(
                    primary_policy.official_action_loss.detach().float().item()
                ),
                "action_backend": primary_policy.action_backend,
                "wla_host_evidence_arm": args.wla_host_evidence_arm,
                "official_moe_regularizer": float(
                    primary_policy.official_moe_regularizer.detach().float().item()
                ),
                "official_policy_loss": float(
                    primary_policy.official_total_loss.detach().float().item()
                ),
                "omitted_static_action_loss": (
                    None
                    if omitted_policy is None
                    else float(omitted_policy.official_action_loss.detach().float().item())
                ),
                "omitted_static_policy_loss": (
                    None
                    if omitted_policy is None
                    else float(omitted_policy.official_total_loss.detach().float().item())
                ),
                "effective_training_action_loss": float(
                    effective_action_loss.detach().float().item()
                ),
                "effective_training_policy_loss": float(
                    effective_policy_loss.detach().float().item()
                ),
                "frame_losses": [
                    {
                        "total": float(value.total.detach().float().item()),
                        "mask_focal": float(value.mask_focal.detach().float().item()),
                        "mask_dice": float(value.mask_dice.detach().float().item()),
                        "existence_focal": float(value.existence_focal.detach().float().item()),
                        "ownership_nll": float(value.ownership_nll.detach().float().item()),
                    }
                    for value in frame_losses
                ],
                "row_bindings": report_row_bindings,
                "staged_row_bindings": [
                    [list(pair) for pair in bindings] for bindings in staged_bindings
                ],
                "correction_branch_count": report_correction_branch_count,
                "current_grid_branch": report_current_grid_branch,
                "omitted_static_branch": report_omitted_static_branch,
                "omitted_static_action_branch": report_omitted_static_action_branch,
                "omitted_static_rematerialization": (
                    report_omitted_static_rematerialization
                ),
                "sequential_factual_gradient_storage": (args.sequential_factual_gradient_storage),
                "gradient_metrics": gradient_metrics,
                "videomt_shared_query_gradient_diagnostic": (
                    shared_query_gradient_report
                ),
                "step_time_s": step_time,
                "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
                "peak_cuda_reserved_bytes": peak_reserved,
                "posterior_bank_sha256": _posterior_bank_digest(bank),
                "visual_artifacts": visual_artifacts,
            }
            _append_rank_metric(
                metric_handle,
                step_report,
                durable=(
                    cadence.metrics_due(global_step)
                    or _anchor_evaluation_due(args=args, global_step=global_step)
                ),
            )
            rank_window.append(step_report)
            checkpoint_due = (
                cadence.checkpoint_due(global_step)
                or _scientific_terminal_checkpoint_due(
                    stop_after_step=args.stop_after_step,
                    global_step=global_step,
                )
                or _causal_checkpoint_due(
                    mode=args.causal_ablation_mode,
                    global_step=global_step,
                )
                or _acceptance_checkpoint_due(
                    mode=args.acceptance_mode,
                    global_step=global_step,
                )
            )
            if checkpoint_due:
                _publish_checkpoint(
                    args=args,
                    checkpointer=checkpointer,
                    policy=policy,
                    optimizer=optimizer,
                    bank=bank,
                    global_step=global_step,
                    implementation_sha256=implementation_sha256,
                    model_family_sha256=model_family_sha256,
                    stream_plan_sha256=stream_plan.plan_sha256,
                    temporal_sha256=temporal_config.digest,
                    predictive_manifest_sha256=predictive_manifest_sha256,
                    current_manifest_sha256=current_manifest_sha256,
                    evidence_sha256=evidence_sha256,
                    execution_contract_sha256=execution_contract_sha256,
                    torch_module=torch,
                    numpy_module=np,
                    device=device,
                    rank=rank,
                    dist=dist,
                    source_model=(
                        videomt_runtime.model if adr207_native_query else None
                    ),
                    source_optimizer=(videomt_optimizer if adr207_native_query else None),
                    source_scheduler=(videomt_scheduler if adr207_native_query else None),
                    wsa_scheduler=(
                        wsa_scheduler if _adr221_full_source_wsa_active(args) else None
                    ),
                    wla_scheduler=(
                        wla_scheduler if _wla_complete_active(args) else None
                    ),
                )
                latest_checkpoint_global_step = global_step
            if (
                cadence.visual_due(global_step)
                or _anchor_evaluation_due(args=args, global_step=global_step)
                or _acceptance_terminal_evidence_due(
                    mode=args.acceptance_mode,
                    global_step=global_step,
                )
            ):
                if adr207_native_query:
                    if (
                        videomt_source_batch is None
                        or videomt_transaction is None
                        or native_joint_result is None
                    ):
                        raise RuntimeError("ADR-207 visual audit lost its source transaction")
                    visual_artifacts = render_native_videomt_query_visuals(
                        output_root=args.run_dir,
                        global_step=global_step,
                        input_weight_global_step=global_step - 1,
                        rank=rank,
                        normalized_padded_rgb=(
                            videomt_source_batch.normalized_padded_rgb
                        ),
                        clip_targets=videomt_source_batch.clip_targets,
                        identity_keys=videomt_source_batch.identity_keys,
                        source_output=videomt_transaction.current_output,
                        sample_keys=primary_batch.routing.sample_keys,
                    )
                    visual_schema = "picf-next.native-videomt-query-visual-manifest/v1"
                    visual_directory_name = "native_videomt_query_visuals"
                else:
                    if result is None:
                        raise RuntimeError("legacy visual audit lost its objective")
                    visual_artifacts = render_task_independent_entity_visuals(
                        output_root=args.run_dir,
                        global_step=global_step,
                        input_weight_global_step=global_step - 1,
                        rank=rank,
                        host_items=planned.training.host_items,
                        model_inputs=primary_batch.model_inputs,
                        relation=result.relations[0],
                        target_bundle=result.targets[0],
                        set_loss=frame_losses[0],
                        sample_keys=primary_batch.routing.sample_keys,
                        merge_size=merge_size,
                    )
                    visual_schema = "picf-next.task-independent-entity-visual-manifest/v1"
                    visual_directory_name = "entity_visuals"
                step_report["visual_artifacts"] = visual_artifacts
                visual_manifest_path = (
                    args.run_dir
                    / visual_directory_name
                    / f"step_{global_step:08d}"
                    / f"rank_{rank}"
                    / "artifacts.json"
                )
                write_text_durable_exclusive(
                    visual_manifest_path,
                    json.dumps(
                        {
                            "schema": visual_schema,
                            "global_step": global_step,
                            "rank": rank,
                            "artifacts": visual_artifacts,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            if rank == 0:
                print(
                    json.dumps(
                        {
                            "event": "task_independent_full_step",
                            "global_step": global_step,
                            "objective_total": step_report["objective_total"],
                            "official_action_loss": step_report["official_action_loss"],
                            "step_time_s": step_time,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            _emit_progress(
                "step_completed",
                rank=rank,
                global_step=global_step,
                details={
                    "official_action_loss": step_report["official_action_loss"],
                    "step_time_s": step_time,
                },
            )
            if cadence.metrics_due(global_step):
                _publish_metric_window(
                    run_dir=args.run_dir,
                    global_step=global_step,
                    rank_window=rank_window,
                    rank=rank,
                    dist=dist,
                )
                rank_window = []
            if videomt_runtime is not None and adr207_native_query:
                videomt_runtime.reset_state()
            result = None
            native_joint_result = None
            shared_query_gradient_moments_local = None
            shared_query_gradient_report = None
            videomt_transaction = None
            videomt_source_batch = None
            objective = None
            frame_losses = None
            omitted_policy = None
            effective_action_loss = None
            effective_policy_loss = None
            primary_policy = None
            source_objective = None
            native_relation = None
            report_videomt_source_objective = None
            posterior = None
            dcp_step_action_outputs = None
            captured = None
            capture_context = None
            sequential_plan = None
            replayed_prior = None
            omitted_result = None
            factual_gradient_spill = None
            prior_entry_execution_state = None
            omitted_entry_execution_state = None
            rollout_factory = None
            source_mask = None
            omitted_static = None
            posterior_adoption_route = None
            batches = None
            continuation_batches = None
            egress_batch = None
            gc.collect()
            torch.cuda.empty_cache()
            _emit_progress(
                "step_graph_released",
                rank=rank,
                global_step=global_step,
                details={
                    "allocated_gib": round(
                        torch.cuda.memory_allocated(device) / 2**30,
                        3,
                    ),
                    "reserved_gib": round(
                        torch.cuda.memory_reserved(device) / 2**30,
                        3,
                    ),
                },
            )
            causal_report = run_causal_diagnostic(
                diagnostic_step=global_step,
                prepared_batch=prepared,
                collated_batch=primary_batch,
                planned_batch=planned,
                prior_host_steps=(
                    None if prior_host_steps_by_batch is None else prior_host_steps_by_batch[0]
                ),
            )
            if rank == 0 and causal_report is not None:
                print(
                    json.dumps(
                        {
                            "event": (
                                "adr149_two_pass_filter_diagnostic"
                                if args.posterior_architecture == "two_pass_v3"
                                else (
                                    "layerwise_posterior_causal_diagnostic"
                                    if args.posterior_architecture == "layerwise_v2"
                                    else "adr146_causal_diagnostic"
                                )
                            ),
                            "global_step": global_step,
                            "mode": causal_report["mode"],
                            "eligible": causal_report["eligible"],
                            "variant_entity_losses": {
                                name: values["entity_loss"]
                                for name, values in causal_report["variants"].items()
                            },
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if _action_evaluation_active(args) and global_step in action_evaluation_steps:
                run_registered_action_evaluations(global_step)
            stop_request = [False]
            if rank == 0:
                stop_request[0] = _external_stop_requested(
                    run_dir=args.run_dir,
                    checkpoint_due=checkpoint_due,
                )
            dist.broadcast_object_list(stop_request, src=0)
            if stop_request[0]:
                break

        if metric_handle is not None:
            metric_handle.flush()
            os.fsync(metric_handle.fileno())
        if rank_window:
            gathered_tail: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(gathered_tail, {"rank": rank, "steps": rank_window})
            if rank == 0:
                output = args.run_dir / "metrics" / f"tail_through_{global_step:08d}.json"
                write_text_durable_exclusive(
                    output,
                    json.dumps(
                        {
                            "schema": METRIC_WINDOW_SCHEMA,
                            "partial": True,
                            "end_global_step": global_step,
                            "rank_reports": gathered_tail,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
        if dcp_acceptance:
            if dcp_continuation is None:
                raise RuntimeError("DCP acceptance produced no exact step-2 continuation")
            checkpoint_dir = args.run_dir / "checkpoints" / "global_step_1"
            checkpoint_evidence: list[Any] = [None]
            checkpoint_error: list[str | None] = [None]
            if rank == 0:
                try:
                    checkpoint_report_path = checkpoint_dir / "task_independent_checkpoint.json"
                    checkpoint_report = json.loads(
                        checkpoint_report_path.read_text(encoding="utf-8")
                    )
                    if (
                        checkpoint_report.get("schema") != CHECKPOINT_SCHEMA
                        or checkpoint_report.get("status") != "PASS"
                        or checkpoint_report.get("global_step") != 1
                    ):
                        raise RuntimeError("DCP step-1 checkpoint report contract differs")
                    if args.acceptance_mode == "dcp-restored":
                        uninterrupted_report = validate_action_dcp_phase_report(
                            json.loads(
                                (args.run_dir / "acceptance" / "dcp_uninterrupted.json").read_text(
                                    encoding="utf-8"
                                )
                            ),
                            expected_phase="uninterrupted",
                        )
                        checkpoint_artifact_sha256 = uninterrupted_report[
                            "checkpoint_artifact_sha256"
                        ]
                    else:
                        checkpoint_artifact_sha256 = directory_tree_sha256(checkpoint_dir)
                    checkpoint_evidence[0] = {
                        "rank_boundaries": checkpoint_report["rank_boundaries"],
                        "checkpoint_artifact_sha256": checkpoint_artifact_sha256,
                    }
                except BaseException as error:
                    checkpoint_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(checkpoint_error, src=0)
            if checkpoint_error[0] is not None:
                raise RuntimeError(f"DCP checkpoint evidence failed: {checkpoint_error[0]}")
            dist.broadcast_object_list(checkpoint_evidence, src=0)
            saved_boundary = aggregate_rank_state_digests(checkpoint_evidence[0]["rank_boundaries"])
            if args.acceptance_mode == "dcp-restored":
                if loaded_boundary is None:
                    raise RuntimeError("DCP restored process has no loaded boundary")
                restored_rank_boundaries: list[Any] = [None for _ in range(WORLD_SIZE)]
                dist.all_gather_object(
                    restored_rank_boundaries,
                    {"rank": rank, "boundary": loaded_boundary},
                )
                observed_boundary = aggregate_rank_state_digests(restored_rank_boundaries)
                if observed_boundary != saved_boundary:
                    raise RuntimeError("DCP loaded distributed boundary differs from checkpoint")
                phase = "restored"
            else:
                observed_boundary = saved_boundary
                phase = "uninterrupted"

            rank_processes: list[Any] = [None for _ in range(WORLD_SIZE)]
            dist.all_gather_object(
                rank_processes,
                {
                    "rank": rank,
                    "pid": os.getpid(),
                    "start_ticks": current_process_start_ticks(),
                },
            )
            dcp_report = make_action_dcp_phase_report(
                phase=phase,
                process_sha256=process_set_sha256(
                    phase=phase,
                    rank_processes=rank_processes,
                ),
                checkpoint_artifact_sha256=checkpoint_evidence[0]["checkpoint_artifact_sha256"],
                boundary=observed_boundary,
                next_step=dcp_continuation,
            )
            dcp_publication_error: list[str | None] = [None]
            if rank == 0:
                try:
                    acceptance_root = args.run_dir / "acceptance"
                    acceptance_root.mkdir(parents=True, exist_ok=True)
                    write_text_durable_exclusive(
                        acceptance_root / f"dcp_{phase}.json",
                        json.dumps(dcp_report, indent=2, sort_keys=True) + "\n",
                    )
                except BaseException as error:
                    dcp_publication_error[0] = f"{type(error).__name__}: {error}"
            dist.broadcast_object_list(dcp_publication_error, src=0)
            if dcp_publication_error[0] is not None:
                raise RuntimeError(f"DCP phase publication failed: {dcp_publication_error[0]}")
            dist.barrier()
        videomt_stage_pq_rank_summaries: list[Any] = [None for _ in range(WORLD_SIZE)]
        local_videomt_stage_pq_summary = {
            "rank": rank,
            "idle_placement": getattr(
                args,
                "videomt_idle_placement",
                VIDEOMT_IDLE_PLACEMENT_CUDA_RESIDENT,
            ),
            "current_device": videomt_stage_pq_current_device,
            "cuda_loads": videomt_stage_pq_cuda_loads,
            "cpu_offloads": videomt_stage_pq_cpu_offloads,
            "placement_seconds": videomt_stage_pq_placement_seconds,
            "eligible_collations": videomt_stage_pq_eligible_collations,
            "ineligible_short_prefix_collations": (
                videomt_stage_pq_ineligible_short_prefix_collations
            ),
            "first_execution_receipt_written": (
                videomt_stage_pq_first_execution_receipt_written
            ),
        }
        if (
            _videomt_stage_pq_active(args)
            and videomt_stage_pq_eligible_collations > 0
            and not videomt_stage_pq_first_execution_receipt_written
        ):
            raise RuntimeError("VidEoMT executed without publishing its first-execution receipt")
        dist.all_gather_object(
            videomt_stage_pq_rank_summaries,
            local_videomt_stage_pq_summary,
        )
        if rank == 0:
            if action_evaluation_snapshot_reports:
                observed_steps = [
                    item["checkpoint_global_step"] for item in action_evaluation_snapshot_reports
                ]
                expected_steps = [step for step in action_evaluation_steps if step <= global_step]
                curve_status = (
                    "PASS"
                    if global_step >= action_evaluation_steps[-1]
                    and observed_steps == list(action_evaluation_steps)
                    else "PARTIAL"
                )
                if observed_steps != expected_steps:
                    raise RuntimeError("ADR-149 action evaluation curve checkpoints changed")
                evaluation_inputs = {
                    item["evaluation_input_sha256"] for item in action_evaluation_snapshot_reports
                }
                if len(evaluation_inputs) != 1:
                    raise RuntimeError("ADR-149 action evaluation inputs changed across curve")
                write_text_durable_exclusive(
                    args.run_dir / f"action_evaluation_curve_step_{global_step:08d}.json",
                    json.dumps(
                        {
                            "schema": "picf-next.adr149-cold-action-curve/v1",
                            "status": curve_status,
                            "completed_global_step": global_step,
                            "registered_steps": list(action_evaluation_steps),
                            "snapshots": action_evaluation_snapshot_reports,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            if causal_warm_action_evaluation_snapshot_reports:
                observed_warm_steps = [
                    item["checkpoint_global_step"]
                    for item in causal_warm_action_evaluation_snapshot_reports
                ]
                expected_warm_steps = [
                    step
                    for step in ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS
                    if step <= global_step
                ]
                if observed_warm_steps != expected_warm_steps:
                    raise RuntimeError(
                        "ADR-210 causal-warm evaluation checkpoints changed"
                    )
                write_text_durable_exclusive(
                    args.run_dir
                    / f"causal_warm_action_evaluation_curve_step_{global_step:08d}.json",
                    json.dumps(
                        {
                            "schema": "picf-next.adr210-causal-warm-action-curve/v1",
                            "status": (
                                "PASS"
                                if global_step
                                >= ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS[-1]
                                else "PARTIAL"
                            ),
                            "completed_global_step": global_step,
                            "registered_steps": list(
                                ADR210_CAUSAL_WARM_ACTION_EVALUATION_STEPS
                            ),
                            "snapshots": (
                                causal_warm_action_evaluation_snapshot_reports
                            ),
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n",
                )
            summary = {
                "schema": RUNNER_SCHEMA,
                "status": "COMPLETE" if global_step == TOTAL_STEPS else "EARLY_STOP",
                "completed_global_step": global_step,
                "latest_checkpoint_global_step": latest_checkpoint_global_step,
                "saved_global_step": (
                    latest_checkpoint_global_step if latest_checkpoint_global_step > 0 else None
                ),
                "declared_total_steps": TOTAL_STEPS,
                "loaded_boundary_sha256": loaded_boundary,
                "videomt_stage_pq_rank_summaries": videomt_stage_pq_rank_summaries,
            }
            output = args.run_dir / f"run_summary_step_{global_step:08d}.json"
            write_text_durable_exclusive(
                output,
                json.dumps(summary, indent=2, sort_keys=True) + "\n",
            )
        dist.barrier()
    finally:
        if metric_handle is not None:
            metric_handle.close()
        if run_lease is not None:
            run_lease.close()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

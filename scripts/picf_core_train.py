from __future__ import annotations

import argparse
import ast
import contextlib
import dataclasses
import faulthandler
import fnmatch
import json
import logging
import math
import os
import random
import re
import signal
import shutil
import sys
import time
import traceback
from collections import deque
from collections.abc import Iterator
from collections.abc import Mapping
from collections.abc import MutableMapping
from collections.abc import Sequence
from pathlib import Path
from typing import Any
from typing import TextIO

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from PIL import ImageDraw
from torch.nn.parallel import DistributedDataParallel
from torch.nn.parameter import UninitializedParameter
import tqdm.auto as tqdm

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer
except Exception:  # pragma: no cover - depends on torch distributed build
    ZeroRedundancyOptimizer = None

try:
    from torch.distributed.fsdp import BackwardPrefetch
    from torch.distributed.fsdp import FullyShardedDataParallel
    from torch.distributed.fsdp import FullOptimStateDictConfig
    from torch.distributed.fsdp import FullStateDictConfig
    from torch.distributed.fsdp import ShardingStrategy
    from torch.distributed.fsdp import StateDictType
except Exception:  # pragma: no cover - depends on torch distributed build
    BackwardPrefetch = None
    FullyShardedDataParallel = None
    FullOptimStateDictConfig = None
    FullStateDictConfig = None
    ShardingStrategy = None
    StateDictType = None

try:
    import wandb
except Exception:  # pragma: no cover - import availability depends on env
    wandb = None

from openpi.picf.anytouch.config import AnyTouchConfig
from openpi.picf.action_normalization import PicfActionNormalizer
from openpi.picf.action_normalization import default_calvin_action_norm_stats_path
from openpi.picf.contracts import PicfObservation
from openpi.picf.core import PicfCoreConfig
from openpi.picf.core import PicfFullCore
from openpi.picf.core import PicfTransitionLossBreakdown
from openpi.picf.core import PicfTransitionLossConfig
from openpi.picf.core import compute_transition_loss
from openpi.picf.core import future_targets_from_current_targets
from openpi.picf.core import make_action_only_transition_loss
from openpi.picf.paligemma.config import PaliGemmaSemanticConfig
from openpi.picf.paligemma.wrapper import PaliGemmaSemanticEncoder
from openpi.picf.policy import PicfPi05Policy
from openpi.picf.pointcloud_picf import CalvinDepthToPicfPointCloud
from openpi.picf.replay.calvin_replay import _calvin_tactile_packet
from openpi.picf.replay.calvin_replay import _resolve_tactile_calibration
from openpi.picf.sonata.config import SonataPointConfig
from openpi.picf.sonata.wrapper import SonataPointFeatureExtractor
from openpi.picf.vjepa.config import VjepaVisualConfig
from openpi.training.calvin_dataset import CalvinLangSegmentDataset


_DEFAULT_TACTILE_SENSOR_NAMES = ("digit", "gelsight_mini")
_DEFAULT_TACTILE_SENSOR_OFFSETS_M = ((0.01, 0.0, 0.0), (-0.01, 0.0, 0.0))
_MVTRACK_TRACKLET_KEYS = (
    "tracklet_xy",
    "tracklet_velocity",
    "tracklet_visibility",
    "tracklet_confidence",
    "tracklet_ids",
    "tracklet_view_ids",
    "tracklet_age",
)
_MVTRACK_PROPOSAL_KEYS = (
    "proposal_centers_xy",
    "proposal_boxes_xyxy",
    "proposal_objectness",
    "proposal_view_ids",
    "proposal_source_ids",
    "proposal_age",
    "proposal_mask_xy",
    "proposal_mask_weights",
    "proposal_mask_offsets",
)
_SPEC_DEFAULTS = PicfCoreConfig()
_LOSS_DEFAULTS = PicfTransitionLossConfig()
_OBJECT_SCAFFOLD_DECAY_FIELDS = (
    "lambda_anchor_object_pull",
    "lambda_object_explanation_point",
    "lambda_object_explanation_contact",
    "lambda_object_explanation_duplicate",
    "lambda_object_explanation_background",
    "lambda_mapg_support_diversity",
)
_RETRYABLE_FIRST_STEP_ERRORS = (
    "PICF core requires a valid xyzrgb point cloud on the first control step.",
    "PICF core requires non-empty local xyzrgb support on the first control step.",
    "PICF core requires non-empty local xyzrgb support on window step",
)


def _picf_mode_enabled(value: str | argparse.Namespace | None) -> bool:
    if isinstance(value, argparse.Namespace):
        value = getattr(value, "picf_mode", "enabled")
    return str("enabled" if value is None else value).lower().replace("-", "_") == "enabled"
_COMPAT_ALLOWED_MISSING_KEYS = (
    "core.semantic_prefix_proj.*",
    "core.posterior_to_control_proj.*",
    "core.global_post_to_control_proj.*",
    "core.innovation_to_control_proj.*",
    "core.proprio_to_control_proj.*",
    "core.task_query_tokens",
    "core.task_global_query_tokens",
    "core.task_instruction_query_tokens",
    "core.posterior_slot_hidden",
    "core.posterior_slot_token",
    "core.task_query_conditioner.*",
    "core.task_public_reader.*",
    "core.task_visual_reread.*",
    "core.task_tactile_reread.*",
    "core.task_point_reread.*",
    "core.task_self.*",
    "core.task_geom_proj.*",
    "core.task_to_control_proj.*",
    "core.task_global_to_control_proj.*",
    "core.instruction_to_control_proj.*",
    "core.control_role_embedding.*",
    "core.predictive_physical_role_embedding.*",
    "core.physical_pred_to_conditioned_proj.*",
    "core.predictive_conditioned_role_embedding.*",
    "core.control_query_tokens",
    "core.predictive_query_tokens",
    "core.pi_prefix_query_tokens",
    "core.pi_prefix_reader.*",
    "core.action_prefix_teacher_tokens",
    "core.action_prefix_teacher_initialized",
    "core.future_condition_reader.*",
    "core.predictive_semantic_world.*",
    "core.predictive_state_proj.*",
    "core.binding_signature_proj.*",
    "core.tactile_patch_token_proj.*",
    "core.binding_quadratic_diag",
    "core.binding_low_rank_left.*",
    "core.binding_low_rank_right.*",
    "semantic_encoder.encoder.action_context_in_proj.*",
    "semantic_encoder.encoder.action_context_q_proj.*",
    "semantic_encoder.encoder.action_context_k_proj.*",
    "semantic_encoder.encoder.action_context_v_proj.*",
    "semantic_encoder.encoder.action_context_out_proj.*",
    "semantic_encoder.encoder.action_context_gate_logit",
    "semantic_encoder.encoder.action_context_flow_residual_gate_logit",
    "semantic_encoder.encoder.action_context_readout_*",
    "semantic_encoder.encoder.action_expert_router_*",
    "policy.semantic_encoder.encoder.action_context_in_proj.*",
    "policy.semantic_encoder.encoder.action_context_q_proj.*",
    "policy.semantic_encoder.encoder.action_context_k_proj.*",
    "policy.semantic_encoder.encoder.action_context_v_proj.*",
    "policy.semantic_encoder.encoder.action_context_out_proj.*",
    "policy.semantic_encoder.encoder.action_context_gate_logit",
    "policy.semantic_encoder.encoder.action_context_flow_residual_gate_logit",
    "policy.semantic_encoder.encoder.action_context_readout_*",
    "policy.semantic_encoder.encoder.action_expert_router_*",
    "encoder.action_context_in_proj.*",
    "encoder.action_context_q_proj.*",
    "encoder.action_context_k_proj.*",
    "encoder.action_context_v_proj.*",
    "encoder.action_context_out_proj.*",
    "encoder.action_context_gate_logit",
    "encoder.action_context_flow_residual_gate_logit",
    "encoder.action_context_readout_*",
    "encoder.action_expert_router_*",
    "action_context_in_proj.*",
    "action_context_q_proj.*",
    "action_context_k_proj.*",
    "action_context_v_proj.*",
    "action_context_out_proj.*",
    "action_context_gate_logit",
    "action_context_flow_residual_gate_logit",
    "action_context_readout_*",
    "action_expert_router_*",
    "semantic_prefix_proj.*",
    "posterior_to_control_proj.*",
    "global_post_to_control_proj.*",
    "innovation_to_control_proj.*",
    "proprio_to_control_proj.*",
    "task_query_tokens",
    "task_global_query_tokens",
    "task_instruction_query_tokens",
    "posterior_slot_hidden",
    "posterior_slot_token",
    "task_query_conditioner.*",
    "task_public_reader.*",
    "task_visual_reread.*",
    "task_tactile_reread.*",
    "task_point_reread.*",
    "task_self.*",
    "task_geom_proj.*",
    "task_to_control_proj.*",
    "task_global_to_control_proj.*",
    "instruction_to_control_proj.*",
    "control_role_embedding.*",
    "predictive_physical_role_embedding.*",
    "physical_pred_to_conditioned_proj.*",
    "predictive_conditioned_role_embedding.*",
    "control_query_tokens",
    "predictive_query_tokens",
    "pi_prefix_query_tokens",
    "pi_prefix_reader.*",
    "action_prefix_teacher_tokens",
    "action_prefix_teacher_initialized",
    "future_condition_reader.*",
    "predictive_semantic_world.*",
    "predictive_state_proj.*",
    "binding_signature_proj.*",
    "tactile_patch_token_proj.*",
    "binding_quadratic_diag",
    "binding_low_rank_left.*",
    "binding_low_rank_right.*",
)
_COMPAT_ALLOWED_UNEXPECTED_KEYS = (
    "core.semantic_summary_proj.*",
    "core.predictive_semantic_reads.*",
    "core.control_semantic_reads.*",
    "core.control_pool.*",
    "semantic_summary_proj.*",
    "predictive_semantic_reads.*",
    "control_semantic_reads.*",
    "control_pool.*",
    "core.control_semantic_prefix_tokens",
    "core.predictive_conditioned_query_tokens",
    "core.task_fused_public_reader.*",
    "core.task_visual_public_reader.*",
    "core.task_point_public_reader.*",
    "core.task_tactile_public_reader.*",
    "control_semantic_prefix_tokens",
    "predictive_conditioned_query_tokens",
    "task_fused_public_reader.*",
    "task_visual_public_reader.*",
    "task_point_public_reader.*",
    "task_tactile_public_reader.*",
)

_DEBUG_INDEX_GUARDS_INSTALLED = False
_DEBUG_INDEX_TRACE: deque[dict[str, Any]] = deque(maxlen=64)


class _NullTactileEncoder:
    def encode_sensor_clips(self, *, clips_by_sensor, backgrounds_by_sensor, poses_by_sensor):
        del clips_by_sensor, backgrounds_by_sensor, poses_by_sensor
        return None


class _NullVisualEncoder:
    def encode_clip(self, _clip):
        raise AssertionError("visual_map_override should bypass encoder use in picf_core_train")


class _ZeroSemanticEncoder(torch.nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def encode_observation(self, observation: PicfObservation) -> torch.Tensor:
        del observation
        return torch.zeros((1, self.dim), dtype=torch.float32)


@dataclasses.dataclass(frozen=True)
class _DistributedRuntimeEnv:
    torch_distributed_debug: str
    torch_distributed_debug_source: str
    allow_torch_distributed_debug_detail: bool
    torch_nccl_enable_monitoring: str | None
    torch_nccl_heartbeat_timeout_sec: str | None
    pytorch_cuda_alloc_conf: str | None
    pytorch_cuda_alloc_conf_source: str


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() not in {"", "0", "false", "no", "off"}


def _configure_distributed_runtime_env(*, world_size: int, rank: int) -> _DistributedRuntimeEnv:
    requested_raw = os.environ.get("TORCH_DISTRIBUTED_DEBUG", "").strip()
    requested = requested_raw.upper()
    allow_detail = _env_flag("OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL")
    applied = requested
    source = "inherited"

    if int(world_size) > 1:
        if not requested:
            applied = "INFO"
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = applied
            source = "defaulted_for_ddp"
        elif requested == "DETAIL" and not allow_detail:
            raise RuntimeError(
                "DDP runtime guard: TORCH_DISTRIBUTED_DEBUG=DETAIL is not allowed by default. "
                "DETAIL is reserved for explicit distributed-runtime debugging because it can destabilize "
                "standalone TCPStore/NCCL tracing. Set OPENPI_ALLOW_TORCH_DISTRIBUTED_DEBUG_DETAIL=1 to override."
            )
        else:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = requested
            applied = requested

    return _DistributedRuntimeEnv(
        torch_distributed_debug=applied or os.environ.get("TORCH_DISTRIBUTED_DEBUG", "").strip().upper(),
        torch_distributed_debug_source=source,
        allow_torch_distributed_debug_detail=allow_detail,
        torch_nccl_enable_monitoring=os.environ.get("TORCH_NCCL_ENABLE_MONITORING"),
        torch_nccl_heartbeat_timeout_sec=os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"),
        pytorch_cuda_alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        pytorch_cuda_alloc_conf_source="inherited",
    )


def _configure_cuda_allocator_env(*, requested_device: str, training_strategy: str, world_size: int) -> tuple[str | None, str]:
    if not str(requested_device).startswith("cuda"):
        return os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "not_applicable"
    if str(training_strategy).lower().replace("-", "_") != "fsdp_full_shard" or int(world_size) <= 1:
        return os.environ.get("PYTORCH_CUDA_ALLOC_CONF"), "not_applicable"

    requested = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "").strip()
    if requested:
        if "expandable_segments" not in requested.lower():
            raise RuntimeError(
                "FSDP runtime guard: all-backbone CUDA training expects "
                "PYTORCH_CUDA_ALLOC_CONF to include expandable_segments:True. "
                f"Got {requested!r}."
            )
        return requested, "inherited"

    applied = "expandable_segments:True"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = applied
    return applied, "defaulted_for_fsdp"


def _register_fault_dump_handler(*, stream: TextIO | None = None) -> bool:
    """Register SIGUSR1 -> faulthandler stack dump for live hang diagnosis."""
    if not hasattr(signal, "SIGUSR1"):
        return False
    try:
        try:
            faulthandler.unregister(signal.SIGUSR1)
        except RuntimeError:
            pass
        faulthandler.register(signal.SIGUSR1, file=stream or sys.stderr, all_threads=True, chain=False)
    except (RuntimeError, ValueError, OSError):
        return False
    return True


def _debug_index_stack() -> str:
    stack = traceback.format_stack(limit=8)
    return "".join(stack[:-2]).strip()


def _debug_index_tensor_summary(index: torch.Tensor) -> str:
    return (
        f"shape={tuple(index.shape)} dtype={index.dtype} device={index.device} "
        f"min={int(index.min().item()) if index.numel() > 0 else 'n/a'} "
        f"max={int(index.max().item()) if index.numel() > 0 else 'n/a'}"
    )


def _record_debug_index_event(op_name: str, **payload: Any) -> None:
    entry = {"op": str(op_name)}
    entry.update(payload)
    _DEBUG_INDEX_TRACE.append(entry)


def _dump_debug_index_trace() -> list[dict[str, Any]]:
    return list(_DEBUG_INDEX_TRACE)


def _debug_check_integer_index(
    *,
    op_name: str,
    data: torch.Tensor,
    dim: int,
    index: torch.Tensor,
) -> None:
    if index.dtype not in (
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        return
    if dim < 0:
        dim += int(data.dim())
    if dim < 0 or dim >= int(data.dim()) or index.numel() == 0:
        return
    size = int(data.shape[dim])
    min_idx = int(index.min().item())
    max_idx = int(index.max().item())
    if min_idx < -size or max_idx >= size:
        raise RuntimeError(
            f"{op_name} integer index out of bounds: dim={dim} size={size} "
            f"data.shape={tuple(data.shape)} index={_debug_index_tensor_summary(index)}\n"
            f"stack:\n{_debug_index_stack()}"
        )


def _debug_check_getitem_indices(
    *,
    op_name: str,
    data: torch.Tensor,
    index: Any,
) -> None:
    if not isinstance(index, tuple):
        index_items = (index,)
    else:
        index_items = index
    dim = 0
    for item in index_items:
        if item is Ellipsis:
            remaining = int(data.dim()) - sum(
                0 if sub is None or sub is Ellipsis else 1 for sub in index_items[index_items.index(item) + 1 :]
            )
            dim = remaining
            continue
        if item is None:
            continue
        if isinstance(item, slice):
            dim += 1
            continue
        if isinstance(item, int):
            size = int(data.shape[dim]) if dim < int(data.dim()) else 0
            if item < -size or item >= size:
                raise RuntimeError(
                    f"{op_name} scalar index out of bounds: dim={dim} size={size} "
                    f"index={item} data.shape={tuple(data.shape)}\nstack:\n{_debug_index_stack()}"
                )
            dim += 1
            continue
        if isinstance(item, torch.Tensor):
            if item.dtype == torch.bool:
                continue
            _debug_check_integer_index(op_name=op_name, data=data, dim=dim, index=item)
            dim += 1


def _install_debug_tensor_index_guards() -> None:
    global _DEBUG_INDEX_GUARDS_INSTALLED
    if _DEBUG_INDEX_GUARDS_INSTALLED:
        return

    original_getitem = torch.Tensor.__getitem__
    original_setitem = torch.Tensor.__setitem__
    original_index_select = torch.Tensor.index_select
    original_gather = torch.gather
    original_tensor_gather = torch.Tensor.gather
    original_scatter = torch.Tensor.scatter_
    original_embedding = torch.nn.functional.embedding
    original_torch_embedding = torch.embedding

    def guarded_getitem(self: torch.Tensor, index: Any):
        _record_debug_index_event(op_name="tensor.__getitem__", data_shape=tuple(self.shape))
        _debug_check_getitem_indices(op_name="tensor.__getitem__", data=self, index=index)
        return original_getitem(self, index)

    def guarded_setitem(self: torch.Tensor, index: Any, value: Any):
        _record_debug_index_event(op_name="tensor.__setitem__", data_shape=tuple(self.shape))
        _debug_check_getitem_indices(op_name="tensor.__setitem__", data=self, index=index)
        return original_setitem(self, index, value)

    def guarded_index_select(self: torch.Tensor, dim: int, index: torch.Tensor):
        _record_debug_index_event(
            op_name="tensor.index_select",
            data_shape=tuple(self.shape),
            dim=int(dim),
            index=_debug_index_tensor_summary(index),
        )
        _debug_check_integer_index(op_name="tensor.index_select", data=self, dim=dim, index=index)
        return original_index_select(self, dim, index)

    def guarded_gather(input_tensor: torch.Tensor, dim: int, index: torch.Tensor, *args: Any, **kwargs: Any):
        _record_debug_index_event(
            op_name="torch.gather",
            data_shape=tuple(input_tensor.shape),
            dim=int(dim),
            index=_debug_index_tensor_summary(index),
        )
        _debug_check_integer_index(op_name="torch.gather", data=input_tensor, dim=dim, index=index)
        return original_gather(input_tensor, dim, index, *args, **kwargs)

    def guarded_tensor_gather(self: torch.Tensor, dim: int, index: torch.Tensor, *args: Any, **kwargs: Any):
        _record_debug_index_event(
            op_name="tensor.gather",
            data_shape=tuple(self.shape),
            dim=int(dim),
            index=_debug_index_tensor_summary(index),
        )
        _debug_check_integer_index(op_name="tensor.gather", data=self, dim=dim, index=index)
        return original_tensor_gather(self, dim, index, *args, **kwargs)

    def guarded_scatter(self: torch.Tensor, dim: int, index: torch.Tensor, src: Any, *args: Any, **kwargs: Any):
        _record_debug_index_event(
            op_name="tensor.scatter_",
            data_shape=tuple(self.shape),
            dim=int(dim),
            index=_debug_index_tensor_summary(index),
        )
        _debug_check_integer_index(op_name="tensor.scatter_", data=self, dim=dim, index=index)
        return original_scatter(self, dim, index, src, *args, **kwargs)

    def guarded_embedding(
        input_tensor: torch.Tensor,
        weight: torch.Tensor,
        padding_idx: int | None = None,
        *args: Any,
        **kwargs: Any,
    ):
        _record_debug_index_event(
            op_name="torch.nn.functional.embedding",
            weight_shape=tuple(weight.shape),
            padding_idx=None if padding_idx is None else int(padding_idx),
            index=_debug_index_tensor_summary(input_tensor),
        )
        _debug_check_integer_index(op_name="torch.nn.functional.embedding", data=weight, dim=0, index=input_tensor)
        return original_embedding(input_tensor, weight, padding_idx, *args, **kwargs)

    def guarded_torch_embedding(weight: torch.Tensor, input_tensor: torch.Tensor, *args: Any, **kwargs: Any):
        _record_debug_index_event(
            op_name="torch.embedding",
            weight_shape=tuple(weight.shape),
            index=_debug_index_tensor_summary(input_tensor),
        )
        _debug_check_integer_index(op_name="torch.embedding", data=weight, dim=0, index=input_tensor)
        return original_torch_embedding(weight, input_tensor, *args, **kwargs)

    torch.Tensor.__getitem__ = guarded_getitem
    torch.Tensor.__setitem__ = guarded_setitem
    torch.Tensor.index_select = guarded_index_select
    torch.gather = guarded_gather
    torch.Tensor.gather = guarded_tensor_gather
    torch.Tensor.scatter_ = guarded_scatter
    torch.nn.functional.embedding = guarded_embedding
    torch.embedding = guarded_torch_embedding
    _DEBUG_INDEX_GUARDS_INSTALLED = True


def _init_logging() -> None:
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Formatter(logging.Formatter):
        def format(self, record: logging.LogRecord) -> str:
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = _Formatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            handler.setFormatter(formatter)


def _rgb_visual_override(rgb: np.ndarray, grid: int = 8) -> torch.Tensor:
    rgb_t = torch.as_tensor(np.asarray(rgb, dtype=np.float32) / 255.0, dtype=torch.float32)
    pooled = torch.nn.functional.adaptive_avg_pool2d(rgb_t.permute(2, 0, 1)[None, :], (grid, grid))[0]
    return pooled.permute(1, 2, 0).contiguous()


def _rgb_uint8(rgb: np.ndarray) -> np.ndarray:
    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array


def _apply_picf_photometric_augmentation(
    image: np.ndarray,
    *,
    rng: np.random.Generator,
    strength: str,
) -> np.ndarray:
    """Apply geometry-preserving RGB jitter for full-PICF training.

    This deliberately avoids crop/rotation/warp. PICF point/depth/camera
    geometry must stay aligned with the image evidence.
    """
    source = np.asarray(image)
    if source.ndim != 3 or source.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB image for photometric augmentation, got shape={source.shape}.")
    if str(strength).lower().replace("-", "_") == "reference":
        brightness = float(rng.uniform(0.7, 1.3))
        contrast = float(rng.uniform(0.6, 1.4))
        saturation = float(rng.uniform(0.5, 1.5))
    elif str(strength).lower().replace("-", "_") == "conservative":
        brightness = float(rng.uniform(0.85, 1.15))
        contrast = float(rng.uniform(0.8, 1.2))
        saturation = float(rng.uniform(0.75, 1.25))
    else:
        raise ValueError(f"picf_photometric_strength must be one of {{'conservative', 'reference'}}, got {strength!r}.")

    dtype = source.dtype
    if dtype == np.uint8:
        x = source.astype(np.float32) / 255.0
        output_scale = 255.0
    else:
        x = source.astype(np.float32)
        if float(np.nanmax(x)) > 1.5:
            x = x / 255.0
            output_scale = 255.0
        elif float(np.nanmin(x)) < -0.1:
            x = (x + 1.0) * 0.5
            output_scale = None
        else:
            output_scale = None

    x = np.clip(x, 0.0, 1.0)
    x = x * brightness
    mean = np.mean(x, axis=(0, 1), keepdims=True)
    x = (x - mean) * contrast + mean
    gray = np.mean(x, axis=-1, keepdims=True)
    x = gray + (x - gray) * saturation
    x = np.clip(x, 0.0, 1.0)

    if dtype == np.uint8:
        return np.clip(np.round(x * 255.0), 0.0, 255.0).astype(np.uint8)
    if output_scale == 255.0:
        return np.clip(x * 255.0, 0.0, 255.0).astype(dtype, copy=False)
    if float(np.nanmin(source.astype(np.float32))) < -0.1:
        return (x * 2.0 - 1.0).astype(dtype, copy=False)
    return x.astype(dtype, copy=False)


def _decode_visual_real_prediction(
    visual_real: torch.Tensor | None,
    *,
    grid: int,
    upscale: int,
) -> np.ndarray | None:
    if visual_real is None:
        return None
    flat = visual_real.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    expected = 3 * (int(grid) ** 2)
    if flat.numel() != expected:
        raise ValueError(f"Expected visual_real with {expected} values, got {flat.numel()}.")
    image = flat.reshape(3, int(grid), int(grid)).permute(1, 2, 0).numpy()
    image = np.clip(image, 0.0, 1.0)
    image_uint8 = np.clip(np.round(image * 255.0), 0.0, 255.0).astype(np.uint8)
    if int(upscale) > 1:
        side = int(grid) * int(upscale)
        image_uint8 = np.asarray(
            Image.fromarray(image_uint8).resize((side, side), resample=Image.Resampling.NEAREST)
        )
    return image_uint8


def _write_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgb_uint8(image)).save(path)


def _write_gif(path: Path, frames: list[np.ndarray], *, duration_ms: int = 400) -> None:
    if not frames:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pil_frames = [Image.fromarray(_rgb_uint8(frame)) for frame in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=int(duration_ms),
        loop=0,
    )


def _resize_rgb_for_concat(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Return a uint8 RGB image with the exact width/height needed for concat grids."""
    rgb = _rgb_uint8(image)
    if rgb.shape[1] == int(size[0]) and rgb.shape[0] == int(size[1]):
        return rgb
    return np.asarray(Image.fromarray(rgb).resize(size, resample=Image.Resampling.BILINEAR))


def _save_visual_diagnostics(
    *,
    output_dir: Path,
    step: int,
    window: "_TransitionWindow",
    physical_visual_real_seq: list[torch.Tensor | None],
    semantic_visual_real_seq: list[torch.Tensor | None],
    visual_real_grid: int,
    visual_real_upscale: int,
) -> None:
    diag_dir = output_dir / "diagnostics" / f"{int(step):06d}"
    if diag_dir.exists():
        shutil.rmtree(diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    gt_frames = [_rgb_uint8(frame.rgb_static) for frame in window.frames]
    physical_frames: list[np.ndarray] = [gt_frames[0]]
    semantic_frames: list[np.ndarray] = [gt_frames[0]]
    records: list[dict[str, Any]] = []

    for index, predicted in enumerate(physical_visual_real_seq, start=1):
        decoded = _decode_visual_real_prediction(
            predicted,
            grid=visual_real_grid,
            upscale=visual_real_upscale,
        )
        if decoded is not None:
            _write_png(diag_dir / f"pred_physical_t{index}.png", decoded)
            physical_frames.append(decoded)
    for index, predicted in enumerate(semantic_visual_real_seq, start=1):
        decoded = _decode_visual_real_prediction(
            predicted,
            grid=visual_real_grid,
            upscale=visual_real_upscale,
        )
        if decoded is not None:
            _write_png(diag_dir / f"pred_semantic_t{index}.png", decoded)
            semantic_frames.append(decoded)

    for index, frame in enumerate(window.frames):
        rgb = _rgb_uint8(frame.rgb_static)
        _write_png(diag_dir / f"gt_static_t{index}.png", rgb)
        records.append(
            {
                "t": int(index),
                "step_id": int(frame.step_id),
                "segment_id": int(frame.segment_id),
                "timestamp_s": float(frame.timestamp_s),
                "reset_scaffold": bool(frame.reset_scaffold),
            }
        )

    compare_rows: list[np.ndarray] = []
    for transition_index in range(len(window.frames) - 1):
        current = gt_frames[transition_index]
        target_next = gt_frames[transition_index + 1]
        physical = _decode_visual_real_prediction(
            physical_visual_real_seq[transition_index] if transition_index < len(physical_visual_real_seq) else None,
            grid=visual_real_grid,
            upscale=visual_real_upscale,
        )
        semantic = _decode_visual_real_prediction(
            semantic_visual_real_seq[transition_index] if transition_index < len(semantic_visual_real_seq) else None,
            grid=visual_real_grid,
            upscale=visual_real_upscale,
        )
        if physical is not None:
            compare_size = physical.shape[1], physical.shape[0]
        elif semantic is not None:
            compare_size = semantic.shape[1], semantic.shape[0]
        else:
            side = max(1, int(visual_real_grid) * max(1, int(visual_real_upscale)))
            compare_size = side, side
        if physical is None:
            physical = np.zeros((compare_size[1], compare_size[0], 3), dtype=np.uint8)
        if semantic is None:
            semantic = np.zeros((compare_size[1], compare_size[0], 3), dtype=np.uint8)
        current_ref = _resize_rgb_for_concat(current, compare_size)
        physical = _resize_rgb_for_concat(physical, compare_size)
        semantic = _resize_rgb_for_concat(semantic, compare_size)
        target_ref = _resize_rgb_for_concat(target_next, compare_size)
        row = np.concatenate([current_ref, physical, semantic, target_ref], axis=1)
        compare_rows.append(row)
    if compare_rows:
        _write_png(diag_dir / "compare_grid.png", np.concatenate(compare_rows, axis=0))

    _write_gif(diag_dir / "gt_window_static.gif", gt_frames)
    _write_gif(diag_dir / "pred_physical_window_static.gif", physical_frames)
    _write_gif(diag_dir / "pred_semantic_window_static.gif", semantic_frames)

    metadata = {
        "step": int(step),
        "prompt": window.prompt,
        "segment_id": int(window.segment_id),
        "start_step_id": int(window.start_step_id),
        "visual_real_grid": int(visual_real_grid),
        "visual_real_upscale": int(visual_real_upscale),
        "diagnostic_note": (
            "pred_* images are upsampled visual_real predictions from the PICF future head; "
            "they are coarse 4x4 RGB reconstructions, not full-resolution generated video frames."
        ),
        "frames": records,
    }
    (diag_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


_ANCHOR_OVERLAY_ROLE_COLORS: tuple[tuple[int, int, int], ...] = (
    (36, 137, 255),
    (255, 149, 0),
    (23, 190, 80),
    (220, 80, 255),
    (255, 64, 64),
    (48, 220, 220),
)


def _to_cpu_tensor_or_none(value: torch.Tensor | None) -> torch.Tensor | None:
    if value is None:
        return None
    return value.detach().to(device="cpu")


def _json_scalar(value: Any) -> float | int | bool | str | None:
    if value is None:
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, torch.Tensor):
        detached = value.detach().to(device="cpu")
        if detached.numel() == 1:
            return _json_scalar(detached.item())
        return None
    if isinstance(value, np.generic):
        return value.item()
    return None


def _anchor_source_snapshot(
    *,
    name: str,
    x: torch.Tensor | None,
    role_ids: torch.Tensor | None = None,
    confidence: torch.Tensor | None = None,
    active: torch.Tensor | None = None,
    downstream_weight: torch.Tensor | None = None,
    support_mass: torch.Tensor | None = None,
    recycle_gate: torch.Tensor | None = None,
    geometry_valid: torch.Tensor | None = None,
    address_update_rate: torch.Tensor | None = None,
    file_competition_demoted_mass: torch.Tensor | None = None,
    owner_transport_mass: torch.Tensor | None = None,
    owner_transport_confidence: torch.Tensor | None = None,
    owner_transport_dist_to_standard: torch.Tensor | None = None,
    owner_transport_dist_after_fusion: torch.Tensor | None = None,
    support_signature: torch.Tensor | None = None,
    binding_signature: torch.Tensor | None = None,
) -> dict[str, Any] | None:
    x_cpu = _to_cpu_tensor_or_none(x)
    if x_cpu is None or x_cpu.ndim != 2 or x_cpu.shape[-1] < 3 or x_cpu.shape[0] == 0:
        return None
    count = int(x_cpu.shape[0])

    def _aligned(value: torch.Tensor | None, *, dtype: torch.dtype | None = None) -> torch.Tensor | None:
        value_cpu = _to_cpu_tensor_or_none(value)
        if value_cpu is None or value_cpu.numel() == 0:
            return None
        value_cpu = value_cpu.reshape(-1)
        if value_cpu.shape[0] != count:
            return None
        if dtype is not None:
            value_cpu = value_cpu.to(dtype=dtype)
        return value_cpu

    def _aligned_matrix(value: torch.Tensor | None) -> torch.Tensor | None:
        value_cpu = _to_cpu_tensor_or_none(value)
        if value_cpu is None or value_cpu.numel() == 0 or value_cpu.ndim != 2:
            return None
        if value_cpu.shape[0] != count:
            return None
        return value_cpu.to(dtype=torch.float32)

    return {
        "name": str(name),
        "x": x_cpu[:, :3].to(dtype=torch.float32),
        "role_ids": _aligned(role_ids, dtype=torch.long),
        "confidence": _aligned(confidence, dtype=torch.float32),
        "active": _aligned(active, dtype=torch.float32),
        "downstream_weight": _aligned(downstream_weight, dtype=torch.float32),
        "support_mass": _aligned(support_mass, dtype=torch.float32),
        "recycle_gate": _aligned(recycle_gate, dtype=torch.float32),
        "geometry_valid": _aligned(geometry_valid, dtype=torch.bool),
        "address_update_rate": _aligned(address_update_rate, dtype=torch.float32),
        "file_competition_demoted_mass": _aligned(file_competition_demoted_mass, dtype=torch.float32),
        "owner_transport_mass": _aligned(owner_transport_mass, dtype=torch.float32),
        "owner_transport_confidence": _aligned(owner_transport_confidence, dtype=torch.float32),
        "owner_transport_dist_to_standard": _aligned(owner_transport_dist_to_standard, dtype=torch.float32),
        "owner_transport_dist_after_fusion": _aligned(owner_transport_dist_after_fusion, dtype=torch.float32),
        "support_signature": _aligned_matrix(support_signature),
        "binding_signature": _aligned_matrix(binding_signature),
    }


def _anchor_overlay_snapshot_from_output(
    output: Any,
    observation: PicfObservation,
    *,
    dump_signatures: bool = False,
) -> dict[str, Any] | None:
    state = getattr(output, "state", None)
    if state is None:
        return None
    sources: list[dict[str, Any]] = []
    graph = getattr(state, "anchor_prior_graph", None)
    if graph is not None:
        graph_source = _anchor_source_snapshot(
            name="graph",
            x=getattr(graph, "anchor_x", None),
            role_ids=getattr(graph, "anchor_roles", None),
            confidence=getattr(graph, "anchor_confidence", None),
            active=getattr(graph, "anchor_active", None),
            downstream_weight=getattr(graph, "anchor_downstream_weight", None),
            geometry_valid=getattr(graph, "geometry_valid", None),
            support_signature=getattr(graph, "support_signature", None) if dump_signatures else None,
            binding_signature=getattr(graph, "binding_signature", None) if dump_signatures else None,
        )
        if graph_source is not None:
            sources.append(graph_source)
    posterior = getattr(state, "posterior", None)
    if posterior is not None:
        posterior_active = None
        file_active = _to_cpu_tensor_or_none(getattr(posterior, "file_competition_active", None))
        if file_active is not None and file_active.ndim == 1:
            posterior_active = file_active.to(dtype=torch.float32)
        posterior_alpha = _to_cpu_tensor_or_none(getattr(posterior, "alpha", None))
        posterior_support = _to_cpu_tensor_or_none(getattr(posterior, "support_mass", None))
        posterior_recycle = _to_cpu_tensor_or_none(getattr(posterior, "recycle_gate", None))
        if (
            posterior_active is None
            and posterior_alpha is not None
            and posterior_support is not None
            and posterior_recycle is not None
            and posterior_alpha.numel() == posterior_support.numel() == posterior_recycle.numel()
        ):
            # Fallback for old checkpoints without file-competition state. New
            # runs prefer posterior.file_competition_active so the image shows
            # no-object demotions instead of treating all persistent capacity
            # as active object files.
            posterior_active = (
                (posterior_alpha.reshape(-1) >= 0.25)
                & (posterior_support.reshape(-1) >= 0.05)
                & (posterior_recycle.reshape(-1) <= 0.5)
            ).to(dtype=torch.float32)
        posterior_source = _anchor_source_snapshot(
            name="posterior",
            x=getattr(posterior, "x", None),
            role_ids=getattr(posterior, "role_ids", None),
            confidence=getattr(posterior, "alpha", None),
            active=posterior_active,
            support_mass=getattr(posterior, "support_mass", None),
            recycle_gate=getattr(posterior, "recycle_gate", None),
            address_update_rate=getattr(posterior, "address_update_rate", None),
            file_competition_demoted_mass=getattr(posterior, "file_competition_demoted_mass", None),
            owner_transport_mass=getattr(posterior, "owner_transport_mass", None),
            owner_transport_confidence=getattr(posterior, "owner_transport_confidence", None),
            owner_transport_dist_to_standard=getattr(posterior, "owner_transport_dist_to_standard", None),
            owner_transport_dist_after_fusion=getattr(posterior, "owner_transport_dist_after_fusion", None),
            support_signature=getattr(posterior, "support_signature", None) if dump_signatures else None,
            binding_signature=getattr(posterior, "binding_signature", None) if dump_signatures else None,
        )
        if posterior_source is not None:
            sources.append(posterior_source)
    task_readout = getattr(state, "task_readout", None)
    if task_readout is not None:
        task_x = getattr(task_readout, "x", None)
        task_roles = getattr(task_readout, "local_role_ids", None)
        task_valid = getattr(task_readout, "geometry_valid", None)
        task_active = None
        if isinstance(task_valid, torch.Tensor):
            task_active = task_valid.to(dtype=torch.float32)
        task_source = _anchor_source_snapshot(
            name="task",
            x=task_x,
            role_ids=task_roles,
            confidence=task_active,
            active=task_active,
            geometry_valid=task_valid,
        )
        if task_source is not None:
            sources.append(task_source)
    if not sources:
        return None
    debug = {}
    debug_map = getattr(output, "debug", {}) or {}
    for key in (
        "aqr_same_role_support_overlap_max",
        "aqr_active_same_role_support_overlap_max",
        "aqr_same_role_object_core_overlap_max",
        "aqr_active_same_role_object_core_overlap_max",
        "aqr_effective_anchor_count",
        "aqr_active_anchor_count",
        "aqr_context_anchor_count",
        "aqr_context_downstream_weight_mean",
        "aqr_reserve_anchor_fraction",
        "posterior_recycle_rate",
        "posterior_identity_switch_rate",
        "posterior_file_competition_duplicate_overlap_max",
        "posterior_file_competition_active_count",
        "posterior_file_competition_birth_count",
        "posterior_support_mass_final_mean",
        "posterior_owner_transport_mass_max",
        "posterior_owner_transport_confidence_max",
        "posterior_owner_transport_applied_fraction",
        "owm_proposal_tokens",
        "aqr_temporal_view_mass_0",
        "aqr_temporal_view_mass_1",
    ):
        value = _json_scalar(debug_map.get(key))
        if value is not None:
            debug[key] = value
    proposals: dict[str, torch.Tensor] = {}
    token_field = getattr(state, "token_field", None)
    proposal = getattr(token_field, "proposal", None) if token_field is not None else None
    if proposal is not None:
        for key, attr in (
            ("centers_xy", "centers_xy"),
            ("boxes_xyxy", "boxes_xyxy"),
            ("objectness", "objectness"),
            ("view_ids", "view_ids"),
            ("source_ids", "source_ids"),
            ("age", "age"),
            ("mask_xy", "mask_xy"),
            ("mask_weights", "mask_weights"),
            ("mask_offsets", "mask_offsets"),
        ):
            value = getattr(proposal, attr, None)
            value_cpu = _to_cpu_tensor_or_none(value)
            if value_cpu is not None:
                proposals[key] = value_cpu
    return {
        "image": _rgb_uint8(observation.rgb_static),
        "segment_id": int(observation.segment_id),
        "step_id": int(observation.step_id),
        "timestamp_s": float(observation.timestamp_s),
        "prompt": str(observation.prompt or ""),
        "sources": sources,
        "debug": debug,
        "proposals": proposals,
    }


def _camera_model_to_static_projection(core: PicfFullCore, points: torch.Tensor, image_shape: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    height = int(image_shape[0])
    width = int(image_shape[1])
    points_np = np.asarray(points.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32)
    if points_np.ndim != 2 or points_np.shape[-1] < 3 or points_np.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=bool)
    camera = getattr(core, "camera_model", None)
    if camera is None:
        return np.zeros((points_np.shape[0], 2), dtype=np.float32), np.zeros((points_np.shape[0],), dtype=bool)
    C_T_W = getattr(camera, "C_T_W", None)
    if C_T_W is None:
        W_T_C = getattr(camera, "W_T_C", None)
        if W_T_C is None:
            return np.zeros((points_np.shape[0], 2), dtype=np.float32), np.zeros((points_np.shape[0],), dtype=bool)
        C_T_W = np.linalg.inv(np.asarray(W_T_C, dtype=np.float32))
    C_T_W_np = np.asarray(C_T_W, dtype=np.float32)
    if C_T_W_np.shape != (4, 4):
        return np.zeros((points_np.shape[0], 2), dtype=np.float32), np.zeros((points_np.shape[0],), dtype=bool)

    fx = getattr(camera, "fx", None)
    fy = getattr(camera, "fy", None)
    cx = getattr(camera, "cx", None)
    cy = getattr(camera, "cy", None)
    if fx is None or fy is None or cx is None or cy is None:
        K = getattr(camera, "K", None)
        if K is None:
            return np.zeros((points_np.shape[0], 2), dtype=np.float32), np.zeros((points_np.shape[0],), dtype=bool)
        K_np = np.asarray(K, dtype=np.float32)
        fx, fy, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]

    homo = np.concatenate([points_np[:, :3], np.ones((points_np.shape[0], 1), dtype=np.float32)], axis=-1)
    points_cam = (C_T_W_np @ homo.T).T[:, :3]
    z = points_cam[:, 2]
    denom = np.maximum(z, 1e-6)
    uv = np.zeros((points_np.shape[0], 2), dtype=np.float32)
    uv[:, 0] = (float(fx) * points_cam[:, 0] / denom) + float(cx)
    uv[:, 1] = (float(fy) * points_cam[:, 1] / denom) + float(cy)
    visible = (
        np.isfinite(uv[:, 0])
        & np.isfinite(uv[:, 1])
        & (z > 0.0)
        & (uv[:, 0] >= 0.0)
        & (uv[:, 0] <= float(width - 1))
        & (uv[:, 1] >= 0.0)
        & (uv[:, 1] <= float(height - 1))
    )
    return uv, visible


def _save_anchor_overlay_diagnostic(
    *,
    output_dir: Path,
    step: int,
    snapshot: dict[str, Any] | None,
    core: PicfFullCore,
    max_anchors: int,
) -> None:
    if snapshot is None:
        return
    overlay_dir = output_dir / "anchor_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    image = np.asarray(snapshot["image"], dtype=np.uint8)
    records: list[dict[str, Any]] = []
    draw_items: list[dict[str, Any]] = []
    radius_by_source = {"graph": 4, "posterior": 6, "task": 5}

    proposal_records: list[dict[str, Any]] = []
    proposal_draw_items: list[dict[str, Any]] = []
    proposals = snapshot.get("proposals", {})
    if isinstance(proposals, dict):
        centers = proposals.get("centers_xy")
        boxes = proposals.get("boxes_xyxy")
        objectness = proposals.get("objectness")
        view_ids = proposals.get("view_ids")
        source_ids = proposals.get("source_ids")
        age = proposals.get("age")
        mask_xy = proposals.get("mask_xy")
        mask_weights = proposals.get("mask_weights")
        mask_offsets = proposals.get("mask_offsets")
        if isinstance(centers, torch.Tensor) and isinstance(boxes, torch.Tensor):
            centers_np = np.asarray(centers.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32)
            boxes_np = np.asarray(boxes.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32)
            objectness_np = (
                np.asarray(objectness.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32).reshape(-1)
                if isinstance(objectness, torch.Tensor)
                else np.zeros((centers_np.shape[0],), dtype=np.float32)
            )
            view_ids_np = (
                np.asarray(view_ids.detach().to(device="cpu", dtype=torch.long), dtype=np.int64).reshape(-1)
                if isinstance(view_ids, torch.Tensor)
                else np.zeros((centers_np.shape[0],), dtype=np.int64)
            )
            source_ids_np = (
                np.asarray(source_ids.detach().to(device="cpu", dtype=torch.long), dtype=np.int64).reshape(-1)
                if isinstance(source_ids, torch.Tensor)
                else np.zeros((centers_np.shape[0],), dtype=np.int64)
            )
            age_np = (
                np.asarray(age.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32).reshape(-1)
                if isinstance(age, torch.Tensor)
                else np.zeros((centers_np.shape[0],), dtype=np.float32)
            )
            mask_xy_np = (
                np.asarray(mask_xy.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32).reshape(-1, 2)
                if isinstance(mask_xy, torch.Tensor) and mask_xy.numel() > 0
                else np.zeros((0, 2), dtype=np.float32)
            )
            mask_weights_np = (
                np.asarray(mask_weights.detach().to(device="cpu", dtype=torch.float32), dtype=np.float32).reshape(-1)
                if isinstance(mask_weights, torch.Tensor) and mask_weights.numel() > 0
                else np.zeros((0,), dtype=np.float32)
            )
            mask_offsets_np = (
                np.asarray(mask_offsets.detach().to(device="cpu", dtype=torch.long), dtype=np.int64).reshape(-1)
                if isinstance(mask_offsets, torch.Tensor) and mask_offsets.numel() > 0
                else np.zeros((0,), dtype=np.int64)
            )
            h, w = int(image.shape[0]), int(image.shape[1])
            count = min(int(centers_np.shape[0]), int(boxes_np.shape[0]))
            for idx in range(count):
                # Overlay only static-view proposals on the static RGB image. Wrist
                # proposals stay in JSON because drawing them in static pixels would
                # imply an extrinsic calibration we do not have.
                view_id = int(view_ids_np[idx]) if idx < view_ids_np.shape[0] else 0
                score = float(objectness_np[idx]) if idx < objectness_np.shape[0] else 0.0
                box = np.clip(boxes_np[idx], 0.0, 1.0)
                center = np.clip(centers_np[idx], 0.0, 1.0)
                rec = {
                    "index": int(idx),
                    "center_xy_norm": [float(center[0]), float(center[1])],
                    "box_xyxy_norm": [float(v) for v in box.tolist()],
                    "objectness": score,
                    "age": float(age_np[idx]) if idx < age_np.shape[0] else 0.0,
                    "view_id": view_id,
                    "source_id": int(source_ids_np[idx]) if idx < source_ids_np.shape[0] else -1,
                }
                mask_points: list[tuple[int, int, float]] = []
                if mask_offsets_np.shape[0] >= count + 1 and mask_xy_np.shape[0] == mask_weights_np.shape[0]:
                    start = int(mask_offsets_np[idx])
                    end = int(mask_offsets_np[idx + 1])
                    start = max(0, min(start, int(mask_xy_np.shape[0])))
                    end = max(start, min(end, int(mask_xy_np.shape[0])))
                    if end > start:
                        rec["mask_sample_count"] = int(end - start)
                        for xy, weight in zip(mask_xy_np[start:end], mask_weights_np[start:end], strict=False):
                            x_pix = int(round(float(np.clip(xy[0], 0.0, 1.0)) * max(w - 1, 1)))
                            y_pix = int(round(float(np.clip(xy[1], 0.0, 1.0)) * max(h - 1, 1)))
                            mask_points.append((x_pix, y_pix, float(np.clip(weight, 0.0, 1.0))))
                    else:
                        rec["mask_sample_count"] = 0
                else:
                    rec["mask_sample_count"] = 0
                proposal_records.append(rec)
                if view_id != 0:
                    continue
                x0, y0, x1, y1 = box.tolist()
                proposal_draw_items.append(
                    {
                        "index": int(idx),
                        "box": (
                            int(round(x0 * max(w - 1, 1))),
                            int(round(y0 * max(h - 1, 1))),
                            int(round(x1 * max(w - 1, 1))),
                            int(round(y1 * max(h - 1, 1))),
                        ),
                        "center": (
                            int(round(float(center[0]) * max(w - 1, 1))),
                            int(round(float(center[1]) * max(h - 1, 1))),
                        ),
                        "objectness": score,
                        "age": float(age_np[idx]) if idx < age_np.shape[0] else 0.0,
                        "mask_points": mask_points,
                    }
                )

    for source_index, source in enumerate(snapshot.get("sources", [])):
        name = str(source.get("name", f"source{source_index}"))
        x = source.get("x")
        if not isinstance(x, torch.Tensor):
            continue
        uv, visible = _camera_model_to_static_projection(core, x, image.shape)
        count = min(int(x.shape[0]), max(int(max_anchors), 0))
        roles = source.get("role_ids")
        confidence = source.get("confidence")
        active = source.get("active")
        support_mass = source.get("support_mass")
        recycle_gate = source.get("recycle_gate")
        geometry_valid = source.get("geometry_valid")
        address_update_rate = source.get("address_update_rate")
        file_competition_demoted_mass = source.get("file_competition_demoted_mass")
        owner_transport_mass = source.get("owner_transport_mass")
        owner_transport_confidence = source.get("owner_transport_confidence")
        owner_transport_dist_to_standard = source.get("owner_transport_dist_to_standard")
        support_signature = source.get("support_signature")
        binding_signature = source.get("binding_signature")
        downstream_weight = source.get("downstream_weight")
        for idx in range(count):
            role = int(roles[idx].item()) if isinstance(roles, torch.Tensor) else -1
            color = _ANCHOR_OVERLAY_ROLE_COLORS[role % len(_ANCHOR_OVERLAY_ROLE_COLORS)] if role >= 0 else (245, 245, 245)
            px = float(uv[idx, 0]) if idx < uv.shape[0] else float("nan")
            py = float(uv[idx, 1]) if idx < uv.shape[0] else float("nan")
            is_visible = bool(visible[idx]) if idx < visible.shape[0] else False
            rec = {
                "source": name,
                "index": int(idx),
                "role": int(role),
                "world_xyz": [float(v) for v in x[idx, :3].tolist()],
                "pixel_xy": [px, py] if np.isfinite(px) and np.isfinite(py) else None,
                "visible": is_visible,
                "confidence": float(confidence[idx].item()) if isinstance(confidence, torch.Tensor) else None,
                "active": float(active[idx].item()) if isinstance(active, torch.Tensor) else None,
                "downstream_weight": float(downstream_weight[idx].item()) if isinstance(downstream_weight, torch.Tensor) else None,
                "support_mass": float(support_mass[idx].item()) if isinstance(support_mass, torch.Tensor) else None,
                "recycle_gate": float(recycle_gate[idx].item()) if isinstance(recycle_gate, torch.Tensor) else None,
                "geometry_valid": bool(geometry_valid[idx].item()) if isinstance(geometry_valid, torch.Tensor) else None,
                "address_update_rate": (
                    float(address_update_rate[idx].item()) if isinstance(address_update_rate, torch.Tensor) else None
                ),
                "file_competition_demoted_mass": (
                    float(file_competition_demoted_mass[idx].item())
                    if isinstance(file_competition_demoted_mass, torch.Tensor)
                    else None
                ),
                "owner_transport_mass": (
                    float(owner_transport_mass[idx].item()) if isinstance(owner_transport_mass, torch.Tensor) else None
                ),
                "owner_transport_confidence": (
                    float(owner_transport_confidence[idx].item())
                    if isinstance(owner_transport_confidence, torch.Tensor)
                    else None
                ),
                "owner_transport_dist_to_standard": (
                    float(owner_transport_dist_to_standard[idx].item())
                    if isinstance(owner_transport_dist_to_standard, torch.Tensor)
                    else None
                ),
            }
            if isinstance(support_signature, torch.Tensor):
                rec["support_signature"] = [float(v) for v in support_signature[idx].tolist()]
            if isinstance(binding_signature, torch.Tensor):
                rec["binding_signature"] = [float(v) for v in binding_signature[idx].tolist()]
            records.append(rec)
            if not is_visible:
                continue
            active_value = float(active[idx].item()) if isinstance(active, torch.Tensor) else 1.0
            is_active = active_value > 0.5
            downstream_value = float(downstream_weight[idx].item()) if isinstance(downstream_weight, torch.Tensor) else active_value
            draw_items.append(
                {
                    "source": name,
                    "index": int(idx),
                    "role": int(role),
                    "color": tuple(int(v) for v in color),
                    "pixel_xy": (int(round(px)), int(round(py))),
                    "active": bool(is_active),
                    "downstream_weight": downstream_value,
                    "radius": int(radius_by_source.get(name, 5)),
                }
            )

    prompt = str(snapshot.get("prompt", "")).strip().lower()
    safe_prompt = "".join(ch if ch.isalnum() else "_" for ch in prompt)
    safe_prompt = "_".join(part for part in safe_prompt.split("_") if part)[:80]
    suffix = f"__{safe_prompt}" if safe_prompt else ""
    image_variants: dict[str, str] = {}

    def _draw_proposals(draw: ImageDraw.ImageDraw, *, draw_boxes: bool, draw_masks: bool) -> None:
        if not proposal_draw_items:
            return
        mask_palette = (
            (40, 220, 90),
            (255, 190, 40),
            (80, 170, 255),
            (235, 80, 255),
            (255, 90, 90),
            (90, 230, 230),
        )
        for prop in proposal_draw_items:
            color = mask_palette[int(prop["index"]) % len(mask_palette)]
            if draw_masks:
                for x_pix, y_pix, weight in prop.get("mask_points", []):
                    radius = max(1, int(round(1.5 + (2.5 * float(weight)))))
                    fill = tuple(int((0.35 + 0.65 * float(weight)) * channel) for channel in color)
                    draw.ellipse((x_pix - radius, y_pix - radius, x_pix + radius, y_pix + radius), fill=fill)
            if not draw_boxes:
                continue
            x0, y0, x1, y1 = prop["box"]
            score = float(prop["objectness"])
            box_color = color if score >= 0.85 else tuple(max(0, int(channel * 0.75)) for channel in color)
            draw.rectangle((x0, y0, x1, y1), outline=box_color, width=1)
            cx, cy = prop["center"]
            draw.ellipse((cx - 2, cy - 2, cx + 2, cy + 2), outline=box_color, width=1)
            age = float(prop.get("age", 0.0))
            age_tag = f"/a{age:.0f}" if age > 0.0 else ""
            draw.text((x0 + 1, max(y0 - 10, 0)), f"s{int(prop['index'])}:{score:.2f}{age_tag}", fill=box_color)

    def _render_overlay_variant(
        *,
        include_inactive: bool,
        variant_name: str,
        variant_note: str,
        draw_anchors: bool = True,
        draw_boxes: bool = True,
        draw_masks: bool = False,
        dim_background: bool = False,
    ) -> None:
        base = image.copy()
        if dim_background:
            base = np.asarray(base, dtype=np.float32)
            base = np.clip(base * 0.35, 0.0, 255.0).astype(np.uint8)
        pil = Image.fromarray(base)
        draw = ImageDraw.Draw(pil)
        _draw_proposals(draw, draw_boxes=draw_boxes, draw_masks=draw_masks)
        total_drawn = 0
        if draw_anchors:
            for item in draw_items:
                is_active = bool(item["active"])
                if not is_active and not include_inactive:
                    continue
                x0, y0 = item["pixel_xy"]
                radius = int(item["radius"])
                name = str(item["source"])
                outline = item["color"] if is_active else (130, 130, 130)
                if name == "posterior":
                    draw.ellipse((x0 - radius, y0 - radius, x0 + radius, y0 + radius), outline=outline, width=2)
                    draw.line((x0 - radius, y0, x0 + radius, y0), fill=outline, width=1)
                    draw.line((x0, y0 - radius, x0, y0 + radius), fill=outline, width=1)
                elif name == "task":
                    draw.polygon(
                        ((x0, y0 - radius), (x0 + radius, y0), (x0, y0 + radius), (x0 - radius, y0)),
                        outline=outline,
                    )
                else:
                    width = 2 if is_active else 1
                    draw.rectangle((x0 - radius, y0 - radius, x0 + radius, y0 + radius), outline=outline, width=width)
                if name == "posterior":
                    prefix = "p"
                elif name == "task":
                    prefix = "t"
                else:
                    prefix = "g"
                label = prefix + str(int(item["index"]))
                if not is_active:
                    label += "i"
                if name == "graph" and float(item.get("downstream_weight", 0.0)) > 0.0 and not is_active:
                    label += "c"
                draw.text((x0 + radius + 2, y0 - radius - 2), label, fill=outline)
                total_drawn += 1
        header = (
            f"step {int(step)} | {variant_name} | visible anchors {total_drawn} | "
            "graph square, posterior circle"
        )
        draw.rectangle((0, 0, min(pil.width, max(420, 8 * len(header))), 18), fill=(0, 0, 0))
        draw.text((4, 3), header, fill=(255, 255, 255))
        image_path = overlay_dir / f"step_{int(step):06d}{suffix}__{variant_name}.png"
        pil.save(image_path)
        image_variants[variant_name] = str(image_path.name)
        note_path = overlay_dir / f"step_{int(step):06d}{suffix}__{variant_name}.txt"
        note_path.write_text(variant_note + "\n", encoding="utf-8")

    _render_overlay_variant(
        include_inactive=True,
        variant_name="with_gray",
        variant_note=(
            "Includes context and reserve/no-object files in gray. Use this view to audit fixed posterior capacity, "
            "duplicate demotion, and whether reserve files are accumulating near an object."
        ),
    )
    _render_overlay_variant(
        include_inactive=False,
        variant_name="active_only",
        variant_note=(
            "Hides context/reserve no-object files. Use this view to judge full-weight active object binding and "
            "posterior files without gray reserve clutter."
        ),
    )
    if proposal_draw_items:
        _render_overlay_variant(
            include_inactive=False,
            variant_name="sidecar_proposals",
            variant_note=(
                "Draws static-view sidecar proposal boxes and active object files. Wrist proposals are recorded in JSON "
                "but not projected into static pixels without an explicit wrist-to-static calibration."
            ),
        )
        _render_overlay_variant(
            include_inactive=False,
            variant_name="mask_only",
            variant_note=(
                "Dims the RGB image and draws only static-view sidecar mask samples, colored by proposal id and "
                "weighted by sidecar support intensity. Use this to judge whether the sidecar mask itself is clean."
            ),
            draw_anchors=False,
            draw_boxes=True,
            draw_masks=True,
            dim_background=True,
        )
        _render_overlay_variant(
            include_inactive=False,
            variant_name="mask_active",
            variant_note=(
                "Sidecar mask samples plus full-weight active graph/posterior/task anchors. Use this to judge "
                "whether active anchors bind to the sidecar object support."
            ),
            draw_boxes=True,
            draw_masks=True,
            dim_background=True,
        )
        _render_overlay_variant(
            include_inactive=True,
            variant_name="mask_with_gray",
            variant_note=(
                "Sidecar mask samples plus all visible anchors, including gray reserve/context rows. Use this to "
                "diagnose whether reserve files compete with object files."
            ),
            draw_boxes=True,
            draw_masks=True,
            dim_background=True,
        )

    metadata_path = overlay_dir / f"step_{int(step):06d}{suffix}.json"
    metadata = {
        "step": int(step),
        "segment_id": int(snapshot.get("segment_id", -1)),
        "step_id": int(snapshot.get("step_id", -1)),
        "timestamp_s": float(snapshot.get("timestamp_s", 0.0)),
        "prompt": str(snapshot.get("prompt", "")),
        "image_shape": [int(v) for v in image.shape],
        "max_anchors_per_source": int(max_anchors),
        "image_variants": image_variants,
        "debug": dict(snapshot.get("debug", {})),
        "anchors": records,
        "proposals": proposal_records,
        "note": (
            "Static-view projection of graph and posterior 3D anchors. "
            "Invisible anchors are retained in JSON but not drawn. with_gray includes context/reserve graph or posterior "
            "anchors in gray and labels them with an 'i' suffix; active_only hides those non-full-weight files. "
            "Sidecar proposal boxes are frozen offline proposal evidence; they do not overwrite posterior identity. "
            "This is a training diagnostic only and does not change the loss or forward path."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _default_vjepa_checkpoint(model_name: str) -> str | None:
    filename_by_model = {
        "vjepa2_1_vit_base_384": "vjepa2_1_vitb_dist_vitG_384.pt",
        "vjepa2_1_vit_large_384": "vjepa2_1_vitl_dist_vitG_384.pt",
        "vjepa2_1_vit_giant_384": "vjepa2_1_vitg_384.pt",
        "vjepa2_1_vit_gigantic_384": "vjepa2_1_vitG_384.pt",
    }
    filename = filename_by_model.get(str(model_name))
    if filename is None:
        return None
    candidate = Path("checkpoints") / "foundation" / "vjepa2_1" / str(model_name) / filename
    return str(candidate) if candidate.is_file() else None


def _default_anytouch_checkpoint() -> str | None:
    candidate = Path("checkpoints") / "foundation" / "anytouch2" / "checkpoint-4frames.pth"
    return str(candidate) if candidate.is_file() else None


def _default_tactile_backgrounds_path() -> str | None:
    candidates = (
        Path("assets") / "calvin" / "tactile_backgrounds.npz",
        Path("assets") / "calvin" / "tactile_backgrounds.npy.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _default_tactile_calibration_path() -> str | None:
    candidates = (
        Path("assets") / "calvin" / "tactile_fingertip_calibration.json",
        Path("assets") / "calvin" / "tcp_tactile_calibration.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    return None


def _default_tactile_contact_stats_path() -> str | None:
    candidate = Path("assets") / "calvin" / "tactile_contact_stats.json"
    return str(candidate) if candidate.is_file() else None


def _default_action_norm_stats_path() -> str | None:
    candidate = default_calvin_action_norm_stats_path()
    return str(candidate) if candidate.is_file() else None


def _default_sonata_checkpoint() -> str | None:
    candidate = Path("src") / "pretrain" / "SpatialLM_Sonata_encoder.pth"
    return str(candidate) if candidate.is_file() else None


def _default_paligemma_model_name() -> str:
    return "google/paligemma2-3b-pt-224"


def _default_paligemma_checkpoint() -> str | None:
    candidates = (
        Path("checkpoints") / "foundation" / "pi05_base_pytorch",
        Path("checkpoints") / "pi05_base_pytorch",
        Path("/mnt/checkpoints/pi05_base_pytorch"),
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "model.safetensors").is_file():
            return str(candidate)
        if candidate.is_file() and candidate.suffix == ".safetensors":
            return str(candidate)
    return None


def _default_paligemma_config_json(checkpoint_path: str | None) -> str | None:
    if checkpoint_path is None:
        return None
    candidate = Path(checkpoint_path).expanduser()
    if candidate.is_dir():
        config_json = candidate / "config.json"
        return str(config_json) if config_json.is_file() else None
    if candidate.is_file():
        config_json = candidate.parent / "config.json"
        return str(config_json) if config_json.is_file() else None
    return None


def _parse_tactile_sensor_names(raw: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(raw, (list, tuple)):
        names = tuple(str(part).strip() for part in raw if str(part).strip())
    else:
        parsed = None
        text = str(raw).strip()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            names = tuple(str(part).strip() for part in parsed if str(part).strip())
        else:
            names = tuple(part.strip() for part in text.split(",") if part.strip())
    if not names:
        raise ValueError("Expected at least one tactile sensor name.")
    return names


def _parse_int_tuple(raw: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(raw, (list, tuple)):
        values = tuple(int(part) for part in raw)
    else:
        parsed = None
        text = str(raw).strip()
        if not text:
            return tuple()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
        if isinstance(parsed, (list, tuple)):
            values = tuple(int(part) for part in parsed)
        else:
            values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    return values


def _parse_tactile_sensor_offsets(
    raw: str | tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]]
) -> tuple[tuple[float, float, float], ...]:
    parsed = raw if isinstance(raw, (list, tuple)) else None
    if parsed is None:
        text = str(raw).strip()
        if text.startswith(("(", "[")):
            try:
                parsed = ast.literal_eval(text)
            except (SyntaxError, ValueError):
                parsed = None
    offsets: list[tuple[float, float, float]] = []
    if isinstance(parsed, (list, tuple)):
        for item in parsed:
            if not isinstance(item, (list, tuple)) or len(item) != 3:
                raise ValueError(
                    "Each tactile sensor offset must have exactly three values; "
                    f"got {item!r}."
                )
            offsets.append((float(item[0]), float(item[1]), float(item[2])))
    else:
        for block in str(raw).split(";"):
            block = block.strip()
            if not block:
                continue
            values = [float(piece.strip()) for piece in block.split(",") if piece.strip()]
            if len(values) != 3:
                raise ValueError(
                    "Each tactile sensor offset must have exactly three comma-separated floats; "
                    f"got {block!r}."
                )
            offsets.append((values[0], values[1], values[2]))
    if not offsets:
        raise ValueError("Expected at least one tactile sensor offset triplet.")
    return tuple(offsets)


def _load_tactile_backgrounds_npz(path: str | None) -> dict[str, np.ndarray] | None:
    if path is None:
        return None
    payload = np.load(Path(path).expanduser(), allow_pickle=False)
    return {str(key): np.asarray(payload[key]) for key in payload.files}


def _load_tactile_contact_stats_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return dict(payload)


def _load_tactile_calibration_json(path: str | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return dict(payload)


def _apply_foundation_profile(args: argparse.Namespace) -> None:
    if not bool(args.use_foundation_backbones):
        return
    args.point_backbone = "sonata"
    args.visual_mode = "encoder"
    args.tactile_mode = "encoder"
    args.semantic_mode = "paligemma"
    args.semantic_source = "auto"
    args.use_tactile = True
    args.visual_trainable = True
    args.tactile_trainable = True
    args.point_backbone_trainable = True
    args.semantic_trainable = True
    args.diagnostic_interval = 0


def _looks_like_legacy_blind_sam_sidecar(path: str | Path | None) -> bool:
    """Return true for archived blind-SAM sidecar roots.

    Generic proposal_* sidecars remain part of the MVTrack contract.  The
    rejected path is specifically the class-agnostic automatic SAM producer,
    whose outputs repeatedly included wall, robot, and drawer fragments.
    """

    if path is None:
        return False
    text = str(path).lower()
    legacy_needles = (
        "sam_proposal",
        "sam-proposal",
        "sam_proposals",
        "sam-proposals",
        "blind_sam",
        "blind-sam",
        "segment-anything",
    )
    return any(needle in text for needle in legacy_needles)


_LEGACY_SPEC_DEFAULT_OVERRIDES = {
    # These fields were added after the 4-22 ablation artifacts.  Some remote
    # probe worktrees can have a newer trainer script paired with an older
    # PicfCoreConfig, so old-args replay must not assume the config object has
    # every newer attribute.
    "posterior_binding_signature_memory_enabled": True,
    "posterior_binding_signature_update_rate": 0.20,
    "posterior_binding_signature_update_max_rate": 0.50,
    "posterior_binding_signature_min_support": 0.02,
    "posterior_binding_signature_owner_weight": 0.50,
    "posterior_binding_signature_dispersion_gate_enabled": True,
    "posterior_binding_signature_measurement_min_std": 0.05,
    "posterior_binding_signature_measurement_margin_min": 0.25,
    "posterior_binding_signature_measurement_margin_temperature": 0.10,
}


def _spec_default(name: str, fallback: Any | None = None) -> Any:
    if hasattr(_SPEC_DEFAULTS, name):
        return getattr(_SPEC_DEFAULTS, name)
    if name in _LEGACY_SPEC_DEFAULT_OVERRIDES:
        return _LEGACY_SPEC_DEFAULT_OVERRIDES[name]
    return fallback


def _normalize_train_args(args: argparse.Namespace) -> None:
    args.training_strategy = str(getattr(args, "training_strategy", "ddp")).lower().replace("-", "_")
    args.picf_mode = str(getattr(args, "picf_mode", "enabled")).lower().replace("-", "_")
    args.picf_trainable_scope = str(getattr(args, "picf_trainable_scope", "all")).lower().replace("-", "_")
    if args.picf_trainable_scope not in {"all", "anchor_only", "policy_only"}:
        raise ValueError(
            "picf_trainable_scope must be one of {'all', 'anchor_only', 'policy_only'}, "
            f"got {getattr(args, 'picf_trainable_scope', None)!r}."
        )
    args.burnin_mode = str(getattr(args, "burnin_mode", "full")).lower().replace("-", "_")
    args.burnin_steps = int(getattr(args, "burnin_steps", 0) or 0)
    args.step_indexed_window_rng = bool(getattr(args, "step_indexed_window_rng", True))
    args.effector_persistent_anchors = int(
        getattr(args, "effector_persistent_anchors", _SPEC_DEFAULTS.effector_persistent_anchors)
    )
    args.effector_observation_anchors = int(
        getattr(args, "effector_observation_anchors", _SPEC_DEFAULTS.effector_observation_anchors)
    )
    args.task_effector_queries = int(getattr(args, "task_effector_queries", _SPEC_DEFAULTS.task_effector_queries))
    args.global_scene_point_cap = int(getattr(args, "global_scene_point_cap", _SPEC_DEFAULTS.global_scene_point_cap))
    args.scene_anchor_border_patches = float(
        getattr(args, "scene_anchor_border_patches", _SPEC_DEFAULTS.scene_anchor_border_patches)
    )
    if getattr(args, "visual_feature_mode", None) is None:
        args.visual_feature_mode = "auto"
    if getattr(args, "vjepa_feature_cache_root", None) is None:
        args.vjepa_feature_cache_root = None
    if getattr(args, "vjepa_feature_cache_mode", None) is None:
        args.vjepa_feature_cache_mode = "off"
    if getattr(args, "vjepa_feature_cache_temporal_slices", None) is None:
        args.vjepa_feature_cache_temporal_slices = 4
    else:
        args.vjepa_feature_cache_temporal_slices = int(args.vjepa_feature_cache_temporal_slices)
    if getattr(args, "vjepa_feature_cache_storage_dtype", None) is None:
        args.vjepa_feature_cache_storage_dtype = "bfloat16"
    args.visual_real_grid = int(getattr(args, "visual_real_grid", _SPEC_DEFAULTS.visual_real_grid))
    if int(getattr(args, "diagnostic_visual_upscale", 64)) == 64 and int(args.visual_real_grid) >= 32:
        # The historical 4x4 visual-real target used 64x upscaling to make
        # 256x256 diagnostic videos.  The live 64x64 target should keep the
        # same approximate display size instead of writing 4096x4096 frames.
        args.diagnostic_visual_upscale = max(1, 256 // int(args.visual_real_grid))
    args.effective_unroll_steps = int(getattr(args, "unroll_steps", 1)) + int(args.burnin_steps)
    perception_finetune_mode = str(getattr(args, "perception_finetune_mode", "auto")).lower().replace("-", "_")
    if perception_finetune_mode not in {"auto", "full", "frozen"}:
        raise ValueError(
            "perception_finetune_mode must be one of {'auto', 'full', 'frozen'}, "
            f"got {getattr(args, 'perception_finetune_mode', None)!r}."
        )
    args.perception_finetune_mode = perception_finetune_mode
    augmentation_mode = str(getattr(args, "picf_augmentation_mode", "off")).lower().replace("-", "_")
    if augmentation_mode not in {"off", "photometric", "multimodal_geometry"}:
        raise ValueError(
            "picf_augmentation_mode must be one of {'off', 'photometric', 'multimodal_geometry'}, "
            f"got {getattr(args, 'picf_augmentation_mode', None)!r}."
        )
    args.picf_augmentation_mode = augmentation_mode
    args.picf_photometric_strength = str(
        getattr(args, "picf_photometric_strength", "conservative")
    ).lower().replace("-", "_")
    if getattr(args, "grad_clip_mode", None) is None:
        args.grad_clip_mode = "percentile"
    else:
        args.grad_clip_mode = str(args.grad_clip_mode).lower()
    args.optimizer_sharding = str(getattr(args, "optimizer_sharding", "none")).lower()
    optimizer_checkpoint_mode = str(getattr(args, "optimizer_checkpoint_mode", "auto")).lower().replace("-", "_")
    args.optimizer_checkpoint_mode = optimizer_checkpoint_mode
    if perception_finetune_mode == "frozen":
        args.point_backbone_trainable = False
        args.tactile_trainable = False
        args.visual_trainable = False
        args.visual_finetune_mode = "frozen"
    elif perception_finetune_mode == "full":
        args.point_backbone_trainable = True
        args.tactile_trainable = True
        args.visual_trainable = True
        args.visual_finetune_mode = "full"
    visual_finetune_mode = str(getattr(args, "visual_finetune_mode", "auto")).lower().replace("-", "_")
    if visual_finetune_mode == "auto":
        visual_finetune_mode = "full" if bool(getattr(args, "visual_trainable", False)) else "frozen"
    if visual_finetune_mode not in {"full", "frozen"}:
        raise ValueError(
            "visual_finetune_mode must be one of {'auto', 'full', 'frozen'} before normalization, "
            f"got {getattr(args, 'visual_finetune_mode', None)!r}."
        )
    args.visual_finetune_mode = visual_finetune_mode
    args.visual_trainable = bool(visual_finetune_mode == "full")
    if args.picf_trainable_scope == "anchor_only":
        args.perception_finetune_mode = "frozen"
        args.point_backbone_trainable = False
        args.tactile_trainable = False
        args.visual_trainable = False
        args.visual_finetune_mode = "frozen"
        args.semantic_trainable = False
        args.semantic_gradient_checkpointing = False
        args.window_activation_checkpointing = False
    if args.picf_trainable_scope == "policy_only":
        args.perception_finetune_mode = "frozen"
        args.point_backbone_trainable = False
        args.tactile_trainable = False
        args.visual_trainable = False
        args.visual_finetune_mode = "frozen"
    if not _picf_mode_enabled(args):
        args.point_backbone = "rgb"
        args.point_backbone_trainable = False
        args.visual_mode = "stub"
        args.visual_finetune_mode = "frozen"
        args.visual_trainable = False
        args.tactile_mode = "stub"
        args.tactile_trainable = False
        args.use_tactile = False
        args.use_scene_obs = False
        args.vl_anchor_router_enabled = False
        args.mapg_enabled = False
        args.aqr_mapg_enabled = False
        args.require_pi0_action_generator = True
    if getattr(args, "grad_clip_percentile", None) is None:
        args.grad_clip_percentile = 75.0
    else:
        args.grad_clip_percentile = float(args.grad_clip_percentile)
    if getattr(args, "grad_clip_window", None) is None:
        args.grad_clip_window = 100
    else:
        args.grad_clip_window = int(args.grad_clip_window)
    if getattr(args, "control_query_tokens", None) is None:
        args.control_query_tokens = int(_SPEC_DEFAULTS.control_query_tokens)
    if getattr(args, "predictive_query_tokens", None) is None:
        args.predictive_query_tokens = int(_SPEC_DEFAULTS.predictive_query_tokens)
    if getattr(args, "task_local_queries", None) is None:
        args.task_local_queries = int(_SPEC_DEFAULTS.task_local_queries)
    if getattr(args, "task_global_queries", None) is None:
        args.task_global_queries = int(_SPEC_DEFAULTS.task_global_queries)
    if getattr(args, "task_instruction_queries", None) is None:
        args.task_instruction_queries = int(_SPEC_DEFAULTS.task_instruction_queries)
    if getattr(args, "task_self_layers", None) is None:
        args.task_self_layers = int(_SPEC_DEFAULTS.task_self_layers)
    if getattr(args, "conditioned_control_queries", None) is None:
        args.conditioned_control_queries = int(_SPEC_DEFAULTS.conditioned_control_queries)
    if getattr(args, "pi_prefix_queries", None) is None:
        args.pi_prefix_queries = int(_SPEC_DEFAULTS.pi_prefix_queries)
    if getattr(args, "conditioned_future_queries", None) is None:
        args.conditioned_future_queries = int(_SPEC_DEFAULTS.conditioned_future_queries)
    if getattr(args, "task_visual_reread_topk", None) is None:
        args.task_visual_reread_topk = int(_SPEC_DEFAULTS.task_visual_reread_topk)
    if getattr(args, "task_tactile_reread_groups", None) is None:
        args.task_tactile_reread_groups = int(_SPEC_DEFAULTS.task_tactile_reread_groups)
    if getattr(args, "task_point_reread_topk", None) is None:
        args.task_point_reread_topk = int(_SPEC_DEFAULTS.task_point_reread_topk)
    if getattr(args, "vl_anchor_router_enabled", None) is None:
        args.vl_anchor_router_enabled = bool(_SPEC_DEFAULTS.vl_anchor_router_enabled)
    if getattr(args, "vl_grounding_view", None) is None:
        args.vl_grounding_view = str(_SPEC_DEFAULTS.vl_grounding_view)
    if getattr(args, "vl_anchor_modes", None) is None:
        args.vl_anchor_modes = int(_SPEC_DEFAULTS.vl_anchor_modes)
    if getattr(args, "vl_anchor_nms_radius_m", None) is None:
        args.vl_anchor_nms_radius_m = float(_SPEC_DEFAULTS.vl_anchor_nms_radius_m)
    if getattr(args, "vl_anchor_local_sigma_m", None) is None:
        args.vl_anchor_local_sigma_m = float(_SPEC_DEFAULTS.vl_anchor_local_sigma_m)
    if getattr(args, "vl_min_visible_mass", None) is None:
        args.vl_min_visible_mass = float(_SPEC_DEFAULTS.vl_min_visible_mass)
    if getattr(args, "vl_heatmap_temperature", None) is None:
        args.vl_heatmap_temperature = float(_SPEC_DEFAULTS.vl_heatmap_temperature)
    if getattr(args, "vl_obs_anchor_gate_init", None) is None:
        args.vl_obs_anchor_gate_init = float(_SPEC_DEFAULTS.vl_obs_anchor_gate_init)
    if getattr(args, "vl_task_point_gate_init", None) is None:
        args.vl_task_point_gate_init = float(_SPEC_DEFAULTS.vl_task_point_gate_init)
    if getattr(args, "vl_posterior_bind_gate_init", None) is None:
        args.vl_posterior_bind_gate_init = float(_SPEC_DEFAULTS.vl_posterior_bind_gate_init)
    if getattr(args, "vl_prior_bias_clip", None) is None:
        args.vl_prior_bias_clip = float(_SPEC_DEFAULTS.vl_prior_bias_clip)
    if getattr(args, "mapg_enabled", None) is None:
        args.mapg_enabled = bool(_SPEC_DEFAULTS.mapg_enabled)
    if getattr(args, "mapg_anchor_count", None) is None:
        args.mapg_anchor_count = int(_SPEC_DEFAULTS.mapg_anchor_count)
    if getattr(args, "mapg_message_rounds", None) is None:
        args.mapg_message_rounds = int(_SPEC_DEFAULTS.mapg_message_rounds)
    if getattr(args, "mapg_visual_sigma_patches", None) is None:
        args.mapg_visual_sigma_patches = float(_SPEC_DEFAULTS.mapg_visual_sigma_patches)
    if getattr(args, "mapg_tactile_sigma_m", None) is None:
        args.mapg_tactile_sigma_m = float(_SPEC_DEFAULTS.mapg_tactile_sigma_m)
    if getattr(args, "mapg_posterior_sigma_m", None) is None:
        args.mapg_posterior_sigma_m = float(_SPEC_DEFAULTS.mapg_posterior_sigma_m)
    if getattr(args, "mapg_confidence_floor", None) is None:
        args.mapg_confidence_floor = float(_SPEC_DEFAULTS.mapg_confidence_floor)
    if getattr(args, "mapg_assignment_sinkhorn_iters", None) is None:
        args.mapg_assignment_sinkhorn_iters = int(_SPEC_DEFAULTS.mapg_assignment_sinkhorn_iters)
    if getattr(args, "mapg_assignment_temperature", None) is None:
        args.mapg_assignment_temperature = float(_SPEC_DEFAULTS.mapg_assignment_temperature)
    if getattr(args, "mapg_assignment_quality_uniform_mix", None) is None:
        args.mapg_assignment_quality_uniform_mix = float(_SPEC_DEFAULTS.mapg_assignment_quality_uniform_mix)
    if getattr(args, "mapg_mode_confidence_threshold", None) is None:
        args.mapg_mode_confidence_threshold = float(_SPEC_DEFAULTS.mapg_mode_confidence_threshold)
    if getattr(args, "mapg_obs_gate_init", None) is None:
        args.mapg_obs_gate_init = float(_SPEC_DEFAULTS.mapg_obs_gate_init)
    if getattr(args, "mapg_task_gate_init", None) is None:
        args.mapg_task_gate_init = float(_SPEC_DEFAULTS.mapg_task_gate_init)
    if getattr(args, "mapg_posterior_gate_init", None) is None:
        args.mapg_posterior_gate_init = float(_SPEC_DEFAULTS.mapg_posterior_gate_init)
    if getattr(args, "mapg_control_gate_init", None) is None:
        args.mapg_control_gate_init = float(_SPEC_DEFAULTS.mapg_control_gate_init)
    if getattr(args, "mapg_obs_point_mix_floor", None) is None:
        args.mapg_obs_point_mix_floor = float(_SPEC_DEFAULTS.mapg_obs_point_mix_floor)
    if getattr(args, "mapg_prior_bias_clip", None) is None:
        args.mapg_prior_bias_clip = float(_SPEC_DEFAULTS.mapg_prior_bias_clip)
    if getattr(args, "aqr_mapg_enabled", None) is None:
        args.aqr_mapg_enabled = bool(_SPEC_DEFAULTS.aqr_mapg_enabled)
    if getattr(args, "aqr_pg_grounding_enabled", None) is None:
        args.aqr_pg_grounding_enabled = bool(_SPEC_DEFAULTS.aqr_pg_grounding_enabled)
    if getattr(args, "aqr_pg_image_support_enabled", None) is None:
        args.aqr_pg_image_support_enabled = bool(_SPEC_DEFAULTS.aqr_pg_image_support_enabled)
    for _name in (
        "aqr_query_count_physical",
        "aqr_query_count_task",
        "aqr_query_rounds",
        "aqr_sinkhorn_iters",
        "aqr_vjepa_temporal_tokens",
        "aqr_active_slot_min_per_role",
        "aqr_active_slot_max_per_role",
        "posterior_file_competition_min_per_role",
        "posterior_file_competition_max_per_role",
        "evidence_cache_len",
        "vjepa_max_views",
        "tracklet_max_tokens",
        "proposal_max_tokens",
        "binding_signature_dim",
        "binding_low_rank_signature_rank",
        "binding_signature_centering_min_tokens",
        "local_refinement_topk",
    ):
        if getattr(args, _name, None) is None:
            setattr(args, _name, int(getattr(_SPEC_DEFAULTS, _name)))
    if getattr(args, "aqr_vjepa_temporal_mode", None) is None:
        args.aqr_vjepa_temporal_mode = str(_SPEC_DEFAULTS.aqr_vjepa_temporal_mode)
    if getattr(args, "recycle_residual_norm_mode", None) is None:
        args.recycle_residual_norm_mode = str(_SPEC_DEFAULTS.recycle_residual_norm_mode)
    if getattr(args, "binding_signature_score_calibration_mode", None) is None:
        args.binding_signature_score_calibration_mode = str(_SPEC_DEFAULTS.binding_signature_score_calibration_mode)
    for _name in (
        "aqr_vjepa_temporal_include_delta",
        "vjepa_multiview_enabled",
        "aqr_ownership_prior_enabled",
        "aqr_active_slot_filter_enabled",
        "aqr_active_slot_geometry_duplicate_enabled",
        "posterior_owner_active_gate_enabled",
        "posterior_file_competition_enabled",
        "posterior_file_competition_geometry_duplicate_enabled",
        "evidence_cache_enabled",
        "tracklet_memory_enabled",
        "proposal_memory_enabled",
        "binding_signature_centering_enabled",
        "binding_signature_score_calibration_enabled",
        "posterior_binding_signature_memory_enabled",
        "posterior_binding_signature_dispersion_gate_enabled",
        "legacy_local_refinement_opt_in",
        "local_refinement_enabled",
        "slot_jepa_enabled",
        "support_prediction_enabled",
        "ordinal_relation_enabled",
        "ordinal_weak_target_enabled",
        "action_prefix_stopgrad",
    ):
        if getattr(args, _name, None) is None:
            setattr(args, _name, bool(_spec_default(_name)))
    for _name in (
        "aqr_sinkhorn_temperature",
        "aqr_pg_image_support_weight",
        "aqr_pg_entropy_threshold",
        "aqr_pg_peak_threshold",
        "aqr_pg_bias_weight",
        "aqr_support_bias_clip",
        "aqr_ownership_prior_weight",
        "aqr_ownership_point_prior_weight",
        "aqr_ownership_point_prior_sigma_m",
        "aqr_ownership_temporal_prior_weight",
        "aqr_ownership_prior_uniform_mix",
        "aqr_active_slot_min_confidence",
        "aqr_active_slot_overlap_threshold",
        "aqr_active_slot_relative_score_threshold",
        "aqr_active_slot_geometry_duplicate_sigma_m",
        "aqr_active_slot_geometry_duplicate_threshold",
        "posterior_owner_active_min",
        "posterior_owner_active_bias",
        "posterior_file_competition_min_support",
        "posterior_file_competition_relative_score_threshold",
        "posterior_file_competition_support_overlap_threshold",
        "posterior_file_competition_geometry_sigma_m",
        "posterior_file_competition_geometry_threshold",
        "evidence_cache_read_weight",
        "evidence_cache_innovation_downweight",
        "evidence_cache_address_weight",
        "evidence_cache_content_weight",
        "evidence_cache_role_weight",
        "tactile_evidence_prob_floor",
        "tracklet_confidence_floor",
        "tracklet_read_weight",
        "proposal_confidence_floor",
        "proposal_read_weight",
        "proposal_age_decay_steps",
        "proposal_point_bridge_weight",
        "proposal_point_bridge_edge_tau",
        "task_owner_proposal_point_bridge_weight",
        "task_owner_visual_bias_weight",
        "task_owner_proposal_bias_weight",
        "task_owner_proposal_point_bias_weight",
        "task_owner_proposal_objectness_power",
        "bind_support_signature_weight",
        "bind_embedding_signature_weight",
        "bind_quadratic_signature_weight",
        "bind_low_rank_signature_weight",
        "binding_signature_score_min_std",
        "binding_signature_score_clip",
        "posterior_binding_signature_update_rate",
        "posterior_binding_signature_update_max_rate",
        "posterior_binding_signature_min_support",
        "posterior_binding_signature_owner_weight",
        "posterior_binding_signature_measurement_min_std",
        "posterior_binding_signature_measurement_margin_min",
        "posterior_binding_signature_measurement_margin_temperature",
        "bind_address_weight",
        "bind_address_innovation_downweight",
        "address_update_rate",
        "address_update_max_rate",
        "local_refinement_weight",
        "local_refinement_binding_weight",
        "aqr_obs_gate_init",
        "aqr_task_gate_init",
        "aqr_posterior_gate_init",
        "aqr_control_gate_init",
    ):
        if getattr(args, _name, None) is None:
            setattr(args, _name, float(_spec_default(_name)))
    if getattr(args, "require_pi0_action_generator", None) is None:
        args.require_pi0_action_generator = bool(_SPEC_DEFAULTS.require_pi0_action_generator)
    if getattr(args, "semantic_prefix_dropout_prob", None) is None:
        args.semantic_prefix_dropout_prob = float(_SPEC_DEFAULTS.semantic_prefix_dropout_prob)
    if getattr(args, "enable_aux_budgeting", None) is None:
        args.enable_aux_budgeting = True
    if getattr(args, "aux_budget_physical_ratio", None) is None:
        args.aux_budget_physical_ratio = 0.20
    if getattr(args, "aux_budget_semantic_ratio", None) is None:
        args.aux_budget_semantic_ratio = 0.10
    if getattr(args, "aux_budget_alignment_ratio", None) is None:
        args.aux_budget_alignment_ratio = 0.05
    if getattr(args, "aux_budget_floor", None) is None:
        args.aux_budget_floor = 0.25
    if getattr(args, "tactile_aux_force_scale", None) is None:
        args.tactile_aux_force_scale = 1.0
    if getattr(args, "tactile_aux_indent_scale", None) is None:
        args.tactile_aux_indent_scale = 5e-4
    if getattr(args, "tactile_aux_pressure_scale", None) is None:
        args.tactile_aux_pressure_scale = 0.1
    if getattr(args, "action_normalization", None) is None:
        args.action_normalization = "quantile"
    if getattr(args, "action_output_clip", None) is not None:
        args.action_output_clip = float(args.action_output_clip)
    if getattr(args, "action_normalization", "quantile") != "none" and getattr(args, "action_norm_stats_path", None) is None:
        args.action_norm_stats_path = _default_action_norm_stats_path()
    prompt_state_mode = str(getattr(args, "prompt_state_normalization", "inherit")).lower()
    if prompt_state_mode == "inherit":
        prompt_state_mode = str(getattr(args, "action_normalization", "quantile")).lower()
    args.prompt_state_normalization = prompt_state_mode
    if args.prompt_state_normalization != "none" and getattr(args, "prompt_state_norm_stats_path", None) is None:
        args.prompt_state_norm_stats_path = getattr(args, "action_norm_stats_path", None) or _default_action_norm_stats_path()
    if getattr(args, "tactile_aux_pose_scale", None) is None:
        args.tactile_aux_pose_scale = float(getattr(args, "crop_radius_m", _SPEC_DEFAULTS.crop_radius_m))
    if getattr(args, "tactile_aux_huber_delta", None) is None:
        args.tactile_aux_huber_delta = 1.0
    if getattr(args, "predictive_semantic_reads", None) is None:
        args.predictive_semantic_reads = int(_SPEC_DEFAULTS.predictive_semantic_reads)
    if getattr(args, "control_semantic_reads", None) is None:
        args.control_semantic_reads = int(_SPEC_DEFAULTS.control_semantic_reads)
    if getattr(args, "tokenwise_ff_chunk_size", None) is None:
        args.tokenwise_ff_chunk_size = int(_SPEC_DEFAULTS.tokenwise_ff_chunk_size)
    else:
        args.tokenwise_ff_chunk_size = int(args.tokenwise_ff_chunk_size)
    if getattr(args, "semantic_tokenwise_chunk_size", None) is None:
        args.semantic_tokenwise_chunk_size = 0
    else:
        args.semantic_tokenwise_chunk_size = int(args.semantic_tokenwise_chunk_size)
    semantic_projection_chunk_was_none = getattr(args, "semantic_projection_chunk_size", None) is None
    semantic_mlp_chunk_was_none = getattr(args, "semantic_mlp_chunk_size", None) is None
    if semantic_projection_chunk_was_none:
        args.semantic_projection_chunk_size = int(args.semantic_tokenwise_chunk_size)
    else:
        args.semantic_projection_chunk_size = int(args.semantic_projection_chunk_size)
    if semantic_mlp_chunk_was_none:
        args.semantic_mlp_chunk_size = int(args.semantic_tokenwise_chunk_size)
    else:
        args.semantic_mlp_chunk_size = int(args.semantic_mlp_chunk_size)
    if args.training_strategy == "fsdp_full_shard":
        if int(args.tokenwise_ff_chunk_size) <= 0:
            args.tokenwise_ff_chunk_size = 64
        if (
            str(getattr(args, "semantic_mode", "zero")) == "paligemma"
            and bool(getattr(args, "semantic_trainable", False))
        ):
            if int(args.semantic_tokenwise_chunk_size) <= 0:
                args.semantic_tokenwise_chunk_size = 64
            if semantic_projection_chunk_was_none and int(args.semantic_projection_chunk_size) <= 0:
                args.semantic_projection_chunk_size = 128
            elif int(args.semantic_projection_chunk_size) <= 0:
                args.semantic_projection_chunk_size = int(args.semantic_tokenwise_chunk_size)
            if semantic_mlp_chunk_was_none and int(args.semantic_mlp_chunk_size) <= 0:
                args.semantic_mlp_chunk_size = 64
            elif int(args.semantic_mlp_chunk_size) <= 0:
                args.semantic_mlp_chunk_size = int(args.semantic_tokenwise_chunk_size)
    if args.warmup_steps is None:
        args.warmup_steps = max(1, int(round(0.02 * float(args.num_train_steps))))
    else:
        args.warmup_steps = int(args.warmup_steps)
    if getattr(args, "point_backbone_lr_scale", None) is None:
        args.point_backbone_lr_scale = 0.25
    if getattr(args, "visual_lr_scale", None) is None:
        args.visual_lr_scale = 0.25
    if getattr(args, "tactile_lr_scale", None) is None:
        args.tactile_lr_scale = 0.25
    if getattr(args, "semantic_lr_scale", None) is None:
        args.semantic_lr_scale = 0.25
    args.semantic_trainable_scope = str(
        getattr(args, "semantic_trainable_scope", "backbone_only")
    ).strip().lower().replace("-", "_")
    if getattr(args, "picf_core_lr_scale", None) is None:
        args.picf_core_lr_scale = 1.0
    if getattr(args, "policy_head_lr_scale", None) is None:
        args.policy_head_lr_scale = 1.0
    args.picf_core_lr_runtime_mode = str(
        getattr(args, "picf_core_lr_runtime_mode", "constant")
    ).lower().replace("-", "_")
    if getattr(args, "action_prefix_norm_mode", None) is None:
        args.action_prefix_norm_mode = str(_SPEC_DEFAULTS.action_prefix_norm_mode)
    if getattr(args, "action_prefix_rms_target", None) is None:
        args.action_prefix_rms_target = float(_SPEC_DEFAULTS.action_prefix_rms_target)
    if getattr(args, "action_prefix_norm_eps", None) is None:
        args.action_prefix_norm_eps = float(_SPEC_DEFAULTS.action_prefix_norm_eps)
    if getattr(args, "action_prefix_value_clip", None) is None:
        args.action_prefix_value_clip = float(_SPEC_DEFAULTS.action_prefix_value_clip)
    if getattr(args, "picf_action_condition_enabled", None) is None:
        args.picf_action_condition_enabled = bool(_SPEC_DEFAULTS.picf_action_condition_enabled)
    if getattr(args, "action_prefix_output_gate", None) is None:
        args.action_prefix_output_gate = float(_SPEC_DEFAULTS.action_prefix_output_gate)
    if getattr(args, "action_prefix_teacher_mode", None) is None:
        args.action_prefix_teacher_mode = str(_SPEC_DEFAULTS.action_prefix_teacher_mode)
    if getattr(args, "action_prefix_teacher_ema_decay", None) is None:
        args.action_prefix_teacher_ema_decay = float(_SPEC_DEFAULTS.action_prefix_teacher_ema_decay)
    if getattr(args, "action_prefix_teacher_blend", None) is None:
        args.action_prefix_teacher_blend = float(_SPEC_DEFAULTS.action_prefix_teacher_blend)
    if getattr(args, "lambda_action_prefix_trust", None) is None:
        args.lambda_action_prefix_trust = float(_SPEC_DEFAULTS.lambda_action_prefix_trust)
    if getattr(args, "pt_bag_radius_m", None) is None:
        args.pt_bag_radius_m = 0.045
        args.pt_bag_radius_m_source = "default"
    else:
        args.pt_bag_radius_m = float(args.pt_bag_radius_m)
        args.pt_bag_radius_m_source = "cli"
    if getattr(args, "pt_bag_sigma_m", None) is None:
        args.pt_bag_sigma_m = 0.015
        args.pt_bag_sigma_m_source = "default"
    else:
        args.pt_bag_sigma_m = float(args.pt_bag_sigma_m)
        args.pt_bag_sigma_m_source = "cli"
    args.semantic_gradient_checkpointing_disabled_for_accum = False
    if (
        int(args.accum_steps) > 1
        and bool(getattr(args, "semantic_gradient_checkpointing", False))
        and str(getattr(args, "semantic_mode", "zero")) == "paligemma"
    ):
        # HF PaliGemma gradient checkpointing is reentrant and collides with DDP
        # gradient accumulation, causing "mark ready twice" errors on repeated
        # backwards within one optimizer step.
        args.semantic_gradient_checkpointing = False
        args.semantic_gradient_checkpointing_disabled_for_accum = True

def _resolve_action_normalizer(args: argparse.Namespace) -> PicfActionNormalizer | None:
    mode = str(getattr(args, "action_normalization", "quantile"))
    if mode == "none":
        return None
    path = getattr(args, "action_norm_stats_path", None)
    if path is None:
        raise FileNotFoundError(
            "Action normalization requested but no norm_stats path was resolved. "
            "Pass --action-norm-stats-path or place norm_stats.json under assets/pi05_calvin_sonata/calvin/."
        )
    return PicfActionNormalizer.from_path(path, mode=mode)


def _warn_single_gpu_foundation_accum_risk(
    args: argparse.Namespace,
    *,
    world_size: int,
    logger: logging.Logger,
) -> None:
    if (
        bool(getattr(args, "use_foundation_backbones", False))
        and str(getattr(args, "device", "")).startswith("cuda")
        and int(world_size) == 1
        and int(getattr(args, "accum_steps", 1)) > 1
    ):
        logger.warning(
            "Single-GPU full foundation training with accum_steps=%s is OOM-prone on ~40 GiB GPUs. "
            "Prefer accum_steps=1 on one GPU, or move to 2-GPU DDP if you need effective_global_batch>1.",
            int(args.accum_steps),
        )


def _validate_train_args(args: argparse.Namespace) -> None:
    if str(getattr(args, "picf_mode", "enabled")) not in {"enabled", "ablated"}:
        raise ValueError(
            "picf_mode must be one of {'enabled', 'ablated'}, "
            f"got {getattr(args, 'picf_mode', None)!r}."
        )
    args.aqr_role_layout = str(
        getattr(args, "aqr_role_layout", _SPEC_DEFAULTS.aqr_role_layout)
    ).lower().replace("-", "_")
    if args.aqr_role_layout not in {"structured", "no_effector", "object_contact_context", "object_only", "object"}:
        raise ValueError(
            "aqr_role_layout must be one of {'structured', 'no_effector', 'object_contact_context', 'object_only'}, "
            f"got {getattr(args, 'aqr_role_layout', None)!r}."
        )
    positive_int_fields = (
        "num_train_steps",
        "log_interval",
        "save_interval",
        "diagnostic_visual_upscale",
        "accum_steps",
        "max_empty_window_retries",
        "unroll_steps",
        "stride",
        "max_points",
        "visual_grid",
        "visual_real_grid",
        "visual_num_frames",
        "visual_img_size",
        "visual_patch_size",
        "visual_tubelet_size",
        "tactile_num_frames",
        "tactile_stride",
        "pt_bag_kmin",
        "hidden_dim",
        "posterior_hidden_dim",
        "latent_dim",
        "innovation_dim",
        "control_dim",
        "semantic_dim",
        "semantic_cross_dim",
        "future_hidden_dim",
        "persistent_anchors",
        "observation_anchors",
        "global_scene_point_cap",
        "fusion_layers",
        "posterior_layers",
        "predictive_layers",
        "control_layers",
        "control_query_tokens",
        "predictive_query_tokens",
        "task_local_queries",
        "task_global_queries",
        "task_instruction_queries",
        "task_self_layers",
        "conditioned_control_queries",
        "pi_prefix_queries",
        "conditioned_future_queries",
        "predictive_semantic_reads",
        "control_semantic_reads",
        "attention_heads",
        "future_vote_heads",
        "task_visual_reread_topk",
        "task_tactile_reread_groups",
        "task_point_reread_topk",
        "mapg_anchor_count",
        "mapg_message_rounds",
    )
    for name in positive_int_fields:
        value = int(getattr(args, name))
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")
    no_effector_probe_fields = (
        "effector_persistent_anchors",
        "effector_observation_anchors",
        "task_effector_queries",
    )
    for name in no_effector_probe_fields:
        value = int(getattr(args, name))
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    if args.aqr_role_layout == "structured":
        for name in no_effector_probe_fields:
            value = int(getattr(args, name))
            if value < 1:
                raise ValueError(
                    f"{name} must be >= 1 for structured aqr_role_layout; "
                    "use --aqr-role-layout object_only/no_effector for a clean no-effector probe."
                )
    if int(getattr(args, "burnin_steps", 0)) < 0:
        raise ValueError(f"burnin_steps must be >= 0, got {args.burnin_steps}.")
    if str(getattr(args, "burnin_mode", "full")) not in {"full", "state_only"}:
        raise ValueError(
            "burnin_mode must be one of {'full', 'state_only'}, "
            f"got {getattr(args, 'burnin_mode', None)!r}."
        )
    if int(getattr(args, "effective_unroll_steps", 0)) != (
        int(args.unroll_steps) + int(getattr(args, "burnin_steps", 0))
    ):
        raise ValueError(
            "effective_unroll_steps must equal burnin_steps + unroll_steps, "
            f"got effective_unroll_steps={getattr(args, 'effective_unroll_steps', None)} "
            f"burnin_steps={getattr(args, 'burnin_steps', None)} unroll_steps={args.unroll_steps}."
        )
    if int(args.burnin_steps) > 0 and not _picf_mode_enabled(args):
        raise ValueError("burnin_steps > 0 requires picf_mode=enabled; ablated PI0.5 has no PICF carry to burn in.")
    if bool(getattr(args, "vl_anchor_router_enabled", False)) and not _picf_mode_enabled(args):
        raise ValueError("vl_anchor_router_enabled requires picf_mode=enabled.")
    if bool(getattr(args, "vl_anchor_router_enabled", False)) and str(getattr(args, "semantic_mode", "zero")) != "paligemma":
        raise ValueError("vl_anchor_router_enabled requires semantic_mode=paligemma so PaliGemma image tokens exist.")
    if bool(getattr(args, "mapg_enabled", False)) and not _picf_mode_enabled(args):
        raise ValueError("mapg_enabled requires picf_mode=enabled.")
    if bool(getattr(args, "mapg_enabled", False)) and str(getattr(args, "semantic_mode", "zero")) != "paligemma":
        raise ValueError("mapg_enabled requires semantic_mode=paligemma so MAPG can build PaliGemma grounding priors.")
    if bool(getattr(args, "aqr_mapg_enabled", False)) and not _picf_mode_enabled(args):
        raise ValueError("aqr_mapg_enabled requires picf_mode=enabled.")
    if bool(getattr(args, "aqr_mapg_enabled", False)) and str(getattr(args, "semantic_mode", "zero")) != "paligemma":
        raise ValueError("aqr_mapg_enabled requires semantic_mode=paligemma for typed support memory and weak PaliGemma priors.")
    if bool(getattr(args, "aqr_mapg_enabled", False)) and bool(getattr(args, "mapg_enabled", False)):
        raise ValueError("aqr_mapg_enabled is the direct-final graph path; do not enable legacy mapg_enabled at the same time.")
    if (
        _looks_like_legacy_blind_sam_sidecar(getattr(args, "mvtrack_sidecar_root", None))
        and not bool(getattr(args, "allow_legacy_blind_sam_sidecar", False))
    ):
        raise ValueError(
            "mvtrack_sidecar_root appears to be an archived blind-SAM proposal root. "
            "Blind automatic SAM proposals are rejected for current PICF-AQR-OWM training. "
            "Use contact/task/tracklet-aware sidecars instead, or pass "
            "--allow-legacy-blind-sam-sidecar only for historical reproduction."
        )
    if int(args.warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {args.warmup_steps}.")
    if int(getattr(args, "keep_last_checkpoints", 0)) < 0:
        raise ValueError(
            f"keep_last_checkpoints must be >= 0, got {getattr(args, 'keep_last_checkpoints', None)}."
        )
    if float(args.lr) <= 0.0:
        raise ValueError(f"lr must be > 0, got {args.lr}.")
    if float(args.min_lr) < 0.0:
        raise ValueError(f"min_lr must be >= 0, got {args.min_lr}.")
    if float(args.min_lr) > float(args.lr):
        raise ValueError(f"min_lr must be <= lr, got min_lr={args.min_lr} lr={args.lr}.")
    if float(args.weight_decay) < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {args.weight_decay}.")
    if str(args.grad_clip_mode) not in {"fixed", "percentile"}:
        raise ValueError(
            "grad_clip_mode must be one of {'fixed', 'percentile'}, "
            f"got {args.grad_clip_mode!r}."
        )
    if str(args.training_strategy) not in {"ddp", "fsdp_full_shard"}:
        raise ValueError(
            "training_strategy must be one of {'ddp', 'fsdp_full_shard'}, "
            f"got {args.training_strategy!r}."
        )
    if str(args.optimizer_sharding) not in {"none", "zero1"}:
        raise ValueError(
            "optimizer_sharding must be one of {'none', 'zero1'}, "
            f"got {args.optimizer_sharding!r}."
        )
    if str(args.training_strategy) == "fsdp_full_shard" and str(args.optimizer_sharding) != "none":
        raise ValueError(
            "training_strategy=fsdp_full_shard is incompatible with optimizer_sharding. "
            "FSDP full_shard already shards parameter, gradient, and optimizer state."
        )
    if bool(getattr(args, "window_activation_checkpointing", False)) and str(args.training_strategy) != "fsdp_full_shard":
        raise ValueError(
            "window_activation_checkpointing is reserved for training_strategy=fsdp_full_shard, "
            f"got training_strategy={args.training_strategy!r}."
        )
    if str(args.optimizer_checkpoint_mode) not in {"auto", "full", "model_only"}:
        raise ValueError(
            "optimizer_checkpoint_mode must be one of {'auto', 'full', 'model-only'}, "
            f"got {args.optimizer_checkpoint_mode!r}."
        )
    if float(args.grad_clip_norm) < 0.0:
        raise ValueError(f"grad_clip_norm must be >= 0, got {args.grad_clip_norm}.")
    if not (0.0 <= float(args.grad_clip_percentile) <= 100.0):
        raise ValueError(
            "grad_clip_percentile must be in [0, 100], "
            f"got {args.grad_clip_percentile}."
        )
    if int(args.grad_clip_window) < 1:
        raise ValueError(f"grad_clip_window must be >= 1, got {args.grad_clip_window}.")
    if int(getattr(args, "tokenwise_ff_chunk_size", 0)) < 0:
        raise ValueError(f"tokenwise_ff_chunk_size must be >= 0, got {args.tokenwise_ff_chunk_size}.")
    if int(getattr(args, "semantic_tokenwise_chunk_size", 0)) < 0:
        raise ValueError(
            "semantic_tokenwise_chunk_size must be >= 0, "
            f"got {args.semantic_tokenwise_chunk_size}."
        )
    if int(getattr(args, "semantic_projection_chunk_size", 0)) < 0:
        raise ValueError(
            "semantic_projection_chunk_size must be >= 0, "
            f"got {args.semantic_projection_chunk_size}."
        )
    if int(getattr(args, "semantic_mlp_chunk_size", 0)) < 0:
        raise ValueError(
            "semantic_mlp_chunk_size must be >= 0, "
            f"got {args.semantic_mlp_chunk_size}."
        )
    if getattr(args, "action_output_clip", None) is not None and float(args.action_output_clip) <= 0.0:
        raise ValueError(f"action_output_clip must be > 0 when provided, got {args.action_output_clip}.")
    if float(args.crop_radius_m) <= 0.0:
        raise ValueError(f"crop_radius_m must be > 0, got {args.crop_radius_m}.")
    if float(args.scene_anchor_border_patches) < 0.0:
        raise ValueError(f"scene_anchor_border_patches must be >= 0, got {args.scene_anchor_border_patches}.")
    for name in ("effector_persistent_anchors", "effector_observation_anchors", "task_effector_queries", "global_scene_point_cap"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"{name} must be >= 0, got {getattr(args, name)}.")
    if int(getattr(args, "vl_anchor_modes", 0)) < 0:
        raise ValueError(f"vl_anchor_modes must be >= 0, got {getattr(args, 'vl_anchor_modes')}.")
    if int(args.effector_persistent_anchors) > int(args.persistent_anchors):
        raise ValueError("effector_persistent_anchors must be <= persistent_anchors.")
    if int(args.effector_observation_anchors) > int(args.observation_anchors):
        raise ValueError("effector_observation_anchors must be <= observation_anchors.")
    if int(args.task_effector_queries) > int(args.task_local_queries):
        raise ValueError("task_effector_queries must be <= task_local_queries.")
    if float(args.point_focus_sigma_m) <= 0.0:
        raise ValueError(f"point_focus_sigma_m must be > 0, got {args.point_focus_sigma_m}.")
    if int(args.diagnostic_interval) < 0:
        raise ValueError(f"diagnostic_interval must be >= 0, got {args.diagnostic_interval}.")
    if int(getattr(args, "anchor_overlay_interval", 0)) < 0:
        raise ValueError(f"anchor_overlay_interval must be >= 0, got {getattr(args, 'anchor_overlay_interval', None)}.")
    if int(getattr(args, "anchor_overlay_max_anchors", 64)) < 0:
        raise ValueError(
            f"anchor_overlay_max_anchors must be >= 0, got {getattr(args, 'anchor_overlay_max_anchors', None)}."
        )
    if bool(getattr(args, "anchor_overlay_dump_signatures", False)) and int(getattr(args, "anchor_overlay_interval", 0)) <= 0:
        raise ValueError("--anchor-overlay-dump-signatures requires --anchor-overlay-interval > 0.")
    if float(getattr(args, "logical_batch_action_bucket_ema_decay", 0.98)) < 0.0 or float(
        getattr(args, "logical_batch_action_bucket_ema_decay", 0.98)
    ) >= 1.0:
        raise ValueError(
            "logical_batch_action_bucket_ema_decay must be in [0, 1), "
            f"got {getattr(args, 'logical_batch_action_bucket_ema_decay', None)}."
        )
    if float(getattr(args, "logical_batch_action_bucket_scale_min", 0.5)) <= 0.0:
        raise ValueError(
            "logical_batch_action_bucket_scale_min must be > 0, "
            f"got {getattr(args, 'logical_batch_action_bucket_scale_min', None)}."
        )
    if float(getattr(args, "logical_batch_action_bucket_scale_max", 1.5)) < float(
        getattr(args, "logical_batch_action_bucket_scale_min", 0.5)
    ):
        raise ValueError(
            "logical_batch_action_bucket_scale_max must be >= logical_batch_action_bucket_scale_min, "
            f"got min={getattr(args, 'logical_batch_action_bucket_scale_min', None)} "
            f"max={getattr(args, 'logical_batch_action_bucket_scale_max', None)}."
        )
    if int(getattr(args, "logical_batch_action_bucket_min_count", 2)) < 1:
        raise ValueError(
            "logical_batch_action_bucket_min_count must be >= 1, "
            f"got {getattr(args, 'logical_batch_action_bucket_min_count', None)}."
        )
    gradient_surgery = str(getattr(args, "logical_batch_gradient_surgery", "off")).lower().replace("-", "_")
    if gradient_surgery not in {"off", "pcgrad", "cagrad"}:
        raise ValueError(
            "logical_batch_gradient_surgery must be one of {'off', 'pcgrad', 'cagrad'}, "
            f"got {getattr(args, 'logical_batch_gradient_surgery', None)!r}."
        )
    gradient_surgery_groups = {
        str(group).strip()
        for group in str(getattr(args, "logical_batch_gradient_surgery_groups", "")).split(",")
        if str(group).strip()
    }
    valid_gradient_surgery_groups = {"semantic", "policy_head", "picf_core"}
    invalid_gradient_surgery_groups = sorted(gradient_surgery_groups - valid_gradient_surgery_groups)
    if invalid_gradient_surgery_groups:
        raise ValueError(
            "logical_batch_gradient_surgery_groups contains unsupported groups "
            f"{invalid_gradient_surgery_groups!r}; valid groups are {sorted(valid_gradient_surgery_groups)!r}."
        )
    if gradient_surgery != "off":
        if not gradient_surgery_groups:
            raise ValueError("--logical-batch-gradient-surgery requires at least one target group.")
        if not bool(getattr(args, "logical_batch_bucket_normalization", False)):
            raise ValueError(
                "--logical-batch-gradient-surgery requires --logical-batch-bucket-normalization so per-micro "
                "gradients correspond to the controlled logical-batch objective."
            )
    semantic_scope_choices = {
        "all",
        "backbone_only",
        "model_only",
        "action_head_only",
        "action_adapter_only",
        "action_head_and_adapter",
    }
    if str(getattr(args, "semantic_trainable_scope", "backbone_only")) not in semantic_scope_choices:
        raise ValueError(
            "semantic_trainable_scope must be one of "
            "{'all', 'backbone_only', 'model_only', 'action_head_only', "
            "'action_adapter_only', 'action_head_and_adapter'}, "
            f"got {getattr(args, 'semantic_trainable_scope', None)!r}."
        )
    action_flow_loss = str(getattr(args, "semantic_action_flow_loss", "mse")).strip().lower().replace("-", "_")
    if action_flow_loss not in {"mse", "l2", "l1", "mae", "huber", "smooth_l1", "smoothl1"}:
        raise ValueError(
            "semantic_action_flow_loss must be one of {'mse', 'l1', 'huber', 'smooth_l1'}, "
            f"got {getattr(args, 'semantic_action_flow_loss', None)!r}."
        )
    if float(getattr(args, "semantic_action_flow_huber_delta", 1.0)) <= 0.0:
        raise ValueError("semantic_action_flow_huber_delta must be > 0.")
    if float(getattr(args, "semantic_action_flow_time_alpha", 1.5)) <= 0.0:
        raise ValueError("semantic_action_flow_time_alpha must be > 0.")
    if float(getattr(args, "semantic_action_flow_time_beta", 1.0)) <= 0.0:
        raise ValueError("semantic_action_flow_time_beta must be > 0.")
    readout_loss = (
        str(getattr(args, "semantic_action_context_readout_aux_loss", "smooth_l1")).strip().lower().replace("-", "_")
    )
    if readout_loss not in {"mse", "l2", "l1", "mae", "huber", "smooth_l1", "smoothl1"}:
        raise ValueError(
            "semantic_action_context_readout_aux_loss must be one of {'mse', 'l1', 'huber', 'smooth_l1'}, "
            f"got {getattr(args, 'semantic_action_context_readout_aux_loss', None)!r}."
        )
    if float(getattr(args, "semantic_action_context_readout_aux_weight", 0.0)) < 0.0:
        raise ValueError("semantic_action_context_readout_aux_weight must be >= 0.")
    if float(getattr(args, "semantic_action_context_readout_aux_huber_delta", 1.0)) <= 0.0:
        raise ValueError("semantic_action_context_readout_aux_huber_delta must be > 0.")
    if float(getattr(args, "semantic_action_context_token_aux_weight", 0.0)) < 0.0:
        raise ValueError("semantic_action_context_token_aux_weight must be >= 0.")
    if int(getattr(args, "semantic_action_context_token_aux_bins", 256)) < 2:
        raise ValueError("semantic_action_context_token_aux_bins must be >= 2.")
    if float(getattr(args, "semantic_action_context_token_aux_clip", 1.0)) <= 0.0:
        raise ValueError("semantic_action_context_token_aux_clip must be > 0.")
    if float(getattr(args, "semantic_action_context_flow_residual_time_floor", 0.05)) <= 0.0:
        raise ValueError("semantic_action_context_flow_residual_time_floor must be > 0.")
    if int(getattr(args, "semantic_action_expert_router_experts", 4)) < 1:
        raise ValueError("semantic_action_expert_router_experts must be >= 1.")
    if int(getattr(args, "semantic_action_expert_router_rank", 64)) < 1:
        raise ValueError("semantic_action_expert_router_rank must be >= 1.")
    if float(getattr(args, "semantic_action_expert_router_temperature", 1.0)) <= 0.0:
        raise ValueError("semantic_action_expert_router_temperature must be > 0.")
    for name in (
        "point_backbone_lr_scale",
        "visual_lr_scale",
        "tactile_lr_scale",
        "semantic_lr_scale",
        "picf_core_lr_scale",
        "policy_head_lr_scale",
    ):
        value = float(getattr(args, name))
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}.")
    for name in ("vl_anchor_nms_radius_m", "vl_anchor_local_sigma_m", "vl_heatmap_temperature"):
        value = float(getattr(args, name))
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}.")
    for name in ("vl_min_visible_mass", "vl_prior_bias_clip"):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    for name in (
        "lambda_action_pos",
        "lambda_action_rot",
        "lambda_action_gripper",
        "lambda_visual_latent",
        "lambda_visual_real",
        "lambda_tactile_real",
        "lambda_point_real",
        "lambda_semantic_future_aux",
        "lambda_anchor_pv",
        "lambda_pv_weak",
        "lambda_pt",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    for name in (
        "aux_budget_physical_ratio",
        "aux_budget_semantic_ratio",
        "aux_budget_alignment_ratio",
        "aux_budget_floor",
        "tactile_aux_force_scale",
        "tactile_aux_indent_scale",
        "tactile_aux_pressure_scale",
        "tactile_aux_pose_scale",
        "tactile_aux_huber_delta",
    ):
        value = float(getattr(args, name))
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0, got {value}.")
    if float(args.pt_bag_radius_m) <= 0.0:
        raise ValueError(f"pt_bag_radius_m must be > 0, got {args.pt_bag_radius_m}.")
    if float(args.pt_bag_sigma_m) <= 0.0:
        raise ValueError(f"pt_bag_sigma_m must be > 0, got {args.pt_bag_sigma_m}.")
    if float(args.pt_back_slack_m) < 0.0:
        raise ValueError(f"pt_back_slack_m must be >= 0, got {args.pt_back_slack_m}.")
    if not (0.0 <= float(args.p_align_off) <= float(args.p_align_on) <= 1.0):
        raise ValueError(
            "p_align_off / p_align_on must satisfy 0 <= p_align_off <= p_align_on <= 1; "
            f"got p_align_off={args.p_align_off} p_align_on={args.p_align_on}."
        )
    if float(args.tactile_anchor_prob_on) < 0.0 or float(args.tactile_anchor_prob_on) > 1.0:
        raise ValueError(
            f"tactile_anchor_prob_on must be in [0, 1], got {args.tactile_anchor_prob_on}."
        )
    if float(args.tactile_evidence_prob_floor) < 0.0 or float(args.tactile_evidence_prob_floor) > 1.0:
        raise ValueError(
            f"tactile_evidence_prob_floor must be in [0, 1], got {args.tactile_evidence_prob_floor}."
        )
    if float(args.tactile_evidence_prob_floor) > float(args.tactile_anchor_prob_on):
        raise ValueError(
            "tactile_evidence_prob_floor must be <= tactile_anchor_prob_on; "
            f"got floor={args.tactile_evidence_prob_floor} on={args.tactile_anchor_prob_on}."
        )
    if int(args.hidden_dim) % int(args.attention_heads) != 0:
        raise ValueError(
            "hidden_dim must be divisible by attention_heads; "
            f"got hidden_dim={args.hidden_dim} attention_heads={args.attention_heads}."
        )
    if int(args.semantic_dim) % int(args.attention_heads) != 0:
        raise ValueError(
            "semantic_dim must be divisible by attention_heads; "
            f"got semantic_dim={args.semantic_dim} attention_heads={args.attention_heads}."
        )
    if int(args.semantic_cross_dim) % int(args.attention_heads) != 0:
        raise ValueError(
            "semantic_cross_dim must be divisible by attention_heads; "
            f"got semantic_cross_dim={args.semantic_cross_dim} attention_heads={args.attention_heads}."
        )
    if not (0.0 <= float(args.predictive_semantic_dropout_prob) < 1.0):
        raise ValueError(
            "predictive_semantic_dropout_prob must be in [0, 1); "
            f"got {args.predictive_semantic_dropout_prob}."
        )
    if not (0.0 <= float(args.semantic_prefix_dropout_prob) < 1.0):
        raise ValueError(
            "semantic_prefix_dropout_prob must be in [0, 1); "
            f"got {args.semantic_prefix_dropout_prob}."
        )
    if str(args.device).startswith("cpu") and args.point_backbone == "sonata":
        raise RuntimeError(
            "point_backbone=sonata currently requires CUDA. "
            "Use --device cuda or switch to --point-backbone rgb."
        )
    if str(getattr(args, "action_normalization", "quantile")) != "none":
        path = getattr(args, "action_norm_stats_path", None)
        if path is None or not Path(path).expanduser().is_file():
            raise FileNotFoundError(
                "Action normalization requires a valid norm_stats.json. "
                f"Got action_norm_stats_path={path!r}."
            )
    if str(getattr(args, "prompt_state_normalization", "none")) != "none":
        path = getattr(args, "prompt_state_norm_stats_path", None)
        if path is None or not Path(path).expanduser().is_file():
            raise FileNotFoundError(
                "Prompt-state normalization requires a valid norm_stats.json. "
                f"Got prompt_state_norm_stats_path={path!r}."
            )
    if not _picf_mode_enabled(args) and str(getattr(args, "semantic_mode", "zero")) != "paligemma":
        raise ValueError(
            "picf_mode=ablated requires semantic_mode=paligemma so the PI0.5 action path remains available."
        )
    if str(getattr(args, "picf_augmentation_mode", "off")) == "multimodal_geometry":
        raise NotImplementedError(
            "picf_augmentation_mode=multimodal_geometry is reserved for a future synchronized "
            "RGB/depth/point/camera augmentation path. Use 'off' or 'photometric'."
        )
    if str(getattr(args, "picf_photometric_strength", "conservative")) not in {"conservative", "reference"}:
        raise ValueError(
            "picf_photometric_strength must be one of {'conservative', 'reference'}, "
            f"got {getattr(args, 'picf_photometric_strength', None)!r}."
        )


def _validate_backbone_args(args: argparse.Namespace) -> None:
    args.tactile_sensor_names = _parse_tactile_sensor_names(args.tactile_sensor_names)
    args.tactile_sensor_offsets_m = _parse_tactile_sensor_offsets(args.tactile_sensor_offsets_m)
    if len(args.tactile_sensor_names) != len(args.tactile_sensor_offsets_m):
        raise ValueError(
            "tactile_sensor_names and tactile_sensor_offsets_m must describe the same number of sensors. "
            f"Got {len(args.tactile_sensor_names)} names and {len(args.tactile_sensor_offsets_m)} offsets."
        )
    if args.visual_mode == "encoder":
        args.visual_checkpoint_path = args.visual_checkpoint_path or _default_vjepa_checkpoint(args.visual_model_name)
        if args.visual_checkpoint_path is None:
            raise FileNotFoundError(
                "visual_mode=encoder requires a V-JEPA checkpoint. "
                "Pass --visual-checkpoint-path or download one into checkpoints/foundation/vjepa2_1/."
            )
    if args.tactile_mode == "encoder":
        args.use_tactile = True
        args.tactile_checkpoint_path = args.tactile_checkpoint_path or _default_anytouch_checkpoint()
        args.tactile_backgrounds_path = args.tactile_backgrounds_path or _default_tactile_backgrounds_path()
        args.tactile_calibration_path = args.tactile_calibration_path or _default_tactile_calibration_path()
        args.tactile_contact_stats_path = (
            args.tactile_contact_stats_path or _default_tactile_contact_stats_path()
        )
        if args.tactile_checkpoint_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder requires an AnyTouch2 checkpoint. "
                "Pass --tactile-checkpoint-path or download checkpoint-4frames.pth into checkpoints/foundation/anytouch2/."
            )
        if args.tactile_backgrounds_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder on CALVIN requires calibrated tactile backgrounds. "
                "Pass --tactile-backgrounds-path or place tactile_backgrounds.npz under assets/calvin/."
            )
        if args.tactile_calibration_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder on CALVIN requires fingertip geometry calibration. "
                "Pass --tactile-calibration-path or place tactile_fingertip_calibration.json under assets/calvin/."
            )
        tactile_calibration_payload = _load_tactile_calibration_json(args.tactile_calibration_path)
        if tactile_calibration_payload is None:
            raise FileNotFoundError(
                f"Failed to load tactile fingertip calibration from {args.tactile_calibration_path!r}."
            )
        recommended_radius = tactile_calibration_payload.get("recommended_pt_bag_radius_m")
        if recommended_radius is not None and getattr(args, "pt_bag_radius_m_source", "") == "default":
            args.pt_bag_radius_m = float(recommended_radius)
            args.pt_bag_radius_m_source = "calibration"
        recommended_sigma = tactile_calibration_payload.get("recommended_pt_bag_sigma_m")
        if recommended_sigma is not None and getattr(args, "pt_bag_sigma_m_source", "") == "default":
            args.pt_bag_sigma_m = float(recommended_sigma)
            args.pt_bag_sigma_m_source = "calibration"
        if args.tactile_contact_stats_path is None:
            raise FileNotFoundError(
                "tactile_mode=encoder on CALVIN requires calibrated tactile contact thresholds. "
                "Pass --tactile-contact-stats-path or place tactile_contact_stats.json under assets/calvin/."
            )
        stats = _load_tactile_contact_stats_json(args.tactile_contact_stats_path)
        if stats is None:
            raise FileNotFoundError(
                f"Failed to load tactile contact stats from {args.tactile_contact_stats_path!r}."
            )
        if args.tactile_contact_tau_on is None:
            args.tactile_contact_tau_on = float(stats["tau_on"])
        if args.tactile_contact_tau_off is None:
            args.tactile_contact_tau_off = float(stats["tau_off"])
        if args.tactile_contact_temperature is None:
            temp = stats.get("temperature")
            if temp is None:
                temp = max(0.5 * (float(args.tactile_contact_tau_on) - float(args.tactile_contact_tau_off)), 1e-3)
            args.tactile_contact_temperature = float(temp)
        if float(args.tactile_contact_tau_on) <= float(args.tactile_contact_tau_off):
            raise ValueError(
                "tactile contact thresholds must satisfy tau_on > tau_off; "
                f"got tau_on={args.tactile_contact_tau_on} tau_off={args.tactile_contact_tau_off}."
            )
        if float(args.tactile_contact_temperature) <= 0.0:
            raise ValueError(
                f"tactile_contact_temperature must be > 0, got {args.tactile_contact_temperature}."
            )
    if args.point_backbone == "sonata":
        args.sonata_checkpoint_path = args.sonata_checkpoint_path or _default_sonata_checkpoint()
        if args.sonata_checkpoint_path is None:
            raise FileNotFoundError(
                "point_backbone=sonata requires a Sonata checkpoint. "
                "Pass --sonata-checkpoint-path or place SpatialLM_Sonata_encoder.pth under src/pretrain/."
            )
    if args.semantic_mode == "paligemma":
        args.semantic_checkpoint_path = args.semantic_checkpoint_path or _default_paligemma_checkpoint()
        args.semantic_checkpoint_config_path = (
            args.semantic_checkpoint_config_path or _default_paligemma_config_json(args.semantic_checkpoint_path)
        )
        if args.semantic_source not in {"auto", "hf", "pi0_pytorch"}:
            raise ValueError(
                f"semantic_source must be one of auto|hf|pi0_pytorch, got {args.semantic_source!r}."
            )
        if args.semantic_source == "pi0_pytorch" and args.semantic_checkpoint_path is None:
            raise FileNotFoundError(
                "semantic_source=pi0_pytorch requires --semantic-checkpoint-path or a detectable local pi05_base_pytorch checkpoint."
            )
        if args.semantic_source == "hf" and not args.semantic_model_name:
            raise ValueError("semantic_source=hf requires --semantic-model-name or a non-empty default model id.")
        if args.semantic_source == "auto" and args.semantic_checkpoint_path is None and not args.semantic_model_name:
            raise ValueError(
                "semantic_mode=paligemma requires either a local semantic checkpoint or a non-empty --semantic-model-name."
            )


def _setup_distributed(requested_device: str) -> tuple[bool, int, int, torch.device, _DistributedRuntimeEnv]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    use_ddp = world_size > 1
    runtime_env = _configure_distributed_runtime_env(world_size=world_size, rank=rank)
    local_rank_env = os.environ.get("LOCAL_RANK")
    if use_ddp and local_rank_env is None:
        raise RuntimeError("LOCAL_RANK must be set when running under DDP (use torchrun or set LOCAL_RANK per process).")
    local_rank = int(local_rank_env or "0")
    if str(requested_device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested device={requested_device!r}, but CUDA is not available.")
        if use_ddp:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            if ":" in requested_device:
                device = torch.device(requested_device)
            else:
                device = torch.device("cuda:0")
            torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if device.type == "cuda" else "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    return use_ddp, rank, world_size, device, runtime_env


def _is_fsdp_training(args: argparse.Namespace) -> bool:
    return str(getattr(args, "training_strategy", "ddp")).lower().replace("-", "_") == "fsdp_full_shard"


def _is_fsdp_model(model: torch.nn.Module) -> bool:
    return FullyShardedDataParallel is not None and isinstance(model, FullyShardedDataParallel)


def _unwrap_training_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    if _is_fsdp_model(model):
        return model.module
    return model


def _object_scaffold_decay_scale(args: argparse.Namespace, step: int) -> float:
    mode = str(getattr(args, "object_scaffold_decay_mode", "none")).lower()
    if mode == "none":
        return 1.0
    start = int(getattr(args, "object_scaffold_decay_start_step", 0))
    end = int(getattr(args, "object_scaffold_decay_end_step", 0))
    floor = min(max(float(getattr(args, "object_scaffold_decay_floor", 1.0)), 0.0), 1.0)
    if end <= start:
        return floor
    if step < start:
        return 1.0
    if step >= end:
        return floor
    progress = min(max(float(step - start) / float(end - start), 0.0), 1.0)
    if mode == "linear":
        return 1.0 + (floor - 1.0) * progress
    if mode == "cosine":
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))
    raise ValueError(f"Unsupported object_scaffold_decay_mode: {mode!r}")


def _scheduled_loss_config(
    base: PicfTransitionLossConfig,
    *,
    args: argparse.Namespace,
    step: int,
) -> tuple[PicfTransitionLossConfig, float]:
    """Return the per-step loss contract for weak object scaffold teachers.

    Contact-motion/tracklet sidecars are weak measurement evidence, not hard
    labels.  A schedule lets them bootstrap slot ownership early and then decay
    to a small shaping term so action learning and dense context remain the
    dominant long-run objective.
    """

    scale = _object_scaffold_decay_scale(args, step)
    if scale == 1.0:
        return base, scale
    updates = {
        field: float(getattr(base, field)) * scale
        for field in _OBJECT_SCAFFOLD_DECAY_FIELDS
    }
    return dataclasses.replace(base, **updates), scale


def _training_model_no_sync(
    model: torch.nn.Module,
    *,
    enabled: bool,
) -> contextlib.AbstractContextManager[None]:
    if enabled and callable(getattr(model, "no_sync", None)):
        return model.no_sync()
    return contextlib.nullcontext()


def _fsdp_device_id(device: torch.device) -> int | torch.device:
    if device.type == "cuda":
        return int(device.index if device.index is not None else 0)
    return device


_FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES = 512 * 1024 * 1024


def _fsdp_wrap_kwargs(*, device: torch.device) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "device_id": _fsdp_device_id(device),
        "use_orig_params": False,
        "limit_all_gathers": True,
    }
    if BackwardPrefetch is not None:
        kwargs["backward_prefetch"] = BackwardPrefetch.BACKWARD_POST
    return kwargs


def _fsdp_sharded_child_modules(model: "_PicfWindowTrainer") -> list[torch.nn.Module]:
    children: list[torch.nn.Module] = []
    core = model.core
    for module in (
        model.semantic_encoder if isinstance(model.semantic_encoder, torch.nn.Module) else None,
        getattr(core, "point_feature_extractor", None),
        getattr(core, "visual_encoder", None),
        getattr(core, "tactile_encoder", None),
        getattr(core, "token_fusion", None),
        getattr(core, "obs_self", None),
        getattr(core, "posterior_self", None),
        getattr(core, "task_self", None),
        getattr(core, "predictive_world", None),
        getattr(core, "predictive_semantic_world", None),
        getattr(core, "control_world", None),
    ):
        if module is None:
            continue
        if not isinstance(module, torch.nn.Module):
            continue
        if not any(bool(getattr(param, "requires_grad", False)) for param in module.parameters()):
            continue
        children.append(module)
    return children


def _module_has_trainable_params(module: torch.nn.Module | None) -> bool:
    if module is None:
        return False
    return any(bool(getattr(param, "requires_grad", False)) for param in module.parameters())


def _fsdp_root_ignored_modules(model: "_PicfWindowTrainer") -> list[torch.nn.Module]:
    core = model.core
    ignored: list[torch.nn.Module] = []
    ignore_core = isinstance(core, torch.nn.Module) and not _is_fsdp_model(core) and not _module_has_trainable_params(core)
    if ignore_core:
        ignored.append(core)
    for module in (
        model.semantic_encoder if isinstance(model.semantic_encoder, torch.nn.Module) else None,
        None if ignore_core else getattr(core, "point_feature_extractor", None),
        None if ignore_core else getattr(core, "visual_encoder", None),
        None if ignore_core else getattr(core, "tactile_encoder", None),
    ):
        if not isinstance(module, torch.nn.Module):
            continue
        if _is_fsdp_model(module):
            continue
        if _module_has_trainable_params(module):
            continue
        ignored.append(module)
    return ignored


def _fsdp_frozen_states_excluding_modules(
    module: torch.nn.Module,
    *,
    ignored_modules: Sequence[torch.nn.Module],
) -> list[torch.nn.Parameter]:
    """Collect frozen root-managed params that must be excluded from flat FSDP.

    `picf_trainable_scope=anchor_only` intentionally leaves a sparse set of
    anchor/router parameters trainable inside otherwise frozen PICF modules.
    FSDP with `use_orig_params=False` requires every flattened handle to have
    uniform `requires_grad`; ignoring frozen root-managed states preserves that
    diagnostic contract without relaxing the trainable allowlist.
    """

    ignored_ids = {id(mod) for mod in ignored_modules}
    frozen: list[torch.nn.Parameter] = []

    def _visit(current: torch.nn.Module) -> None:
        if id(current) in ignored_ids or _is_fsdp_model(current):
            return
        for param in current.parameters(recurse=False):
            if not bool(getattr(param, "requires_grad", False)):
                frozen.append(param)
        for child in current.children():
            _visit(child)

    _visit(module)
    return frozen


def _assign_fsdp_wrapped_child_module(
    model: "_PicfWindowTrainer",
    *,
    original: torch.nn.Module,
    wrapped: torch.nn.Module,
) -> None:
    if original is model.semantic_encoder:
        model.semantic_encoder = wrapped
        model.policy.semantic_encoder = wrapped
        return

    core = model.core
    for attr_name in (
        "point_feature_extractor",
        "visual_encoder",
        "tactile_encoder",
        "token_fusion",
        "obs_self",
        "posterior_self",
        "task_self",
        "predictive_world",
        "predictive_semantic_world",
        "control_world",
    ):
        if getattr(core, attr_name, None) is original:
            setattr(core, attr_name, wrapped)
            return

    raise RuntimeError(
        "FSDP child module enumeration returned a module that could not be reattached "
        f"to the trainer graph: type={type(original).__name__} id={id(original)}"
    )


def _promote_non_trainable_nonfloating_params_to_buffers(module: torch.nn.Module) -> None:
    for name, param in list(module.named_parameters(recurse=False)):
        if bool(getattr(param, "requires_grad", False)):
            continue
        if torch.is_floating_point(param) or torch.is_complex(param):
            continue
        delattr(module, name)
        module.register_buffer(name, param.detach(), persistent=True)
    for child in module.children():
        _promote_non_trainable_nonfloating_params_to_buffers(child)


def _collect_fsdp_managed_params_excluding_nested_fsdp(module: torch.nn.Module) -> list[torch.nn.Parameter]:
    params = list(module.parameters(recurse=False))
    for child in module.children():
        if _is_fsdp_model(child):
            continue
        params.extend(_collect_fsdp_managed_params_excluding_nested_fsdp(child))
    return params


def _fsdp_param_storage_bytes(params: Sequence[torch.nn.Parameter]) -> int:
    total = 0
    for param in params:
        total += int(param.numel()) * int(param.element_size())
    return int(total)


def _fsdp_wrap_uniform_dtype_subtrees(
    module: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    if _is_fsdp_model(module):
        return module
    if FullyShardedDataParallel is None:
        raise RuntimeError("training_strategy=fsdp_full_shard requires torch.distributed.fsdp to be available.")

    _promote_non_trainable_nonfloating_params_to_buffers(module)

    remaining_params = _collect_fsdp_managed_params_excluding_nested_fsdp(module)
    if not remaining_params:
        return module

    remaining_dtypes = {param.dtype for param in remaining_params}
    subtree_param_bytes = _fsdp_param_storage_bytes(remaining_params)
    if len(remaining_dtypes) == 1 and (
        subtree_param_bytes <= _FSDP_UNIFORM_WRAP_MAX_PARAM_BYTES or not any(True for _ in module.children())
    ):
        return FullyShardedDataParallel(module, **_fsdp_wrap_kwargs(device=device))

    for child_name, child in list(module.named_children()):
        wrapped_child = _fsdp_wrap_uniform_dtype_subtrees(child, device=device)
        if wrapped_child is not child:
            setattr(module, child_name, wrapped_child)

    remaining_params = _collect_fsdp_managed_params_excluding_nested_fsdp(module)
    if not remaining_params:
        return module

    direct_params = [param for param in module.parameters(recurse=False) if bool(getattr(param, "requires_grad", False))]
    direct_dtypes = {param.dtype for param in direct_params}
    if len(direct_dtypes) > 1:
        raise RuntimeError(
            "FSDP wrapping requires uniform trainable parameter dtype inside a single module, "
            f"but {type(module).__name__} has direct trainable params with dtypes={sorted(str(dtype) for dtype in direct_dtypes)}."
        )

    unresolved_children: list[str] = []
    for child_name, child in module.named_children():
        if _is_fsdp_model(child):
            continue
        child_dtypes = {
            param.dtype for param in _collect_fsdp_managed_params_excluding_nested_fsdp(child)
        }
        if len(child_dtypes) > 1:
            unresolved_children.append(
                f"{child_name}:{type(child).__name__}:{sorted(str(dtype) for dtype in child_dtypes)}"
            )
    if unresolved_children:
        raise RuntimeError(
            "Unable to recursively split a mixed-dtype subtree for FSDP wrapping. "
            f"module={type(module).__name__} unresolved_children={unresolved_children}"
        )

    if direct_params:
        return FullyShardedDataParallel(module, **_fsdp_wrap_kwargs(device=device))

    return module


def _fsdp_wrap_root_with_ignored_non_dominant_dtypes(
    module: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Wrap a mixed-dtype root as one FSDP boundary, ignoring minority dtypes.

    The PI0/PaliGemma semantic stack contains a hand-written dual-branch Gemma
    forward that walks language-model and action-expert layers directly instead
    of re-entering each internal submodule via ``module(...)``. Recursive FSDP
    splitting inside that stack therefore breaks parameter-view materialization:
    the custom forward can cross into nested sharded subtrees without hitting
    their pre-forward unshard hooks.

    For that specific training path we keep one semantic FSDP boundary and leave
    both frozen states and minority-dtype trainable states unsharded via
    ``ignored_states``.  The frozen-state exclusion is required for partial
    trainable scopes such as ``action_head_and_adapter`` because flat-parameter
    FSDP with ``use_orig_params=False`` requires every managed flattened handle
    to have uniform ``requires_grad``.  This preserves the custom forward while
    still full-sharding the active semantic action/adapter parameters.
    """

    if _is_fsdp_model(module):
        return module
    if FullyShardedDataParallel is None:
        raise RuntimeError("training_strategy=fsdp_full_shard requires torch.distributed.fsdp to be available.")

    _promote_non_trainable_nonfloating_params_to_buffers(module)

    managed_params = _collect_fsdp_managed_params_excluding_nested_fsdp(module)
    trainable_params = [param for param in managed_params if bool(getattr(param, "requires_grad", False))]
    if not trainable_params:
        return module

    dtype_numel: dict[torch.dtype, int] = {}
    for param in trainable_params:
        dtype_numel[param.dtype] = dtype_numel.get(param.dtype, 0) + int(param.numel())

    dominant_dtype = max(dtype_numel, key=dtype_numel.get)
    frozen_states = [param for param in managed_params if not bool(getattr(param, "requires_grad", False))]
    ignored_states = [
        *frozen_states,
        *[param for param in trainable_params if param.dtype != dominant_dtype],
    ]

    return FullyShardedDataParallel(
        module,
        ignored_states=ignored_states or None,
        **_fsdp_wrap_kwargs(device=device),
    )


def _prepare_semantic_runtime_leaf_fsdp(
    module: torch.nn.Module,
    *,
    device: torch.device,
) -> torch.nn.Module:
    """Wrap directly-called semantic hot leaves before the outer semantic root wrap.

    The PI0/PaliGemma runtime mixes standard module forwards with a custom
    dual-branch path that explicitly calls a subset of heavy child modules
    (`embed_tokens`, attention projections, MLPs, projector heads). Those
    leaves must become explicit nested FSDP modules first; otherwise the outer
    semantic root remains the only sharding boundary and the whole stack is
    materialized together.
    """

    specs_fn = getattr(module, "fsdp_runtime_leaf_module_specs", None)
    if specs_fn is None:
        return module
    for parent, child_name, mode in specs_fn():
        child = getattr(parent, child_name, None)
        if child is None or _is_fsdp_model(child):
            continue
        if mode == "uniform_recursive":
            wrapped = _fsdp_wrap_uniform_dtype_subtrees(child, device=device)
        elif mode == "mixed_root":
            wrapped = _fsdp_wrap_root_with_ignored_non_dominant_dtypes(child, device=device)
        else:
            raise ValueError(
                f"Unsupported semantic runtime leaf FSDP mode={mode!r} for child={child_name!r}."
            )
        setattr(parent, child_name, wrapped)
    return module


def _wrap_model_for_training_strategy(
    model: "_PicfWindowTrainer",
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> torch.nn.Module:
    if not _is_fsdp_training(args):
        return model
    if FullyShardedDataParallel is None:
        raise RuntimeError("training_strategy=fsdp_full_shard requires torch.distributed.fsdp to be available.")
    child_modules = _fsdp_sharded_child_modules(model)
    for child in child_modules:
        if child is model.semantic_encoder:
            child = _prepare_semantic_runtime_leaf_fsdp(child, device=device)
            wrapped = _fsdp_wrap_root_with_ignored_non_dominant_dtypes(child, device=device)
        else:
            wrapped = _fsdp_wrap_uniform_dtype_subtrees(child, device=device)
        _assign_fsdp_wrapped_child_module(model, original=child, wrapped=wrapped)
    ignored_modules = _fsdp_root_ignored_modules(model)
    root_wrap_kwargs = _fsdp_wrap_kwargs(device=device)
    if str(getattr(args, "picf_trainable_scope", "all")).lower().replace("-", "_") == "anchor_only":
        ignored_module_states = [param for module in ignored_modules for param in module.parameters()]
        ignored_frozen_states = _fsdp_frozen_states_excluding_modules(model, ignored_modules=ignored_modules)
        ignored_states_by_id: dict[int, torch.nn.Parameter] = {}
        for param in [*ignored_module_states, *ignored_frozen_states]:
            ignored_states_by_id[id(param)] = param
        ignored_states = list(ignored_states_by_id.values())
        if ignored_states:
            # PyTorch FSDP rejects mixing modules and parameters inside
            # ignored_states, so expand ignored modules to their parameters.
            root_wrap_kwargs["ignored_states"] = ignored_states
    elif ignored_modules:
        root_wrap_kwargs["ignored_modules"] = ignored_modules
    return FullyShardedDataParallel(model, **root_wrap_kwargs)


def _cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _distributed_barrier(*, use_ddp: bool, device: torch.device) -> None:
    if not use_ddp or not dist.is_initialized():
        return
    if device.type == "cuda" and device.index is not None:
        dist.barrier(device_ids=[device.index])
    else:
        dist.barrier()


def _is_main(rank: int) -> bool:
    return rank == 0


def _seed_everything(seed: int, rank: int) -> None:
    mixed = int(seed) + (1009 * int(rank))
    random.seed(mixed)
    np.random.seed(mixed)
    torch.manual_seed(mixed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(mixed)


def _step_indexed_window_rng(
    *,
    seed: int,
    rank: int,
    step: int,
    micro_step: int,
    retry_count: int = 0,
) -> np.random.Generator:
    """Return a deterministic sampler RNG keyed by the global optimizer step.

    Resume safety matters for PICF action-loss analysis: if a run resumes from
    step 7000 but reuses a stateful RNG initialized only from `seed + rank`, it
    replays the same early sampled-window stream and can make train-window loss
    look like a model rebound.  Keying the RNG by global step keeps sampling
    deterministic while making continuation semantics independent of whether
    the process reached the step in one run or via checkpoint resume.
    """

    words = (
        int(seed) & 0xFFFFFFFF,
        int(rank) & 0xFFFFFFFF,
        int(step) & 0xFFFFFFFF,
        int(micro_step) & 0xFFFFFFFF,
        int(retry_count) & 0xFFFFFFFF,
        0xA7B5_2026,
    )
    return np.random.default_rng(np.random.SeedSequence(words))


def _calvin_prompt_bucket(prompt: str) -> str:
    """Coarse CALVIN task family used only for sampler balancing/trace.

    The bucket is intentionally derived from language metadata rather than
    model outputs.  It does not enter the model and cannot leak action labels;
    it only prevents a long run from issuing many consecutive optimizer
    updates dominated by one task family.
    """

    text = str(prompt).strip().lower().replace("_", " ")
    if not text:
        return "other"
    if "drawer" in text:
        return "drawer"
    if "button" in text or "switch" in text or "light" in text or "led" in text:
        return "switch_button_light"
    if "slider" in text or "slide" in text:
        return "slider"
    if "push" in text and "block" in text:
        return "block_push"
    if ("lift" in text or "grasp" in text or "pick" in text) and "block" in text:
        return "block_lift"
    if "block" in text:
        return "block_other"
    return "other"


_CALVIN_BUCKET_SAMPLING_MODES = {"round_robin", "task_uniform", "trajectory", "temperature"}


def _normalize_bucket_sampling_mode(value: str) -> str:
    mode = str(value or "round_robin").strip().lower().replace("-", "_")
    if mode not in _CALVIN_BUCKET_SAMPLING_MODES:
        raise ValueError(
            "calvin bucket sampling mode must be one of "
            f"{sorted(_CALVIN_BUCKET_SAMPLING_MODES)}, got {value!r}."
        )
    return mode


def _parse_bucket_weight_spec(spec: str | None) -> dict[str, float]:
    """Parse a strict VLA-Foundry-style bucket ratio specification.

    Format: ``bucket=weight,bucket=weight``.  Keys may be exact bucket names,
    sanitized metric fragments, glob patterns, or ``*`` for the default weight.
    """

    text = str(spec or "").strip()
    if not text:
        return {}
    result: dict[str, float] = {}
    for raw_part in re.split(r"[,;]", text):
        part = raw_part.strip()
        if not part:
            continue
        if "=" in part:
            key, raw_value = part.split("=", 1)
        elif ":" in part:
            key, raw_value = part.split(":", 1)
        else:
            raise ValueError(
                "Bucket weight spec entries must use key=value or key:value; "
                f"got {part!r} in {spec!r}."
            )
        key = key.strip()
        if not key:
            raise ValueError(f"Bucket weight spec contains an empty key in {spec!r}.")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"Invalid bucket weight {raw_value!r} for key {key!r}.") from exc
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"Bucket weight for {key!r} must be finite and non-negative, got {value}.")
        result[key] = value
    return result


def _compute_bucket_sampling_weights(
    *,
    bucket_names: Sequence[str],
    bucket_sizes: Mapping[str, int],
    mode: str,
    temperature_alpha: float,
    weight_spec: str | None,
) -> dict[str, float]:
    """Return normalized target q_b for CALVIN task-bucket sampling."""

    names = [str(name) for name in bucket_names]
    if not names:
        return {}
    mode = _normalize_bucket_sampling_mode(mode)
    alpha = float(temperature_alpha)
    if not math.isfinite(alpha):
        raise ValueError(f"calvin_bucket_temperature_alpha must be finite, got {temperature_alpha}.")
    overrides = _parse_bucket_weight_spec(weight_spec)
    if overrides:
        default_weight = overrides.get("*", overrides.get("__default__"))
        weights: dict[str, float] = {}
        for name in names:
            matched: list[float] = []
            for key, value in overrides.items():
                if key in {"*", "__default__"}:
                    continue
                if key == name or key == _metric_key_fragment(name) or fnmatch.fnmatch(name, key):
                    matched.append(float(value))
            if matched:
                weights[name] = float(matched[-1])
            elif default_weight is not None:
                weights[name] = float(default_weight)
            else:
                raise ValueError(
                    "Bucket weight spec did not cover bucket "
                    f"{name!r}. Add an explicit entry or '*=<weight>'."
                )
    else:
        if mode in {"round_robin", "task_uniform"}:
            weights = {name: 1.0 for name in names}
        elif mode == "trajectory":
            weights = {name: float(max(int(bucket_sizes.get(name, 0)), 0)) for name in names}
        else:
            weights = {
                name: float(max(int(bucket_sizes.get(name, 0)), 0)) ** max(alpha, 0.0)
                for name in names
            }
    total = float(sum(max(float(value), 0.0) for value in weights.values()))
    if total <= 0.0:
        raise ValueError(
            "CALVIN bucket sampling weights sum to zero. "
            f"mode={mode!r} alpha={alpha} spec={weight_spec!r} sizes={dict(bucket_sizes)!r}."
        )
    return {name: max(float(weights[name]), 0.0) / total for name in names}


def _normalize_bucket_weights_with_floor_cap(
    raw_weights: Mapping[str, float],
    *,
    floor: float,
    cap: float,
) -> dict[str, float]:
    """Normalize bucket weights under per-bucket floor/cap constraints."""

    names = [str(name) for name in raw_weights]
    if not names:
        return {}
    n = len(names)
    floor = max(float(floor), 0.0)
    cap = max(float(cap), 0.0)
    if floor * float(n) > 1.0:
        floor = 1.0 / float(n)
    if cap * float(n) < 1.0:
        cap = 1.0 / float(n)
    values = {
        name: max(float(raw_weights.get(name, 0.0)), 0.0)
        for name in names
    }
    if sum(values.values()) <= 0.0:
        values = {name: 1.0 for name in names}

    result = {name: floor for name in names}
    remaining = max(1.0 - floor * float(n), 0.0)
    free = set(names)
    while free and remaining > 1e-12:
        free_total = float(sum(values[name] for name in free))
        if free_total <= 0.0:
            share = remaining / float(len(free))
            proposal = {name: result[name] + share for name in free}
        else:
            proposal = {
                name: result[name] + remaining * float(values[name]) / free_total
                for name in free
            }
        capped = [name for name, value in proposal.items() if value > cap]
        if not capped:
            for name, value in proposal.items():
                result[name] = float(value)
            remaining = 0.0
            break
        for name in capped:
            result[name] = float(cap)
            free.remove(name)
        remaining = max(1.0 - float(sum(result.values())), 0.0)
    total = float(sum(result.values()))
    if total <= 0.0:
        return {name: 1.0 / float(n) for name in names}
    return {name: float(value) / total for name, value in result.items()}


def _zscore_by_name(values: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    data = np.asarray([float(values.get(str(name), 0.0)) for name in names], dtype=np.float64)
    if data.size == 0:
        return {}
    mean = float(np.mean(data))
    std = float(np.std(data))
    if std <= 1e-12 or not math.isfinite(std):
        return {str(name): 0.0 for name in names}
    return {str(name): float((float(values.get(str(name), 0.0)) - mean) / std) for name in names}


def _dynamic_bucket_sampling_weights(
    *,
    bucket_names: Sequence[str],
    base_weights: Mapping[str, float],
    loss_ema: Mapping[str, float],
    previous_loss_ema: Mapping[str, float],
    counts: Mapping[str, int],
    step: int,
    warmup_steps: int,
    min_count: int,
    eta: float,
    gamma: float,
    clip: float,
    min_mass_fraction: float,
    max_weight: float,
) -> tuple[dict[str, float], dict[str, float | int | bool]]:
    """Return bounded PiKE-style adaptive task-bucket weights.

    The update only changes the sampler target distribution.  It does not
    change labels, loss definitions, action targets, or PICF semantics.
    """

    names = [str(name) for name in bucket_names]
    if not names:
        return {}, {"logical_batch_dynamic_mixing_enabled": False}
    base = _normalize_bucket_weights_with_floor_cap(
        {name: max(float(base_weights.get(name, 0.0)), 0.0) for name in names},
        floor=0.0,
        cap=1.0,
    )
    if int(step) < int(warmup_steps):
        info: dict[str, float | int | bool] = {
            "logical_batch_dynamic_mixing_enabled": True,
            "logical_batch_dynamic_mixing_active": False,
            "logical_batch_dynamic_mixing_reason_id": 1,
            "logical_batch_dynamic_mixing_observed_bucket_count": int(
                sum(1 for name in names if int(counts.get(name, 0)) > 0)
            ),
        }
        return base, info
    valid_names = [
        name
        for name in names
        if int(counts.get(name, 0)) >= int(min_count) and float(loss_ema.get(name, 0.0)) > 0.0
    ]
    if len(valid_names) < 2:
        info = {
            "logical_batch_dynamic_mixing_enabled": True,
            "logical_batch_dynamic_mixing_active": False,
            "logical_batch_dynamic_mixing_reason_id": 2,
            "logical_batch_dynamic_mixing_observed_bucket_count": int(len(valid_names)),
        }
        return base, info

    loss_values = {name: float(loss_ema.get(name, 0.0)) for name in valid_names}
    progress_values = {
        name: float(previous_loss_ema.get(name, loss_ema.get(name, 0.0))) - float(loss_ema.get(name, 0.0))
        for name in valid_names
    }
    loss_z = _zscore_by_name(loss_values, valid_names)
    progress_z = _zscore_by_name(progress_values, valid_names)
    eta = max(float(eta), 0.0)
    gamma = float(gamma)
    clip = max(float(clip), 0.0)
    raw: dict[str, float] = {}
    lag_values: dict[str, float] = {}
    for name in names:
        if name in valid_names:
            lag = float(loss_z.get(name, 0.0)) - gamma * float(progress_z.get(name, 0.0))
            if clip > 0.0:
                lag = min(max(lag, -clip), clip)
            lag_values[name] = float(lag)
            raw[name] = float(base.get(name, 0.0)) * math.exp(eta * lag)
        else:
            lag_values[name] = 0.0
            raw[name] = float(base.get(name, 0.0))
    q_min = max(float(min_mass_fraction), 0.0) / float(len(names))
    q_max = float(max_weight)
    weights = _normalize_bucket_weights_with_floor_cap(raw, floor=q_min, cap=q_max)
    info = {
        "logical_batch_dynamic_mixing_enabled": True,
        "logical_batch_dynamic_mixing_active": True,
        "logical_batch_dynamic_mixing_reason_id": 0,
        "logical_batch_dynamic_mixing_observed_bucket_count": int(len(valid_names)),
        "logical_batch_dynamic_mixing_eta": float(eta),
        "logical_batch_dynamic_mixing_gamma": float(gamma),
        "logical_batch_dynamic_mixing_clip": float(clip),
        "logical_batch_dynamic_mixing_min_weight": float(min(weights.values())),
        "logical_batch_dynamic_mixing_max_weight": float(max(weights.values())),
    }
    for name in names:
        fragment = _metric_key_fragment(name)
        info[f"logical_batch_dynamic_weight_{fragment}"] = float(weights.get(name, 0.0))
        info[f"logical_batch_dynamic_lag_{fragment}"] = float(lag_values.get(name, 0.0))
        info[f"logical_batch_dynamic_loss_ema_{fragment}"] = float(loss_ema.get(name, 0.0))
        info[f"logical_batch_dynamic_count_{fragment}"] = int(counts.get(name, 0))
    return weights, info


def _bucket_sequence_for_logical_step(
    *,
    bucket_names: Sequence[str],
    target_bucket_weights: Mapping[str, float],
    mode: str,
    weight_spec: str | None,
    seed: int,
    step: int,
    world_size: int,
    accum_steps: int,
    without_replacement: bool = True,
) -> tuple[str, ...]:
    """Return the global bucket sequence for one optimizer step.

    The sequence length is ``world_size * accum_steps`` and is shared by every
    rank.  Each rank then takes its rank/micro-step slice.  This implements the
    logical-batch contract from VLA-style data mixing: bucket choice is decided
    once per optimizer step, while the concrete segment/window inside each
    bucket remains independently randomized.
    """

    names = tuple(str(name) for name in bucket_names)
    if not names:
        return ()
    global_micro_count = max(int(world_size), 1) * max(int(accum_steps), 1)
    if global_micro_count <= 0:
        return ()

    normalized_mode = _normalize_bucket_sampling_mode(mode)
    if normalized_mode == "round_robin" and not str(weight_spec or "").strip():
        base = int(step) * global_micro_count
        return tuple(names[int(base + offset) % len(names)] for offset in range(global_micro_count))

    raw_weights = np.asarray(
        [max(float(target_bucket_weights.get(str(name), 0.0)), 0.0) for name in names],
        dtype=np.float64,
    )
    positive = raw_weights > 0.0
    if not bool(np.any(positive)):
        raise ValueError(
            "Bucket sampling target weights contain no positive mass: "
            f"names={names!r} weights={dict(target_bucket_weights)!r}."
        )

    rng = np.random.default_rng(
        np.random.SeedSequence(
            (
                int(seed) & 0xFFFFFFFF,
                int(step) & 0xFFFFFFFF,
                int(global_micro_count) & 0xFFFFFFFF,
                0xB00C_2026,
            )
        )
    )
    eligible_names = np.asarray([names[index] for index, keep in enumerate(positive) if bool(keep)], dtype=object)
    eligible_weights = raw_weights[positive]
    eligible_weights = eligible_weights / float(np.sum(eligible_weights))

    if not bool(without_replacement):
        return tuple(
            str(bucket)
            for bucket in rng.choice(
                eligible_names,
                size=int(global_micro_count),
                replace=True,
                p=eligible_weights,
            )
        )

    sequence: list[str] = []
    while len(sequence) < int(global_micro_count):
        take = min(int(global_micro_count) - len(sequence), int(len(eligible_names)))
        chosen = rng.choice(
            eligible_names,
            size=int(take),
            replace=False,
            p=eligible_weights,
        )
        sequence.extend(str(bucket) for bucket in np.asarray(chosen, dtype=object).reshape(-1))
    return tuple(sequence)


def _reduce_mean(value: float, *, device: torch.device, world_size: int) -> float:
    if world_size <= 1:
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= float(world_size)
    return float(tensor.item())


def _reduce_sum(value: float, *, device: torch.device, world_size: int) -> float:
    if world_size <= 1:
        return float(value)
    tensor = torch.tensor([value], device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return float(tensor.item())


def _all_gather_python_object(value: Any, *, use_ddp: bool, world_size: int) -> list[Any]:
    if not bool(use_ddp) or int(world_size) <= 1:
        return [value]
    gathered: list[Any] = [None for _ in range(int(world_size))]
    dist.all_gather_object(gathered, value)
    return gathered


def _metric_key_fragment(value: str) -> str:
    fragment = re.sub(r"[^0-9a-zA-Z]+", "_", str(value).strip().lower())
    fragment = fragment.strip("_")
    return fragment or "unknown"


def _logical_batch_loss_scales(
    local_buckets: Sequence[str],
    *,
    enabled: bool,
    use_ddp: bool,
    world_size: int,
    target_bucket_weights: Mapping[str, float] | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Return local backward scales for a task-balanced logical optimizer step.

    DDP averages gradients across ranks.  To realize a global estimator weight
    `w_m` for each selected micro-window, each rank must call backward with
    `world_size * w_m`.
    """

    local_buckets = [str(bucket) for bucket in local_buckets]
    if not local_buckets:
        return [], {
            "logical_batch_enabled": bool(enabled),
            "logical_batch_global_micro_count": 0,
            "logical_batch_distinct_bucket_count": 0,
        }
    if not bool(enabled):
        scale = 1.0 / float(len(local_buckets))
        return [scale for _ in local_buckets], {
            "logical_batch_enabled": False,
            "logical_batch_global_micro_count": int(len(local_buckets) * max(int(world_size), 1)),
            "logical_batch_distinct_bucket_count": 0,
        }

    gathered = _all_gather_python_object(local_buckets, use_ddp=use_ddp, world_size=int(world_size))
    global_buckets: list[str] = []
    for item in gathered:
        global_buckets.extend(str(bucket) for bucket in (item or []))
    bucket_counts: dict[str, int] = {}
    for bucket in global_buckets:
        bucket_counts[bucket] = int(bucket_counts.get(bucket, 0)) + 1
    distinct_buckets = sorted(bucket_counts)
    if not distinct_buckets:
        scale = 1.0 / float(len(local_buckets))
        return [scale for _ in local_buckets], {
            "logical_batch_enabled": True,
            "logical_batch_global_micro_count": int(len(global_buckets)),
            "logical_batch_distinct_bucket_count": 0,
        }

    raw_target_weights = {str(key): float(value) for key, value in (target_bucket_weights or {}).items()}
    if raw_target_weights:
        selected_target_total = float(
            sum(max(raw_target_weights.get(str(bucket), 0.0), 0.0) for bucket in distinct_buckets)
        )
        if selected_target_total <= 0.0:
            raise ValueError(
                "Logical-batch target weights assign zero total mass to selected buckets: "
                f"selected={distinct_buckets!r} target={raw_target_weights!r}."
            )
        bucket_weights = {
            str(bucket): max(raw_target_weights.get(str(bucket), 0.0), 0.0) / selected_target_total
            for bucket in distinct_buckets
        }
    else:
        uniform_bucket_weight = 1.0 / float(len(distinct_buckets))
        bucket_weights = {str(bucket): uniform_bucket_weight for bucket in distinct_buckets}
    ddp_multiplier = float(max(int(world_size), 1))
    scales = [
        ddp_multiplier * float(bucket_weights[str(bucket)]) / float(max(bucket_counts[str(bucket)], 1))
        for bucket in local_buckets
    ]
    return scales, {
        "logical_batch_enabled": True,
        "logical_batch_global_micro_count": int(len(global_buckets)),
        "logical_batch_distinct_bucket_count": int(len(distinct_buckets)),
        "logical_batch_bucket_counts": {str(bucket): int(count) for bucket, count in sorted(bucket_counts.items())},
        "logical_batch_bucket_target_weights": {
            str(bucket): float(bucket_weights[str(bucket)]) for bucket in sorted(bucket_weights)
        },
    }


def _logical_action_bucket_scale(
    bucket: str,
    *,
    ema: Mapping[str, float],
    counts: Mapping[str, int],
    min_count: int,
    scale_min: float,
    scale_max: float,
) -> float:
    """Return a bounded action-gradient scale for one task bucket.

    The displayed action metrics remain unscaled.  This scale only changes the
    backward action component so high-loss/noisy buckets do not dominate Adam
    momentum when each physical step has few tasks.
    """

    bucket = str(bucket)
    bucket_count = int(counts.get(bucket, 0))
    bucket_ema = float(ema.get(bucket, 0.0))
    if bucket_count < int(min_count) or bucket_ema <= 0.0:
        return 1.0
    valid = [
        float(value)
        for key, value in ema.items()
        if int(counts.get(str(key), 0)) >= int(min_count) and float(value) > 0.0
    ]
    if not valid:
        return 1.0
    target = float(sum(valid) / float(len(valid)))
    if target <= 0.0:
        return 1.0
    scale = target / bucket_ema
    return float(min(max(scale, float(scale_min)), float(scale_max)))


def _update_logical_action_bucket_ema(
    ema: MutableMapping[str, float],
    counts: MutableMapping[str, int],
    *,
    bucket: str,
    value: float,
    decay: float,
    bucket_names: Sequence[str],
    use_ddp: bool,
    device: torch.device,
) -> None:
    """Update per-bucket action-loss EMA with rank-synchronized observations."""

    bucket_names = [str(name) for name in bucket_names]
    if not bucket_names:
        return
    bucket_to_index = {str(name): idx for idx, name in enumerate(bucket_names)}
    if str(bucket) not in bucket_to_index:
        return
    payload = torch.zeros((len(bucket_names), 2), device=device, dtype=torch.float32)
    idx = int(bucket_to_index[str(bucket)])
    payload[idx, 0] = float(value)
    payload[idx, 1] = 1.0
    if bool(use_ddp):
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    decay = float(decay)
    for name, (loss_sum, count_sum) in zip(bucket_names, payload.detach().cpu().tolist(), strict=True):
        count = int(round(float(count_sum)))
        if count <= 0:
            continue
        mean_value = float(loss_sum) / float(count)
        previous_count = int(counts.get(str(name), 0))
        previous_ema = float(ema.get(str(name), mean_value))
        ema[str(name)] = float(decay * previous_ema + (1.0 - decay) * mean_value) if previous_count > 0 else mean_value
        counts[str(name)] = previous_count + count


def _optimizer_owner_group_name(name: str) -> str:
    canonical = _canonical_param_owner_name(name)
    if canonical.startswith("core."):
        return "picf_core"
    if canonical.startswith("semantic_encoder."):
        return "semantic"
    return "policy_head"


def _logical_batch_gradient_surgery_params(
    model: torch.nn.Module,
    *,
    groups: Sequence[str],
) -> list[torch.nn.Parameter]:
    requested = {str(group).strip() for group in groups if str(group).strip()}
    if not requested:
        return []
    params: list[torch.nn.Parameter] = []
    seen: set[int] = set()
    for name, param in model.named_parameters():
        if not bool(getattr(param, "requires_grad", False)):
            continue
        if isinstance(param, UninitializedParameter):
            continue
        if _optimizer_owner_group_name(name) not in requested:
            continue
        if id(param) in seen:
            continue
        params.append(param)
        seen.add(id(param))
    return params


def _logical_batch_autograd_grads(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
) -> list[torch.Tensor | None]:
    if not params:
        return []
    grads = torch.autograd.grad(
        loss,
        list(params),
        retain_graph=True,
        allow_unused=True,
    )
    return [None if grad is None else grad.detach().clone() for grad in grads]


def _grad_list_dot(a: Sequence[torch.Tensor | None], b: Sequence[torch.Tensor | None]) -> torch.Tensor:
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    for tensor in list(a) + list(b):
        if tensor is not None:
            device = tensor.device
            dtype = tensor.dtype
            break
    if device is None:
        return torch.zeros((), dtype=torch.float32)
    total = torch.zeros((), device=device, dtype=dtype or torch.float32)
    for ga, gb in zip(a, b, strict=True):
        if ga is None or gb is None:
            continue
        total = total + torch.sum(ga * gb)
    return total


def _pcgrad_project_and_sum(
    grad_lists: Sequence[Sequence[torch.Tensor | None]],
    *,
    eps: float = 1e-12,
) -> list[torch.Tensor | None]:
    """Return the summed PCGrad gradient over local task/micro gradients."""

    if not grad_lists:
        return []
    projected: list[list[torch.Tensor | None]] = [
        [None if grad is None else grad.clone() for grad in grads]
        for grads in grad_lists
    ]
    references: list[list[torch.Tensor | None]] = [
        [None if grad is None else grad.detach() for grad in grads]
        for grads in grad_lists
    ]
    for idx, grads in enumerate(projected):
        for ref_idx, ref in enumerate(references):
            if idx == ref_idx:
                continue
            dot = _grad_list_dot(grads, ref)
            if float(dot.detach().cpu().item()) >= 0.0:
                continue
            denom = _grad_list_dot(ref, ref).detach()
            if float(denom.detach().cpu().item()) <= float(eps):
                continue
            coeff = dot / (denom + float(eps))
            for param_idx, (grad, ref_grad) in enumerate(zip(grads, ref, strict=True)):
                if grad is None or ref_grad is None:
                    continue
                grads[param_idx] = grad - coeff.to(device=grad.device, dtype=grad.dtype) * ref_grad
    combined: list[torch.Tensor | None] = []
    for param_idx in range(len(projected[0])):
        total: torch.Tensor | None = None
        for grads in projected:
            grad = grads[param_idx]
            if grad is None:
                continue
            total = grad.clone() if total is None else total + grad
        combined.append(total)
    return combined


def _project_simplex_vector(vector: torch.Tensor) -> torch.Tensor:
    """Project a 1-D tensor onto the probability simplex."""

    if vector.ndim != 1:
        raise ValueError(f"Expected a 1-D tensor for simplex projection, got shape={tuple(vector.shape)}.")
    if vector.numel() == 0:
        return vector
    sorted_values, _ = torch.sort(vector, descending=True)
    cssv = torch.cumsum(sorted_values, dim=0) - 1.0
    ranks = torch.arange(1, int(vector.numel()) + 1, device=vector.device, dtype=vector.dtype)
    support = sorted_values - cssv / ranks > 0
    if not bool(torch.any(support).item()):
        return torch.full_like(vector, 1.0 / float(vector.numel()))
    rho = int(torch.nonzero(support, as_tuple=False)[-1].item())
    theta = cssv[rho] / float(rho + 1)
    return torch.clamp(vector - theta, min=0.0)


def _gradient_surgery_gram_matrix(
    grad_lists: Sequence[Sequence[torch.Tensor | None]],
) -> torch.Tensor:
    """Return the CPU float64 Gram matrix for local task/micro gradients."""

    task_count = int(len(grad_lists))
    gram = torch.zeros((task_count, task_count), dtype=torch.float64)
    for row in range(task_count):
        for col in range(row, task_count):
            value = _grad_list_dot(grad_lists[row], grad_lists[col]).detach().to(device="cpu", dtype=torch.float64)
            gram[row, col] = value
            gram[col, row] = value
    return gram


def _combine_grad_lists_with_coefficients(
    grad_lists: Sequence[Sequence[torch.Tensor | None]],
    coefficients: Sequence[float],
) -> list[torch.Tensor | None]:
    if not grad_lists:
        return []
    if len(grad_lists) != len(coefficients):
        raise ValueError("Gradient list / coefficient length mismatch.")
    combined: list[torch.Tensor | None] = []
    for param_idx in range(len(grad_lists[0])):
        total: torch.Tensor | None = None
        for grads, coeff in zip(grad_lists, coefficients, strict=True):
            grad = grads[param_idx]
            if grad is None:
                continue
            scaled = grad * float(coeff)
            total = scaled.clone() if total is None else total + scaled
        combined.append(total)
    return combined


def _cagrad_project_and_sum(
    grad_lists: Sequence[Sequence[torch.Tensor | None]],
    *,
    alpha: float = 0.4,
    iters: int = 20,
    eps: float = 1e-12,
    rescale_to_raw_norm: bool = True,
) -> list[torch.Tensor | None]:
    """Return a scoped CAGrad update over local task/micro gradients.

    This follows the CAGrad dual form over the simplex:

        min_w <g_w, g_0> + alpha * ||g_0|| * ||g_w||

    where g_0 is the raw logical-batch gradient and g_w is a convex
    combination of local task gradients.  The optional norm-preserving rescale
    keeps the adapter update magnitude comparable to the raw logical-batch
    gradient, which is important here because CAGrad is used as a scoped
    bridge/action conflict diagnostic rather than a whole-model optimizer.
    """

    if not grad_lists:
        return []
    task_count = int(len(grad_lists))
    if task_count == 1 or float(alpha) <= 0.0:
        return _combine_grad_lists_with_coefficients(grad_lists, [1.0] * task_count)

    gram = _gradient_surgery_gram_matrix(grad_lists)
    raw_coeff = torch.ones((task_count,), dtype=torch.float64)
    raw_norm_sq = torch.clamp(raw_coeff @ gram @ raw_coeff, min=0.0)
    raw_norm = torch.sqrt(raw_norm_sq + float(eps))
    if not bool(torch.isfinite(raw_norm).item()) or float(raw_norm.item()) <= float(eps):
        return _combine_grad_lists_with_coefficients(grad_lists, [1.0] * task_count)

    weights = torch.full((task_count,), 1.0 / float(task_count), dtype=torch.float64)
    # A small, deterministic projected-gradient solver is sufficient because
    # K == logical micro count is small in all production gates.
    lipschitz = float(torch.linalg.matrix_norm(gram, ord=2).item()) + float(eps)
    step_size = 1.0 / (lipschitz * (1.0 + float(alpha)) + float(eps))
    for _ in range(max(1, int(iters))):
        gw_norm = torch.sqrt(torch.clamp(weights @ gram @ weights, min=0.0) + float(eps))
        grad_obj = gram @ raw_coeff + float(alpha) * raw_norm * (gram @ weights) / gw_norm
        weights = _project_simplex_vector(weights - step_size * grad_obj)

    gw_norm = torch.sqrt(torch.clamp(weights @ gram @ weights, min=0.0) + float(eps))
    if not bool(torch.isfinite(gw_norm).item()) or float(gw_norm.item()) <= float(eps):
        return _combine_grad_lists_with_coefficients(grad_lists, [1.0] * task_count)
    cagrad_coeff = raw_coeff + float(alpha) * float(raw_norm.item() / gw_norm.item()) * weights
    if bool(rescale_to_raw_norm):
        update_norm_sq = torch.clamp(cagrad_coeff @ gram @ cagrad_coeff, min=0.0)
        update_norm = torch.sqrt(update_norm_sq + float(eps))
        if bool(torch.isfinite(update_norm).item()) and float(update_norm.item()) > float(eps):
            cagrad_coeff = cagrad_coeff * float(raw_norm.item() / update_norm.item())
    return _combine_grad_lists_with_coefficients(
        grad_lists,
        [float(value) for value in cagrad_coeff.tolist()],
    )


def _assign_and_sync_gradient_surgery_grads(
    params: Sequence[torch.nn.Parameter],
    grads: Sequence[torch.Tensor | None],
    *,
    use_ddp: bool,
    world_size: int,
) -> None:
    if len(params) != len(grads):
        raise ValueError("Gradient surgery parameter/gradient length mismatch.")
    for param, grad in zip(params, grads, strict=True):
        if grad is None:
            param.grad = None
            continue
        assigned = grad.to(device=param.device, dtype=param.dtype)
        if bool(use_ddp):
            dist.all_reduce(assigned, op=dist.ReduceOp.SUM)
            assigned = assigned / float(max(int(world_size), 1))
        param.grad = assigned


def _grad_norm(parameters: Iterator[torch.nn.Parameter]) -> float:
    sq_sum = 0.0
    found = False
    for param in parameters:
        if param.grad is None:
            continue
        grad = param.grad.detach()
        sq_sum += float(torch.sum(grad * grad).item())
        found = True
    return float(math.sqrt(sq_sum)) if found else 0.0


def _fsdp_global_grad_l2_norm(model: torch.nn.Module) -> float:
    local_sq: torch.Tensor | None = None
    for param in model.parameters():
        if param.grad is None:
            continue
        grad = param.grad.detach()
        grad64 = grad.to(dtype=torch.float64)
        grad_sq = torch.sum(grad64 * grad64)
        if local_sq is None:
            local_sq = grad_sq
        else:
            local_sq = local_sq + grad_sq
    if local_sq is None:
        return 0.0
    if dist.is_initialized():
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    return float(torch.sqrt(local_sq).item())


def _fsdp_clip_grad_norm_exact(model: torch.nn.Module, *, max_norm: float, eps: float = 1e-6) -> float:
    total_norm = _fsdp_global_grad_l2_norm(model)
    if total_norm <= 0.0:
        return 0.0
    clip_coef = float(max_norm) / (float(total_norm) + float(eps))
    if clip_coef >= 1.0:
        return float(total_norm)
    with torch.no_grad():
        for param in model.parameters():
            if param.grad is None:
                continue
            param.grad.mul_(clip_coef)
    return float(total_norm)


def _grad_norm_for_training_model(model: torch.nn.Module) -> float:
    if _is_fsdp_model(model):
        return _fsdp_global_grad_l2_norm(model)
    return _grad_norm(model.parameters())


def _clip_grad_norm_for_training_model(model: torch.nn.Module, *, max_norm: float) -> float:
    if _is_fsdp_model(model):
        return _fsdp_clip_grad_norm_exact(model, max_norm=float(max_norm))
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(max_norm)))


def _postclip_grad_norm_for_logging(
    *,
    preclip_grad_norm: float,
    grad_clip_threshold: float | None,
    grad_clip_applied: bool,
) -> float:
    """Return the post-clip norm estimate without issuing another distributed grad scan."""

    preclip = float(preclip_grad_norm)
    if not bool(grad_clip_applied) or grad_clip_threshold is None:
        return preclip
    return float(min(preclip, float(grad_clip_threshold)))


@dataclasses.dataclass
class _GradClipController:
    mode: str
    fixed_norm: float
    percentile: float
    window: int
    history: deque[float] = dataclasses.field(default_factory=deque)

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "_GradClipController":
        return cls(
            mode=str(getattr(args, "grad_clip_mode", "fixed")).lower(),
            fixed_norm=float(getattr(args, "grad_clip_norm", 0.0)),
            percentile=float(getattr(args, "grad_clip_percentile", 75.0)),
            window=int(getattr(args, "grad_clip_window", 100)),
            history=deque(maxlen=int(getattr(args, "grad_clip_window", 100))),
        )

    def threshold(self) -> float | None:
        if self.mode == "fixed":
            return float(self.fixed_norm) if float(self.fixed_norm) > 0.0 else None
        if len(self.history) < int(self.window):
            return None
        return float(np.percentile(np.asarray(self.history, dtype=np.float32), float(self.percentile)))

    def history_size(self) -> int:
        return int(len(self.history))

    def observe(self, preclip_grad_norm: float) -> None:
        self.history.append(float(preclip_grad_norm))

    def state_dict(self) -> dict[str, Any]:
        return {
            "mode": str(self.mode),
            "fixed_norm": float(self.fixed_norm),
            "percentile": float(self.percentile),
            "window": int(self.window),
            "history": [float(value) for value in self.history],
        }

    def load_state_dict(self, payload: dict[str, Any] | None) -> bool:
        if not payload:
            return False
        payload_mode = str(payload.get("mode", "")).lower()
        if payload_mode != str(self.mode):
            return False
        if int(payload.get("window", -1)) != int(self.window):
            return False
        if payload_mode == "percentile":
            if not math.isclose(
                float(payload.get("percentile", math.nan)),
                float(self.percentile),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                return False
        elif payload_mode == "fixed":
            if not math.isclose(
                float(payload.get("fixed_norm", math.nan)),
                float(self.fixed_norm),
                rel_tol=0.0,
                abs_tol=1e-6,
            ):
                return False
        self.history.clear()
        for value in payload.get("history", ()):
            self.history.append(float(value))
        return True


def _optimizer_param_group_lookup(optimizer: torch.optim.Optimizer | None) -> dict[int, str]:
    if optimizer is None:
        return {}
    lookup: dict[int, str] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        for param in group.get("params", ()):
            lookup[id(param)] = name
    return lookup


def _optimizer_group_grad_metrics(optimizer: torch.optim.Optimizer | None) -> dict[str, float]:
    if optimizer is None:
        return {}
    metrics: dict[str, float] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("name", f"group_{index}"))
        safe_name = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_") or f"group_{index}"
        sq_sum = 0.0
        max_abs = 0.0
        elem_count = 0
        param_count = 0
        for param in group.get("params", ()):
            if isinstance(param, UninitializedParameter) or param.grad is None:
                continue
            grad = torch.nan_to_num(param.grad.detach().to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            sq_sum += float(torch.sum(grad.square()).item())
            max_abs = max(max_abs, float(grad.abs().max().item()) if grad.numel() > 0 else 0.0)
            elem_count += int(grad.numel())
            param_count += 1
        metrics[f"grad_norm_group_{safe_name}"] = float(math.sqrt(max(sq_sum, 0.0)))
        metrics[f"grad_absmax_group_{safe_name}"] = float(max_abs)
        metrics[f"grad_param_tensors_group_{safe_name}"] = float(param_count)
        metrics[f"grad_elements_group_{safe_name}"] = float(elem_count)
        metrics[f"lr_group_{safe_name}"] = float(group.get("lr", 0.0))
    return metrics


def _tensor_finite_abs_max(tensor: torch.Tensor) -> float:
    if tensor.numel() == 0:
        return 0.0
    safe = torch.nan_to_num(tensor.detach(), nan=0.0, posinf=0.0, neginf=0.0)
    return float(safe.abs().max().item())


def _collect_nonfinite_gradient_diagnostics(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_items: int = 16,
) -> dict[str, Any]:
    group_lookup = _optimizer_param_group_lookup(optimizer)
    count = 0
    samples: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if isinstance(param, UninitializedParameter):
            continue
        grad = param.grad
        if grad is None:
            continue
        grad_detached = grad.detach()
        if bool(torch.isfinite(grad_detached).all().item()):
            continue
        count += 1
        if len(samples) >= max_items:
            continue
        samples.append(
            {
                "name": str(name),
                "group": group_lookup.get(id(param)),
                "shape": tuple(int(dim) for dim in param.shape),
                "grad_has_nan": bool(torch.isnan(grad_detached).any().item()),
                "grad_has_inf": bool(torch.isinf(grad_detached).any().item()),
                "grad_abs_max_finite": _tensor_finite_abs_max(grad_detached),
                "param_has_nan": bool(torch.isnan(param.detach()).any().item()),
                "param_has_inf": bool(torch.isinf(param.detach()).any().item()),
                "param_abs_max_finite": _tensor_finite_abs_max(param.detach()),
            }
        )
    return {
        "nonfinite_grad_count": int(count),
        "samples": samples,
    }


def _collect_nonfinite_parameter_diagnostics(
    model: torch.nn.Module,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    max_items: int = 16,
) -> dict[str, Any]:
    group_lookup = _optimizer_param_group_lookup(optimizer)
    count = 0
    samples: list[dict[str, Any]] = []
    for name, param in model.named_parameters():
        if isinstance(param, UninitializedParameter):
            continue
        param_detached = param.detach()
        if bool(torch.isfinite(param_detached).all().item()):
            continue
        count += 1
        if len(samples) >= max_items:
            continue
        samples.append(
            {
                "name": str(name),
                "group": group_lookup.get(id(param)),
                "shape": tuple(int(dim) for dim in param.shape),
                "param_has_nan": bool(torch.isnan(param_detached).any().item()),
                "param_has_inf": bool(torch.isinf(param_detached).any().item()),
                "param_abs_max_finite": _tensor_finite_abs_max(param_detached),
            }
        )
    return {
        "nonfinite_param_count": int(count),
        "samples": samples,
    }


@dataclasses.dataclass(frozen=True)
class _TransitionWindow:
    segment_id: int
    start_step_id: int
    prompt: str
    frames: tuple[PicfObservation, ...]


@dataclasses.dataclass(frozen=True)
class _PreparedTrainingMicroWindow:
    micro_step: int
    flat_index: int
    sampled_bucket: str
    retry_count: int
    point_counts: tuple[int, ...]
    window: _TransitionWindow


@dataclasses.dataclass(frozen=True)
class _SegmentSamplingSlot:
    segment_id: int
    first_valid_start_step_id: int
    valid_start_exclusive: int

    @property
    def num_valid_starts(self) -> int:
        return int(self.valid_start_exclusive - self.first_valid_start_step_id)


@dataclasses.dataclass(frozen=True)
class _PendingTransitionLoss:
    index: int
    output: Any
    next_observation: PicfObservation
    next_visual_map_override: torch.Tensor | np.ndarray | None
    action_target: torch.Tensor | np.ndarray | None
    flow_override: dict[str, torch.Tensor] | None
    candidate_density: torch.Tensor
    tactile_contact_prob_mean: torch.Tensor
    tactile_active_rate: torch.Tensor
    owm_debug_metrics: dict[str, torch.Tensor]
    policy_forward_sec: float


def _flow_training_total(flow: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
    if flow is None:
        return None
    return flow.get("training_total", flow["total"])


def _flow_training_component(flow: dict[str, torch.Tensor] | None, component: str) -> torch.Tensor | None:
    if flow is None:
        return None
    training_key = f"training_{component}"
    return flow.get(training_key, flow[component])


def _flow_default_equiv(flow: dict[str, torch.Tensor] | None) -> torch.Tensor | None:
    if flow is None:
        return None
    return flow.get("mse_total", flow["total"])


def _window_trace_record(
    window: _TransitionWindow,
    *,
    global_step: int,
    micro_step: int,
    rank: int,
    flat_index: int,
    retry_count: int,
    point_counts: tuple[int, ...],
) -> dict[str, Any]:
    """Compact sample/data-window trace for action-rebound attribution.

    The recurrent late rebound can only be separated from optimizer drift if
    the exact window distribution is visible.  This record is deliberately
    metadata-only: it does not affect model inputs, gradients, or sampling.
    """

    first = window.frames[0]
    action = None if first.action is None else np.asarray(first.action, dtype=np.float32)
    action_chunk = None if first.action_chunk is None else np.asarray(first.action_chunk, dtype=np.float32)

    def _norm(value: np.ndarray | None) -> float | None:
        if value is None or value.size == 0:
            return None
        return float(np.linalg.norm(value.reshape(-1)))

    def _count(value: Any | None) -> int:
        if value is None:
            return 0
        try:
            return int(np.asarray(value).shape[0])
        except Exception:
            return 0

    return {
        "global_step": int(global_step),
        "micro_step": int(micro_step),
        "rank": int(rank),
        "flat_index": int(flat_index),
        "segment": int(window.segment_id),
        "start_step": int(window.start_step_id),
        "prompt": str(window.prompt),
        "prompt_bucket": _calvin_prompt_bucket(str(window.prompt)),
        "retry_count": int(retry_count),
        "point_counts": [int(count) for count in point_counts],
        "action_norm": _norm(action),
        "action_chunk_norm": _norm(action_chunk),
        "action_chunk_first_norm": _norm(None if action_chunk is None else action_chunk[:1]),
        "action_chunk_last_norm": _norm(None if action_chunk is None else action_chunk[-1:]),
        "proposal_count": _count(first.proposal_centers_xy),
        "proposal_mask_point_count": _count(first.proposal_mask_xy),
        "tracklet_count": _count(first.tracklet_xy),
    }


def _load_action_chunk(
    reader,
    *,
    step_id: int,
    segment_end: int,
    action_horizon: int,
    current_action: np.ndarray | None = None,
    action_key: str = "rel_actions",
) -> np.ndarray | None:
    if int(action_horizon) <= 1:
        return None
    if current_action is None:
        current = reader.read_npz(step_id, keys=[action_key])[action_key]
    else:
        current = np.asarray(current_action, dtype=np.float32)
    actions = [np.asarray(current, dtype=np.float32)]
    last = actions[0]
    for future_step in range(step_id + 1, step_id + int(action_horizon)):
        if future_step < int(segment_end):
            last = np.asarray(reader.read_npz(future_step, keys=[action_key])[action_key], dtype=np.float32)
        actions.append(last)
    return np.stack(actions, axis=0)


def _read_optional_npz_fields(reader: Any, step_id: int, keys: Sequence[str]) -> dict[str, np.ndarray]:
    """Read optional per-episode arrays without turning absent modalities into errors."""

    keys = tuple(str(k) for k in keys)
    if not keys:
        return {}
    optional_reader = getattr(reader, "read_npz_optional", None)
    if callable(optional_reader):
        return dict(optional_reader(step_id, list(keys)))
    try:
        payload = reader.read_npz(step_id, keys=None)
    except Exception:
        return {}
    return {key: payload[key] for key in keys if key in payload}


def _read_npz_required_optional(
    reader: Any,
    step_id: int,
    *,
    required: Sequence[str],
    optional: Sequence[str],
) -> dict[str, np.ndarray]:
    required = tuple(str(k) for k in required)
    optional = tuple(str(k) for k in optional)
    combined_reader = getattr(reader, "read_npz_required_optional", None)
    if callable(combined_reader):
        return dict(combined_reader(step_id, required=list(required), optional=list(optional)))
    frame = dict(reader.read_npz(step_id, keys=list(required)))
    if optional:
        frame.update(_read_optional_npz_fields(reader, step_id, optional))
    return frame


def _read_mvtrack_sidecar_fields(
    sidecar_root: str | Path | None,
    *,
    split: str,
    step_id: int,
    keys: Sequence[str],
    proposal_nearest_max_gap: int = 0,
) -> dict[str, np.ndarray]:
    """Read optional tracklet/proposal sidecar arrays for one CALVIN frame.

    The main CALVIN zip/directory remains immutable. Offline contact/task/
    tracklet sidecar jobs can write npz files with the same episode ids; the
    trainer then merges only optional MVTrack fields into the frame. Archived
    blind-SAM roots are rejected by argument validation unless explicitly
    allowed for historical reproduction.
    """

    if sidecar_root is None:
        return {}
    root = Path(sidecar_root)
    proposal_key_set = {str(key) for key in _MVTRACK_PROPOSAL_KEYS if str(key) != "proposal_age"}
    requested_proposal_keys = [str(key) for key in keys if str(key).startswith("proposal_")]

    def _proposal_count_from_frame(frame: Mapping[str, np.ndarray]) -> int:
        if "proposal_centers_xy" in frame:
            return int(np.asarray(frame["proposal_centers_xy"]).reshape(-1, 2).shape[0])
        if "proposal_boxes_xyxy" in frame:
            return int(np.asarray(frame["proposal_boxes_xyxy"]).reshape(-1, 4).shape[0])
        return 0

    def _proposal_count_from_path(path: Path) -> int:
        try:
            with np.load(path, allow_pickle=False) as data:
                if "proposal_centers_xy" in data.files:
                    return int(np.asarray(data["proposal_centers_xy"]).reshape(-1, 2).shape[0])
                if "proposal_boxes_xyxy" in data.files:
                    return int(np.asarray(data["proposal_boxes_xyxy"]).reshape(-1, 4).shape[0])
        except Exception:
            return 0
        return 0

    def _nearest_proposal_path(max_gap: int) -> tuple[int, Path] | None:
        for gap in range(1, max(int(max_gap), 0) + 1):
            for signed_gap in (-gap, gap):
                for candidate in (
                    root / split / f"episode_{int(step_id + signed_gap):07d}.npz",
                    root / f"episode_{int(step_id + signed_gap):07d}.npz",
                ):
                    if not candidate.exists():
                        continue
                    if _proposal_count_from_path(candidate) <= 0:
                        continue
                    return abs(int(signed_gap)), candidate
        return None

    candidates = (
        root / split / f"episode_{int(step_id):07d}.npz",
        root / f"episode_{int(step_id):07d}.npz",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    temporal_gap = 0
    if path is None:
        max_gap = max(int(proposal_nearest_max_gap), 0)
        if max_gap <= 0 or not requested_proposal_keys:
            return {}
        best = _nearest_proposal_path(max_gap)
        if best is None:
            return {}
        temporal_gap, path = best
    with np.load(path, allow_pickle=False) as data:
        frame = {key: data[key] for key in keys if key in data.files}
    # Tracklet sidecar files may exist for frames where the sparse contact/task
    # proposal generator intentionally emitted no current proposal.  Do not let
    # that current tracklet file block proposal borrowing: merge nearest
    # non-empty proposal/mask fields into the same frame and rely on
    # `proposal_age` decay downstream.  This preserves current-frame tracklets
    # while keeping proposal evidence sparse and age-aware.
    if (
        path is not None
        and temporal_gap == 0
        and requested_proposal_keys
        and int(proposal_nearest_max_gap) > 0
        and _proposal_count_from_frame(frame) <= 0
    ):
        best = _nearest_proposal_path(int(proposal_nearest_max_gap))
        if best is not None:
            temporal_gap, proposal_path = best
            with np.load(proposal_path, allow_pickle=False) as data:
                for key in requested_proposal_keys:
                    if key == "proposal_age":
                        continue
                    if key in proposal_key_set and key in data.files:
                        frame[key] = data[key]
    if temporal_gap > 0 and any(key in frame for key in ("proposal_centers_xy", "proposal_boxes_xyxy")):
        count = 0
        if "proposal_centers_xy" in frame:
            count = int(np.asarray(frame["proposal_centers_xy"]).reshape(-1, 2).shape[0])
        elif "proposal_boxes_xyxy" in frame:
            count = int(np.asarray(frame["proposal_boxes_xyxy"]).reshape(-1, 4).shape[0])
        frame["proposal_age"] = np.full((count,), float(temporal_gap), dtype=np.float32)
    elif any(key in frame for key in ("proposal_centers_xy", "proposal_boxes_xyxy")):
        count = 0
        if "proposal_centers_xy" in frame:
            count = int(np.asarray(frame["proposal_centers_xy"]).reshape(-1, 2).shape[0])
        elif "proposal_boxes_xyxy" in frame:
            count = int(np.asarray(frame["proposal_boxes_xyxy"]).reshape(-1, 4).shape[0])
        frame.setdefault("proposal_age", np.zeros((count,), dtype=np.float32))
    return frame


class _CalvinTransitionSource:
    def __init__(
        self,
        root: str,
        *,
        split: str,
        backend: str,
        unroll_steps: int,
        action_horizon: int = 16,
        use_wrist_rgb: bool = True,
        use_tactile: bool = False,
        tactile_sensor_names: tuple[str, ...] = _DEFAULT_TACTILE_SENSOR_NAMES,
        tactile_sensor_offsets_m: tuple[tuple[float, float, float], ...] = _DEFAULT_TACTILE_SENSOR_OFFSETS_M,
        tactile_calibration: dict[str, object] | str | Path | None = None,
        tactile_backgrounds_by_sensor: dict[str, np.ndarray] | None = None,
        use_scene_obs: bool = False,
        load_tracklet_fields: bool = False,
        load_proposal_fields: bool = False,
        frame_dt_s: float = 1.0 / 30.0,
        action_normalizer: PicfActionNormalizer | None = None,
        augmentation_mode: str = "off",
        photometric_strength: str = "conservative",
        mvtrack_sidecar_root: str | Path | None = None,
        mvtrack_sidecar_proposal_nearest_max_gap: int = 0,
        segment_indices: Sequence[int] | None = None,
        bucket_sampling_mode: str = "round_robin",
        bucket_temperature_alpha: float = 0.0,
        bucket_weight_spec: str | None = None,
        bucket_sample_without_replacement: bool = True,
    ) -> None:
        if int(unroll_steps) < 1:
            raise ValueError(f"unroll_steps must be >= 1, got {unroll_steps}")
        if int(action_horizon) < 1:
            raise ValueError(f"action_horizon must be >= 1, got {action_horizon}")
        self.dataset = CalvinLangSegmentDataset(
            root=root,
            split=split,
            action_horizon=int(action_horizon),
            backend=backend,
            use_wrist_rgb=use_wrist_rgb,
            sample_within_segment=False,
        )
        self.reader = self.dataset.reader
        self.segments = self.dataset.segments
        self.split = split
        self.backend = backend
        self.unroll_steps = int(unroll_steps)
        self.action_horizon = int(action_horizon)
        self.use_wrist_rgb = bool(use_wrist_rgb)
        self.use_tactile = bool(use_tactile)
        self.tactile_sensor_names = tuple(tactile_sensor_names)
        self.tactile_sensor_offsets_m = tuple(tuple(offset) for offset in tactile_sensor_offsets_m)
        calibration_payload = tactile_calibration
        explicit_legacy_offsets = (
            tactile_sensor_offsets_m is not None
            and tuple(tuple(float(value) for value in offset) for offset in tactile_sensor_offsets_m) != _DEFAULT_TACTILE_SENSOR_OFFSETS_M
        )
        if tactile_calibration is None and explicit_legacy_offsets:
            calibration_payload = {"sensor_centers_local": tactile_sensor_offsets_m}
        self.tactile_calibration = _resolve_tactile_calibration(calibration_payload)
        self.tactile_backgrounds_by_sensor = None if tactile_backgrounds_by_sensor is None else {
            str(name): np.asarray(image) for name, image in tactile_backgrounds_by_sensor.items()
        }
        self.use_scene_obs = bool(use_scene_obs)
        self.load_tracklet_fields = bool(load_tracklet_fields)
        self.load_proposal_fields = bool(load_proposal_fields)
        self.frame_dt_s = float(frame_dt_s)
        self.action_normalizer = action_normalizer
        self.augmentation_mode = str(augmentation_mode).lower().replace("-", "_")
        self.photometric_strength = str(photometric_strength).lower().replace("-", "_")
        self.mvtrack_sidecar_root = None if mvtrack_sidecar_root is None else Path(mvtrack_sidecar_root)
        self.mvtrack_sidecar_proposal_nearest_max_gap = int(mvtrack_sidecar_proposal_nearest_max_gap)
        if self.augmentation_mode not in {"off", "photometric", "multimodal_geometry"}:
            raise ValueError(
                "augmentation_mode must be one of {'off', 'photometric', 'multimodal_geometry'}, "
                f"got {augmentation_mode!r}."
            )
        if self.augmentation_mode == "multimodal_geometry":
            raise NotImplementedError(
                "picf_augmentation_mode=multimodal_geometry is intentionally not implemented yet. "
                "Full PICF requires synchronized RGB/depth/point/camera transforms."
            )
        # Keep the exhaustive valid window-start index for diagnostics and
        # compatibility utilities, but do not use it as the training sampling
        # distribution. Training should remain segment-uniform, matching the
        # historical CALVIN dataset contract more closely.
        self.window_index: list[tuple[int, int]] = []
        self.segment_sampling_slots: list[_SegmentSamplingSlot] = []
        selected_segment_ids = list(range(len(self.segments))) if segment_indices is None else [int(value) for value in segment_indices]
        for segment_id in selected_segment_ids:
            if segment_id < 0 or segment_id >= len(self.segments):
                raise ValueError(f"calvin_segment_indices contains out-of-range segment id {segment_id}; valid [0,{len(self.segments)-1}].")
            segment = self.segments[int(segment_id)]
            max_start_exclusive = segment.end - (self.unroll_steps + self.action_horizon - 1)
            for step_id in range(segment.start, max_start_exclusive):
                self.window_index.append((int(segment_id), step_id))
            if segment.start < max_start_exclusive:
                self.segment_sampling_slots.append(
                    _SegmentSamplingSlot(
                        segment_id=int(segment_id),
                        first_valid_start_step_id=int(segment.start),
                        valid_start_exclusive=int(max_start_exclusive),
                    )
                )
        if not self.segment_sampling_slots:
            raise RuntimeError(
                "No valid CALVIN transition windows found for "
                f"split={split}, backend={backend}, unroll_steps={unroll_steps}, action_horizon={action_horizon}."
            )
        self.bucket_to_slot_indices: dict[str, list[int]] = {}
        for slot_index, slot in enumerate(self.segment_sampling_slots):
            bucket = _calvin_prompt_bucket(self.segments[int(slot.segment_id)].lang)
            self.bucket_to_slot_indices.setdefault(bucket, []).append(int(slot_index))
        self.bucket_names: tuple[str, ...] = tuple(
            sorted(bucket for bucket, indices in self.bucket_to_slot_indices.items() if indices)
        )
        self.bucket_sampling_mode = _normalize_bucket_sampling_mode(bucket_sampling_mode)
        self.bucket_temperature_alpha = float(bucket_temperature_alpha)
        self.bucket_weight_spec = str(bucket_weight_spec or "").strip()
        self.bucket_sample_without_replacement = bool(bucket_sample_without_replacement)
        self.bucket_segment_counts: dict[str, int] = {
            str(bucket): int(len(indices)) for bucket, indices in sorted(self.bucket_to_slot_indices.items())
        }
        self.bucket_target_weights: dict[str, float] = _compute_bucket_sampling_weights(
            bucket_names=self.bucket_names,
            bucket_sizes=self.bucket_segment_counts,
            mode=self.bucket_sampling_mode,
            temperature_alpha=self.bucket_temperature_alpha,
            weight_spec=self.bucket_weight_spec,
        )

    def __len__(self) -> int:
        return len(self.segment_sampling_slots)

    def balanced_bucket_slot_index(
        self,
        *,
        seed: int,
        rank: int,
        world_size: int,
        step: int,
        micro_step: int,
        accum_steps: int,
        retry_count: int = 0,
    ) -> tuple[int, str, np.random.Generator]:
        """Return a deterministic task-bucket-balanced segment slot.

        This is a sampling-only repair for action-loss rebound diagnostics: the
        optimizer sees a controlled sequence of task families across
        rank/micro-step slots, while each selected segment still samples a
        random valid start step through the same step-indexed RNG.
        """

        sample_rng = _step_indexed_window_rng(
            seed=int(seed),
            rank=int(rank),
            step=int(step),
            micro_step=int(micro_step),
            retry_count=int(retry_count),
        )
        if not self.bucket_names:
            return int(sample_rng.integers(0, len(self))), "unbucketed", sample_rng
        bucket_sequence = _bucket_sequence_for_logical_step(
            bucket_names=self.bucket_names,
            target_bucket_weights=self.bucket_target_weights,
            mode=self.bucket_sampling_mode,
            weight_spec=self.bucket_weight_spec,
            seed=int(seed),
            step=int(step),
            world_size=int(world_size),
            accum_steps=int(accum_steps),
            without_replacement=bool(self.bucket_sample_without_replacement),
        )
        global_micro_in_step = int(rank) * max(int(accum_steps), 1) + int(micro_step)
        bucket = str(bucket_sequence[int(global_micro_in_step) % len(bucket_sequence)])
        candidates = self.bucket_to_slot_indices[bucket]
        slot_index = int(candidates[int(sample_rng.integers(0, len(candidates)))])
        return slot_index, bucket, sample_rng

    def close(self) -> None:
        self.reader.close()

    def _load_frame(
        self,
        segment_id: int,
        step_id: int,
        *,
        reset_scaffold: bool,
        rng: np.random.Generator | None = None,
    ) -> PicfObservation:
        segment = self.segments[segment_id]
        keys = ["rgb_static", "depth_static", "depth_gripper", "robot_obs", "rel_actions"]
        if self.use_wrist_rgb:
            keys.append("rgb_gripper")
        if self.use_tactile:
            keys.extend(["rgb_tactile", "depth_tactile"])
        if self.use_scene_obs:
            keys.append("scene_obs")
        optional_keys: list[str] = []
        if self.load_tracklet_fields:
            optional_keys.extend(_MVTRACK_TRACKLET_KEYS)
        if self.load_proposal_fields:
            optional_keys.extend(_MVTRACK_PROPOSAL_KEYS)
        frame = _read_npz_required_optional(self.reader, step_id, required=keys, optional=optional_keys)
        if optional_keys:
            frame.update(
                _read_mvtrack_sidecar_fields(
                    self.mvtrack_sidecar_root,
                    split=self.split,
                    step_id=step_id,
                    keys=optional_keys,
                    proposal_nearest_max_gap=(
                        self.mvtrack_sidecar_proposal_nearest_max_gap if self.load_proposal_fields else 0
                    ),
                )
            )
        if self.augmentation_mode == "photometric":
            jitter_rng = rng if rng is not None else np.random.default_rng()
            frame["rgb_static"] = _apply_picf_photometric_augmentation(
                frame["rgb_static"],
                rng=jitter_rng,
                strength=self.photometric_strength,
            )
            if self.use_wrist_rgb and frame.get("rgb_gripper") is not None:
                frame["rgb_gripper"] = _apply_picf_photometric_augmentation(
                    frame["rgb_gripper"],
                    rng=jitter_rng,
                    strength=self.photometric_strength,
                )
        timestamp_s = float(step_id) * self.frame_dt_s
        action = frame.get("rel_actions")
        action_chunk = _load_action_chunk(
            self.reader,
            step_id=step_id,
            segment_end=segment.end,
            action_horizon=self.action_horizon,
            current_action=action,
        )
        if action is not None and self.action_normalizer is not None:
            action = self.action_normalizer.normalize_np(action)
        if action_chunk is not None and self.action_normalizer is not None:
            action_chunk = self.action_normalizer.normalize_np(action_chunk)
        tactile = (
            _calvin_tactile_packet(
                frame,
                timestamp_s=timestamp_s,
                sensor_names=self.tactile_sensor_names,
                robot_obs=frame["robot_obs"],
                calibration=self.tactile_calibration,
                background_rgb_by_sensor=self.tactile_backgrounds_by_sensor,
            )
            if self.use_tactile
            else None
        )
        return PicfObservation(
            rgb_static=frame["rgb_static"],
            depth_static=frame["depth_static"],
            depth_gripper=frame.get("depth_gripper"),
            robot_obs=frame["robot_obs"],
            prompt=segment.lang,
            step_id=int(step_id),
            segment_id=int(segment_id),
            timestamp_s=timestamp_s,
            reset_scaffold=bool(reset_scaffold),
            rgb_gripper=frame.get("rgb_gripper"),
            scene_obs=frame.get("scene_obs"),
            proprio=frame["robot_obs"],
            action=action,
            action_chunk=action_chunk,
            tactile=tactile,
            tracklet_xy=frame.get("tracklet_xy"),
            tracklet_velocity=frame.get("tracklet_velocity"),
            tracklet_visibility=frame.get("tracklet_visibility"),
            tracklet_confidence=frame.get("tracklet_confidence"),
            tracklet_ids=frame.get("tracklet_ids"),
            tracklet_view_ids=frame.get("tracklet_view_ids"),
            tracklet_age=frame.get("tracklet_age"),
            proposal_centers_xy=frame.get("proposal_centers_xy"),
            proposal_boxes_xyxy=frame.get("proposal_boxes_xyxy"),
            proposal_objectness=frame.get("proposal_objectness"),
            proposal_view_ids=frame.get("proposal_view_ids"),
            proposal_source_ids=frame.get("proposal_source_ids"),
            proposal_age=frame.get("proposal_age"),
            proposal_mask_xy=frame.get("proposal_mask_xy"),
            proposal_mask_weights=frame.get("proposal_mask_weights"),
            proposal_mask_offsets=frame.get("proposal_mask_offsets"),
        )

    def sample_window_metadata(
        self,
        slot_index: int,
        *,
        rng: np.random.Generator | None = None,
    ) -> tuple[int, int]:
        slot = self.segment_sampling_slots[int(slot_index)]
        if rng is None:
            start_step_id = int(slot.first_valid_start_step_id)
        else:
            start_step_id = int(rng.integers(slot.first_valid_start_step_id, slot.valid_start_exclusive))
        return int(slot.segment_id), int(start_step_id)

    def window(self, flat_index: int, *, rng: np.random.Generator | None = None) -> _TransitionWindow:
        segment_id, start_step_id = self.sample_window_metadata(int(flat_index), rng=rng)
        return self.window_from_metadata(segment_id=segment_id, start_step_id=start_step_id, rng=rng)

    def window_from_metadata(
        self,
        *,
        segment_id: int,
        start_step_id: int,
        rng: np.random.Generator | None = None,
    ) -> _TransitionWindow:
        segment_id = int(segment_id)
        start_step_id = int(start_step_id)
        if segment_id < 0 or segment_id >= len(self.segments):
            raise ValueError(f"segment_id out of range: {segment_id}; valid [0,{len(self.segments)-1}].")
        segment = self.segments[segment_id]
        max_start_exclusive = int(segment.end) - (self.unroll_steps + self.action_horizon - 1)
        if start_step_id < int(segment.start) or start_step_id >= max_start_exclusive:
            raise ValueError(
                "start_step_id out of valid range for segment "
                f"{segment_id}: got {start_step_id}, valid [{int(segment.start)},{max_start_exclusive})."
            )
        frames = tuple(
            self._load_frame(
                segment_id,
                start_step_id + offset,
                reset_scaffold=(offset == 0),
                rng=rng,
            )
            for offset in range(self.unroll_steps + 1)
        )
        return _TransitionWindow(
            segment_id=int(segment_id),
            start_step_id=int(start_step_id),
            prompt=self.segments[segment_id].lang,
            frames=frames,
        )

def _loss_component_or_zero(losses: Any, name: str, reference: torch.Tensor | None = None) -> torch.Tensor:
    value = getattr(losses, name, None)
    if value is not None:
        return value
    if reference is None:
        reference = getattr(losses, "pt", None)
    if reference is None:
        reference = getattr(losses, "total", None)
    if reference is None:
        raise AttributeError(f"loss object has no {name!r}, 'pt', or 'total' component")
    return reference * 0.0


OWM_DEBUG_METRIC_KEYS: tuple[str, ...] = (
    "aqr_temporal_support_entropy_mean",
    "pi_prefix_norm_mode_enabled",
    "pi_prefix_norm_mean",
    "pi_prefix_norm_max",
    "pi_prefix_rms_mean",
    "pi_prefix_rms_max",
    "pi_prefix_max_abs",
    "pi_prefix_nonfinite_count",
    "pi_prefix_pre_rms_mean",
    "pi_prefix_pre_rms_max",
    "pi_prefix_post_rms_mean",
    "pi_prefix_post_rms_max",
    "pi_prefix_scale_mean",
    "pi_prefix_scale_min",
    "pi_prefix_scale_max",
    "aqr_temporal_support_time_mass_t0",
    "aqr_temporal_support_time_mass_t1",
    "aqr_temporal_view_mass_0",
    "aqr_temporal_view_mass_1",
    "aqr_pg_support_entropy_mean",
    "aqr_pg_support_max",
    "aqr_pg_support_peak_mean",
    "aqr_tracklet_support_entropy_mean",
    "aqr_tracklet_support_max",
    "aqr_proposal_support_entropy_mean",
    "aqr_proposal_support_max",
    "aqr_proposal_shape_quality_mean",
    "aqr_proposal_shape_quality_max",
    "aqr_proposal_shape_quality_nonzero_fraction",
    "aqr_proposal_point_bridge_entropy_mean",
    "aqr_proposal_point_bridge_max",
    "aqr_task_owner_point_bridge_entropy_mean",
    "aqr_task_owner_point_bridge_max",
    "aqr_task_owner_point_bridge_nonzero_fraction",
    "aqr_proposal_anchor_seed_row_count",
    "aqr_proposal_anchor_seed_nonzero_fraction",
    "aqr_proposal_anchor_seed_point_max",
    "aqr_proposal_anchor_seed_entropy_mean",
    "aqr_proposal_anchor_seed_assignment_max",
    "aqr_object_candidate_assigned_row_count",
    "aqr_object_candidate_assigned_candidate_count",
    "aqr_object_candidate_assignment_max",
    "aqr_object_candidate_owner_row_count",
    "aqr_object_candidate_owner_candidate_count",
    "aqr_object_candidate_owner_assignment_max",
    "aqr_object_candidate_owner_point_row_count",
    "aqr_object_candidate_owner_point_max",
    "aqr_object_candidate_coverage_mean",
    "aqr_object_candidate_coverage_max",
    "aqr_object_candidate_background_mean",
    "aqr_object_candidate_duplicate_overlap_max",
    "aqr_object_explanation_quality_mean",
    "aqr_object_explanation_quality_max",
    "aqr_object_explanation_quality_active_mean",
    "aqr_task_owner_visual_prior_entropy",
    "aqr_task_owner_visual_prior_max",
    "aqr_task_owner_proposal_score_max",
    "aqr_task_owner_proposal_score_mean",
    "aqr_task_owner_proposal_score_nonzero_fraction",
    "aqr_task_owner_proposal_score_entropy",
    "aqr_task_owner_proposal_selected_count",
    "aqr_task_owner_proposal_shape_quality_mean",
    "aqr_task_owner_anchor_score_max",
    "aqr_task_owner_anchor_score_mean",
    "aqr_task_owner_anchor_score_nonzero_fraction",
    "aqr_local_support_entropy_mean",
    "aqr_same_role_local_overlap_max",
    "aqr_same_role_local_true_overlap_max",
    "aqr_same_role_local_true_overlap_mean",
    "aqr_same_role_local_jaccard_max",
    "aqr_same_role_local_jaccard_mean",
    "aqr_same_role_anchor_binding_signature_overlap_max",
    "aqr_same_role_anchor_binding_signature_overlap_mean",
    "aqr_same_role_obs_binding_signature_overlap_max",
    "aqr_same_role_obs_binding_signature_overlap_mean",
    "aqr_local_source_mass_visual",
    "aqr_local_source_mass_temporal",
    "aqr_local_source_mass_point",
    "aqr_local_source_mass_tracklet",
    "aqr_local_source_mass_proposal",
    "aqr_effective_anchor_count",
    "aqr_same_role_support_overlap_max",
    "aqr_same_role_object_core_overlap_max",
    "aqr_same_role_object_core_overlap_mean",
    "oeml_valid",
    "oeml_anchor_quality_mean",
    "oeml_anchor_quality_max",
    "oeml_duplicate_overlap_max",
    "oeml_duplicate_overlap_mean",
    "oeml_feature_variance_mean",
    "oeml_point_spatial_variance_mean",
    "oeml_point_quality_mean",
    "oeml_point_quality_max",
    "oeml_contact_explanation_score",
    "oeml_visual_background_mean",
    "oeml_point_background_mean",
    "object_candidate_owner_geometry_rows",
    "object_candidate_owner_geometry_dist_mean",
    "object_candidate_owner_geometry_dist_max",
    "object_candidate_owner_geometry_active_dist_mean",
    "object_candidate_owner_geometry_active_dist_max",
    "aqr_active_anchor_count",
    "aqr_inactive_anchor_fraction",
    "aqr_context_anchor_count",
    "aqr_context_downstream_weight_mean",
    "aqr_reserve_anchor_fraction",
    "aqr_downstream_same_role_support_overlap_max",
    "aqr_downstream_same_role_object_core_overlap_max",
    "aqr_context_same_role_support_overlap_max",
    "aqr_context_same_role_object_core_overlap_max",
    "aqr_reserve_same_role_support_overlap_max",
    "aqr_reserve_same_role_support_overlap_mean",
    "aqr_reserve_same_role_object_core_overlap_max",
    "aqr_reserve_same_role_object_core_overlap_mean",
    "aqr_active_same_role_support_overlap_max",
    "aqr_active_same_role_support_overlap_mean",
    "aqr_active_same_role_object_core_overlap_max",
    "aqr_active_same_role_object_core_overlap_mean",
    "aqr_active_anchor_count_role_0",
    "aqr_active_anchor_count_role_1",
    "aqr_active_anchor_count_role_2",
    "aqr_active_anchor_count_role_3",
    "vcap_enabled",
    "vcap_proposal_count",
    "vcap_stop_entropy",
    "vcap_active_prob_mean",
    "vcap_unexplained_evidence",
    "vcap_duplicate_cost",
    "vcap_count_cost",
    "vcap_continuity_cost",
    "vcap_matched_old_file_fraction",
    "vcap_birth_fraction",
    "vcap_noobject_fraction",
    "vcap_action_grad_scale",
    "aqr_ownership_prior_enabled",
    "aqr_ownership_prior_weight",
    "aqr_ownership_point_prior_weight",
    "aqr_ownership_point_prior_sigma_m",
    "aqr_ownership_temporal_prior_weight",
    "aqr_same_role_support_competition_enabled",
    "aqr_same_role_support_competition_weight",
    "posterior_identity_switch_rate",
    "posterior_identity_switch_rate_stable",
    "posterior_identity_switch_rate_nonrecycled",
    "posterior_identity_switch_rate_recycled",
    "posterior_stable_slot_fraction",
    "posterior_binding_top1_margin_mean",
    "posterior_binding_top1_margin_min",
    "posterior_binding_top1_margin_stable_mean",
    "posterior_file_self_signature_sim_mean",
    "posterior_file_best_other_signature_margin_mean",
    "posterior_file_potential_swap_rate",
    "posterior_file_calibrated_self_signature_sim_mean",
    "posterior_file_calibrated_best_other_signature_margin_mean",
    "posterior_file_calibrated_potential_swap_rate",
    "posterior_file_calibrated_signature_score_std",
    "posterior_active_file_fraction",
    "posterior_active_file_self_signature_sim_mean",
    "posterior_active_file_best_other_signature_margin_mean",
    "posterior_active_file_potential_swap_rate",
    "posterior_active_file_calibrated_self_signature_sim_mean",
    "posterior_active_file_calibrated_best_other_signature_margin_mean",
    "posterior_active_file_calibrated_potential_swap_rate",
    "posterior_binding_signature_linear_score_mean",
    "posterior_binding_signature_linear_score_abs_mean",
    "posterior_binding_signature_quadratic_score_mean",
    "posterior_binding_signature_quadratic_score_abs_mean",
    "posterior_binding_signature_low_rank_score_mean",
    "posterior_binding_signature_low_rank_score_abs_mean",
    "posterior_binding_signature_combined_score_mean",
    "posterior_binding_signature_combined_score_abs_mean",
    "posterior_binding_signature_calibrated_score_mean",
    "posterior_binding_signature_calibrated_score_abs_mean",
    "posterior_binding_signature_calibrated_score_std",
    "posterior_binding_signature_calibrated_top1_margin_mean",
    "posterior_binding_signature_gate_mean",
    "posterior_binding_signature_update_rate_mean",
    "posterior_binding_signature_measurement_trust_mean",
    "posterior_binding_signature_memory_keep_rate_mean",
    "posterior_binding_signature_measurement_score_std",
    "posterior_binding_signature_measurement_margin_mean",
    "posterior_binding_signature_measurement_dispersion_gate_mean",
    "posterior_recycle_rate",
    "posterior_recycle_rate_effector",
    "posterior_recycle_rate_scene",
    "posterior_active_file_recycle_rate",
    "posterior_inactive_file_recycle_rate",
    "posterior_recycle_gate_std",
    "posterior_recycle_gate_min",
    "posterior_recycle_gate_max",
    "posterior_recycle_logit_mean",
    "posterior_recycle_logit_std",
    "posterior_recycle_logit_min",
    "posterior_recycle_logit_max",
    "posterior_support_mass_raw_mean",
    "posterior_support_mass_final_mean",
    "posterior_prior_var_mean",
    "posterior_prior_alpha_mean",
    "posterior_residual_summary_norm",
    "posterior_dustbin_mass_raw",
    "posterior_dustbin_mass_final",
    "posterior_lifecycle_assignment_confidence_mean",
    "posterior_lifecycle_support_entropy_mean",
    "posterior_lifecycle_support_margin_mean",
    "posterior_lifecycle_owner_reliability_mean",
    "posterior_lifecycle_survival_prob_mean",
    "posterior_lifecycle_reset_allowance_mean",
    "posterior_lifecycle_recycle_raw_mean",
    "posterior_lifecycle_inactive_dustbin_mass",
    "posterior_lifecycle_unexplained_dustbin_mass",
    "posterior_file_competition_active_mean",
    "posterior_file_competition_active_count",
    "posterior_file_competition_demoted_mass_mean",
    "posterior_file_competition_demoted_mass_max",
    "posterior_file_competition_duplicate_overlap_max",
    "posterior_file_competition_active_duplicate_overlap_max",
    "posterior_file_competition_birth_active_mean",
    "posterior_file_competition_birth_count",
    "posterior_file_competition_birth_share_mean",
    "posterior_file_competition_birth_share_max",
    "posterior_identity_innovation_risk",
    "posterior_address_update_rate_mean",
    "posterior_address_update_rate_max",
    "posterior_owner_transport_mass_mean",
    "posterior_owner_transport_mass_max",
    "posterior_owner_transport_confidence_mean",
    "posterior_owner_transport_confidence_max",
    "posterior_owner_transport_dist_to_standard_mean",
    "posterior_owner_transport_dist_to_standard_max",
    "posterior_owner_transport_dist_after_fusion_mean",
    "posterior_owner_transport_dist_after_fusion_max",
    "posterior_owner_transport_active_count",
    "posterior_owner_transport_active_confidence_mean",
    "posterior_owner_transport_active_confidence_max",
    "posterior_owner_transport_active_dist_mean",
    "posterior_owner_transport_active_dist_min",
    "posterior_owner_transport_active_dist_max",
    "posterior_owner_transport_active_dist_after_fusion_mean",
    "posterior_owner_transport_active_dist_after_fusion_min",
    "posterior_owner_transport_active_dist_after_fusion_max",
    "posterior_owner_transport_applied_fraction",
    "posterior_owner_active_gate_enabled",
    "posterior_owner_active_score_mean",
    "posterior_owner_active_score_max",
    "posterior_owner_active_eligible_fraction",
    "owm_temporal_visual_tokens",
    "owm_tracklet_tokens",
    "owm_tracklet_valid_fraction",
    "owm_proposal_tokens",
    "owm_proposal_valid_fraction",
    "owm_posterior_support_signature_mean",
    "owm_posterior_binding_signature_norm_mean",
    "evidence_cache_trust_mean",
    "evidence_cache_age_mean",
    "tactile_contact_prob_max",
    "tactile_evidence_rate",
    "tactile_evidence_weight_mean",
    "tactile_evidence_weight_max",
    "innovation_norm_visual",
    "innovation_norm_point",
    "innovation_norm_tactile",
    "owm_ordinal_active",
    "owm_ordinal_target_rank",
    "owm_ordinal_confidence",
    "pi_prefix_gate_mean",
    "pi_prefix_gate_min",
    "pi_prefix_gate_max",
    "pi_prefix_teacher_mode_enabled",
    "pi_prefix_teacher_trust_loss",
    "pi_prefix_teacher_trust_raw",
    "pi_prefix_teacher_delta_rms",
    "pi_prefix_teacher_cos_to_teacher",
    "pi_prefix_teacher_blend",
    "pi_prefix_teacher_ema_decay",
    "pi_context_token_count",
    "pi_context_gate",
    "pi_context_post_rms_mean",
    "pi_context_fused_prefix_token_count",
    "pi_context_attention_entropy_mean",
    "pi_context_fused_post_rms_mean",
    "pi_context_adapter_token_count",
    "pi_context_adapter_gate",
    "pi_context_adapter_attention_entropy_mean",
    "pi_context_adapter_residual_rms_mean",
    "pi_context_readout_enabled",
    "pi_context_readout_loss",
    "pi_context_readout_mse",
    "pi_context_readout_weighted_total",
    "pi_context_readout_weight",
    "pi_context_readout_token_count",
    "pi_context_readout_attention_entropy_mean",
    "pi_context_token_aux_enabled",
    "pi_context_token_aux_loss",
    "pi_context_token_aux_accuracy",
    "pi_context_token_aux_weighted_total",
    "pi_context_token_aux_weight",
    "pi_context_token_aux_bins",
    "pi_context_token_aux_clip",
    "pi_context_token_aux_token_count",
    "pi_context_token_aux_attention_entropy_mean",
    "pi_context_flow_residual_enabled",
    "pi_context_flow_residual_gate",
    "pi_context_flow_residual_token_count",
    "pi_context_flow_residual_rms_mean",
    "pi_context_flow_context_velocity_rms_mean",
    "pi_context_flow_context_target_mse",
    "pi_context_flow_residual_time_floor",
    "pi_context_flow_base_mse",
    "pi_context_flow_adapted_mse",
    "pi_context_flow_gain_mse_delta",
    "pi_action_flow_objective_mode_id",
    "pi_action_flow_time_mean",
    "pi_action_expert_router_enabled",
    "pi_action_expert_router_gate",
    "pi_action_expert_router_entropy_mean",
    "pi_action_expert_router_top_weight_mean",
    "pi_action_expert_router_residual_rms_mean",
    "pi_context_probe_mode_id",
    "pi_context_probe_delta_rms_mean",
    "pi_context_probe_post_rms_mean",
    "pi_prefix_probe_mode_id",
    "pi_prefix_probe_delta_rms_mean",
    "pi_prefix_probe_post_rms_mean",
    "pi_action_condition_token_count",
)


def _owm_debug_metrics_from_output(output: Any, reference: torch.Tensor) -> dict[str, torch.Tensor]:
    debug = getattr(output, "debug", {}) or {}
    return {
        key: torch.as_tensor(float(debug.get(key, 0.0)), device=reference.device, dtype=reference.dtype)
        for key in OWM_DEBUG_METRIC_KEYS
    }


def _accumulate_owm_debug_metrics(
    metrics: dict[str, torch.Tensor],
    update: dict[str, torch.Tensor],
) -> None:
    for key, value in update.items():
        metrics[key] = metrics.get(key, value * 0.0) + value


@dataclasses.dataclass
class _MetricAccumulator:
    loss_total: float = 0.0
    loss_action: float = 0.0
    loss_action_default_equiv: float = 0.0
    loss_action_weight_scale: float = 0.0
    loss_action_active7: float = 0.0
    loss_action_pos: float = 0.0
    loss_action_rot: float = 0.0
    loss_action_gripper: float = 0.0
    loss_visual_latent: float = 0.0
    loss_visual_real: float = 0.0
    loss_tactile_real: float = 0.0
    loss_tactile_map: float = 0.0
    loss_tactile_aux: float = 0.0
    loss_point_real: float = 0.0
    loss_semantic_future_aux: float = 0.0
    loss_semantic_group_raw: float = 0.0
    loss_semantic_group_capped: float = 0.0
    loss_physical_aux: float = 0.0
    loss_physical_aux_capped: float = 0.0
    loss_alignment: float = 0.0
    loss_alignment_raw: float = 0.0
    loss_total_minus_action: float = 0.0
    loss_action_prefix_trust: float = 0.0
    loss_anchor_pv: float = 0.0
    loss_anchor_object_pull: float = 0.0
    loss_anchor_object_pull_graph: float = 0.0
    loss_anchor_object_pull_posterior: float = 0.0
    loss_anchor_object_pull_graph_weight_sum: float = 0.0
    loss_anchor_object_pull_posterior_weight_sum: float = 0.0
    loss_anchor_object_pull_target_mass_mean: float = 0.0
    loss_anchor_object_pull_target_quality_mean: float = 0.0
    loss_pv_weak: float = 0.0
    loss_pt: float = 0.0
    loss_vl_router: float = 0.0
    loss_vl_heatmap_task: float = 0.0
    loss_vl_heatmap_effector: float = 0.0
    loss_vl_heatmap_interaction: float = 0.0
    loss_vl_point_consistency: float = 0.0
    loss_vl_anchor_diversity: float = 0.0
    loss_mapg_graph: float = 0.0
    loss_mapg_siglip: float = 0.0
    loss_mapg_vicreg: float = 0.0
    loss_mapg_cycle: float = 0.0
    loss_mapg_masked_modality: float = 0.0
    loss_mapg_routing: float = 0.0
    loss_mapg_support_diversity: float = 0.0
    loss_mapg_geometry_diversity: float = 0.0
    loss_slot_jepa: float = 0.0
    loss_slot_jepa_direction: float = 0.0
    loss_slot_jepa_log_norm: float = 0.0
    loss_slot_jepa_pred_norm: float = 0.0
    loss_slot_jepa_target_norm: float = 0.0
    loss_slot_jepa_matched_target_norm: float = 0.0
    loss_support_pred: float = 0.0
    loss_binding_consistency: float = 0.0
    loss_aqr_denoising: float = 0.0
    loss_slot_quality: float = 0.0
    loss_vcap: float = 0.0
    loss_object_explanation: float = 0.0
    loss_object_explanation_feature: float = 0.0
    loss_object_explanation_point: float = 0.0
    loss_object_explanation_contact: float = 0.0
    loss_object_explanation_duplicate: float = 0.0
    loss_object_explanation_background: float = 0.0
    physical_aux_budget_scale: float = 0.0
    semantic_aux_budget_scale: float = 0.0
    alignment_budget_scale: float = 0.0
    candidate_density: float = 0.0
    tactile_contact_prob_mean: float = 0.0
    tactile_active_rate: float = 0.0
    owm_debug_metrics: dict[str, float] = dataclasses.field(default_factory=dict)
    num_windows: int = 0

    def update(
        self,
        losses: PicfTransitionLossBreakdown,
        *,
        candidate_density: float,
        tactile_contact_prob_mean: float = 0.0,
        tactile_active_rate: float = 0.0,
    ) -> None:
        self.loss_total += float(losses.total.item())
        self.loss_action += float(losses.action.item())
        self.loss_action_default_equiv += float(losses.action_default_equiv.item())
        self.loss_action_weight_scale += float(losses.action_weight_scale.item())
        self.loss_action_active7 += float(losses.action_active7.item())
        self.loss_action_pos += float(losses.action_pos.item())
        self.loss_action_rot += float(losses.action_rot.item())
        self.loss_action_gripper += float(losses.action_gripper.item())
        self.loss_visual_latent += float(losses.visual_latent.item())
        self.loss_visual_real += float(losses.visual_real.item())
        self.loss_tactile_real += float(losses.tactile_real.item())
        self.loss_tactile_map += float(losses.tactile_map.item())
        self.loss_tactile_aux += float(losses.tactile_aux.item())
        self.loss_point_real += float(losses.point_real.item())
        self.loss_semantic_future_aux += float(losses.semantic_future_aux.item())
        self.loss_semantic_group_raw += float(losses.semantic_group_raw.item())
        self.loss_semantic_group_capped += float(losses.semantic_group_capped.item())
        self.loss_physical_aux += float(losses.physical_aux.item())
        self.loss_physical_aux_capped += float(losses.physical_aux_capped.item())
        self.loss_alignment += float(losses.alignment.item())
        self.loss_alignment_raw += float(losses.alignment_raw.item())
        self.loss_total_minus_action += float(losses.total_minus_action.item())
        self.loss_action_prefix_trust += float(_loss_component_or_zero(losses, "action_prefix_trust").item())
        self.loss_anchor_pv += float(losses.anchor_pv.item())
        self.loss_anchor_object_pull += float(losses.anchor_object_pull.item())
        self.loss_anchor_object_pull_graph += float(_loss_component_or_zero(losses, "anchor_object_pull_graph").item())
        self.loss_anchor_object_pull_posterior += float(_loss_component_or_zero(losses, "anchor_object_pull_posterior").item())
        self.loss_anchor_object_pull_graph_weight_sum += float(
            _loss_component_or_zero(losses, "anchor_object_pull_graph_weight_sum").item()
        )
        self.loss_anchor_object_pull_posterior_weight_sum += float(
            _loss_component_or_zero(losses, "anchor_object_pull_posterior_weight_sum").item()
        )
        self.loss_anchor_object_pull_target_mass_mean += float(
            _loss_component_or_zero(losses, "anchor_object_pull_target_mass_mean").item()
        )
        self.loss_anchor_object_pull_target_quality_mean += float(
            _loss_component_or_zero(losses, "anchor_object_pull_target_quality_mean").item()
        )
        self.loss_pv_weak += float(losses.pv_weak.item())
        self.loss_pt += float(losses.pt.item())
        self.loss_vl_router += float(losses.vl_router.item())
        self.loss_vl_heatmap_task += float(losses.vl_heatmap_task.item())
        self.loss_vl_heatmap_effector += float(losses.vl_heatmap_effector.item())
        self.loss_vl_heatmap_interaction += float(losses.vl_heatmap_interaction.item())
        self.loss_vl_point_consistency += float(losses.vl_point_consistency.item())
        self.loss_vl_anchor_diversity += float(losses.vl_anchor_diversity.item())
        self.loss_mapg_graph += float(losses.mapg_graph.item())
        self.loss_mapg_siglip += float(losses.mapg_siglip.item())
        self.loss_mapg_vicreg += float(losses.mapg_vicreg.item())
        self.loss_mapg_cycle += float(losses.mapg_cycle.item())
        self.loss_mapg_masked_modality += float(losses.mapg_masked_modality.item())
        self.loss_mapg_routing += float(losses.mapg_routing.item())
        self.loss_mapg_support_diversity += float(losses.mapg_support_diversity.item())
        self.loss_mapg_geometry_diversity += float(losses.mapg_geometry_diversity.item())
        self.loss_slot_jepa += float(_loss_component_or_zero(losses, "slot_jepa").item())
        self.loss_slot_jepa_direction += float(_loss_component_or_zero(losses, "slot_jepa_direction").item())
        self.loss_slot_jepa_log_norm += float(_loss_component_or_zero(losses, "slot_jepa_log_norm").item())
        self.loss_slot_jepa_pred_norm += float(_loss_component_or_zero(losses, "slot_jepa_pred_norm").item())
        self.loss_slot_jepa_target_norm += float(_loss_component_or_zero(losses, "slot_jepa_target_norm").item())
        self.loss_slot_jepa_matched_target_norm += float(
            _loss_component_or_zero(losses, "slot_jepa_matched_target_norm").item()
        )
        self.loss_support_pred += float(_loss_component_or_zero(losses, "support_pred").item())
        self.loss_binding_consistency += float(_loss_component_or_zero(losses, "binding_consistency").item())
        self.loss_aqr_denoising += float(_loss_component_or_zero(losses, "aqr_denoising").item())
        self.loss_slot_quality += float(_loss_component_or_zero(losses, "slot_quality").item())
        self.loss_vcap += float(_loss_component_or_zero(losses, "vcap").item())
        self.loss_object_explanation += float(_loss_component_or_zero(losses, "object_explanation").item())
        self.loss_object_explanation_feature += float(_loss_component_or_zero(losses, "object_explanation_feature").item())
        self.loss_object_explanation_point += float(_loss_component_or_zero(losses, "object_explanation_point").item())
        self.loss_object_explanation_contact += float(_loss_component_or_zero(losses, "object_explanation_contact").item())
        self.loss_object_explanation_duplicate += float(_loss_component_or_zero(losses, "object_explanation_duplicate").item())
        self.loss_object_explanation_background += float(_loss_component_or_zero(losses, "object_explanation_background").item())
        self.physical_aux_budget_scale += float(losses.physical_aux_budget_scale.item())
        self.semantic_aux_budget_scale += float(losses.semantic_aux_budget_scale.item())
        self.alignment_budget_scale += float(losses.alignment_budget_scale.item())
        self.candidate_density += float(candidate_density)
        self.tactile_contact_prob_mean += float(tactile_contact_prob_mean)
        self.tactile_active_rate += float(tactile_active_rate)
        self.num_windows += 1

    def update_from_outputs(self, outputs: dict[str, torch.Tensor]) -> None:
        self.loss_total += float(outputs["loss_total"].detach().item())
        self.loss_action += float(outputs["loss_action"].detach().item())
        self.loss_action_default_equiv += float(outputs.get("loss_action_default_equiv", outputs["loss_action"]).detach().item())
        self.loss_action_weight_scale += float(outputs.get("loss_action_weight_scale", outputs["loss_action"] * 0.0 + 1.0).detach().item())
        self.loss_action_active7 += float(outputs["loss_action_active7"].detach().item())
        self.loss_action_pos += float(outputs["loss_action_pos"].detach().item())
        self.loss_action_rot += float(outputs["loss_action_rot"].detach().item())
        self.loss_action_gripper += float(outputs["loss_action_gripper"].detach().item())
        self.loss_visual_latent += float(outputs["loss_visual_latent"].detach().item())
        self.loss_visual_real += float(outputs["loss_visual_real"].detach().item())
        self.loss_tactile_real += float(outputs["loss_tactile_real"].detach().item())
        self.loss_tactile_map += float(outputs["loss_tactile_map"].detach().item())
        self.loss_tactile_aux += float(outputs["loss_tactile_aux"].detach().item())
        self.loss_point_real += float(outputs["loss_point_real"].detach().item())
        self.loss_semantic_future_aux += float(outputs["loss_semantic_future_aux"].detach().item())
        self.loss_semantic_group_raw += float(outputs["loss_semantic_group_raw"].detach().item())
        self.loss_semantic_group_capped += float(outputs["loss_semantic_group_capped"].detach().item())
        self.loss_physical_aux += float(outputs["loss_physical_aux"].detach().item())
        self.loss_physical_aux_capped += float(outputs["loss_physical_aux_capped"].detach().item())
        self.loss_alignment += float(outputs["loss_alignment"].detach().item())
        self.loss_alignment_raw += float(outputs["loss_alignment_raw"].detach().item())
        self.loss_total_minus_action += float(outputs["loss_total_minus_action"].detach().item())
        self.loss_action_prefix_trust += float(outputs.get("loss_action_prefix_trust", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_anchor_pv += float(outputs["loss_anchor_pv"].detach().item())
        self.loss_anchor_object_pull += float(outputs["loss_anchor_object_pull"].detach().item())
        self.loss_anchor_object_pull_graph += float(outputs.get("loss_anchor_object_pull_graph", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_anchor_object_pull_posterior += float(outputs.get("loss_anchor_object_pull_posterior", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_anchor_object_pull_graph_weight_sum += float(
            outputs.get("loss_anchor_object_pull_graph_weight_sum", outputs["loss_pt"] * 0.0).detach().item()
        )
        self.loss_anchor_object_pull_posterior_weight_sum += float(
            outputs.get("loss_anchor_object_pull_posterior_weight_sum", outputs["loss_pt"] * 0.0).detach().item()
        )
        self.loss_anchor_object_pull_target_mass_mean += float(
            outputs.get("loss_anchor_object_pull_target_mass_mean", outputs["loss_pt"] * 0.0).detach().item()
        )
        self.loss_anchor_object_pull_target_quality_mean += float(
            outputs.get("loss_anchor_object_pull_target_quality_mean", outputs["loss_pt"] * 0.0).detach().item()
        )
        self.loss_pv_weak += float(outputs["loss_pv_weak"].detach().item())
        self.loss_pt += float(outputs["loss_pt"].detach().item())
        self.loss_vl_router += float(outputs.get("loss_vl_router", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_vl_heatmap_task += float(outputs.get("loss_vl_heatmap_task", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_vl_heatmap_effector += float(outputs.get("loss_vl_heatmap_effector", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_vl_heatmap_interaction += float(outputs.get("loss_vl_heatmap_interaction", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_vl_point_consistency += float(outputs.get("loss_vl_point_consistency", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_vl_anchor_diversity += float(outputs.get("loss_vl_anchor_diversity", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_graph += float(outputs.get("loss_mapg_graph", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_siglip += float(outputs.get("loss_mapg_siglip", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_vicreg += float(outputs.get("loss_mapg_vicreg", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_cycle += float(outputs.get("loss_mapg_cycle", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_masked_modality += float(outputs.get("loss_mapg_masked_modality", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_routing += float(outputs.get("loss_mapg_routing", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_support_diversity += float(outputs.get("loss_mapg_support_diversity", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_mapg_geometry_diversity += float(outputs.get("loss_mapg_geometry_diversity", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_jepa += float(outputs.get("loss_slot_jepa", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_jepa_direction += float(outputs.get("loss_slot_jepa_direction", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_jepa_log_norm += float(outputs.get("loss_slot_jepa_log_norm", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_jepa_pred_norm += float(outputs.get("loss_slot_jepa_pred_norm", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_jepa_target_norm += float(outputs.get("loss_slot_jepa_target_norm", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_jepa_matched_target_norm += float(
            outputs.get("loss_slot_jepa_matched_target_norm", outputs["loss_pt"] * 0.0).detach().item()
        )
        self.loss_support_pred += float(outputs.get("loss_support_pred", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_binding_consistency += float(outputs.get("loss_binding_consistency", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_aqr_denoising += float(outputs.get("loss_aqr_denoising", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_slot_quality += float(outputs.get("loss_slot_quality", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_vcap += float(outputs.get("loss_vcap", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_object_explanation += float(outputs.get("loss_object_explanation", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_object_explanation_feature += float(outputs.get("loss_object_explanation_feature", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_object_explanation_point += float(outputs.get("loss_object_explanation_point", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_object_explanation_contact += float(outputs.get("loss_object_explanation_contact", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_object_explanation_duplicate += float(outputs.get("loss_object_explanation_duplicate", outputs["loss_pt"] * 0.0).detach().item())
        self.loss_object_explanation_background += float(outputs.get("loss_object_explanation_background", outputs["loss_pt"] * 0.0).detach().item())
        self.physical_aux_budget_scale += float(outputs["physical_aux_budget_scale"].detach().item())
        self.semantic_aux_budget_scale += float(outputs["semantic_aux_budget_scale"].detach().item())
        self.alignment_budget_scale += float(outputs["alignment_budget_scale"].detach().item())
        self.candidate_density += float(outputs["projective_candidate_density"].detach().item())
        self.tactile_contact_prob_mean += float(outputs.get("tactile_contact_prob_mean", 0.0).detach().item())
        self.tactile_active_rate += float(outputs.get("tactile_active_rate", 0.0).detach().item())
        for key in OWM_DEBUG_METRIC_KEYS:
            value = outputs.get(key)
            if value is not None:
                self.owm_debug_metrics[key] = self.owm_debug_metrics.get(key, 0.0) + float(value.detach().item())
        self.num_windows += 1

    def averages(self) -> dict[str, float]:
        denom = max(self.num_windows, 1)
        return {
            "loss_total": self.loss_total / denom,
            "loss_action": self.loss_action / denom,
            "loss_action_default_equiv": self.loss_action_default_equiv / denom,
            "loss_action_weight_scale": self.loss_action_weight_scale / denom,
            "loss_action_active7": self.loss_action_active7 / denom,
            "loss_action_pos": self.loss_action_pos / denom,
            "loss_action_rot": self.loss_action_rot / denom,
            "loss_action_gripper": self.loss_action_gripper / denom,
            "loss_visual_latent": self.loss_visual_latent / denom,
            "loss_visual_real": self.loss_visual_real / denom,
            "loss_tactile_real": self.loss_tactile_real / denom,
            "loss_tactile_map": self.loss_tactile_map / denom,
            "loss_tactile_aux": self.loss_tactile_aux / denom,
            "loss_point_real": self.loss_point_real / denom,
            "loss_semantic_future_aux": self.loss_semantic_future_aux / denom,
            "loss_semantic_group_raw": self.loss_semantic_group_raw / denom,
            "loss_semantic_group_capped": self.loss_semantic_group_capped / denom,
            "loss_physical_aux": self.loss_physical_aux / denom,
            "loss_physical_aux_capped": self.loss_physical_aux_capped / denom,
            "loss_alignment": self.loss_alignment / denom,
            "loss_alignment_raw": self.loss_alignment_raw / denom,
            "loss_total_minus_action": self.loss_total_minus_action / denom,
            "loss_action_prefix_trust": self.loss_action_prefix_trust / denom,
            "loss_anchor_pv": self.loss_anchor_pv / denom,
            "loss_anchor_object_pull": self.loss_anchor_object_pull / denom,
            "loss_anchor_object_pull_graph": self.loss_anchor_object_pull_graph / denom,
            "loss_anchor_object_pull_posterior": self.loss_anchor_object_pull_posterior / denom,
            "loss_anchor_object_pull_graph_weight_sum": self.loss_anchor_object_pull_graph_weight_sum / denom,
            "loss_anchor_object_pull_posterior_weight_sum": self.loss_anchor_object_pull_posterior_weight_sum / denom,
            "loss_anchor_object_pull_target_mass_mean": self.loss_anchor_object_pull_target_mass_mean / denom,
            "loss_anchor_object_pull_target_quality_mean": self.loss_anchor_object_pull_target_quality_mean / denom,
            "loss_pv_weak": self.loss_pv_weak / denom,
            "loss_pt": self.loss_pt / denom,
            "loss_vl_router": self.loss_vl_router / denom,
            "loss_vl_heatmap_task": self.loss_vl_heatmap_task / denom,
            "loss_vl_heatmap_effector": self.loss_vl_heatmap_effector / denom,
            "loss_vl_heatmap_interaction": self.loss_vl_heatmap_interaction / denom,
            "loss_vl_point_consistency": self.loss_vl_point_consistency / denom,
            "loss_vl_anchor_diversity": self.loss_vl_anchor_diversity / denom,
            "loss_mapg_graph": self.loss_mapg_graph / denom,
            "loss_mapg_siglip": self.loss_mapg_siglip / denom,
            "loss_mapg_vicreg": self.loss_mapg_vicreg / denom,
            "loss_mapg_cycle": self.loss_mapg_cycle / denom,
            "loss_mapg_masked_modality": self.loss_mapg_masked_modality / denom,
            "loss_mapg_routing": self.loss_mapg_routing / denom,
            "loss_mapg_support_diversity": self.loss_mapg_support_diversity / denom,
            "loss_mapg_geometry_diversity": self.loss_mapg_geometry_diversity / denom,
            "loss_slot_jepa": self.loss_slot_jepa / denom,
            "loss_slot_jepa_direction": self.loss_slot_jepa_direction / denom,
            "loss_slot_jepa_log_norm": self.loss_slot_jepa_log_norm / denom,
            "loss_slot_jepa_pred_norm": self.loss_slot_jepa_pred_norm / denom,
            "loss_slot_jepa_target_norm": self.loss_slot_jepa_target_norm / denom,
            "loss_slot_jepa_matched_target_norm": self.loss_slot_jepa_matched_target_norm / denom,
            "loss_support_pred": self.loss_support_pred / denom,
            "loss_binding_consistency": self.loss_binding_consistency / denom,
            "loss_aqr_denoising": self.loss_aqr_denoising / denom,
            "loss_slot_quality": self.loss_slot_quality / denom,
            "loss_vcap": self.loss_vcap / denom,
            "loss_object_explanation": self.loss_object_explanation / denom,
            "loss_object_explanation_feature": self.loss_object_explanation_feature / denom,
            "loss_object_explanation_point": self.loss_object_explanation_point / denom,
            "loss_object_explanation_contact": self.loss_object_explanation_contact / denom,
            "loss_object_explanation_duplicate": self.loss_object_explanation_duplicate / denom,
            "loss_object_explanation_background": self.loss_object_explanation_background / denom,
            "physical_aux_budget_scale": self.physical_aux_budget_scale / denom,
            "semantic_aux_budget_scale": self.semantic_aux_budget_scale / denom,
            "alignment_budget_scale": self.alignment_budget_scale / denom,
            "projective_candidate_density": self.candidate_density / denom,
            "tactile_contact_prob_mean": self.tactile_contact_prob_mean / denom,
            "tactile_active_rate": self.tactile_active_rate / denom,
            **{key: self.owm_debug_metrics.get(key, 0.0) / denom for key in OWM_DEBUG_METRIC_KEYS},
        }


_BUCKET_LOG_OUTPUT_KEYS = (
    "loss_total",
    "loss_action_default_equiv",
    "loss_total_minus_action",
    "loss_anchor_pv",
    "loss_anchor_object_pull",
    "loss_mapg_routing",
    "loss_slot_jepa",
)


@dataclasses.dataclass
class _BucketMetricAccumulator:
    counts: dict[str, int] = dataclasses.field(default_factory=dict)
    totals: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)

    def update_from_outputs(self, bucket: str, outputs: dict[str, torch.Tensor]) -> None:
        bucket_key = str(bucket)
        self.counts[bucket_key] = int(self.counts.get(bucket_key, 0)) + 1
        bucket_totals = self.totals.setdefault(bucket_key, {})
        zero_reference = outputs.get("loss_total")
        for key in _BUCKET_LOG_OUTPUT_KEYS:
            value = outputs.get(key)
            if value is None:
                if zero_reference is None:
                    continue
                value = zero_reference * 0.0
            bucket_totals[key] = float(bucket_totals.get(key, 0.0)) + float(value.detach().item())

    def records(self) -> dict[str, dict[str, float]]:
        records: dict[str, dict[str, float]] = {}
        for bucket, count in sorted(self.counts.items()):
            denom = max(int(count), 1)
            values: dict[str, float] = {"count": float(denom)}
            for key, total in sorted(self.totals.get(bucket, {}).items()):
                values[key] = float(total) / float(denom)
            records[str(bucket)] = values
        return records


def _merge_bucket_metric_records(records: Sequence[dict[str, dict[str, float]]]) -> dict[str, dict[str, float]]:
    merged_counts: dict[str, float] = {}
    merged_totals: dict[str, dict[str, float]] = {}
    for record in records:
        for bucket, values in (record or {}).items():
            count = float(values.get("count", 0.0))
            if count <= 0.0:
                continue
            merged_counts[bucket] = float(merged_counts.get(bucket, 0.0)) + count
            totals = merged_totals.setdefault(bucket, {})
            for key, value in values.items():
                if key == "count":
                    continue
                totals[key] = float(totals.get(key, 0.0)) + float(value) * count
    merged: dict[str, dict[str, float]] = {}
    for bucket, count in sorted(merged_counts.items()):
        denom = max(float(count), 1.0)
        values = {"count": float(count)}
        for key, total in sorted(merged_totals.get(bucket, {}).items()):
            values[key] = float(total) / denom
        merged[bucket] = values
    return merged


def _flatten_bucket_metric_records(records: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
    flat: dict[str, float] = {}
    for bucket, values in sorted((records or {}).items()):
        bucket_fragment = _metric_key_fragment(str(bucket))
        for key, value in sorted(values.items()):
            key_fragment = _metric_key_fragment(str(key))
            flat[f"bucket_{bucket_fragment}_{key_fragment}"] = float(value)
    return flat


class _PicfWindowTrainer(torch.nn.Module):
    def __init__(
        self,
        core: PicfFullCore,
        *,
        semantic_encoder: torch.nn.Module | None,
        visual_grid: int,
        use_visual_override: bool,
        loss_config: PicfTransitionLossConfig | None = None,
        picf_mode: str = "enabled",
        burnin_steps: int = 0,
        burnin_mode: str = "full",
    ) -> None:
        super().__init__()
        self.core = core
        self.semantic_encoder = semantic_encoder
        self.picf_mode = str(picf_mode).lower().replace("-", "_")
        self.burnin_steps = int(burnin_steps)
        self.burnin_mode = str(burnin_mode).lower().replace("-", "_")
        self.policy = PicfPi05Policy(
            core=core,
            semantic_encoder=semantic_encoder,
            picf_enabled=_picf_mode_enabled(self.picf_mode),
        )
        self.visual_grid = int(visual_grid)
        self.use_visual_override = bool(use_visual_override)
        self.loss_config = loss_config or PicfTransitionLossConfig()

    def _future_targets_override_from_observed(self, observed: Any | None) -> Any | None:
        if observed is None:
            return None
        current_targets = getattr(observed, "current_targets", None)
        availability = getattr(observed, "availability", None)
        posterior = getattr(observed, "posterior", None)
        if current_targets is None or availability is None:
            return None
        return future_targets_from_current_targets(current_targets, availability, posterior=posterior)

    @staticmethod
    def _loss_metrics(
        losses: PicfTransitionLossBreakdown,
        *,
        candidate_density: torch.Tensor,
        tactile_contact_prob_mean: torch.Tensor,
        tactile_active_rate: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "loss_action": losses.action,
            "loss_action_default_equiv": losses.action_default_equiv,
            "loss_action_weight_scale": losses.action_weight_scale,
            "loss_action_active7": losses.action_active7,
            "loss_action_pos": losses.action_pos,
            "loss_action_rot": losses.action_rot,
            "loss_action_gripper": losses.action_gripper,
            "loss_visual_latent": losses.visual_latent,
            "loss_visual_real": losses.visual_real,
            "loss_tactile_real": losses.tactile_real,
            "loss_tactile_map": losses.tactile_map,
            "loss_tactile_aux": losses.tactile_aux,
            "loss_point_real": losses.point_real,
            "loss_semantic_future_aux": losses.semantic_future_aux,
            "loss_semantic_group_raw": losses.semantic_group_raw,
            "loss_semantic_group_capped": losses.semantic_group_capped,
            "loss_physical_aux": losses.physical_aux,
            "loss_physical_aux_capped": losses.physical_aux_capped,
            "loss_alignment": losses.alignment,
            "loss_alignment_raw": losses.alignment_raw,
            "loss_total_minus_action": losses.total_minus_action,
            "loss_action_prefix_trust": _loss_component_or_zero(losses, "action_prefix_trust"),
            "loss_anchor_pv": losses.anchor_pv,
            "loss_anchor_object_pull": losses.anchor_object_pull,
            "loss_anchor_object_pull_graph": _loss_component_or_zero(losses, "anchor_object_pull_graph"),
            "loss_anchor_object_pull_posterior": _loss_component_or_zero(losses, "anchor_object_pull_posterior"),
            "loss_anchor_object_pull_graph_weight_sum": _loss_component_or_zero(losses, "anchor_object_pull_graph_weight_sum"),
            "loss_anchor_object_pull_posterior_weight_sum": _loss_component_or_zero(
                losses, "anchor_object_pull_posterior_weight_sum"
            ),
            "loss_anchor_object_pull_target_mass_mean": _loss_component_or_zero(losses, "anchor_object_pull_target_mass_mean"),
            "loss_anchor_object_pull_target_quality_mean": _loss_component_or_zero(
                losses, "anchor_object_pull_target_quality_mean"
            ),
            "loss_pv_weak": losses.pv_weak,
            "loss_pt": losses.pt,
            "loss_vl_router": losses.vl_router,
            "loss_vl_heatmap_task": losses.vl_heatmap_task,
            "loss_vl_heatmap_effector": losses.vl_heatmap_effector,
            "loss_vl_heatmap_interaction": losses.vl_heatmap_interaction,
            "loss_vl_point_consistency": losses.vl_point_consistency,
            "loss_vl_anchor_diversity": losses.vl_anchor_diversity,
            "loss_mapg_graph": losses.mapg_graph,
            "loss_mapg_siglip": losses.mapg_siglip,
            "loss_mapg_vicreg": losses.mapg_vicreg,
            "loss_mapg_cycle": losses.mapg_cycle,
            "loss_mapg_masked_modality": losses.mapg_masked_modality,
            "loss_mapg_routing": losses.mapg_routing,
            "loss_mapg_support_diversity": losses.mapg_support_diversity,
            "loss_mapg_geometry_diversity": losses.mapg_geometry_diversity,
            "loss_slot_jepa": _loss_component_or_zero(losses, "slot_jepa"),
            "loss_slot_jepa_direction": _loss_component_or_zero(losses, "slot_jepa_direction"),
            "loss_slot_jepa_log_norm": _loss_component_or_zero(losses, "slot_jepa_log_norm"),
            "loss_slot_jepa_pred_norm": _loss_component_or_zero(losses, "slot_jepa_pred_norm"),
            "loss_slot_jepa_target_norm": _loss_component_or_zero(losses, "slot_jepa_target_norm"),
            "loss_slot_jepa_matched_target_norm": _loss_component_or_zero(losses, "slot_jepa_matched_target_norm"),
            "loss_support_pred": _loss_component_or_zero(losses, "support_pred"),
            "loss_binding_consistency": _loss_component_or_zero(losses, "binding_consistency"),
            "loss_aqr_denoising": _loss_component_or_zero(losses, "aqr_denoising"),
            "loss_slot_quality": _loss_component_or_zero(losses, "slot_quality"),
            "loss_vcap": _loss_component_or_zero(losses, "vcap"),
            "loss_object_explanation": _loss_component_or_zero(losses, "object_explanation"),
            "loss_object_explanation_feature": _loss_component_or_zero(losses, "object_explanation_feature"),
            "loss_object_explanation_point": _loss_component_or_zero(losses, "object_explanation_point"),
            "loss_object_explanation_contact": _loss_component_or_zero(losses, "object_explanation_contact"),
            "loss_object_explanation_duplicate": _loss_component_or_zero(losses, "object_explanation_duplicate"),
            "loss_object_explanation_background": _loss_component_or_zero(losses, "object_explanation_background"),
            "physical_aux_budget_scale": losses.physical_aux_budget_scale,
            "semantic_aux_budget_scale": losses.semantic_aux_budget_scale,
            "alignment_budget_scale": losses.alignment_budget_scale,
            "projective_candidate_density": candidate_density,
            "tactile_contact_prob_mean": tactile_contact_prob_mean,
            "tactile_active_rate": tactile_active_rate,
        }

    @staticmethod
    def _accumulate_loss_metrics(
        metrics: dict[str, torch.Tensor] | None,
        update: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if metrics is None:
            return update
        for key, value in update.items():
            metrics[key] = metrics[key] + value
        return metrics

    def _forward_action_only_window(
        self,
        window: _TransitionWindow,
        *,
        capture_visual_diagnostics: bool = False,
        capture_anchor_overlay: bool = False,
        capture_anchor_overlay_signatures: bool = False,
        debug_phase_label: str | None = None,
    ) -> dict[str, Any]:
        totals: list[torch.Tensor] = []
        metrics: dict[str, torch.Tensor] | None = None
        for index in range(len(window.frames) - 1):
            if debug_phase_label is not None:
                logging.info("%s transition=%s begin", debug_phase_label, index)
            current = dataclasses.replace(window.frames[index], reset_scaffold=(index == 0))
            step_start = time.perf_counter()
            action_chunk_target = current.action_chunk if current.action_chunk is not None else current.action
            policy_forward = self.policy.forward_train_transition(
                previous=None,
                current=current,
                visual_map_override=None,
                action_chunk_target=action_chunk_target,
            )
            policy_forward_sec = time.perf_counter() - step_start
            if debug_phase_label is not None:
                logging.info(
                    "%s transition=%s policy_forward_sec=%.3f",
                    debug_phase_label,
                    index,
                    policy_forward_sec,
                )
            if policy_forward.flow_override is None:
                raise RuntimeError(
                    "picf_mode=ablated expects PI0.5 flow loss outputs on every training transition."
                )
            reference = policy_forward.flow_override["predicted_action"]
            zero = torch.zeros((), device=reference.device, dtype=reference.dtype)
            losses = make_action_only_transition_loss(
                reference=reference,
                action_loss_override=_flow_training_total(policy_forward.flow_override),
                action_default_equiv_override=_flow_default_equiv(policy_forward.flow_override),
                action_pos_override=_flow_training_component(policy_forward.flow_override, "action_pos"),
                action_rot_override=_flow_training_component(policy_forward.flow_override, "action_rot"),
                action_gripper_override=_flow_training_component(policy_forward.flow_override, "action_gripper"),
            )
            totals.append(losses.total)
            metrics = self._accumulate_loss_metrics(
                metrics,
                self._loss_metrics(
                    losses,
                    candidate_density=zero,
                    tactile_contact_prob_mean=zero,
                    tactile_active_rate=zero,
                ),
            )
        assert metrics is not None
        denom = float(len(window.frames) - 1)
        result: dict[str, Any] = {
            "loss_total": torch.stack(totals).mean(),
            "loss_action": metrics["loss_action"] / denom,
            "loss_action_default_equiv": metrics["loss_action_default_equiv"] / denom,
            "loss_action_weight_scale": metrics["loss_action_weight_scale"] / denom,
            "loss_action_active7": metrics["loss_action_active7"] / denom,
            "loss_action_pos": metrics["loss_action_pos"] / denom,
            "loss_action_rot": metrics["loss_action_rot"] / denom,
            "loss_action_gripper": metrics["loss_action_gripper"] / denom,
            "loss_visual_latent": metrics["loss_visual_latent"] / denom,
            "loss_visual_real": metrics["loss_visual_real"] / denom,
            "loss_tactile_real": metrics["loss_tactile_real"] / denom,
            "loss_tactile_map": metrics["loss_tactile_map"] / denom,
            "loss_tactile_aux": metrics["loss_tactile_aux"] / denom,
            "loss_point_real": metrics["loss_point_real"] / denom,
            "loss_semantic_future_aux": metrics["loss_semantic_future_aux"] / denom,
            "loss_semantic_group_raw": metrics["loss_semantic_group_raw"] / denom,
            "loss_semantic_group_capped": metrics["loss_semantic_group_capped"] / denom,
            "loss_physical_aux": metrics["loss_physical_aux"] / denom,
            "loss_physical_aux_capped": metrics["loss_physical_aux_capped"] / denom,
            "loss_alignment": metrics["loss_alignment"] / denom,
            "loss_alignment_raw": metrics["loss_alignment_raw"] / denom,
            "loss_total_minus_action": metrics["loss_total_minus_action"] / denom,
            "loss_action_prefix_trust": metrics["loss_action_prefix_trust"] / denom,
            "loss_anchor_pv": metrics["loss_anchor_pv"] / denom,
            "loss_anchor_object_pull": metrics["loss_anchor_object_pull"] / denom,
            "loss_anchor_object_pull_graph": metrics["loss_anchor_object_pull_graph"] / denom,
            "loss_anchor_object_pull_posterior": metrics["loss_anchor_object_pull_posterior"] / denom,
            "loss_anchor_object_pull_graph_weight_sum": metrics["loss_anchor_object_pull_graph_weight_sum"] / denom,
            "loss_anchor_object_pull_posterior_weight_sum": metrics["loss_anchor_object_pull_posterior_weight_sum"] / denom,
            "loss_anchor_object_pull_target_mass_mean": metrics["loss_anchor_object_pull_target_mass_mean"] / denom,
            "loss_anchor_object_pull_target_quality_mean": metrics["loss_anchor_object_pull_target_quality_mean"] / denom,
            "loss_pv_weak": metrics["loss_pv_weak"] / denom,
            "loss_pt": metrics["loss_pt"] / denom,
            "loss_vl_router": metrics["loss_vl_router"] / denom,
            "loss_vl_heatmap_task": metrics["loss_vl_heatmap_task"] / denom,
            "loss_vl_heatmap_effector": metrics["loss_vl_heatmap_effector"] / denom,
            "loss_vl_heatmap_interaction": metrics["loss_vl_heatmap_interaction"] / denom,
            "loss_vl_point_consistency": metrics["loss_vl_point_consistency"] / denom,
            "loss_vl_anchor_diversity": metrics["loss_vl_anchor_diversity"] / denom,
            "loss_mapg_graph": metrics["loss_mapg_graph"] / denom,
            "loss_mapg_siglip": metrics["loss_mapg_siglip"] / denom,
            "loss_mapg_vicreg": metrics["loss_mapg_vicreg"] / denom,
            "loss_mapg_cycle": metrics["loss_mapg_cycle"] / denom,
            "loss_mapg_masked_modality": metrics["loss_mapg_masked_modality"] / denom,
            "loss_mapg_routing": metrics["loss_mapg_routing"] / denom,
            "loss_mapg_support_diversity": metrics["loss_mapg_support_diversity"] / denom,
            "loss_mapg_geometry_diversity": metrics["loss_mapg_geometry_diversity"] / denom,
            "loss_slot_jepa": metrics["loss_slot_jepa"] / denom,
            "loss_support_pred": metrics["loss_support_pred"] / denom,
            "loss_binding_consistency": metrics["loss_binding_consistency"] / denom,
            "loss_aqr_denoising": metrics["loss_aqr_denoising"] / denom,
            "loss_slot_quality": metrics["loss_slot_quality"] / denom,
            "loss_vcap": metrics["loss_vcap"] / denom,
            "loss_object_explanation": metrics["loss_object_explanation"] / denom,
            "loss_object_explanation_feature": metrics["loss_object_explanation_feature"] / denom,
            "loss_object_explanation_point": metrics["loss_object_explanation_point"] / denom,
            "loss_object_explanation_contact": metrics["loss_object_explanation_contact"] / denom,
            "loss_object_explanation_duplicate": metrics["loss_object_explanation_duplicate"] / denom,
            "loss_object_explanation_background": metrics["loss_object_explanation_background"] / denom,
            "physical_aux_budget_scale": metrics["physical_aux_budget_scale"] / denom,
            "semantic_aux_budget_scale": metrics["semantic_aux_budget_scale"] / denom,
            "alignment_budget_scale": metrics["alignment_budget_scale"] / denom,
            "projective_candidate_density": metrics["projective_candidate_density"] / denom,
            "tactile_contact_prob_mean": metrics["tactile_contact_prob_mean"] / denom,
            "tactile_active_rate": metrics["tactile_active_rate"] / denom,
        }
        zero_metric = result["loss_total"] * 0.0
        for key in OWM_DEBUG_METRIC_KEYS:
            result[key] = zero_metric
        if capture_visual_diagnostics:
            result["diagnostic_physical_visual_real_seq"] = []
            result["diagnostic_semantic_visual_real_seq"] = []
        if capture_anchor_overlay:
            result["diagnostic_anchor_overlay"] = None
        return result

    def forward(
        self,
        window: _TransitionWindow,
        *,
        capture_visual_diagnostics: bool = False,
        capture_anchor_overlay: bool = False,
        capture_anchor_overlay_signatures: bool = False,
        debug_phase_label: str | None = None,
    ) -> dict[str, Any]:
        if not bool(getattr(self.policy, "picf_enabled", True)):
            return self._forward_action_only_window(
                window,
                capture_visual_diagnostics=capture_visual_diagnostics,
                capture_anchor_overlay=capture_anchor_overlay,
                capture_anchor_overlay_signatures=capture_anchor_overlay_signatures,
                debug_phase_label=debug_phase_label,
            )
        previous = None
        metrics: dict[str, torch.Tensor] | None = None
        totals: list[torch.Tensor] = []
        physical_visual_real_seq: list[torch.Tensor | None] = []
        semantic_visual_real_seq: list[torch.Tensor | None] = []
        anchor_overlay_snapshot: dict[str, Any] | None = None
        pending: _PendingTransitionLoss | None = None
        transition_count = len(window.frames) - 1
        train_start_index = min(max(int(self.burnin_steps), 0), transition_count - 1)
        for index in range(transition_count):
            if debug_phase_label is not None:
                logging.info("%s transition=%s begin", debug_phase_label, index)
            current = dataclasses.replace(window.frames[index], reset_scaffold=(index == 0))
            nxt = dataclasses.replace(window.frames[index + 1], reset_scaffold=False)
            current_visual = _rgb_visual_override(current.rgb_static, grid=self.visual_grid) if self.use_visual_override else None
            next_visual = _rgb_visual_override(nxt.rgb_static, grid=self.visual_grid) if self.use_visual_override else None
            if index < train_start_index:
                step_start = time.perf_counter()
                with torch.no_grad():
                    if self.burnin_mode == "state_only":
                        burnin_state = self.policy.burnin_recurrent_transition(
                            previous=previous,
                            current=current,
                            visual_map_override=current_visual,
                        )
                        burnin_forward = None
                    else:
                        burnin_forward = self.policy.forward_train_transition(
                            previous=previous,
                            current=current,
                            visual_map_override=current_visual,
                            action_chunk_target=None,
                        )
                        burnin_state = burnin_forward.next_state
                burnin_forward_sec = time.perf_counter() - step_start
                if debug_phase_label is not None:
                    logging.info(
                        "%s transition=%s burnin_mode=%s burnin_forward_sec=%.3f",
                        debug_phase_label,
                        index,
                        self.burnin_mode,
                        burnin_forward_sec,
                    )
                if burnin_state is None:
                    raise RuntimeError("PICF burn-in transition did not produce a recurrent state.")
                if capture_visual_diagnostics and burnin_forward is not None:
                    physical_visual_real = burnin_forward.output.state.predictive.physical_prediction_cache.visual_real
                    semantic_visual_real = burnin_forward.output.state.predictive.prediction_cache.visual_real
                    physical_visual_real_seq.append(
                        None if physical_visual_real is None else physical_visual_real.detach().to(device="cpu")
                    )
                    semantic_visual_real_seq.append(
                        None if semantic_visual_real is None else semantic_visual_real.detach().to(device="cpu")
                    )
                elif capture_visual_diagnostics:
                    physical_visual_real = burnin_state.predictive.physical_prediction_cache.visual_real
                    physical_visual_real_seq.append(
                        None if physical_visual_real is None else physical_visual_real.detach().to(device="cpu")
                    )
                    semantic_visual_real_seq.append(None)
                previous = burnin_state
                continue
            step_start = time.perf_counter()
            action_chunk_target = current.action_chunk if current.action_chunk is not None else current.action
            policy_forward = self.policy.forward_train_transition(
                previous=previous,
                current=current,
                visual_map_override=current_visual,
                action_chunk_target=action_chunk_target,
            )
            output = policy_forward.output
            if capture_anchor_overlay:
                anchor_overlay_snapshot = _anchor_overlay_snapshot_from_output(
                    output,
                    current,
                    dump_signatures=capture_anchor_overlay_signatures,
                )
            flow_override = policy_forward.flow_override
            policy_forward_sec = time.perf_counter() - step_start
            if debug_phase_label is not None:
                logging.info(
                    "%s transition=%s policy_forward_sec=%.3f",
                    debug_phase_label,
                    index,
                    policy_forward_sec,
                )
            if capture_visual_diagnostics:
                physical_visual_real = output.state.predictive.physical_prediction_cache.visual_real
                semantic_visual_real = output.state.predictive.prediction_cache.visual_real
                physical_visual_real_seq.append(
                    None if physical_visual_real is None else physical_visual_real.detach().to(device="cpu")
                )
                semantic_visual_real_seq.append(
                    None if semantic_visual_real is None else semantic_visual_real.detach().to(device="cpu")
                )
            candidate_density = torch.as_tensor(
                float(output.debug.get("projective_candidate_density", 0.0)),
                device=self.core.device,
                dtype=self.core.dtype,
            )
            tactile_contact_prob_mean = torch.as_tensor(
                float(output.debug.get("tactile_contact_prob_mean", 0.0)),
                device=self.core.device,
                dtype=self.core.dtype,
            )
            tactile_active_rate = torch.as_tensor(
                float(output.debug.get("tactile_active_rate", 0.0)),
                device=self.core.device,
                dtype=self.core.dtype,
            )
            if pending is not None:
                loss_start = time.perf_counter()
                losses = compute_transition_loss(
                    self.core,
                    pending.output,
                    current,
                    action_target=pending.action_target,
                    next_visual_map_override=pending.next_visual_map_override,
                    config=self.loss_config,
                    action_loss_override=_flow_training_total(pending.flow_override),
                    action_default_equiv_override=_flow_default_equiv(pending.flow_override),
                    action_pos_override=_flow_training_component(pending.flow_override, "action_pos"),
                    action_rot_override=_flow_training_component(pending.flow_override, "action_rot"),
                    action_gripper_override=_flow_training_component(pending.flow_override, "action_gripper"),
                    action_prefix_trust_override=(
                        None
                        if pending.flow_override is None
                        else pending.flow_override.get("picf_action_prefix_trust_loss")
                    ),
                    future_targets_override=self._future_targets_override_from_observed(policy_forward.observed),
                )
                loss_sec = time.perf_counter() - loss_start
                if debug_phase_label is not None:
                    logging.info(
                        "%s transition=%s loss_sec=%.3f total_transition_sec=%.3f",
                        debug_phase_label,
                        pending.index,
                        loss_sec,
                        pending.policy_forward_sec + loss_sec,
                    )
                totals.append(losses.total)
                metrics_candidate_density = pending.candidate_density
                metrics_tactile_contact_prob_mean = pending.tactile_contact_prob_mean
                metrics_tactile_active_rate = pending.tactile_active_rate
                metrics_owm_debug_metrics = pending.owm_debug_metrics
            else:
                losses = None
                metrics_candidate_density = None
                metrics_tactile_contact_prob_mean = None
                metrics_tactile_active_rate = None
                metrics_owm_debug_metrics = None
            pending = _PendingTransitionLoss(
                index=index,
                output=output,
                next_observation=nxt,
                next_visual_map_override=next_visual,
                action_target=current.action,
                flow_override=flow_override,
                candidate_density=candidate_density,
                tactile_contact_prob_mean=tactile_contact_prob_mean,
                tactile_active_rate=tactile_active_rate,
                owm_debug_metrics=_owm_debug_metrics_from_output(output, candidate_density),
                policy_forward_sec=policy_forward_sec,
            )
            if metrics is None:
                if losses is None:
                    previous = policy_forward.next_state
                    continue
                metrics = {
                    "loss_action": losses.action,
                    "loss_action_default_equiv": losses.action_default_equiv,
                    "loss_action_weight_scale": losses.action_weight_scale,
                    "loss_action_active7": losses.action_active7,
                    "loss_action_pos": losses.action_pos,
                    "loss_action_rot": losses.action_rot,
                    "loss_action_gripper": losses.action_gripper,
                    "loss_visual_latent": losses.visual_latent,
                    "loss_visual_real": losses.visual_real,
                    "loss_tactile_real": losses.tactile_real,
                    "loss_tactile_map": losses.tactile_map,
                    "loss_tactile_aux": losses.tactile_aux,
                    "loss_point_real": losses.point_real,
                    "loss_semantic_future_aux": losses.semantic_future_aux,
                    "loss_semantic_group_raw": losses.semantic_group_raw,
                    "loss_semantic_group_capped": losses.semantic_group_capped,
                    "loss_physical_aux": losses.physical_aux,
                    "loss_physical_aux_capped": losses.physical_aux_capped,
                    "loss_alignment": losses.alignment,
                    "loss_alignment_raw": losses.alignment_raw,
                    "loss_total_minus_action": losses.total_minus_action,
                    "loss_action_prefix_trust": _loss_component_or_zero(losses, "action_prefix_trust"),
                    "loss_anchor_pv": losses.anchor_pv,
                    "loss_anchor_object_pull": losses.anchor_object_pull,
                    "loss_anchor_object_pull_graph": _loss_component_or_zero(losses, "anchor_object_pull_graph"),
                    "loss_anchor_object_pull_posterior": _loss_component_or_zero(losses, "anchor_object_pull_posterior"),
                    "loss_anchor_object_pull_graph_weight_sum": _loss_component_or_zero(losses, "anchor_object_pull_graph_weight_sum"),
                    "loss_anchor_object_pull_posterior_weight_sum": _loss_component_or_zero(
                        losses, "anchor_object_pull_posterior_weight_sum"
                    ),
                    "loss_anchor_object_pull_target_mass_mean": _loss_component_or_zero(
                        losses, "anchor_object_pull_target_mass_mean"
                    ),
                    "loss_anchor_object_pull_target_quality_mean": _loss_component_or_zero(
                        losses, "anchor_object_pull_target_quality_mean"
                    ),
                    "loss_pv_weak": losses.pv_weak,
                    "loss_pt": losses.pt,
                    "loss_vl_router": losses.vl_router,
                    "loss_vl_heatmap_task": losses.vl_heatmap_task,
                    "loss_vl_heatmap_effector": losses.vl_heatmap_effector,
                    "loss_vl_heatmap_interaction": losses.vl_heatmap_interaction,
                    "loss_vl_point_consistency": losses.vl_point_consistency,
                    "loss_vl_anchor_diversity": losses.vl_anchor_diversity,
                    "loss_mapg_graph": losses.mapg_graph,
                    "loss_mapg_siglip": losses.mapg_siglip,
                    "loss_mapg_vicreg": losses.mapg_vicreg,
                    "loss_mapg_cycle": losses.mapg_cycle,
                    "loss_mapg_masked_modality": losses.mapg_masked_modality,
                    "loss_mapg_routing": losses.mapg_routing,
                    "loss_mapg_support_diversity": losses.mapg_support_diversity,
                    "loss_mapg_geometry_diversity": losses.mapg_geometry_diversity,
                    "loss_slot_jepa": _loss_component_or_zero(losses, "slot_jepa"),
                    "loss_slot_jepa_direction": _loss_component_or_zero(losses, "slot_jepa_direction"),
                    "loss_slot_jepa_log_norm": _loss_component_or_zero(losses, "slot_jepa_log_norm"),
                    "loss_slot_jepa_pred_norm": _loss_component_or_zero(losses, "slot_jepa_pred_norm"),
                    "loss_slot_jepa_target_norm": _loss_component_or_zero(losses, "slot_jepa_target_norm"),
                    "loss_slot_jepa_matched_target_norm": _loss_component_or_zero(losses, "slot_jepa_matched_target_norm"),
                    "loss_support_pred": _loss_component_or_zero(losses, "support_pred"),
                    "loss_binding_consistency": _loss_component_or_zero(losses, "binding_consistency"),
                    "loss_aqr_denoising": _loss_component_or_zero(losses, "aqr_denoising"),
                    "loss_slot_quality": _loss_component_or_zero(losses, "slot_quality"),
                    "loss_vcap": _loss_component_or_zero(losses, "vcap"),
                    "loss_object_explanation": _loss_component_or_zero(losses, "object_explanation"),
                    "loss_object_explanation_feature": _loss_component_or_zero(losses, "object_explanation_feature"),
                    "loss_object_explanation_point": _loss_component_or_zero(losses, "object_explanation_point"),
                    "loss_object_explanation_contact": _loss_component_or_zero(losses, "object_explanation_contact"),
                    "loss_object_explanation_duplicate": _loss_component_or_zero(losses, "object_explanation_duplicate"),
                    "loss_object_explanation_background": _loss_component_or_zero(losses, "object_explanation_background"),
                    "physical_aux_budget_scale": losses.physical_aux_budget_scale,
                    "semantic_aux_budget_scale": losses.semantic_aux_budget_scale,
                    "alignment_budget_scale": losses.alignment_budget_scale,
                    "projective_candidate_density": metrics_candidate_density,
                    "tactile_contact_prob_mean": metrics_tactile_contact_prob_mean,
                    "tactile_active_rate": metrics_tactile_active_rate,
                }
                _accumulate_owm_debug_metrics(metrics, metrics_owm_debug_metrics or {})
            else:
                if losses is None:
                    previous = policy_forward.next_state
                    continue
                metrics["loss_action"] = metrics["loss_action"] + losses.action
                metrics["loss_action_default_equiv"] = metrics["loss_action_default_equiv"] + losses.action_default_equiv
                metrics["loss_action_weight_scale"] = metrics["loss_action_weight_scale"] + losses.action_weight_scale
                metrics["loss_action_active7"] = metrics["loss_action_active7"] + losses.action_active7
                metrics["loss_action_pos"] = metrics["loss_action_pos"] + losses.action_pos
                metrics["loss_action_rot"] = metrics["loss_action_rot"] + losses.action_rot
                metrics["loss_action_gripper"] = metrics["loss_action_gripper"] + losses.action_gripper
                metrics["loss_visual_latent"] = metrics["loss_visual_latent"] + losses.visual_latent
                metrics["loss_visual_real"] = metrics["loss_visual_real"] + losses.visual_real
                metrics["loss_tactile_real"] = metrics["loss_tactile_real"] + losses.tactile_real
                metrics["loss_tactile_map"] = metrics["loss_tactile_map"] + losses.tactile_map
                metrics["loss_tactile_aux"] = metrics["loss_tactile_aux"] + losses.tactile_aux
                metrics["loss_point_real"] = metrics["loss_point_real"] + losses.point_real
                metrics["loss_semantic_future_aux"] = metrics["loss_semantic_future_aux"] + losses.semantic_future_aux
                metrics["loss_semantic_group_raw"] = metrics["loss_semantic_group_raw"] + losses.semantic_group_raw
                metrics["loss_semantic_group_capped"] = metrics["loss_semantic_group_capped"] + losses.semantic_group_capped
                metrics["loss_physical_aux"] = metrics["loss_physical_aux"] + losses.physical_aux
                metrics["loss_physical_aux_capped"] = metrics["loss_physical_aux_capped"] + losses.physical_aux_capped
                metrics["loss_alignment"] = metrics["loss_alignment"] + losses.alignment
                metrics["loss_alignment_raw"] = metrics["loss_alignment_raw"] + losses.alignment_raw
                metrics["loss_total_minus_action"] = metrics["loss_total_minus_action"] + losses.total_minus_action
                metrics["loss_action_prefix_trust"] = metrics["loss_action_prefix_trust"] + _loss_component_or_zero(
                    losses, "action_prefix_trust"
                )
                metrics["loss_anchor_pv"] = metrics["loss_anchor_pv"] + losses.anchor_pv
                metrics["loss_anchor_object_pull"] = metrics["loss_anchor_object_pull"] + losses.anchor_object_pull
                metrics["loss_anchor_object_pull_graph"] = metrics["loss_anchor_object_pull_graph"] + _loss_component_or_zero(
                    losses, "anchor_object_pull_graph"
                )
                metrics["loss_anchor_object_pull_posterior"] = (
                    metrics["loss_anchor_object_pull_posterior"] + _loss_component_or_zero(losses, "anchor_object_pull_posterior")
                )
                metrics["loss_anchor_object_pull_graph_weight_sum"] = (
                    metrics["loss_anchor_object_pull_graph_weight_sum"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_graph_weight_sum")
                )
                metrics["loss_anchor_object_pull_posterior_weight_sum"] = (
                    metrics["loss_anchor_object_pull_posterior_weight_sum"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_posterior_weight_sum")
                )
                metrics["loss_anchor_object_pull_target_mass_mean"] = (
                    metrics["loss_anchor_object_pull_target_mass_mean"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_target_mass_mean")
                )
                metrics["loss_anchor_object_pull_target_quality_mean"] = (
                    metrics["loss_anchor_object_pull_target_quality_mean"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_target_quality_mean")
                )
                metrics["loss_pv_weak"] = metrics["loss_pv_weak"] + losses.pv_weak
                metrics["loss_pt"] = metrics["loss_pt"] + losses.pt
                metrics["loss_vl_router"] = metrics["loss_vl_router"] + losses.vl_router
                metrics["loss_vl_heatmap_task"] = metrics["loss_vl_heatmap_task"] + losses.vl_heatmap_task
                metrics["loss_vl_heatmap_effector"] = metrics["loss_vl_heatmap_effector"] + losses.vl_heatmap_effector
                metrics["loss_vl_heatmap_interaction"] = metrics["loss_vl_heatmap_interaction"] + losses.vl_heatmap_interaction
                metrics["loss_vl_point_consistency"] = metrics["loss_vl_point_consistency"] + losses.vl_point_consistency
                metrics["loss_vl_anchor_diversity"] = metrics["loss_vl_anchor_diversity"] + losses.vl_anchor_diversity
                metrics["loss_mapg_graph"] = metrics["loss_mapg_graph"] + losses.mapg_graph
                metrics["loss_mapg_siglip"] = metrics["loss_mapg_siglip"] + losses.mapg_siglip
                metrics["loss_mapg_vicreg"] = metrics["loss_mapg_vicreg"] + losses.mapg_vicreg
                metrics["loss_mapg_cycle"] = metrics["loss_mapg_cycle"] + losses.mapg_cycle
                metrics["loss_mapg_masked_modality"] = metrics["loss_mapg_masked_modality"] + losses.mapg_masked_modality
                metrics["loss_mapg_routing"] = metrics["loss_mapg_routing"] + losses.mapg_routing
                metrics["loss_mapg_support_diversity"] = metrics["loss_mapg_support_diversity"] + losses.mapg_support_diversity
                metrics["loss_mapg_geometry_diversity"] = metrics["loss_mapg_geometry_diversity"] + losses.mapg_geometry_diversity
                metrics["loss_slot_jepa"] = metrics["loss_slot_jepa"] + _loss_component_or_zero(losses, "slot_jepa")
                metrics["loss_slot_jepa_direction"] = metrics["loss_slot_jepa_direction"] + _loss_component_or_zero(losses, "slot_jepa_direction")
                metrics["loss_slot_jepa_log_norm"] = metrics["loss_slot_jepa_log_norm"] + _loss_component_or_zero(losses, "slot_jepa_log_norm")
                metrics["loss_slot_jepa_pred_norm"] = metrics["loss_slot_jepa_pred_norm"] + _loss_component_or_zero(losses, "slot_jepa_pred_norm")
                metrics["loss_slot_jepa_target_norm"] = metrics["loss_slot_jepa_target_norm"] + _loss_component_or_zero(losses, "slot_jepa_target_norm")
                metrics["loss_slot_jepa_matched_target_norm"] = (
                    metrics["loss_slot_jepa_matched_target_norm"]
                    + _loss_component_or_zero(losses, "slot_jepa_matched_target_norm")
                )
                metrics["loss_support_pred"] = metrics["loss_support_pred"] + _loss_component_or_zero(losses, "support_pred")
                metrics["loss_binding_consistency"] = metrics["loss_binding_consistency"] + _loss_component_or_zero(
                    losses, "binding_consistency"
                )
                metrics["loss_aqr_denoising"] = metrics["loss_aqr_denoising"] + _loss_component_or_zero(
                    losses, "aqr_denoising"
                )
                metrics["loss_slot_quality"] = metrics["loss_slot_quality"] + _loss_component_or_zero(
                    losses, "slot_quality"
                )
                metrics["loss_vcap"] = metrics["loss_vcap"] + _loss_component_or_zero(losses, "vcap")
                metrics["loss_object_explanation"] = metrics["loss_object_explanation"] + _loss_component_or_zero(
                    losses, "object_explanation"
                )
                metrics["loss_object_explanation_feature"] = metrics["loss_object_explanation_feature"] + _loss_component_or_zero(
                    losses, "object_explanation_feature"
                )
                metrics["loss_object_explanation_point"] = metrics["loss_object_explanation_point"] + _loss_component_or_zero(
                    losses, "object_explanation_point"
                )
                metrics["loss_object_explanation_contact"] = metrics["loss_object_explanation_contact"] + _loss_component_or_zero(
                    losses, "object_explanation_contact"
                )
                metrics["loss_object_explanation_duplicate"] = metrics["loss_object_explanation_duplicate"] + _loss_component_or_zero(
                    losses, "object_explanation_duplicate"
                )
                metrics["loss_object_explanation_background"] = metrics["loss_object_explanation_background"] + _loss_component_or_zero(
                    losses, "object_explanation_background"
                )
                metrics["physical_aux_budget_scale"] = metrics["physical_aux_budget_scale"] + losses.physical_aux_budget_scale
                metrics["semantic_aux_budget_scale"] = metrics["semantic_aux_budget_scale"] + losses.semantic_aux_budget_scale
                metrics["alignment_budget_scale"] = metrics["alignment_budget_scale"] + losses.alignment_budget_scale
                metrics["projective_candidate_density"] = metrics["projective_candidate_density"] + metrics_candidate_density
                metrics["tactile_contact_prob_mean"] = metrics["tactile_contact_prob_mean"] + metrics_tactile_contact_prob_mean
                metrics["tactile_active_rate"] = metrics["tactile_active_rate"] + metrics_tactile_active_rate
                _accumulate_owm_debug_metrics(metrics, metrics_owm_debug_metrics or {})
            previous = policy_forward.next_state

        if pending is not None:
            loss_start = time.perf_counter()
            losses = compute_transition_loss(
                self.core,
                pending.output,
                pending.next_observation,
                action_target=pending.action_target,
                next_visual_map_override=pending.next_visual_map_override,
                config=self.loss_config,
                action_loss_override=_flow_training_total(pending.flow_override),
                action_default_equiv_override=_flow_default_equiv(pending.flow_override),
                action_pos_override=_flow_training_component(pending.flow_override, "action_pos"),
                action_rot_override=_flow_training_component(pending.flow_override, "action_rot"),
                action_gripper_override=_flow_training_component(pending.flow_override, "action_gripper"),
                action_prefix_trust_override=(
                    None
                    if pending.flow_override is None
                    else pending.flow_override.get("picf_action_prefix_trust_loss")
                ),
            )
            loss_sec = time.perf_counter() - loss_start
            if debug_phase_label is not None:
                logging.info(
                    "%s transition=%s loss_sec=%.3f total_transition_sec=%.3f",
                    debug_phase_label,
                    pending.index,
                    loss_sec,
                    pending.policy_forward_sec + loss_sec,
                )
            totals.append(losses.total)
            if metrics is None:
                metrics = {
                    "loss_action": losses.action,
                    "loss_action_default_equiv": losses.action_default_equiv,
                    "loss_action_weight_scale": losses.action_weight_scale,
                    "loss_action_active7": losses.action_active7,
                    "loss_action_pos": losses.action_pos,
                    "loss_action_rot": losses.action_rot,
                    "loss_action_gripper": losses.action_gripper,
                    "loss_visual_latent": losses.visual_latent,
                    "loss_visual_real": losses.visual_real,
                    "loss_tactile_real": losses.tactile_real,
                    "loss_tactile_map": losses.tactile_map,
                    "loss_tactile_aux": losses.tactile_aux,
                    "loss_point_real": losses.point_real,
                    "loss_semantic_future_aux": losses.semantic_future_aux,
                    "loss_semantic_group_raw": losses.semantic_group_raw,
                    "loss_semantic_group_capped": losses.semantic_group_capped,
                    "loss_physical_aux": losses.physical_aux,
                    "loss_physical_aux_capped": losses.physical_aux_capped,
                    "loss_alignment": losses.alignment,
                    "loss_alignment_raw": losses.alignment_raw,
                    "loss_total_minus_action": losses.total_minus_action,
                    "loss_action_prefix_trust": _loss_component_or_zero(losses, "action_prefix_trust"),
                    "loss_anchor_pv": losses.anchor_pv,
                    "loss_anchor_object_pull": losses.anchor_object_pull,
                    "loss_anchor_object_pull_graph": _loss_component_or_zero(losses, "anchor_object_pull_graph"),
                    "loss_anchor_object_pull_posterior": _loss_component_or_zero(losses, "anchor_object_pull_posterior"),
                    "loss_anchor_object_pull_graph_weight_sum": _loss_component_or_zero(losses, "anchor_object_pull_graph_weight_sum"),
                    "loss_anchor_object_pull_posterior_weight_sum": _loss_component_or_zero(
                        losses, "anchor_object_pull_posterior_weight_sum"
                    ),
                    "loss_anchor_object_pull_target_mass_mean": _loss_component_or_zero(
                        losses, "anchor_object_pull_target_mass_mean"
                    ),
                    "loss_anchor_object_pull_target_quality_mean": _loss_component_or_zero(
                        losses, "anchor_object_pull_target_quality_mean"
                    ),
                    "loss_pv_weak": losses.pv_weak,
                    "loss_pt": losses.pt,
                    "loss_vl_router": losses.vl_router,
                    "loss_vl_heatmap_task": losses.vl_heatmap_task,
                    "loss_vl_heatmap_effector": losses.vl_heatmap_effector,
                    "loss_vl_heatmap_interaction": losses.vl_heatmap_interaction,
                    "loss_vl_point_consistency": losses.vl_point_consistency,
                    "loss_vl_anchor_diversity": losses.vl_anchor_diversity,
                    "loss_mapg_graph": losses.mapg_graph,
                    "loss_mapg_siglip": losses.mapg_siglip,
                    "loss_mapg_vicreg": losses.mapg_vicreg,
                    "loss_mapg_cycle": losses.mapg_cycle,
                    "loss_mapg_masked_modality": losses.mapg_masked_modality,
                    "loss_mapg_routing": losses.mapg_routing,
                    "loss_mapg_support_diversity": losses.mapg_support_diversity,
                    "loss_mapg_geometry_diversity": losses.mapg_geometry_diversity,
                    "loss_slot_jepa": _loss_component_or_zero(losses, "slot_jepa"),
                    "loss_slot_jepa_direction": _loss_component_or_zero(losses, "slot_jepa_direction"),
                    "loss_slot_jepa_log_norm": _loss_component_or_zero(losses, "slot_jepa_log_norm"),
                    "loss_slot_jepa_pred_norm": _loss_component_or_zero(losses, "slot_jepa_pred_norm"),
                    "loss_slot_jepa_target_norm": _loss_component_or_zero(losses, "slot_jepa_target_norm"),
                    "loss_slot_jepa_matched_target_norm": _loss_component_or_zero(losses, "slot_jepa_matched_target_norm"),
                    "loss_support_pred": _loss_component_or_zero(losses, "support_pred"),
                    "loss_binding_consistency": _loss_component_or_zero(losses, "binding_consistency"),
                    "loss_aqr_denoising": _loss_component_or_zero(losses, "aqr_denoising"),
                    "loss_slot_quality": _loss_component_or_zero(losses, "slot_quality"),
                    "loss_vcap": _loss_component_or_zero(losses, "vcap"),
                    "loss_object_explanation": _loss_component_or_zero(losses, "object_explanation"),
                    "loss_object_explanation_feature": _loss_component_or_zero(losses, "object_explanation_feature"),
                    "loss_object_explanation_point": _loss_component_or_zero(losses, "object_explanation_point"),
                    "loss_object_explanation_contact": _loss_component_or_zero(losses, "object_explanation_contact"),
                    "loss_object_explanation_duplicate": _loss_component_or_zero(losses, "object_explanation_duplicate"),
                    "loss_object_explanation_background": _loss_component_or_zero(losses, "object_explanation_background"),
                    "physical_aux_budget_scale": losses.physical_aux_budget_scale,
                    "semantic_aux_budget_scale": losses.semantic_aux_budget_scale,
                    "alignment_budget_scale": losses.alignment_budget_scale,
                    "projective_candidate_density": pending.candidate_density,
                    "tactile_contact_prob_mean": pending.tactile_contact_prob_mean,
                    "tactile_active_rate": pending.tactile_active_rate,
                }
                _accumulate_owm_debug_metrics(metrics, pending.owm_debug_metrics)
            else:
                metrics["loss_action"] = metrics["loss_action"] + losses.action
                metrics["loss_action_default_equiv"] = metrics["loss_action_default_equiv"] + losses.action_default_equiv
                metrics["loss_action_weight_scale"] = metrics["loss_action_weight_scale"] + losses.action_weight_scale
                metrics["loss_action_active7"] = metrics["loss_action_active7"] + losses.action_active7
                metrics["loss_action_pos"] = metrics["loss_action_pos"] + losses.action_pos
                metrics["loss_action_rot"] = metrics["loss_action_rot"] + losses.action_rot
                metrics["loss_action_gripper"] = metrics["loss_action_gripper"] + losses.action_gripper
                metrics["loss_visual_latent"] = metrics["loss_visual_latent"] + losses.visual_latent
                metrics["loss_visual_real"] = metrics["loss_visual_real"] + losses.visual_real
                metrics["loss_tactile_real"] = metrics["loss_tactile_real"] + losses.tactile_real
                metrics["loss_tactile_map"] = metrics["loss_tactile_map"] + losses.tactile_map
                metrics["loss_tactile_aux"] = metrics["loss_tactile_aux"] + losses.tactile_aux
                metrics["loss_point_real"] = metrics["loss_point_real"] + losses.point_real
                metrics["loss_semantic_future_aux"] = metrics["loss_semantic_future_aux"] + losses.semantic_future_aux
                metrics["loss_semantic_group_raw"] = metrics["loss_semantic_group_raw"] + losses.semantic_group_raw
                metrics["loss_semantic_group_capped"] = metrics["loss_semantic_group_capped"] + losses.semantic_group_capped
                metrics["loss_physical_aux"] = metrics["loss_physical_aux"] + losses.physical_aux
                metrics["loss_physical_aux_capped"] = metrics["loss_physical_aux_capped"] + losses.physical_aux_capped
                metrics["loss_alignment"] = metrics["loss_alignment"] + losses.alignment
                metrics["loss_alignment_raw"] = metrics["loss_alignment_raw"] + losses.alignment_raw
                metrics["loss_total_minus_action"] = metrics["loss_total_minus_action"] + losses.total_minus_action
                metrics["loss_action_prefix_trust"] = metrics["loss_action_prefix_trust"] + _loss_component_or_zero(
                    losses, "action_prefix_trust"
                )
                metrics["loss_anchor_pv"] = metrics["loss_anchor_pv"] + losses.anchor_pv
                metrics["loss_anchor_object_pull"] = metrics["loss_anchor_object_pull"] + losses.anchor_object_pull
                metrics["loss_anchor_object_pull_graph"] = metrics["loss_anchor_object_pull_graph"] + _loss_component_or_zero(
                    losses, "anchor_object_pull_graph"
                )
                metrics["loss_anchor_object_pull_posterior"] = (
                    metrics["loss_anchor_object_pull_posterior"] + _loss_component_or_zero(losses, "anchor_object_pull_posterior")
                )
                metrics["loss_anchor_object_pull_graph_weight_sum"] = (
                    metrics["loss_anchor_object_pull_graph_weight_sum"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_graph_weight_sum")
                )
                metrics["loss_anchor_object_pull_posterior_weight_sum"] = (
                    metrics["loss_anchor_object_pull_posterior_weight_sum"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_posterior_weight_sum")
                )
                metrics["loss_anchor_object_pull_target_mass_mean"] = (
                    metrics["loss_anchor_object_pull_target_mass_mean"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_target_mass_mean")
                )
                metrics["loss_anchor_object_pull_target_quality_mean"] = (
                    metrics["loss_anchor_object_pull_target_quality_mean"]
                    + _loss_component_or_zero(losses, "anchor_object_pull_target_quality_mean")
                )
                metrics["loss_pv_weak"] = metrics["loss_pv_weak"] + losses.pv_weak
                metrics["loss_pt"] = metrics["loss_pt"] + losses.pt
                metrics["loss_vl_router"] = metrics["loss_vl_router"] + losses.vl_router
                metrics["loss_vl_heatmap_task"] = metrics["loss_vl_heatmap_task"] + losses.vl_heatmap_task
                metrics["loss_vl_heatmap_effector"] = metrics["loss_vl_heatmap_effector"] + losses.vl_heatmap_effector
                metrics["loss_vl_heatmap_interaction"] = metrics["loss_vl_heatmap_interaction"] + losses.vl_heatmap_interaction
                metrics["loss_vl_point_consistency"] = metrics["loss_vl_point_consistency"] + losses.vl_point_consistency
                metrics["loss_vl_anchor_diversity"] = metrics["loss_vl_anchor_diversity"] + losses.vl_anchor_diversity
                metrics["loss_mapg_graph"] = metrics["loss_mapg_graph"] + losses.mapg_graph
                metrics["loss_mapg_siglip"] = metrics["loss_mapg_siglip"] + losses.mapg_siglip
                metrics["loss_mapg_vicreg"] = metrics["loss_mapg_vicreg"] + losses.mapg_vicreg
                metrics["loss_mapg_cycle"] = metrics["loss_mapg_cycle"] + losses.mapg_cycle
                metrics["loss_mapg_masked_modality"] = metrics["loss_mapg_masked_modality"] + losses.mapg_masked_modality
                metrics["loss_mapg_routing"] = metrics["loss_mapg_routing"] + losses.mapg_routing
                metrics["loss_mapg_support_diversity"] = metrics["loss_mapg_support_diversity"] + losses.mapg_support_diversity
                metrics["loss_mapg_geometry_diversity"] = metrics["loss_mapg_geometry_diversity"] + losses.mapg_geometry_diversity
                metrics["loss_slot_jepa"] = metrics["loss_slot_jepa"] + _loss_component_or_zero(losses, "slot_jepa")
                metrics["loss_slot_jepa_direction"] = metrics["loss_slot_jepa_direction"] + _loss_component_or_zero(losses, "slot_jepa_direction")
                metrics["loss_slot_jepa_log_norm"] = metrics["loss_slot_jepa_log_norm"] + _loss_component_or_zero(losses, "slot_jepa_log_norm")
                metrics["loss_slot_jepa_pred_norm"] = metrics["loss_slot_jepa_pred_norm"] + _loss_component_or_zero(losses, "slot_jepa_pred_norm")
                metrics["loss_slot_jepa_target_norm"] = metrics["loss_slot_jepa_target_norm"] + _loss_component_or_zero(losses, "slot_jepa_target_norm")
                metrics["loss_slot_jepa_matched_target_norm"] = (
                    metrics["loss_slot_jepa_matched_target_norm"]
                    + _loss_component_or_zero(losses, "slot_jepa_matched_target_norm")
                )
                metrics["loss_support_pred"] = metrics["loss_support_pred"] + _loss_component_or_zero(losses, "support_pred")
                metrics["loss_binding_consistency"] = metrics["loss_binding_consistency"] + _loss_component_or_zero(
                    losses, "binding_consistency"
                )
                metrics["loss_aqr_denoising"] = metrics["loss_aqr_denoising"] + _loss_component_or_zero(
                    losses, "aqr_denoising"
                )
                metrics["loss_slot_quality"] = metrics["loss_slot_quality"] + _loss_component_or_zero(
                    losses, "slot_quality"
                )
                metrics["loss_vcap"] = metrics["loss_vcap"] + _loss_component_or_zero(losses, "vcap")
                metrics["loss_object_explanation"] = metrics["loss_object_explanation"] + _loss_component_or_zero(
                    losses, "object_explanation"
                )
                metrics["loss_object_explanation_feature"] = metrics["loss_object_explanation_feature"] + _loss_component_or_zero(
                    losses, "object_explanation_feature"
                )
                metrics["loss_object_explanation_point"] = metrics["loss_object_explanation_point"] + _loss_component_or_zero(
                    losses, "object_explanation_point"
                )
                metrics["loss_object_explanation_contact"] = metrics["loss_object_explanation_contact"] + _loss_component_or_zero(
                    losses, "object_explanation_contact"
                )
                metrics["loss_object_explanation_duplicate"] = metrics["loss_object_explanation_duplicate"] + _loss_component_or_zero(
                    losses, "object_explanation_duplicate"
                )
                metrics["loss_object_explanation_background"] = metrics["loss_object_explanation_background"] + _loss_component_or_zero(
                    losses, "object_explanation_background"
                )
                metrics["physical_aux_budget_scale"] = metrics["physical_aux_budget_scale"] + losses.physical_aux_budget_scale
                metrics["semantic_aux_budget_scale"] = metrics["semantic_aux_budget_scale"] + losses.semantic_aux_budget_scale
                metrics["alignment_budget_scale"] = metrics["alignment_budget_scale"] + losses.alignment_budget_scale
                metrics["projective_candidate_density"] = metrics["projective_candidate_density"] + pending.candidate_density
                metrics["tactile_contact_prob_mean"] = metrics["tactile_contact_prob_mean"] + pending.tactile_contact_prob_mean
                metrics["tactile_active_rate"] = metrics["tactile_active_rate"] + pending.tactile_active_rate
                _accumulate_owm_debug_metrics(metrics, pending.owm_debug_metrics)

        assert metrics is not None
        denom = float(len(totals))
        mean_total = torch.stack(totals).mean()
        result: dict[str, Any] = {
            "loss_total": mean_total,
            "loss_action": metrics["loss_action"] / denom,
            "loss_action_default_equiv": metrics["loss_action_default_equiv"] / denom,
            "loss_action_weight_scale": metrics["loss_action_weight_scale"] / denom,
            "loss_action_active7": metrics["loss_action_active7"] / denom,
            "loss_action_pos": metrics["loss_action_pos"] / denom,
            "loss_action_rot": metrics["loss_action_rot"] / denom,
            "loss_action_gripper": metrics["loss_action_gripper"] / denom,
            "loss_visual_latent": metrics["loss_visual_latent"] / denom,
            "loss_visual_real": metrics["loss_visual_real"] / denom,
            "loss_tactile_real": metrics["loss_tactile_real"] / denom,
            "loss_tactile_map": metrics["loss_tactile_map"] / denom,
            "loss_tactile_aux": metrics["loss_tactile_aux"] / denom,
            "loss_point_real": metrics["loss_point_real"] / denom,
            "loss_semantic_future_aux": metrics["loss_semantic_future_aux"] / denom,
            "loss_semantic_group_raw": metrics["loss_semantic_group_raw"] / denom,
            "loss_semantic_group_capped": metrics["loss_semantic_group_capped"] / denom,
            "loss_physical_aux": metrics["loss_physical_aux"] / denom,
            "loss_physical_aux_capped": metrics["loss_physical_aux_capped"] / denom,
            "loss_alignment": metrics["loss_alignment"] / denom,
            "loss_alignment_raw": metrics["loss_alignment_raw"] / denom,
            "loss_total_minus_action": metrics["loss_total_minus_action"] / denom,
            "loss_action_prefix_trust": metrics["loss_action_prefix_trust"] / denom,
            "loss_anchor_pv": metrics["loss_anchor_pv"] / denom,
            "loss_anchor_object_pull": metrics["loss_anchor_object_pull"] / denom,
            "loss_anchor_object_pull_graph": metrics["loss_anchor_object_pull_graph"] / denom,
            "loss_anchor_object_pull_posterior": metrics["loss_anchor_object_pull_posterior"] / denom,
            "loss_anchor_object_pull_graph_weight_sum": metrics["loss_anchor_object_pull_graph_weight_sum"] / denom,
            "loss_anchor_object_pull_posterior_weight_sum": metrics["loss_anchor_object_pull_posterior_weight_sum"] / denom,
            "loss_anchor_object_pull_target_mass_mean": metrics["loss_anchor_object_pull_target_mass_mean"] / denom,
            "loss_anchor_object_pull_target_quality_mean": metrics["loss_anchor_object_pull_target_quality_mean"] / denom,
            "loss_pv_weak": metrics["loss_pv_weak"] / denom,
            "loss_pt": metrics["loss_pt"] / denom,
            "loss_vl_router": metrics["loss_vl_router"] / denom,
            "loss_vl_heatmap_task": metrics["loss_vl_heatmap_task"] / denom,
            "loss_vl_heatmap_effector": metrics["loss_vl_heatmap_effector"] / denom,
            "loss_vl_heatmap_interaction": metrics["loss_vl_heatmap_interaction"] / denom,
            "loss_vl_point_consistency": metrics["loss_vl_point_consistency"] / denom,
            "loss_vl_anchor_diversity": metrics["loss_vl_anchor_diversity"] / denom,
            "loss_mapg_graph": metrics["loss_mapg_graph"] / denom,
            "loss_mapg_siglip": metrics["loss_mapg_siglip"] / denom,
            "loss_mapg_vicreg": metrics["loss_mapg_vicreg"] / denom,
            "loss_mapg_cycle": metrics["loss_mapg_cycle"] / denom,
            "loss_mapg_masked_modality": metrics["loss_mapg_masked_modality"] / denom,
            "loss_mapg_routing": metrics["loss_mapg_routing"] / denom,
            "loss_mapg_support_diversity": metrics["loss_mapg_support_diversity"] / denom,
            "loss_mapg_geometry_diversity": metrics["loss_mapg_geometry_diversity"] / denom,
            "loss_slot_jepa": metrics["loss_slot_jepa"] / denom,
            "loss_slot_jepa_direction": metrics["loss_slot_jepa_direction"] / denom,
            "loss_slot_jepa_log_norm": metrics["loss_slot_jepa_log_norm"] / denom,
            "loss_slot_jepa_pred_norm": metrics["loss_slot_jepa_pred_norm"] / denom,
            "loss_slot_jepa_target_norm": metrics["loss_slot_jepa_target_norm"] / denom,
            "loss_slot_jepa_matched_target_norm": metrics["loss_slot_jepa_matched_target_norm"] / denom,
            "loss_support_pred": metrics["loss_support_pred"] / denom,
            "loss_binding_consistency": metrics["loss_binding_consistency"] / denom,
            "loss_aqr_denoising": metrics["loss_aqr_denoising"] / denom,
            "loss_slot_quality": metrics["loss_slot_quality"] / denom,
            "loss_vcap": metrics["loss_vcap"] / denom,
            "loss_object_explanation": metrics["loss_object_explanation"] / denom,
            "loss_object_explanation_feature": metrics["loss_object_explanation_feature"] / denom,
            "loss_object_explanation_point": metrics["loss_object_explanation_point"] / denom,
            "loss_object_explanation_contact": metrics["loss_object_explanation_contact"] / denom,
            "loss_object_explanation_duplicate": metrics["loss_object_explanation_duplicate"] / denom,
            "loss_object_explanation_background": metrics["loss_object_explanation_background"] / denom,
            "physical_aux_budget_scale": metrics["physical_aux_budget_scale"] / denom,
            "semantic_aux_budget_scale": metrics["semantic_aux_budget_scale"] / denom,
            "alignment_budget_scale": metrics["alignment_budget_scale"] / denom,
            "projective_candidate_density": metrics["projective_candidate_density"] / denom,
            "tactile_contact_prob_mean": metrics["tactile_contact_prob_mean"] / denom,
            "tactile_active_rate": metrics["tactile_active_rate"] / denom,
        }
        for key in OWM_DEBUG_METRIC_KEYS:
            result[key] = metrics.get(key, result["loss_total"] * 0.0) / denom
        if capture_visual_diagnostics:
            result["diagnostic_physical_visual_real_seq"] = physical_visual_real_seq
            result["diagnostic_semantic_visual_real_seq"] = semantic_visual_real_seq
        if capture_anchor_overlay:
            result["diagnostic_anchor_overlay"] = anchor_overlay_snapshot
        return result


_WINDOW_OUTPUT_TENSOR_KEYS: tuple[str, ...] = (
    "loss_total",
    "loss_action",
    "loss_action_default_equiv",
    "loss_action_weight_scale",
    "loss_action_active7",
    "loss_action_pos",
    "loss_action_rot",
    "loss_action_gripper",
    "loss_visual_latent",
    "loss_visual_real",
    "loss_tactile_real",
    "loss_tactile_map",
    "loss_tactile_aux",
    "loss_point_real",
    "loss_semantic_future_aux",
    "loss_semantic_group_raw",
    "loss_semantic_group_capped",
    "loss_physical_aux",
    "loss_physical_aux_capped",
    "loss_alignment",
    "loss_alignment_raw",
    "loss_total_minus_action",
    "loss_action_prefix_trust",
    "loss_anchor_pv",
    "loss_anchor_object_pull",
    "loss_anchor_object_pull_graph",
    "loss_anchor_object_pull_posterior",
    "loss_anchor_object_pull_graph_weight_sum",
    "loss_anchor_object_pull_posterior_weight_sum",
    "loss_anchor_object_pull_target_mass_mean",
    "loss_anchor_object_pull_target_quality_mean",
    "loss_pv_weak",
    "loss_pt",
    "loss_vl_router",
    "loss_vl_heatmap_task",
    "loss_vl_heatmap_effector",
    "loss_vl_heatmap_interaction",
    "loss_vl_point_consistency",
    "loss_vl_anchor_diversity",
    "loss_mapg_graph",
    "loss_mapg_siglip",
    "loss_mapg_vicreg",
    "loss_mapg_cycle",
    "loss_mapg_masked_modality",
    "loss_mapg_routing",
    "loss_mapg_support_diversity",
    "loss_mapg_geometry_diversity",
    "loss_slot_jepa",
    "loss_slot_jepa_direction",
    "loss_slot_jepa_log_norm",
    "loss_slot_jepa_pred_norm",
    "loss_slot_jepa_target_norm",
    "loss_slot_jepa_matched_target_norm",
    "loss_support_pred",
    "loss_binding_consistency",
    "loss_aqr_denoising",
    "loss_slot_quality",
    "loss_vcap",
    "loss_object_explanation",
    "loss_object_explanation_feature",
    "loss_object_explanation_point",
    "loss_object_explanation_contact",
    "loss_object_explanation_duplicate",
    "loss_object_explanation_background",
    "physical_aux_budget_scale",
    "semantic_aux_budget_scale",
    "alignment_budget_scale",
    "projective_candidate_density",
    "tactile_contact_prob_mean",
    "tactile_active_rate",
    *OWM_DEBUG_METRIC_KEYS,
)


def _window_outputs_to_tensor_tuple(outputs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
    return tuple(outputs[key] for key in _WINDOW_OUTPUT_TENSOR_KEYS)


def _window_outputs_from_tensor_tuple(outputs: Sequence[torch.Tensor]) -> dict[str, torch.Tensor]:
    if len(outputs) != len(_WINDOW_OUTPUT_TENSOR_KEYS):
        raise RuntimeError(
            "Checkpointed window-output tuple does not match the canonical loss/metric contract: "
            f"expected {len(_WINDOW_OUTPUT_TENSOR_KEYS)} tensors, got {len(outputs)}."
        )
    return {
        key: tensor
        for key, tensor in zip(_WINDOW_OUTPUT_TENSOR_KEYS, outputs, strict=True)
    }


def _checkpoint_dummy_input(model: torch.nn.Module) -> torch.Tensor:
    for param in model.parameters():
        if bool(getattr(param, "requires_grad", False)):
            dtype = param.dtype if torch.is_floating_point(param) else torch.float32
            return torch.zeros((), device=param.device, dtype=dtype, requires_grad=True)
    raise RuntimeError("Window activation checkpointing requires at least one trainable parameter.")


def _is_retryable_first_step_error(exc: RuntimeError) -> bool:
    message = str(exc)
    return any(pattern in message for pattern in _RETRYABLE_FIRST_STEP_ERRORS)


def _focus_centers_world_from_observation(observation: PicfObservation) -> np.ndarray:
    if observation.G_t is None:
        raise ValueError("PICF focus center construction requires observation.G_t to be set.")
    centers = [np.asarray(observation.G_t[:3, 3], dtype=np.float32)]
    packet = observation.tactile
    if packet is not None:
        for sensor in packet.sensors:
            if not sensor.valid:
                continue
            sensor_pose_world = np.asarray(observation.G_t, dtype=np.float32) @ np.asarray(sensor.T_sens_to_wrist, dtype=np.float32)
            centers.append(np.asarray(sensor_pose_world[:3, 3], dtype=np.float32))
    return np.stack(centers, axis=0)


def _pointcloud_payload_from_observation(
    core: PicfFullCore,
    observation: PicfObservation,
) -> dict[str, np.ndarray | float]:
    return {
        "rgb_static": observation.rgb_static,
        "depth_static": observation.depth_static,
        "rgb_gripper": observation.rgb_gripper,
        "depth_gripper": observation.depth_gripper,
        "robot_obs": observation.robot_obs,
        "focus_centers_world": _focus_centers_world_from_observation(observation),
        "focus_radius_m": core.config.crop_radius_m,
    }


def _ensure_window_has_valid_first_step_xyzrgb_support(
    trainer: _PicfWindowTrainer,
    window: _TransitionWindow,
) -> tuple[int, ...]:
    """Validate the data-only first-step contract before entering DDP forward.

    Catching retryable first-step errors *inside* a DDP-wrapped forward can leave
    the reducer in an unfinished state on the rank that aborted early. The
    first-step legality check depends only on the current observation and the
    calibrated point-cloud crop, so it is safe to preflight it before
    `model(window)` enters the distributed graph.
    """
    core = trainer.core
    first = window.frames[0]
    if first.G_t is None:
        first.G_t = core.local_frame.make_transform(first.robot_obs)
    if first.point_set is None:
        first.point_set = core.pointcloud_builder(_pointcloud_payload_from_observation(core, first))
    meta = core._build_runtime_meta(first, None)
    if not meta.point_contract_ok:
        raise RuntimeError(_RETRYABLE_FIRST_STEP_ERRORS[0])
    point_counts: list[int] = []
    for offset, frame in enumerate(window.frames):
        if frame.G_t is None:
            frame.G_t = core.local_frame.make_transform(frame.robot_obs)
        if frame.point_set is None:
            frame.point_set = core.pointcloud_builder(_pointcloud_payload_from_observation(core, frame))
        frame_context = core._point_subset(frame)
        point_count = int(frame_context.points_local.shape[0])
        point_counts.append(point_count)
        if point_count == 0:
            if offset == 0:
                raise RuntimeError(_RETRYABLE_FIRST_STEP_ERRORS[1])
            raise RuntimeError(f"{_RETRYABLE_FIRST_STEP_ERRORS[2]} {offset}.")
    return tuple(point_counts)


def _lr_for_step(step: int, *, base_lr: float, warmup_steps: int, min_lr: float, total_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * float(step + 1) / float(warmup_steps)
    if total_steps <= warmup_steps:
        return base_lr
    progress = min(max((step - warmup_steps) / float(total_steps - warmup_steps), 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (base_lr - min_lr) * cosine


def _set_optimizer_lr(
    optimizer: torch.optim.Optimizer,
    lr: float,
    *,
    group_runtime_multipliers: dict[str, float] | None = None,
) -> None:
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        name = str(group.get("name", ""))
        runtime = 1.0 if group_runtime_multipliers is None else float(group_runtime_multipliers.get(name, 1.0))
        group["lr"] = lr * scale * runtime


def _picf_core_runtime_lr_multiplier(args: argparse.Namespace, *, step: int) -> float:
    mode = str(getattr(args, "picf_core_lr_runtime_mode", "constant")).lower().replace("-", "_")
    if mode in {"", "constant", "none", "off"}:
        return 1.0
    if mode != "block_alternating":
        raise ValueError(f"Unsupported picf_core_lr_runtime_mode={mode!r}.")
    start = int(getattr(args, "picf_core_lr_block_start_step", 0))
    if int(step) < start:
        return 0.0
    cycle = max(int(getattr(args, "picf_core_lr_block_cycle_steps", 0)), 1)
    active = max(int(getattr(args, "picf_core_lr_block_active_steps", 0)), 0)
    if active <= 0:
        return 0.0
    phase = (int(step) - start) % cycle
    return 1.0 if phase < min(active, cycle) else 0.0


class _OptimizerCollection:
    """Small optimizer facade used when Zero-1 needs dtype-partitioned optimizers."""

    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("_OptimizerCollection requires at least one optimizer.")
        self.optimizers = list(optimizers)
        self._refresh_param_groups()

    def _refresh_param_groups(self) -> None:
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]

    def zero_grad(self, *args: Any, **kwargs: Any) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(*args, **kwargs)

    def step(self, *args: Any, **kwargs: Any) -> list[Any]:
        return [optimizer.step(*args, **kwargs) for optimizer in self.optimizers]

    def consolidate_state_dict(self, *, to: int = 0) -> None:
        for optimizer in self.optimizers:
            consolidate = getattr(optimizer, "consolidate_state_dict", None)
            if callable(consolidate):
                consolidate(to=to)

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": "picf_optimizer_collection_v1",
            "optimizers": [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if state_dict.get("format") != "picf_optimizer_collection_v1":
            raise ValueError("Optimizer checkpoint is not a PICF optimizer collection.")
        optimizer_states = state_dict.get("optimizers")
        if not isinstance(optimizer_states, list) or len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                "Optimizer collection checkpoint does not match the current dtype-partitioned optimizer layout."
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states, strict=True):
            optimizer.load_state_dict(optimizer_state)
        self._refresh_param_groups()


def _checkpoint_dir_for_step(output_dir: Path, step: int) -> Path:
    return output_dir / f"{int(step)}"


def _prune_old_checkpoints(output_dir: Path, *, keep_last: int) -> list[Path]:
    """Remove old numeric step checkpoint directories after a successful save."""

    keep_last = int(keep_last)
    if keep_last <= 0 or not output_dir.exists():
        return []
    step_dirs: list[tuple[int, Path]] = []
    for path in output_dir.iterdir():
        if not path.is_dir() or path.name.startswith("tmp_"):
            continue
        try:
            step = int(path.name)
        except ValueError:
            continue
        step_dirs.append((step, path))
    step_dirs.sort(key=lambda item: item[0])
    removed: list[Path] = []
    for _, path in step_dirs[:-keep_last]:
        shutil.rmtree(path)
        removed.append(path)
    return removed


def _latest_checkpoint_step(output_dir: Path) -> int | None:
    if not output_dir.exists():
        return None
    steps = [
        int(path.name)
        for path in output_dir.iterdir()
        if path.is_dir() and path.name.isdigit() and not path.name.startswith("tmp_")
    ]
    return max(steps) if steps else None


def _resolve_resume_path(*, output_dir: Path, latest_path: Path) -> Path | None:
    latest_step = _latest_checkpoint_step(output_dir)
    if latest_step is not None:
        return _checkpoint_dir_for_step(output_dir, latest_step)
    if not latest_path.exists():
        return None
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    checkpoint_dir = payload.get("checkpoint_dir")
    if checkpoint_dir is not None:
        candidate = Path(checkpoint_dir)
        if candidate.exists():
            return candidate
    return latest_path


def _should_save_optimizer_state(*, args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "optimizer_checkpoint_mode", "auto")).lower().replace("-", "_")
    if mode == "full":
        return True
    if mode == "model_only":
        return False
    if mode != "auto":
        raise ValueError(f"Unsupported optimizer checkpoint mode: {mode!r}.")
    if not _picf_mode_enabled(args):
        # PI0.5-only ablation checkpoints are primarily used for parity/eval
        # runs. Saving full optimizer state for the semantic stack is both
        # unnecessary in the default case and extremely expensive.
        return False
    return str(getattr(args, "optimizer_sharding", "none")).lower() != "zero1"


def _fsdp_full_state_dict_context(
    model: torch.nn.Module,
    *,
    rank0_only: bool,
) -> contextlib.AbstractContextManager[None]:
    assert FullyShardedDataParallel is not None
    assert FullStateDictConfig is not None
    assert FullOptimStateDictConfig is not None
    assert StateDictType is not None
    return FullyShardedDataParallel.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=bool(rank0_only)),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=bool(rank0_only)),
    )


_ABLATED_SEMANTIC_ONLY_CHECKPOINT_FORMAT = "picf_ablated_semantic_only_v1"


def _reset_child_fsdp_root_markers(module: torch.nn.Module) -> None:
    """Undo child-only FSDP lazy-root marking after nested state-dict operations.

    The ablated semantic-only checkpoint path intentionally talks only to the
    nested ``semantic_encoder`` subtree so that a lazy PICF core can remain
    uninitialized. However, entering ``FULL_STATE_DICT`` context on that child
    FSDP subtree may lazily mark the child wrapper as a temporary local root.
    If we leave that marker behind, the outer trainer-level FSDP root will hit
    PyTorch's ``Non-root FSDP instance's `_is_root` ...`` assertion on the next
    forward after reload.

    Resetting only the temporarily promoted child roots restores the pre-forward
    lazy-init state without weakening the checkpoint contract or changing the
    saved tensor values.
    """
    if FullyShardedDataParallel is None:
        return
    for submodule in module.modules():
        if isinstance(submodule, FullyShardedDataParallel) and getattr(submodule, "_is_root", None) is True:
            submodule._is_root = None


def _ablated_semantic_only_checkpoint_enabled(
    *,
    args: argparse.Namespace | None,
    module: torch.nn.Module,
) -> bool:
    if args is not None and _picf_mode_enabled(args):
        return False
    semantic_encoder = getattr(module, "semantic_encoder", None)
    return isinstance(semantic_encoder, torch.nn.Module)


def _build_ablated_semantic_only_model_state(
    *,
    module: torch.nn.Module,
) -> dict[str, Any]:
    semantic_encoder = getattr(module, "semantic_encoder", None)
    if not isinstance(semantic_encoder, torch.nn.Module):
        raise RuntimeError(
            "picf_mode=ablated checkpointing requires a semantic_encoder module."
        )
    if _is_fsdp_model(semantic_encoder):
        with _fsdp_full_state_dict_context(semantic_encoder, rank0_only=True):
            semantic_state = semantic_encoder.state_dict()
        _reset_child_fsdp_root_markers(semantic_encoder)
    else:
        semantic_state = semantic_encoder.state_dict()
    return {
        "checkpoint_model_format": _ABLATED_SEMANTIC_ONLY_CHECKPOINT_FORMAT,
        "semantic_encoder": semantic_state,
    }


def _is_ablated_semantic_only_model_state(state: Any) -> bool:
    return bool(
        isinstance(state, dict)
        and state.get("checkpoint_model_format") == _ABLATED_SEMANTIC_ONLY_CHECKPOINT_FORMAT
        and "semantic_encoder" in state
    )


def _load_ablated_semantic_only_model_state(
    *,
    module: torch.nn.Module,
    model_state: dict[str, Any],
) -> None:
    semantic_encoder = getattr(module, "semantic_encoder", None)
    if not isinstance(semantic_encoder, torch.nn.Module):
        raise RuntimeError(
            "Ablated semantic-only checkpoint load requires the target model to expose semantic_encoder."
        )
    if _is_fsdp_model(semantic_encoder):
        with _fsdp_full_state_dict_context(semantic_encoder, rank0_only=False):
            semantic_encoder.load_state_dict(model_state["semantic_encoder"], strict=True)
        _reset_child_fsdp_root_markers(semantic_encoder)
    else:
        semantic_encoder.load_state_dict(model_state["semantic_encoder"], strict=True)


def _ablated_semantic_only_optimizer_state(
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    rank0_only: bool,
) -> dict[str, Any]:
    semantic_encoder = getattr(module, "semantic_encoder", None)
    if not isinstance(semantic_encoder, torch.nn.Module):
        raise RuntimeError(
            "Ablated semantic-only optimizer checkpointing requires the target model to expose semantic_encoder."
        )
    if _is_fsdp_model(semantic_encoder):
        with _fsdp_full_state_dict_context(semantic_encoder, rank0_only=rank0_only):
            optimizer_state = FullyShardedDataParallel.optim_state_dict(semantic_encoder, optimizer)
        _reset_child_fsdp_root_markers(semantic_encoder)
        return optimizer_state
    return optimizer.state_dict()


def _load_ablated_semantic_only_optimizer_state(
    *,
    module: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    optimizer_state: dict[str, Any],
) -> None:
    semantic_encoder = getattr(module, "semantic_encoder", None)
    if not isinstance(semantic_encoder, torch.nn.Module):
        raise RuntimeError(
            "Ablated semantic-only optimizer checkpoint load requires the target model to expose semantic_encoder."
        )
    if _is_fsdp_model(semantic_encoder):
        with _fsdp_full_state_dict_context(semantic_encoder, rank0_only=False):
            optimizer.load_state_dict(
                FullyShardedDataParallel.optim_state_dict_to_load(semantic_encoder, optimizer, optimizer_state)
            )
        _reset_child_fsdp_root_markers(semantic_encoder)
    else:
        optimizer.load_state_dict(optimizer_state)


def _save_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    rank: int = 0,
    device: torch.device | None = None,
    grad_clip_controller: _GradClipController | None = None,
    save_optimizer_state: bool = True,
) -> None:
    module = _unwrap_training_model(model)
    final_dir = _checkpoint_dir_for_step(output_dir, step)
    tmp_dir = output_dir / f"tmp_{int(step)}"
    latest_path = output_dir / "latest.pt"

    is_main = _is_main(int(rank))

    if device is None:
        device = getattr(module, "device", torch.device("cpu"))

    if _is_fsdp_model(model):
        if _ablated_semantic_only_checkpoint_enabled(args=args, module=module):
            model_state = _build_ablated_semantic_only_model_state(module=module)
            optimizer_state = (
                None
                if not save_optimizer_state
                else _ablated_semantic_only_optimizer_state(
                    module=module,
                    optimizer=optimizer,
                    rank0_only=True,
                )
            )
        else:
            with _fsdp_full_state_dict_context(model, rank0_only=True):
                model_state = model.state_dict()
                optimizer_state = (
                    None
                    if not save_optimizer_state
                    else FullyShardedDataParallel.optim_state_dict(model, optimizer)
                )
        _distributed_barrier(use_ddp=dist.is_initialized(), device=device)
        if not is_main:
            return
    else:
        if not is_main:
            return
        if _ablated_semantic_only_checkpoint_enabled(args=args, module=module):
            model_state = _build_ablated_semantic_only_model_state(module=module)
        else:
            model_state = module.state_dict()
        optimizer_state = optimizer.state_dict() if save_optimizer_state else None

    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model_state, tmp_dir / "model.pt")
    if save_optimizer_state and optimizer_state is not None:
        torch.save(optimizer_state, tmp_dir / "optimizer.pt")
    metadata = {
        "step": int(step),
        "args": vars(args),
        "timestamp": time.time(),
        "checkpoint_format": "picf_trainer_v2",
        "optimizer_state_saved": bool(save_optimizer_state),
        "optimizer_checkpoint_mode": str(getattr(args, "optimizer_checkpoint_mode", "auto")),
        "grad_clip_controller": None if grad_clip_controller is None else grad_clip_controller.state_dict(),
    }
    torch.save(metadata, tmp_dir / "metadata.pt")

    if final_dir.exists():
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    torch.save(
        {
            "step": int(step),
            "checkpoint_dir": str(final_dir),
        },
        latest_path,
    )


def _consolidate_optimizer_state_for_checkpoint(
    optimizer: torch.optim.Optimizer,
    *,
    rank: int,
) -> None:
    """Gather sharded optimizer state onto rank 0 before checkpointing.

    ZeroRedundancyOptimizer shards Adam moments across ranks.  Its
    `consolidate_state_dict(to=0)` call is collective, so every rank must enter
    it before rank 0 calls `state_dict()` in `_save_checkpoint`.
    """
    consolidate = getattr(optimizer, "consolidate_state_dict", None)
    if not callable(consolidate):
        return
    consolidate(to=0)
    if int(rank) == 0 and callable(getattr(optimizer, "state_dict", None)):
        # Fail here, before writing model.pt, if consolidation was incomplete.
        optimizer.state_dict()


def _matches_compat_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatch(str(name), pattern) for pattern in patterns)


def _filter_shape_mismatched_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    module_state = module.state_dict()
    filtered: dict[str, Any] = {}
    mismatched: dict[str, str] = {}
    for key, value in state_dict.items():
        target = module_state.get(key)
        if target is None:
            filtered[key] = value
            continue
        if hasattr(value, "shape") and hasattr(target, "shape"):
            value_shape = tuple(value.shape)
            target_shape = tuple(target.shape)
            if value_shape != target_shape:
                mismatched[key] = f"{key}: checkpoint_shape={value_shape} model_shape={target_shape}"
                continue
        filtered[key] = value
    return filtered, mismatched


def _load_state_dict_picf_compat(module: torch.nn.Module, state_dict: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    filtered_state_dict, shape_mismatches = _filter_shape_mismatched_state_dict(module, state_dict)
    info = module.load_state_dict(filtered_state_dict, strict=False)
    mismatch_keys = set(shape_mismatches)
    missing = [
        key
        for key in info.missing_keys
        if key not in mismatch_keys and not _matches_compat_pattern(key, _COMPAT_ALLOWED_MISSING_KEYS)
    ]
    unexpected = [
        key for key in info.unexpected_keys if not _matches_compat_pattern(key, _COMPAT_ALLOWED_UNEXPECTED_KEYS)
    ]
    if missing or unexpected:
        raise RuntimeError(
            "PICF compatibility checkpoint load failed. "
            f"missing={missing} unexpected={unexpected}"
        )
    return list(info.missing_keys), list(info.unexpected_keys), list(shape_mismatches.values())


def _load_checkpoint(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    optimizer_checkpoint_mode: str = "auto",
    grad_clip_controller: _GradClipController | None = None,
) -> int:
    module = _unwrap_training_model(model)
    optimizer_checkpoint_mode = str(optimizer_checkpoint_mode or "auto").lower().replace("-", "_")
    load_optimizer_state = optimizer_checkpoint_mode != "model_only"
    if path.is_dir():
        model_state = torch.load(path / "model.pt", map_location=device, weights_only=False)
        metadata = torch.load(path / "metadata.pt", map_location=device, weights_only=False)
        optimizer_path = path / "optimizer.pt"
        optimizer_state = (
            torch.load(optimizer_path, map_location=device, weights_only=False)
            if load_optimizer_state and optimizer_path.exists()
            else None
        )
        optimizer_loaded = optimizer_state is not None
        if not load_optimizer_state and optimizer_path.exists():
            logging.info(
                "optimizer_checkpoint_mode=%s; skipping optimizer state from checkpoint %s.",
                optimizer_checkpoint_mode,
                path,
            )
        if _is_fsdp_model(model):
            if _is_ablated_semantic_only_model_state(model_state):
                _load_ablated_semantic_only_model_state(module=module, model_state=model_state)
                if optimizer_loaded:
                    _load_ablated_semantic_only_optimizer_state(
                        module=module,
                        optimizer=optimizer,
                        optimizer_state=optimizer_state,
                    )
            else:
                with _fsdp_full_state_dict_context(model, rank0_only=False):
                    try:
                        model.load_state_dict(model_state, strict=True)
                    except RuntimeError:
                        try:
                            missing, unexpected, shape_mismatches = _load_state_dict_picf_compat(model, model_state)
                            logging.warning(
                                "Loaded FSDP PICF trainer checkpoint with compatibility migration. "
                                "missing_keys=%s unexpected_keys=%s shape_mismatch_keys=%s. "
                                "Optimizer state will be reinitialized.",
                                missing,
                                unexpected,
                                shape_mismatches,
                            )
                            optimizer_loaded = False
                        except RuntimeError as compat_exc:
                            raise RuntimeError(
                                "FSDP checkpoint load failed. Compatibility migration is restricted to "
                                "explicitly allowed PICF extension parameters; load a checkpoint written by "
                                "the same architecture or add a narrow compatibility rule."
                            ) from compat_exc
                    if optimizer_loaded:
                        optimizer.load_state_dict(
                            FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, optimizer_state)
                        )
            if grad_clip_controller is not None and not grad_clip_controller.load_state_dict(metadata.get("grad_clip_controller")):
                logging.info("Gradient clip controller state not restored from checkpoint; starting with fresh history.")
            return int(metadata.get("step", 0))
        if _is_ablated_semantic_only_model_state(model_state):
            _load_ablated_semantic_only_model_state(module=module, model_state=model_state)
        else:
            try:
                module.load_state_dict(model_state, strict=True)
            except RuntimeError:
                try:
                    missing, unexpected, shape_mismatches = _load_state_dict_picf_compat(module, model_state)
                    logging.warning(
                        "Loaded PICF trainer checkpoint with compatibility migration. "
                        "missing_keys=%s unexpected_keys=%s shape_mismatch_keys=%s. "
                        "Optimizer state will be reinitialized.",
                        missing,
                        unexpected,
                        shape_mismatches,
                    )
                    optimizer_loaded = False
                except RuntimeError:
                    try:
                        module.core.load_state_dict(model_state, strict=True)
                    except RuntimeError:
                        missing, unexpected, shape_mismatches = _load_state_dict_picf_compat(module.core, model_state)
                        logging.warning(
                            "Loaded PICF core-only checkpoint with compatibility migration. "
                            "missing_keys=%s unexpected_keys=%s shape_mismatch_keys=%s. "
                            "Optimizer state will be reinitialized.",
                            missing,
                            unexpected,
                            shape_mismatches,
                        )
                        optimizer_loaded = False
        if optimizer_loaded:
            try:
                optimizer.load_state_dict(optimizer_state)
            except ValueError:
                logging.warning(
                    "Optimizer state dict is incompatible with the current PICF architecture. Reinitializing optimizer."
                )
        else:
            logging.info("No optimizer state found in checkpoint %s; optimizer will be reinitialized.", path)
        if grad_clip_controller is not None and not grad_clip_controller.load_state_dict(metadata.get("grad_clip_controller")):
            logging.info("Gradient clip controller state not restored from checkpoint; starting with fresh history.")
        return int(metadata.get("step", 0))

    payload = torch.load(path, map_location=device, weights_only=False)
    checkpoint_dir = payload.get("checkpoint_dir")
    if checkpoint_dir is not None and "model" not in payload:
        return _load_checkpoint(
            path=Path(checkpoint_dir),
            model=model,
            optimizer=optimizer,
            device=device,
            optimizer_checkpoint_mode=optimizer_checkpoint_mode,
            grad_clip_controller=grad_clip_controller,
        )
    optimizer_loaded = bool(load_optimizer_state)
    if not load_optimizer_state and payload.get("optimizer") is not None:
        logging.info(
            "optimizer_checkpoint_mode=%s; skipping optimizer payload from checkpoint %s.",
            optimizer_checkpoint_mode,
            path,
        )
    if _is_fsdp_model(model):
        if _is_ablated_semantic_only_model_state(payload["model"]):
            _load_ablated_semantic_only_model_state(module=module, model_state=payload["model"])
            optimizer_payload = payload.get("optimizer")
            if optimizer_loaded and optimizer_payload is not None:
                _load_ablated_semantic_only_optimizer_state(
                    module=module,
                    optimizer=optimizer,
                    optimizer_state=optimizer_payload,
                )
            elif optimizer_loaded:
                logging.info("No optimizer payload found in checkpoint; optimizer will be reinitialized.")
        else:
            with _fsdp_full_state_dict_context(model, rank0_only=False):
                try:
                    model.load_state_dict(payload["model"], strict=True)
                except RuntimeError as exc:
                    raise RuntimeError(
                        "FSDP payload load failed. Compatibility migration is not supported for "
                        "training_strategy=fsdp_full_shard; load a checkpoint written by the same architecture."
                    ) from exc
                optimizer_payload = payload.get("optimizer")
                if optimizer_loaded and optimizer_payload is not None:
                    optimizer.load_state_dict(
                        FullyShardedDataParallel.optim_state_dict_to_load(model, optimizer, optimizer_payload)
                    )
                elif optimizer_loaded:
                    logging.info("No optimizer payload found in checkpoint; optimizer will be reinitialized.")
        if grad_clip_controller is not None and not grad_clip_controller.load_state_dict(payload.get("grad_clip_controller")):
            logging.info("Gradient clip controller state not restored from payload; starting with fresh history.")
        return int(payload.get("step", 0))
    if _is_ablated_semantic_only_model_state(payload["model"]):
        _load_ablated_semantic_only_model_state(module=module, model_state=payload["model"])
    else:
        try:
            module.load_state_dict(payload["model"], strict=True)
        except RuntimeError:
            try:
                missing, unexpected, shape_mismatches = _load_state_dict_picf_compat(module, payload["model"])
                logging.warning(
                    "Loaded PICF trainer payload with compatibility migration. "
                    "missing_keys=%s unexpected_keys=%s shape_mismatch_keys=%s. "
                    "Optimizer state will be reinitialized.",
                    missing,
                    unexpected,
                    shape_mismatches,
                )
                optimizer_loaded = False
            except RuntimeError:
                try:
                    module.core.load_state_dict(payload["model"], strict=True)
                except RuntimeError:
                    missing, unexpected, shape_mismatches = _load_state_dict_picf_compat(module.core, payload["model"])
                    logging.warning(
                        "Loaded PICF core payload with compatibility migration. "
                        "missing_keys=%s unexpected_keys=%s shape_mismatch_keys=%s. "
                        "Optimizer state will be reinitialized.",
                        missing,
                        unexpected,
                        shape_mismatches,
                    )
                    optimizer_loaded = False
    optimizer_payload = payload.get("optimizer")
    if optimizer_loaded and optimizer_payload is not None:
        try:
            optimizer.load_state_dict(optimizer_payload)
        except ValueError:
            logging.warning("Optimizer payload is incompatible with the current PICF architecture. Reinitializing optimizer.")
    elif optimizer_loaded:
        logging.info("No optimizer payload found in checkpoint; optimizer will be reinitialized.")
    if grad_clip_controller is not None and not grad_clip_controller.load_state_dict(payload.get("grad_clip_controller")):
        logging.info("Gradient clip controller state not restored from payload; starting with fresh history.")
    return int(payload.get("step", 0))


def _load_checkpoint_sequential_across_ranks(
    *,
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    rank: int,
    world_size: int,
    optimizer_checkpoint_mode: str = "auto",
    grad_clip_controller: _GradClipController | None = None,
) -> int:
    """Avoid shared-filesystem page-read stalls by serializing DDP checkpoint loads.

    Foundation checkpoints are still loaded independently during model construction,
    but the large resume checkpoint for PICF trainer state should only be read by one
    rank at a time on networked storage.
    """
    if _is_fsdp_model(model):
        loaded_step = _load_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            device=device,
            optimizer_checkpoint_mode=optimizer_checkpoint_mode,
            grad_clip_controller=grad_clip_controller,
        )
        _distributed_barrier(use_ddp=world_size > 1, device=device)
        if world_size > 1:
            step_tensor = torch.tensor([int(loaded_step)], device=device, dtype=torch.int64)
            dist.broadcast(step_tensor, src=0)
            return int(step_tensor.item())
        return int(loaded_step)
    if world_size <= 1:
        return _load_checkpoint(
            path=path,
            model=model,
            optimizer=optimizer,
            device=device,
            optimizer_checkpoint_mode=optimizer_checkpoint_mode,
            grad_clip_controller=grad_clip_controller,
        )
    loaded_step = 0
    for load_rank in range(int(world_size)):
        if rank == load_rank:
            loaded_step = _load_checkpoint(
                path=path,
                model=model,
                optimizer=optimizer,
                device=device,
                optimizer_checkpoint_mode=optimizer_checkpoint_mode,
                grad_clip_controller=grad_clip_controller,
            )
        _distributed_barrier(use_ddp=True, device=device)
    step_tensor = torch.tensor([int(loaded_step)], device=device, dtype=torch.int64)
    dist.broadcast(step_tensor, src=0)
    return int(step_tensor.item())


def _build_model_sequential_across_ranks(
    args: argparse.Namespace,
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> tuple[PicfFullCore, torch.nn.Module | None, bool]:
    """Build the training stack on each rank without serialized barriers.

    The v2.2 full-finetune path already spends significant wall clock in large
    checkpoint loads. Serializing that work rank-by-rank turns startup into a
    multi-stage stall without improving training semantics. The standard path now
    lets each rank construct its stack independently and relies on the regular
    distributed initialization immediately after model build.
    """
    del rank, world_size
    return _build_model(args, device=device)


def _build_model(args: argparse.Namespace, *, device: torch.device) -> tuple[PicfFullCore, torch.nn.Module | None, bool]:
    picf_enabled = _picf_mode_enabled(args)

    def _arg_or_default(name: str, default: Any) -> Any:
        value = getattr(args, name, None)
        return default if value is None else value

    builder = CalvinDepthToPicfPointCloud(args.calvin_root, stride=args.stride, max_points=args.max_points)
    config = PicfCoreConfig(
        device=str(device),
        hidden_dim=args.hidden_dim,
        posterior_hidden_dim=args.posterior_hidden_dim,
        latent_dim=args.latent_dim,
        innovation_dim=args.innovation_dim,
        control_dim=args.control_dim,
        semantic_dim=args.semantic_dim,
        semantic_cross_dim=args.semantic_cross_dim,
        future_hidden_dim=args.future_hidden_dim,
        persistent_anchors=args.persistent_anchors,
        observation_anchors=args.observation_anchors,
        effector_persistent_anchors=int(_arg_or_default("effector_persistent_anchors", _SPEC_DEFAULTS.effector_persistent_anchors)),
        effector_observation_anchors=int(_arg_or_default("effector_observation_anchors", _SPEC_DEFAULTS.effector_observation_anchors)),
        posterior_slot_identity_std=float(
            _arg_or_default("posterior_slot_identity_std", _SPEC_DEFAULTS.posterior_slot_identity_std)
        ),
        task_slot_identity_std=float(_arg_or_default("task_slot_identity_std", _SPEC_DEFAULTS.task_slot_identity_std)),
        posterior_bootstrap_from_observation=bool(
            _arg_or_default("posterior_bootstrap_from_observation", _SPEC_DEFAULTS.posterior_bootstrap_from_observation)
        ),
        fusion_layers=args.fusion_layers,
        posterior_layers=args.posterior_layers,
        predictive_layers=args.predictive_layers,
        control_layers=args.control_layers,
        control_query_tokens=args.control_query_tokens,
        predictive_query_tokens=args.predictive_query_tokens,
        task_local_queries=args.task_local_queries,
        task_effector_queries=int(_arg_or_default("task_effector_queries", _SPEC_DEFAULTS.task_effector_queries)),
        task_global_queries=args.task_global_queries,
        task_instruction_queries=args.task_instruction_queries,
        task_self_layers=args.task_self_layers,
        conditioned_control_queries=args.conditioned_control_queries,
        pi_prefix_queries=args.pi_prefix_queries,
        conditioned_future_queries=args.conditioned_future_queries,
        predictive_semantic_reads=args.predictive_semantic_reads,
        control_semantic_reads=args.control_semantic_reads,
        predictive_semantic_dropout_prob=args.predictive_semantic_dropout_prob,
        semantic_prefix_dropout_prob=args.semantic_prefix_dropout_prob,
        attention_heads=args.attention_heads,
        future_vote_heads=args.future_vote_heads,
        crop_radius_m=float(_arg_or_default("crop_radius_m", _SPEC_DEFAULTS.crop_radius_m)),
        global_scene_point_cap=int(_arg_or_default("global_scene_point_cap", _SPEC_DEFAULTS.global_scene_point_cap)),
        scene_anchor_border_patches=float(
            _arg_or_default("scene_anchor_border_patches", _SPEC_DEFAULTS.scene_anchor_border_patches)
        ),
        point_focus_sigma_m=float(_arg_or_default("point_focus_sigma_m", _SPEC_DEFAULTS.point_focus_sigma_m)),
        tactile_contact_tau_on=float(_arg_or_default("tactile_contact_tau_on", _SPEC_DEFAULTS.tactile_contact_tau_on)),
        tactile_contact_tau_off=float(_arg_or_default("tactile_contact_tau_off", _SPEC_DEFAULTS.tactile_contact_tau_off)),
        tactile_contact_temperature=float(
            _arg_or_default("tactile_contact_temperature", _SPEC_DEFAULTS.tactile_contact_temperature)
        ),
        tactile_contact_ema_beta=float(
            _arg_or_default("tactile_contact_ema_beta", _SPEC_DEFAULTS.tactile_contact_ema_beta)
        ),
        tactile_evidence_prob_floor=float(
            _arg_or_default("tactile_evidence_prob_floor", _SPEC_DEFAULTS.tactile_evidence_prob_floor)
        ),
        tactile_anchor_prob_on=float(_arg_or_default("tactile_anchor_prob_on", _SPEC_DEFAULTS.tactile_anchor_prob_on)),
        tactile_attach_to_object_owner=bool(
            _arg_or_default("tactile_attach_to_object_owner", _SPEC_DEFAULTS.tactile_attach_to_object_owner)
        ),
        task_visual_reread_topk=int(_arg_or_default("task_visual_reread_topk", _SPEC_DEFAULTS.task_visual_reread_topk)),
        task_tactile_reread_groups=int(
            _arg_or_default("task_tactile_reread_groups", _SPEC_DEFAULTS.task_tactile_reread_groups)
        ),
        task_point_reread_topk=int(_arg_or_default("task_point_reread_topk", _SPEC_DEFAULTS.task_point_reread_topk)),
        vl_anchor_router_enabled=bool(_arg_or_default("vl_anchor_router_enabled", _SPEC_DEFAULTS.vl_anchor_router_enabled)),
        vl_grounding_view=str(_arg_or_default("vl_grounding_view", _SPEC_DEFAULTS.vl_grounding_view)),
        vl_anchor_modes=int(_arg_or_default("vl_anchor_modes", _SPEC_DEFAULTS.vl_anchor_modes)),
        vl_anchor_nms_radius_m=float(
            _arg_or_default("vl_anchor_nms_radius_m", _SPEC_DEFAULTS.vl_anchor_nms_radius_m)
        ),
        vl_anchor_local_sigma_m=float(
            _arg_or_default("vl_anchor_local_sigma_m", _SPEC_DEFAULTS.vl_anchor_local_sigma_m)
        ),
        vl_min_visible_mass=float(_arg_or_default("vl_min_visible_mass", _SPEC_DEFAULTS.vl_min_visible_mass)),
        vl_heatmap_temperature=float(
            _arg_or_default("vl_heatmap_temperature", _SPEC_DEFAULTS.vl_heatmap_temperature)
        ),
        vl_obs_anchor_gate_init=float(
            _arg_or_default("vl_obs_anchor_gate_init", _SPEC_DEFAULTS.vl_obs_anchor_gate_init)
        ),
        vl_task_point_gate_init=float(
            _arg_or_default("vl_task_point_gate_init", _SPEC_DEFAULTS.vl_task_point_gate_init)
        ),
        vl_posterior_bind_gate_init=float(
            _arg_or_default("vl_posterior_bind_gate_init", _SPEC_DEFAULTS.vl_posterior_bind_gate_init)
        ),
        vl_prior_bias_clip=float(_arg_or_default("vl_prior_bias_clip", _SPEC_DEFAULTS.vl_prior_bias_clip)),
        mapg_enabled=bool(_arg_or_default("mapg_enabled", _SPEC_DEFAULTS.mapg_enabled)),
        mapg_anchor_count=int(_arg_or_default("mapg_anchor_count", _SPEC_DEFAULTS.mapg_anchor_count)),
        mapg_message_rounds=int(_arg_or_default("mapg_message_rounds", _SPEC_DEFAULTS.mapg_message_rounds)),
        mapg_visual_sigma_patches=float(
            _arg_or_default("mapg_visual_sigma_patches", _SPEC_DEFAULTS.mapg_visual_sigma_patches)
        ),
        mapg_tactile_sigma_m=float(_arg_or_default("mapg_tactile_sigma_m", _SPEC_DEFAULTS.mapg_tactile_sigma_m)),
        mapg_posterior_sigma_m=float(
            _arg_or_default("mapg_posterior_sigma_m", _SPEC_DEFAULTS.mapg_posterior_sigma_m)
        ),
        mapg_confidence_floor=float(_arg_or_default("mapg_confidence_floor", _SPEC_DEFAULTS.mapg_confidence_floor)),
        mapg_assignment_sinkhorn_iters=int(
            _arg_or_default("mapg_assignment_sinkhorn_iters", _SPEC_DEFAULTS.mapg_assignment_sinkhorn_iters)
        ),
        mapg_assignment_temperature=float(
            _arg_or_default("mapg_assignment_temperature", _SPEC_DEFAULTS.mapg_assignment_temperature)
        ),
        mapg_assignment_quality_uniform_mix=float(
            _arg_or_default("mapg_assignment_quality_uniform_mix", _SPEC_DEFAULTS.mapg_assignment_quality_uniform_mix)
        ),
        mapg_mode_confidence_threshold=float(
            _arg_or_default("mapg_mode_confidence_threshold", _SPEC_DEFAULTS.mapg_mode_confidence_threshold)
        ),
        mapg_obs_gate_init=float(_arg_or_default("mapg_obs_gate_init", _SPEC_DEFAULTS.mapg_obs_gate_init)),
        mapg_task_gate_init=float(_arg_or_default("mapg_task_gate_init", _SPEC_DEFAULTS.mapg_task_gate_init)),
        mapg_posterior_gate_init=float(
            _arg_or_default("mapg_posterior_gate_init", _SPEC_DEFAULTS.mapg_posterior_gate_init)
        ),
        mapg_control_gate_init=float(
            _arg_or_default("mapg_control_gate_init", _SPEC_DEFAULTS.mapg_control_gate_init)
        ),
        mapg_obs_point_mix_floor=float(
            _arg_or_default("mapg_obs_point_mix_floor", _SPEC_DEFAULTS.mapg_obs_point_mix_floor)
        ),
        mapg_prior_bias_clip=float(_arg_or_default("mapg_prior_bias_clip", _SPEC_DEFAULTS.mapg_prior_bias_clip)),
        aqr_mapg_enabled=bool(_arg_or_default("aqr_mapg_enabled", _SPEC_DEFAULTS.aqr_mapg_enabled)),
        aqr_query_count_physical=int(
            _arg_or_default("aqr_query_count_physical", _SPEC_DEFAULTS.aqr_query_count_physical)
        ),
        aqr_query_count_task=int(_arg_or_default("aqr_query_count_task", _SPEC_DEFAULTS.aqr_query_count_task)),
        aqr_role_layout=str(_arg_or_default("aqr_role_layout", _SPEC_DEFAULTS.aqr_role_layout)),
        aqr_query_rounds=int(_arg_or_default("aqr_query_rounds", _SPEC_DEFAULTS.aqr_query_rounds)),
        aqr_sinkhorn_iters=int(_arg_or_default("aqr_sinkhorn_iters", _SPEC_DEFAULTS.aqr_sinkhorn_iters)),
        aqr_sinkhorn_temperature=float(
            _arg_or_default("aqr_sinkhorn_temperature", _SPEC_DEFAULTS.aqr_sinkhorn_temperature)
        ),
        aqr_pg_grounding_enabled=bool(
            _arg_or_default("aqr_pg_grounding_enabled", _SPEC_DEFAULTS.aqr_pg_grounding_enabled)
        ),
        aqr_pg_image_support_enabled=bool(
            _arg_or_default("aqr_pg_image_support_enabled", _SPEC_DEFAULTS.aqr_pg_image_support_enabled)
        ),
        aqr_pg_image_support_weight=float(
            _arg_or_default("aqr_pg_image_support_weight", _SPEC_DEFAULTS.aqr_pg_image_support_weight)
        ),
        aqr_pg_entropy_threshold=float(
            _arg_or_default("aqr_pg_entropy_threshold", _SPEC_DEFAULTS.aqr_pg_entropy_threshold)
        ),
        aqr_pg_peak_threshold=float(
            _arg_or_default("aqr_pg_peak_threshold", _SPEC_DEFAULTS.aqr_pg_peak_threshold)
        ),
        aqr_pg_bias_weight=float(_arg_or_default("aqr_pg_bias_weight", _SPEC_DEFAULTS.aqr_pg_bias_weight)),
        aqr_support_bias_clip=float(
            _arg_or_default("aqr_support_bias_clip", _SPEC_DEFAULTS.aqr_support_bias_clip)
        ),
        aqr_ownership_prior_enabled=bool(
            _arg_or_default("aqr_ownership_prior_enabled", _SPEC_DEFAULTS.aqr_ownership_prior_enabled)
        ),
        aqr_ownership_prior_weight=float(
            _arg_or_default("aqr_ownership_prior_weight", _SPEC_DEFAULTS.aqr_ownership_prior_weight)
        ),
        aqr_ownership_point_prior_weight=float(
            _arg_or_default("aqr_ownership_point_prior_weight", _SPEC_DEFAULTS.aqr_ownership_point_prior_weight)
        ),
        aqr_ownership_point_prior_sigma_m=float(
            _arg_or_default("aqr_ownership_point_prior_sigma_m", _SPEC_DEFAULTS.aqr_ownership_point_prior_sigma_m)
        ),
        aqr_ownership_temporal_prior_weight=float(
            _arg_or_default("aqr_ownership_temporal_prior_weight", _SPEC_DEFAULTS.aqr_ownership_temporal_prior_weight)
        ),
        aqr_ownership_prior_uniform_mix=float(
            _arg_or_default("aqr_ownership_prior_uniform_mix", _SPEC_DEFAULTS.aqr_ownership_prior_uniform_mix)
        ),
        aqr_same_role_support_competition_enabled=bool(
            _arg_or_default(
                "aqr_same_role_support_competition_enabled",
                _SPEC_DEFAULTS.aqr_same_role_support_competition_enabled,
            )
        ),
        aqr_same_role_support_competition_weight=float(
            _arg_or_default(
                "aqr_same_role_support_competition_weight",
                _SPEC_DEFAULTS.aqr_same_role_support_competition_weight,
            )
        ),
        aqr_same_role_support_competition_iters=int(
            _arg_or_default(
                "aqr_same_role_support_competition_iters",
                _SPEC_DEFAULTS.aqr_same_role_support_competition_iters,
            )
        ),
        aqr_same_role_support_competition_physical_only=bool(
            _arg_or_default(
                "aqr_same_role_support_competition_physical_only",
                _SPEC_DEFAULTS.aqr_same_role_support_competition_physical_only,
            )
        ),
        aqr_active_slot_filter_enabled=bool(
            _arg_or_default("aqr_active_slot_filter_enabled", _SPEC_DEFAULTS.aqr_active_slot_filter_enabled)
        ),
        aqr_active_slot_min_per_role=int(
            _arg_or_default("aqr_active_slot_min_per_role", _SPEC_DEFAULTS.aqr_active_slot_min_per_role)
        ),
        aqr_active_slot_max_per_role=int(
            _arg_or_default("aqr_active_slot_max_per_role", _SPEC_DEFAULTS.aqr_active_slot_max_per_role)
        ),
        aqr_active_slot_min_confidence=float(
            _arg_or_default("aqr_active_slot_min_confidence", _SPEC_DEFAULTS.aqr_active_slot_min_confidence)
        ),
        aqr_active_slot_overlap_threshold=float(
            _arg_or_default("aqr_active_slot_overlap_threshold", _SPEC_DEFAULTS.aqr_active_slot_overlap_threshold)
        ),
        aqr_active_slot_relative_score_threshold=float(
            _arg_or_default(
                "aqr_active_slot_relative_score_threshold",
                _SPEC_DEFAULTS.aqr_active_slot_relative_score_threshold,
            )
        ),
        aqr_active_slot_geometry_duplicate_enabled=bool(
            _arg_or_default(
                "aqr_active_slot_geometry_duplicate_enabled",
                _SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_enabled,
            )
        ),
        aqr_active_slot_geometry_duplicate_sigma_m=float(
            _arg_or_default(
                "aqr_active_slot_geometry_duplicate_sigma_m",
                _SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_sigma_m,
            )
        ),
        aqr_active_slot_geometry_duplicate_threshold=float(
            _arg_or_default(
                "aqr_active_slot_geometry_duplicate_threshold",
                _SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_threshold,
            )
        ),
        vcap_enabled=bool(_arg_or_default("vcap_enabled", _SPEC_DEFAULTS.vcap_enabled)),
        vcap_max_active=int(_arg_or_default("vcap_max_active", _SPEC_DEFAULTS.vcap_max_active)),
        vcap_min_active=int(_arg_or_default("vcap_min_active", _SPEC_DEFAULTS.vcap_min_active)),
        vcap_stop_threshold=float(_arg_or_default("vcap_stop_threshold", _SPEC_DEFAULTS.vcap_stop_threshold)),
        vcap_action_grad_scale=float(
            _arg_or_default("vcap_action_grad_scale", _SPEC_DEFAULTS.vcap_action_grad_scale)
        ),
        aqr_context_slot_enabled=bool(
            _arg_or_default("aqr_context_slot_enabled", _SPEC_DEFAULTS.aqr_context_slot_enabled)
        ),
        aqr_context_slot_weight=float(
            _arg_or_default("aqr_context_slot_weight", _SPEC_DEFAULTS.aqr_context_slot_weight)
        ),
        aqr_context_slot_min_confidence=float(
            _arg_or_default(
                "aqr_context_slot_min_confidence",
                _SPEC_DEFAULTS.aqr_context_slot_min_confidence,
            )
        ),
        aqr_context_slot_min_score=float(
            _arg_or_default("aqr_context_slot_min_score", _SPEC_DEFAULTS.aqr_context_slot_min_score)
        ),
        aqr_context_slot_duplicate_overlap_threshold=float(
            _arg_or_default(
                "aqr_context_slot_duplicate_overlap_threshold",
                _SPEC_DEFAULTS.aqr_context_slot_duplicate_overlap_threshold,
            )
        ),
        aqr_context_slot_deduplicate_enabled=bool(
            _arg_or_default(
                "aqr_context_slot_deduplicate_enabled",
                _SPEC_DEFAULTS.aqr_context_slot_deduplicate_enabled,
            )
        ),
        aqr_context_slot_max_per_role=int(
            _arg_or_default("aqr_context_slot_max_per_role", _SPEC_DEFAULTS.aqr_context_slot_max_per_role)
        ),
        aqr_context_slot_self_overlap_threshold=float(
            _arg_or_default(
                "aqr_context_slot_self_overlap_threshold",
                _SPEC_DEFAULTS.aqr_context_slot_self_overlap_threshold,
            )
        ),
        aqr_context_slot_self_support_overlap_enabled=bool(
            _arg_or_default(
                "aqr_context_slot_self_support_overlap_enabled",
                _SPEC_DEFAULTS.aqr_context_slot_self_support_overlap_enabled,
            )
        ),
        aqr_context_slot_self_support_overlap_threshold=float(
            _arg_or_default(
                "aqr_context_slot_self_support_overlap_threshold",
                _SPEC_DEFAULTS.aqr_context_slot_self_support_overlap_threshold,
            )
        ),
        aqr_context_slot_active_support_overlap_enabled=bool(
            _arg_or_default(
                "aqr_context_slot_active_support_overlap_enabled",
                _SPEC_DEFAULTS.aqr_context_slot_active_support_overlap_enabled,
            )
        ),
        aqr_context_slot_active_support_overlap_threshold=float(
            _arg_or_default(
                "aqr_context_slot_active_support_overlap_threshold",
                _SPEC_DEFAULTS.aqr_context_slot_active_support_overlap_threshold,
            )
        ),
        aqr_context_slot_quality_gate_enabled=bool(
            _arg_or_default(
                "aqr_context_slot_quality_gate_enabled",
                _SPEC_DEFAULTS.aqr_context_slot_quality_gate_enabled,
            )
        ),
        aqr_slot_quality_owner_active_floor=float(
            _arg_or_default(
                "aqr_slot_quality_owner_active_floor",
                _SPEC_DEFAULTS.aqr_slot_quality_owner_active_floor,
            )
        ),
        aqr_control_graph_attention_bias_enabled=bool(
            _arg_or_default(
                "aqr_control_graph_attention_bias_enabled",
                _SPEC_DEFAULTS.aqr_control_graph_attention_bias_enabled,
            )
        ),
        aqr_control_graph_token_scaling_enabled=bool(
            _arg_or_default(
                "aqr_control_graph_token_scaling_enabled",
                _SPEC_DEFAULTS.aqr_control_graph_token_scaling_enabled,
            )
        ),
        aqr_control_graph_state_embedding_enabled=bool(
            _arg_or_default(
                "aqr_control_graph_state_embedding_enabled",
                _SPEC_DEFAULTS.aqr_control_graph_state_embedding_enabled,
            )
        ),
        aqr_control_graph_bias_min=float(
            _arg_or_default("aqr_control_graph_bias_min", _SPEC_DEFAULTS.aqr_control_graph_bias_min)
        ),
        posterior_owner_active_gate_enabled=bool(
            _arg_or_default(
                "posterior_owner_active_gate_enabled",
                _SPEC_DEFAULTS.posterior_owner_active_gate_enabled,
            )
        ),
        posterior_owner_active_min=float(
            _arg_or_default("posterior_owner_active_min", _SPEC_DEFAULTS.posterior_owner_active_min)
        ),
        posterior_owner_active_bias=float(
            _arg_or_default("posterior_owner_active_bias", _SPEC_DEFAULTS.posterior_owner_active_bias)
        ),
        aqr_vjepa_temporal_mode=str(
            _arg_or_default("aqr_vjepa_temporal_mode", _SPEC_DEFAULTS.aqr_vjepa_temporal_mode)
        ),
        aqr_vjepa_temporal_tokens=int(
            _arg_or_default("aqr_vjepa_temporal_tokens", _SPEC_DEFAULTS.aqr_vjepa_temporal_tokens)
        ),
        aqr_vjepa_temporal_include_delta=bool(
            _arg_or_default(
                "aqr_vjepa_temporal_include_delta",
                _SPEC_DEFAULTS.aqr_vjepa_temporal_include_delta,
            )
        ),
        vjepa_multiview_enabled=bool(
            _arg_or_default("vjepa_multiview_enabled", _SPEC_DEFAULTS.vjepa_multiview_enabled)
        ),
        vjepa_views=tuple(_SPEC_DEFAULTS.vjepa_views),
        vjepa_max_views=int(_arg_or_default("vjepa_max_views", _SPEC_DEFAULTS.vjepa_max_views)),
        aqr_obs_gate_init=float(_arg_or_default("aqr_obs_gate_init", _SPEC_DEFAULTS.aqr_obs_gate_init)),
        aqr_task_gate_init=float(_arg_or_default("aqr_task_gate_init", _SPEC_DEFAULTS.aqr_task_gate_init)),
        aqr_posterior_gate_init=float(
            _arg_or_default("aqr_posterior_gate_init", _SPEC_DEFAULTS.aqr_posterior_gate_init)
        ),
        aqr_control_gate_init=float(
            _arg_or_default("aqr_control_gate_init", _SPEC_DEFAULTS.aqr_control_gate_init)
        ),
        evidence_cache_enabled=bool(
            _arg_or_default("evidence_cache_enabled", _SPEC_DEFAULTS.evidence_cache_enabled)
        ),
        evidence_cache_len=int(_arg_or_default("evidence_cache_len", _SPEC_DEFAULTS.evidence_cache_len)),
        evidence_cache_read_weight=float(
            _arg_or_default("evidence_cache_read_weight", _SPEC_DEFAULTS.evidence_cache_read_weight)
        ),
        evidence_cache_innovation_downweight=float(
            _arg_or_default(
                "evidence_cache_innovation_downweight",
                _SPEC_DEFAULTS.evidence_cache_innovation_downweight,
            )
        ),
        evidence_cache_address_weight=float(
            _arg_or_default("evidence_cache_address_weight", _SPEC_DEFAULTS.evidence_cache_address_weight)
        ),
        evidence_cache_content_weight=float(
            _arg_or_default("evidence_cache_content_weight", _SPEC_DEFAULTS.evidence_cache_content_weight)
        ),
        evidence_cache_role_weight=float(
            _arg_or_default("evidence_cache_role_weight", _SPEC_DEFAULTS.evidence_cache_role_weight)
        ),
        tracklet_memory_enabled=bool(
            _arg_or_default("tracklet_memory_enabled", _SPEC_DEFAULTS.tracklet_memory_enabled)
        ),
        tracklet_max_tokens=int(_arg_or_default("tracklet_max_tokens", _SPEC_DEFAULTS.tracklet_max_tokens)),
        tracklet_confidence_floor=float(
            _arg_or_default("tracklet_confidence_floor", _SPEC_DEFAULTS.tracklet_confidence_floor)
        ),
        tracklet_read_weight=float(_arg_or_default("tracklet_read_weight", _SPEC_DEFAULTS.tracklet_read_weight)),
        proposal_memory_enabled=bool(
            _arg_or_default("proposal_memory_enabled", _SPEC_DEFAULTS.proposal_memory_enabled)
        ),
        proposal_max_tokens=int(_arg_or_default("proposal_max_tokens", _SPEC_DEFAULTS.proposal_max_tokens)),
        proposal_confidence_floor=float(
            _arg_or_default("proposal_confidence_floor", _SPEC_DEFAULTS.proposal_confidence_floor)
        ),
        proposal_read_weight=float(_arg_or_default("proposal_read_weight", _SPEC_DEFAULTS.proposal_read_weight)),
        proposal_age_decay_steps=float(
            _arg_or_default("proposal_age_decay_steps", _SPEC_DEFAULTS.proposal_age_decay_steps)
        ),
        proposal_shape_quality_enabled=bool(
            _arg_or_default("proposal_shape_quality_enabled", _SPEC_DEFAULTS.proposal_shape_quality_enabled)
        ),
        proposal_shape_area_min=float(
            _arg_or_default("proposal_shape_area_min", _SPEC_DEFAULTS.proposal_shape_area_min)
        ),
        proposal_shape_area_max=float(
            _arg_or_default("proposal_shape_area_max", _SPEC_DEFAULTS.proposal_shape_area_max)
        ),
        proposal_shape_aspect_min=float(
            _arg_or_default("proposal_shape_aspect_min", _SPEC_DEFAULTS.proposal_shape_aspect_min)
        ),
        proposal_context_quality_power=float(
            _arg_or_default("proposal_context_quality_power", _SPEC_DEFAULTS.proposal_context_quality_power)
        ),
        proposal_point_bridge_weight=float(
            _arg_or_default("proposal_point_bridge_weight", _SPEC_DEFAULTS.proposal_point_bridge_weight)
        ),
        proposal_point_bridge_edge_tau=float(
            _arg_or_default("proposal_point_bridge_edge_tau", _SPEC_DEFAULTS.proposal_point_bridge_edge_tau)
        ),
        proposal_mask_point_tau=float(
            _arg_or_default("proposal_mask_point_tau", _SPEC_DEFAULTS.proposal_mask_point_tau)
        ),
        proposal_anchor_seed_enabled=bool(
            _arg_or_default("proposal_anchor_seed_enabled", _SPEC_DEFAULTS.proposal_anchor_seed_enabled)
        ),
        proposal_anchor_seed_pre_reader_enabled=bool(
            _arg_or_default(
                "proposal_anchor_seed_pre_reader_enabled",
                _SPEC_DEFAULTS.proposal_anchor_seed_pre_reader_enabled,
            )
        ),
        proposal_anchor_seed_rows=int(
            _arg_or_default("proposal_anchor_seed_rows", _SPEC_DEFAULTS.proposal_anchor_seed_rows)
        ),
        proposal_anchor_seed_weight=float(
            _arg_or_default("proposal_anchor_seed_weight", _SPEC_DEFAULTS.proposal_anchor_seed_weight)
        ),
        proposal_anchor_seed_token_weight=float(
            _arg_or_default("proposal_anchor_seed_token_weight", _SPEC_DEFAULTS.proposal_anchor_seed_token_weight)
        ),
        proposal_anchor_seed_score_floor=float(
            _arg_or_default("proposal_anchor_seed_score_floor", _SPEC_DEFAULTS.proposal_anchor_seed_score_floor)
        ),
        proposal_anchor_seed_point_topk=int(
            _arg_or_default("proposal_anchor_seed_point_topk", _SPEC_DEFAULTS.proposal_anchor_seed_point_topk)
        ),
        proposal_anchor_seed_point_power=float(
            _arg_or_default("proposal_anchor_seed_point_power", _SPEC_DEFAULTS.proposal_anchor_seed_point_power)
        ),
        object_candidate_assignment_enabled=bool(
            _arg_or_default(
                "object_candidate_assignment_enabled",
                _SPEC_DEFAULTS.object_candidate_assignment_enabled,
            )
        ),
        object_candidate_assignment_temperature=float(
            _arg_or_default(
                "object_candidate_assignment_temperature",
                _SPEC_DEFAULTS.object_candidate_assignment_temperature,
            )
        ),
        object_candidate_background_prior=float(
            _arg_or_default("object_candidate_background_prior", _SPEC_DEFAULTS.object_candidate_background_prior)
        ),
        object_candidate_background_quality_weight=float(
            _arg_or_default(
                "object_candidate_background_quality_weight",
                _SPEC_DEFAULTS.object_candidate_background_quality_weight,
            )
        ),
        object_candidate_row_support_floor=float(
            _arg_or_default(
                "object_candidate_row_support_floor",
                _SPEC_DEFAULTS.object_candidate_row_support_floor,
            )
        ),
        object_candidate_eligible_roles=_parse_int_tuple(
            _arg_or_default(
                "object_candidate_eligible_roles",
                _SPEC_DEFAULTS.object_candidate_eligible_roles,
            )
        ),
        object_candidate_max_rows_per_candidate=int(
            _arg_or_default(
                "object_candidate_max_rows_per_candidate",
                _SPEC_DEFAULTS.object_candidate_max_rows_per_candidate,
            )
        ),
        object_candidate_row_capacity=float(
            _arg_or_default("object_candidate_row_capacity", _SPEC_DEFAULTS.object_candidate_row_capacity)
        ),
        object_candidate_row_capacity_iters=int(
            _arg_or_default(
                "object_candidate_row_capacity_iters",
                _SPEC_DEFAULTS.object_candidate_row_capacity_iters,
            )
        ),
        object_candidate_point_weight=float(
            _arg_or_default("object_candidate_point_weight", _SPEC_DEFAULTS.object_candidate_point_weight)
        ),
        object_candidate_proposal_weight=float(
            _arg_or_default("object_candidate_proposal_weight", _SPEC_DEFAULTS.object_candidate_proposal_weight)
        ),
        object_candidate_seed_weight=float(
            _arg_or_default("object_candidate_seed_weight", _SPEC_DEFAULTS.object_candidate_seed_weight)
        ),
        object_candidate_task_owner_weight=float(
            _arg_or_default("object_candidate_task_owner_weight", _SPEC_DEFAULTS.object_candidate_task_owner_weight)
        ),
        object_candidate_anchor_score_weight=float(
            _arg_or_default(
                "object_candidate_anchor_score_weight",
                _SPEC_DEFAULTS.object_candidate_anchor_score_weight,
            )
        ),
        object_candidate_point_mix=float(
            _arg_or_default("object_candidate_point_mix", _SPEC_DEFAULTS.object_candidate_point_mix)
        ),
        object_candidate_proposal_mix=float(
            _arg_or_default("object_candidate_proposal_mix", _SPEC_DEFAULTS.object_candidate_proposal_mix)
        ),
        object_candidate_min_shape_quality=float(
            _arg_or_default("object_candidate_min_shape_quality", _SPEC_DEFAULTS.object_candidate_min_shape_quality)
        ),
        object_candidate_owner_transport_enabled=bool(
            _arg_or_default(
                "object_candidate_owner_transport_enabled",
                _SPEC_DEFAULTS.object_candidate_owner_transport_enabled,
            )
        ),
        object_candidate_owner_roles=_parse_int_tuple(
            _arg_or_default(
                "object_candidate_owner_roles",
                _SPEC_DEFAULTS.object_candidate_owner_roles,
            )
        ),
        object_candidate_owner_min_share=float(
            _arg_or_default("object_candidate_owner_min_share", _SPEC_DEFAULTS.object_candidate_owner_min_share)
        ),
        object_candidate_owner_point_mix=float(
            _arg_or_default("object_candidate_owner_point_mix", _SPEC_DEFAULTS.object_candidate_owner_point_mix)
        ),
        object_candidate_owner_geometry_mix=float(
            _arg_or_default("object_candidate_owner_geometry_mix", _SPEC_DEFAULTS.object_candidate_owner_geometry_mix)
        ),
        posterior_owner_transport_candidate_geometry_mix=float(
            _arg_or_default(
                "posterior_owner_transport_candidate_geometry_mix",
                _SPEC_DEFAULTS.posterior_owner_transport_candidate_geometry_mix,
            )
        ),
        object_explanation_point_core_mass=float(
            _arg_or_default("object_explanation_point_core_mass", _SPEC_DEFAULTS.object_explanation_point_core_mass)
        ),
        object_explanation_point_core_topk=int(
            _arg_or_default("object_explanation_point_core_topk", _SPEC_DEFAULTS.object_explanation_point_core_topk)
        ),
        object_explanation_point_loss_clip=float(
            _arg_or_default("object_explanation_point_loss_clip", _SPEC_DEFAULTS.object_explanation_point_loss_clip)
        ),
        task_owner_proposal_point_bridge_weight=float(
            _arg_or_default(
                "task_owner_proposal_point_bridge_weight",
                _SPEC_DEFAULTS.task_owner_proposal_point_bridge_weight,
            )
        ),
        task_owner_bias_enabled=bool(
            _arg_or_default("task_owner_bias_enabled", _SPEC_DEFAULTS.task_owner_bias_enabled)
        ),
        task_owner_visual_bias_weight=float(
            _arg_or_default("task_owner_visual_bias_weight", _SPEC_DEFAULTS.task_owner_visual_bias_weight)
        ),
        task_owner_proposal_bias_weight=float(
            _arg_or_default("task_owner_proposal_bias_weight", _SPEC_DEFAULTS.task_owner_proposal_bias_weight)
        ),
        task_owner_proposal_point_bias_weight=float(
            _arg_or_default(
                "task_owner_proposal_point_bias_weight",
                _SPEC_DEFAULTS.task_owner_proposal_point_bias_weight,
            )
        ),
        task_owner_proposal_objectness_power=float(
            _arg_or_default("task_owner_proposal_objectness_power", _SPEC_DEFAULTS.task_owner_proposal_objectness_power)
        ),
        task_owner_proposal_static_only=bool(
            _arg_or_default("task_owner_proposal_static_only", _SPEC_DEFAULTS.task_owner_proposal_static_only)
        ),
        task_owner_proposal_topk=int(
            _arg_or_default("task_owner_proposal_topk", _SPEC_DEFAULTS.task_owner_proposal_topk)
        ),
        task_owner_proposal_score_floor=float(
            _arg_or_default("task_owner_proposal_score_floor", _SPEC_DEFAULTS.task_owner_proposal_score_floor)
        ),
        bind_support_signature_weight=float(
            _arg_or_default("bind_support_signature_weight", _SPEC_DEFAULTS.bind_support_signature_weight)
        ),
        bind_embedding_signature_weight=float(
            _arg_or_default("bind_embedding_signature_weight", _SPEC_DEFAULTS.bind_embedding_signature_weight)
        ),
        bind_quadratic_signature_weight=float(
            _arg_or_default("bind_quadratic_signature_weight", _SPEC_DEFAULTS.bind_quadratic_signature_weight)
        ),
        bind_low_rank_signature_weight=float(
            _arg_or_default("bind_low_rank_signature_weight", _SPEC_DEFAULTS.bind_low_rank_signature_weight)
        ),
        binding_signature_dim=int(_arg_or_default("binding_signature_dim", _SPEC_DEFAULTS.binding_signature_dim)),
        binding_low_rank_signature_rank=int(
            _arg_or_default("binding_low_rank_signature_rank", _SPEC_DEFAULTS.binding_low_rank_signature_rank)
        ),
        binding_signature_centering_enabled=bool(
            _arg_or_default(
                "binding_signature_centering_enabled",
                _SPEC_DEFAULTS.binding_signature_centering_enabled,
            )
        ),
        binding_signature_centering_min_tokens=int(
            _arg_or_default(
                "binding_signature_centering_min_tokens",
                _SPEC_DEFAULTS.binding_signature_centering_min_tokens,
            )
        ),
        binding_signature_score_calibration_enabled=bool(
            _arg_or_default(
                "binding_signature_score_calibration_enabled",
                _SPEC_DEFAULTS.binding_signature_score_calibration_enabled,
            )
        ),
        binding_signature_score_calibration_mode=str(
            _arg_or_default(
                "binding_signature_score_calibration_mode",
                _SPEC_DEFAULTS.binding_signature_score_calibration_mode,
            )
        ),
        binding_signature_score_min_std=float(
            _arg_or_default(
                "binding_signature_score_min_std",
                _SPEC_DEFAULTS.binding_signature_score_min_std,
            )
        ),
        binding_signature_score_clip=float(
            _arg_or_default(
                "binding_signature_score_clip",
                _SPEC_DEFAULTS.binding_signature_score_clip,
            )
        ),
        posterior_binding_signature_memory_enabled=bool(
            _arg_or_default(
                "posterior_binding_signature_memory_enabled",
                _spec_default("posterior_binding_signature_memory_enabled"),
            )
        ),
        posterior_binding_signature_dispersion_gate_enabled=bool(
            _arg_or_default(
                "posterior_binding_signature_dispersion_gate_enabled",
                _spec_default("posterior_binding_signature_dispersion_gate_enabled"),
            )
        ),
        posterior_binding_signature_update_rate=float(
            _arg_or_default(
                "posterior_binding_signature_update_rate",
                _spec_default("posterior_binding_signature_update_rate"),
            )
        ),
        posterior_binding_signature_update_max_rate=float(
            _arg_or_default(
                "posterior_binding_signature_update_max_rate",
                _spec_default("posterior_binding_signature_update_max_rate"),
            )
        ),
        posterior_binding_signature_min_support=float(
            _arg_or_default(
                "posterior_binding_signature_min_support",
                _spec_default("posterior_binding_signature_min_support"),
            )
        ),
        posterior_binding_signature_owner_weight=float(
            _arg_or_default(
                "posterior_binding_signature_owner_weight",
                _spec_default("posterior_binding_signature_owner_weight"),
            )
        ),
        posterior_binding_signature_measurement_min_std=float(
            _arg_or_default(
                "posterior_binding_signature_measurement_min_std",
                _spec_default("posterior_binding_signature_measurement_min_std"),
            )
        ),
        posterior_binding_signature_measurement_margin_min=float(
            _arg_or_default(
                "posterior_binding_signature_measurement_margin_min",
                _spec_default("posterior_binding_signature_measurement_margin_min"),
            )
        ),
        posterior_binding_signature_measurement_margin_temperature=float(
            _arg_or_default(
                "posterior_binding_signature_measurement_margin_temperature",
                _spec_default("posterior_binding_signature_measurement_margin_temperature"),
            )
        ),
        bind_address_weight=float(_arg_or_default("bind_address_weight", _SPEC_DEFAULTS.bind_address_weight)),
        bind_address_innovation_downweight=float(
            _arg_or_default("bind_address_innovation_downweight", _SPEC_DEFAULTS.bind_address_innovation_downweight)
        ),
        address_update_rate=float(_arg_or_default("address_update_rate", _SPEC_DEFAULTS.address_update_rate)),
        address_update_max_rate=float(
            _arg_or_default("address_update_max_rate", _SPEC_DEFAULTS.address_update_max_rate)
        ),
        posterior_occupancy_prior_enabled=bool(
            _arg_or_default(
                "posterior_occupancy_prior_enabled",
                _SPEC_DEFAULTS.posterior_occupancy_prior_enabled,
            )
        ),
        posterior_occupancy_prior_weight=float(
            _arg_or_default(
                "posterior_occupancy_prior_weight",
                _SPEC_DEFAULTS.posterior_occupancy_prior_weight,
            )
        ),
        posterior_occupancy_prior_sigma_m=float(
            _arg_or_default(
                "posterior_occupancy_prior_sigma_m",
                _SPEC_DEFAULTS.posterior_occupancy_prior_sigma_m,
            )
        ),
        posterior_occupancy_prior_clip=float(
            _arg_or_default(
                "posterior_occupancy_prior_clip",
                _SPEC_DEFAULTS.posterior_occupancy_prior_clip,
            )
        ),
        observation_anchor_seed_point_mix=float(
            _arg_or_default(
                "observation_anchor_seed_point_mix",
                _SPEC_DEFAULTS.observation_anchor_seed_point_mix,
            )
        ),
        recycle_normalize_residual_summary=bool(
            _arg_or_default(
                "recycle_normalize_residual_summary",
                _SPEC_DEFAULTS.recycle_normalize_residual_summary,
            )
        ),
        recycle_residual_norm_mode=str(
            _arg_or_default("recycle_residual_norm_mode", _SPEC_DEFAULTS.recycle_residual_norm_mode)
        ),
        recycle_logit_clamp=float(_arg_or_default("recycle_logit_clamp", _SPEC_DEFAULTS.recycle_logit_clamp)),
        posterior_slotwise_recycle_residual=bool(
            _arg_or_default(
                "posterior_slotwise_recycle_residual",
                _SPEC_DEFAULTS.posterior_slotwise_recycle_residual,
            )
        ),
        posterior_lifecycle_calibration_enabled=bool(
            _arg_or_default(
                "posterior_lifecycle_calibration_enabled",
                _SPEC_DEFAULTS.posterior_lifecycle_calibration_enabled,
            )
        ),
        posterior_lifecycle_support_min=float(
            _arg_or_default(
                "posterior_lifecycle_support_min",
                _SPEC_DEFAULTS.posterior_lifecycle_support_min,
            )
        ),
        posterior_lifecycle_support_temperature=float(
            _arg_or_default(
                "posterior_lifecycle_support_temperature",
                _SPEC_DEFAULTS.posterior_lifecycle_support_temperature,
            )
        ),
        posterior_lifecycle_margin_min=float(
            _arg_or_default(
                "posterior_lifecycle_margin_min",
                _SPEC_DEFAULTS.posterior_lifecycle_margin_min,
            )
        ),
        posterior_lifecycle_margin_temperature=float(
            _arg_or_default(
                "posterior_lifecycle_margin_temperature",
                _SPEC_DEFAULTS.posterior_lifecycle_margin_temperature,
            )
        ),
        posterior_lifecycle_entropy_weight=float(
            _arg_or_default(
                "posterior_lifecycle_entropy_weight",
                _SPEC_DEFAULTS.posterior_lifecycle_entropy_weight,
            )
        ),
        posterior_lifecycle_owner_weight=float(
            _arg_or_default(
                "posterior_lifecycle_owner_weight",
                _SPEC_DEFAULTS.posterior_lifecycle_owner_weight,
            )
        ),
        posterior_lifecycle_innovation_downweight=float(
            _arg_or_default(
                "posterior_lifecycle_innovation_downweight",
                _SPEC_DEFAULTS.posterior_lifecycle_innovation_downweight,
            )
        ),
        posterior_owner_transport_enabled=bool(
            _arg_or_default(
                "posterior_owner_transport_enabled",
                _SPEC_DEFAULTS.posterior_owner_transport_enabled,
            )
        ),
        posterior_owner_transport_roles=_parse_int_tuple(
            _arg_or_default(
                "posterior_owner_transport_roles",
                _SPEC_DEFAULTS.posterior_owner_transport_roles,
            )
        ),
        posterior_owner_transport_max_per_role=int(
            _arg_or_default(
                "posterior_owner_transport_max_per_role",
                _SPEC_DEFAULTS.posterior_owner_transport_max_per_role,
            )
        ),
        posterior_owner_transport_max_rate=float(
            _arg_or_default(
                "posterior_owner_transport_max_rate",
                _SPEC_DEFAULTS.posterior_owner_transport_max_rate,
            )
        ),
        posterior_owner_transport_precision_gain=float(
            _arg_or_default(
                "posterior_owner_transport_precision_gain",
                _SPEC_DEFAULTS.posterior_owner_transport_precision_gain,
            )
        ),
        posterior_owner_transport_min_mass=float(
            _arg_or_default(
                "posterior_owner_transport_min_mass",
                _SPEC_DEFAULTS.posterior_owner_transport_min_mass,
            )
        ),
        posterior_owner_transport_direct_candidate_assignment=bool(
            _arg_or_default(
                "posterior_owner_transport_direct_candidate_assignment",
                _SPEC_DEFAULTS.posterior_owner_transport_direct_candidate_assignment,
            )
        ),
        posterior_owner_transport_direct_candidate_min_score=float(
            _arg_or_default(
                "posterior_owner_transport_direct_candidate_min_score",
                _SPEC_DEFAULTS.posterior_owner_transport_direct_candidate_min_score,
            )
        ),
        posterior_owner_transport_assignment_floor=float(
            _arg_or_default(
                "posterior_owner_transport_assignment_floor",
                _SPEC_DEFAULTS.posterior_owner_transport_assignment_floor,
            )
        ),
        posterior_owner_transport_reliability_floor=float(
            _arg_or_default(
                "posterior_owner_transport_reliability_floor",
                _SPEC_DEFAULTS.posterior_owner_transport_reliability_floor,
            )
        ),
        posterior_owner_transport_covariance_scale=float(
            _arg_or_default(
                "posterior_owner_transport_covariance_scale",
                _SPEC_DEFAULTS.posterior_owner_transport_covariance_scale,
            )
        ),
        posterior_owner_transport_inactive_prior=float(
            _arg_or_default(
                "posterior_owner_transport_inactive_prior",
                _SPEC_DEFAULTS.posterior_owner_transport_inactive_prior,
            )
        ),
        posterior_owner_transport_activates_file=bool(
            _arg_or_default(
                "posterior_owner_transport_activates_file",
                _SPEC_DEFAULTS.posterior_owner_transport_activates_file,
            )
        ),
        posterior_owner_transport_active_threshold=float(
            _arg_or_default(
                "posterior_owner_transport_active_threshold",
                _SPEC_DEFAULTS.posterior_owner_transport_active_threshold,
            )
        ),
        posterior_file_competition_enabled=bool(
            _arg_or_default(
                "posterior_file_competition_enabled",
                _SPEC_DEFAULTS.posterior_file_competition_enabled,
            )
        ),
        posterior_file_competition_min_per_role=int(
            _arg_or_default(
                "posterior_file_competition_min_per_role",
                _SPEC_DEFAULTS.posterior_file_competition_min_per_role,
            )
        ),
        posterior_file_competition_max_per_role=int(
            _arg_or_default(
                "posterior_file_competition_max_per_role",
                _SPEC_DEFAULTS.posterior_file_competition_max_per_role,
            )
        ),
        posterior_file_competition_min_support=float(
            _arg_or_default(
                "posterior_file_competition_min_support",
                _SPEC_DEFAULTS.posterior_file_competition_min_support,
            )
        ),
        posterior_file_competition_relative_score_threshold=float(
            _arg_or_default(
                "posterior_file_competition_relative_score_threshold",
                _SPEC_DEFAULTS.posterior_file_competition_relative_score_threshold,
            )
        ),
        posterior_file_competition_support_overlap_threshold=float(
            _arg_or_default(
                "posterior_file_competition_support_overlap_threshold",
                _SPEC_DEFAULTS.posterior_file_competition_support_overlap_threshold,
            )
        ),
        posterior_file_competition_geometry_duplicate_enabled=bool(
            _arg_or_default(
                "posterior_file_competition_geometry_duplicate_enabled",
                _SPEC_DEFAULTS.posterior_file_competition_geometry_duplicate_enabled,
            )
        ),
        posterior_file_competition_geometry_sigma_m=float(
            _arg_or_default(
                "posterior_file_competition_geometry_sigma_m",
                _SPEC_DEFAULTS.posterior_file_competition_geometry_sigma_m,
            )
        ),
        posterior_file_competition_geometry_threshold=float(
            _arg_or_default(
                "posterior_file_competition_geometry_threshold",
                _SPEC_DEFAULTS.posterior_file_competition_geometry_threshold,
            )
        ),
        posterior_birth_competition_enabled=bool(
            _arg_or_default(
                "posterior_birth_competition_enabled",
                _SPEC_DEFAULTS.posterior_birth_competition_enabled,
            )
        ),
        posterior_birth_competition_max_per_role=int(
            _arg_or_default(
                "posterior_birth_competition_max_per_role",
                _SPEC_DEFAULTS.posterior_birth_competition_max_per_role,
            )
        ),
        posterior_birth_competition_min_score=float(
            _arg_or_default(
                "posterior_birth_competition_min_score",
                _SPEC_DEFAULTS.posterior_birth_competition_min_score,
            )
        ),
        posterior_birth_competition_inactive_only=bool(
            _arg_or_default(
                "posterior_birth_competition_inactive_only",
                _SPEC_DEFAULTS.posterior_birth_competition_inactive_only,
            )
        ),
        posterior_birth_alpha_suppression_power=float(
            _arg_or_default(
                "posterior_birth_alpha_suppression_power",
                _SPEC_DEFAULTS.posterior_birth_alpha_suppression_power,
            )
        ),
        legacy_local_refinement_opt_in=bool(
            _arg_or_default(
                "legacy_local_refinement_opt_in",
                _SPEC_DEFAULTS.legacy_local_refinement_opt_in,
            )
        ),
        local_refinement_enabled=bool(
            _arg_or_default("local_refinement_enabled", _SPEC_DEFAULTS.local_refinement_enabled)
        ),
        local_refinement_topk=int(_arg_or_default("local_refinement_topk", _SPEC_DEFAULTS.local_refinement_topk)),
        local_refinement_weight=float(
            _arg_or_default("local_refinement_weight", _SPEC_DEFAULTS.local_refinement_weight)
        ),
        local_refinement_binding_weight=float(
            _arg_or_default("local_refinement_binding_weight", _SPEC_DEFAULTS.local_refinement_binding_weight)
        ),
        slot_jepa_enabled=bool(_arg_or_default("slot_jepa_enabled", _SPEC_DEFAULTS.slot_jepa_enabled)),
        support_prediction_enabled=bool(
            _arg_or_default("support_prediction_enabled", _SPEC_DEFAULTS.support_prediction_enabled)
        ),
        ordinal_relation_enabled=bool(
            _arg_or_default("ordinal_relation_enabled", _SPEC_DEFAULTS.ordinal_relation_enabled)
        ),
        ordinal_weak_target_enabled=bool(
            _arg_or_default("ordinal_weak_target_enabled", _SPEC_DEFAULTS.ordinal_weak_target_enabled)
        ),
        lambda_vl_heatmap_task=float(_arg_or_default("lambda_vl_heatmap_task", _SPEC_DEFAULTS.lambda_vl_heatmap_task)),
        lambda_vl_heatmap_effector=float(
            _arg_or_default("lambda_vl_heatmap_effector", _SPEC_DEFAULTS.lambda_vl_heatmap_effector)
        ),
        lambda_vl_heatmap_interaction=float(
            _arg_or_default("lambda_vl_heatmap_interaction", _SPEC_DEFAULTS.lambda_vl_heatmap_interaction)
        ),
        lambda_vl_point_consistency=float(
            _arg_or_default("lambda_vl_point_consistency", _SPEC_DEFAULTS.lambda_vl_point_consistency)
        ),
        lambda_vl_anchor_diversity=float(
            _arg_or_default("lambda_vl_anchor_diversity", _SPEC_DEFAULTS.lambda_vl_anchor_diversity)
        ),
        visual_real_grid=int(_arg_or_default("visual_real_grid", _SPEC_DEFAULTS.visual_real_grid)),
        action_output_clip=getattr(args, "action_output_clip", None),
        tokenwise_ff_chunk_size=int(_arg_or_default("tokenwise_ff_chunk_size", _SPEC_DEFAULTS.tokenwise_ff_chunk_size)),
        require_pi0_action_generator=bool(_arg_or_default("require_pi0_action_generator", _SPEC_DEFAULTS.require_pi0_action_generator)),
        action_prefix_stopgrad=bool(_arg_or_default("action_prefix_stopgrad", _SPEC_DEFAULTS.action_prefix_stopgrad)),
        action_prefix_norm_mode=str(
            _arg_or_default("action_prefix_norm_mode", _SPEC_DEFAULTS.action_prefix_norm_mode)
        ),
        action_prefix_rms_target=float(
            _arg_or_default("action_prefix_rms_target", _SPEC_DEFAULTS.action_prefix_rms_target)
        ),
        action_prefix_norm_eps=float(
            _arg_or_default("action_prefix_norm_eps", _SPEC_DEFAULTS.action_prefix_norm_eps)
        ),
        action_prefix_value_clip=float(
            _arg_or_default("action_prefix_value_clip", _SPEC_DEFAULTS.action_prefix_value_clip)
        ),
        picf_action_condition_enabled=bool(
            _arg_or_default("picf_action_condition_enabled", _SPEC_DEFAULTS.picf_action_condition_enabled)
        ),
        action_prefix_output_gate=float(
            _arg_or_default("action_prefix_output_gate", _SPEC_DEFAULTS.action_prefix_output_gate)
        ),
        action_prefix_teacher_mode=str(
            _arg_or_default("action_prefix_teacher_mode", _SPEC_DEFAULTS.action_prefix_teacher_mode)
        ),
        action_prefix_teacher_ema_decay=float(
            _arg_or_default("action_prefix_teacher_ema_decay", _SPEC_DEFAULTS.action_prefix_teacher_ema_decay)
        ),
        action_prefix_teacher_blend=float(
            _arg_or_default("action_prefix_teacher_blend", _SPEC_DEFAULTS.action_prefix_teacher_blend)
        ),
        lambda_action_prefix_trust=float(
            _arg_or_default("lambda_action_prefix_trust", _SPEC_DEFAULTS.lambda_action_prefix_trust)
        ),
        action_context_tokens=int(_arg_or_default("action_context_tokens", _SPEC_DEFAULTS.action_context_tokens)),
        action_context_integration=str(
            _arg_or_default("action_context_integration", _SPEC_DEFAULTS.action_context_integration)
        ),
        action_context_stopgrad=bool(
            _arg_or_default("action_context_stopgrad", _SPEC_DEFAULTS.action_context_stopgrad)
        ),
        action_context_norm_mode=str(
            _arg_or_default("action_context_norm_mode", _SPEC_DEFAULTS.action_context_norm_mode)
        ),
        action_context_rms_target=float(
            _arg_or_default("action_context_rms_target", _SPEC_DEFAULTS.action_context_rms_target)
        ),
        action_context_output_gate=float(
            _arg_or_default("action_context_output_gate", _SPEC_DEFAULTS.action_context_output_gate)
        ),
        action_context_include_query_tokens=bool(
            _arg_or_default(
                "action_context_include_query_tokens",
                _SPEC_DEFAULTS.action_context_include_query_tokens,
            )
        ),
    )
    point_feature_extractor = None
    if picf_enabled and args.point_backbone == "sonata":
        point_feature_extractor = SonataPointFeatureExtractor(
            SonataPointConfig(
                checkpoint_path=args.sonata_checkpoint_path,
                stage_name=args.sonata_stage_name,
                device=str(device),
                dtype=args.sonata_dtype,
                trainable=bool(args.point_backbone_trainable),
                allow_random_init=False,
                enable_flash=not bool(args.sonata_disable_flash),
            )
        )

    if picf_enabled and args.visual_mode == "encoder":
        visual_config = VjepaVisualConfig(
            model_name=args.visual_model_name,
            checkpoint_path=args.visual_checkpoint_path,
            checkpoint_key=args.visual_checkpoint_key,
            camera_json_path=args.calvin_root,
            device=str(device),
            dtype=args.visual_dtype,
            trainable=bool(args.visual_trainable),
            feature_mode=args.visual_feature_mode,
            use_activation_checkpointing=bool(args.visual_activation_checkpointing),
            feature_cache_root=args.vjepa_feature_cache_root,
            feature_cache_mode=args.vjepa_feature_cache_mode,
            feature_cache_temporal_slices=args.vjepa_feature_cache_temporal_slices,
            feature_cache_storage_dtype=args.vjepa_feature_cache_storage_dtype,
            img_size=args.visual_img_size,
            num_frames=args.visual_num_frames,
            patch_size=args.visual_patch_size,
            tubelet_size=args.visual_tubelet_size,
            use_last_two_mean=bool(args.visual_use_last_two_mean),
        )
        visual_encoder = None
        use_visual_override = False
    else:
        visual_config = VjepaVisualConfig(
            camera_json_path=args.calvin_root,
            arch_name_override="vit_tiny",
            img_size=64,
            num_frames=4,
            device=str(device),
            dtype="float32",
            feature_mode=args.visual_feature_mode,
        )
        visual_encoder = _NullVisualEncoder()
        use_visual_override = bool(picf_enabled)

    tactile_config = None
    tactile_encoder = None
    if picf_enabled and args.tactile_mode == "encoder":
        tactile_contact_stats = _load_tactile_contact_stats_json(args.tactile_contact_stats_path)
        tactile_config = AnyTouchConfig(
            checkpoint_path=args.tactile_checkpoint_path,
            device=str(device),
            dtype=args.tactile_dtype,
            trainable=bool(args.tactile_trainable),
            num_frames=args.tactile_num_frames,
            stride=args.tactile_stride,
            allow_random_init=False,
            require_background=True,
            contact_stats_payload=tactile_contact_stats,
        )
    else:
        tactile_encoder = _NullTactileEncoder()

    semantic_encoder: torch.nn.Module | None
    if args.semantic_mode == "paligemma":
        semantic_encoder = PaliGemmaSemanticEncoder(
            PaliGemmaSemanticConfig(
                source=args.semantic_source,
                model_name=args.semantic_model_name,
                checkpoint_path=args.semantic_checkpoint_path,
                checkpoint_config_path=args.semantic_checkpoint_config_path,
                revision=args.semantic_revision,
                paligemma_variant=args.semantic_paligemma_variant,
                action_expert_variant=args.semantic_action_expert_variant,
                device=str(device),
                dtype=args.semantic_dtype,
                trainable=bool(args.semantic_trainable),
                trainable_scope=str(getattr(args, "semantic_trainable_scope", "backbone_only")),
                gradient_checkpointing=bool(args.semantic_gradient_checkpointing),
                include_gripper_image=bool(args.semantic_use_gripper),
                max_length=args.semantic_max_length,
                prompt_state_normalization=str(getattr(args, "prompt_state_normalization", "none")),
                prompt_state_norm_stats_path=getattr(args, "prompt_state_norm_stats_path", None),
                action_horizon=int(args.action_horizon),
                tokenwise_chunk_size=int(getattr(args, "semantic_tokenwise_chunk_size", 0)),
                projection_chunk_size=int(getattr(args, "semantic_projection_chunk_size", 0)),
                mlp_chunk_size=int(getattr(args, "semantic_mlp_chunk_size", 0)),
                action_context_adapter_gate_init=float(
                    getattr(args, "semantic_action_context_adapter_gate_init", -2.0)
                ),
                action_context_adapter_rms_cap=bool(
                    getattr(args, "semantic_action_context_adapter_rms_cap", True)
                ),
                action_flow_loss=str(getattr(args, "semantic_action_flow_loss", "mse")),
                action_flow_huber_delta=float(getattr(args, "semantic_action_flow_huber_delta", 1.0)),
                action_flow_time_alpha=float(getattr(args, "semantic_action_flow_time_alpha", 1.5)),
                action_flow_time_beta=float(getattr(args, "semantic_action_flow_time_beta", 1.0)),
                action_context_readout_aux_weight=float(
                    getattr(args, "semantic_action_context_readout_aux_weight", 0.0)
                ),
                action_context_readout_aux_loss=str(
                    getattr(args, "semantic_action_context_readout_aux_loss", "smooth_l1")
                ),
                action_context_readout_aux_huber_delta=float(
                    getattr(args, "semantic_action_context_readout_aux_huber_delta", 1.0)
                ),
                action_context_token_aux_weight=float(
                    getattr(args, "semantic_action_context_token_aux_weight", 0.0)
                ),
                action_context_token_aux_bins=int(
                    getattr(args, "semantic_action_context_token_aux_bins", 256)
                ),
                action_context_token_aux_clip=float(
                    getattr(args, "semantic_action_context_token_aux_clip", 1.0)
                ),
                action_context_flow_residual_enabled=bool(
                    getattr(args, "semantic_action_context_flow_residual_enabled", False)
                ),
                action_context_flow_residual_gate_init=float(
                    getattr(args, "semantic_action_context_flow_residual_gate_init", -2.0)
                ),
                action_context_flow_residual_time_floor=float(
                    getattr(args, "semantic_action_context_flow_residual_time_floor", 0.05)
                ),
                action_context_flow_residual_rms_cap=bool(
                    getattr(args, "semantic_action_context_flow_residual_rms_cap", True)
                ),
                action_expert_router_enabled=bool(
                    getattr(args, "semantic_action_expert_router_enabled", False)
                ),
                action_expert_router_experts=int(getattr(args, "semantic_action_expert_router_experts", 4)),
                action_expert_router_rank=int(getattr(args, "semantic_action_expert_router_rank", 64)),
                action_expert_router_gate_init=float(
                    getattr(args, "semantic_action_expert_router_gate_init", -2.5)
                ),
                action_expert_router_temperature=float(
                    getattr(args, "semantic_action_expert_router_temperature", 1.0)
                ),
                action_expert_router_rms_cap=bool(
                    getattr(args, "semantic_action_expert_router_rms_cap", True)
                ),
            )
        )
    else:
        semantic_encoder = None

    core = PicfFullCore(
        builder,
        config=config,
        point_feature_extractor=point_feature_extractor,
        visual_config=visual_config,
        visual_encoder=visual_encoder,
        tactile_config=tactile_config,
        tactile_encoder=tactile_encoder,
    )
    if not picf_enabled:
        _freeze_initialized_module_parameters(core)
    return core, semantic_encoder, use_visual_override


def _build_loss_config(args: argparse.Namespace) -> PicfTransitionLossConfig:
    defaults = _LOSS_DEFAULTS
    return PicfTransitionLossConfig(
        lambda_action_pos=float(getattr(args, "lambda_action_pos", defaults.lambda_action_pos)),
        lambda_action_rot=float(getattr(args, "lambda_action_rot", defaults.lambda_action_rot)),
        lambda_action_gripper=float(getattr(args, "lambda_action_gripper", defaults.lambda_action_gripper)),
        lambda_visual_latent=float(getattr(args, "lambda_visual_latent", defaults.lambda_visual_latent)),
        lambda_visual_real=float(getattr(args, "lambda_visual_real", defaults.lambda_visual_real)),
        lambda_tactile_real=float(getattr(args, "lambda_tactile_real", defaults.lambda_tactile_real)),
        lambda_point_real=float(getattr(args, "lambda_point_real", defaults.lambda_point_real)),
        lambda_semantic_future_aux=float(getattr(args, "lambda_semantic_future_aux", defaults.lambda_semantic_future_aux)),
        lambda_anchor_pv=float(getattr(args, "lambda_anchor_pv", defaults.lambda_anchor_pv)),
        lambda_anchor_object_pull=float(getattr(args, "lambda_anchor_object_pull", defaults.lambda_anchor_object_pull)),
        anchor_object_pull_sigma_m=float(getattr(args, "anchor_object_pull_sigma_m", defaults.anchor_object_pull_sigma_m)),
        anchor_object_pull_confirmation_threshold=float(
            getattr(args, "anchor_object_pull_confirmation_threshold", defaults.anchor_object_pull_confirmation_threshold)
        ),
        anchor_object_pull_allowed_roles=_parse_int_tuple(
            getattr(args, "anchor_object_pull_allowed_roles", defaults.anchor_object_pull_allowed_roles)
        ),
        anchor_object_pull_graph_weight=float(
            getattr(args, "anchor_object_pull_graph_weight", defaults.anchor_object_pull_graph_weight)
        ),
        anchor_object_pull_posterior_weight=float(
            getattr(args, "anchor_object_pull_posterior_weight", defaults.anchor_object_pull_posterior_weight)
        ),
        anchor_object_pull_target_quality_gate_enabled=bool(
            getattr(
                args,
                "anchor_object_pull_target_quality_gate_enabled",
                defaults.anchor_object_pull_target_quality_gate_enabled,
            )
        ),
        anchor_object_pull_target_quality_sigma_m=float(
            getattr(
                args,
                "anchor_object_pull_target_quality_sigma_m",
                defaults.anchor_object_pull_target_quality_sigma_m,
            )
        ),
        anchor_object_pull_target_quality_min=float(
            getattr(args, "anchor_object_pull_target_quality_min", defaults.anchor_object_pull_target_quality_min)
        ),
        anchor_object_pull_target_quality_power=float(
            getattr(args, "anchor_object_pull_target_quality_power", defaults.anchor_object_pull_target_quality_power)
        ),
        anchor_object_pull_target_core_mass=float(
            getattr(args, "anchor_object_pull_target_core_mass", defaults.anchor_object_pull_target_core_mass)
        ),
        anchor_object_pull_target_core_topk=int(
            getattr(args, "anchor_object_pull_target_core_topk", defaults.anchor_object_pull_target_core_topk)
        ),
        lambda_pv_weak=float(getattr(args, "lambda_pv_weak", defaults.lambda_pv_weak)),
        lambda_pt=float(getattr(args, "lambda_pt", defaults.lambda_pt)),
        anchor_pv_object_gate_enabled=bool(
            getattr(args, "anchor_pv_object_gate_enabled", defaults.anchor_pv_object_gate_enabled)
        ),
        anchor_pv_active_object_gate_only=bool(
            getattr(args, "anchor_pv_active_object_gate_only", defaults.anchor_pv_active_object_gate_only)
        ),
        anchor_pv_object_gate_floor=float(
            getattr(args, "anchor_pv_object_gate_floor", defaults.anchor_pv_object_gate_floor)
        ),
        anchor_pv_object_normalize_by_object_mass=bool(
            getattr(
                args,
                "anchor_pv_object_normalize_by_object_mass",
                defaults.anchor_pv_object_normalize_by_object_mass,
            )
        ),
        anchor_pv_object_distribution_loss=bool(
            getattr(args, "anchor_pv_object_distribution_loss", defaults.anchor_pv_object_distribution_loss)
        ),
        anchor_pv_object_distribution_confirmed_only=bool(
            getattr(
                args,
                "anchor_pv_object_distribution_confirmed_only",
                defaults.anchor_pv_object_distribution_confirmed_only,
            )
        ),
        anchor_pv_object_distribution_confirmation_threshold=float(
            getattr(
                args,
                "anchor_pv_object_distribution_confirmation_threshold",
                defaults.anchor_pv_object_distribution_confirmation_threshold,
            )
        ),
        tau_pv=float(getattr(args, "tau_pv", defaults.tau_pv)),
        tau_pt=float(getattr(args, "tau_pt", defaults.tau_pt)),
        tau_route_p=float(getattr(args, "tau_route_p", defaults.tau_route_p)),
        tau_route_v=float(getattr(args, "tau_route_v", defaults.tau_route_v)),
        pt_bag_radius_m=float(getattr(args, "pt_bag_radius_m", defaults.pt_bag_radius_m)),
        pt_bag_sigma_m=float(getattr(args, "pt_bag_sigma_m", defaults.pt_bag_sigma_m)),
        pt_bag_kmin=int(getattr(args, "pt_bag_kmin", defaults.pt_bag_kmin)),
        pt_back_slack_m=float(getattr(args, "pt_back_slack_m", defaults.pt_back_slack_m)),
        p_align_on=float(getattr(args, "p_align_on", defaults.p_align_on)),
        p_align_off=float(getattr(args, "p_align_off", defaults.p_align_off)),
        tactile_aux_force_scale=float(getattr(args, "tactile_aux_force_scale", defaults.tactile_aux_force_scale)),
        tactile_aux_indent_scale=float(getattr(args, "tactile_aux_indent_scale", defaults.tactile_aux_indent_scale)),
        tactile_aux_pressure_scale=float(getattr(args, "tactile_aux_pressure_scale", defaults.tactile_aux_pressure_scale)),
        tactile_aux_pose_scale=float(getattr(args, "tactile_aux_pose_scale", getattr(args, "crop_radius_m", 0.10))),
        tactile_aux_huber_delta=float(getattr(args, "tactile_aux_huber_delta", defaults.tactile_aux_huber_delta)),
        enable_aux_budgeting=bool(getattr(args, "enable_aux_budgeting", defaults.enable_aux_budgeting)),
        aux_budget_physical_ratio=float(getattr(args, "aux_budget_physical_ratio", defaults.aux_budget_physical_ratio)),
        aux_budget_semantic_ratio=float(getattr(args, "aux_budget_semantic_ratio", defaults.aux_budget_semantic_ratio)),
        aux_budget_alignment_ratio=float(getattr(args, "aux_budget_alignment_ratio", defaults.aux_budget_alignment_ratio)),
        aux_budget_floor=float(getattr(args, "aux_budget_floor", defaults.aux_budget_floor)),
        aux_budget_alignment_floor=float(getattr(args, "aux_budget_alignment_floor", defaults.aux_budget_alignment_floor)),
        lambda_vl_heatmap_task=float(getattr(args, "lambda_vl_heatmap_task", defaults.lambda_vl_heatmap_task)),
        lambda_vl_heatmap_effector=float(getattr(args, "lambda_vl_heatmap_effector", defaults.lambda_vl_heatmap_effector)),
        lambda_vl_heatmap_interaction=float(getattr(args, "lambda_vl_heatmap_interaction", defaults.lambda_vl_heatmap_interaction)),
        lambda_vl_point_consistency=float(getattr(args, "lambda_vl_point_consistency", defaults.lambda_vl_point_consistency)),
        lambda_vl_anchor_diversity=float(getattr(args, "lambda_vl_anchor_diversity", defaults.lambda_vl_anchor_diversity)),
        vl_heatmap_sigma_patches=float(getattr(args, "vl_heatmap_sigma_patches", defaults.vl_heatmap_sigma_patches)),
        vl_point_consistency_eps=float(getattr(args, "vl_point_consistency_eps", defaults.vl_point_consistency_eps)),
        vl_anchor_diversity_radius_m=float(getattr(args, "vl_anchor_diversity_radius_m", defaults.vl_anchor_diversity_radius_m)),
        lambda_mapg_siglip=float(getattr(args, "lambda_mapg_siglip", defaults.lambda_mapg_siglip)),
        lambda_mapg_vicreg=float(getattr(args, "lambda_mapg_vicreg", defaults.lambda_mapg_vicreg)),
        lambda_mapg_cycle=float(getattr(args, "lambda_mapg_cycle", defaults.lambda_mapg_cycle)),
        lambda_mapg_masked_modality=float(getattr(args, "lambda_mapg_masked_modality", defaults.lambda_mapg_masked_modality)),
        lambda_mapg_routing=float(getattr(args, "lambda_mapg_routing", defaults.lambda_mapg_routing)),
        lambda_mapg_support_diversity=float(getattr(args, "lambda_mapg_support_diversity", defaults.lambda_mapg_support_diversity)),
        lambda_mapg_geometry_diversity=float(getattr(args, "lambda_mapg_geometry_diversity", defaults.lambda_mapg_geometry_diversity)),
        lambda_slot_jepa=float(getattr(args, "lambda_slot_jepa", defaults.lambda_slot_jepa)),
        lambda_support_pred=float(getattr(args, "lambda_support_pred", defaults.lambda_support_pred)),
        lambda_binding_consistency=float(getattr(args, "lambda_binding_consistency", defaults.lambda_binding_consistency)),
        lambda_aqr_denoising=float(getattr(args, "lambda_aqr_denoising", defaults.lambda_aqr_denoising)),
        lambda_slot_quality=float(getattr(args, "lambda_slot_quality", defaults.lambda_slot_quality)),
        aqr_denoising_active_object_only=bool(
            getattr(args, "aqr_denoising_active_object_only", defaults.aqr_denoising_active_object_only)
        ),
        aqr_denoising_confirmed_object_only=bool(
            getattr(args, "aqr_denoising_confirmed_object_only", defaults.aqr_denoising_confirmed_object_only)
        ),
        aqr_denoising_confirmation_threshold=float(
            getattr(args, "aqr_denoising_confirmation_threshold", defaults.aqr_denoising_confirmation_threshold)
        ),
        lambda_vcap_unexplained=float(getattr(args, "lambda_vcap_unexplained", defaults.lambda_vcap_unexplained)),
        lambda_vcap_duplicate=float(getattr(args, "lambda_vcap_duplicate", defaults.lambda_vcap_duplicate)),
        lambda_vcap_count=float(getattr(args, "lambda_vcap_count", defaults.lambda_vcap_count)),
        lambda_vcap_continuity=float(getattr(args, "lambda_vcap_continuity", defaults.lambda_vcap_continuity)),
        lambda_object_explanation_feature=float(
            getattr(args, "lambda_object_explanation_feature", defaults.lambda_object_explanation_feature)
        ),
        lambda_object_explanation_point=float(
            getattr(args, "lambda_object_explanation_point", defaults.lambda_object_explanation_point)
        ),
        lambda_object_explanation_contact=float(
            getattr(args, "lambda_object_explanation_contact", defaults.lambda_object_explanation_contact)
        ),
        lambda_object_explanation_duplicate=float(
            getattr(args, "lambda_object_explanation_duplicate", defaults.lambda_object_explanation_duplicate)
        ),
        lambda_object_explanation_background=float(
            getattr(args, "lambda_object_explanation_background", defaults.lambda_object_explanation_background)
        ),
        object_explanation_active_object_only=bool(
            getattr(args, "object_explanation_active_object_only", defaults.object_explanation_active_object_only)
        ),
        object_explanation_duplicate_margin=float(
            getattr(args, "object_explanation_duplicate_margin", defaults.object_explanation_duplicate_margin)
        ),
        object_explanation_point_loss_clip=float(
            getattr(args, "object_explanation_point_loss_clip", defaults.object_explanation_point_loss_clip)
        ),
        object_explanation_point_quality_gate_enabled=bool(
            getattr(
                args,
                "object_explanation_point_quality_gate_enabled",
                defaults.object_explanation_point_quality_gate_enabled,
            )
        ),
        object_explanation_point_quality_min=float(
            getattr(args, "object_explanation_point_quality_min", defaults.object_explanation_point_quality_min)
        ),
        object_explanation_point_quality_power=float(
            getattr(args, "object_explanation_point_quality_power", defaults.object_explanation_point_quality_power)
        ),
        object_explanation_point_outlier_prior=float(
            getattr(args, "object_explanation_point_outlier_prior", defaults.object_explanation_point_outlier_prior)
        ),
        detach_action_loss_from_picf=bool(getattr(args, "detach_action_loss_from_picf", defaults.detach_action_loss_from_picf)),
        mapg_siglip_tau=float(getattr(args, "mapg_siglip_tau", defaults.mapg_siglip_tau)),
        mapg_vicreg_var_target=float(getattr(args, "mapg_vicreg_var_target", defaults.mapg_vicreg_var_target)),
        mapg_vicreg_cov_weight=float(getattr(args, "mapg_vicreg_cov_weight", defaults.mapg_vicreg_cov_weight)),
        mapg_support_div_margin_visual=float(getattr(args, "mapg_support_div_margin_visual", defaults.mapg_support_div_margin_visual)),
        mapg_support_div_margin_point=float(getattr(args, "mapg_support_div_margin_point", defaults.mapg_support_div_margin_point)),
        mapg_support_div_margin_tactile=float(getattr(args, "mapg_support_div_margin_tactile", defaults.mapg_support_div_margin_tactile)),
        mapg_support_div_margin_posterior=float(getattr(args, "mapg_support_div_margin_posterior", defaults.mapg_support_div_margin_posterior)),
        mapg_support_div_sigma_visual_patches=float(getattr(args, "mapg_support_div_sigma_visual_patches", defaults.mapg_support_div_sigma_visual_patches)),
        mapg_support_div_sigma_point_m=float(getattr(args, "mapg_support_div_sigma_point_m", defaults.mapg_support_div_sigma_point_m)),
        mapg_support_div_direct_visual_weight=float(
            getattr(args, "mapg_support_div_direct_visual_weight", defaults.mapg_support_div_direct_visual_weight)
        ),
        mapg_support_div_local_candidate_weight=float(
            getattr(args, "mapg_support_div_local_candidate_weight", defaults.mapg_support_div_local_candidate_weight)
        ),
        mapg_support_div_local_margin=float(getattr(args, "mapg_support_div_local_margin", defaults.mapg_support_div_local_margin)),
        mapg_support_div_tail_topk=int(getattr(args, "mapg_support_div_tail_topk", defaults.mapg_support_div_tail_topk)),
        mapg_geometry_diversity_margin=float(getattr(args, "mapg_geometry_diversity_margin", defaults.mapg_geometry_diversity_margin)),
        mapg_geometry_diversity_jitter_m=float(getattr(args, "mapg_geometry_diversity_jitter_m", defaults.mapg_geometry_diversity_jitter_m)),
    )


def _materialize_model_parameters(
    model: _PicfWindowTrainer,
    *,
    source: _CalvinTransitionSource,
    rank: int,
) -> None:
    """Run one no-grad window to initialize lazy modules before DDP wrapping."""
    warmup_index = int(rank) % max(len(source), 1)
    warmup_window = source.window(warmup_index)
    was_training = model.training
    model.train()
    with torch.no_grad():
        _ = model(warmup_window)
        core = model.core
        picf_enabled = bool(getattr(getattr(model, "policy", None), "picf_enabled", True))
        if picf_enabled and any(
            isinstance(param, UninitializedParameter)
            for name, param in model.named_parameters()
            if name.startswith("core.prior_")
        ):
            # A valid unroll_steps=1 training run never calls the recurrent
            # prior during the normal warmup window. Materialize that lazy
            # branch explicitly with one no-grad recurrent transition rather
            # than weakening the uninitialized-parameter check.
            first = dataclasses.replace(warmup_window.frames[0], reset_scaffold=True)
            second_index = min(1, len(warmup_window.frames) - 1)
            second = dataclasses.replace(warmup_window.frames[second_index], reset_scaffold=False)
            first_visual = (
                _rgb_visual_override(first.rgb_static, grid=model.visual_grid)
                if model.use_visual_override
                else None
            )
            second_visual = (
                _rgb_visual_override(second.rgb_static, grid=model.visual_grid)
                if model.use_visual_override
                else None
            )
            first_action_chunk = first.action_chunk if first.action_chunk is not None else first.action
            second_action_chunk = second.action_chunk if second.action_chunk is not None else second.action
            first_forward = model.policy.forward_train_transition(
                previous=None,
                current=first,
                visual_map_override=first_visual,
                action_chunk_target=first_action_chunk,
            )
            _ = model.policy.forward_train_transition(
                previous=first_forward.next_state,
                current=second,
                visual_map_override=second_visual,
                action_chunk_target=second_action_chunk,
            )
        if picf_enabled and isinstance(core.tactile_token_proj.weight, UninitializedParameter):
            # picf_core_train uses a null tactile encoder, so tactile lazy layers
            # need an explicit placeholder init before DDP inspects parameters.
            tactile_pooled_dim = 4 * 768
            tactile_sensor_in = torch.zeros(
                (1, (2 * tactile_pooled_dim) + 9),
                device=core.device,
                dtype=core.dtype,
            )
            tactile_tokens = core.tactile_token_proj(tactile_sensor_in)
            _ = core.tactile_align_proj(tactile_tokens)
        tracklet_token_proj = getattr(core, "tracklet_token_proj", None)
        if picf_enabled and tracklet_token_proj is not None and isinstance(tracklet_token_proj.weight, UninitializedParameter):
            # Tracklet evidence is optional and absent from raw CALVIN batches.
            # Initialize its lazy adapter explicitly so FSDP/DDP sees the same
            # anchor-only parameter set whether or not precomputed tracks exist.
            tracklet_in = torch.zeros(
                (1, 23),
                device=core.device,
                dtype=core.dtype,
            )
            _ = tracklet_token_proj(tracklet_in)
        proposal_token_proj = getattr(core, "proposal_token_proj", None)
        if picf_enabled and proposal_token_proj is not None and isinstance(proposal_token_proj.weight, UninitializedParameter):
            # Proposal memory is also optional; keep its typed-evidence adapter
            # materialized instead of weakening the strict lazy-param audit.
            proposal_in = torch.zeros(
                (1, 26),
                device=core.device,
                dtype=core.dtype,
            )
            _ = proposal_token_proj(proposal_in)
        if picf_enabled and isinstance(core.tactile_error_encoder.weight, UninitializedParameter):
            tactile_error_in = torch.zeros(
                (1, 3 * core.config.tactile_real_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.tactile_error_encoder(tactile_error_in)
        if picf_enabled and isinstance(core.visual_error_encoder.weight, UninitializedParameter):
            visual_error_in = torch.zeros(
                (1, 3 * core.config.hidden_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.visual_error_encoder(visual_error_in)
        if picf_enabled and isinstance(core.visual_real_error_encoder.weight, UninitializedParameter):
            visual_real_error_in = torch.zeros(
                (1, 3 * core.config.visual_real_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.visual_real_error_encoder(visual_real_error_in)
        if picf_enabled and isinstance(core.point_error_encoder.weight, UninitializedParameter):
            point_error_in = torch.zeros(
                (1, 3 * core.config.point_real_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.point_error_encoder(point_error_in)
        if picf_enabled and isinstance(core.innovation_proj.weight, UninitializedParameter):
            branch_dim = max(core.config.hidden_dim // 4, 32)
            innovation_in = torch.zeros(
                (1, (4 * branch_dim) + 4),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.innovation_proj(innovation_in)
        binding_signature_proj = getattr(core, "binding_signature_proj", None)
        if picf_enabled and binding_signature_proj is not None and isinstance(binding_signature_proj.weight, UninitializedParameter):
            # The pairwise binding-signature adapter is used whenever typed
            # supports are present, but materialize it explicitly so optional
            # support no-op batches expose the same anchor-only parameter set.
            binding_in = torch.zeros(
                (1, core.config.hidden_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = binding_signature_proj(binding_in)
        if picf_enabled and isinstance(core.tactile_route_reread.key_proj.weight, UninitializedParameter):
            tactile_dense_dim = _infer_tactile_dense_dim(core)
            dummy_queries = torch.zeros(
                (1, core.config.tactile_group_proposals, core.config.hidden_dim),
                device=core.device,
                dtype=core.dtype,
            )
            dummy_keys = torch.zeros(
                (1, 1, tactile_dense_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.tactile_route_reread(dummy_queries, dummy_keys)
        if picf_enabled and isinstance(core.tactile_native_reread.key_proj.weight, UninitializedParameter):
            tactile_dense_dim = _infer_tactile_dense_dim(core)
            tactile_query_count = max(int(getattr(core.config, "tactile_latent_tokens", 1)), 1)
            dummy_queries = torch.zeros(
                (1, tactile_query_count, core.config.hidden_dim),
                device=core.device,
                dtype=core.dtype,
            )
            dummy_keys = torch.zeros(
                (1, 1, tactile_dense_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.tactile_native_reread(dummy_queries, dummy_keys)
        if picf_enabled and isinstance(core.task_tactile_reread.key_proj.weight, UninitializedParameter):
            tactile_dense_dim = _infer_tactile_dense_dim(core)
            task_query_count = max(
                int(core.config.task_local_queries)
                + int(core.config.task_global_queries)
                + int(core.config.task_instruction_queries),
                1,
            )
            dummy_queries = torch.zeros(
                (1, task_query_count, core.config.hidden_dim),
                device=core.device,
                dtype=core.dtype,
            )
            dummy_keys = torch.zeros(
                (1, 1, tactile_dense_dim),
                device=core.device,
                dtype=core.dtype,
            )
            _ = core.task_tactile_reread(dummy_queries, dummy_keys)
    model.zero_grad(set_to_none=True)
    remaining_uninitialized = [
        name
        for name, param in model.named_parameters()
        if isinstance(param, UninitializedParameter) and bool(getattr(param, "requires_grad", False))
    ]
    if remaining_uninitialized:
        raise RuntimeError(
            "Uninitialized parameters remain after warmup materialization: "
            + ", ".join(sorted(remaining_uninitialized))
        )
    if not was_training:
        model.eval()


def _infer_tactile_dense_dim(core: torch.nn.Module) -> int:
    """Return the PICF-space tactile dense width used by tactile rereads.

    AnyTouch patch tokens are encoder-native when they leave the tactile
    backbone, but PICF immediately maps them through
    `core.tactile_patch_token_proj` before storing them in dense tactile memory.
    The lazy reread blocks therefore must be materialized at `hidden_dim`.
    Initializing them with the AnyTouch native width makes the warmup contract
    disagree with the real forward path and crashes on the first dense contact.
    """

    return int(getattr(getattr(core, "config", None), "hidden_dim", 512))


def _prepare_output_dir(
    *, output_dir: Path, args: argparse.Namespace, is_main: bool, use_ddp: bool, device: torch.device
) -> None:
    if is_main:
        if args.resume and args.overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive.")
        if not args.resume:
            if output_dir.exists():
                has_content = any(output_dir.iterdir())
                if args.overwrite:
                    shutil.rmtree(output_dir)
                    logging.info("Overwriting checkpoint directory: %s", output_dir)
                elif has_content:
                    raise FileExistsError(
                        f"Checkpoint directory {output_dir} already exists. Use --resume or --overwrite."
                    )
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
        else:
            if not output_dir.exists():
                raise FileNotFoundError(f"Checkpoint directory does not exist for resume: {output_dir}")
    _distributed_barrier(use_ddp=use_ddp, device=device)


def _collect_trainable_params(module: torch.nn.Module | None) -> list[torch.nn.Parameter]:
    if module is None:
        return []
    return [param for param in module.parameters() if getattr(param, "requires_grad", False)]


def _freeze_initialized_module_parameters(module: torch.nn.Module | None) -> None:
    if module is None:
        return
    for param in module.parameters():
        if isinstance(param, UninitializedParameter):
            param.requires_grad = False
            continue
        param.requires_grad_(False)


_ANCHOR_ONLY_TRAINABLE_PATTERNS: tuple[str, ...] = (
    "core.modality_embedding.*",
    "core.point_token_proj.*",
    "core.visual_token_proj.*",
    "core.tactile_token_proj.*",
    "core.tracklet_token_proj.*",
    "core.proposal_token_proj.*",
    "core.point_align_proj.*",
    "core.visual_align_proj.*",
    "core.tactile_align_proj.*",
    "core.binding_signature_proj.*",
    "core.binding_quadratic_diag",
    "core.binding_low_rank_left.*",
    "core.binding_low_rank_right.*",
    "core.temporal_visual_time_proj.*",
    "core.temporal_visual_view_embedding.*",
    "core.projective_bias_head.*",
    "core.token_fusion.*",
    "core.obs_reader.*",
    "core.obs_self.*",
    "core.activity_head.*",
    "core.recycle_head.*",
    "core.residual_mu_head.*",
    "core.residual_logvar_head.*",
    "core.residual_h_head.*",
    "core.residual_c_head.*",
    "core.posterior_slot_hidden",
    "core.posterior_slot_token",
    "core.anchor_seed_proj.*",
    "core.anchor_reader.*",
    "core.contact_head.*",
    "core.prior_lstm.*",
    "core.post_write_proj.*",
    "core.post_lstm.*",
    "core.posterior_token_proj.*",
    "core.posterior_self.*",
    "core.posterior_pool.*",
    "core.aqr_*",
    "core.mapg_obs_gate_logit",
    "core.mapg_task_gate_logit",
    "core.mapg_posterior_gate_logit",
    "core.slot_support_pred_head.*",
    "core.evidence_delta.*",
    "core.evidence_gate.*",
)

_RECYCLE_PATH_TRAINABLE_PATTERNS: tuple[str, ...] = (
    "core.recycle_head.*",
    "core.residual_mu_head.*",
    "core.residual_logvar_head.*",
    "core.residual_h_head.*",
    "core.residual_c_head.*",
)


def _safe_parameter_numel(param: torch.nn.Parameter) -> int:
    if isinstance(param, UninitializedParameter):
        return 0
    return int(param.numel())


def _set_parameter_trainable(param: torch.nn.Parameter, trainable: bool) -> None:
    if isinstance(param, UninitializedParameter):
        param.requires_grad = bool(trainable)
        return
    param.requires_grad_(bool(trainable))


def _matches_any_pattern(name: str, patterns: tuple[str, ...]) -> bool:
    return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)


def _apply_picf_trainable_scope(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    """Apply high-level parameter trainability scopes after lazy materialization.

    `anchor_only` is a diagnostic profile: it freezes perception, semantic,
    PI0.5 action/control, and predictive heads while leaving the typed-evidence
    anchor router, posterior binding, address/cache/local support path, and
    observation-anchor adapters trainable. This keeps large-batch anchor
    convergence probes isolated from policy-capacity changes.
    """

    scope = str(getattr(args, "picf_trainable_scope", "all")).lower().replace("-", "_")
    if scope == "all":
        trainable_names = [name for name, param in model.named_parameters() if getattr(param, "requires_grad", False)]
        trainable_numel = sum(
            _safe_parameter_numel(param) for _name, param in model.named_parameters() if getattr(param, "requires_grad", False)
        )
        total_numel = sum(_safe_parameter_numel(param) for _name, param in model.named_parameters())
        return {
            "scope": scope,
            "trainable_param_tensors": len(trainable_names),
            "trainable_numel": int(trainable_numel),
            "total_numel": int(total_numel),
            "matched_names_sample": trainable_names[:24],
        }
    if scope == "policy_only":
        for name, param in model.named_parameters():
            if name.startswith("core."):
                _set_parameter_trainable(param, False)
        trainable_names = [name for name, param in model.named_parameters() if getattr(param, "requires_grad", False)]
        trainable_numel = sum(
            _safe_parameter_numel(param) for _name, param in model.named_parameters() if getattr(param, "requires_grad", False)
        )
        total_numel = sum(_safe_parameter_numel(param) for _name, param in model.named_parameters())
        if not trainable_names:
            raise RuntimeError("picf_trainable_scope=policy_only matched no trainable non-core parameters.")
        if any(name.startswith("core.") for name in trainable_names):
            raise RuntimeError("picf_trainable_scope=policy_only left PICF core parameters trainable.")
        info = {
            "scope": scope,
            "trainable_param_tensors": len(trainable_names),
            "trainable_numel": int(trainable_numel),
            "total_numel": int(total_numel),
            "matched_names_sample": trainable_names[:24],
        }
        if logger is not None:
            frozen_numel = int(total_numel) - int(trainable_numel)
            logger.info(
                "Trainable scope: scope=%s trainable_tensors=%s trainable_numel=%s frozen_numel=%s total_numel=%s frozen_prefix=core.",
                scope,
                len(trainable_names),
                int(trainable_numel),
                frozen_numel,
                int(total_numel),
            )
            logger.info("Trainable scope sample: %s", ", ".join(trainable_names[:24]))
        return info
    if scope != "anchor_only":
        raise ValueError(f"Unsupported picf_trainable_scope={scope!r}.")

    matched_names: list[str] = []
    for _name, param in model.named_parameters():
        _set_parameter_trainable(param, False)
    for name, param in model.named_parameters():
        if _matches_any_pattern(name, _ANCHOR_ONLY_TRAINABLE_PATTERNS):
            _set_parameter_trainable(param, True)
            matched_names.append(name)

    trainable_numel = sum(
        _safe_parameter_numel(param) for _name, param in model.named_parameters() if getattr(param, "requires_grad", False)
    )
    total_numel = sum(_safe_parameter_numel(param) for _name, param in model.named_parameters())
    if not matched_names:
        raise RuntimeError("picf_trainable_scope=anchor_only matched no parameters; aborting to avoid a no-op run.")
    if trainable_numel <= 0:
        # Lazy params can report zero before materialization in some tests, but
        # the real train path materializes first. Treat a fully zero set as a
        # hard diagnostic failure in live training.
        raise RuntimeError("picf_trainable_scope=anchor_only produced zero initialized trainable parameters.")
    info = {
        "scope": scope,
        "trainable_param_tensors": len(matched_names),
        "trainable_numel": int(trainable_numel),
        "total_numel": int(total_numel),
        "matched_names_sample": matched_names[:24],
    }
    if logger is not None:
        frozen_numel = int(total_numel) - int(trainable_numel)
        logger.info(
            "Trainable scope: scope=%s trainable_tensors=%s trainable_numel=%s frozen_numel=%s total_numel=%s patterns=%s",
            scope,
            len(matched_names),
            int(trainable_numel),
            frozen_numel,
            int(total_numel),
            ",".join(_ANCHOR_ONLY_TRAINABLE_PATTERNS),
        )
        logger.info("Trainable scope sample: %s", ", ".join(matched_names[:24]))
    return info


def _freeze_recycle_path_parameters(model: torch.nn.Module) -> dict[str, Any]:
    matched_names: list[str] = []
    for name, param in model.named_parameters():
        if _matches_any_pattern(name, _RECYCLE_PATH_TRAINABLE_PATTERNS):
            _set_parameter_trainable(param, False)
            matched_names.append(name)
    return {
        "frozen_recycle_param_tensors": len(matched_names),
        "frozen_recycle_names_sample": matched_names[:24],
    }


def _split_optimizer_groups_by_dense_type(groups: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """Partition param groups by tensor type for ZeroRedundancyOptimizer.

    PyTorch ZeroRedundancyOptimizer rejects a single optimizer containing both
    CUDA float32 and CUDA bfloat16 parameters.  Splitting by `param.type()`
    preserves each group's lr scale while keeping every ZRO instance
    homogeneous.
    """
    partitions: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for group in groups:
        params_by_type: dict[str, list[torch.nn.Parameter]] = {}
        type_order: list[str] = []
        for param in group["params"]:
            if isinstance(param, UninitializedParameter):
                # Lazy modules that remain dormant after warmup should not
                # participate in ZeRO dtype partitioning. They already report
                # zero params in optimizer metadata and excluding them keeps
                # the sharded optimizer path aligned with the non-sharded path.
                continue
            tensor_type = str(param.type())
            if tensor_type not in params_by_type:
                params_by_type[tensor_type] = []
                type_order.append(tensor_type)
            params_by_type[tensor_type].append(param)
        for tensor_type in type_order:
            if tensor_type not in partitions:
                partitions[tensor_type] = []
                order.append(tensor_type)
            split_group = {key: value for key, value in group.items() if key != "params"}
            split_group["params"] = params_by_type[tensor_type]
            split_group["dense_type"] = tensor_type
            partitions[tensor_type].append(split_group)
    return [(tensor_type, partitions[tensor_type]) for tensor_type in order]


def _canonical_param_owner_name(name: str) -> str:
    """Normalize wrapper prefixes before assigning optimizer groups.

    FSDP exposes root parameters under ``_fsdp_wrapped_module.*`` and DDP uses
    ``module.*``.  Optimizer grouping is semantic, not wrapper-specific: a
    wrapped ``core.foo`` parameter must still be assigned to the slow
    ``picf_core`` group rather than falling through to ``policy_head``.
    """

    parts = str(name).split(".")
    while parts and parts[0] in {"module", "_fsdp_wrapped_module"}:
        parts = parts[1:]
    return ".".join(parts)


def _build_optimizer(
    model: _PicfWindowTrainer,
    *,
    args: argparse.Namespace,
) -> tuple[torch.optim.Optimizer, list[dict[str, Any]]]:
    groups: list[dict[str, Any]] = []
    used_ids: set[int] = set()

    def _safe_numel(param: torch.nn.Parameter) -> int:
        if isinstance(param, UninitializedParameter):
            return 0
        return int(param.numel())

    def _append_group(name: str, params: list[torch.nn.Parameter], lr_scale: float) -> None:
        if not params:
            return
        unique = [param for param in params if id(param) not in used_ids]
        if not unique:
            return
        used_ids.update(id(param) for param in unique)
        groups.append(
            {
                "name": name,
                "params": unique,
                "lr": float(args.lr) * float(lr_scale),
                "lr_scale": float(lr_scale),
            }
        )

    point_params = _collect_trainable_params(model.core.point_feature_extractor if isinstance(model.core.point_feature_extractor, torch.nn.Module) else None)
    visual_params = _collect_trainable_params(model.core.visual_encoder if isinstance(model.core.visual_encoder, torch.nn.Module) else None)
    tactile_params = _collect_trainable_params(model.core.tactile_encoder if isinstance(model.core.tactile_encoder, torch.nn.Module) else None)
    semantic_params = _collect_trainable_params(model.semantic_encoder)

    _append_group("point_backbone", point_params, args.point_backbone_lr_scale)
    _append_group("visual_backbone", visual_params, args.visual_lr_scale)
    _append_group("tactile_backbone", tactile_params, args.tactile_lr_scale)
    _append_group("semantic_backbone", semantic_params, args.semantic_lr_scale)

    remaining_named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if getattr(param, "requires_grad", False) and id(param) not in used_ids
    ]
    core_params = [param for name, param in remaining_named_params if _canonical_param_owner_name(name).startswith("core.")]
    policy_head_params = [
        param for name, param in remaining_named_params if not _canonical_param_owner_name(name).startswith("core.")
    ]
    _append_group("picf_core", core_params, args.picf_core_lr_scale)
    _append_group("policy_head", policy_head_params, args.policy_head_lr_scale)
    if not groups:
        raise RuntimeError("No trainable parameters found for optimizer; check backbone and semantic trainability settings.")
    optimizer_sharding = str(getattr(args, "optimizer_sharding", "none")).lower()
    if optimizer_sharding == "zero1":
        if not (dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1):
            raise RuntimeError("optimizer_sharding=zero1 requires an initialized multi-rank distributed process group.")
        if ZeroRedundancyOptimizer is None:
            raise RuntimeError("optimizer_sharding=zero1 requires torch.distributed.optim.ZeroRedundancyOptimizer.")
        partitioned_groups = _split_optimizer_groups_by_dense_type(groups)
        zero_optimizers = [
            ZeroRedundancyOptimizer(
                dtype_groups,
                optimizer_class=torch.optim.AdamW,
                lr=args.lr,
                betas=(0.9, 0.95),
                weight_decay=args.weight_decay,
            )
            for _dense_type, dtype_groups in partitioned_groups
        ]
        optimizer = zero_optimizers[0] if len(zero_optimizers) == 1 else _OptimizerCollection(zero_optimizers)
    else:
        optimizer = torch.optim.AdamW(
            groups,
            lr=args.lr,
            betas=(0.9, 0.95),
            weight_decay=args.weight_decay,
        )
    reported_groups = optimizer.param_groups if optimizer_sharding == "zero1" else groups
    group_info = [
        {
            "name": group.get("name", f"group_{idx}"),
            "lr": float(group["lr"]),
            "lr_scale": float(group.get("lr_scale", 1.0)),
            "num_params": int(sum(_safe_numel(param) for param in group["params"])),
            "optimizer_sharding": optimizer_sharding,
            "dense_type": group.get("dense_type"),
        }
        for idx, group in enumerate(reported_groups)
    ]
    return optimizer, group_info


def _init_wandb(*, args: argparse.Namespace, output_dir: Path, resuming: bool, enabled: bool) -> bool:
    if not enabled:
        return False
    if wandb is None:
        raise RuntimeError("wandb is not installed, but wandb logging is enabled.")
    run_name = args.wandb_run_name or args.exp_name
    wandb_id_path = output_dir / "wandb_id.txt"
    init_kwargs = {
        "project": args.project_name,
        "name": run_name,
        "mode": args.wandb_mode,
        "config": vars(args),
    }
    if resuming and wandb_id_path.exists():
        run_id = wandb_id_path.read_text(encoding="utf-8").strip()
        init_kwargs.update({"id": run_id, "resume": "must"})
        wandb.init(**init_kwargs)
        return True
    wandb.init(**init_kwargs)
    if getattr(wandb, "run", None) is not None:
        wandb_id_path.write_text(str(wandb.run.id), encoding="utf-8")
    return True


def train(args: argparse.Namespace) -> None:
    alloc_conf, alloc_conf_source = _configure_cuda_allocator_env(
        requested_device=args.device,
        training_strategy=str(getattr(args, "training_strategy", "ddp")),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
    )
    use_ddp, rank, world_size, device, runtime_env = _setup_distributed(args.device)
    runtime_env = dataclasses.replace(
        runtime_env,
        pytorch_cuda_alloc_conf=alloc_conf,
        pytorch_cuda_alloc_conf_source=alloc_conf_source,
    )
    wandb_active = False
    is_main = False
    source: _CalvinTransitionSource | None = None
    action_normalizer: PicfActionNormalizer | None = None
    pbar: Any = None
    fault_dump_handle: TextIO | None = None
    fault_dump_path: Path | None = None
    try:
        _seed_everything(args.seed, rank)
        is_main = _is_main(rank)
        if is_main:
            _warn_single_gpu_foundation_accum_risk(args, world_size=world_size, logger=logging.getLogger())
            logging.info(
                "Distributed runtime env: world_size=%s torch_distributed_debug=%s source=%s allow_detail=%s "
                "torch_nccl_enable_monitoring=%s torch_nccl_heartbeat_timeout_sec=%s "
                "pytorch_cuda_alloc_conf=%s alloc_source=%s",
                world_size,
                runtime_env.torch_distributed_debug or None,
                runtime_env.torch_distributed_debug_source,
                runtime_env.allow_torch_distributed_debug_detail,
                runtime_env.torch_nccl_enable_monitoring,
                runtime_env.torch_nccl_heartbeat_timeout_sec,
                runtime_env.pytorch_cuda_alloc_conf,
                runtime_env.pytorch_cuda_alloc_conf_source,
            )
        output_dir = Path(args.checkpoint_base_dir) / "picf_core" / args.exp_name
        latest_path = output_dir / "latest.pt"
        metrics_path = output_dir / "metrics.jsonl"
        window_trace_path = output_dir / f"window_trace_rank{rank}.jsonl"
        _prepare_output_dir(output_dir=output_dir, args=args, is_main=is_main, use_ddp=use_ddp, device=device)
        fault_dump_path = output_dir / f"stackdump_rank{rank}.log"
        try:
            fault_dump_handle = fault_dump_path.open("a", encoding="utf-8")
        except OSError:
            fault_dump_handle = None
        fault_dump_registered = _register_fault_dump_handler(stream=fault_dump_handle)
        action_normalizer = _resolve_action_normalizer(args)
        calvin_segment_indices = None
        if args.calvin_segment_indices:
            calvin_segment_indices = [int(part) for part in str(args.calvin_segment_indices).split(",") if part.strip()]
        source = _CalvinTransitionSource(
            args.calvin_root,
            split=args.split,
            backend=args.backend,
            unroll_steps=args.effective_unroll_steps,
            action_horizon=args.action_horizon,
            use_tactile=bool(args.use_tactile),
            tactile_sensor_names=args.tactile_sensor_names,
            tactile_sensor_offsets_m=args.tactile_sensor_offsets_m,
            tactile_calibration=args.tactile_calibration_path,
            tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(args.tactile_backgrounds_path),
            use_scene_obs=bool(args.use_scene_obs),
            load_tracklet_fields=bool(getattr(args, "tracklet_memory_enabled", _SPEC_DEFAULTS.tracklet_memory_enabled)),
            load_proposal_fields=bool(getattr(args, "proposal_memory_enabled", _SPEC_DEFAULTS.proposal_memory_enabled)),
            mvtrack_sidecar_root=args.mvtrack_sidecar_root,
            mvtrack_sidecar_proposal_nearest_max_gap=int(
                getattr(args, "mvtrack_sidecar_proposal_nearest_max_gap", 0)
            ),
            action_normalizer=action_normalizer,
            augmentation_mode=args.picf_augmentation_mode,
            photometric_strength=args.picf_photometric_strength,
            segment_indices=calvin_segment_indices,
            bucket_sampling_mode=getattr(args, "calvin_bucket_sampling_mode", "round_robin"),
            bucket_temperature_alpha=float(getattr(args, "calvin_bucket_temperature_alpha", 0.0)),
            bucket_weight_spec=getattr(args, "calvin_bucket_weight_spec", ""),
            bucket_sample_without_replacement=bool(
                getattr(args, "calvin_bucket_sample_without_replacement", True)
            ),
        )
        expected_logical_task_count = int(world_size) * int(args.accum_steps)
        requested_logical_task_count = int(getattr(args, "logical_batch_task_count", 0))
        logical_batch_normalization = bool(getattr(args, "logical_batch_bucket_normalization", False))
        logical_batch_logging = bool(getattr(args, "logical_batch_log_bucket_metrics", False))
        logical_action_bucket_ema_normalization = bool(
            getattr(args, "logical_batch_action_bucket_ema_normalization", False)
        )
        logical_action_bucket_ema_decay = float(getattr(args, "logical_batch_action_bucket_ema_decay", 0.98))
        logical_action_bucket_scale_min = float(getattr(args, "logical_batch_action_bucket_scale_min", 0.5))
        logical_action_bucket_scale_max = float(getattr(args, "logical_batch_action_bucket_scale_max", 1.5))
        logical_action_bucket_min_count = int(getattr(args, "logical_batch_action_bucket_min_count", 2))
        logical_action_bucket_loss_ema: dict[str, float] = {}
        logical_action_bucket_loss_ema_counts: dict[str, int] = {}
        logical_dynamic_mixing_enabled = bool(getattr(args, "logical_batch_dynamic_mixing", False))
        logical_dynamic_mixing_base_weights = dict(source.bucket_target_weights)
        logical_dynamic_mixing_loss_ema: dict[str, float] = {}
        logical_dynamic_mixing_loss_ema_previous: dict[str, float] = {}
        logical_dynamic_mixing_loss_ema_counts: dict[str, int] = {}
        logical_dynamic_mixing_decay = float(getattr(args, "logical_batch_dynamic_mixing_decay", 0.95))
        logical_dynamic_mixing_warmup_steps = int(getattr(args, "logical_batch_dynamic_mixing_warmup_steps", 50))
        logical_dynamic_mixing_min_count = int(getattr(args, "logical_batch_dynamic_mixing_min_count", 2))
        logical_dynamic_mixing_eta = float(getattr(args, "logical_batch_dynamic_mixing_eta", 0.25))
        logical_dynamic_mixing_gamma = float(getattr(args, "logical_batch_dynamic_mixing_gamma", 0.5))
        logical_dynamic_mixing_clip = float(getattr(args, "logical_batch_dynamic_mixing_clip", 2.0))
        logical_dynamic_mixing_min_mass_fraction = float(
            getattr(args, "logical_batch_dynamic_mixing_min_mass_fraction", 0.05)
        )
        logical_dynamic_mixing_max_weight = float(getattr(args, "logical_batch_dynamic_mixing_max_weight", 0.35))
        logical_gradient_surgery_mode = str(
            getattr(args, "logical_batch_gradient_surgery", "off")
        ).lower().replace("-", "_")
        logical_gradient_surgery_groups = [
            group.strip()
            for group in str(getattr(args, "logical_batch_gradient_surgery_groups", "semantic")).split(",")
            if group.strip()
        ]
        logical_gradient_surgery_eps = float(getattr(args, "logical_batch_gradient_surgery_eps", 1e-12))
        logical_gradient_surgery_cagrad_alpha = float(
            getattr(args, "logical_batch_gradient_surgery_cagrad_alpha", 0.4)
        )
        logical_gradient_surgery_cagrad_iters = int(
            getattr(args, "logical_batch_gradient_surgery_cagrad_iters", 20)
        )
        logical_gradient_surgery_cagrad_rescale = bool(
            getattr(args, "logical_batch_gradient_surgery_cagrad_rescale", True)
        )
        if requested_logical_task_count > 0 and requested_logical_task_count != expected_logical_task_count:
            raise ValueError(
                "--logical-batch-task-count must equal WORLD_SIZE * --accum-steps for this production "
                f"contract. Requested {requested_logical_task_count}, runtime expects {expected_logical_task_count}."
            )
        if (requested_logical_task_count > 0 or logical_batch_normalization) and not bool(
            getattr(args, "calvin_balanced_bucket_sampler", False)
        ):
            raise ValueError(
                "Logical-batch training requires --calvin-balanced-bucket-sampler so bucket metadata is controlled."
            )
        if (
            (
                str(getattr(args, "calvin_bucket_sampling_mode", "round_robin")).lower().replace("-", "_")
                != "round_robin"
                or str(getattr(args, "calvin_bucket_weight_spec", "") or "").strip()
            )
            and not bool(getattr(args, "calvin_balanced_bucket_sampler", False))
        ):
            raise ValueError(
                "CALVIN bucket sampling modes/weights require --calvin-balanced-bucket-sampler. "
                "Otherwise the dataloader is intentionally global random."
            )
        if logical_batch_normalization and not source.bucket_names:
            raise RuntimeError("Logical-batch bucket normalization requested, but CALVIN source has no task buckets.")
        if logical_action_bucket_ema_normalization and not logical_batch_normalization:
            raise ValueError(
                "--logical-batch-action-bucket-ema-normalization requires --logical-batch-bucket-normalization "
                "so action-component scaling is applied inside a controlled logical-batch estimator."
            )
        if logical_dynamic_mixing_enabled and not logical_batch_normalization:
            raise ValueError(
                "--logical-batch-dynamic-mixing requires --logical-batch-bucket-normalization so sampling q_b(t) "
                "and the logical-batch loss estimator use the same target distribution."
            )
        if is_main:
            logging.info(
                "Logical-batch config: requested_task_count=%s runtime_task_count=%s bucket_normalization=%s "
                "bucket_metrics=%s bucket_sampling_mode=%s bucket_temperature_alpha=%s bucket_weight_spec=%r "
                "bucket_sample_without_replacement=%s action_bucket_ema_normalization=%s "
                "action_bucket_ema_decay=%s action_bucket_scale_min=%s action_bucket_scale_max=%s "
                "action_bucket_min_count=%s dynamic_mixing=%s dynamic_mixing_decay=%s "
                "dynamic_mixing_warmup_steps=%s dynamic_mixing_min_count=%s dynamic_mixing_eta=%s "
                "dynamic_mixing_gamma=%s dynamic_mixing_clip=%s dynamic_mixing_min_mass_fraction=%s "
                "dynamic_mixing_max_weight=%s gradient_surgery=%s gradient_surgery_groups=%s "
                "bucket_names=%s bucket_segment_counts=%s bucket_target_weights=%s",
                requested_logical_task_count,
                expected_logical_task_count,
                logical_batch_normalization,
                logical_batch_logging,
                source.bucket_sampling_mode,
                source.bucket_temperature_alpha,
                source.bucket_weight_spec,
                source.bucket_sample_without_replacement,
                logical_action_bucket_ema_normalization,
                logical_action_bucket_ema_decay,
                logical_action_bucket_scale_min,
                logical_action_bucket_scale_max,
                logical_action_bucket_min_count,
                logical_dynamic_mixing_enabled,
                logical_dynamic_mixing_decay,
                logical_dynamic_mixing_warmup_steps,
                logical_dynamic_mixing_min_count,
                logical_dynamic_mixing_eta,
                logical_dynamic_mixing_gamma,
                logical_dynamic_mixing_clip,
                logical_dynamic_mixing_min_mass_fraction,
                logical_dynamic_mixing_max_weight,
                logical_gradient_surgery_mode,
                ",".join(logical_gradient_surgery_groups),
                ",".join(source.bucket_names),
                source.bucket_segment_counts,
                source.bucket_target_weights,
            )

        core, semantic_encoder, use_visual_override = _build_model_sequential_across_ranks(
            args,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        core = core.to(device)
        base_loss_config = _build_loss_config(args)
        model = _PicfWindowTrainer(
            core,
            semantic_encoder=semantic_encoder,
            visual_grid=args.visual_grid,
            use_visual_override=use_visual_override,
            loss_config=base_loss_config,
            picf_mode=args.picf_mode,
            burnin_steps=args.burnin_steps,
            burnin_mode=args.burnin_mode,
        ).to(device)
        _materialize_model_parameters(model, source=source, rank=rank)
        trainable_scope_info = _apply_picf_trainable_scope(
            model,
            args=args,
            logger=logging.getLogger() if is_main else None,
        )
        if bool(getattr(args, "freeze_recycle_path", False)):
            recycle_freeze_info = _freeze_recycle_path_parameters(model)
            trainable_scope_info.update(recycle_freeze_info)
            if is_main:
                logging.getLogger().info(
                    "Recycle path frozen: tensors=%s sample=%s",
                    recycle_freeze_info["frozen_recycle_param_tensors"],
                    ", ".join(recycle_freeze_info["frozen_recycle_names_sample"]),
                )
        model = _wrap_model_for_training_strategy(model, args=args, device=device)
        optimizer, optimizer_group_info = _build_optimizer(model, args=args)
        grad_clip_controller = _GradClipController.from_args(args)
        if use_ddp and not _is_fsdp_training(args):
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=bool(logical_gradient_surgery_mode == "off"),
                static_graph=False,
            )
        logical_gradient_surgery_params: list[torch.nn.Parameter] = []
        if logical_gradient_surgery_mode != "off":
            logical_gradient_surgery_params = _logical_batch_gradient_surgery_params(
                model,
                groups=logical_gradient_surgery_groups,
            )
            if not logical_gradient_surgery_params:
                raise RuntimeError(
                    "logical_batch_gradient_surgery matched no trainable parameters; "
                    f"groups={logical_gradient_surgery_groups!r}."
                )
        if is_main:
            logging.info(
                "Logical-batch gradient surgery: mode=%s groups=%s param_tensors=%s param_numel=%s",
                logical_gradient_surgery_mode,
                ",".join(logical_gradient_surgery_groups),
                len(logical_gradient_surgery_params),
                int(sum(0 if isinstance(param, UninitializedParameter) else param.numel() for param in logical_gradient_surgery_params)),
            )

        start_step = 0
        resume_path: Path | None = None
        if args.resume_checkpoint is not None:
            resume_path = Path(args.resume_checkpoint)
        elif args.resume:
            resume_path = _resolve_resume_path(output_dir=output_dir, latest_path=latest_path)
        if resume_path is not None:
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
            start_step = _load_checkpoint_sequential_across_ranks(
                path=resume_path,
                model=model,
                optimizer=optimizer,
                device=device,
                rank=rank,
                world_size=world_size,
                optimizer_checkpoint_mode=str(args.optimizer_checkpoint_mode),
                grad_clip_controller=grad_clip_controller,
            )
            logging.info("Resumed from %s at step=%s", resume_path, start_step)
        if is_main:
            logging.info(
                "Gradient clip config: mode=%s fixed_norm=%.4f percentile=%.2f window=%s history_size=%s",
                grad_clip_controller.mode,
                float(grad_clip_controller.fixed_norm),
                float(grad_clip_controller.percentile),
                int(grad_clip_controller.window),
                int(grad_clip_controller.history_size()),
            )

        if is_main:
            wandb_active = _init_wandb(
                args=args,
                output_dir=output_dir,
                resuming=resume_path is not None,
                enabled=bool(args.wandb_enabled and args.wandb_mode != "disabled"),
            )
        _distributed_barrier(use_ddp=use_ddp, device=device)

        rng = np.random.default_rng(args.seed + 17 * rank)
        metric_accum = _MetricAccumulator()
        bucket_metric_accum = _BucketMetricAccumulator()
        last_logical_batch_step_info: dict[str, Any] = {}
        interval_start = time.time()
        steps_in_interval = 0
        retried_windows_interval = 0
        grad_clip_threshold = grad_clip_controller.threshold()
        grad_clip_applied = False
        recent_windows: deque[dict[str, Any]] = deque(maxlen=4)
        window_trace_interval: list[dict[str, Any]] = []
        debug_cuda_sync = os.environ.get("OPENPI_DEBUG_CUDA_SYNC", "").strip() not in {"", "0", "false", "False"}
        debug_autograd_anomaly = os.environ.get("OPENPI_DEBUG_AUTOGRAD_ANOMALY", "").strip() not in {"", "0", "false", "False"}
        debug_tensor_index_guards = os.environ.get("OPENPI_DEBUG_TENSOR_INDEX_GUARDS", "").strip() not in {"", "0", "false", "False"}
        debug_phase_limit = int(os.environ.get("OPENPI_DEBUG_PHASE_LIMIT", "0") or "0")
        verbose_startup_logs = os.environ.get("OPENPI_VERBOSE_STARTUP_LOGS", "").strip() not in {"", "0", "false", "False"}
        if debug_autograd_anomaly:
            torch.autograd.set_detect_anomaly(True)
        if debug_tensor_index_guards:
            _install_debug_tensor_index_guards()
        pbar = (
            tqdm.tqdm(
                total=args.num_train_steps,
                initial=start_step,
                desc="PICF Training",
                dynamic_ncols=True,
                disable=not (is_main and args.progress),
            )
            if is_main
            else None
        )
        if is_main:
            effective_global_batch = int(world_size * args.accum_steps)
            warmup_fraction = 100.0 * float(args.warmup_steps) / float(max(args.num_train_steps, 1))
            logging.info(
                "Training config: world_size=%s training_strategy=%s picf_mode=%s trainable_scope=%s trainable_numel=%s total_numel=%s accum_steps=%s effective_global_batch=%s num_steps=%s lr=%s min_lr=%s warmup=%s save_interval=%s unroll_steps=%s burnin_steps=%s burnin_mode=%s effective_window_steps=%s optimizer_sharding=%s optimizer_checkpoint_mode=%s window_activation_checkpointing=%s fsdp_sync_each_accum_micro=%s step_indexed_window_rng=%s calvin_balanced_bucket_sampler=%s calvin_bucket_sampling_mode=%s calvin_bucket_temperature_alpha=%s calvin_bucket_weight_spec=%r calvin_bucket_sample_without_replacement=%s logical_batch_task_count=%s logical_batch_bucket_normalization=%s logical_batch_log_bucket_metrics=%s calvin_buckets=%s anchor_overlay_interval=%s anchor_overlay_max_anchors=%s anchor_overlay_dump_signatures=%s wandb=%s",
                world_size,
                args.training_strategy,
                args.picf_mode,
                trainable_scope_info["scope"],
                trainable_scope_info["trainable_numel"],
                trainable_scope_info["total_numel"],
                args.accum_steps,
                effective_global_batch,
                args.num_train_steps,
                args.lr,
                args.min_lr,
                args.warmup_steps,
                args.save_interval,
                args.unroll_steps,
                args.burnin_steps,
                args.burnin_mode,
                args.effective_unroll_steps,
                args.optimizer_sharding,
                args.optimizer_checkpoint_mode,
                bool(getattr(args, "window_activation_checkpointing", False)),
                bool(getattr(args, "fsdp_sync_each_accum_micro", False)),
                bool(getattr(args, "step_indexed_window_rng", True)),
                bool(getattr(args, "calvin_balanced_bucket_sampler", False)),
                str(getattr(args, "calvin_bucket_sampling_mode", "round_robin")),
                float(getattr(args, "calvin_bucket_temperature_alpha", 0.0)),
                str(getattr(args, "calvin_bucket_weight_spec", "")),
                bool(getattr(args, "calvin_bucket_sample_without_replacement", True)),
                int(getattr(args, "logical_batch_task_count", 0)),
                bool(getattr(args, "logical_batch_bucket_normalization", False)),
                bool(getattr(args, "logical_batch_log_bucket_metrics", False)),
                ",".join(getattr(source, "bucket_names", ())),
                int(getattr(args, "anchor_overlay_interval", 0)),
                int(getattr(args, "anchor_overlay_max_anchors", 64)),
                bool(getattr(args, "anchor_overlay_dump_signatures", False)),
                bool(args.wandb_enabled and args.wandb_mode != "disabled"),
            )
            logging.info(
                "Tactile geometry config: pt_bag_radius_m=%s (source=%s) pt_bag_sigma_m=%s (source=%s) p_align_off=%s p_align_on=%s tactile_anchor_prob_on=%s",
                args.pt_bag_radius_m,
                getattr(args, "pt_bag_radius_m_source", "unknown"),
                args.pt_bag_sigma_m,
                getattr(args, "pt_bag_sigma_m_source", "unknown"),
                args.p_align_off,
                args.p_align_on,
                args.tactile_anchor_prob_on,
            )
            logging.info(
                "Action contract: normalization=%s norm_stats=%s action_output_clip=%s",
                getattr(args, "action_normalization", "none"),
                getattr(args, "action_norm_stats_path", None),
                getattr(args, "action_output_clip", None),
            )
            logging.info(
                "PICF action interface: prefix_stopgrad=%s prefix_gate=%s prefix_teacher=%s prefix_trust=%s "
                "context_tokens=%s context_integration=%s context_stopgrad=%s context_norm=%s "
                "context_gate=%s context_include_queries=%s",
                bool(getattr(args, "action_prefix_stopgrad", _SPEC_DEFAULTS.action_prefix_stopgrad)),
                float(getattr(args, "action_prefix_output_gate", _SPEC_DEFAULTS.action_prefix_output_gate)),
                str(getattr(args, "action_prefix_teacher_mode", _SPEC_DEFAULTS.action_prefix_teacher_mode)),
                float(getattr(args, "lambda_action_prefix_trust", _SPEC_DEFAULTS.lambda_action_prefix_trust)),
                int(getattr(args, "action_context_tokens", _SPEC_DEFAULTS.action_context_tokens)),
                str(getattr(args, "action_context_integration", _SPEC_DEFAULTS.action_context_integration)),
                bool(getattr(args, "action_context_stopgrad", _SPEC_DEFAULTS.action_context_stopgrad)),
                str(getattr(args, "action_context_norm_mode", _SPEC_DEFAULTS.action_context_norm_mode)),
                float(getattr(args, "action_context_output_gate", _SPEC_DEFAULTS.action_context_output_gate)),
                bool(getattr(args, "action_context_include_query_tokens", _SPEC_DEFAULTS.action_context_include_query_tokens)),
            )
            logging.info(
                "Prompt-state contract: normalization=%s norm_stats=%s inject_state_into_prompt=%s",
                getattr(args, "prompt_state_normalization", "none"),
                getattr(args, "prompt_state_norm_stats_path", None),
                True,
            )
            logging.info(
                "PICF core config: hidden=%s posterior_hidden=%s latent=%s innovation=%s control=%s semantic=%s semantic_cross=%s future_hidden=%s persistent_anchors=%s observation_anchors=%s effector_persistent_anchors=%s effector_observation_anchors=%s posterior_slot_identity_std=%s task_slot_identity_std=%s posterior_bootstrap_from_observation=%s global_scene_point_cap=%s fusion_layers=%s posterior_layers=%s predictive_layers=%s control_layers=%s control_query_tokens=%s predictive_query_tokens=%s task_local_queries=%s task_effector_queries=%s task_global_queries=%s task_instruction_queries=%s task_self_layers=%s conditioned_control_queries=%s pi_prefix_queries=%s conditioned_future_queries=%s task_visual_reread_topk=%s task_tactile_reread_groups=%s task_point_reread_topk=%s visual_real_grid=%s visual_real_dim=%s require_pi0_action_generator=%s predictive_semantic_reads=%s control_semantic_reads=%s predictive_semantic_dropout_prob=%s semantic_prefix_dropout_prob=%s attention_heads=%s future_vote_heads=%s",
                args.hidden_dim,
                args.posterior_hidden_dim,
                args.latent_dim,
                args.innovation_dim,
                args.control_dim,
                args.semantic_dim,
                args.semantic_cross_dim,
                args.future_hidden_dim,
                args.persistent_anchors,
                args.observation_anchors,
                args.effector_persistent_anchors,
                args.effector_observation_anchors,
                getattr(args, "posterior_slot_identity_std", _SPEC_DEFAULTS.posterior_slot_identity_std),
                getattr(args, "task_slot_identity_std", _SPEC_DEFAULTS.task_slot_identity_std),
                getattr(
                    args,
                    "posterior_bootstrap_from_observation",
                    _SPEC_DEFAULTS.posterior_bootstrap_from_observation,
                ),
                args.global_scene_point_cap,
                args.fusion_layers,
                args.posterior_layers,
                args.predictive_layers,
                args.control_layers,
                args.control_query_tokens,
                args.predictive_query_tokens,
                args.task_local_queries,
                args.task_effector_queries,
                args.task_global_queries,
                args.task_instruction_queries,
                args.task_self_layers,
                args.conditioned_control_queries,
                args.pi_prefix_queries,
                args.conditioned_future_queries,
                args.task_visual_reread_topk,
                args.task_tactile_reread_groups,
                args.task_point_reread_topk,
                args.visual_real_grid,
                3 * int(args.visual_real_grid) * int(args.visual_real_grid),
                bool(args.require_pi0_action_generator),
                args.predictive_semantic_reads,
                args.control_semantic_reads,
                args.predictive_semantic_dropout_prob,
                args.semantic_prefix_dropout_prob,
                args.attention_heads,
                args.future_vote_heads,
            )
            logging.info(
                "v2.2 semantic/control contract: the full PaliGemma token sequence remains width=%s and stays native in PI0.5 semantic/action generation; core control/future no longer take raw semantic-prefix tokens directly, task readout builds semantic-conditioned current-step context from public_read_memory=[fused_tokens, visual_tokens] plus private dense rereads; point tokens are split into local effector and global scene pools for role-aware observation/task slots; conditioned control/future consume task context together with up-projected physical posterior/global_post/innovation/proprio/physical_pred tokens; semantic_cross_dim/predictive_semantic_reads/control_semantic_reads remain compatibility fields that do not restore direct raw-semantic injection into the core trunks.",
                args.semantic_dim,
            )
            logging.info(
                "Tokenwise exact-memory contract: core_ff_chunk_size=%s semantic_tokenwise_chunk_size=%s semantic_projection_chunk_size=%s semantic_mlp_chunk_size=%s",
                int(getattr(args, "tokenwise_ff_chunk_size", 0)),
                int(getattr(args, "semantic_tokenwise_chunk_size", 0)),
                int(getattr(args, "semantic_projection_chunk_size", 0)),
                int(getattr(args, "semantic_mlp_chunk_size", 0)),
            )
            logging.info(
                "VL-guided anchor router contract: enabled=%s view=%s anchor_modes=%s nms_radius_m=%s local_sigma_m=%s min_visible_mass=%s heatmap_temperature=%s obs_gate_init=%s task_gate_init=%s posterior_bind_gate_init=%s prior_bias_clip=%s",
                bool(getattr(args, "vl_anchor_router_enabled", False)),
                getattr(args, "vl_grounding_view", _SPEC_DEFAULTS.vl_grounding_view),
                int(getattr(args, "vl_anchor_modes", _SPEC_DEFAULTS.vl_anchor_modes)),
                float(getattr(args, "vl_anchor_nms_radius_m", _SPEC_DEFAULTS.vl_anchor_nms_radius_m)),
                float(getattr(args, "vl_anchor_local_sigma_m", _SPEC_DEFAULTS.vl_anchor_local_sigma_m)),
                float(getattr(args, "vl_min_visible_mass", _SPEC_DEFAULTS.vl_min_visible_mass)),
                float(getattr(args, "vl_heatmap_temperature", _SPEC_DEFAULTS.vl_heatmap_temperature)),
                float(getattr(args, "vl_obs_anchor_gate_init", _SPEC_DEFAULTS.vl_obs_anchor_gate_init)),
                float(getattr(args, "vl_task_point_gate_init", _SPEC_DEFAULTS.vl_task_point_gate_init)),
                float(getattr(args, "vl_posterior_bind_gate_init", _SPEC_DEFAULTS.vl_posterior_bind_gate_init)),
                float(getattr(args, "vl_prior_bias_clip", _SPEC_DEFAULTS.vl_prior_bias_clip)),
            )
            logging.info(
                "MAPG anchor prior graph contract: enabled=%s anchors=%s message_rounds=%s visual_sigma_patches=%s tactile_sigma_m=%s posterior_sigma_m=%s confidence_floor=%s assignment_sinkhorn_iters=%s assignment_temperature=%s assignment_quality_uniform_mix=%s mode_confidence_threshold=%s obs_gate_init=%s task_gate_init=%s posterior_gate_init=%s control_gate_init=%s obs_point_mix_floor=%s prior_bias_clip=%s losses(siglip=%s vicreg=%s cycle=%s masked=%s routing=%s support_div=%s geom_div=%s)",
                bool(getattr(args, "mapg_enabled", False)),
                int(getattr(args, "mapg_anchor_count", _SPEC_DEFAULTS.mapg_anchor_count)),
                int(getattr(args, "mapg_message_rounds", _SPEC_DEFAULTS.mapg_message_rounds)),
                float(getattr(args, "mapg_visual_sigma_patches", _SPEC_DEFAULTS.mapg_visual_sigma_patches)),
                float(getattr(args, "mapg_tactile_sigma_m", _SPEC_DEFAULTS.mapg_tactile_sigma_m)),
                float(getattr(args, "mapg_posterior_sigma_m", _SPEC_DEFAULTS.mapg_posterior_sigma_m)),
                float(getattr(args, "mapg_confidence_floor", _SPEC_DEFAULTS.mapg_confidence_floor)),
                int(getattr(args, "mapg_assignment_sinkhorn_iters", _SPEC_DEFAULTS.mapg_assignment_sinkhorn_iters)),
                float(getattr(args, "mapg_assignment_temperature", _SPEC_DEFAULTS.mapg_assignment_temperature)),
                float(getattr(args, "mapg_assignment_quality_uniform_mix", _SPEC_DEFAULTS.mapg_assignment_quality_uniform_mix)),
                float(getattr(args, "mapg_mode_confidence_threshold", _SPEC_DEFAULTS.mapg_mode_confidence_threshold)),
                float(getattr(args, "mapg_obs_gate_init", _SPEC_DEFAULTS.mapg_obs_gate_init)),
                float(getattr(args, "mapg_task_gate_init", _SPEC_DEFAULTS.mapg_task_gate_init)),
                float(getattr(args, "mapg_posterior_gate_init", _SPEC_DEFAULTS.mapg_posterior_gate_init)),
                float(getattr(args, "mapg_control_gate_init", _SPEC_DEFAULTS.mapg_control_gate_init)),
                float(getattr(args, "mapg_obs_point_mix_floor", _SPEC_DEFAULTS.mapg_obs_point_mix_floor)),
                float(getattr(args, "mapg_prior_bias_clip", _SPEC_DEFAULTS.mapg_prior_bias_clip)),
                float(getattr(args, "lambda_mapg_siglip", _LOSS_DEFAULTS.lambda_mapg_siglip)),
                float(getattr(args, "lambda_mapg_vicreg", _LOSS_DEFAULTS.lambda_mapg_vicreg)),
                float(getattr(args, "lambda_mapg_cycle", _LOSS_DEFAULTS.lambda_mapg_cycle)),
                float(getattr(args, "lambda_mapg_masked_modality", _LOSS_DEFAULTS.lambda_mapg_masked_modality)),
                float(getattr(args, "lambda_mapg_routing", _LOSS_DEFAULTS.lambda_mapg_routing)),
                float(getattr(args, "lambda_mapg_support_diversity", _LOSS_DEFAULTS.lambda_mapg_support_diversity)),
                float(getattr(args, "lambda_mapg_geometry_diversity", _LOSS_DEFAULTS.lambda_mapg_geometry_diversity)),
            )
            logging.info(
                "AQR-OWM direct-final graph contract: enabled=%s physical_queries=%s task_queries=%s query_rounds=%s sinkhorn_iters=%s sinkhorn_temperature=%s pg_grounding_enabled=%s pg_image_support_enabled=%s pg_image_support_weight=%s pg_entropy_threshold=%s pg_peak_threshold=%s pg_bias_weight=%s support_bias_clip=%s ownership_prior_enabled=%s ownership_prior_weight=%s ownership_point_prior_weight=%s ownership_point_prior_sigma_m=%s ownership_temporal_prior_weight=%s ownership_uniform_mix=%s same_role_support_competition_enabled=%s same_role_support_competition_weight=%s same_role_support_competition_iters=%s same_role_support_competition_physical_only=%s active_slot_filter_enabled=%s active_slot_min_per_role=%s active_slot_max_per_role=%s active_slot_min_confidence=%s active_slot_overlap_threshold=%s active_slot_relative_score_threshold=%s active_slot_geometry_duplicate_enabled=%s active_slot_geometry_duplicate_sigma_m=%s active_slot_geometry_duplicate_threshold=%s context_slot_enabled=%s context_slot_weight=%s context_slot_min_confidence=%s context_slot_min_score=%s context_slot_duplicate_overlap_threshold=%s context_slot_deduplicate_enabled=%s context_slot_max_per_role=%s context_slot_self_overlap_threshold=%s context_slot_self_support_overlap_enabled=%s context_slot_self_support_overlap_threshold=%s context_slot_active_support_overlap_enabled=%s context_slot_active_support_overlap_threshold=%s context_slot_quality_gate_enabled=%s control_graph_attention_bias_enabled=%s control_graph_token_scaling_enabled=%s control_graph_state_embedding_enabled=%s control_graph_bias_min=%s posterior_owner_active_gate_enabled=%s posterior_owner_active_min=%s posterior_owner_active_bias=%s posterior_file_competition_enabled=%s posterior_file_competition_min_per_role=%s posterior_file_competition_max_per_role=%s posterior_file_competition_min_support=%s posterior_file_competition_overlap_threshold=%s posterior_file_competition_geometry_duplicate_enabled=%s posterior_file_competition_geometry_sigma_m=%s posterior_file_competition_geometry_threshold=%s vjepa_temporal_mode=%s vjepa_temporal_tokens=%s vjepa_temporal_delta=%s evidence_cache_enabled=%s evidence_cache_len=%s evidence_cache_read_weight=%s evidence_cache_innovation_downweight=%s tracklet_memory_enabled=%s proposal_memory_enabled=%s posterior_occupancy_prior_enabled=%s posterior_occupancy_prior_weight=%s posterior_occupancy_prior_sigma_m=%s posterior_occupancy_prior_clip=%s observation_anchor_seed_point_mix=%s recycle_residual_norm_mode=%s posterior_slotwise_recycle_residual=%s legacy_local_refinement_opt_in=%s local_refinement_enabled=%s local_refinement_weight=%s local_refinement_binding_weight=%s slot_jepa_enabled=%s support_prediction_enabled=%s ordinal_relation_enabled=%s losses(slot_jepa=%s support_pred=%s bind=%s denoise=%s) obs_gate_init=%s task_gate_init=%s posterior_gate_init=%s control_gate_init=%s legacy_mapg_builder_enabled=%s vl_router_enabled=%s",
                bool(getattr(args, "aqr_mapg_enabled", False)),
                int(getattr(args, "aqr_query_count_physical", _SPEC_DEFAULTS.aqr_query_count_physical)),
                int(getattr(args, "aqr_query_count_task", _SPEC_DEFAULTS.aqr_query_count_task)),
                int(getattr(args, "aqr_query_rounds", _SPEC_DEFAULTS.aqr_query_rounds)),
                int(getattr(args, "aqr_sinkhorn_iters", _SPEC_DEFAULTS.aqr_sinkhorn_iters)),
                float(getattr(args, "aqr_sinkhorn_temperature", _SPEC_DEFAULTS.aqr_sinkhorn_temperature)),
                bool(getattr(args, "aqr_pg_grounding_enabled", _SPEC_DEFAULTS.aqr_pg_grounding_enabled)),
                bool(getattr(args, "aqr_pg_image_support_enabled", _SPEC_DEFAULTS.aqr_pg_image_support_enabled)),
                float(getattr(args, "aqr_pg_image_support_weight", _SPEC_DEFAULTS.aqr_pg_image_support_weight)),
                float(getattr(args, "aqr_pg_entropy_threshold", _SPEC_DEFAULTS.aqr_pg_entropy_threshold)),
                float(getattr(args, "aqr_pg_peak_threshold", _SPEC_DEFAULTS.aqr_pg_peak_threshold)),
                float(getattr(args, "aqr_pg_bias_weight", _SPEC_DEFAULTS.aqr_pg_bias_weight)),
                float(getattr(args, "aqr_support_bias_clip", _SPEC_DEFAULTS.aqr_support_bias_clip)),
                bool(getattr(args, "aqr_ownership_prior_enabled", _SPEC_DEFAULTS.aqr_ownership_prior_enabled)),
                float(getattr(args, "aqr_ownership_prior_weight", _SPEC_DEFAULTS.aqr_ownership_prior_weight)),
                float(getattr(args, "aqr_ownership_point_prior_weight", _SPEC_DEFAULTS.aqr_ownership_point_prior_weight)),
                float(getattr(args, "aqr_ownership_point_prior_sigma_m", _SPEC_DEFAULTS.aqr_ownership_point_prior_sigma_m)),
                float(
                    getattr(
                        args,
                        "aqr_ownership_temporal_prior_weight",
                        _SPEC_DEFAULTS.aqr_ownership_temporal_prior_weight,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_ownership_prior_uniform_mix",
                        _SPEC_DEFAULTS.aqr_ownership_prior_uniform_mix,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_same_role_support_competition_enabled",
                        _SPEC_DEFAULTS.aqr_same_role_support_competition_enabled,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_same_role_support_competition_weight",
                        _SPEC_DEFAULTS.aqr_same_role_support_competition_weight,
                    )
                ),
                int(
                    getattr(
                        args,
                        "aqr_same_role_support_competition_iters",
                        _SPEC_DEFAULTS.aqr_same_role_support_competition_iters,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_same_role_support_competition_physical_only",
                        _SPEC_DEFAULTS.aqr_same_role_support_competition_physical_only,
                    )
                ),
                bool(getattr(args, "aqr_active_slot_filter_enabled", _SPEC_DEFAULTS.aqr_active_slot_filter_enabled)),
                int(getattr(args, "aqr_active_slot_min_per_role", _SPEC_DEFAULTS.aqr_active_slot_min_per_role)),
                int(getattr(args, "aqr_active_slot_max_per_role", _SPEC_DEFAULTS.aqr_active_slot_max_per_role)),
                float(getattr(args, "aqr_active_slot_min_confidence", _SPEC_DEFAULTS.aqr_active_slot_min_confidence)),
                float(
                    getattr(
                        args,
                        "aqr_active_slot_overlap_threshold",
                        _SPEC_DEFAULTS.aqr_active_slot_overlap_threshold,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_active_slot_relative_score_threshold",
                        _SPEC_DEFAULTS.aqr_active_slot_relative_score_threshold,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_active_slot_geometry_duplicate_enabled",
                        _SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_enabled,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_active_slot_geometry_duplicate_sigma_m",
                        _SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_sigma_m,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_active_slot_geometry_duplicate_threshold",
                        _SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_threshold,
                    )
                ),
                bool(getattr(args, "aqr_context_slot_enabled", _SPEC_DEFAULTS.aqr_context_slot_enabled)),
                float(getattr(args, "aqr_context_slot_weight", _SPEC_DEFAULTS.aqr_context_slot_weight)),
                float(
                    getattr(
                        args,
                        "aqr_context_slot_min_confidence",
                        _SPEC_DEFAULTS.aqr_context_slot_min_confidence,
                    )
                ),
                float(getattr(args, "aqr_context_slot_min_score", _SPEC_DEFAULTS.aqr_context_slot_min_score)),
                float(
                    getattr(
                        args,
                        "aqr_context_slot_duplicate_overlap_threshold",
                        _SPEC_DEFAULTS.aqr_context_slot_duplicate_overlap_threshold,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_context_slot_deduplicate_enabled",
                        _SPEC_DEFAULTS.aqr_context_slot_deduplicate_enabled,
                    )
                ),
                int(getattr(args, "aqr_context_slot_max_per_role", _SPEC_DEFAULTS.aqr_context_slot_max_per_role)),
                float(
                    getattr(
                        args,
                        "aqr_context_slot_self_overlap_threshold",
                        _SPEC_DEFAULTS.aqr_context_slot_self_overlap_threshold,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_context_slot_self_support_overlap_enabled",
                        _SPEC_DEFAULTS.aqr_context_slot_self_support_overlap_enabled,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_context_slot_self_support_overlap_threshold",
                        _SPEC_DEFAULTS.aqr_context_slot_self_support_overlap_threshold,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_context_slot_active_support_overlap_enabled",
                        _SPEC_DEFAULTS.aqr_context_slot_active_support_overlap_enabled,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_context_slot_active_support_overlap_threshold",
                        _SPEC_DEFAULTS.aqr_context_slot_active_support_overlap_threshold,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_context_slot_quality_gate_enabled",
                        _SPEC_DEFAULTS.aqr_context_slot_quality_gate_enabled,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_control_graph_attention_bias_enabled",
                        _SPEC_DEFAULTS.aqr_control_graph_attention_bias_enabled,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_control_graph_token_scaling_enabled",
                        _SPEC_DEFAULTS.aqr_control_graph_token_scaling_enabled,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "aqr_control_graph_state_embedding_enabled",
                        _SPEC_DEFAULTS.aqr_control_graph_state_embedding_enabled,
                    )
                ),
                float(
                    getattr(
                        args,
                        "aqr_control_graph_bias_min",
                        _SPEC_DEFAULTS.aqr_control_graph_bias_min,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "posterior_owner_active_gate_enabled",
                        _SPEC_DEFAULTS.posterior_owner_active_gate_enabled,
                    )
                ),
                float(getattr(args, "posterior_owner_active_min", _SPEC_DEFAULTS.posterior_owner_active_min)),
                float(getattr(args, "posterior_owner_active_bias", _SPEC_DEFAULTS.posterior_owner_active_bias)),
                bool(getattr(args, "posterior_file_competition_enabled", _SPEC_DEFAULTS.posterior_file_competition_enabled)),
                int(getattr(args, "posterior_file_competition_min_per_role", _SPEC_DEFAULTS.posterior_file_competition_min_per_role)),
                int(getattr(args, "posterior_file_competition_max_per_role", _SPEC_DEFAULTS.posterior_file_competition_max_per_role)),
                float(getattr(args, "posterior_file_competition_min_support", _SPEC_DEFAULTS.posterior_file_competition_min_support)),
                float(
                    getattr(
                        args,
                        "posterior_file_competition_support_overlap_threshold",
                        _SPEC_DEFAULTS.posterior_file_competition_support_overlap_threshold,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "posterior_file_competition_geometry_duplicate_enabled",
                        _SPEC_DEFAULTS.posterior_file_competition_geometry_duplicate_enabled,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_file_competition_geometry_sigma_m",
                        _SPEC_DEFAULTS.posterior_file_competition_geometry_sigma_m,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_file_competition_geometry_threshold",
                        _SPEC_DEFAULTS.posterior_file_competition_geometry_threshold,
                    )
                ),
                str(getattr(args, "aqr_vjepa_temporal_mode", _SPEC_DEFAULTS.aqr_vjepa_temporal_mode)),
                int(getattr(args, "aqr_vjepa_temporal_tokens", _SPEC_DEFAULTS.aqr_vjepa_temporal_tokens)),
                bool(
                    getattr(
                        args,
                        "aqr_vjepa_temporal_include_delta",
                        _SPEC_DEFAULTS.aqr_vjepa_temporal_include_delta,
                    )
                ),
                bool(getattr(args, "evidence_cache_enabled", _SPEC_DEFAULTS.evidence_cache_enabled)),
                int(getattr(args, "evidence_cache_len", _SPEC_DEFAULTS.evidence_cache_len)),
                float(getattr(args, "evidence_cache_read_weight", _SPEC_DEFAULTS.evidence_cache_read_weight)),
                float(
                    getattr(
                        args,
                        "evidence_cache_innovation_downweight",
                        _SPEC_DEFAULTS.evidence_cache_innovation_downweight,
                    )
                ),
                bool(getattr(args, "tracklet_memory_enabled", _SPEC_DEFAULTS.tracklet_memory_enabled)),
                bool(getattr(args, "proposal_memory_enabled", _SPEC_DEFAULTS.proposal_memory_enabled)),
                bool(getattr(args, "posterior_occupancy_prior_enabled", _SPEC_DEFAULTS.posterior_occupancy_prior_enabled)),
                float(getattr(args, "posterior_occupancy_prior_weight", _SPEC_DEFAULTS.posterior_occupancy_prior_weight)),
                float(getattr(args, "posterior_occupancy_prior_sigma_m", _SPEC_DEFAULTS.posterior_occupancy_prior_sigma_m)),
                float(getattr(args, "posterior_occupancy_prior_clip", _SPEC_DEFAULTS.posterior_occupancy_prior_clip)),
                float(getattr(args, "observation_anchor_seed_point_mix", _SPEC_DEFAULTS.observation_anchor_seed_point_mix)),
                str(getattr(args, "recycle_residual_norm_mode", _SPEC_DEFAULTS.recycle_residual_norm_mode)),
                bool(
                    getattr(
                        args,
                        "posterior_slotwise_recycle_residual",
                        _SPEC_DEFAULTS.posterior_slotwise_recycle_residual,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "legacy_local_refinement_opt_in",
                        _SPEC_DEFAULTS.legacy_local_refinement_opt_in,
                    )
                ),
                bool(getattr(args, "local_refinement_enabled", _SPEC_DEFAULTS.local_refinement_enabled)),
                float(getattr(args, "local_refinement_weight", _SPEC_DEFAULTS.local_refinement_weight)),
                float(
                    getattr(
                        args,
                        "local_refinement_binding_weight",
                        _SPEC_DEFAULTS.local_refinement_binding_weight,
                    )
                ),
                bool(getattr(args, "slot_jepa_enabled", _SPEC_DEFAULTS.slot_jepa_enabled)),
                bool(getattr(args, "support_prediction_enabled", _SPEC_DEFAULTS.support_prediction_enabled)),
                bool(getattr(args, "ordinal_relation_enabled", _SPEC_DEFAULTS.ordinal_relation_enabled)),
                float(getattr(args, "lambda_slot_jepa", _LOSS_DEFAULTS.lambda_slot_jepa)),
                float(getattr(args, "lambda_support_pred", _LOSS_DEFAULTS.lambda_support_pred)),
                float(getattr(args, "lambda_binding_consistency", _LOSS_DEFAULTS.lambda_binding_consistency)),
                float(getattr(args, "lambda_aqr_denoising", _LOSS_DEFAULTS.lambda_aqr_denoising)),
                float(getattr(args, "aqr_obs_gate_init", _SPEC_DEFAULTS.aqr_obs_gate_init)),
                float(getattr(args, "aqr_task_gate_init", _SPEC_DEFAULTS.aqr_task_gate_init)),
                float(getattr(args, "aqr_posterior_gate_init", _SPEC_DEFAULTS.aqr_posterior_gate_init)),
                float(getattr(args, "aqr_control_gate_init", _SPEC_DEFAULTS.aqr_control_gate_init)),
                bool(getattr(args, "mapg_enabled", False)),
                bool(getattr(args, "vl_anchor_router_enabled", False)),
            )
            logging.info(
                "Posterior owner-transport contract: enabled=%s roles=%s max_per_role=%s max_rate=%s precision_gain=%s min_mass=%s direct_candidate_assignment=%s direct_candidate_min_score=%s assignment_floor=%s reliability_floor=%s covariance_scale=%s inactive_prior=%s activates_file=%s active_threshold=%s. "
                "Accepted object/contact responsibility is closed into posterior object-file geometry instead of staying only in the transient graph.",
                bool(getattr(args, "posterior_owner_transport_enabled", _SPEC_DEFAULTS.posterior_owner_transport_enabled)),
                str(getattr(args, "posterior_owner_transport_roles", _SPEC_DEFAULTS.posterior_owner_transport_roles)),
                int(
                    getattr(
                        args,
                        "posterior_owner_transport_max_per_role",
                        _SPEC_DEFAULTS.posterior_owner_transport_max_per_role,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_max_rate",
                        _SPEC_DEFAULTS.posterior_owner_transport_max_rate,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_precision_gain",
                        _SPEC_DEFAULTS.posterior_owner_transport_precision_gain,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_min_mass",
                        _SPEC_DEFAULTS.posterior_owner_transport_min_mass,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "posterior_owner_transport_direct_candidate_assignment",
                        _SPEC_DEFAULTS.posterior_owner_transport_direct_candidate_assignment,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_direct_candidate_min_score",
                        _SPEC_DEFAULTS.posterior_owner_transport_direct_candidate_min_score,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_assignment_floor",
                        _SPEC_DEFAULTS.posterior_owner_transport_assignment_floor,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_reliability_floor",
                        _SPEC_DEFAULTS.posterior_owner_transport_reliability_floor,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_covariance_scale",
                        _SPEC_DEFAULTS.posterior_owner_transport_covariance_scale,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_inactive_prior",
                        _SPEC_DEFAULTS.posterior_owner_transport_inactive_prior,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "posterior_owner_transport_activates_file",
                        _SPEC_DEFAULTS.posterior_owner_transport_activates_file,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_owner_transport_active_threshold",
                        _SPEC_DEFAULTS.posterior_owner_transport_active_threshold,
                    )
                ),
            )
            logging.info(
                "Posterior birth transport contract: enabled=%s max_per_role=%s min_score=%s inactive_only=%s alpha_suppression_power=%s. "
                "Dustbin residual can only feed bounded reserve births; it is not broadcast into every inactive posterior file.",
                bool(
                    getattr(
                        args,
                        "posterior_birth_competition_enabled",
                        _SPEC_DEFAULTS.posterior_birth_competition_enabled,
                    )
                ),
                int(
                    getattr(
                        args,
                        "posterior_birth_competition_max_per_role",
                        _SPEC_DEFAULTS.posterior_birth_competition_max_per_role,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_birth_competition_min_score",
                        _SPEC_DEFAULTS.posterior_birth_competition_min_score,
                    )
                ),
                bool(
                    getattr(
                        args,
                        "posterior_birth_competition_inactive_only",
                        _SPEC_DEFAULTS.posterior_birth_competition_inactive_only,
                    )
                ),
                float(
                    getattr(
                        args,
                        "posterior_birth_alpha_suppression_power",
                        _SPEC_DEFAULTS.posterior_birth_alpha_suppression_power,
                    )
                ),
            )
            logging.info(
                "Backbone contract: point=%s(trainable=%s flash_requested=%s) visual=%s(finetune_mode=%s trainable=%s) tactile=%s(trainable=%s) semantic=%s(trainable=%s scope=%s)",
                args.point_backbone,
                bool(args.point_backbone_trainable),
                bool(not args.sonata_disable_flash),
                args.visual_mode,
                args.visual_finetune_mode,
                bool(args.visual_trainable),
                args.tactile_mode,
                bool(args.tactile_trainable),
                args.semantic_mode,
                bool(args.semantic_trainable),
                str(getattr(args, "semantic_trainable_scope", "backbone_only")),
            )
            logging.info(
                "Action-flow objective contract: train_loss=%s huber_delta=%s time_beta=(%s,%s) "
                "canonical_mse_report=loss_action_default_equiv",
                str(getattr(args, "semantic_action_flow_loss", "mse")),
                float(getattr(args, "semantic_action_flow_huber_delta", 1.0)),
                float(getattr(args, "semantic_action_flow_time_alpha", 1.5)),
                float(getattr(args, "semantic_action_flow_time_beta", 1.0)),
            )
            logging.info(
                "Action-context readout auxiliary contract: weight=%s loss=%s huber_delta=%s. "
                "This trains PICF context motor-readability and leaves loss_action_default_equiv as canonical flow MSE.",
                float(getattr(args, "semantic_action_context_readout_aux_weight", 0.0)),
                str(getattr(args, "semantic_action_context_readout_aux_loss", "smooth_l1")),
                float(getattr(args, "semantic_action_context_readout_aux_huber_delta", 1.0)),
            )
            logging.info(
                "Action-context token auxiliary contract: weight=%s bins=%s clip=%s. "
                "This is a PICF-local FAST-style action-token CE over discretized action chunks; "
                "it shares the context readout state and keeps loss_action_default_equiv canonical.",
                float(getattr(args, "semantic_action_context_token_aux_weight", 0.0)),
                int(getattr(args, "semantic_action_context_token_aux_bins", 256)),
                float(getattr(args, "semantic_action_context_token_aux_clip", 1.0)),
            )
            logging.info(
                "Action-context deployed flow residual contract: enabled=%s gate_init=%s time_floor=%s "
                "rms_cap=%s. This changes the native PI0.5 flow velocity in train and sampling.",
                bool(getattr(args, "semantic_action_context_flow_residual_enabled", False)),
                float(getattr(args, "semantic_action_context_flow_residual_gate_init", -2.0)),
                float(getattr(args, "semantic_action_context_flow_residual_time_floor", 0.05)),
                bool(getattr(args, "semantic_action_context_flow_residual_rms_cap", True)),
            )
            logging.info(
                "Action-expert router contract: enabled=%s experts=%s rank=%s gate_init=%s temperature=%s "
                "rms_cap=%s scope=%s",
                bool(getattr(args, "semantic_action_expert_router_enabled", False)),
                int(getattr(args, "semantic_action_expert_router_experts", 4)),
                int(getattr(args, "semantic_action_expert_router_rank", 64)),
                float(getattr(args, "semantic_action_expert_router_gate_init", -2.5)),
                float(getattr(args, "semantic_action_expert_router_temperature", 1.0)),
                bool(getattr(args, "semantic_action_expert_router_rms_cap", True)),
                str(getattr(args, "semantic_trainable_scope", "backbone_only")),
            )
            logging.info(
                "Frozen feature cache contract: vjepa_mode=%s vjepa_root=%s vjepa_temporal_slices=%s "
                "vjepa_storage_dtype=%s valid_only_when_visual_trainable_false=%s",
                str(getattr(args, "vjepa_feature_cache_mode", "off")),
                str(getattr(args, "vjepa_feature_cache_root", None)),
                int(getattr(args, "vjepa_feature_cache_temporal_slices", 4)),
                str(getattr(args, "vjepa_feature_cache_storage_dtype", "bfloat16")),
                bool(not args.visual_trainable),
            )
            logging.info(
                "MVTrack sidecar contract: mvtrack_sidecar_root=%s proposal_nearest_max_gap=%s proposal_age_decay_steps=%s; optional tracklet/proposal arrays are offline typed evidence and do not mutate CALVIN frames.",
                args.mvtrack_sidecar_root,
                int(getattr(args, "mvtrack_sidecar_proposal_nearest_max_gap", 0)),
                float(getattr(args, "proposal_age_decay_steps", _SPEC_DEFAULTS.proposal_age_decay_steps)),
            )
            logging.info(
                "Proposal reference-anchor seed contract: enabled=%s pre_reader=%s rows=%s weight=%s token_weight=%s score_floor=%s point_topk=%s point_power=%s; sidecar proposals condition physical measurement rows before point reread but do not bypass AQR/posterior.",
                bool(getattr(args, "proposal_anchor_seed_enabled", _SPEC_DEFAULTS.proposal_anchor_seed_enabled)),
                bool(
                    getattr(
                        args,
                        "proposal_anchor_seed_pre_reader_enabled",
                        _SPEC_DEFAULTS.proposal_anchor_seed_pre_reader_enabled,
                    )
                ),
                int(getattr(args, "proposal_anchor_seed_rows", _SPEC_DEFAULTS.proposal_anchor_seed_rows)),
                float(getattr(args, "proposal_anchor_seed_weight", _SPEC_DEFAULTS.proposal_anchor_seed_weight)),
                float(getattr(args, "proposal_anchor_seed_token_weight", _SPEC_DEFAULTS.proposal_anchor_seed_token_weight)),
                float(getattr(args, "proposal_anchor_seed_score_floor", _SPEC_DEFAULTS.proposal_anchor_seed_score_floor)),
                int(getattr(args, "proposal_anchor_seed_point_topk", _SPEC_DEFAULTS.proposal_anchor_seed_point_topk)),
                float(getattr(args, "proposal_anchor_seed_point_power", _SPEC_DEFAULTS.proposal_anchor_seed_point_power)),
            )
            logging.info(
                "Frozen-perception/augmentation contract: perception_finetune_mode=%s picf_augmentation_mode=%s photometric_strength=%s semantic_max_length=%s",
                args.perception_finetune_mode,
                args.picf_augmentation_mode,
                args.picf_photometric_strength,
                int(args.semantic_max_length),
            )
            if not _picf_mode_enabled(args):
                logging.info(
                    "PI0.5-only ablation contract: PICF recurrent/control/future branches are disabled; the trainer builds the native PI0.5 semantic action path, freezes unused PICF core parameters, and uses extra_prefix_tokens=None."
                )
            compact_startup_logging = bool(use_ddp and not verbose_startup_logs)
            logging.info("Startup logging: compact=%s", compact_startup_logging)
            if bool(getattr(args, "semantic_gradient_checkpointing_disabled_for_accum", False)):
                logging.info(
                    "Semantic contract: disabled PaliGemma gradient checkpointing because accum_steps=%s > 1; "
                    "this avoids DDP 'mark ready twice' failures during gradient accumulation.",
                    args.accum_steps,
                )
            if not compact_startup_logging:
                logging.info(
                    "Backbone runtime types: point=%s semantic=%s",
                    type(core.point_feature_extractor).__name__ if core.point_feature_extractor is not None else "none",
                    type(semantic_encoder).__name__ if semantic_encoder is not None else "none",
                )
                if _is_fsdp_training(args) and str(getattr(args, "semantic_mode", "zero")) == "paligemma":
                    logging.info(
                        "Semantic contract: FSDP keeps the PI0/PaliGemma stack at one shard boundary, "
                        "so non-reentrant semantic gradient checkpointing remains enabled when requested."
                    )
                    logging.info(
                        "FSDP memory contract: FULL_SHARD uses BACKWARD_POST prefetch and the PICF core "
                        "transformer stacks run train-time activation recompute to reduce backward peak memory."
                    )
                    logging.info(
                        "FSDP shard contract: large uniform-dtype subtrees recurse to a 512MiB per-boundary "
                        "budget and safe core stacks (token_fusion/obs_self/posterior_self/task_self/"
                        "predictive_world/predictive_semantic_world/control_world) shard before the root wrapper."
                    )
                logging.info(
                    "Semantic checkpointing request: enabled=%s non_reentrant=%s",
                    bool(args.semantic_gradient_checkpointing),
                    bool(getattr(args, "semantic_gradient_checkpointing_non_reentrant", False)),
                )
                logging.info(
                    "LR contract: cosine decay with warmup_steps=%s (%.2f%% of total steps).",
                    args.warmup_steps,
                    warmup_fraction,
                )
                logging.info(
                    "Window contract: first-step empty xyzrgb windows will be resampled up to %s times per micro-step.",
                    args.max_empty_window_retries,
                )
                logging.info(
                    "Window RNG contract: step_indexed=%s. Step-indexed mode keys sampling by "
                    "(seed, rank, global_step, micro_step, retry) so checkpoint resume does not replay "
                    "the early train-window stream.",
                    bool(args.step_indexed_window_rng),
                )
                logging.info("CUDA debug sync: enabled=%s", bool(debug_cuda_sync))
                logging.info("Autograd anomaly detection: enabled=%s", bool(debug_autograd_anomaly))
                logging.info("Tensor index guards: enabled=%s", bool(debug_tensor_index_guards))
                logging.info("Phase timing debug: phase_limit=%s", debug_phase_limit)
                logging.info("Fault dump handler (SIGUSR1): enabled=%s", bool(fault_dump_registered))
                if fault_dump_path is not None:
                    logging.info("Fault dump path: %s", fault_dump_path)
            for group in optimizer_group_info:
                logging.info(
                    "Optimizer group: name=%s lr=%s num_params=%s sharding=%s",
                    group["name"],
                    group["lr"],
                    group["num_params"],
                    group.get("optimizer_sharding", "none"),
                )

        for step in range(start_step, args.num_train_steps):
            debug_phase_enabled = debug_phase_limit > 0 and step < debug_phase_limit
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s step_begin", int(step + 1), rank)
            lr = _lr_for_step(
                step,
                base_lr=args.lr,
                warmup_steps=args.warmup_steps,
                min_lr=args.min_lr,
                total_steps=args.num_train_steps,
            )
            picf_core_runtime_lr_multiplier = _picf_core_runtime_lr_multiplier(args, step=step)
            _set_optimizer_lr(
                optimizer,
                lr,
                group_runtime_multipliers={"picf_core": picf_core_runtime_lr_multiplier},
            )
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s lr_set lr=%.8f", int(step + 1), rank, float(lr))
            optimizer.zero_grad(set_to_none=True)
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s zero_grad_done", int(step + 1), rank)
            current_bucket_target_weights = dict(source.bucket_target_weights)
            dynamic_mixing_step_info: dict[str, float | int | bool] = {}
            if logical_dynamic_mixing_enabled:
                current_bucket_target_weights, dynamic_mixing_step_info = _dynamic_bucket_sampling_weights(
                    bucket_names=source.bucket_names,
                    base_weights=logical_dynamic_mixing_base_weights,
                    loss_ema=logical_dynamic_mixing_loss_ema,
                    previous_loss_ema=logical_dynamic_mixing_loss_ema_previous,
                    counts=logical_dynamic_mixing_loss_ema_counts,
                    step=int(step),
                    warmup_steps=logical_dynamic_mixing_warmup_steps,
                    min_count=logical_dynamic_mixing_min_count,
                    eta=logical_dynamic_mixing_eta,
                    gamma=logical_dynamic_mixing_gamma,
                    clip=logical_dynamic_mixing_clip,
                    min_mass_fraction=logical_dynamic_mixing_min_mass_fraction,
                    max_weight=logical_dynamic_mixing_max_weight,
                )
                source.bucket_target_weights = dict(current_bucket_target_weights)
                logical_dynamic_mixing_loss_ema_previous = dict(logical_dynamic_mixing_loss_ema)
            trainer_module = _unwrap_training_model(model)
            scheduled_loss_config, object_scaffold_decay_scale = _scheduled_loss_config(
                base_loss_config,
                args=args,
                step=step,
            )
            trainer_module.loss_config = scheduled_loss_config
            capture_visual_diagnostics = bool(
                args.diagnostic_interval > 0 and ((step + 1) % args.diagnostic_interval == 0)
            )
            capture_anchor_overlay_step = bool(
                is_main
                and int(getattr(args, "anchor_overlay_interval", 0)) > 0
                and ((step + 1) % int(getattr(args, "anchor_overlay_interval", 0)) == 0)
            )
            if not trainer_module.policy.picf_enabled:
                capture_visual_diagnostics = False
                capture_anchor_overlay_step = False
            use_window_activation_checkpointing = bool(
                getattr(args, "window_activation_checkpointing", False)
                and not capture_visual_diagnostics
                and not capture_anchor_overlay_step
            )
            prepared_micro_windows: list[_PreparedTrainingMicroWindow] = []
            for micro_step in range(args.accum_steps):
                sample_start = time.perf_counter()
                retry_count = 0
                if debug_phase_enabled:
                    logging.info(
                        "phase step=%s micro=%s rank=%s sample_begin",
                        int(step + 1),
                        int(micro_step + 1),
                        rank,
                    )
                while True:
                    sampled_bucket = "unbucketed"
                    if bool(getattr(args, "calvin_balanced_bucket_sampler", False)):
                        flat_index, sampled_bucket, sample_rng = source.balanced_bucket_slot_index(
                            seed=int(args.seed),
                            rank=int(rank),
                            world_size=int(world_size),
                            step=int(step),
                            micro_step=int(micro_step),
                            accum_steps=int(args.accum_steps),
                            retry_count=int(retry_count),
                        )
                    else:
                        sample_rng = (
                            _step_indexed_window_rng(
                                seed=int(args.seed),
                                rank=int(rank),
                                step=int(step),
                                micro_step=int(micro_step),
                                retry_count=int(retry_count),
                            )
                            if bool(args.step_indexed_window_rng)
                            else rng
                        )
                        rng_start = time.perf_counter()
                        flat_index = int(sample_rng.integers(0, len(source)))
                        sampled_bucket = "random"
                    rng_start = time.perf_counter()
                    if debug_phase_enabled:
                        logging.info(
                            "phase step=%s micro=%s rank=%s rng_pick_sec=%.3f flat_index=%s bucket=%s",
                            int(step + 1),
                            int(micro_step + 1),
                            rank,
                            time.perf_counter() - rng_start,
                            flat_index,
                            sampled_bucket,
                        )
                    window_load_start = time.perf_counter()
                    window = source.window(flat_index, rng=sample_rng)
                    if debug_phase_enabled:
                        logging.info(
                            "phase step=%s micro=%s rank=%s window_load_sec=%.3f flat_index=%s segment=%s start_step=%s prompt=%r",
                            int(step + 1),
                            int(micro_step + 1),
                            rank,
                            time.perf_counter() - window_load_start,
                            flat_index,
                            int(window.segment_id),
                            int(window.start_step_id),
                            str(window.prompt),
                        )
                    try:
                        ensure_start = time.perf_counter()
                        if trainer_module.policy.picf_enabled:
                            window_point_counts = _ensure_window_has_valid_first_step_xyzrgb_support(trainer_module, window)
                        else:
                            window_point_counts = tuple()
                        if debug_phase_enabled:
                            logging.info(
                                "phase step=%s micro=%s rank=%s sample_validate_sec=%.3f ensure_sec=%.3f flat_index=%s segment=%s start_step=%s prompt=%r point_counts=%s",
                                int(step + 1),
                                int(micro_step + 1),
                                rank,
                                time.perf_counter() - sample_start,
                                time.perf_counter() - ensure_start,
                                flat_index,
                                int(window.segment_id),
                                int(window.start_step_id),
                                str(window.prompt),
                                tuple(int(count) for count in window_point_counts),
                            )
                        break
                    except RuntimeError as exc:
                        if not _is_retryable_first_step_error(exc):
                            raise
                        retry_count += 1
                        retried_windows_interval += 1
                        if retry_count > args.max_empty_window_retries:
                            raise RuntimeError(
                                "Exceeded max_empty_window_retries while resampling PICF first-step xyzrgb support. "
                                f"Last failing window segment={window.segment_id} start_step={window.start_step_id}."
                            ) from exc
                        if is_main and (retry_count == 1 or retry_count % 8 == 0):
                            logging.warning(
                                "Resampling window due to empty first-step xyzrgb support: segment=%s start_step=%s retry=%s/%s",
                                window.segment_id,
                                window.start_step_id,
                                retry_count,
                                args.max_empty_window_retries,
                        )
                        continue
                prepared_micro_windows.append(
                    _PreparedTrainingMicroWindow(
                        micro_step=int(micro_step),
                        flat_index=int(flat_index),
                        sampled_bucket=str(sampled_bucket),
                        retry_count=int(retry_count),
                        point_counts=tuple(int(count) for count in window_point_counts),
                        window=window,
                    )
                )

            logical_loss_scales, last_logical_batch_step_info = _logical_batch_loss_scales(
                [prepared.sampled_bucket for prepared in prepared_micro_windows],
                enabled=logical_batch_normalization,
                use_ddp=use_ddp,
                world_size=int(world_size),
                target_bucket_weights=current_bucket_target_weights,
            )
            if dynamic_mixing_step_info:
                last_logical_batch_step_info.update(dynamic_mixing_step_info)
            outputs: dict[str, torch.Tensor] = {}
            window = prepared_micro_windows[-1].window
            logical_action_bucket_scales_used: list[float] = []
            logical_action_bucket_scale_by_bucket: dict[str, list[float]] = {}
            logical_gradient_surgery_local_grads: list[list[torch.Tensor | None]] = []
            for prepared, logical_loss_scale in zip(prepared_micro_windows, logical_loss_scales, strict=True):
                micro_step = int(prepared.micro_step)
                flat_index = int(prepared.flat_index)
                sampled_bucket = str(prepared.sampled_bucket)
                retry_count = int(prepared.retry_count)
                window_point_counts = tuple(int(count) for count in prepared.point_counts)
                window = prepared.window
                capture_anchor_overlay = bool(
                    capture_anchor_overlay_step and micro_step == int(args.accum_steps) - 1
                )
                sync_context: Any = _training_model_no_sync(
                    model,
                    enabled=bool(
                        world_size > 1
                        and micro_step < args.accum_steps - 1
                        and not (
                            _is_fsdp_training(args)
                            and bool(getattr(args, "fsdp_sync_each_accum_micro", False))
                        )
                    ),
                )
                # Keep all ranks aligned before entering the DDP-wrapped forward.
                # The exact DDP probe is stable with an explicit post-preflight barrier,
                # while the unconstrained training loop can let one rank run far ahead
                # of the other during semantic prefix preparation.
                if debug_phase_enabled:
                    logging.info(
                        "phase step=%s micro=%s rank=%s preforward_barrier_enter flat_index=%s segment=%s start_step=%s prompt=%r",
                        int(step + 1),
                        int(micro_step + 1),
                        rank,
                        flat_index,
                        int(window.segment_id),
                        int(window.start_step_id),
                        str(window.prompt),
                    )
                _distributed_barrier(use_ddp=use_ddp, device=device)
                if debug_phase_enabled:
                    logging.info(
                        "phase step=%s micro=%s rank=%s preforward_barrier_exit flat_index=%s segment=%s start_step=%s prompt=%r",
                        int(step + 1),
                        int(micro_step + 1),
                        rank,
                        flat_index,
                        int(window.segment_id),
                        int(window.start_step_id),
                        str(window.prompt),
                    )
                with sync_context:
                    recent_windows.append(
                        {
                            "global_step": int(step + 1),
                            "micro_step": int(micro_step + 1),
                            "flat_index": int(flat_index),
                            "segment": int(window.segment_id),
                            "start_step": int(window.start_step_id),
                            "prompt": str(window.prompt),
                            "prompt_bucket": _calvin_prompt_bucket(str(window.prompt)),
                            "sampled_bucket": str(sampled_bucket),
                            "retry_count": int(retry_count),
                            "point_counts": tuple(int(count) for count in window_point_counts),
                            "logical_loss_scale": float(logical_loss_scale),
                        }
                    )
                    trace_entry = _window_trace_record(
                        window,
                        global_step=int(step + 1),
                        micro_step=int(micro_step + 1),
                        rank=rank,
                        flat_index=int(flat_index),
                        retry_count=int(retry_count),
                        point_counts=tuple(int(count) for count in window_point_counts),
                    )
                    trace_entry["sampled_bucket"] = str(sampled_bucket)
                    trace_entry["logical_loss_scale"] = float(logical_loss_scale)
                    window_trace_interval.append(trace_entry)
                    try:
                        forward_label = None
                        if debug_phase_enabled:
                            forward_label = (
                                f"phase step={int(step + 1)} micro={int(micro_step + 1)} rank={rank} "
                                f"segment={int(window.segment_id)} start={int(window.start_step_id)}"
                            )
                            logging.info("%s forward_begin", forward_label)
                        forward_start = time.perf_counter()
                        if use_window_activation_checkpointing and forward_label is None:
                            checkpoint_dummy = _checkpoint_dummy_input(model)

                            def _checkpoint_window_forward(_dummy: torch.Tensor) -> tuple[torch.Tensor, ...]:
                                outputs = model(
                                    window,
                                    capture_visual_diagnostics=False,
                                    capture_anchor_overlay=False,
                                    capture_anchor_overlay_signatures=False,
                                    debug_phase_label=None,
                                )
                                outputs = dict(outputs)
                                outputs["loss_total"] = outputs["loss_total"] + (
                                    _dummy.to(dtype=outputs["loss_total"].dtype) * 0
                                )
                                return _window_outputs_to_tensor_tuple(outputs)

                            outputs = _window_outputs_from_tensor_tuple(
                                torch.utils.checkpoint.checkpoint(
                                    _checkpoint_window_forward,
                                    checkpoint_dummy,
                                    use_reentrant=False,
                                    preserve_rng_state=True,
                                )
                            )
                        else:
                            outputs = model(
                                window,
                                capture_visual_diagnostics=capture_visual_diagnostics,
                                capture_anchor_overlay=capture_anchor_overlay,
                                capture_anchor_overlay_signatures=bool(
                                    getattr(args, "anchor_overlay_dump_signatures", False)
                                ),
                                debug_phase_label=forward_label,
                            )
                        if debug_phase_enabled:
                            logging.info("%s forward_sec=%.3f", forward_label, time.perf_counter() - forward_start)
                        if debug_cuda_sync and device.type == "cuda":
                            torch.cuda.synchronize(device=device)
                        backward_start = time.perf_counter()
                        loss_for_backward = outputs["loss_total"]
                        action_bucket_scale = 1.0
                        if (
                            logical_action_bucket_ema_normalization
                            and "loss_total_minus_action" in outputs
                            and "loss_total" in outputs
                        ):
                            action_bucket_scale = _logical_action_bucket_scale(
                                sampled_bucket,
                                ema=logical_action_bucket_loss_ema,
                                counts=logical_action_bucket_loss_ema_counts,
                                min_count=logical_action_bucket_min_count,
                                scale_min=logical_action_bucket_scale_min,
                                scale_max=logical_action_bucket_scale_max,
                            )
                            non_action_loss = outputs["loss_total_minus_action"]
                            action_component = outputs["loss_total"] - non_action_loss
                            loss_for_backward = non_action_loss + action_component * float(action_bucket_scale)
                            logical_action_bucket_scales_used.append(float(action_bucket_scale))
                            logical_action_bucket_scale_by_bucket.setdefault(str(sampled_bucket), []).append(
                                float(action_bucket_scale)
                            )
                        scaled_loss_for_backward = loss_for_backward * float(logical_loss_scale)
                        if logical_gradient_surgery_mode in {"pcgrad", "cagrad"}:
                            logical_gradient_surgery_local_grads.append(
                                _logical_batch_autograd_grads(
                                    scaled_loss_for_backward,
                                    logical_gradient_surgery_params,
                                )
                            )
                        scaled_loss_for_backward.backward()
                        if logical_action_bucket_ema_normalization and "loss_action_default_equiv" in outputs:
                            _update_logical_action_bucket_ema(
                                logical_action_bucket_loss_ema,
                                logical_action_bucket_loss_ema_counts,
                                bucket=sampled_bucket,
                                value=float(outputs["loss_action_default_equiv"].detach().item()),
                                decay=logical_action_bucket_ema_decay,
                                bucket_names=source.bucket_names,
                                use_ddp=use_ddp,
                                device=device,
                            )
                        if logical_dynamic_mixing_enabled and "loss_action_default_equiv" in outputs:
                            _update_logical_action_bucket_ema(
                                logical_dynamic_mixing_loss_ema,
                                logical_dynamic_mixing_loss_ema_counts,
                                bucket=sampled_bucket,
                                value=float(outputs["loss_action_default_equiv"].detach().item()),
                                decay=logical_dynamic_mixing_decay,
                                bucket_names=source.bucket_names,
                                use_ddp=use_ddp,
                                device=device,
                            )
                        if debug_phase_enabled:
                            logging.info("%s backward_sec=%.3f", forward_label, time.perf_counter() - backward_start)
                        if debug_cuda_sync and device.type == "cuda":
                            torch.cuda.synchronize(device=device)
                    except Exception:
                        logging.exception(
                            "PICF training window failure: rank=%s global_step=%s micro_step=%s flat_index=%s "
                            "segment=%s start_step=%s prompt=%r",
                            rank,
                            int(step + 1),
                            int(micro_step + 1),
                            flat_index,
                            window.segment_id,
                            window.start_step_id,
                            window.prompt,
                        )
                        logging.error("Recent window history before failure: %s", list(recent_windows))
                        if _DEBUG_INDEX_TRACE:
                            logging.error("Recent tensor index trace before failure: %s", _dump_debug_index_trace())
                        raise
                metric_accum.update_from_outputs(outputs)
                if logical_batch_logging:
                    bucket_metric_accum.update_from_outputs(sampled_bucket, outputs)

            if logical_gradient_surgery_mode in {"pcgrad", "cagrad"}:
                if logical_gradient_surgery_mode == "pcgrad":
                    projected_grads = _pcgrad_project_and_sum(
                        logical_gradient_surgery_local_grads,
                        eps=logical_gradient_surgery_eps,
                    )
                    logical_gradient_surgery_mode_id = 1
                else:
                    projected_grads = _cagrad_project_and_sum(
                        logical_gradient_surgery_local_grads,
                        alpha=logical_gradient_surgery_cagrad_alpha,
                        iters=logical_gradient_surgery_cagrad_iters,
                        eps=logical_gradient_surgery_eps,
                        rescale_to_raw_norm=logical_gradient_surgery_cagrad_rescale,
                    )
                    logical_gradient_surgery_mode_id = 2
                _assign_and_sync_gradient_surgery_grads(
                    logical_gradient_surgery_params,
                    projected_grads,
                    use_ddp=use_ddp,
                    world_size=int(world_size),
                )
                last_logical_batch_step_info.update(
                    {
                        "logical_batch_gradient_surgery_enabled": True,
                        "logical_batch_gradient_surgery_mode_id": int(logical_gradient_surgery_mode_id),
                        "logical_batch_gradient_surgery_local_micro_count": int(
                            len(logical_gradient_surgery_local_grads)
                        ),
                        "logical_batch_gradient_surgery_target_param_tensors": int(
                            len(logical_gradient_surgery_params)
                        ),
                        "logical_batch_gradient_surgery_cagrad_alpha": float(
                            logical_gradient_surgery_cagrad_alpha
                        ),
                        "logical_batch_gradient_surgery_cagrad_iters": int(
                            logical_gradient_surgery_cagrad_iters
                        ),
                        "logical_batch_gradient_surgery_cagrad_rescale": bool(
                            logical_gradient_surgery_cagrad_rescale
                        ),
                    }
                )

            if logical_action_bucket_ema_normalization:
                gathered_scale_by_bucket = _all_gather_python_object(
                    logical_action_bucket_scale_by_bucket,
                    use_ddp=use_ddp,
                    world_size=int(world_size),
                )
                global_action_bucket_scale_by_bucket: dict[str, list[float]] = {}
                for rank_record in gathered_scale_by_bucket:
                    if not isinstance(rank_record, Mapping):
                        continue
                    for bucket, values in rank_record.items():
                        bucket_key = str(bucket)
                        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
                            global_action_bucket_scale_by_bucket.setdefault(bucket_key, []).extend(
                                float(value) for value in values
                            )
                global_action_bucket_scales_used = [
                    float(value)
                    for values in global_action_bucket_scale_by_bucket.values()
                    for value in values
                ]
                if global_action_bucket_scales_used:
                    last_logical_batch_step_info.update(
                        {
                            "logical_batch_action_bucket_ema_normalization": True,
                            "logical_batch_action_bucket_scale_mean": float(
                                sum(global_action_bucket_scales_used)
                                / float(len(global_action_bucket_scales_used))
                            ),
                            "logical_batch_action_bucket_scale_min": float(
                                min(global_action_bucket_scales_used)
                            ),
                            "logical_batch_action_bucket_scale_max": float(
                                max(global_action_bucket_scales_used)
                            ),
                        }
                    )
                else:
                    last_logical_batch_step_info["logical_batch_action_bucket_ema_normalization"] = True
                for bucket, values in sorted(global_action_bucket_scale_by_bucket.items()):
                    fragment = _metric_key_fragment(bucket)
                    last_logical_batch_step_info[f"logical_batch_action_bucket_scale_{fragment}"] = float(
                        sum(values) / float(max(len(values), 1))
                    )
                for bucket in source.bucket_names:
                    fragment = _metric_key_fragment(bucket)
                    last_logical_batch_step_info[f"logical_batch_action_bucket_ema_{fragment}"] = float(
                        logical_action_bucket_loss_ema.get(str(bucket), 0.0)
                    )
                    last_logical_batch_step_info[f"logical_batch_action_bucket_ema_count_{fragment}"] = int(
                        logical_action_bucket_loss_ema_counts.get(str(bucket), 0)
                    )

            clip_start = time.perf_counter()
            grad_issue = _collect_nonfinite_gradient_diagnostics(model, optimizer=optimizer, max_items=24)
            if int(grad_issue["nonfinite_grad_count"]) > 0:
                logging.error(
                    "Non-finite gradient detected before grad clipping / optimizer step: %s",
                    grad_issue,
                )
                logging.error("Recent window history before non-finite gradient: %s", list(recent_windows))
                if _DEBUG_INDEX_TRACE:
                    logging.error("Recent tensor index trace before non-finite gradient: %s", _dump_debug_index_trace())
                raise RuntimeError("Non-finite gradients detected before optimizer step.")
            preclip_local_grad = _grad_norm_for_training_model(model)
            grad_clip_threshold = grad_clip_controller.threshold()
            grad_clip_applied = bool(
                grad_clip_threshold is not None and float(preclip_local_grad) > float(grad_clip_threshold)
            )
            if grad_clip_applied:
                _clip_grad_norm_for_training_model(model, max_norm=float(grad_clip_threshold))
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s grad_clip_sec=%.3f", int(step + 1), rank, time.perf_counter() - clip_start)
            opt_start = time.perf_counter()
            optimizer.step()
            param_issue = _collect_nonfinite_parameter_diagnostics(model, optimizer=optimizer, max_items=24)
            if int(param_issue["nonfinite_param_count"]) > 0:
                logging.error("Non-finite parameter detected immediately after optimizer.step: %s", param_issue)
                logging.error("Recent window history before non-finite parameter: %s", list(recent_windows))
                if _DEBUG_INDEX_TRACE:
                    logging.error("Recent tensor index trace before non-finite parameter: %s", _dump_debug_index_trace())
                raise RuntimeError("Non-finite parameters detected after optimizer step.")
            grad_clip_controller.observe(preclip_local_grad)
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s optimizer_step_sec=%.3f", int(step + 1), rank, time.perf_counter() - opt_start)
            steps_in_interval += 1
            current_total = float(outputs["loss_total"].detach().item())

            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix({"loss": f"{current_total:.4f}", "lr": f"{lr:.2e}", "step": int(step + 1)})

            should_log = ((step + 1) % args.log_interval == 0) or ((step + 1) == args.num_train_steps)
            if should_log:
                elapsed = max(time.time() - interval_start, 1e-6)
                local_grad = _postclip_grad_norm_for_logging(
                    preclip_grad_norm=preclip_local_grad,
                    grad_clip_threshold=grad_clip_threshold,
                    grad_clip_applied=grad_clip_applied,
                )
                averages = metric_accum.averages()
                retried_windows = float(retried_windows_interval)
                if world_size > 1:
                    averages = {k: _reduce_mean(v, device=device, world_size=world_size) for k, v in averages.items()}
                    preclip_local_grad = _reduce_mean(preclip_local_grad, device=device, world_size=world_size)
                    local_grad = _reduce_mean(local_grad, device=device, world_size=world_size)
                    retried_windows = _reduce_sum(retried_windows, device=device, world_size=world_size)
                logical_batch_record: dict[str, float | int | bool] = {
                    "logical_batch_enabled": bool(last_logical_batch_step_info.get("logical_batch_enabled", False)),
                    "logical_batch_global_micro_count": int(
                        last_logical_batch_step_info.get("logical_batch_global_micro_count", 0)
                    ),
                    "logical_batch_distinct_bucket_count": int(
                        last_logical_batch_step_info.get("logical_batch_distinct_bucket_count", 0)
                    ),
                }
                for bucket, count in dict(last_logical_batch_step_info.get("logical_batch_bucket_counts", {})).items():
                    logical_batch_record[f"logical_batch_selected_{_metric_key_fragment(str(bucket))}_count"] = int(count)
                for bucket, weight in dict(
                    last_logical_batch_step_info.get("logical_batch_bucket_target_weights", {})
                ).items():
                    logical_batch_record[f"logical_batch_target_{_metric_key_fragment(str(bucket))}_weight"] = float(weight)
                for key, value in last_logical_batch_step_info.items():
                    if not (
                        str(key).startswith("logical_batch_action_bucket_")
                        or str(key).startswith("logical_batch_gradient_surgery_")
                        or str(key).startswith("logical_batch_dynamic_")
                    ):
                        continue
                    if isinstance(value, bool):
                        logical_batch_record[str(key)] = bool(value)
                    elif isinstance(value, int):
                        logical_batch_record[str(key)] = int(value)
                    elif isinstance(value, float):
                        logical_batch_record[str(key)] = float(value)
                bucket_metric_record: dict[str, float] = {}
                if logical_batch_logging:
                    gathered_bucket_records = _all_gather_python_object(
                        bucket_metric_accum.records(),
                        use_ddp=use_ddp,
                        world_size=int(world_size),
                    )
                    merged_bucket_records = _merge_bucket_metric_records(gathered_bucket_records)
                    bucket_metric_record = _flatten_bucket_metric_records(merged_bucket_records)
                if is_main:
                    record = {
                        "step": int(step + 1),
                        "lr": float(lr),
                        "preclip_grad_norm": float(preclip_local_grad),
                        "grad_norm": float(local_grad),
                        "grad_clip_mode": str(grad_clip_controller.mode),
                        "grad_clip_threshold": 0.0 if grad_clip_threshold is None else float(grad_clip_threshold),
                        "grad_clip_threshold_ready": bool(grad_clip_threshold is not None),
                        "grad_clip_applied": bool(grad_clip_applied),
                        "grad_clip_history_size": int(grad_clip_controller.history_size()),
                        "steps_per_sec": float(steps_in_interval / elapsed),
                        "windows_per_sec": float(metric_accum.num_windows / elapsed),
                        "resampled_empty_first_step_windows": int(retried_windows),
                        "object_scaffold_decay_scale": float(object_scaffold_decay_scale),
                        "picf_core_lr_runtime_multiplier": float(picf_core_runtime_lr_multiplier),
                        "picf_core_lr_effective_scale": float(args.picf_core_lr_scale)
                        * float(picf_core_runtime_lr_multiplier),
                        **logical_batch_record,
                        **bucket_metric_record,
                        **_optimizer_group_grad_metrics(optimizer),
                        **averages,
                    }
                    line = json.dumps(record, sort_keys=True)
                    if pbar is not None:
                        pbar.write(line)
                    else:
                        print(line, flush=True)
                    with metrics_path.open("a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                    if wandb_active:
                        wandb.log(record, step=int(step + 1))
                if window_trace_interval:
                    trace_record = {
                        "step": int(step + 1),
                        "rank": int(rank),
                        "world_size": int(world_size),
                        "windows": list(window_trace_interval),
                    }
                    with window_trace_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(trace_record, sort_keys=True) + "\n")
                metric_accum = _MetricAccumulator()
                bucket_metric_accum = _BucketMetricAccumulator()
                window_trace_interval = []
                interval_start = time.time()
                steps_in_interval = 0
                retried_windows_interval = 0

            if is_main and capture_visual_diagnostics:
                if debug_phase_enabled:
                    logging.info("phase step=%s rank=%s visual_diagnostics_begin", int(step + 1), rank)
                _save_visual_diagnostics(
                    output_dir=output_dir,
                    step=step + 1,
                    window=window,
                    physical_visual_real_seq=list(outputs.get("diagnostic_physical_visual_real_seq", [])),
                    semantic_visual_real_seq=list(outputs.get("diagnostic_semantic_visual_real_seq", [])),
                    visual_real_grid=trainer_module.core.config.visual_real_grid,
                    visual_real_upscale=args.diagnostic_visual_upscale,
                )
                if debug_phase_enabled:
                    logging.info("phase step=%s rank=%s visual_diagnostics_done", int(step + 1), rank)

            if is_main and capture_anchor_overlay_step:
                if debug_phase_enabled:
                    logging.info("phase step=%s rank=%s anchor_overlay_begin", int(step + 1), rank)
                try:
                    _save_anchor_overlay_diagnostic(
                        output_dir=output_dir,
                        step=step + 1,
                        snapshot=outputs.get("diagnostic_anchor_overlay"),
                        core=trainer_module.core,
                        max_anchors=int(getattr(args, "anchor_overlay_max_anchors", 64)),
                    )
                except Exception:
                    logging.exception(
                        "Failed to write anchor overlay diagnostic at step=%s; continuing training.",
                        int(step + 1),
                    )
                if debug_phase_enabled:
                    logging.info("phase step=%s rank=%s anchor_overlay_done", int(step + 1), rank)

            should_save = ((step + 1) % args.save_interval == 0) or ((step + 1) == args.num_train_steps)
            if should_save:
                save_optimizer_state = _should_save_optimizer_state(args=args)
                if save_optimizer_state and not _is_fsdp_model(model):
                    _consolidate_optimizer_state_for_checkpoint(optimizer, rank=rank)
                _save_checkpoint(
                    output_dir=output_dir,
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,
                    args=args,
                    rank=rank,
                    device=device,
                    grad_clip_controller=grad_clip_controller,
                    save_optimizer_state=save_optimizer_state,
                )
                if is_main:
                    message = f"[picf_core_train] saved checkpoint step={step + 1} -> {_checkpoint_dir_for_step(output_dir, step + 1)}"
                    if pbar is not None:
                        pbar.write(message)
                    else:
                        print(message, flush=True)
                    pruned = _prune_old_checkpoints(
                        output_dir,
                        keep_last=int(getattr(args, "keep_last_checkpoints", 0)),
                    )
                    for path in pruned:
                        prune_message = f"[picf_core_train] pruned old checkpoint -> {path}"
                        if pbar is not None:
                            pbar.write(prune_message)
                        else:
                            print(prune_message, flush=True)
                    if wandb_active:
                        wandb.log({"checkpoint_step": int(step + 1)}, step=int(step + 1))
                _distributed_barrier(use_ddp=use_ddp, device=device)

    finally:
        if pbar is not None:
            pbar.close()
        if source is not None:
            source.close()
        if is_main and wandb_active:
            wandb.finish()
        if fault_dump_handle is not None:
            fault_dump_handle.close()
        _cleanup_distributed()


def main() -> None:
    _init_logging()
    parser = argparse.ArgumentParser(description="Long-run PICF core training on CALVIN transition windows.")
    parser.add_argument("--calvin-root", required=True)
    parser.add_argument("--split", default="training", choices=["training", "validation"])
    parser.add_argument("--backend", default="dir", choices=["dir", "zip"])
    parser.add_argument(
        "--calvin-segment-indices",
        default=None,
        help=(
            "Optional comma-separated CALVIN segment ids for diagnostics. "
            "Production training should leave this unset; inspected proposal sidecar diagnostics may restrict "
            "sampling to segments that have precomputed proposal coverage."
        ),
    )
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=30000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument(
        "--keep-last-checkpoints",
        type=int,
        default=0,
        help=(
            "Keep only the latest N numeric step checkpoint directories after each successful save. "
            "0 disables pruning. Non-step diagnostics and metadata are never pruned."
        ),
    )
    parser.add_argument("--diagnostic-interval", type=int, default=500)
    parser.add_argument("--diagnostic-visual-upscale", type=int, default=64)
    parser.add_argument(
        "--anchor-overlay-interval",
        type=int,
        default=0,
        help=(
            "Write static-camera anchor position overlays every N optimizer steps. "
            "0 disables the diagnostic. The overlay reuses the real training forward "
            "and saves PNG/JSON under anchor_overlays/ without changing losses."
        ),
    )
    parser.add_argument(
        "--anchor-overlay-max-anchors",
        type=int,
        default=64,
        help="Maximum graph/posterior anchors per source to draw in each anchor overlay image.",
    )
    parser.add_argument(
        "--anchor-overlay-dump-signatures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Also write support_signature and binding_signature vectors into anchor overlay JSON. "
            "This is intended for short IsSameObject diagnostics and is off by default to avoid large long-run JSON files."
        ),
    )
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument(
        "--fsdp-sync-each-accum-micro",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "For training_strategy=fsdp_full_shard with accum_steps>1, synchronize/reduce gradients on every "
            "micro-step instead of wrapping non-final micro-steps in FSDP no_sync().  This is slower because it "
            "communicates every micro-step, but it avoids retaining multiple unsynchronized FSDP gradient shards/full "
            "gradients in memory.  Use it for low-card E21-style K12 attempts such as 4 GPUs x accum=3."
        ),
    )
    parser.add_argument(
        "--calvin-balanced-bucket-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Sample CALVIN segment slots in deterministic coarse task buckets across rank/micro-steps. "
            "This approximates the balanced exact-window objective used by the action-readout causal probes "
            "without changing model inputs or adding losses."
        ),
    )
    parser.add_argument(
        "--calvin-bucket-sampling-mode",
        type=str,
        choices=sorted(_CALVIN_BUCKET_SAMPLING_MODES),
        default="round_robin",
        help=(
            "Target distribution for --calvin-balanced-bucket-sampler. "
            "round_robin keeps the historical deterministic bucket cycle; task_uniform samples task families uniformly; "
            "trajectory samples proportional to segment count; temperature samples proportional to N_b ** alpha."
        ),
    )
    parser.add_argument(
        "--calvin-bucket-temperature-alpha",
        type=float,
        default=0.0,
        help=(
            "Alpha for --calvin-bucket-sampling-mode=temperature. "
            "0 is task-uniform, 1 is trajectory-proportional."
        ),
    )
    parser.add_argument(
        "--calvin-bucket-weight-spec",
        type=str,
        default="",
        help=(
            "Optional explicit VLA-Foundry-style bucket ratio, e.g. "
            "'block_push=1,drawer=1,switch_button_light=1,*=0.5'. "
            "When set, every bucket must match an entry or '*=<weight>'."
        ),
    )
    parser.add_argument(
        "--calvin-bucket-sample-without-replacement",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For non-round-robin bucket modes, sample one shared bucket sequence per optimizer step without "
            "replacement before repeating.  This is the production logical-batch contract: with K<=num_buckets, "
            "one optimizer update cannot be dominated by duplicate task buckets.  Use --no-calvin-bucket-sample-without-replacement "
            "only to reproduce the older independent-with-replacement sampler."
        ),
    )
    parser.add_argument(
        "--logical-batch-task-count",
        type=int,
        default=0,
        help=(
            "Strict logical-batch contract size.  When >0 it must equal WORLD_SIZE * --accum-steps. "
            "This prevents accidentally running a supposed balanced run with the wrong optimizer-step coverage."
        ),
    )
    parser.add_argument(
        "--logical-batch-bucket-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Scale each micro-window loss by a task-bucket-balanced logical-batch estimator before backward. "
            "Requires --calvin-balanced-bucket-sampler."
        ),
    )
    parser.add_argument(
        "--logical-batch-log-bucket-metrics",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Emit flattened per-task-bucket loss metrics into metrics.jsonl. "
            "This is low overhead and intended for diagnosing multi-task gradient coverage."
        ),
    )
    parser.add_argument(
        "--logical-batch-action-bucket-ema-normalization",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Apply a bounded per-bucket EMA scale to only the action component during backward. "
            "Displayed action losses stay unscaled and comparable; this is a controlled F9b "
            "adapter-conflict diagnostic, not a metric change."
        ),
    )
    parser.add_argument("--logical-batch-action-bucket-ema-decay", type=float, default=0.98)
    parser.add_argument("--logical-batch-action-bucket-scale-min", type=float, default=0.5)
    parser.add_argument("--logical-batch-action-bucket-scale-max", type=float, default=1.5)
    parser.add_argument("--logical-batch-action-bucket-min-count", type=int, default=2)
    parser.add_argument(
        "--logical-batch-dynamic-mixing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable bounded PiKE-style dynamic task-bucket sampling.  The base distribution remains "
            "--calvin-bucket-sampling-mode/--calvin-bucket-weight-spec; dynamic mixing only adjusts q_b(t) "
            "from per-bucket action-loss EMA/progress and uses the same q_b(t) in logical-batch loss scaling."
        ),
    )
    parser.add_argument("--logical-batch-dynamic-mixing-decay", type=float, default=0.95)
    parser.add_argument("--logical-batch-dynamic-mixing-warmup-steps", type=int, default=50)
    parser.add_argument("--logical-batch-dynamic-mixing-min-count", type=int, default=2)
    parser.add_argument("--logical-batch-dynamic-mixing-eta", type=float, default=0.25)
    parser.add_argument("--logical-batch-dynamic-mixing-gamma", type=float, default=0.5)
    parser.add_argument("--logical-batch-dynamic-mixing-clip", type=float, default=2.0)
    parser.add_argument("--logical-batch-dynamic-mixing-min-mass-fraction", type=float, default=0.05)
    parser.add_argument("--logical-batch-dynamic-mixing-max-weight", type=float, default=0.35)
    parser.add_argument(
        "--logical-batch-gradient-surgery",
        choices=["off", "pcgrad", "cagrad"],
        default="off",
        help=(
            "Optional adapter-level gradient surgery after logical-batch backward. "
            "pcgrad collects per-micro gradients for the selected parameter groups, projects away negative "
            "task-bucket components, overwrites only those group gradients, and leaves all other gradients unchanged. "
            "cagrad uses a scoped conflict-averse simplex update over the same per-micro gradients."
        ),
    )
    parser.add_argument(
        "--logical-batch-gradient-surgery-groups",
        type=str,
        default="semantic",
        help=(
            "Comma-separated optimizer-owner groups for gradient surgery: semantic, policy_head, picf_core. "
            "The evidence-backed default branch is semantic only; do not include picf_core unless a probe "
            "shows strong structural-loss conflict there."
        ),
    )
    parser.add_argument("--logical-batch-gradient-surgery-eps", type=float, default=1e-12)
    parser.add_argument("--logical-batch-gradient-surgery-cagrad-alpha", type=float, default=0.4)
    parser.add_argument("--logical-batch-gradient-surgery-cagrad-iters", type=int, default=20)
    parser.add_argument(
        "--logical-batch-gradient-surgery-cagrad-rescale",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Rescale the scoped CAGrad update to the raw logical-batch gradient norm. "
            "This keeps adapter/action-head update magnitude comparable while changing conflict direction."
        ),
    )
    parser.add_argument("--max-empty-window-retries", type=int, default=32)
    parser.add_argument("--unroll-steps", type=int, default=2)
    parser.add_argument(
        "--burnin-steps",
        type=int,
        default=0,
        help=(
            "PICF-only no-grad recurrent warmup transitions before the trainable suffix. "
            "When this is >0, --unroll-steps remains the number of transitions that receive "
            "flow/transition losses and gradients; the effective sampled window length is "
            "burnin_steps + unroll_steps."
        ),
    )
    parser.add_argument(
        "--burnin-mode",
        choices=["full", "state_only"],
        default="full",
        help=(
            "Burn-in execution path. 'full' preserves the original full no-grad PICF policy forward. "
            "'state_only' advances only the recurrent carry needed by future physical posterior/innovation, "
            "skipping semantic task readout, conditioned control, PI0.5 flow loss, and conditioned future cache."
        ),
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=16,
        help=(
            "Dataset action horizon to request from CALVIN. "
            "This affects `PicfObservation.action_chunk` construction and valid window start indices."
        ),
    )
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--max-points", type=int, default=1024)
    parser.add_argument("--crop-radius-m", type=float, default=0.10)
    parser.add_argument("--point-focus-sigma-m", type=float, default=_SPEC_DEFAULTS.point_focus_sigma_m)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--picf-mode",
        choices=["enabled", "ablated"],
        default="enabled",
        help=(
            "'enabled' runs the full PICF v2.2 control/future contract; "
            "'ablated' disables PICF recurrent/control/future branches and trains only the native PI0.5 semantic action path."
        ),
    )
    parser.add_argument(
        "--picf-trainable-scope",
        choices=["all", "anchor_only", "policy_only"],
        default="backbone_only",
        help=(
            "High-level parameter trainability scope. 'all' preserves normal training. "
            "'anchor_only' is a diagnostic large-batch probe that freezes perception, semantic, "
            "PI0.5 action/control, and predictive heads while training only the PICF anchor router, "
            "observation-anchor adapters, posterior binding/address, and support/cache/local evidence path. "
            "'policy_only' freezes the whole PICF core while leaving the semantic/action policy path trainable; "
            "use it with structural aux losses disabled to test whether action rebound is caused by moving PICF prefixes."
        ),
    )
    parser.add_argument(
        "--freeze-recycle-path",
        action="store_true",
        help=(
            "Diagnostic only: freeze recycle_head and residual reset heads after applying the trainable scope, "
            "so action loss cannot use posterior recycle/reset as a shortcut."
        ),
    )
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--picf-core-lr-scale",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for trainable parameters under the PICF core namespace. "
            "Use values below 1.0 for two-timescale cotrain: action/semantic policy "
            "can adapt normally while the belief-router prefix moves slowly."
        ),
    )
    parser.add_argument(
        "--policy-head-lr-scale",
        type=float,
        default=1.0,
        help=(
            "LR multiplier for non-semantic, non-PICF trainable policy-head parameters. "
            "Most PI0.5 action-expert parameters live under the semantic encoder and "
            "therefore use --semantic-lr-scale; this group covers any remaining policy adapters."
        ),
    )
    parser.add_argument(
        "--training-strategy",
        choices=["ddp", "fsdp_full_shard"],
        default="ddp",
        help=(
            "Distributed training strategy. 'ddp' keeps full parameter replicas per rank; "
            "'fsdp_full_shard' shards trainable parameters, gradients, and optimizer state across ranks."
        ),
    )
    parser.add_argument(
        "--optimizer-sharding",
        choices=["none", "zero1"],
        default="none",
        help=(
            "Optimizer state sharding mode. 'zero1' uses PyTorch ZeroRedundancyOptimizer "
            "to shard AdamW moments across DDP ranks without freezing or changing model capacity."
        ),
    )
    parser.add_argument(
        "--optimizer-checkpoint-mode",
        choices=["auto", "full", "model-only"],
        default="auto",
        help=(
            "Optimizer checkpoint policy. 'auto' saves full optimizer state for ordinary AdamW "
            "and model-only checkpoints for zero1, avoiding expensive full-state consolidation."
        ),
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--grad-clip-mode", choices=["fixed", "percentile"], default="percentile")
    parser.add_argument("--grad-clip-percentile", type=float, default=75.0)
    parser.add_argument("--grad-clip-window", type=int, default=100)
    parser.add_argument("--action-normalization", choices=["none", "zscore", "quantile"], default="quantile")
    parser.add_argument("--action-norm-stats-path", default=None)
    parser.add_argument("--action-output-clip", type=float, default=None)
    parser.add_argument(
        "--prompt-state-normalization",
        choices=["inherit", "none", "zscore", "quantile"],
        default="inherit",
    )
    parser.add_argument("--prompt-state-norm-stats-path", default=None)
    parser.add_argument("--lambda-action-pos", type=float, default=_LOSS_DEFAULTS.lambda_action_pos)
    parser.add_argument("--lambda-action-rot", type=float, default=_LOSS_DEFAULTS.lambda_action_rot)
    parser.add_argument("--lambda-action-gripper", type=float, default=_LOSS_DEFAULTS.lambda_action_gripper)
    parser.add_argument("--lambda-visual-latent", type=float, default=_LOSS_DEFAULTS.lambda_visual_latent)
    parser.add_argument("--lambda-visual-real", type=float, default=_LOSS_DEFAULTS.lambda_visual_real)
    parser.add_argument("--lambda-tactile-real", type=float, default=_LOSS_DEFAULTS.lambda_tactile_real)
    parser.add_argument("--lambda-point-real", type=float, default=_LOSS_DEFAULTS.lambda_point_real)
    parser.add_argument("--lambda-semantic-future-aux", type=float, default=_LOSS_DEFAULTS.lambda_semantic_future_aux)
    parser.add_argument("--lambda-anchor-pv", type=float, default=_LOSS_DEFAULTS.lambda_anchor_pv)
    parser.add_argument("--lambda-anchor-object-pull", type=float, default=_LOSS_DEFAULTS.lambda_anchor_object_pull)
    parser.add_argument("--anchor-object-pull-sigma-m", type=float, default=_LOSS_DEFAULTS.anchor_object_pull_sigma_m)
    parser.add_argument(
        "--anchor-object-pull-confirmation-threshold",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_confirmation_threshold,
    )
    parser.add_argument(
        "--anchor-object-pull-allowed-roles",
        type=str,
        default=",".join(str(role) for role in _LOSS_DEFAULTS.anchor_object_pull_allowed_roles),
        help=(
            "Comma-separated anchor roles allowed to receive the diagnostic object-pull loss. "
            "Default 1 keeps object sidecar supervision on task-object rows and prevents "
            "effector/gripper rows from being pulled onto the object."
        ),
    )
    parser.add_argument(
        "--anchor-object-pull-graph-weight",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_graph_weight,
        help="Weight of the measurement-graph center term inside anchor_object_pull.",
    )
    parser.add_argument(
        "--anchor-object-pull-posterior-weight",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_posterior_weight,
        help=(
            "Weight of the posterior belief-file center term inside anchor_object_pull. "
            "This closes graph->posterior ownership supervision in object-owner probes."
        ),
    )
    parser.add_argument(
        "--anchor-object-pull-target-quality-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.anchor_object_pull_target_quality_gate_enabled,
        help=(
            "Weight object-pull supervision by compactness of the high-confidence sidecar/proposal target core. "
            "This keeps weak contact-motion masks as measurements rather than hard noisy labels."
        ),
    )
    parser.add_argument(
        "--anchor-object-pull-target-quality-sigma-m",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_target_quality_sigma_m,
    )
    parser.add_argument(
        "--anchor-object-pull-target-quality-min",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_target_quality_min,
    )
    parser.add_argument(
        "--anchor-object-pull-target-quality-power",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_target_quality_power,
    )
    parser.add_argument(
        "--anchor-object-pull-target-core-mass",
        type=float,
        default=_LOSS_DEFAULTS.anchor_object_pull_target_core_mass,
    )
    parser.add_argument(
        "--anchor-object-pull-target-core-topk",
        type=int,
        default=_LOSS_DEFAULTS.anchor_object_pull_target_core_topk,
    )
    parser.add_argument("--lambda-pv-weak", type=float, default=_LOSS_DEFAULTS.lambda_pv_weak)
    parser.add_argument("--lambda-pt", type=float, default=_LOSS_DEFAULTS.lambda_pt)
    parser.add_argument(
        "--anchor-pv-object-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.anchor_pv_object_gate_enabled,
        help=(
            "Gate the anchor-PV object loss by AQR object-routed point/visual support. "
            "Dense background evidence is supervised separately by pv_weak."
        ),
    )
    parser.add_argument(
        "--anchor-pv-active-object-gate-only",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.anchor_pv_active_object_gate_only,
        help=(
            "Use only active object files, not reserve/context rows, when building the object gate for anchor-PV. "
            "Dense background still remains visible through the global PV floor and pv_weak."
        ),
    )
    parser.add_argument("--anchor-pv-object-gate-floor", type=float, default=_LOSS_DEFAULTS.anchor_pv_object_gate_floor)
    parser.add_argument(
        "--anchor-pv-object-normalize-by-object-mass",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.anchor_pv_object_normalize_by_object_mass,
        help=(
            "Normalize anchor-PV object loss over object-confirmed edges instead of all dense projective edges. "
            "Global dense coverage remains in loss_pv_weak."
        ),
    )
    parser.add_argument(
        "--anchor-pv-object-distribution-loss",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.anchor_pv_object_distribution_loss,
        help=(
            "Use per-object-row point/visual distribution consistency for anchor-PV instead of dense edge BCE. "
            "This keeps dense/background PV in pv_weak and trains object slots only on their own support distributions."
        ),
    )
    parser.add_argument(
        "--anchor-pv-object-distribution-confirmed-only",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.anchor_pv_object_distribution_confirmed_only,
        help="Apply distributional anchor-PV only to active rows confirmed by sidecar/proposal/point evidence.",
    )
    parser.add_argument(
        "--anchor-pv-object-distribution-confirmation-threshold",
        type=float,
        default=_LOSS_DEFAULTS.anchor_pv_object_distribution_confirmation_threshold,
    )
    parser.add_argument("--lambda-vl-heatmap-task", type=float, default=_LOSS_DEFAULTS.lambda_vl_heatmap_task)
    parser.add_argument("--lambda-vl-heatmap-effector", type=float, default=_LOSS_DEFAULTS.lambda_vl_heatmap_effector)
    parser.add_argument("--lambda-vl-heatmap-interaction", type=float, default=_LOSS_DEFAULTS.lambda_vl_heatmap_interaction)
    parser.add_argument("--lambda-vl-point-consistency", type=float, default=_LOSS_DEFAULTS.lambda_vl_point_consistency)
    parser.add_argument("--lambda-vl-anchor-diversity", type=float, default=_LOSS_DEFAULTS.lambda_vl_anchor_diversity)
    parser.add_argument("--vl-heatmap-sigma-patches", type=float, default=_LOSS_DEFAULTS.vl_heatmap_sigma_patches)
    parser.add_argument("--vl-point-consistency-eps", type=float, default=_LOSS_DEFAULTS.vl_point_consistency_eps)
    parser.add_argument("--vl-anchor-diversity-radius-m", type=float, default=_SPEC_DEFAULTS.vl_anchor_local_sigma_m)
    parser.add_argument("--lambda-mapg-siglip", type=float, default=_LOSS_DEFAULTS.lambda_mapg_siglip)
    parser.add_argument("--lambda-mapg-vicreg", type=float, default=_LOSS_DEFAULTS.lambda_mapg_vicreg)
    parser.add_argument("--lambda-mapg-cycle", type=float, default=_LOSS_DEFAULTS.lambda_mapg_cycle)
    parser.add_argument("--lambda-mapg-masked-modality", type=float, default=_LOSS_DEFAULTS.lambda_mapg_masked_modality)
    parser.add_argument("--lambda-mapg-routing", type=float, default=_LOSS_DEFAULTS.lambda_mapg_routing)
    parser.add_argument("--lambda-mapg-support-diversity", type=float, default=_LOSS_DEFAULTS.lambda_mapg_support_diversity)
    parser.add_argument("--lambda-mapg-geometry-diversity", type=float, default=_LOSS_DEFAULTS.lambda_mapg_geometry_diversity)
    parser.add_argument("--lambda-slot-jepa", type=float, default=_LOSS_DEFAULTS.lambda_slot_jepa)
    parser.add_argument("--lambda-support-pred", type=float, default=_LOSS_DEFAULTS.lambda_support_pred)
    parser.add_argument("--lambda-binding-consistency", type=float, default=_LOSS_DEFAULTS.lambda_binding_consistency)
    parser.add_argument("--lambda-aqr-denoising", type=float, default=_LOSS_DEFAULTS.lambda_aqr_denoising)
    parser.add_argument(
        "--lambda-slot-quality",
        type=float,
        default=_LOSS_DEFAULTS.lambda_slot_quality,
        help=(
            "Weak BCE loss for the adaptive object/no-object/duplicate slot-quality head. "
            "Targets are derived from sidecar/tracklet/point/contact measurements and default to zero unless explicitly enabled."
        ),
    )
    parser.add_argument(
        "--aqr-denoising-active-object-only",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.aqr_denoising_active_object_only,
        help="Apply AQR support denoising only to active object files, excluding reserve/no-object rows.",
    )
    parser.add_argument(
        "--aqr-denoising-confirmed-object-only",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.aqr_denoising_confirmed_object_only,
        help=(
            "Apply AQR support denoising only to active rows confirmed by object-candidate/proposal/point evidence."
        ),
    )
    parser.add_argument(
        "--aqr-denoising-confirmation-threshold",
        type=float,
        default=_LOSS_DEFAULTS.aqr_denoising_confirmation_threshold,
    )
    parser.add_argument("--lambda-vcap-unexplained", type=float, default=_LOSS_DEFAULTS.lambda_vcap_unexplained)
    parser.add_argument("--lambda-vcap-duplicate", type=float, default=_LOSS_DEFAULTS.lambda_vcap_duplicate)
    parser.add_argument("--lambda-vcap-count", type=float, default=_LOSS_DEFAULTS.lambda_vcap_count)
    parser.add_argument("--lambda-vcap-continuity", type=float, default=_LOSS_DEFAULTS.lambda_vcap_continuity)
    parser.add_argument("--lambda-object-explanation-feature", type=float, default=_LOSS_DEFAULTS.lambda_object_explanation_feature)
    parser.add_argument("--lambda-object-explanation-point", type=float, default=_LOSS_DEFAULTS.lambda_object_explanation_point)
    parser.add_argument("--lambda-object-explanation-contact", type=float, default=_LOSS_DEFAULTS.lambda_object_explanation_contact)
    parser.add_argument("--lambda-object-explanation-duplicate", type=float, default=_LOSS_DEFAULTS.lambda_object_explanation_duplicate)
    parser.add_argument("--lambda-object-explanation-background", type=float, default=_LOSS_DEFAULTS.lambda_object_explanation_background)
    parser.add_argument(
        "--object-explanation-active-object-only",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.object_explanation_active_object_only,
        help="Compute object-explanation feature/duplicate terms over active object files only.",
    )
    parser.add_argument("--object-explanation-duplicate-margin", type=float, default=_LOSS_DEFAULTS.object_explanation_duplicate_margin)
    parser.add_argument(
        "--object-explanation-point-loss-clip",
        type=float,
        default=_LOSS_DEFAULTS.object_explanation_point_loss_clip,
        help=(
            "Upper bound for the OEML point compactness auxiliary. "
            "This prevents noisy sidecar mask tails from becoming hard unbounded labels."
        ),
    )
    parser.add_argument(
        "--object-explanation-point-quality-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_LOSS_DEFAULTS.object_explanation_point_quality_gate_enabled,
        help=(
            "Weight the OEML point compactness auxiliary by detached point-quality instead "
            "of treating noisy/sparse sidecar point masks as equally reliable hard labels."
        ),
    )
    parser.add_argument(
        "--object-explanation-point-quality-min",
        type=float,
        default=_LOSS_DEFAULTS.object_explanation_point_quality_min,
    )
    parser.add_argument(
        "--object-explanation-point-quality-power",
        type=float,
        default=_LOSS_DEFAULTS.object_explanation_point_quality_power,
    )
    parser.add_argument(
        "--object-explanation-point-outlier-prior",
        type=float,
        default=_LOSS_DEFAULTS.object_explanation_point_outlier_prior,
        help="Robust outlier prior for the OEML point compactness mixture loss.",
    )
    parser.add_argument(
        "--object-scaffold-decay-mode",
        choices=["none", "linear", "cosine"],
        default="none",
        help=(
            "Optional curriculum for weak object-scaffold losses. "
            "Use cosine for production: sidecar/contact/tracklet teachers bootstrap ownership early, "
            "then decay so action and dense context dominate the long run."
        ),
    )
    parser.add_argument(
        "--object-scaffold-decay-start-step",
        type=int,
        default=0,
        help="Global step where object-scaffold decay starts. Resume uses the loaded global step.",
    )
    parser.add_argument(
        "--object-scaffold-decay-end-step",
        type=int,
        default=0,
        help="Global step where object-scaffold decay reaches its floor.",
    )
    parser.add_argument(
        "--object-scaffold-decay-floor",
        type=float,
        default=1.0,
        help=(
            "Long-run multiplier for object-scaffold losses. "
            "A floor around 0.10 keeps weak sidecar shaping near action-loss scale."
        ),
    )
    parser.add_argument(
        "--picf-action-detach-from-anchor",
        dest="detach_action_loss_from_picf",
        action="store_true",
        help=(
            "Diagnostic only: report action loss normally but stop its gradient before it reaches "
            "PICF anchor/posterior parameters. Use to isolate action-gradient effects on recycle/binding."
        ),
    )
    parser.add_argument("--no-picf-action-detach-from-anchor", dest="detach_action_loss_from_picf", action="store_false")
    parser.set_defaults(detach_action_loss_from_picf=_LOSS_DEFAULTS.detach_action_loss_from_picf)
    parser.add_argument(
        "--picf-action-prefix-stopgrad",
        dest="action_prefix_stopgrad",
        action="store_true",
        help=(
            "Stop gradients from PI0.5 action-flow loss at PICF pi-prefix tokens while still allowing "
            "the action generator side to receive its normal loss. This is the cotrain-safe bridge "
            "for preventing action gradients from using posterior recycle/reset as a shortcut."
        ),
    )
    parser.add_argument("--no-picf-action-prefix-stopgrad", dest="action_prefix_stopgrad", action="store_false")
    parser.set_defaults(action_prefix_stopgrad=_SPEC_DEFAULTS.action_prefix_stopgrad)
    parser.add_argument(
        "--action-prefix-norm-mode",
        choices=["none", "rmsnorm", "rms_norm", "rmscap", "rms_cap", "layernorm", "layer_norm"],
        default=_SPEC_DEFAULTS.action_prefix_norm_mode,
        help=(
            "Normalize only the PICF-to-action prefix interface. This stabilizes the action-visible "
            "belief prefix without changing PICF internal belief dynamics."
        ),
    )
    parser.add_argument(
        "--action-prefix-rms-target",
        type=float,
        default=_SPEC_DEFAULTS.action_prefix_rms_target,
        help="Target RMS for action prefix interface normalization/capping.",
    )
    parser.add_argument(
        "--action-prefix-norm-eps",
        type=float,
        default=_SPEC_DEFAULTS.action_prefix_norm_eps,
        help="Numerical epsilon for action prefix interface normalization.",
    )
    parser.add_argument(
        "--action-prefix-value-clip",
        type=float,
        default=_SPEC_DEFAULTS.action_prefix_value_clip,
        help="Optional absolute value clip after action prefix normalization; <=0 disables clipping.",
    )
    parser.add_argument(
        "--picf-action-condition-enabled",
        dest="picf_action_condition_enabled",
        action="store_true",
        help=(
            "Allow PICF conditioned-control prefixes/context to enter the PI0.5 action path. "
            "Use --no-picf-action-condition-enabled for a clean causal control where PICF still "
            "computes and trains auxiliary belief losses, but action receives no PICF condition."
        ),
    )
    parser.add_argument(
        "--no-picf-action-condition-enabled",
        dest="picf_action_condition_enabled",
        action="store_false",
    )
    parser.set_defaults(picf_action_condition_enabled=_SPEC_DEFAULTS.picf_action_condition_enabled)
    parser.add_argument(
        "--action-prefix-output-gate",
        type=float,
        default=_SPEC_DEFAULTS.action_prefix_output_gate,
        help=(
            "Fixed 0..1 scalar gate applied to PICF extra-prefix tokens after interface normalization. "
            "This is the training-time counterpart of gated VLA conditioning: it bounds how much "
            "a moving PICF prefix can perturb the pretrained action stream."
        ),
    )
    parser.add_argument(
        "--action-prefix-teacher-mode",
        choices=["off", "ema"],
        default=_SPEC_DEFAULTS.action_prefix_teacher_mode,
        help=(
            "Train-time target-network bridge for PICF action prefixes. ema feeds PI0.5 a slow "
            "teacher prefix while the online prefix continues to train through PICF losses."
        ),
    )
    parser.add_argument(
        "--action-prefix-teacher-ema-decay",
        type=float,
        default=_SPEC_DEFAULTS.action_prefix_teacher_ema_decay,
        help="EMA decay for the action-prefix teacher buffer.",
    )
    parser.add_argument(
        "--action-prefix-teacher-blend",
        type=float,
        default=_SPEC_DEFAULTS.action_prefix_teacher_blend,
        help="0..1 blend from online prefix to EMA teacher prefix; 1 uses only the teacher.",
    )
    parser.add_argument(
        "--lambda-action-prefix-trust",
        type=float,
        default=_SPEC_DEFAULTS.lambda_action_prefix_trust,
        help=(
            "Auxiliary trust-region weight between online PICF prefix and EMA teacher prefix. "
            "It enters alignment/total_minus_action, not loss_action_default_equiv."
        ),
    )
    parser.add_argument(
        "--action-context-tokens",
        type=int,
        default=_SPEC_DEFAULTS.action_context_tokens,
        help=(
            "Expose up to this many conditioned-control context tokens to the PI action path. "
            "The integration mode decides whether they are appended or fused into the fixed PI prefix."
        ),
    )
    parser.add_argument(
        "--action-context-integration",
        choices=["append", "prefix_fusion", "suffix_cross_attention"],
        default=_SPEC_DEFAULTS.action_context_integration,
        help=(
            "How action-context tokens enter PI0.5. append increases prefix length and shifts suffix "
            "positions; prefix_fusion keeps the PI prefix length fixed by reading context through a "
            "bounded residual attention adapter; suffix_cross_attention keeps the native PI prefix and "
            "lets action suffix tokens read PICF context through the semantic action adapter."
        ),
    )
    parser.add_argument(
        "--picf-action-context-stopgrad",
        dest="action_context_stopgrad",
        action="store_true",
        help="Detach appended action-context tokens from action gradients while still exposing them to PI0.5.",
    )
    parser.add_argument("--no-picf-action-context-stopgrad", dest="action_context_stopgrad", action="store_false")
    parser.set_defaults(action_context_stopgrad=_SPEC_DEFAULTS.action_context_stopgrad)
    parser.add_argument(
        "--action-context-norm-mode",
        choices=["none", "rmsnorm", "rms_norm", "rmscap", "rms_cap", "layernorm", "layer_norm"],
        default=_SPEC_DEFAULTS.action_context_norm_mode,
        help="Normalize appended action-context tokens independently from the compressed PI prefix.",
    )
    parser.add_argument(
        "--action-context-rms-target",
        type=float,
        default=_SPEC_DEFAULTS.action_context_rms_target,
        help="Target RMS for appended action-context tokens.",
    )
    parser.add_argument(
        "--action-context-output-gate",
        type=float,
        default=_SPEC_DEFAULTS.action_context_output_gate,
        help="Fixed 0..1 gate applied to appended action-context tokens after normalization.",
    )
    parser.add_argument(
        "--action-context-include-query-tokens",
        dest="action_context_include_query_tokens",
        action="store_true",
        help="Include conditioned-control query tokens in the appended action context.",
    )
    parser.add_argument(
        "--no-action-context-include-query-tokens",
        dest="action_context_include_query_tokens",
        action="store_false",
    )
    parser.set_defaults(action_context_include_query_tokens=_SPEC_DEFAULTS.action_context_include_query_tokens)
    parser.add_argument(
        "--picf-core-lr-runtime-mode",
        choices=["constant", "block_alternating"],
        default="constant",
        help=(
            "Runtime schedule for the picf_core optimizer group. block_alternating updates PICF core "
            "only during short active windows, then gives the action path stationary-prefix recovery steps."
        ),
    )
    parser.add_argument(
        "--picf-core-lr-block-start-step",
        type=int,
        default=0,
        help="Global optimizer step at which block_alternating PICF core updates begin.",
    )
    parser.add_argument(
        "--picf-core-lr-block-cycle-steps",
        type=int,
        default=0,
        help="Cycle length for block_alternating PICF core LR scheduling.",
    )
    parser.add_argument(
        "--picf-core-lr-block-active-steps",
        type=int,
        default=0,
        help="Number of PICF-core update steps at the beginning of each block_alternating cycle.",
    )
    parser.add_argument("--mapg-siglip-tau", type=float, default=_LOSS_DEFAULTS.mapg_siglip_tau)
    parser.add_argument("--mapg-vicreg-var-target", type=float, default=_LOSS_DEFAULTS.mapg_vicreg_var_target)
    parser.add_argument("--mapg-vicreg-cov-weight", type=float, default=_LOSS_DEFAULTS.mapg_vicreg_cov_weight)
    parser.add_argument("--mapg-support-div-margin-visual", type=float, default=_LOSS_DEFAULTS.mapg_support_div_margin_visual)
    parser.add_argument("--mapg-support-div-margin-point", type=float, default=_LOSS_DEFAULTS.mapg_support_div_margin_point)
    parser.add_argument("--mapg-support-div-margin-tactile", type=float, default=_LOSS_DEFAULTS.mapg_support_div_margin_tactile)
    parser.add_argument("--mapg-support-div-margin-posterior", type=float, default=_LOSS_DEFAULTS.mapg_support_div_margin_posterior)
    parser.add_argument("--mapg-support-div-sigma-visual-patches", type=float, default=_LOSS_DEFAULTS.mapg_support_div_sigma_visual_patches)
    parser.add_argument("--mapg-support-div-sigma-point-m", type=float, default=_LOSS_DEFAULTS.mapg_support_div_sigma_point_m)
    parser.add_argument("--mapg-support-div-direct-visual-weight", type=float, default=_LOSS_DEFAULTS.mapg_support_div_direct_visual_weight)
    parser.add_argument("--mapg-support-div-local-candidate-weight", type=float, default=_LOSS_DEFAULTS.mapg_support_div_local_candidate_weight)
    parser.add_argument("--mapg-support-div-local-margin", type=float, default=_LOSS_DEFAULTS.mapg_support_div_local_margin)
    parser.add_argument("--mapg-support-div-tail-topk", type=int, default=_LOSS_DEFAULTS.mapg_support_div_tail_topk)
    parser.add_argument("--mapg-geometry-diversity-margin", type=float, default=_LOSS_DEFAULTS.mapg_geometry_diversity_margin)
    parser.add_argument("--mapg-geometry-diversity-jitter-m", type=float, default=_LOSS_DEFAULTS.mapg_geometry_diversity_jitter_m)
    parser.add_argument("--enable-aux-budgeting", dest="enable_aux_budgeting", action="store_true")
    parser.add_argument("--disable-aux-budgeting", dest="enable_aux_budgeting", action="store_false")
    parser.add_argument("--aux-budget-physical-ratio", type=float, default=_LOSS_DEFAULTS.aux_budget_physical_ratio)
    parser.add_argument("--aux-budget-semantic-ratio", type=float, default=_LOSS_DEFAULTS.aux_budget_semantic_ratio)
    parser.add_argument("--aux-budget-alignment-ratio", type=float, default=_LOSS_DEFAULTS.aux_budget_alignment_ratio)
    parser.add_argument("--aux-budget-floor", type=float, default=_LOSS_DEFAULTS.aux_budget_floor)
    parser.add_argument("--aux-budget-alignment-floor", type=float, default=_LOSS_DEFAULTS.aux_budget_alignment_floor)
    parser.add_argument("--tau-pv", type=float, default=_LOSS_DEFAULTS.tau_pv)
    parser.add_argument("--tau-pt", type=float, default=_LOSS_DEFAULTS.tau_pt)
    parser.add_argument("--tau-route-p", type=float, default=_LOSS_DEFAULTS.tau_route_p)
    parser.add_argument("--tau-route-v", type=float, default=_LOSS_DEFAULTS.tau_route_v)
    parser.add_argument("--pt-bag-radius-m", type=float, default=None)
    parser.add_argument("--pt-bag-sigma-m", type=float, default=None)
    parser.add_argument("--pt-bag-kmin", type=int, default=_LOSS_DEFAULTS.pt_bag_kmin)
    parser.add_argument("--pt-back-slack-m", type=float, default=_LOSS_DEFAULTS.pt_back_slack_m)
    parser.add_argument("--p-align-on", type=float, default=_LOSS_DEFAULTS.p_align_on)
    parser.add_argument("--p-align-off", type=float, default=_LOSS_DEFAULTS.p_align_off)
    parser.add_argument("--tactile-aux-force-scale", type=float, default=_LOSS_DEFAULTS.tactile_aux_force_scale)
    parser.add_argument("--tactile-aux-indent-scale", type=float, default=_LOSS_DEFAULTS.tactile_aux_indent_scale)
    parser.add_argument("--tactile-aux-pressure-scale", type=float, default=_LOSS_DEFAULTS.tactile_aux_pressure_scale)
    parser.add_argument("--tactile-aux-pose-scale", type=float, default=None)
    parser.add_argument("--tactile-aux-huber-delta", type=float, default=_LOSS_DEFAULTS.tactile_aux_huber_delta)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--step-indexed-window-rng",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Key sampled-window RNG by (seed, rank, global_step, micro_step, retry). "
            "This is resume-safe and prevents a checkpoint continuation from replaying "
            "the early train-window stream. Use --no-step-indexed-window-rng only to "
            "reproduce legacy May-2026 rebound diagnostics."
        ),
    )
    parser.add_argument("--project-name", default="openpi")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--visual-grid", type=int, default=8)
    parser.add_argument(
        "--visual-real-grid",
        type=int,
        default=_SPEC_DEFAULTS.visual_real_grid,
        help=(
            "Side length of the RGB future target used by loss_visual_real. "
            "The v2.2 default is 64, replacing the historical 4x4 diagnostic target. "
            "Use 4 only for lightweight unit tests or compatibility probes."
        ),
    )
    parser.add_argument("--use-foundation-backbones", action="store_true")
    parser.add_argument(
        "--perception-finetune-mode",
        choices=["auto", "full", "frozen"],
        default="auto",
        help=(
            "High-level perception backbone trainability profile. 'auto' preserves existing per-module flags; "
            "'full' trains V-JEPA/Sonata/AnyTouch; 'frozen' freezes those perception backbones while leaving "
            "semantic/action and PICF adaptation modules trainable."
        ),
    )
    parser.add_argument(
        "--picf-augmentation-mode",
        choices=["off", "photometric", "multimodal_geometry"],
        default="off",
        help=(
            "Full-PICF train-time augmentation policy. 'off' is the default. 'photometric' applies "
            "geometry-preserving RGB color jitter. 'multimodal_geometry' is reserved and fail-fast until "
            "synchronized RGB/depth/point/camera transforms are implemented."
        ),
    )
    parser.add_argument(
        "--picf-photometric-strength",
        choices=["conservative", "reference"],
        default="conservative",
        help="Photometric jitter strength when --picf-augmentation-mode=photometric.",
    )
    parser.add_argument("--point-backbone", choices=["rgb", "sonata"], default="rgb")
    parser.add_argument("--point-backbone-trainable", action="store_true")
    parser.add_argument("--sonata-checkpoint-path", default=None)
    parser.add_argument("--sonata-stage-name", default="enc4")
    parser.add_argument("--sonata-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--sonata-disable-flash", action="store_true")
    parser.add_argument("--point-backbone-lr-scale", type=float, default=0.25)
    parser.add_argument("--visual-mode", choices=["stub", "encoder"], default="stub")
    parser.add_argument("--visual-finetune-mode", choices=["auto", "full", "frozen"], default="auto")
    parser.add_argument("--visual-trainable", action="store_true")
    parser.add_argument("--visual-model-name", default="vjepa2_1_vit_base_384")
    parser.add_argument("--visual-checkpoint-path", default=None)
    parser.add_argument("--visual-checkpoint-key", default=None)
    parser.add_argument("--visual-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument(
        "--visual-feature-mode",
        choices=["auto", "hierarchical", "final"],
        default="auto",
        help=(
            "V-JEPA feature layout. 'auto' preserves historical behavior "
            "(hierarchical when trainable, final when frozen). Use 'hierarchical' "
            "with --perception-finetune-mode frozen when PICF should keep the same "
            "visual feature contract as the full-train profile."
        ),
    )
    parser.add_argument("--visual-lr-scale", type=float, default=0.25)
    parser.add_argument("--visual-activation-checkpointing", action="store_true")
    parser.add_argument("--visual-img-size", type=int, default=384)
    parser.add_argument("--visual-num-frames", type=int, default=64)
    parser.add_argument("--visual-patch-size", type=int, default=16)
    parser.add_argument("--visual-tubelet-size", type=int, default=2)
    parser.add_argument("--visual-use-last-two-mean", action="store_true")
    parser.add_argument(
        "--vjepa-feature-cache-root",
        default=None,
        help=(
            "Optional deterministic frozen V-JEPA feature cache root. "
            "Only valid when the V-JEPA backbone is frozen."
        ),
    )
    parser.add_argument(
        "--vjepa-feature-cache-mode",
        choices=["off", "read", "read_or_encode"],
        default="off",
        help=(
            "V-JEPA frozen feature cache mode. 'read' fails closed on missing/stale entries; "
            "'read_or_encode' reads valid entries and writes misses."
        ),
    )
    parser.add_argument(
        "--vjepa-feature-cache-temporal-slices",
        type=int,
        default=4,
        help=(
            "Number of most-recent V-JEPA temporal maps to persist per frozen-feature cache entry. "
            "PICF consumes current_map/recent_maps, so a bounded suffix preserves the runtime objective "
            "while avoiding full 32-slice dense-volume cache files."
        ),
    )
    parser.add_argument(
        "--vjepa-feature-cache-storage-dtype",
        choices=["float32", "bfloat16", "float16"],
        default="bfloat16",
        help="On-disk dtype for deterministic frozen V-JEPA feature cache entries.",
    )
    parser.add_argument("--tactile-mode", choices=["stub", "encoder"], default="stub")
    parser.add_argument("--tactile-trainable", action="store_true")
    parser.add_argument("--tactile-checkpoint-path", default=None)
    parser.add_argument("--tactile-dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--tactile-lr-scale", type=float, default=0.25)
    parser.add_argument("--tactile-num-frames", type=int, default=4)
    parser.add_argument("--tactile-stride", type=int, default=2)
    parser.add_argument("--use-tactile", action="store_true")
    parser.add_argument("--tactile-sensor-names", default="digit,gelsight_mini")
    parser.add_argument("--tactile-sensor-offsets-m", default="0.01,0,0;-0.01,0,0")
    parser.add_argument("--tactile-calibration-path", default=None)
    parser.add_argument("--tactile-backgrounds-path", default=None)
    parser.add_argument("--tactile-contact-stats-path", default=None)
    parser.add_argument("--tactile-contact-tau-on", type=float, default=None)
    parser.add_argument("--tactile-contact-tau-off", type=float, default=None)
    parser.add_argument("--tactile-contact-temperature", type=float, default=None)
    parser.add_argument("--tactile-contact-ema-beta", type=float, default=_SPEC_DEFAULTS.tactile_contact_ema_beta)
    parser.add_argument("--tactile-evidence-prob-floor", type=float, default=_SPEC_DEFAULTS.tactile_evidence_prob_floor)
    parser.add_argument("--tactile-anchor-prob-on", type=float, default=_SPEC_DEFAULTS.tactile_anchor_prob_on)
    parser.add_argument(
        "--tactile-attach-to-object-owner",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.tactile_attach_to_object_owner,
        help=(
            "Route tactile/contact tokens to role-1 object owners during assignment. "
            "This treats touch as evidence about the contacted object rather than an independent effector owner."
        ),
    )
    parser.add_argument("--use-scene-obs", action="store_true")
    parser.add_argument(
        "--semantic-mode",
        choices=["zero", "paligemma"],
        default="paligemma",
        help="Semantic encoder mode. The PICF-AQR-OWM production default is paligemma; use zero only with --no-aqr-mapg-enabled ablations.",
    )
    parser.add_argument("--semantic-source", choices=["auto", "hf", "pi0_pytorch"], default="auto")
    parser.add_argument("--semantic-model-name", default=_default_paligemma_model_name())
    parser.add_argument("--semantic-checkpoint-path", default=None)
    parser.add_argument("--semantic-checkpoint-config-path", default=None)
    parser.add_argument("--semantic-revision", default=None)
    parser.add_argument("--semantic-paligemma-variant", default="gemma_2b")
    parser.add_argument("--semantic-action-expert-variant", default="gemma_300m")
    parser.add_argument("--semantic-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--semantic-trainable", action="store_true")
    parser.add_argument(
        "--semantic-trainable-scope",
        choices=[
            "all",
            "backbone_only",
            "model_only",
            "action_head_only",
            "action_adapter_only",
            "action_head_and_adapter",
        ],
        default="all",
        help=(
            "Scope for trainable PaliGemma/PI0 semantic parameters when --semantic-trainable is set. "
            "'backbone_only'/'model_only' are the production default and reproduce the historical "
            "fast full-cotrain boundary by "
            "training paligemma_with_expert while freezing wrapper-local action_in/out projections "
            "and time MLPs. 'all' additionally trains those wrapper-local flow/time heads and is "
            "reserved for explicit diagnostics because it is materially slower under FSDP. "
            "'action_head_only' freezes the PaliGemma/Gemma backbone/expert and "
            "trains only action_in/out projections plus time MLPs for causal probes. "
            "'action_adapter_only' trains only the PICF-to-action suffix cross-attention adapter; "
            "'action_head_and_adapter' trains both the wrapper-local action head and that adapter."
        ),
    )
    parser.add_argument(
        "--semantic-action-context-adapter-gate-init",
        type=float,
        default=-2.0,
        help="Initial logit for the action-side PICF context adapter gate; sigmoid(-2) ~= 0.12.",
    )
    parser.add_argument(
        "--semantic-action-context-adapter-rms-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap action-context adapter residual RMS to the current action suffix RMS.",
    )
    parser.add_argument(
        "--semantic-action-flow-loss",
        choices=["mse", "l1", "huber", "smooth_l1"],
        default="mse",
        help=(
            "Training objective for PI0.5 flow velocity. The canonical MSE report remains "
            "loss_action_default_equiv for comparison with historical 4-22 runs; non-MSE modes "
            "only change the gradient objective."
        ),
    )
    parser.add_argument(
        "--semantic-action-flow-huber-delta",
        type=float,
        default=1.0,
        help="Delta/beta for huber or smooth_l1 action-flow objective.",
    )
    parser.add_argument(
        "--semantic-action-flow-time-alpha",
        type=float,
        default=1.5,
        help="Beta distribution alpha for PI0.5 action-flow training time sampling.",
    )
    parser.add_argument(
        "--semantic-action-flow-time-beta",
        type=float,
        default=1.0,
        help="Beta distribution beta for PI0.5 action-flow training time sampling.",
    )
    parser.add_argument(
        "--semantic-action-context-readout-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the G22 context-only action readout auxiliary. "
            "The auxiliary asks PICF action-context tokens alone to predict the action chunk; "
            "loss_action_default_equiv remains the canonical PI0.5 flow MSE."
        ),
    )
    parser.add_argument(
        "--semantic-action-context-readout-aux-loss",
        choices=["mse", "l1", "huber", "smooth_l1"],
        default="smooth_l1",
        help="Objective used by the context-only action readout auxiliary.",
    )
    parser.add_argument(
        "--semantic-action-context-readout-aux-huber-delta",
        type=float,
        default=1.0,
        help="Delta/beta for huber or smooth_l1 context-readout auxiliary.",
    )
    parser.add_argument(
        "--semantic-action-context-token-aux-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the G26 FAST-style PICF action-token auxiliary. "
            "PICF context readout states classify discretized action chunks; "
            "loss_action_default_equiv remains the canonical PI0.5 flow MSE."
        ),
    )
    parser.add_argument(
        "--semantic-action-context-token-aux-bins",
        type=int,
        default=256,
        help="Number of uniform bins per action dimension for the G26 action-token auxiliary.",
    )
    parser.add_argument(
        "--semantic-action-context-token-aux-clip",
        type=float,
        default=1.0,
        help="Symmetric action value clip used before quantization by the G26 action-token auxiliary.",
    )
    parser.add_argument(
        "--semantic-action-context-flow-residual-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable the G25 deployed-path PICF context residual for PI0.5 flow. "
            "The context readout predicts an action chunk target and converts it to a bounded "
            "flow velocity residual used by both training and sampling."
        ),
    )
    parser.add_argument(
        "--semantic-action-context-flow-residual-gate-init",
        type=float,
        default=-2.0,
        help="Initial logit for the deployed context-flow residual gate; sigmoid(-2) ~= 0.12.",
    )
    parser.add_argument(
        "--semantic-action-context-flow-residual-time-floor",
        type=float,
        default=0.05,
        help="Minimum flow time used when converting context target readout to velocity.",
    )
    parser.add_argument(
        "--semantic-action-context-flow-residual-rms-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap deployed context-flow residual RMS to the native PI0.5 velocity RMS.",
    )
    parser.add_argument(
        "--semantic-action-expert-router-enabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable semantic/PICF-conditioned low-rank expert routing on PI0.5 action suffix tokens. "
            "This is action-expert-only routing; it does not MoE the VLM backbone."
        ),
    )
    parser.add_argument(
        "--semantic-action-expert-router-experts",
        type=int,
        default=4,
        help="Number of low-rank action suffix experts for --semantic-action-expert-router-enabled.",
    )
    parser.add_argument(
        "--semantic-action-expert-router-rank",
        type=int,
        default=64,
        help="Low-rank bottleneck size for each action expert router residual.",
    )
    parser.add_argument(
        "--semantic-action-expert-router-gate-init",
        type=float,
        default=-2.5,
        help="Initial logit for the action-expert router residual gate; sigmoid(-2.5) ~= 0.076.",
    )
    parser.add_argument(
        "--semantic-action-expert-router-temperature",
        type=float,
        default=1.0,
        help="Softmax temperature for action-expert router mixture weights.",
    )
    parser.add_argument(
        "--semantic-action-expert-router-rms-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Cap action-expert router residual RMS to current action suffix RMS.",
    )
    parser.add_argument("--semantic-lr-scale", type=float, default=0.25)
    parser.add_argument("--semantic-gradient-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--window-activation-checkpointing", action="store_true")
    parser.add_argument("--semantic-use-gripper", action="store_true")
    parser.add_argument("--semantic-max-length", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=_SPEC_DEFAULTS.hidden_dim)
    parser.add_argument("--posterior-hidden-dim", type=int, default=_SPEC_DEFAULTS.posterior_hidden_dim)
    parser.add_argument("--latent-dim", type=int, default=_SPEC_DEFAULTS.latent_dim)
    parser.add_argument("--innovation-dim", type=int, default=_SPEC_DEFAULTS.innovation_dim)
    parser.add_argument("--control-dim", type=int, default=_SPEC_DEFAULTS.control_dim)
    parser.add_argument("--semantic-dim", type=int, default=_SPEC_DEFAULTS.semantic_dim)
    parser.add_argument("--semantic-cross-dim", type=int, default=_SPEC_DEFAULTS.semantic_cross_dim)
    parser.add_argument("--future-hidden-dim", type=int, default=_SPEC_DEFAULTS.future_hidden_dim)
    parser.add_argument("--persistent-anchors", type=int, default=_SPEC_DEFAULTS.persistent_anchors)
    parser.add_argument("--observation-anchors", type=int, default=_SPEC_DEFAULTS.observation_anchors)
    parser.add_argument("--effector-persistent-anchors", type=int, default=_SPEC_DEFAULTS.effector_persistent_anchors)
    parser.add_argument("--effector-observation-anchors", type=int, default=_SPEC_DEFAULTS.effector_observation_anchors)
    parser.add_argument(
        "--posterior-slot-identity-std",
        type=float,
        default=_SPEC_DEFAULTS.posterior_slot_identity_std,
        help=(
            "Stddev for persistent posterior slot identity seeds. Keep nonzero "
            "for object-file symmetry breaking; setting it to 0 reproduces the "
            "legacy symmetric posterior initialization."
        ),
    )
    parser.add_argument(
        "--task-slot-identity-std",
        type=float,
        default=_SPEC_DEFAULTS.task_slot_identity_std,
        help="Stddev for task query identity seeds.",
    )
    parser.add_argument(
        "--posterior-bootstrap-from-observation",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_bootstrap_from_observation,
        help=(
            "Initialize posterior geometry from current observation anchors by "
            "per-role farthest-point selection on the first step."
        ),
    )
    parser.add_argument("--fusion-layers", type=int, default=_SPEC_DEFAULTS.fusion_layers)
    parser.add_argument("--posterior-layers", type=int, default=_SPEC_DEFAULTS.posterior_layers)
    parser.add_argument("--predictive-layers", type=int, default=_SPEC_DEFAULTS.predictive_layers)
    parser.add_argument("--control-layers", type=int, default=_SPEC_DEFAULTS.control_layers)
    parser.add_argument("--control-query-tokens", type=int, default=_SPEC_DEFAULTS.control_query_tokens)
    parser.add_argument("--predictive-query-tokens", type=int, default=_SPEC_DEFAULTS.predictive_query_tokens)
    parser.add_argument("--task-local-queries", type=int, default=_SPEC_DEFAULTS.task_local_queries)
    parser.add_argument("--task-effector-queries", type=int, default=_SPEC_DEFAULTS.task_effector_queries)
    parser.add_argument("--task-global-queries", type=int, default=_SPEC_DEFAULTS.task_global_queries)
    parser.add_argument("--task-instruction-queries", type=int, default=_SPEC_DEFAULTS.task_instruction_queries)
    parser.add_argument("--task-self-layers", type=int, default=_SPEC_DEFAULTS.task_self_layers)
    parser.add_argument("--conditioned-control-queries", type=int, default=_SPEC_DEFAULTS.conditioned_control_queries)
    parser.add_argument("--pi-prefix-queries", type=int, default=_SPEC_DEFAULTS.pi_prefix_queries)
    parser.add_argument("--conditioned-future-queries", type=int, default=_SPEC_DEFAULTS.conditioned_future_queries)
    parser.add_argument("--predictive-semantic-reads", type=int, default=_SPEC_DEFAULTS.predictive_semantic_reads)
    parser.add_argument("--control-semantic-reads", type=int, default=_SPEC_DEFAULTS.control_semantic_reads)
    parser.add_argument("--predictive-semantic-dropout-prob", type=float, default=_SPEC_DEFAULTS.predictive_semantic_dropout_prob)
    parser.add_argument("--semantic-prefix-dropout-prob", type=float, default=_SPEC_DEFAULTS.semantic_prefix_dropout_prob)
    parser.add_argument("--global-scene-point-cap", type=int, default=_SPEC_DEFAULTS.global_scene_point_cap)
    parser.add_argument("--scene-anchor-border-patches", type=float, default=_SPEC_DEFAULTS.scene_anchor_border_patches)
    parser.add_argument("--task-visual-reread-topk", type=int, default=_SPEC_DEFAULTS.task_visual_reread_topk)
    parser.add_argument("--task-tactile-reread-groups", type=int, default=_SPEC_DEFAULTS.task_tactile_reread_groups)
    parser.add_argument("--task-point-reread-topk", type=int, default=_SPEC_DEFAULTS.task_point_reread_topk)
    parser.add_argument(
        "--vl-anchor-router-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.vl_anchor_router_enabled,
        help=(
            "Enable the PaliGemma-guided 2D-to-3D anchor router. "
            "The default is off; when enabled it requires picf_mode=enabled and semantic_mode=paligemma."
        ),
    )
    parser.add_argument("--vl-grounding-view", default=_SPEC_DEFAULTS.vl_grounding_view)
    parser.add_argument("--vl-anchor-modes", type=int, default=_SPEC_DEFAULTS.vl_anchor_modes)
    parser.add_argument("--vl-anchor-nms-radius-m", type=float, default=_SPEC_DEFAULTS.vl_anchor_nms_radius_m)
    parser.add_argument("--vl-anchor-local-sigma-m", type=float, default=_SPEC_DEFAULTS.vl_anchor_local_sigma_m)
    parser.add_argument("--vl-min-visible-mass", type=float, default=_SPEC_DEFAULTS.vl_min_visible_mass)
    parser.add_argument("--vl-heatmap-temperature", type=float, default=_SPEC_DEFAULTS.vl_heatmap_temperature)
    parser.add_argument("--vl-obs-anchor-gate-init", type=float, default=_SPEC_DEFAULTS.vl_obs_anchor_gate_init)
    parser.add_argument("--vl-task-point-gate-init", type=float, default=_SPEC_DEFAULTS.vl_task_point_gate_init)
    parser.add_argument("--vl-posterior-bind-gate-init", type=float, default=_SPEC_DEFAULTS.vl_posterior_bind_gate_init)
    parser.add_argument("--vl-prior-bias-clip", type=float, default=_SPEC_DEFAULTS.vl_prior_bias_clip)
    parser.add_argument(
        "--mapg-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.mapg_enabled,
        help=(
            "Enable MAPG-PICF: modality-native anchor prior graph over PaliGemma, "
            "V-JEPA, point/Sonata, tactile/AnyTouch, posterior, observation anchors, "
            "task readout, posterior binding, and PI0.5 action prefix."
        ),
    )
    parser.add_argument("--mapg-anchor-count", type=int, default=_SPEC_DEFAULTS.mapg_anchor_count)
    parser.add_argument("--mapg-message-rounds", type=int, default=_SPEC_DEFAULTS.mapg_message_rounds)
    parser.add_argument("--mapg-visual-sigma-patches", type=float, default=_SPEC_DEFAULTS.mapg_visual_sigma_patches)
    parser.add_argument("--mapg-tactile-sigma-m", type=float, default=_SPEC_DEFAULTS.mapg_tactile_sigma_m)
    parser.add_argument("--mapg-posterior-sigma-m", type=float, default=_SPEC_DEFAULTS.mapg_posterior_sigma_m)
    parser.add_argument("--mapg-confidence-floor", type=float, default=_SPEC_DEFAULTS.mapg_confidence_floor)
    parser.add_argument("--mapg-assignment-sinkhorn-iters", type=int, default=_SPEC_DEFAULTS.mapg_assignment_sinkhorn_iters)
    parser.add_argument("--mapg-assignment-temperature", type=float, default=_SPEC_DEFAULTS.mapg_assignment_temperature)
    parser.add_argument("--mapg-assignment-quality-uniform-mix", type=float, default=_SPEC_DEFAULTS.mapg_assignment_quality_uniform_mix)
    parser.add_argument("--mapg-mode-confidence-threshold", type=float, default=_SPEC_DEFAULTS.mapg_mode_confidence_threshold)
    parser.add_argument("--mapg-obs-gate-init", type=float, default=_SPEC_DEFAULTS.mapg_obs_gate_init)
    parser.add_argument("--mapg-task-gate-init", type=float, default=_SPEC_DEFAULTS.mapg_task_gate_init)
    parser.add_argument("--mapg-posterior-gate-init", type=float, default=_SPEC_DEFAULTS.mapg_posterior_gate_init)
    parser.add_argument("--mapg-control-gate-init", type=float, default=_SPEC_DEFAULTS.mapg_control_gate_init)
    parser.add_argument("--mapg-obs-point-mix-floor", type=float, default=_SPEC_DEFAULTS.mapg_obs_point_mix_floor)
    parser.add_argument("--mapg-prior-bias-clip", type=float, default=_SPEC_DEFAULTS.mapg_prior_bias_clip)
    parser.add_argument(
        "--aqr-mapg-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_mapg_enabled,
        help=(
            "Enable direct-final AQR-MAPG: learned physical/task anchor queries over typed "
            "visual/point/tactile/posterior support memory. This is the production default "
            "and is mutually exclusive with legacy mapg_enabled."
        ),
    )
    parser.add_argument("--aqr-query-count-physical", type=int, default=_SPEC_DEFAULTS.aqr_query_count_physical)
    parser.add_argument("--aqr-query-count-task", type=int, default=_SPEC_DEFAULTS.aqr_query_count_task)
    parser.add_argument(
        "--aqr-role-layout",
        type=str,
        default=_SPEC_DEFAULTS.aqr_role_layout,
        choices=("structured", "no_effector", "object_contact_context", "object_only", "object"),
        help=(
            "AQR graph role layout. structured keeps the historical blue effector role; "
            "object_only makes every AQR graph row a role-1 object-owner row for isolated binding probes."
        ),
    )
    parser.add_argument("--aqr-query-rounds", type=int, default=_SPEC_DEFAULTS.aqr_query_rounds)
    parser.add_argument("--aqr-sinkhorn-iters", type=int, default=_SPEC_DEFAULTS.aqr_sinkhorn_iters)
    parser.add_argument("--aqr-sinkhorn-temperature", type=float, default=_SPEC_DEFAULTS.aqr_sinkhorn_temperature)
    parser.add_argument(
        "--aqr-pg-grounding-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_pg_grounding_enabled,
        help=(
            "Explicitly enable the PaliGemma heatmap/grounding head for AQR diagnostics "
            "or ablations. Default is off; PaliGemma semantic conditioning remains active."
        ),
    )
    parser.add_argument(
        "--aqr-pg-image-support-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_pg_image_support_enabled,
        help=(
            "Let AQR task queries read PaliGemma image tokens as typed visual-semantic support "
            "and project that support onto the V-JEPA grid. This is not the PaliGemma heatmap head."
        ),
    )
    parser.add_argument("--aqr-pg-image-support-weight", type=float, default=_SPEC_DEFAULTS.aqr_pg_image_support_weight)
    parser.add_argument("--aqr-pg-entropy-threshold", type=float, default=_SPEC_DEFAULTS.aqr_pg_entropy_threshold)
    parser.add_argument("--aqr-pg-peak-threshold", type=float, default=_SPEC_DEFAULTS.aqr_pg_peak_threshold)
    parser.add_argument("--aqr-pg-bias-weight", type=float, default=_SPEC_DEFAULTS.aqr_pg_bias_weight)
    parser.add_argument("--aqr-support-bias-clip", type=float, default=_SPEC_DEFAULTS.aqr_support_bias_clip)
    parser.add_argument(
        "--aqr-ownership-prior-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_ownership_prior_enabled,
        help=(
            "Enable low-amplitude role-local ownership priors for AQR support reads. "
            "This breaks identical same-role assignment rows before Sinkhorn; it is not an auxiliary loss."
        ),
    )
    parser.add_argument("--aqr-ownership-prior-weight", type=float, default=_SPEC_DEFAULTS.aqr_ownership_prior_weight)
    parser.add_argument(
        "--aqr-ownership-point-prior-weight",
        type=float,
        default=_SPEC_DEFAULTS.aqr_ownership_point_prior_weight,
    )
    parser.add_argument(
        "--aqr-ownership-point-prior-sigma-m",
        type=float,
        default=_SPEC_DEFAULTS.aqr_ownership_point_prior_sigma_m,
    )
    parser.add_argument(
        "--aqr-ownership-temporal-prior-weight",
        type=float,
        default=_SPEC_DEFAULTS.aqr_ownership_temporal_prior_weight,
    )
    parser.add_argument(
        "--aqr-ownership-prior-uniform-mix",
        type=float,
        default=_SPEC_DEFAULTS.aqr_ownership_prior_uniform_mix,
    )
    parser.add_argument(
        "--aqr-same-role-support-competition-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_same_role_support_competition_enabled,
        help=(
            "Apply role-local evidence competition among same-role physical object files after AQR support reads. "
            "This is a measurement-routing invariant, not a loss."
        ),
    )
    parser.add_argument(
        "--aqr-same-role-support-competition-weight",
        type=float,
        default=_SPEC_DEFAULTS.aqr_same_role_support_competition_weight,
    )
    parser.add_argument(
        "--aqr-same-role-support-competition-iters",
        type=int,
        default=_SPEC_DEFAULTS.aqr_same_role_support_competition_iters,
    )
    parser.add_argument(
        "--aqr-same-role-support-competition-physical-only",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_same_role_support_competition_physical_only,
    )
    parser.add_argument(
        "--aqr-active-slot-filter-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_active_slot_filter_enabled,
        help=(
            "Allow only distinct high-confidence same-role AQR anchors to participate in obs/task "
            "assignment; redundant anchors remain as inactive/dustbin candidates."
        ),
    )
    parser.add_argument("--aqr-active-slot-min-per-role", type=int, default=_SPEC_DEFAULTS.aqr_active_slot_min_per_role)
    parser.add_argument("--aqr-active-slot-max-per-role", type=int, default=_SPEC_DEFAULTS.aqr_active_slot_max_per_role)
    parser.add_argument("--aqr-active-slot-min-confidence", type=float, default=_SPEC_DEFAULTS.aqr_active_slot_min_confidence)
    parser.add_argument(
        "--aqr-active-slot-overlap-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_active_slot_overlap_threshold,
    )
    parser.add_argument(
        "--aqr-active-slot-relative-score-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_active_slot_relative_score_threshold,
        help=(
            "Role-local evidence threshold for object-conditional active selection. "
            "A non-minimum same-role anchor whose score is below this fraction of the role-best score "
            "is treated as a dustbin candidate."
        ),
    )
    parser.add_argument(
        "--aqr-active-slot-geometry-duplicate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_enabled,
        help="Use point/posterior geometry proximity as an additional same-role duplicate signal for active-slot dustbinning.",
    )
    parser.add_argument(
        "--aqr-active-slot-geometry-duplicate-sigma-m",
        type=float,
        default=_SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_sigma_m,
    )
    parser.add_argument(
        "--aqr-active-slot-geometry-duplicate-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_active_slot_geometry_duplicate_threshold,
    )
    parser.add_argument(
        "--vcap-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.vcap_enabled,
        help=(
            "Enable the disabled-by-default Variable-Cardinality Active Proposal allocator. "
            "VCAP only initializes active AQR proposal rows; it must not replace dense memory or posterior correction."
        ),
    )
    parser.add_argument("--vcap-max-active", type=int, default=_SPEC_DEFAULTS.vcap_max_active)
    parser.add_argument("--vcap-min-active", type=int, default=_SPEC_DEFAULTS.vcap_min_active)
    parser.add_argument("--vcap-stop-threshold", type=float, default=_SPEC_DEFAULTS.vcap_stop_threshold)
    parser.add_argument("--vcap-action-grad-scale", type=float, default=_SPEC_DEFAULTS.vcap_action_grad_scale)
    parser.add_argument(
        "--aqr-context-slot-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_context_slot_enabled,
        help=(
            "Route inactive-but-real scene anchors as low-weight action context instead of forcing "
            "every non-active anchor into a zero-weight reserve/dustbin path."
        ),
    )
    parser.add_argument("--aqr-context-slot-weight", type=float, default=_SPEC_DEFAULTS.aqr_context_slot_weight)
    parser.add_argument(
        "--aqr-context-slot-min-confidence",
        type=float,
        default=_SPEC_DEFAULTS.aqr_context_slot_min_confidence,
    )
    parser.add_argument("--aqr-context-slot-min-score", type=float, default=_SPEC_DEFAULTS.aqr_context_slot_min_score)
    parser.add_argument(
        "--aqr-context-slot-duplicate-overlap-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_context_slot_duplicate_overlap_threshold,
    )
    parser.add_argument(
        "--aqr-context-slot-deduplicate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_context_slot_deduplicate_enabled,
        help=(
            "Expose only distinct low-priority context anchors to the control graph. "
            "Dense typed memory is unchanged; duplicate context rows remain reserve capacity."
        ),
    )
    parser.add_argument(
        "--aqr-context-slot-max-per-role",
        type=int,
        default=_SPEC_DEFAULTS.aqr_context_slot_max_per_role,
        help="Maximum duplicate-suppressed context anchors per role; <=0 means no explicit cap.",
    )
    parser.add_argument(
        "--aqr-context-slot-self-overlap-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_context_slot_self_overlap_threshold,
        help="Suppress context-context rows whose object-core/geometry overlap exceeds this threshold.",
    )
    parser.add_argument(
        "--aqr-context-slot-self-support-overlap-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_context_slot_self_support_overlap_enabled,
        help=(
            "Also suppress context-context rows whose full support maps overlap strongly. "
            "This catches diffuse duplicate context rows that object-core overlap intentionally tolerates."
        ),
    )
    parser.add_argument(
        "--aqr-context-slot-self-support-overlap-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_context_slot_self_support_overlap_threshold,
        help="Full-support overlap threshold for context-context duplicate suppression.",
    )
    parser.add_argument(
        "--aqr-context-slot-active-support-overlap-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_context_slot_active_support_overlap_enabled,
        help=(
            "Also suppress context rows whose full support overlaps an already-active object owner. "
            "This keeps duplicate owner rows out of the control graph without deleting dense typed memory."
        ),
    )
    parser.add_argument(
        "--aqr-context-slot-active-support-overlap-threshold",
        type=float,
        default=_SPEC_DEFAULTS.aqr_context_slot_active_support_overlap_threshold,
        help="Full-support overlap threshold for context rows compared against active owners.",
    )
    parser.add_argument(
        "--aqr-context-slot-quality-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_context_slot_quality_gate_enabled,
        help=(
            "Opt-in ablation: multiply accepted context rows by learned slot-quality context weights. "
            "The maintained default keeps deterministic context evidence visible and only gates active rows by slot quality."
        ),
    )
    parser.add_argument(
        "--aqr-slot-quality-owner-active-floor",
        type=float,
        default=_SPEC_DEFAULTS.aqr_slot_quality_owner_active_floor,
        help=(
            "Minimum active-quality contribution for rows with accepted owner-candidate evidence. "
            "This prevents QASA-style quality from suppressing a valid contact/motion owner measurement during early training."
        ),
    )
    parser.add_argument(
        "--aqr-control-graph-attention-bias-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_control_graph_attention_bias_enabled,
        help=(
            "Use active/context/reserve downstream weights as graph-token attention priors "
            "inside the control world and downstream PI0.5 readers."
        ),
    )
    parser.add_argument(
        "--aqr-control-graph-token-scaling-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_control_graph_token_scaling_enabled,
        help=(
            "Legacy opt-in: multiply control graph token embeddings by downstream weight. "
            "The maintained path keeps embeddings intact and uses attention bias instead."
        ),
    )
    parser.add_argument(
        "--aqr-control-graph-state-embedding-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_control_graph_state_embedding_enabled,
    )
    parser.add_argument("--aqr-control-graph-bias-min", type=float, default=_SPEC_DEFAULTS.aqr_control_graph_bias_min)
    parser.add_argument(
        "--posterior-owner-active-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_owner_active_gate_enabled,
        help=(
            "Carry active-slot owner/reserve selection into posterior binding so inactive "
            "reserve observation anchors route to dustbin instead of object files."
        ),
    )
    parser.add_argument("--posterior-owner-active-min", type=float, default=_SPEC_DEFAULTS.posterior_owner_active_min)
    parser.add_argument("--posterior-owner-active-bias", type=float, default=_SPEC_DEFAULTS.posterior_owner_active_bias)
    parser.add_argument(
        "--aqr-vjepa-temporal-mode",
        default=_SPEC_DEFAULTS.aqr_vjepa_temporal_mode,
        choices=["disabled", "last_only", "last_two_tokens", "last_mean_delta", "last4_tokens"],
        help="Controls the typed V-JEPA temporal support path. last_two_tokens is the OWM default.",
    )
    parser.add_argument("--aqr-vjepa-temporal-tokens", type=int, default=_SPEC_DEFAULTS.aqr_vjepa_temporal_tokens)
    parser.add_argument(
        "--aqr-vjepa-temporal-include-delta",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.aqr_vjepa_temporal_include_delta,
        help="Append a recent-frame delta token map to the explicit V-JEPA temporal support memory.",
    )
    parser.add_argument(
        "--vjepa-multiview-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.vjepa_multiview_enabled,
        help="Encode static and wrist/gripper V-JEPA clip buffers as typed temporal views.",
    )
    parser.add_argument("--vjepa-max-views", type=int, default=_SPEC_DEFAULTS.vjepa_max_views)
    parser.add_argument(
        "--evidence-cache-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.evidence_cache_enabled,
        help="Enable the posterior-grounded fixed-ring evidence cache. The cache is read only from previous carry.",
    )
    parser.add_argument("--evidence-cache-len", type=int, default=_SPEC_DEFAULTS.evidence_cache_len)
    parser.add_argument("--evidence-cache-read-weight", type=float, default=_SPEC_DEFAULTS.evidence_cache_read_weight)
    parser.add_argument(
        "--evidence-cache-innovation-downweight",
        type=float,
        default=_SPEC_DEFAULTS.evidence_cache_innovation_downweight,
    )
    parser.add_argument("--evidence-cache-address-weight", type=float, default=_SPEC_DEFAULTS.evidence_cache_address_weight)
    parser.add_argument("--evidence-cache-content-weight", type=float, default=_SPEC_DEFAULTS.evidence_cache_content_weight)
    parser.add_argument("--evidence-cache-role-weight", type=float, default=_SPEC_DEFAULTS.evidence_cache_role_weight)
    parser.add_argument(
        "--tracklet-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.tracklet_memory_enabled,
    )
    parser.add_argument("--tracklet-max-tokens", type=int, default=_SPEC_DEFAULTS.tracklet_max_tokens)
    parser.add_argument("--tracklet-confidence-floor", type=float, default=_SPEC_DEFAULTS.tracklet_confidence_floor)
    parser.add_argument("--tracklet-read-weight", type=float, default=_SPEC_DEFAULTS.tracklet_read_weight)
    parser.add_argument(
        "--proposal-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.proposal_memory_enabled,
        help=(
            "Enable optional pseudo-proposal typed memory. Default is off for production "
            "because blind automatic mask proposals were noisy in task-object binding diagnostics; "
            "turn this on only for contact/task-guided proposal sidecars or explicit ablations."
        ),
    )
    parser.add_argument("--proposal-max-tokens", type=int, default=_SPEC_DEFAULTS.proposal_max_tokens)
    parser.add_argument("--proposal-confidence-floor", type=float, default=_SPEC_DEFAULTS.proposal_confidence_floor)
    parser.add_argument("--proposal-read-weight", type=float, default=_SPEC_DEFAULTS.proposal_read_weight)
    parser.add_argument("--proposal-age-decay-steps", type=float, default=_SPEC_DEFAULTS.proposal_age_decay_steps)
    parser.add_argument(
        "--proposal-shape-quality-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.proposal_shape_quality_enabled,
        help="Softly downweight wall/edge/robot-fragment sidecar proposals before they affect proposal reads or task-owner point bridges.",
    )
    parser.add_argument("--proposal-shape-area-min", type=float, default=_SPEC_DEFAULTS.proposal_shape_area_min)
    parser.add_argument("--proposal-shape-area-max", type=float, default=_SPEC_DEFAULTS.proposal_shape_area_max)
    parser.add_argument("--proposal-shape-aspect-min", type=float, default=_SPEC_DEFAULTS.proposal_shape_aspect_min)
    parser.add_argument("--proposal-context-quality-power", type=float, default=_SPEC_DEFAULTS.proposal_context_quality_power)
    parser.add_argument("--proposal-point-bridge-weight", type=float, default=_SPEC_DEFAULTS.proposal_point_bridge_weight)
    parser.add_argument("--proposal-point-bridge-edge-tau", type=float, default=_SPEC_DEFAULTS.proposal_point_bridge_edge_tau)
    parser.add_argument("--proposal-mask-point-tau", type=float, default=_SPEC_DEFAULTS.proposal_mask_point_tau)
    parser.add_argument(
        "--proposal-anchor-seed-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.proposal_anchor_seed_enabled,
        help=(
            "Turn inspected task/contact proposals into bounded physical-row reference point priors. "
            "This is a reference-query transport path, not a hard label or dense-token pruning."
        ),
    )
    parser.add_argument("--proposal-anchor-seed-rows", type=int, default=_SPEC_DEFAULTS.proposal_anchor_seed_rows)
    parser.add_argument(
        "--proposal-anchor-seed-pre-reader-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.proposal_anchor_seed_pre_reader_enabled,
        help=(
            "Apply accepted proposal/mask seed priors before the point reader so object slots are "
            "conditioned like modern object-centric initializers instead of being pulled back only after dense attention."
        ),
    )
    parser.add_argument("--proposal-anchor-seed-weight", type=float, default=_SPEC_DEFAULTS.proposal_anchor_seed_weight)
    parser.add_argument("--proposal-anchor-seed-token-weight", type=float, default=_SPEC_DEFAULTS.proposal_anchor_seed_token_weight)
    parser.add_argument("--proposal-anchor-seed-score-floor", type=float, default=_SPEC_DEFAULTS.proposal_anchor_seed_score_floor)
    parser.add_argument("--proposal-anchor-seed-point-topk", type=int, default=_SPEC_DEFAULTS.proposal_anchor_seed_point_topk)
    parser.add_argument("--proposal-anchor-seed-point-power", type=float, default=_SPEC_DEFAULTS.proposal_anchor_seed_point_power)
    parser.add_argument(
        "--object-candidate-assignment-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.object_candidate_assignment_enabled,
        help=(
            "Let inspected sidecar proposal masks compete for physical object slots with a background residual. "
            "This is soft object-candidate measurement routing, not a hard mask label."
        ),
    )
    parser.add_argument(
        "--object-candidate-assignment-temperature",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_assignment_temperature,
    )
    parser.add_argument("--object-candidate-background-prior", type=float, default=_SPEC_DEFAULTS.object_candidate_background_prior)
    parser.add_argument(
        "--object-candidate-background-quality-weight",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_background_quality_weight,
    )
    parser.add_argument(
        "--object-candidate-row-support-floor",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_row_support_floor,
    )
    parser.add_argument(
        "--object-candidate-eligible-roles",
        type=str,
        default=",".join(str(role) for role in _SPEC_DEFAULTS.object_candidate_eligible_roles),
        help=(
            "Comma-separated physical roles allowed to explain sidecar object candidates. "
            "Default 1,2 means task-object rows plus contact/interaction rows; role 0 effector "
            "is intentionally excluded from object ownership."
        ),
    )
    parser.add_argument(
        "--object-candidate-max-rows-per-candidate",
        type=int,
        default=_SPEC_DEFAULTS.object_candidate_max_rows_per_candidate,
        help=(
            "Maximum physical rows allowed to explain one proposal candidate. "
            "Default 2 permits one task-object owner plus one contact/interaction bridge while still preventing raw clones."
        ),
    )
    parser.add_argument(
        "--object-candidate-row-capacity",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_row_capacity,
        help="Soft capacity on total proposal-candidate mass per physical row before background residual absorbs overflow.",
    )
    parser.add_argument(
        "--object-candidate-row-capacity-iters",
        type=int,
        default=_SPEC_DEFAULTS.object_candidate_row_capacity_iters,
        help="Number of row-capacity normalization passes for proposal-candidate assignment.",
    )
    parser.add_argument("--object-candidate-point-weight", type=float, default=_SPEC_DEFAULTS.object_candidate_point_weight)
    parser.add_argument("--object-candidate-proposal-weight", type=float, default=_SPEC_DEFAULTS.object_candidate_proposal_weight)
    parser.add_argument("--object-candidate-seed-weight", type=float, default=_SPEC_DEFAULTS.object_candidate_seed_weight)
    parser.add_argument("--object-candidate-task-owner-weight", type=float, default=_SPEC_DEFAULTS.object_candidate_task_owner_weight)
    parser.add_argument("--object-candidate-anchor-score-weight", type=float, default=_SPEC_DEFAULTS.object_candidate_anchor_score_weight)
    parser.add_argument("--object-candidate-point-mix", type=float, default=_SPEC_DEFAULTS.object_candidate_point_mix)
    parser.add_argument("--object-candidate-proposal-mix", type=float, default=_SPEC_DEFAULTS.object_candidate_proposal_mix)
    parser.add_argument("--object-candidate-min-shape-quality", type=float, default=_SPEC_DEFAULTS.object_candidate_min_shape_quality)
    parser.add_argument(
        "--object-candidate-owner-transport-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.object_candidate_owner_transport_enabled,
        help=(
            "Transport accepted object/contact candidates into role-1 object-owner geometry. "
            "This keeps role-2 as contact bridge without letting it replace the object file."
        ),
    )
    parser.add_argument(
        "--object-candidate-owner-roles",
        type=str,
        default=",".join(str(role) for role in _SPEC_DEFAULTS.object_candidate_owner_roles),
        help="Comma-separated physical roles that may receive object-owner transport. Default 1.",
    )
    parser.add_argument(
        "--object-candidate-owner-min-share",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_owner_min_share,
        help="Minimum covered candidate mass copied into the selected object-owner row.",
    )
    parser.add_argument(
        "--object-candidate-owner-point-mix",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_owner_point_mix,
        help="Mix ratio for owner transport point priors into the candidate point prior.",
    )
    parser.add_argument(
        "--object-candidate-owner-geometry-mix",
        type=float,
        default=_SPEC_DEFAULTS.object_candidate_owner_geometry_mix,
        help=(
            "Mix accepted owner-candidate point-mask geometry into graph anchor geometry before active selection. "
            "This turns inspected sidecar masks into soft measurement centers instead of mere support priors."
        ),
    )
    parser.add_argument(
        "--posterior-owner-transport-candidate-geometry-mix",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_candidate_geometry_mix,
        help="Mix accepted owner-candidate geometry into posterior owner transport measurements.",
    )
    parser.add_argument(
        "--object-explanation-point-core-mass",
        type=float,
        default=_SPEC_DEFAULTS.object_explanation_point_core_mass,
        help=(
            "Fraction of each object-mask row mass used for OEML point compactness. "
            "This keeps the auxiliary compactness target robust to weak sidecar tails without pruning dense memory."
        ),
    )
    parser.add_argument(
        "--object-explanation-point-core-topk",
        type=int,
        default=_SPEC_DEFAULTS.object_explanation_point_core_topk,
        help="Maximum point tokens used for OEML point compactness per row; 0 disables the top-k cap.",
    )
    parser.add_argument(
        "--task-owner-proposal-point-bridge-weight",
        type=float,
        default=_SPEC_DEFAULTS.task_owner_proposal_point_bridge_weight,
    )
    parser.add_argument(
        "--task-owner-bias-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.task_owner_bias_enabled,
        help="Transfer task-query visual support into physical scene object/proposal reads as soft measurement bias.",
    )
    parser.add_argument("--task-owner-visual-bias-weight", type=float, default=_SPEC_DEFAULTS.task_owner_visual_bias_weight)
    parser.add_argument("--task-owner-proposal-bias-weight", type=float, default=_SPEC_DEFAULTS.task_owner_proposal_bias_weight)
    parser.add_argument(
        "--task-owner-proposal-point-bias-weight",
        type=float,
        default=_SPEC_DEFAULTS.task_owner_proposal_point_bias_weight,
    )
    parser.add_argument(
        "--task-owner-proposal-objectness-power",
        type=float,
        default=_SPEC_DEFAULTS.task_owner_proposal_objectness_power,
    )
    parser.add_argument(
        "--task-owner-proposal-static-only",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.task_owner_proposal_static_only,
        help="Restrict task-owner proposal projection to static-view proposals unless view geometry is explicitly available.",
    )
    parser.add_argument("--task-owner-proposal-topk", type=int, default=_SPEC_DEFAULTS.task_owner_proposal_topk)
    parser.add_argument("--task-owner-proposal-score-floor", type=float, default=_SPEC_DEFAULTS.task_owner_proposal_score_floor)
    parser.add_argument(
        "--mvtrack-sidecar-root",
        type=str,
        default=None,
        help=(
            "Optional root containing per-frame MVTrack sidecar npz files with tracklet_* "
            "and/or proposal_* arrays. Supported layouts: <root>/<split>/episode_XXXXXXX.npz "
            "or <root>/episode_XXXXXXX.npz. Proposal generation stays offline and sidecar-only."
        ),
    )
    parser.add_argument(
        "--allow-legacy-blind-sam-sidecar",
        action="store_true",
        default=False,
        help=(
            "Historical reproduction only. Allows mvtrack sidecar roots whose names look like "
            "archived blind automatic SAM proposal outputs. Do not use for current training."
        ),
    )
    parser.add_argument(
        "--mvtrack-sidecar-proposal-nearest-max-gap",
        type=int,
        default=0,
        help=(
            "Allow sparse proposal sidecars to serve the nearest proposal frame within this many CALVIN steps. "
            "Borrowed proposal objectness is exponentially decayed by proposal_age_decay_steps and recorded as proposal_age."
        ),
    )
    parser.add_argument("--bind-support-signature-weight", type=float, default=_SPEC_DEFAULTS.bind_support_signature_weight)
    parser.add_argument("--bind-embedding-signature-weight", type=float, default=_SPEC_DEFAULTS.bind_embedding_signature_weight)
    parser.add_argument("--bind-quadratic-signature-weight", type=float, default=_SPEC_DEFAULTS.bind_quadratic_signature_weight)
    parser.add_argument("--bind-low-rank-signature-weight", type=float, default=_SPEC_DEFAULTS.bind_low_rank_signature_weight)
    parser.add_argument("--binding-signature-dim", type=int, default=_SPEC_DEFAULTS.binding_signature_dim)
    parser.add_argument("--binding-low-rank-signature-rank", type=int, default=_SPEC_DEFAULTS.binding_low_rank_signature_rank)
    parser.add_argument(
        "--binding-signature-centering-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.binding_signature_centering_enabled,
        help="Center projected typed-memory binding keys before support pooling to remove scene/modality common-mode.",
    )
    parser.add_argument(
        "--binding-signature-centering-min-tokens",
        type=int,
        default=_SPEC_DEFAULTS.binding_signature_centering_min_tokens,
    )
    parser.add_argument(
        "--binding-signature-score-calibration-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.binding_signature_score_calibration_enabled,
        help="Calibrate pairwise binding-signature scores into relative IsSameObject assignment logits.",
    )
    parser.add_argument(
        "--binding-signature-score-calibration-mode",
        type=str,
        default=_SPEC_DEFAULTS.binding_signature_score_calibration_mode,
        choices=("double_center_zscore", "double_center", "row_zscore", "row_center", "global_center", "zscore"),
    )
    parser.add_argument(
        "--binding-signature-score-min-std",
        type=float,
        default=_SPEC_DEFAULTS.binding_signature_score_min_std,
    )
    parser.add_argument(
        "--binding-signature-score-clip",
        type=float,
        default=_SPEC_DEFAULTS.binding_signature_score_clip,
    )
    parser.add_argument(
        "--posterior-binding-signature-memory-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_binding_signature_memory_enabled,
        help="Keep posterior object-file binding signatures as a trusted-memory state instead of overwriting them every frame.",
    )
    parser.add_argument(
        "--posterior-binding-signature-update-rate",
        type=float,
        default=_spec_default("posterior_binding_signature_update_rate"),
    )
    parser.add_argument(
        "--posterior-binding-signature-update-max-rate",
        type=float,
        default=_spec_default("posterior_binding_signature_update_max_rate"),
    )
    parser.add_argument(
        "--posterior-binding-signature-min-support",
        type=float,
        default=_spec_default("posterior_binding_signature_min_support"),
    )
    parser.add_argument(
        "--posterior-binding-signature-owner-weight",
        type=float,
        default=_spec_default("posterior_binding_signature_owner_weight"),
    )
    parser.add_argument(
        "--posterior-binding-signature-dispersion-gate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_binding_signature_dispersion_gate_enabled,
        help=(
            "Require the current binding-signature measurement to have relative IsSameObject "
            "dispersion before it can update posterior file identity memory."
        ),
    )
    parser.add_argument(
        "--posterior-binding-signature-measurement-min-std",
        type=float,
        default=_spec_default("posterior_binding_signature_measurement_min_std"),
    )
    parser.add_argument(
        "--posterior-binding-signature-measurement-margin-min",
        type=float,
        default=_spec_default("posterior_binding_signature_measurement_margin_min"),
    )
    parser.add_argument(
        "--posterior-binding-signature-measurement-margin-temperature",
        type=float,
        default=_spec_default("posterior_binding_signature_measurement_margin_temperature"),
    )
    parser.add_argument("--bind-address-weight", type=float, default=_SPEC_DEFAULTS.bind_address_weight)
    parser.add_argument(
        "--bind-address-innovation-downweight",
        type=float,
        default=_SPEC_DEFAULTS.bind_address_innovation_downweight,
    )
    parser.add_argument("--address-update-rate", type=float, default=_SPEC_DEFAULTS.address_update_rate)
    parser.add_argument("--address-update-max-rate", type=float, default=_SPEC_DEFAULTS.address_update_max_rate)
    parser.add_argument(
        "--posterior-occupancy-prior-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_occupancy_prior_enabled,
        help=(
            "Enable the same-role posterior object-file occupancy prior. "
            "This label-free measurement prior prevents multiple same-role posterior rows "
            "from correcting to the same broad observation centroid."
        ),
    )
    parser.add_argument(
        "--posterior-occupancy-prior-weight",
        type=float,
        default=_SPEC_DEFAULTS.posterior_occupancy_prior_weight,
    )
    parser.add_argument(
        "--posterior-occupancy-prior-sigma-m",
        type=float,
        default=_SPEC_DEFAULTS.posterior_occupancy_prior_sigma_m,
    )
    parser.add_argument(
        "--posterior-occupancy-prior-clip",
        type=float,
        default=_SPEC_DEFAULTS.posterior_occupancy_prior_clip,
    )
    parser.add_argument(
        "--observation-anchor-seed-point-mix",
        type=float,
        default=_SPEC_DEFAULTS.observation_anchor_seed_point_mix,
        help=(
            "Mix seed-point one-hot coverage back into observation-anchor point weights. "
            "This keeps same-role observation hypotheses spatially distinct before posterior association."
        ),
    )
    parser.add_argument(
        "--recycle-normalize-residual-summary",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.recycle_normalize_residual_summary,
        help=(
            "Normalize the dustbin residual summary before the recycle gate. "
            "This keeps reset probability from being driven by unbounded residual magnitude."
        ),
    )
    parser.add_argument(
        "--recycle-residual-norm-mode",
        choices=("layernorm", "rmsnorm", "none"),
        default=_SPEC_DEFAULTS.recycle_residual_norm_mode,
        help=(
            "Recycle residual-summary normalization family. layernorm is the verified default; "
            "rmsnorm is the conservative ablation that preserves residual mean/DC; none is diagnostic only."
        ),
    )
    parser.add_argument("--recycle-logit-clamp", type=float, default=_SPEC_DEFAULTS.recycle_logit_clamp)
    parser.add_argument(
        "--posterior-slotwise-recycle-residual",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_slotwise_recycle_residual,
        help=(
            "Use each posterior slot's own raw measurement mixture for recycle/reset residuals. "
            "The default avoids resetting multiple same-role object files from one global dustbin vector."
        ),
    )
    parser.add_argument(
        "--posterior-lifecycle-calibration-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_lifecycle_calibration_enabled,
        help=(
            "Factor posterior survival/reset/dustbin calibration from raw recycle logits. "
            "Stable high-support low-innovation object files are protected from reset."
        ),
    )
    parser.add_argument(
        "--posterior-lifecycle-support-min",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_support_min,
    )
    parser.add_argument(
        "--posterior-lifecycle-support-temperature",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_support_temperature,
    )
    parser.add_argument(
        "--posterior-lifecycle-margin-min",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_margin_min,
    )
    parser.add_argument(
        "--posterior-lifecycle-margin-temperature",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_margin_temperature,
    )
    parser.add_argument(
        "--posterior-lifecycle-entropy-weight",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_entropy_weight,
    )
    parser.add_argument(
        "--posterior-lifecycle-owner-weight",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_owner_weight,
    )
    parser.add_argument(
        "--posterior-lifecycle-innovation-downweight",
        type=float,
        default=_SPEC_DEFAULTS.posterior_lifecycle_innovation_downweight,
    )
    parser.add_argument(
        "--posterior-owner-transport-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_owner_transport_enabled,
        help=(
            "Close graph/object-owner responsibility into posterior object-file geometry. "
            "This is the belief-write closure for sidecar/contact object masks, not an auxiliary loss."
        ),
    )
    parser.add_argument(
        "--posterior-owner-transport-roles",
        type=str,
        default=",".join(str(role) for role in _SPEC_DEFAULTS.posterior_owner_transport_roles),
        help="Comma-separated posterior physical roles allowed to accept object-owner transported geometry.",
    )
    parser.add_argument(
        "--posterior-owner-transport-max-per-role",
        type=int,
        default=_SPEC_DEFAULTS.posterior_owner_transport_max_per_role,
        help="Maximum posterior files per role that may accept transported owner geometry in one update.",
    )
    parser.add_argument(
        "--posterior-owner-transport-max-rate",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_max_rate,
        help="Maximum confidence cap for transported owner geometry before precision fusion.",
    )
    parser.add_argument(
        "--posterior-owner-transport-precision-gain",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_precision_gain,
        help=(
            "Precision multiplier for accepted owner geometry. "
            "This makes sidecar/contact ownership a high-precision posterior measurement rather than a weak convex interpolation."
        ),
    )
    parser.add_argument(
        "--posterior-owner-transport-min-mass",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_min_mass,
        help="Minimum transported owner mass required before posterior geometry can be rewritten.",
    )
    parser.add_argument(
        "--posterior-owner-transport-direct-candidate-assignment",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_owner_transport_direct_candidate_assignment,
        help=(
            "Assign validated graph-owner candidates directly to posterior files before owner geometry is averaged. "
            "This is the slot-responsibility write-back path; the obs-averaged transport remains fallback."
        ),
    )
    parser.add_argument(
        "--posterior-owner-transport-direct-candidate-min-score",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_direct_candidate_min_score,
        help="Minimum candidate-to-file responsibility score required for direct owner write-through.",
    )
    parser.add_argument(
        "--posterior-owner-transport-assignment-floor",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_assignment_floor,
        help="Lifecycle assignment confidence floor for accepting transported owner geometry.",
    )
    parser.add_argument(
        "--posterior-owner-transport-reliability-floor",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_reliability_floor,
        help="Lifecycle owner reliability floor for accepting transported owner geometry.",
    )
    parser.add_argument(
        "--posterior-owner-transport-covariance-scale",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_covariance_scale,
        help="Covariance scale applied to transported owner geometry before posterior fusion.",
    )
    parser.add_argument(
        "--posterior-owner-transport-inactive-prior",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_inactive_prior,
        help=(
            "Soft prior for an inactive file that receives transported owner responsibility. "
            "This prevents lifecycle gates from hard-blocking a newly explained object file."
        ),
    )
    parser.add_argument(
        "--posterior-owner-transport-activates-file",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_owner_transport_activates_file,
        help=(
            "Allow transported owner responsibility to make the selected posterior file action-visible "
            "before the final per-role file cap."
        ),
    )
    parser.add_argument(
        "--posterior-owner-transport-active-threshold",
        type=float,
        default=_SPEC_DEFAULTS.posterior_owner_transport_active_threshold,
        help="Owner-transport confidence that maps to a full downstream active file gate.",
    )
    parser.add_argument(
        "--posterior-file-competition-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_file_competition_enabled,
        help=(
            "Demote duplicate same-role posterior object-file assignments into no-object/dustbin before posterior write. "
            "This is an object-file explaining-away step, not an auxiliary loss."
        ),
    )
    parser.add_argument(
        "--posterior-file-competition-min-per-role",
        type=int,
        default=_SPEC_DEFAULTS.posterior_file_competition_min_per_role,
    )
    parser.add_argument(
        "--posterior-file-competition-max-per-role",
        type=int,
        default=_SPEC_DEFAULTS.posterior_file_competition_max_per_role,
        help="Maximum active posterior files per role after duplicate demotion; 0 disables this cap.",
    )
    parser.add_argument(
        "--posterior-file-competition-min-support",
        type=float,
        default=_SPEC_DEFAULTS.posterior_file_competition_min_support,
    )
    parser.add_argument(
        "--posterior-file-competition-relative-score-threshold",
        type=float,
        default=_SPEC_DEFAULTS.posterior_file_competition_relative_score_threshold,
    )
    parser.add_argument(
        "--posterior-file-competition-support-overlap-threshold",
        type=float,
        default=_SPEC_DEFAULTS.posterior_file_competition_support_overlap_threshold,
    )
    parser.add_argument(
        "--posterior-file-competition-geometry-duplicate-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_file_competition_geometry_duplicate_enabled,
    )
    parser.add_argument(
        "--posterior-file-competition-geometry-sigma-m",
        type=float,
        default=_SPEC_DEFAULTS.posterior_file_competition_geometry_sigma_m,
    )
    parser.add_argument(
        "--posterior-file-competition-geometry-threshold",
        type=float,
        default=_SPEC_DEFAULTS.posterior_file_competition_geometry_threshold,
    )
    parser.add_argument(
        "--posterior-birth-competition-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_birth_competition_enabled,
        help=(
            "Select a bounded number of inactive reserve files that may consume demoted dustbin evidence. "
            "This prevents one no-object residual from being broadcast into every recycled posterior file."
        ),
    )
    parser.add_argument(
        "--posterior-birth-competition-max-per-role",
        type=int,
        default=_SPEC_DEFAULTS.posterior_birth_competition_max_per_role,
    )
    parser.add_argument(
        "--posterior-birth-competition-min-score",
        type=float,
        default=_SPEC_DEFAULTS.posterior_birth_competition_min_score,
    )
    parser.add_argument(
        "--posterior-birth-competition-inactive-only",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.posterior_birth_competition_inactive_only,
    )
    parser.add_argument(
        "--posterior-birth-alpha-suppression-power",
        type=float,
        default=_SPEC_DEFAULTS.posterior_birth_alpha_suppression_power,
    )
    parser.add_argument(
        "--legacy-local-refinement-opt-in",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.legacy_local_refinement_opt_in,
        help=(
            "Archived ablation-only opt-in for the legacy local top-k reread. "
            "Production ignores --local-refinement-enabled unless this flag is also true."
        ),
    )
    parser.add_argument(
        "--local-refinement-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.local_refinement_enabled,
        help=(
            "Legacy archived local top-k reread switch. It remains inactive unless "
            "--legacy-local-refinement-opt-in is also set."
        ),
    )
    parser.add_argument("--local-refinement-topk", type=int, default=_SPEC_DEFAULTS.local_refinement_topk)
    parser.add_argument(
        "--local-refinement-weight",
        type=float,
        default=_SPEC_DEFAULTS.local_refinement_weight,
        help="Legacy archived local top-k reread residual weight; production default is 0.0.",
    )
    parser.add_argument(
        "--local-refinement-binding-weight",
        type=float,
        default=_SPEC_DEFAULTS.local_refinement_binding_weight,
        help=(
            "Archived optional same-object subspace reranking strength for legacy local refinement top-k. "
            "This uses the existing binding_signature projection and does not add a new loss."
        ),
    )
    parser.add_argument(
        "--slot-jepa-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.slot_jepa_enabled,
        help="Materialize slot-level predictive states. Its auxiliary loss remains controlled by lambda-slot-jepa.",
    )
    parser.add_argument(
        "--support-prediction-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.support_prediction_enabled,
    )
    parser.add_argument(
        "--ordinal-relation-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.ordinal_relation_enabled,
    )
    parser.add_argument(
        "--ordinal-weak-target-enabled",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.ordinal_weak_target_enabled,
    )
    parser.add_argument("--aqr-obs-gate-init", type=float, default=_SPEC_DEFAULTS.aqr_obs_gate_init)
    parser.add_argument("--aqr-task-gate-init", type=float, default=_SPEC_DEFAULTS.aqr_task_gate_init)
    parser.add_argument("--aqr-posterior-gate-init", type=float, default=_SPEC_DEFAULTS.aqr_posterior_gate_init)
    parser.add_argument("--aqr-control-gate-init", type=float, default=_SPEC_DEFAULTS.aqr_control_gate_init)
    parser.add_argument(
        "--require-pi0-action-generator",
        action=argparse.BooleanOptionalAction,
        default=_SPEC_DEFAULTS.require_pi0_action_generator,
    )
    parser.add_argument("--tokenwise-ff-chunk-size", type=int, default=_SPEC_DEFAULTS.tokenwise_ff_chunk_size)
    parser.add_argument("--semantic-tokenwise-chunk-size", type=int, default=0)
    parser.add_argument("--semantic-projection-chunk-size", type=int, default=None)
    parser.add_argument("--semantic-mlp-chunk-size", type=int, default=None)
    parser.add_argument("--attention-heads", type=int, default=_SPEC_DEFAULTS.attention_heads)
    parser.add_argument("--future-vote-heads", type=int, default=_SPEC_DEFAULTS.future_vote_heads)
    parser.set_defaults(
        wandb_enabled=True,
        progress=True,
        enable_aux_budgeting=True,
        visual_activation_checkpointing=True,
        semantic_gradient_checkpointing=True,
        semantic_use_gripper=True,
    )
    args = parser.parse_args()
    _apply_foundation_profile(args)
    _normalize_train_args(args)
    _validate_train_args(args)
    _validate_backbone_args(args)
    train(args)


if __name__ == "__main__":
    main()

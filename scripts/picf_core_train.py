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
import signal
import shutil
import sys
import time
import traceback
from collections import deque
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from typing import TextIO

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
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
_SPEC_DEFAULTS = PicfCoreConfig()
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
    "core.future_condition_reader.*",
    "core.predictive_semantic_world.*",
    "core.predictive_state_proj.*",
    "semantic_prefix_proj.*",
    "posterior_to_control_proj.*",
    "global_post_to_control_proj.*",
    "innovation_to_control_proj.*",
    "proprio_to_control_proj.*",
    "task_query_tokens",
    "task_global_query_tokens",
    "task_instruction_query_tokens",
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
    "future_condition_reader.*",
    "predictive_semantic_world.*",
    "predictive_state_proj.*",
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
        if physical is None:
            physical = np.zeros_like(target_next)
        if semantic is None:
            semantic = np.zeros_like(target_next)
        compare_size = physical.shape[1], physical.shape[0]
        current_ref = np.asarray(
            Image.fromarray(current).resize(compare_size, resample=Image.Resampling.BILINEAR)
        )
        target_ref = np.asarray(
            Image.fromarray(target_next).resize(compare_size, resample=Image.Resampling.BILINEAR)
        )
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


def _normalize_train_args(args: argparse.Namespace) -> None:
    args.training_strategy = str(getattr(args, "training_strategy", "ddp")).lower().replace("-", "_")
    args.picf_mode = str(getattr(args, "picf_mode", "enabled")).lower().replace("-", "_")
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
    )
    for name in positive_int_fields:
        value = int(getattr(args, name))
        if value < 1:
            raise ValueError(f"{name} must be >= 1, got {value}.")
    if int(args.warmup_steps) < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {args.warmup_steps}.")
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
    if float(args.point_focus_sigma_m) <= 0.0:
        raise ValueError(f"point_focus_sigma_m must be > 0, got {args.point_focus_sigma_m}.")
    if int(args.diagnostic_interval) < 0:
        raise ValueError(f"diagnostic_interval must be >= 0, got {args.diagnostic_interval}.")
    for name in ("point_backbone_lr_scale", "visual_lr_scale", "tactile_lr_scale", "semantic_lr_scale"):
        value = float(getattr(args, name))
        if value <= 0.0:
            raise ValueError(f"{name} must be > 0, got {value}.")
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
        "lambda_focus_pv",
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
    the minority-dtype trainable parameters unsharded via ``ignored_states``.
    This preserves the custom forward while still full-sharding the bulk of the
    semantic parameters.
    """

    if _is_fsdp_model(module):
        return module
    if FullyShardedDataParallel is None:
        raise RuntimeError("training_strategy=fsdp_full_shard requires torch.distributed.fsdp to be available.")

    _promote_non_trainable_nonfloating_params_to_buffers(module)

    trainable_params = [param for param in module.parameters() if bool(getattr(param, "requires_grad", False))]
    if not trainable_params:
        return module

    dtype_numel: dict[torch.dtype, int] = {}
    for param in trainable_params:
        dtype_numel[param.dtype] = dtype_numel.get(param.dtype, 0) + int(param.numel())

    dominant_dtype = max(dtype_numel, key=dtype_numel.get)
    ignored_states = [param for param in trainable_params if param.dtype != dominant_dtype]

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
    if ignored_modules:
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
    policy_forward_sec: float


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
        frame_dt_s: float = 1.0 / 30.0,
        action_normalizer: PicfActionNormalizer | None = None,
        augmentation_mode: str = "off",
        photometric_strength: str = "conservative",
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
        self.frame_dt_s = float(frame_dt_s)
        self.action_normalizer = action_normalizer
        self.augmentation_mode = str(augmentation_mode).lower().replace("-", "_")
        self.photometric_strength = str(photometric_strength).lower().replace("-", "_")
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
        for segment_id, segment in enumerate(self.segments):
            max_start_exclusive = segment.end - (self.unroll_steps + self.action_horizon - 1)
            for step_id in range(segment.start, max_start_exclusive):
                self.window_index.append((segment_id, step_id))
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

    def __len__(self) -> int:
        return len(self.segment_sampling_slots)

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
        frame = self.reader.read_npz(step_id, keys=keys)
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


@dataclasses.dataclass
class _MetricAccumulator:
    loss_total: float = 0.0
    loss_action: float = 0.0
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
    loss_anchor_pv: float = 0.0
    loss_pv_weak: float = 0.0
    loss_focus_pv: float = 0.0
    loss_pt: float = 0.0
    physical_aux_budget_scale: float = 0.0
    semantic_aux_budget_scale: float = 0.0
    alignment_budget_scale: float = 0.0
    candidate_density: float = 0.0
    tactile_contact_prob_mean: float = 0.0
    tactile_active_rate: float = 0.0
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
        self.loss_anchor_pv += float(losses.anchor_pv.item())
        self.loss_pv_weak += float(losses.pv_weak.item())
        self.loss_focus_pv += float(losses.focus_pv.item())
        self.loss_pt += float(losses.pt.item())
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
        self.loss_anchor_pv += float(outputs["loss_anchor_pv"].detach().item())
        self.loss_pv_weak += float(outputs["loss_pv_weak"].detach().item())
        self.loss_focus_pv += float(outputs["loss_focus_pv"].detach().item())
        self.loss_pt += float(outputs["loss_pt"].detach().item())
        self.physical_aux_budget_scale += float(outputs["physical_aux_budget_scale"].detach().item())
        self.semantic_aux_budget_scale += float(outputs["semantic_aux_budget_scale"].detach().item())
        self.alignment_budget_scale += float(outputs["alignment_budget_scale"].detach().item())
        self.candidate_density += float(outputs["projective_candidate_density"].detach().item())
        self.tactile_contact_prob_mean += float(outputs.get("tactile_contact_prob_mean", 0.0).detach().item())
        self.tactile_active_rate += float(outputs.get("tactile_active_rate", 0.0).detach().item())
        self.num_windows += 1

    def averages(self) -> dict[str, float]:
        denom = max(self.num_windows, 1)
        return {
            "loss_total": self.loss_total / denom,
            "loss_action": self.loss_action / denom,
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
            "loss_anchor_pv": self.loss_anchor_pv / denom,
            "loss_pv_weak": self.loss_pv_weak / denom,
            "loss_focus_pv": self.loss_focus_pv / denom,
            "loss_pt": self.loss_pt / denom,
            "physical_aux_budget_scale": self.physical_aux_budget_scale / denom,
            "semantic_aux_budget_scale": self.semantic_aux_budget_scale / denom,
            "alignment_budget_scale": self.alignment_budget_scale / denom,
            "projective_candidate_density": self.candidate_density / denom,
            "tactile_contact_prob_mean": self.tactile_contact_prob_mean / denom,
            "tactile_active_rate": self.tactile_active_rate / denom,
        }


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
    ) -> None:
        super().__init__()
        self.core = core
        self.semantic_encoder = semantic_encoder
        self.picf_mode = str(picf_mode).lower().replace("-", "_")
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
        if current_targets is None or availability is None:
            return None
        return future_targets_from_current_targets(current_targets, availability)

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
            "loss_anchor_pv": losses.anchor_pv,
            "loss_pv_weak": losses.pv_weak,
            "loss_focus_pv": losses.focus_pv,
            "loss_pt": losses.pt,
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
                action_loss_override=policy_forward.flow_override["total"],
                action_pos_override=policy_forward.flow_override["action_pos"],
                action_rot_override=policy_forward.flow_override["action_rot"],
                action_gripper_override=policy_forward.flow_override["action_gripper"],
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
            "loss_anchor_pv": metrics["loss_anchor_pv"] / denom,
            "loss_pv_weak": metrics["loss_pv_weak"] / denom,
            "loss_focus_pv": metrics["loss_focus_pv"] / denom,
            "loss_pt": metrics["loss_pt"] / denom,
            "physical_aux_budget_scale": metrics["physical_aux_budget_scale"] / denom,
            "semantic_aux_budget_scale": metrics["semantic_aux_budget_scale"] / denom,
            "alignment_budget_scale": metrics["alignment_budget_scale"] / denom,
            "projective_candidate_density": metrics["projective_candidate_density"] / denom,
            "tactile_contact_prob_mean": metrics["tactile_contact_prob_mean"] / denom,
            "tactile_active_rate": metrics["tactile_active_rate"] / denom,
        }
        if capture_visual_diagnostics:
            result["diagnostic_physical_visual_real_seq"] = []
            result["diagnostic_semantic_visual_real_seq"] = []
        return result

    def forward(
        self,
        window: _TransitionWindow,
        *,
        capture_visual_diagnostics: bool = False,
        debug_phase_label: str | None = None,
    ) -> dict[str, Any]:
        if not bool(getattr(self.policy, "picf_enabled", True)):
            return self._forward_action_only_window(
                window,
                capture_visual_diagnostics=capture_visual_diagnostics,
                debug_phase_label=debug_phase_label,
            )
        previous = None
        metrics: dict[str, torch.Tensor] | None = None
        totals: list[torch.Tensor] = []
        physical_visual_real_seq: list[torch.Tensor | None] = []
        semantic_visual_real_seq: list[torch.Tensor | None] = []
        pending: _PendingTransitionLoss | None = None
        for index in range(len(window.frames) - 1):
            if debug_phase_label is not None:
                logging.info("%s transition=%s begin", debug_phase_label, index)
            current = dataclasses.replace(window.frames[index], reset_scaffold=(index == 0))
            nxt = dataclasses.replace(window.frames[index + 1], reset_scaffold=False)
            current_visual = _rgb_visual_override(current.rgb_static, grid=self.visual_grid) if self.use_visual_override else None
            next_visual = _rgb_visual_override(nxt.rgb_static, grid=self.visual_grid) if self.use_visual_override else None
            step_start = time.perf_counter()
            action_chunk_target = current.action_chunk if current.action_chunk is not None else current.action
            policy_forward = self.policy.forward_train_transition(
                previous=previous,
                current=current,
                visual_map_override=current_visual,
                action_chunk_target=action_chunk_target,
            )
            output = policy_forward.output
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
                    action_loss_override=None if pending.flow_override is None else pending.flow_override["total"],
                    action_pos_override=None if pending.flow_override is None else pending.flow_override["action_pos"],
                    action_rot_override=None if pending.flow_override is None else pending.flow_override["action_rot"],
                    action_gripper_override=None if pending.flow_override is None else pending.flow_override["action_gripper"],
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
            else:
                losses = None
                metrics_candidate_density = None
                metrics_tactile_contact_prob_mean = None
                metrics_tactile_active_rate = None
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
                policy_forward_sec=policy_forward_sec,
            )
            if metrics is None:
                if losses is None:
                    previous = policy_forward.next_state
                    continue
                metrics = {
                    "loss_action": losses.action,
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
                    "loss_anchor_pv": losses.anchor_pv,
                    "loss_pv_weak": losses.pv_weak,
                    "loss_focus_pv": losses.focus_pv,
                    "loss_pt": losses.pt,
                    "physical_aux_budget_scale": losses.physical_aux_budget_scale,
                    "semantic_aux_budget_scale": losses.semantic_aux_budget_scale,
                    "alignment_budget_scale": losses.alignment_budget_scale,
                    "projective_candidate_density": metrics_candidate_density,
                    "tactile_contact_prob_mean": metrics_tactile_contact_prob_mean,
                    "tactile_active_rate": metrics_tactile_active_rate,
                }
            else:
                if losses is None:
                    previous = policy_forward.next_state
                    continue
                metrics["loss_action"] = metrics["loss_action"] + losses.action
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
                metrics["loss_anchor_pv"] = metrics["loss_anchor_pv"] + losses.anchor_pv
                metrics["loss_pv_weak"] = metrics["loss_pv_weak"] + losses.pv_weak
                metrics["loss_focus_pv"] = metrics["loss_focus_pv"] + losses.focus_pv
                metrics["loss_pt"] = metrics["loss_pt"] + losses.pt
                metrics["physical_aux_budget_scale"] = metrics["physical_aux_budget_scale"] + losses.physical_aux_budget_scale
                metrics["semantic_aux_budget_scale"] = metrics["semantic_aux_budget_scale"] + losses.semantic_aux_budget_scale
                metrics["alignment_budget_scale"] = metrics["alignment_budget_scale"] + losses.alignment_budget_scale
                metrics["projective_candidate_density"] = metrics["projective_candidate_density"] + metrics_candidate_density
                metrics["tactile_contact_prob_mean"] = metrics["tactile_contact_prob_mean"] + metrics_tactile_contact_prob_mean
                metrics["tactile_active_rate"] = metrics["tactile_active_rate"] + metrics_tactile_active_rate
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
                action_loss_override=None if pending.flow_override is None else pending.flow_override["total"],
                action_pos_override=None if pending.flow_override is None else pending.flow_override["action_pos"],
                action_rot_override=None if pending.flow_override is None else pending.flow_override["action_rot"],
                action_gripper_override=None if pending.flow_override is None else pending.flow_override["action_gripper"],
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
                    "loss_anchor_pv": losses.anchor_pv,
                    "loss_pv_weak": losses.pv_weak,
                    "loss_focus_pv": losses.focus_pv,
                    "loss_pt": losses.pt,
                    "physical_aux_budget_scale": losses.physical_aux_budget_scale,
                    "semantic_aux_budget_scale": losses.semantic_aux_budget_scale,
                    "alignment_budget_scale": losses.alignment_budget_scale,
                    "projective_candidate_density": pending.candidate_density,
                    "tactile_contact_prob_mean": pending.tactile_contact_prob_mean,
                    "tactile_active_rate": pending.tactile_active_rate,
                }
            else:
                metrics["loss_action"] = metrics["loss_action"] + losses.action
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
                metrics["loss_anchor_pv"] = metrics["loss_anchor_pv"] + losses.anchor_pv
                metrics["loss_pv_weak"] = metrics["loss_pv_weak"] + losses.pv_weak
                metrics["loss_focus_pv"] = metrics["loss_focus_pv"] + losses.focus_pv
                metrics["loss_pt"] = metrics["loss_pt"] + losses.pt
                metrics["physical_aux_budget_scale"] = metrics["physical_aux_budget_scale"] + losses.physical_aux_budget_scale
                metrics["semantic_aux_budget_scale"] = metrics["semantic_aux_budget_scale"] + losses.semantic_aux_budget_scale
                metrics["alignment_budget_scale"] = metrics["alignment_budget_scale"] + losses.alignment_budget_scale
                metrics["projective_candidate_density"] = metrics["projective_candidate_density"] + pending.candidate_density
                metrics["tactile_contact_prob_mean"] = metrics["tactile_contact_prob_mean"] + pending.tactile_contact_prob_mean
                metrics["tactile_active_rate"] = metrics["tactile_active_rate"] + pending.tactile_active_rate

        assert metrics is not None
        denom = float(len(window.frames) - 1)
        mean_total = torch.stack(totals).mean()
        result: dict[str, Any] = {
            "loss_total": mean_total,
            "loss_action": metrics["loss_action"] / denom,
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
            "loss_anchor_pv": metrics["loss_anchor_pv"] / denom,
            "loss_pv_weak": metrics["loss_pv_weak"] / denom,
            "loss_focus_pv": metrics["loss_focus_pv"] / denom,
            "loss_pt": metrics["loss_pt"] / denom,
            "physical_aux_budget_scale": metrics["physical_aux_budget_scale"] / denom,
            "semantic_aux_budget_scale": metrics["semantic_aux_budget_scale"] / denom,
            "alignment_budget_scale": metrics["alignment_budget_scale"] / denom,
            "projective_candidate_density": metrics["projective_candidate_density"] / denom,
            "tactile_contact_prob_mean": metrics["tactile_contact_prob_mean"] / denom,
            "tactile_active_rate": metrics["tactile_active_rate"] / denom,
        }
        if capture_visual_diagnostics:
            result["diagnostic_physical_visual_real_seq"] = physical_visual_real_seq
            result["diagnostic_semantic_visual_real_seq"] = semantic_visual_real_seq
        return result


_WINDOW_OUTPUT_TENSOR_KEYS: tuple[str, ...] = (
    "loss_total",
    "loss_action",
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
    "loss_anchor_pv",
    "loss_pv_weak",
    "loss_focus_pv",
    "loss_pt",
    "physical_aux_budget_scale",
    "semantic_aux_budget_scale",
    "alignment_budget_scale",
    "projective_candidate_density",
    "tactile_contact_prob_mean",
    "tactile_active_rate",
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


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        scale = float(group.get("lr_scale", 1.0))
        group["lr"] = lr * scale


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
    grad_clip_controller: _GradClipController | None = None,
) -> int:
    module = _unwrap_training_model(model)
    if path.is_dir():
        model_state = torch.load(path / "model.pt", map_location=device, weights_only=False)
        metadata = torch.load(path / "metadata.pt", map_location=device, weights_only=False)
        optimizer_path = path / "optimizer.pt"
        optimizer_state = (
            torch.load(optimizer_path, map_location=device, weights_only=False)
            if optimizer_path.exists()
            else None
        )
        optimizer_loaded = optimizer_state is not None
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
                    except RuntimeError as exc:
                        raise RuntimeError(
                            "FSDP checkpoint load failed. Compatibility migration is not supported for "
                            "training_strategy=fsdp_full_shard; load a checkpoint written by the same architecture."
                        ) from exc
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
        return _load_checkpoint(path=Path(checkpoint_dir), model=model, optimizer=optimizer, device=device)
    optimizer_loaded = True
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
        fusion_layers=args.fusion_layers,
        posterior_layers=args.posterior_layers,
        predictive_layers=args.predictive_layers,
        control_layers=args.control_layers,
        control_query_tokens=args.control_query_tokens,
        predictive_query_tokens=args.predictive_query_tokens,
        task_local_queries=args.task_local_queries,
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
        point_focus_sigma_m=float(_arg_or_default("point_focus_sigma_m", _SPEC_DEFAULTS.point_focus_sigma_m)),
        tactile_contact_tau_on=float(_arg_or_default("tactile_contact_tau_on", _SPEC_DEFAULTS.tactile_contact_tau_on)),
        tactile_contact_tau_off=float(_arg_or_default("tactile_contact_tau_off", _SPEC_DEFAULTS.tactile_contact_tau_off)),
        tactile_contact_temperature=float(
            _arg_or_default("tactile_contact_temperature", _SPEC_DEFAULTS.tactile_contact_temperature)
        ),
        tactile_contact_ema_beta=float(
            _arg_or_default("tactile_contact_ema_beta", _SPEC_DEFAULTS.tactile_contact_ema_beta)
        ),
        tactile_anchor_prob_on=float(_arg_or_default("tactile_anchor_prob_on", _SPEC_DEFAULTS.tactile_anchor_prob_on)),
        task_visual_reread_topk=int(_arg_or_default("task_visual_reread_topk", _SPEC_DEFAULTS.task_visual_reread_topk)),
        task_tactile_reread_groups=int(
            _arg_or_default("task_tactile_reread_groups", _SPEC_DEFAULTS.task_tactile_reread_groups)
        ),
        task_point_reread_topk=int(_arg_or_default("task_point_reread_topk", _SPEC_DEFAULTS.task_point_reread_topk)),
        action_output_clip=getattr(args, "action_output_clip", None),
        tokenwise_ff_chunk_size=int(_arg_or_default("tokenwise_ff_chunk_size", _SPEC_DEFAULTS.tokenwise_ff_chunk_size)),
        require_pi0_action_generator=bool(_arg_or_default("require_pi0_action_generator", _SPEC_DEFAULTS.require_pi0_action_generator)),
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
                gradient_checkpointing=bool(args.semantic_gradient_checkpointing),
                include_gripper_image=bool(args.semantic_use_gripper),
                max_length=args.semantic_max_length,
                prompt_state_normalization=str(getattr(args, "prompt_state_normalization", "none")),
                prompt_state_norm_stats_path=getattr(args, "prompt_state_norm_stats_path", None),
                action_horizon=int(args.action_horizon),
                tokenwise_chunk_size=int(getattr(args, "semantic_tokenwise_chunk_size", 0)),
                projection_chunk_size=int(getattr(args, "semantic_projection_chunk_size", 0)),
                mlp_chunk_size=int(getattr(args, "semantic_mlp_chunk_size", 0)),
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
    return PicfTransitionLossConfig(
        lambda_action_pos=float(getattr(args, "lambda_action_pos", 2.0)),
        lambda_action_rot=float(getattr(args, "lambda_action_rot", 2.0)),
        lambda_action_gripper=float(getattr(args, "lambda_action_gripper", 2.0)),
        lambda_visual_latent=float(getattr(args, "lambda_visual_latent", 0.2)),
        lambda_visual_real=float(getattr(args, "lambda_visual_real", 0.1)),
        lambda_tactile_real=float(getattr(args, "lambda_tactile_real", 0.3)),
        lambda_point_real=float(getattr(args, "lambda_point_real", 0.3)),
        lambda_semantic_future_aux=float(getattr(args, "lambda_semantic_future_aux", 0.25)),
        lambda_anchor_pv=float(getattr(args, "lambda_anchor_pv", 0.1)),
        lambda_pv_weak=float(getattr(args, "lambda_pv_weak", 0.02)),
        lambda_focus_pv=float(getattr(args, "lambda_focus_pv", 0.0)),
        lambda_pt=float(getattr(args, "lambda_pt", 1.0)),
        tau_pv=float(getattr(args, "tau_pv", 0.07)),
        tau_pt=float(getattr(args, "tau_pt", 0.07)),
        tau_route_p=float(getattr(args, "tau_route_p", 0.1)),
        tau_route_v=float(getattr(args, "tau_route_v", 0.1)),
        pt_bag_radius_m=float(getattr(args, "pt_bag_radius_m", 0.045)),
        pt_bag_sigma_m=float(getattr(args, "pt_bag_sigma_m", 0.015)),
        pt_bag_kmin=int(getattr(args, "pt_bag_kmin", 32)),
        pt_back_slack_m=float(getattr(args, "pt_back_slack_m", 0.008)),
        p_align_on=float(getattr(args, "p_align_on", 0.55)),
        p_align_off=float(getattr(args, "p_align_off", 0.35)),
        tactile_aux_force_scale=float(getattr(args, "tactile_aux_force_scale", 1.0)),
        tactile_aux_indent_scale=float(getattr(args, "tactile_aux_indent_scale", 5e-4)),
        tactile_aux_pressure_scale=float(getattr(args, "tactile_aux_pressure_scale", 0.1)),
        tactile_aux_pose_scale=float(getattr(args, "tactile_aux_pose_scale", getattr(args, "crop_radius_m", 0.10))),
        tactile_aux_huber_delta=float(getattr(args, "tactile_aux_huber_delta", 1.0)),
        enable_aux_budgeting=bool(getattr(args, "enable_aux_budgeting", True)),
        aux_budget_physical_ratio=float(getattr(args, "aux_budget_physical_ratio", 0.20)),
        aux_budget_semantic_ratio=float(getattr(args, "aux_budget_semantic_ratio", 0.10)),
        aux_budget_alignment_ratio=float(getattr(args, "aux_budget_alignment_ratio", 0.05)),
        aux_budget_floor=float(getattr(args, "aux_budget_floor", 0.25)),
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
    """Infer the private AnyTouch dense token width used by tactile reread.

    The tactile public trunk runs at `hidden_dim`, but the dense group memory
    stored by the AnyTouch encoder stays at the encoder-native token width
    (currently CLIP-B/16 hidden size, 768). Warmup materialization must respect
    that native width; forcing `hidden_dim` here will incorrectly initialize the
    lazy reread projections and crash once real tactile groups appear.
    """

    tactile_encoder = getattr(core, "tactile_encoder", None)
    model = getattr(tactile_encoder, "model", None)
    config = getattr(model, "config", None)
    vision_config = getattr(config, "vision_config", None)
    hidden_size = getattr(vision_config, "hidden_size", None)
    if isinstance(hidden_size, int) and hidden_size > 0:
        return int(hidden_size)
    return 768


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

    core_params = [
        param
        for param in model.parameters()
        if getattr(param, "requires_grad", False) and id(param) not in used_ids
    ]
    _append_group("picf_core", core_params, 1.0)
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
        _prepare_output_dir(output_dir=output_dir, args=args, is_main=is_main, use_ddp=use_ddp, device=device)
        fault_dump_path = output_dir / f"stackdump_rank{rank}.log"
        try:
            fault_dump_handle = fault_dump_path.open("a", encoding="utf-8")
        except OSError:
            fault_dump_handle = None
        fault_dump_registered = _register_fault_dump_handler(stream=fault_dump_handle)
        action_normalizer = _resolve_action_normalizer(args)
        source = _CalvinTransitionSource(
            args.calvin_root,
            split=args.split,
            backend=args.backend,
            unroll_steps=args.unroll_steps,
            action_horizon=args.action_horizon,
            use_tactile=bool(args.use_tactile),
            tactile_sensor_names=args.tactile_sensor_names,
            tactile_sensor_offsets_m=args.tactile_sensor_offsets_m,
            tactile_calibration=args.tactile_calibration_path,
            tactile_backgrounds_by_sensor=_load_tactile_backgrounds_npz(args.tactile_backgrounds_path),
            use_scene_obs=bool(args.use_scene_obs),
            action_normalizer=action_normalizer,
            augmentation_mode=args.picf_augmentation_mode,
            photometric_strength=args.picf_photometric_strength,
        )

        core, semantic_encoder, use_visual_override = _build_model_sequential_across_ranks(
            args,
            device=device,
            rank=rank,
            world_size=world_size,
        )
        core = core.to(device)
        model = _PicfWindowTrainer(
            core,
            semantic_encoder=semantic_encoder,
            visual_grid=args.visual_grid,
            use_visual_override=use_visual_override,
            loss_config=_build_loss_config(args),
            picf_mode=args.picf_mode,
        ).to(device)
        _materialize_model_parameters(model, source=source, rank=rank)
        model = _wrap_model_for_training_strategy(model, args=args, device=device)
        optimizer, optimizer_group_info = _build_optimizer(model, args=args)
        grad_clip_controller = _GradClipController.from_args(args)
        if use_ddp and not _is_fsdp_training(args):
            model = DistributedDataParallel(
                model,
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=True,
                gradient_as_bucket_view=True,
                static_graph=False,
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
        interval_start = time.time()
        steps_in_interval = 0
        retried_windows_interval = 0
        grad_clip_threshold = grad_clip_controller.threshold()
        grad_clip_applied = False
        recent_windows: deque[dict[str, Any]] = deque(maxlen=4)
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
                "Training config: world_size=%s training_strategy=%s picf_mode=%s accum_steps=%s effective_global_batch=%s num_steps=%s lr=%s min_lr=%s warmup=%s save_interval=%s optimizer_sharding=%s optimizer_checkpoint_mode=%s window_activation_checkpointing=%s wandb=%s",
                world_size,
                args.training_strategy,
                args.picf_mode,
                args.accum_steps,
                effective_global_batch,
                args.num_train_steps,
                args.lr,
                args.min_lr,
                args.warmup_steps,
                args.save_interval,
                args.optimizer_sharding,
                args.optimizer_checkpoint_mode,
                bool(getattr(args, "window_activation_checkpointing", False)),
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
                "Prompt-state contract: normalization=%s norm_stats=%s inject_state_into_prompt=%s",
                getattr(args, "prompt_state_normalization", "none"),
                getattr(args, "prompt_state_norm_stats_path", None),
                True,
            )
            logging.info(
                "PICF core config: hidden=%s posterior_hidden=%s latent=%s innovation=%s control=%s semantic=%s semantic_cross=%s future_hidden=%s persistent_anchors=%s observation_anchors=%s fusion_layers=%s posterior_layers=%s predictive_layers=%s control_layers=%s control_query_tokens=%s predictive_query_tokens=%s task_local_queries=%s task_global_queries=%s task_instruction_queries=%s task_self_layers=%s conditioned_control_queries=%s pi_prefix_queries=%s conditioned_future_queries=%s task_visual_reread_topk=%s task_tactile_reread_groups=%s task_point_reread_topk=%s require_pi0_action_generator=%s predictive_semantic_reads=%s control_semantic_reads=%s predictive_semantic_dropout_prob=%s semantic_prefix_dropout_prob=%s attention_heads=%s future_vote_heads=%s",
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
                args.fusion_layers,
                args.posterior_layers,
                args.predictive_layers,
                args.control_layers,
                args.control_query_tokens,
                args.predictive_query_tokens,
                args.task_local_queries,
                args.task_global_queries,
                args.task_instruction_queries,
                args.task_self_layers,
                args.conditioned_control_queries,
                args.pi_prefix_queries,
                args.conditioned_future_queries,
                args.task_visual_reread_topk,
                args.task_tactile_reread_groups,
                args.task_point_reread_topk,
                bool(args.require_pi0_action_generator),
                args.predictive_semantic_reads,
                args.control_semantic_reads,
                args.predictive_semantic_dropout_prob,
                args.semantic_prefix_dropout_prob,
                args.attention_heads,
                args.future_vote_heads,
            )
            logging.info(
                "v2.2 semantic/control contract: the full PaliGemma token sequence remains width=%s and stays native in PI0.5 semantic/action generation; core control/future no longer take raw semantic-prefix tokens directly, task readout builds semantic-conditioned current-step context from public_read_memory=[fused_tokens, visual_tokens] plus private dense rereads, and conditioned control/future consume that context together with up-projected physical posterior/global_post/innovation/proprio/physical_pred tokens; semantic_cross_dim/predictive_semantic_reads/control_semantic_reads remain compatibility fields that do not restore direct raw-semantic injection into the core trunks.",
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
                "Backbone contract: point=%s(trainable=%s flash_requested=%s) visual=%s(finetune_mode=%s trainable=%s) tactile=%s(trainable=%s) semantic=%s(trainable=%s)",
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
            _set_optimizer_lr(optimizer, lr)
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s lr_set lr=%.8f", int(step + 1), rank, float(lr))
            optimizer.zero_grad(set_to_none=True)
            if debug_phase_enabled:
                logging.info("phase step=%s rank=%s zero_grad_done", int(step + 1), rank)
            trainer_module = _unwrap_training_model(model)
            capture_visual_diagnostics = bool(
                (args.diagnostic_interval > 0 and ((step + 1) % args.diagnostic_interval == 0))
                or ((step + 1) == args.num_train_steps)
            )
            if not trainer_module.policy.picf_enabled:
                capture_visual_diagnostics = False
            use_window_activation_checkpointing = bool(
                getattr(args, "window_activation_checkpointing", False)
                and not capture_visual_diagnostics
            )
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
                    rng_start = time.perf_counter()
                    flat_index = int(rng.integers(0, len(source)))
                    if debug_phase_enabled:
                        logging.info(
                            "phase step=%s micro=%s rank=%s rng_pick_sec=%.3f flat_index=%s",
                            int(step + 1),
                            int(micro_step + 1),
                            rank,
                            time.perf_counter() - rng_start,
                            flat_index,
                        )
                    window_load_start = time.perf_counter()
                    window = source.window(flat_index, rng=rng)
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
                sync_context: Any = _training_model_no_sync(
                    model,
                    enabled=bool(world_size > 1 and micro_step < args.accum_steps - 1),
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
                            "retry_count": int(retry_count),
                            "point_counts": tuple(int(count) for count in window_point_counts),
                        }
                    )
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
                                debug_phase_label=forward_label,
                            )
                        if debug_phase_enabled:
                            logging.info("%s forward_sec=%.3f", forward_label, time.perf_counter() - forward_start)
                        if debug_cuda_sync and device.type == "cuda":
                            torch.cuda.synchronize(device=device)
                        backward_start = time.perf_counter()
                        (outputs["loss_total"] / float(args.accum_steps)).backward()
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

            if is_main and capture_visual_diagnostics:
                _save_visual_diagnostics(
                    output_dir=output_dir,
                    step=step + 1,
                    window=window,
                    physical_visual_real_seq=list(outputs.get("diagnostic_physical_visual_real_seq", [])),
                    semantic_visual_real_seq=list(outputs.get("diagnostic_semantic_visual_real_seq", [])),
                    visual_real_grid=trainer_module.core.config.visual_real_grid,
                    visual_real_upscale=args.diagnostic_visual_upscale,
                )

            should_log = ((step + 1) % args.log_interval == 0) or ((step + 1) == args.num_train_steps)
            if should_log:
                elapsed = max(time.time() - interval_start, 1e-6)
                local_grad = _grad_norm_for_training_model(model)
                averages = metric_accum.averages()
                retried_windows = float(retried_windows_interval)
                if world_size > 1:
                    averages = {k: _reduce_mean(v, device=device, world_size=world_size) for k, v in averages.items()}
                    preclip_local_grad = _reduce_mean(preclip_local_grad, device=device, world_size=world_size)
                    local_grad = _reduce_mean(local_grad, device=device, world_size=world_size)
                    retried_windows = _reduce_sum(retried_windows, device=device, world_size=world_size)
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
                metric_accum = _MetricAccumulator()
                interval_start = time.time()
                steps_in_interval = 0
                retried_windows_interval = 0

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
    parser.add_argument("--checkpoint-base-dir", required=True)
    parser.add_argument("--exp-name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--num-train-steps", type=int, default=30000)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--diagnostic-interval", type=int, default=500)
    parser.add_argument("--diagnostic-visual-upscale", type=int, default=64)
    parser.add_argument("--accum-steps", type=int, default=1)
    parser.add_argument("--max-empty-window-retries", type=int, default=32)
    parser.add_argument("--unroll-steps", type=int, default=2)
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
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr", type=float, default=2e-5)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
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
    parser.add_argument("--lambda-action-pos", type=float, default=2.0)
    parser.add_argument("--lambda-action-rot", type=float, default=2.0)
    parser.add_argument("--lambda-action-gripper", type=float, default=2.0)
    parser.add_argument("--lambda-visual-latent", type=float, default=0.2)
    parser.add_argument("--lambda-visual-real", type=float, default=0.1)
    parser.add_argument("--lambda-tactile-real", type=float, default=0.3)
    parser.add_argument("--lambda-point-real", type=float, default=0.3)
    parser.add_argument("--lambda-semantic-future-aux", type=float, default=0.25)
    parser.add_argument("--lambda-anchor-pv", type=float, default=0.1)
    parser.add_argument("--lambda-pv-weak", type=float, default=0.02)
    parser.add_argument("--lambda-focus-pv", type=float, default=0.0)
    parser.add_argument("--lambda-pt", type=float, default=1.0)
    parser.add_argument("--enable-aux-budgeting", dest="enable_aux_budgeting", action="store_true")
    parser.add_argument("--disable-aux-budgeting", dest="enable_aux_budgeting", action="store_false")
    parser.add_argument("--aux-budget-physical-ratio", type=float, default=0.20)
    parser.add_argument("--aux-budget-semantic-ratio", type=float, default=0.10)
    parser.add_argument("--aux-budget-alignment-ratio", type=float, default=0.05)
    parser.add_argument("--aux-budget-floor", type=float, default=0.25)
    parser.add_argument("--tau-pv", type=float, default=0.07)
    parser.add_argument("--tau-pt", type=float, default=0.07)
    parser.add_argument("--tau-route-p", type=float, default=0.1)
    parser.add_argument("--tau-route-v", type=float, default=0.1)
    parser.add_argument("--pt-bag-radius-m", type=float, default=None)
    parser.add_argument("--pt-bag-sigma-m", type=float, default=None)
    parser.add_argument("--pt-bag-kmin", type=int, default=32)
    parser.add_argument("--pt-back-slack-m", type=float, default=0.008)
    parser.add_argument("--p-align-on", type=float, default=0.55)
    parser.add_argument("--p-align-off", type=float, default=0.35)
    parser.add_argument("--tactile-aux-force-scale", type=float, default=1.0)
    parser.add_argument("--tactile-aux-indent-scale", type=float, default=_SPEC_DEFAULTS.tau_indent_m)
    parser.add_argument("--tactile-aux-pressure-scale", type=float, default=_SPEC_DEFAULTS.tau_tactile_pressure)
    parser.add_argument("--tactile-aux-pose-scale", type=float, default=None)
    parser.add_argument("--tactile-aux-huber-delta", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project-name", default="openpi")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="offline")
    parser.add_argument("--wandb-enabled", dest="wandb_enabled", action="store_true")
    parser.add_argument("--no-wandb", dest="wandb_enabled", action="store_false")
    parser.add_argument("--progress", dest="progress", action="store_true")
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--visual-grid", type=int, default=8)
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
    parser.add_argument("--tactile-anchor-prob-on", type=float, default=_SPEC_DEFAULTS.tactile_anchor_prob_on)
    parser.add_argument("--use-scene-obs", action="store_true")
    parser.add_argument("--semantic-mode", choices=["zero", "paligemma"], default="zero")
    parser.add_argument("--semantic-source", choices=["auto", "hf", "pi0_pytorch"], default="auto")
    parser.add_argument("--semantic-model-name", default=_default_paligemma_model_name())
    parser.add_argument("--semantic-checkpoint-path", default=None)
    parser.add_argument("--semantic-checkpoint-config-path", default=None)
    parser.add_argument("--semantic-revision", default=None)
    parser.add_argument("--semantic-paligemma-variant", default="gemma_2b")
    parser.add_argument("--semantic-action-expert-variant", default="gemma_300m")
    parser.add_argument("--semantic-dtype", default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--semantic-trainable", action="store_true")
    parser.add_argument("--semantic-lr-scale", type=float, default=0.25)
    parser.add_argument("--semantic-gradient-checkpointing", action="store_true")
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
    parser.add_argument("--fusion-layers", type=int, default=_SPEC_DEFAULTS.fusion_layers)
    parser.add_argument("--posterior-layers", type=int, default=_SPEC_DEFAULTS.posterior_layers)
    parser.add_argument("--predictive-layers", type=int, default=_SPEC_DEFAULTS.predictive_layers)
    parser.add_argument("--control-layers", type=int, default=_SPEC_DEFAULTS.control_layers)
    parser.add_argument("--control-query-tokens", type=int, default=_SPEC_DEFAULTS.control_query_tokens)
    parser.add_argument("--predictive-query-tokens", type=int, default=_SPEC_DEFAULTS.predictive_query_tokens)
    parser.add_argument("--task-local-queries", type=int, default=_SPEC_DEFAULTS.task_local_queries)
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
    parser.add_argument("--task-visual-reread-topk", type=int, default=_SPEC_DEFAULTS.task_visual_reread_topk)
    parser.add_argument("--task-tactile-reread-groups", type=int, default=_SPEC_DEFAULTS.task_tactile_reread_groups)
    parser.add_argument("--task-point-reread-topk", type=int, default=_SPEC_DEFAULTS.task_point_reread_topk)
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

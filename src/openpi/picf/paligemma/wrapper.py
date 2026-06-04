from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import inspect
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from safetensors import safe_open
from transformers import AutoProcessor
from transformers import PaliGemmaForConditionalGeneration

import openpi.models.gemma as _gemma
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import _ensure_transformers_replace_is_ready
from openpi.picf.action_normalization import PicfStateNormalizer
from openpi.picf.contracts import PicfObservation
from openpi.picf.fsdp_utils import module_num_embeddings
from openpi.picf.fsdp_utils import module_parameter_dtype
from openpi.picf.paligemma.config import PaliGemmaSemanticConfig
from openpi.shared import image_tools


@dataclasses.dataclass(frozen=True)
class PaliGemmaSemanticFeatures:
    tokens: torch.Tensor
    summary: torch.Tensor
    prefix_embeddings: torch.Tensor | None = None
    prefix_pad_masks: torch.Tensor | None = None
    prefix_att_masks: torch.Tensor | None = None
    image_tokens: torch.Tensor | None = None
    text_tokens: torch.Tensor | None = None
    image_token_ranges: tuple[tuple[int, int], ...] = ()
    image_grid_shapes: tuple[tuple[int, int], ...] = ()
    image_view_names: tuple[str, ...] = ()
    image_view_transforms: tuple["PaliGemmaViewTransform", ...] = ()


@dataclasses.dataclass(frozen=True)
class PaliGemmaViewTransform:
    original_hw: tuple[int, int]
    target_hw: tuple[int, int]
    resized_hw: tuple[int, int]
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int
    scale_y: float
    scale_x: float


def _masked_position_ids(pad_mask: torch.Tensor) -> torch.Tensor:
    """Build non-negative position ids for padded prefix tokens.

    The original PI0 path uses `cumsum(mask) - 1`, which gives `-1` on padded
    suffix positions. That is mathematically harmless when those tokens are fully
    masked, but some lower-level CUDA kernels still expect non-negative indices.
    Here we keep the same valid-token positions while mapping padded positions to
    `0`, which is equivalent under the attention mask.
    """

    if pad_mask.ndim != 2:
        raise ValueError(f"Expected 2D pad mask, got shape={tuple(pad_mask.shape)}")
    cumsum = torch.cumsum(pad_mask.to(torch.int64), dim=1) - 1
    return torch.where(pad_mask, cumsum, torch.zeros_like(cumsum))


def _runtime_timing_enabled() -> bool:
    return os.environ.get("OPENPI_PICF_TIMING_BREAKDOWN", "").strip().lower() in {"1", "true", "yes", "on"}


def _sync_cuda_for_timing() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timing_start(enabled: bool) -> float:
    if enabled:
        _sync_cuda_for_timing()
        return time.perf_counter()
    return 0.0


def _timing_record(timing: dict[str, float], name: str, start: float, enabled: bool) -> None:
    if not enabled:
        return
    _sync_cuda_for_timing()
    timing[f"{name}_ms"] = (time.perf_counter() - start) * 1000.0


def _recover_flow_target(x_t: torch.Tensor, v_t: torch.Tensor, time_expanded: torch.Tensor) -> torch.Tensor:
    """Recover the denoised target chunk from a PI0.5 flow prediction.

    PI0.5 training uses:
    - `x_t = t * noise + (1 - t) * target`
    - `u_t = noise - target`

    Eliminating `noise` yields `target = x_t - t * u_t`. When the model predicts
    `v_t ~= u_t`, the corresponding action-chunk estimate is therefore
    `x_t - t * v_t`.
    """

    return x_t - (time_expanded * v_t)


def _action_flow_objective_loss(
    target: torch.Tensor,
    pred: torch.Tensor,
    *,
    mode: str = "mse",
    huber_delta: float = 1.0,
) -> torch.Tensor:
    """Action-flow objective with a separate MSE reporting contract.

    PI0.5 parity historically trains and reports MSE on the flow velocity.  G15
    diagnostics need to test robust action objectives without breaking old
    `loss_action_default_equiv` comparisons, so the wrapper computes this
    training objective separately from the canonical MSE report.
    """

    normalized = str(mode).strip().lower().replace("-", "_")
    if normalized in {"", "mse", "l2"}:
        return torch.nn.functional.mse_loss(target, pred)
    if normalized in {"l1", "mae"}:
        return torch.nn.functional.l1_loss(target, pred)
    delta = max(float(huber_delta), 1.0e-6)
    if normalized in {"huber"}:
        return torch.nn.functional.huber_loss(target, pred, delta=delta)
    if normalized in {"smooth_l1", "smoothl1"}:
        return torch.nn.functional.smooth_l1_loss(target, pred, beta=delta)
    raise ValueError(
        "Unsupported action_flow_loss mode "
        f"{mode!r}; expected one of {{'mse', 'l1', 'huber', 'smooth_l1'}}."
    )


def _assert_index_tensor_in_range(indices: torch.Tensor, *, size: int, name: str) -> None:
    if indices.numel() == 0:
        return
    min_value = int(indices.min().item())
    max_value = int(indices.max().item())
    if min_value < 0 or max_value >= int(size):
        raise RuntimeError(
            f"{name} out of range for size={size}: min={min_value} max={max_value} shape={tuple(indices.shape)}"
        )


def _resolve_device(config: PaliGemmaSemanticConfig) -> torch.device:
    if config.device is not None:
        return torch.device(config.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_dtype(config: PaliGemmaSemanticConfig, device: torch.device) -> torch.dtype:
    if config.dtype == "float16":
        return torch.float16 if device.type == "cuda" else torch.float32
    if config.dtype == "bfloat16":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    return torch.float32


def _summary_from_outputs(
    *,
    hidden_states: torch.Tensor,
    image_hidden_states: torch.Tensor | None,
    prompt_mask: torch.Tensor | None,
) -> torch.Tensor:
    if prompt_mask is None:
        txt = hidden_states.mean(dim=1)
    else:
        denom = torch.clamp(prompt_mask.sum(dim=1, keepdim=True), min=1)
        txt = (hidden_states * prompt_mask[..., None]).sum(dim=1) / denom
    if image_hidden_states is None:
        img = torch.zeros_like(txt)
    else:
        img = image_hidden_states.mean(dim=1)
    return torch.cat([txt, img], dim=-1)


def _take_valid_prefix_tokens(hidden_states: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    if hidden_states.ndim != 2:
        raise ValueError(f"Expected hidden_states.ndim == 2, got {hidden_states.ndim}.")
    if pad_mask.ndim != 1:
        raise ValueError(f"Expected pad_mask.ndim == 1, got {pad_mask.ndim}.")
    if hidden_states.shape[0] != pad_mask.shape[0]:
        raise ValueError(
            "PaliGemma prefix token contract violated: "
            f"hidden_states.shape[0]={int(hidden_states.shape[0])} "
            f"!= pad_mask.shape[0]={int(pad_mask.shape[0])}"
        )
    valid = pad_mask.to(dtype=torch.bool)
    valid_len = int(valid.sum().item())
    if valid_len <= 0:
        return hidden_states[:0]
    if valid_len > int(hidden_states.shape[0]):
        raise RuntimeError(
            "PaliGemma prefix token contract violated: "
            f"valid_len={valid_len} exceeds token count={int(hidden_states.shape[0])}"
        )
    if not torch.all(valid[:valid_len]):
        raise RuntimeError("PaliGemma prefix pad mask is not right-padded: found invalid token inside valid prefix.")
    if valid_len < int(valid.shape[0]) and torch.any(valid[valid_len:]):
        raise RuntimeError("PaliGemma prefix pad mask is not right-padded: found valid token after padding tail.")
    return hidden_states[:valid_len]


def _pad_last_dim(x: torch.Tensor, *, dim: int) -> torch.Tensor:
    if int(x.shape[-1]) == int(dim):
        return x
    if int(x.shape[-1]) > int(dim):
        return x[..., :dim]
    return nn.functional.pad(x, (0, int(dim) - int(x.shape[-1])))


def _resolve_checkpoint_file(checkpoint_path: str | None) -> Path | None:
    if checkpoint_path is None:
        return None
    candidate = Path(checkpoint_path).expanduser()
    if candidate.is_dir():
        model_path = candidate / "model.safetensors"
        return model_path if model_path.is_file() else None
    if candidate.is_file() and candidate.suffix == ".safetensors":
        return candidate
    return None


def _checkpoint_cache_root() -> Path:
    raw = os.environ.get("OPENPI_LOCAL_CHECKPOINT_CACHE_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".cache" / "openpi" / "pi0_checkpoints"


def _checkpoint_stage_mode() -> str:
    raw = os.environ.get("OPENPI_STAGE_PI0_CHECKPOINT", "auto").strip().lower()
    if raw in {"1", "true", "yes", "on", "force"}:
        return "on"
    if raw in {"0", "false", "no", "off", "disable", "disabled"}:
        return "off"
    return "auto"


def _path_signature(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _checkpoint_stage_key(checkpoint: Path, config_path: Path | None) -> str:
    payload = {
        "checkpoint": _path_signature(checkpoint),
        "config": _path_signature(config_path) if config_path is not None and config_path.is_file() else None,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]


def _copy_stage_file(source: Path, target: Path) -> None:
    source = source.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    source_stat = source.stat()
    if target.is_file():
        target_stat = target.stat()
        if int(target_stat.st_size) == int(source_stat.st_size) and int(target_stat.st_mtime_ns) == int(source_stat.st_mtime_ns):
            return
    temp_target = target.with_name(f"{target.name}.tmp-{os.getpid()}")
    if temp_target.exists():
        temp_target.unlink()
    shutil.copy2(source, temp_target)
    os.replace(temp_target, target)


def _should_stage_checkpoint_locally(checkpoint: Path) -> bool:
    mode = _checkpoint_stage_mode()
    if mode == "on":
        return True
    if mode == "off":
        return False
    try:
        resolved = checkpoint.expanduser().resolve()
    except FileNotFoundError:
        resolved = checkpoint.expanduser()
    return str(resolved).startswith("/mnt/")


def _stage_pi0_checkpoint_if_needed(
    checkpoint: Path,
    config_path: Path | None,
) -> tuple[Path, Path | None]:
    if not _should_stage_checkpoint_locally(checkpoint):
        return checkpoint, config_path

    cache_root = _checkpoint_cache_root()
    stage_key = _checkpoint_stage_key(checkpoint, config_path)
    stage_dir = cache_root / stage_key
    staged_checkpoint = stage_dir / checkpoint.name
    staged_config = stage_dir / config_path.name if config_path is not None and config_path.is_file() else None

    rank = 0
    world_size = 1
    use_barrier = bool(dist.is_available() and dist.is_initialized())
    if use_barrier:
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())

    if rank == 0:
        start = time.perf_counter()
        stage_dir.mkdir(parents=True, exist_ok=True)
        _copy_stage_file(checkpoint, staged_checkpoint)
        if config_path is not None and config_path.is_file():
            _copy_stage_file(config_path, staged_config)
        logging.info(
            "Staged PI0 checkpoint locally: src=%s dst=%s world_size=%d elapsed_sec=%.3f",
            str(checkpoint),
            str(staged_checkpoint),
            world_size,
            time.perf_counter() - start,
        )
    if use_barrier and world_size > 1:
        dist.barrier()

    if not staged_checkpoint.is_file():
        raise FileNotFoundError(f"Staged PI0 checkpoint missing after local staging: {staged_checkpoint}")
    if staged_config is not None and not staged_config.is_file():
        raise FileNotFoundError(f"Staged PI0 config missing after local staging: {staged_config}")
    return staged_checkpoint, staged_config


def _stage_local_pi0_config(config: PaliGemmaSemanticConfig) -> PaliGemmaSemanticConfig:
    checkpoint = _resolve_checkpoint_file(config.checkpoint_path)
    if checkpoint is None:
        return config
    explicit_config = Path(config.checkpoint_config_path).expanduser() if config.checkpoint_config_path is not None else None
    if explicit_config is not None and not explicit_config.is_file():
        explicit_config = None
    if explicit_config is None:
        sidecar = checkpoint.parent / "config.json"
        explicit_config = sidecar if sidecar.is_file() else None
    staged_checkpoint, staged_config = _stage_pi0_checkpoint_if_needed(checkpoint, explicit_config)
    if staged_checkpoint == checkpoint and staged_config == explicit_config:
        return config
    staged_checkpoint_path = str(staged_checkpoint.parent if staged_checkpoint.name == "model.safetensors" else staged_checkpoint)
    staged_config_path = str(staged_config) if staged_config is not None else None
    return dataclasses.replace(
        config,
        checkpoint_path=staged_checkpoint_path,
        checkpoint_config_path=staged_config_path,
    )


def _resolve_source(config: PaliGemmaSemanticConfig) -> str:
    if config.source != "auto":
        return str(config.source)
    if _resolve_checkpoint_file(config.checkpoint_path) is not None:
        return "pi0_pytorch"
    return "hf"


def _read_pi0_checkpoint_metadata(config: PaliGemmaSemanticConfig) -> dict[str, Any]:
    explicit = config.checkpoint_config_path
    if explicit is not None and Path(explicit).expanduser().is_file():
        return json.loads(Path(explicit).expanduser().read_text(encoding="utf-8"))
    ckpt = _resolve_checkpoint_file(config.checkpoint_path)
    if ckpt is None:
        return {}
    sidecar = ckpt.parent / "config.json"
    if sidecar.is_file():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    return {}


def _repair_missing_tied_embeddings(
    model: nn.Module,
    *,
    missing_keys: list[str],
) -> list[str]:
    """Repair known tied-weight gaps from local PI0 checkpoints.

    The local `pi05_base_pytorch` safetensors only store `lm_head.weight` for the
    PaliGemma branch. HF `PaliGemmaForConditionalGeneration` expects the tied
    input embedding weight under `model.language_model.embed_tokens.weight`.
    When that single key is missing, copy from `lm_head.weight` and keep the
    remainder of the strictness checks intact.
    """

    repaired = list(missing_keys)
    embed_key = "model.language_model.embed_tokens.weight"
    if embed_key not in repaired:
        return repaired

    lm_head = getattr(model, "lm_head", None)
    inner_model = getattr(model, "model", None)
    language_model = getattr(inner_model, "language_model", None)
    embed_tokens = getattr(language_model, "embed_tokens", None)
    if lm_head is None or not hasattr(lm_head, "weight") or embed_tokens is None or not hasattr(embed_tokens, "weight"):
        return repaired
    if tuple(embed_tokens.weight.shape) != tuple(lm_head.weight.shape):
        return repaired

    with torch.no_grad():
        embed_tokens.weight.copy_(lm_head.weight.to(dtype=embed_tokens.weight.dtype, device=embed_tokens.weight.device))
    repaired.remove(embed_key)
    return repaired


def _checkpoint_inputs_require_grad(*args: object) -> bool:
    for arg in args:
        if isinstance(arg, torch.Tensor) and bool(arg.requires_grad):
            return True
    return False


def _enable_gradient_checkpointing_non_reentrant(module: nn.Module) -> tuple[bool, bool]:
    """Enable gradient checkpointing, preferring non-reentrant mode when available.

    Returns:
      enabled: whether gradient checkpointing was enabled at all
      non_reentrant: whether `use_reentrant=False` was successfully requested
    """
    fn = getattr(module, "gradient_checkpointing_enable", None)
    if fn is None:
        return False, False
    try:
        fn(gradient_checkpointing_kwargs={"use_reentrant": False})
        return True, True
    except TypeError:
        fn()
        return True, False


class _HFPaliGemmaSemanticEncoder(nn.Module):
    def __init__(self, config: PaliGemmaSemanticConfig):
        super().__init__()
        self.config = config
        self.device = _resolve_device(config)
        self.dtype = _resolve_dtype(config, self.device)
        self.trainable = bool(config.trainable)
        self.source = "hf"
        self.gradient_checkpointing_enabled = False
        self.gradient_checkpointing_non_reentrant = False
        self._last_runtime_timing: dict[str, float] = {}
        model_id = config.checkpoint_path or config.model_name
        local_only = Path(model_id).expanduser().exists()
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=config.revision,
            local_files_only=local_only,
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            revision=config.revision,
            torch_dtype=self.dtype,
            local_files_only=local_only,
        )
        self.model.to(device=self.device, dtype=self.dtype)
        if self.trainable:
            if hasattr(self.model, "gradient_checkpointing_enable") and config.gradient_checkpointing:
                enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(self.model)
                self.gradient_checkpointing_enabled = enabled
                self.gradient_checkpointing_non_reentrant = non_reentrant
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
        else:
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)

    def _named_views(self, observation: PicfObservation) -> list[tuple[str, np.ndarray]]:
        views = [("static", np.asarray(observation.rgb_static))]
        if self.config.include_gripper_image and observation.rgb_gripper is not None:
            views.append(("gripper", np.asarray(observation.rgb_gripper)))
        return views

    def _views(self, observation: PicfObservation) -> list[np.ndarray]:
        return [image for _, image in self._named_views(observation)]

    def _view_transform(self, image: np.ndarray, *, target_h: int = 224, target_w: int = 224) -> PaliGemmaViewTransform:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image for view transform, got shape={tuple(arr.shape)}")
        cur_h, cur_w = int(arr.shape[0]), int(arr.shape[1])
        if cur_h <= 0 or cur_w <= 0:
            raise ValueError(f"Expected positive image size, got original_hw={(cur_h, cur_w)}")
        ratio = max(cur_w / float(target_w), cur_h / float(target_h))
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)
        pad_top, rem_h = divmod(target_h - resized_h, 2)
        pad_bottom = pad_top + rem_h
        pad_left, rem_w = divmod(target_w - resized_w, 2)
        pad_right = pad_left + rem_w
        return PaliGemmaViewTransform(
            original_hw=(cur_h, cur_w),
            target_hw=(target_h, target_w),
            resized_hw=(resized_h, resized_w),
            pad_top=int(pad_top),
            pad_bottom=int(pad_bottom),
            pad_left=int(pad_left),
            pad_right=int(pad_right),
            scale_y=float(resized_h / max(cur_h, 1)),
            scale_x=float(resized_w / max(cur_w, 1)),
        )

    def _prepare_inputs(self, *, prompt: str, image: np.ndarray) -> dict[str, torch.Tensor]:
        processed = self.processor(
            text=[str(prompt)],
            images=[image],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(self.config.max_length),
        )
        prepared: dict[str, torch.Tensor] = {}
        for key, value in processed.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "pixel_values":
                prepared[key] = value.to(device=self.device, dtype=self.dtype)
            else:
                prepared[key] = value.to(device=self.device)
        return prepared

    def encode_observation(self, observation: PicfObservation) -> PaliGemmaSemanticFeatures:
        views = self._views(observation)
        prompt = str(observation.prompt)
        summaries: list[torch.Tensor] = []
        tokens_list: list[torch.Tensor] = []
        use_grad = bool(self.trainable and self.training)
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            for image in views:
                inputs = self._prepare_inputs(prompt=prompt, image=image)
                outputs = self.model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                    use_cache=False,
                )
                hidden_states = outputs.hidden_states[-1]
                image_hidden_states = outputs.image_hidden_states
                attention_mask = inputs.get("attention_mask")
                input_ids = inputs.get("input_ids")
                image_token_id = getattr(self.model.config, "image_token_index", None)
                if image_token_id is None:
                    image_token_id = getattr(self.model.config, "image_token_id", None)
                prompt_mask = attention_mask
                if prompt_mask is not None and input_ids is not None and image_token_id is not None:
                    prompt_mask = prompt_mask * (input_ids != int(image_token_id)).to(dtype=prompt_mask.dtype)
                attention_mask = inputs.get("attention_mask")
                if attention_mask is None:
                    tokens_list.append(hidden_states[0])
                else:
                    tokens_list.append(_take_valid_prefix_tokens(hidden_states[0], attention_mask[0]))
                summaries.append(
                    _summary_from_outputs(
                        hidden_states=hidden_states,
                        image_hidden_states=image_hidden_states,
                        prompt_mask=prompt_mask,
                    )
                )
        if not summaries or not tokens_list:
            raise RuntimeError("PaliGemma semantic encoder did not receive any image views.")
        return PaliGemmaSemanticFeatures(
            tokens=torch.cat(tokens_list, dim=0),
            summary=torch.stack(summaries, dim=0).mean(dim=0),
        )

    def supports_pi0_action_generation(self) -> bool:
        return False

    def compute_action_flow_loss(self, *args, **kwargs):
        raise RuntimeError("HF PaliGemma semantic encoder does not provide PI0 action generation.")

    def sample_action_chunk(self, *args, **kwargs):
        raise RuntimeError("HF PaliGemma semantic encoder does not provide PI0 action generation.")

    def forward(self, op: str, /, *args: Any, **kwargs: Any):
        if op == "encode_observation":
            return self.encode_observation(*args, **kwargs)
        if op == "compute_action_flow_loss":
            return self.compute_action_flow_loss(*args, **kwargs)
        if op == "sample_action_chunk":
            return self.sample_action_chunk(*args, **kwargs)
        raise ValueError(f"Unsupported semantic forward op: {op!r}")


class _Pi0PaliGemmaSemanticEncoder(nn.Module):
    def __init__(self, config: PaliGemmaSemanticConfig):
        super().__init__()
        self.config = config
        self.device = _resolve_device(config)
        self.dtype = _resolve_dtype(config, self.device)
        self.trainable = bool(config.trainable)
        self.trainable_scope = str(getattr(config, "trainable_scope", "backbone_only")).strip().lower().replace("-", "_")
        if self.trainable_scope not in {
            "all",
            "backbone_only",
            "model_only",
            "action_head_only",
            "action_adapter_only",
            "action_head_and_adapter",
        }:
            raise ValueError(
                "PaliGemmaSemanticConfig.trainable_scope must be one of "
                "{'all', 'backbone_only', 'model_only', 'action_head_only', "
                "'action_adapter_only', 'action_head_and_adapter'}, got "
                f"{getattr(config, 'trainable_scope', None)!r}."
            )
        self.source = "pi0_pytorch"
        self.gradient_checkpointing_enabled = False
        self.gradient_checkpointing_non_reentrant = False

        metadata = _read_pi0_checkpoint_metadata(config)
        paligemma_variant = str(metadata.get("paligemma_variant", config.paligemma_variant))
        precision = str(metadata.get("precision", config.dtype))
        max_token_len = int(metadata.get("max_token_len", config.max_length))
        checkpoint = _resolve_checkpoint_file(config.checkpoint_path)
        if checkpoint is None:
            raise FileNotFoundError(
                "pi0_pytorch semantic source requires a local checkpoint directory or .safetensors file."
            )

        _ensure_transformers_replace_is_ready()
        self.model_action_dim = int(config.action_dim)
        self.action_horizon = int(config.action_horizon)
        self.denoise_steps = int(config.denoise_steps)
        self.action_flow_loss = str(getattr(config, "action_flow_loss", "mse")).strip().lower().replace("-", "_")
        if self.action_flow_loss not in {"mse", "l2", "l1", "mae", "huber", "smooth_l1", "smoothl1"}:
            raise ValueError(
                "PaliGemmaSemanticConfig.action_flow_loss must be one of "
                "{'mse', 'l1', 'huber', 'smooth_l1'}, got "
                f"{getattr(config, 'action_flow_loss', None)!r}."
            )
        self.action_flow_huber_delta = max(float(getattr(config, "action_flow_huber_delta", 1.0)), 1.0e-6)
        self.action_flow_time_alpha = max(float(getattr(config, "action_flow_time_alpha", 1.5)), 1.0e-6)
        self.action_flow_time_beta = max(float(getattr(config, "action_flow_time_beta", 1.0)), 1.0e-6)
        self.action_context_readout_aux_weight = max(
            float(getattr(config, "action_context_readout_aux_weight", 0.0)),
            0.0,
        )
        self.action_context_readout_aux_loss = (
            str(getattr(config, "action_context_readout_aux_loss", "smooth_l1")).strip().lower().replace("-", "_")
        )
        if self.action_context_readout_aux_loss not in {"mse", "l2", "l1", "mae", "huber", "smooth_l1", "smoothl1"}:
            raise ValueError(
                "PaliGemmaSemanticConfig.action_context_readout_aux_loss must be one of "
                "{'mse', 'l1', 'huber', 'smooth_l1'}, got "
                f"{getattr(config, 'action_context_readout_aux_loss', None)!r}."
            )
        self.action_context_readout_aux_huber_delta = max(
            float(getattr(config, "action_context_readout_aux_huber_delta", 1.0)),
            1.0e-6,
        )
        self.action_context_token_aux_weight = max(
            float(getattr(config, "action_context_token_aux_weight", 0.0)),
            0.0,
        )
        self.action_context_token_aux_bins = max(
            int(getattr(config, "action_context_token_aux_bins", 256)),
            2,
        )
        self.action_context_token_aux_clip = max(
            float(getattr(config, "action_context_token_aux_clip", 1.0)),
            1.0e-6,
        )
        self.action_context_flow_residual_enabled = bool(
            getattr(config, "action_context_flow_residual_enabled", False)
        )
        self.action_context_flow_residual_time_floor = max(
            float(getattr(config, "action_context_flow_residual_time_floor", 0.05)),
            1.0e-6,
        )
        self.action_context_flow_residual_rms_cap = bool(
            getattr(config, "action_context_flow_residual_rms_cap", True)
        )
        self.action_expert_router_enabled = bool(getattr(config, "action_expert_router_enabled", False))
        self.action_expert_router_experts = max(int(getattr(config, "action_expert_router_experts", 4)), 1)
        self.action_expert_router_rank = max(int(getattr(config, "action_expert_router_rank", 64)), 1)
        self.action_expert_router_temperature = max(
            float(getattr(config, "action_expert_router_temperature", 1.0)),
            1.0e-6,
        )
        self.action_expert_router_rms_cap = bool(getattr(config, "action_expert_router_rms_cap", True))
        self.paligemma_with_expert = self._build_paligemma_with_expert(
            paligemma_variant=paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            precision=precision,
            pi05=bool(config.pi05),
        )
        paligemma_config = _gemma.get_config(paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_in_proj = nn.Linear(self.model_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, self.model_action_dim)
        if not bool(config.pi05):
            raise RuntimeError("PICF PI0 action restoration only supports pi05=True.")
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.action_context_in_proj = nn.Linear(paligemma_config.width, action_expert_config.width, bias=False)
        self.action_context_q_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_k_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_v_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_out_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_gate_logit = nn.Parameter(
            torch.tensor([float(getattr(config, "action_context_adapter_gate_init", -2.0))], dtype=torch.float32)
        )
        self.action_context_adapter_rms_cap = bool(getattr(config, "action_context_adapter_rms_cap", True))
        self.action_context_readout_query = nn.Parameter(
            torch.empty(self.action_horizon, action_expert_config.width, dtype=torch.float32)
        )
        self.action_context_readout_q_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_readout_k_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_readout_v_proj = nn.Linear(action_expert_config.width, action_expert_config.width, bias=False)
        self.action_context_readout_out_proj = nn.Linear(action_expert_config.width, self.model_action_dim)
        self.action_context_token_readout_out_proj = nn.Linear(
            action_expert_config.width,
            self.model_action_dim * self.action_context_token_aux_bins,
        )
        self.action_context_flow_residual_gate_logit = nn.Parameter(
            torch.tensor(
                [float(getattr(config, "action_context_flow_residual_gate_init", -2.0))],
                dtype=torch.float32,
            )
        )
        self.action_expert_router_summary_proj = nn.Linear(paligemma_config.width, action_expert_config.width, bias=False)
        self.action_expert_router_summary_pair_proj = nn.Linear(
            paligemma_config.width * 2,
            action_expert_config.width,
            bias=False,
        )
        self.action_expert_router_norm = nn.LayerNorm(action_expert_config.width)
        self.action_expert_router_logits = nn.Linear(action_expert_config.width, self.action_expert_router_experts)
        self.action_expert_router_down = nn.ModuleList(
            nn.Linear(action_expert_config.width, self.action_expert_router_rank, bias=False)
            for _ in range(self.action_expert_router_experts)
        )
        self.action_expert_router_up = nn.ModuleList(
            nn.Linear(self.action_expert_router_rank, action_expert_config.width, bias=False)
            for _ in range(self.action_expert_router_experts)
        )
        self.action_expert_router_gate_logit = nn.Parameter(
            torch.tensor([float(getattr(config, "action_expert_router_gate_init", -2.5))], dtype=torch.float32)
        )
        self._reset_action_context_adapter_parameters()
        self._reset_action_context_readout_parameters()
        self._reset_action_expert_router_parameters()
        self._load_full_pi0_weights(checkpoint)
        self._drop_unused_generation_heads()
        self.to(device=self.device)
        self.tokenizer = PaligemmaTokenizer(max_len=max_token_len)
        prompt_state_mode = str(getattr(config, "prompt_state_normalization", "none")).lower()
        prompt_state_path = getattr(config, "prompt_state_norm_stats_path", None)
        self.prompt_state_normalizer = (
            None
            if prompt_state_mode == "none" or prompt_state_path is None
            else PicfStateNormalizer.from_path(prompt_state_path, mode=prompt_state_mode)
        )
        if self.trainable and self._trains_semantic_backbone():
            if hasattr(self.paligemma_with_expert.paligemma, "gradient_checkpointing_disable"):
                self.paligemma_with_expert.paligemma.gradient_checkpointing_disable()
            if config.gradient_checkpointing and hasattr(self, "gradient_checkpointing_enable"):
                enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(self)
                self.gradient_checkpointing_enabled = enabled
                self.gradient_checkpointing_non_reentrant = non_reentrant
            if hasattr(self.paligemma_with_expert.paligemma.config, "use_cache"):
                self.paligemma_with_expert.paligemma.config.use_cache = False
            self.train()
        else:
            self.eval()

        self._apply_trainable_scope()

    @property
    def model(self) -> PaliGemmaWithExpertModel:
        return self.paligemma_with_expert

    def _trains_semantic_backbone(self) -> bool:
        return bool(self.trainable and self.trainable_scope in {"all", "backbone_only", "model_only"})

    def _action_context_adapter_modules(self) -> tuple[nn.Module, ...]:
        modules = (
            getattr(self, "action_context_in_proj", None),
            getattr(self, "action_context_q_proj", None),
            getattr(self, "action_context_k_proj", None),
            getattr(self, "action_context_v_proj", None),
            getattr(self, "action_context_out_proj", None),
        )
        return tuple(module for module in modules if isinstance(module, nn.Module))

    def _action_context_readout_modules(self) -> tuple[nn.Module, ...]:
        modules = (
            getattr(self, "action_context_readout_q_proj", None),
            getattr(self, "action_context_readout_k_proj", None),
            getattr(self, "action_context_readout_v_proj", None),
            getattr(self, "action_context_readout_out_proj", None),
        )
        return tuple(module for module in modules if isinstance(module, nn.Module))

    def _action_context_token_aux_modules(self) -> tuple[nn.Module, ...]:
        modules = (getattr(self, "action_context_token_readout_out_proj", None),)
        return tuple(module for module in modules if isinstance(module, nn.Module))

    def _action_context_token_aux_enabled(self) -> bool:
        return float(getattr(self, "action_context_token_aux_weight", 0.0)) > 0.0

    def _train_action_context_token_aux_modules_if_enabled(self) -> None:
        if not self._action_context_token_aux_enabled():
            return
        for module in self._action_context_token_aux_modules():
            module.train()
            for parameter in module.parameters():
                parameter.requires_grad_(True)

    def _action_expert_router_modules(self) -> tuple[nn.Module, ...]:
        modules = (
            getattr(self, "action_expert_router_summary_proj", None),
            getattr(self, "action_expert_router_summary_pair_proj", None),
            getattr(self, "action_expert_router_norm", None),
            getattr(self, "action_expert_router_logits", None),
            *(getattr(self, "action_expert_router_down", nn.ModuleList())),
            *(getattr(self, "action_expert_router_up", nn.ModuleList())),
        )
        return tuple(module for module in modules if isinstance(module, nn.Module))

    def _reset_action_context_adapter_parameters(self) -> None:
        """Initialize the action-side context adapter as a conservative residual.

        Query/key projections use standard attention initialization.  Value and
        output projections start near identity so a small learned gate can expose
        PICF belief context without requiring the action expert to discover a
        new token vocabulary from scratch.
        """

        if isinstance(getattr(self, "action_context_in_proj", None), nn.Linear):
            nn.init.xavier_uniform_(self.action_context_in_proj.weight)
        nn.init.xavier_uniform_(self.action_context_q_proj.weight)
        nn.init.xavier_uniform_(self.action_context_k_proj.weight)
        nn.init.eye_(self.action_context_v_proj.weight)
        nn.init.eye_(self.action_context_out_proj.weight)

    def _reset_action_context_readout_parameters(self) -> None:
        """Initialize context-only action readout as a conservative auxiliary.

        The readout is not part of inference.  It is a training probe/objective
        that asks whether bounded PICF context alone contains motor-readable
        information.  It therefore must not reuse noisy action suffix tokens,
        otherwise the auxiliary could be solved without repairing `dA/dC`.
        """

        if isinstance(getattr(self, "action_context_readout_query", None), nn.Parameter):
            nn.init.normal_(self.action_context_readout_query, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.action_context_readout_q_proj.weight)
        nn.init.xavier_uniform_(self.action_context_readout_k_proj.weight)
        nn.init.xavier_uniform_(self.action_context_readout_v_proj.weight)
        nn.init.xavier_uniform_(self.action_context_readout_out_proj.weight)
        if self.action_context_readout_out_proj.bias is not None:
            nn.init.zeros_(self.action_context_readout_out_proj.bias)
        token_head = getattr(self, "action_context_token_readout_out_proj", None)
        if isinstance(token_head, nn.Linear):
            nn.init.xavier_uniform_(token_head.weight)
            if token_head.bias is not None:
                nn.init.zeros_(token_head.bias)

    def _reset_action_expert_router_parameters(self) -> None:
        """Initialize action-expert routing as an identity-preserving adapter.

        The router is a task/semantic-conditioned low-rank residual on action
        suffix embeddings.  It starts as an exact no-op because all up
        projections are zero and the learned gate is initialized small.  This
        gives the action expert an expert-routing path without changing the
        restored PI0.5 function before training proves the route useful.
        """

        if isinstance(getattr(self, "action_expert_router_summary_proj", None), nn.Linear):
            nn.init.xavier_uniform_(self.action_expert_router_summary_proj.weight)
        if isinstance(getattr(self, "action_expert_router_summary_pair_proj", None), nn.Linear):
            nn.init.xavier_uniform_(self.action_expert_router_summary_pair_proj.weight)
        if isinstance(getattr(self, "action_expert_router_logits", None), nn.Linear):
            nn.init.zeros_(self.action_expert_router_logits.weight)
            nn.init.zeros_(self.action_expert_router_logits.bias)
        for down in getattr(self, "action_expert_router_down", ()):
            if isinstance(down, nn.Linear):
                nn.init.xavier_uniform_(down.weight)
        for up in getattr(self, "action_expert_router_up", ()):
            if isinstance(up, nn.Linear):
                nn.init.zeros_(up.weight)

    def _apply_trainable_scope(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        if not self.trainable:
            return
        if self.trainable_scope == "all":
            for parameter in self.parameters():
                parameter.requires_grad_(True)
            return
        if self.trainable_scope in {"backbone_only", "model_only"}:
            # Historical full-cotrain contract: train the restored PI0/PaliGemma
            # semantic stack while keeping wrapper-local flow projection/time
            # heads fixed.  This preserves PaliGemma cotrain without moving the
            # small PI0 flow calibration heads into a separate trainable/FSDP
            # path.  The action-context adapter is part of the PICF/action
            # interface, so it remains trainable with the semantic backbone.
            for parameter in self.model.parameters():
                parameter.requires_grad_(True)
            for module in self._action_context_adapter_modules():
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for module in self._action_context_readout_modules():
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            self._train_action_context_token_aux_modules_if_enabled()
            if isinstance(getattr(self, "action_context_gate_logit", None), nn.Parameter):
                self.action_context_gate_logit.requires_grad_(True)
            if isinstance(getattr(self, "action_context_readout_query", None), nn.Parameter):
                self.action_context_readout_query.requires_grad_(True)
            if isinstance(getattr(self, "action_context_flow_residual_gate_logit", None), nn.Parameter):
                self.action_context_flow_residual_gate_logit.requires_grad_(True)
            if bool(getattr(self, "action_expert_router_enabled", False)):
                for module in self._action_expert_router_modules():
                    module.train()
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
                if isinstance(getattr(self, "action_expert_router_gate_logit", None), nn.Parameter):
                    self.action_expert_router_gate_logit.requires_grad_(True)
            return
        if self.trainable_scope == "action_head_only":
            for module in (self.action_in_proj, self.action_out_proj, self.time_mlp_in, self.time_mlp_out):
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            return
        if self.trainable_scope == "action_adapter_only":
            for module in self._action_context_adapter_modules():
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            for module in self._action_context_readout_modules():
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            self._train_action_context_token_aux_modules_if_enabled()
            if isinstance(getattr(self, "action_context_gate_logit", None), nn.Parameter):
                self.action_context_gate_logit.requires_grad_(True)
            if isinstance(getattr(self, "action_context_readout_query", None), nn.Parameter):
                self.action_context_readout_query.requires_grad_(True)
            if isinstance(getattr(self, "action_context_flow_residual_gate_logit", None), nn.Parameter):
                self.action_context_flow_residual_gate_logit.requires_grad_(True)
            if bool(getattr(self, "action_expert_router_enabled", False)):
                for module in self._action_expert_router_modules():
                    module.train()
                    for parameter in module.parameters():
                        parameter.requires_grad_(True)
                if isinstance(getattr(self, "action_expert_router_gate_logit", None), nn.Parameter):
                    self.action_expert_router_gate_logit.requires_grad_(True)
            return
        if self.trainable_scope == "action_head_and_adapter":
            for module in (
                self.action_in_proj,
                self.action_out_proj,
                self.time_mlp_in,
                self.time_mlp_out,
                *self._action_context_adapter_modules(),
                *self._action_context_readout_modules(),
                *(self._action_expert_router_modules() if bool(getattr(self, "action_expert_router_enabled", False)) else ()),
            ):
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
            self._train_action_context_token_aux_modules_if_enabled()
            if isinstance(getattr(self, "action_context_gate_logit", None), nn.Parameter):
                self.action_context_gate_logit.requires_grad_(True)
            if isinstance(getattr(self, "action_context_readout_query", None), nn.Parameter):
                self.action_context_readout_query.requires_grad_(True)
            if isinstance(getattr(self, "action_context_flow_residual_gate_logit", None), nn.Parameter):
                self.action_context_flow_residual_gate_logit.requires_grad_(True)
            if bool(getattr(self, "action_expert_router_enabled", False)) and isinstance(
                getattr(self, "action_expert_router_gate_logit", None),
                nn.Parameter,
            ):
                self.action_expert_router_gate_logit.requires_grad_(True)
            return
        raise AssertionError(f"Unhandled trainable_scope={self.trainable_scope!r}")

    @property
    def last_runtime_timing(self) -> dict[str, float]:
        return dict(self._last_runtime_timing)

    def _build_paligemma_with_expert(
        self,
        *,
        paligemma_variant: str,
        action_expert_variant: str,
        precision: str,
        pi05: bool,
    ) -> PaliGemmaWithExpertModel:
        paligemma_config = _gemma.get_config(paligemma_variant)
        action_expert_config = _gemma.get_config(action_expert_variant)
        config = self.__dict__.get("config")
        tokenwise_chunk_size = int(getattr(config, "tokenwise_chunk_size", 0))
        projection_chunk_size = getattr(config, "projection_chunk_size", None)
        mlp_chunk_size = getattr(config, "mlp_chunk_size", None)
        model_kwargs: dict[str, Any] = {
            "use_adarms": [False, True] if pi05 else [False, False],
            "precision": precision,
            "tokenwise_chunk_size": tokenwise_chunk_size,
            "projection_chunk_size": None if projection_chunk_size is None else int(projection_chunk_size),
            "mlp_chunk_size": None if mlp_chunk_size is None else int(mlp_chunk_size),
        }
        try:
            signature = inspect.signature(PaliGemmaWithExpertModel)
            parameters = signature.parameters
            accepts_var_kwargs = any(
                parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
            )
            if not accepts_var_kwargs:
                model_kwargs = {
                    name: value for name, value in model_kwargs.items() if name in parameters
                }
        except (TypeError, ValueError):
            # Some wrapped/compiled constructors do not expose a signature. In
            # that case keep the full modern argument set and let Python report
            # an actual constructor error if it is incompatible.
            pass
        model = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            **model_kwargs,
        )
        return model

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: dict[str, Any] | None = None) -> None:
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

    def gradient_checkpointing_disable(self) -> None:
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

    def _load_full_pi0_weights(self, checkpoint: Path) -> None:
        paligemma_state: dict[str, torch.Tensor] = {}
        expert_state: dict[str, torch.Tensor] = {}
        head_tensors: dict[str, torch.Tensor] = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                if key.startswith("paligemma_with_expert.paligemma."):
                    paligemma_state[key[len("paligemma_with_expert.paligemma.") :]] = tensor
                elif key.startswith("paligemma_with_expert.gemma_expert."):
                    expert_state[key[len("paligemma_with_expert.gemma_expert.") :]] = tensor
                elif key in {
                    "action_in_proj.weight",
                    "action_in_proj.bias",
                    "action_out_proj.weight",
                    "action_out_proj.bias",
                    "time_mlp_in.weight",
                    "time_mlp_in.bias",
                    "time_mlp_out.weight",
                    "time_mlp_out.bias",
                }:
                    head_tensors[key] = tensor
        if not paligemma_state:
            raise RuntimeError(
                "Local PI0 checkpoint does not contain paligemma_with_expert.paligemma.* weights."
            )
        missing, unexpected = self.paligemma_with_expert.paligemma.load_state_dict(paligemma_state, strict=False)
        if unexpected:
            raise RuntimeError(
                "Unexpected keys when loading local PI0 PaliGemma semantic checkpoint:\n"
                + "\n".join(map(str, unexpected[:200]))
            )
        repaired_missing = _repair_missing_tied_embeddings(self.paligemma_with_expert.paligemma, missing_keys=list(missing))
        bad_missing = [key for key in repaired_missing if key not in {"lm_head.weight"}]
        if bad_missing:
            raise RuntimeError(
                "Unexpected missing keys when loading local PI0 PaliGemma semantic checkpoint:\n"
                + "\n".join(map(str, bad_missing[:200]))
            )
        if not expert_state:
            raise RuntimeError(
                "Local PI0 checkpoint does not contain paligemma_with_expert.gemma_expert.* weights."
            )
        expert_missing, expert_unexpected = self.paligemma_with_expert.gemma_expert.load_state_dict(expert_state, strict=False)
        if expert_missing or expert_unexpected:
            raise RuntimeError(
                "Unexpected Gemma expert checkpoint incompatibility. "
                f"missing[:40]={list(expert_missing)[:40]} unexpected[:40]={list(expert_unexpected)[:40]}"
            )
        self._copy_linear_weight(self.action_in_proj, head_tensors, "action_in_proj")
        self._copy_linear_weight(self.action_out_proj, head_tensors, "action_out_proj")
        self._copy_linear_weight(self.time_mlp_in, head_tensors, "time_mlp_in")
        self._copy_linear_weight(self.time_mlp_out, head_tensors, "time_mlp_out")

    def _copy_linear_weight(self, module: nn.Linear, tensors: dict[str, torch.Tensor], prefix: str) -> None:
        weight_key = f"{prefix}.weight"
        bias_key = f"{prefix}.bias"
        if weight_key not in tensors or bias_key not in tensors:
            raise RuntimeError(f"Checkpoint is missing {weight_key}/{bias_key}.")
        weight = tensors[weight_key]
        bias = tensors[bias_key]
        if tuple(module.weight.shape) != tuple(weight.shape) or tuple(module.bias.shape) != tuple(bias.shape):
            raise RuntimeError(
                f"{prefix} shape mismatch: module weight={tuple(module.weight.shape)} bias={tuple(module.bias.shape)} "
                f"ckpt weight={tuple(weight.shape)} bias={tuple(bias.shape)}"
            )
        with torch.no_grad():
            module.weight.copy_(weight.to(device=module.weight.device, dtype=module.weight.dtype))
            module.bias.copy_(bias.to(device=module.bias.device, dtype=module.bias.dtype))

    def _drop_unused_generation_heads(self) -> None:
        """Remove unused LM heads from the runtime graph.

        PICF/PI0 training consumes hidden states from `paligemma.language_model`
        and `gemma_expert.model`, then uses `action_out_proj` for the final action
        path. The outer causal-LM heads are checkpoint artifacts for tokenizer /
        tie-weight compatibility, but they are not touched by the live training
        forward. Dropping them after checkpoint load keeps the runtime graph
        mathematically unchanged while removing large dead parameters from FSDP
        wrapping and optimizer enumeration.
        """

        if getattr(self.paligemma_with_expert.paligemma, "lm_head", None) is not None:
            self.paligemma_with_expert.paligemma.lm_head = None
        if getattr(self.paligemma_with_expert.gemma_expert, "lm_head", None) is not None:
            self.paligemma_with_expert.gemma_expert.lm_head = None

    def fsdp_runtime_leaf_module_specs(self) -> list[tuple[nn.Module, str, str]]:
        """Expose runtime-hot semantic leaves that can be nested-FSDP wrapped exactly.

        The dual-branch PI0/Gemma path directly calls these child modules during
        training. Wrapping them as explicit nested leaves lets FSDP all-gather
        them on demand instead of materializing the full semantic stack as one
        monolithic boundary.
        """

        specs: list[tuple[nn.Module, str, str]] = []
        paligemma = self.paligemma_with_expert.paligemma
        specs.append((paligemma.language_model, "embed_tokens", "uniform_recursive"))
        for model in (paligemma.language_model, self.paligemma_with_expert.gemma_expert.model):
            for layer in model.layers:
                specs.extend(
                    [
                        (layer.self_attn, "q_proj", "uniform_recursive"),
                        (layer.self_attn, "k_proj", "uniform_recursive"),
                        (layer.self_attn, "v_proj", "uniform_recursive"),
                        (layer.self_attn, "o_proj", "uniform_recursive"),
                        (layer, "mlp", "uniform_recursive"),
                    ]
                )
        specs.extend(
            [
                (self, "action_in_proj", "uniform_recursive"),
                (self, "action_out_proj", "uniform_recursive"),
                (self, "time_mlp_in", "uniform_recursive"),
                (self, "time_mlp_out", "uniform_recursive"),
            ]
        )
        if bool(getattr(self, "action_expert_router_enabled", False)):
            specs.extend(
                [
                    (self, "action_expert_router_summary_proj", "uniform_recursive"),
                    (self, "action_expert_router_summary_pair_proj", "uniform_recursive"),
                    (self, "action_expert_router_norm", "uniform_recursive"),
                    (self, "action_expert_router_logits", "uniform_recursive"),
                ]
            )
            for idx in range(len(getattr(self, "action_expert_router_down", ()))):
                specs.append((self.action_expert_router_down, str(idx), "uniform_recursive"))
            for idx in range(len(getattr(self, "action_expert_router_up", ()))):
                specs.append((self.action_expert_router_up, str(idx), "uniform_recursive"))
        return specs

    def _model_runtime_dtype(self) -> torch.dtype:
        return module_parameter_dtype(self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj)

    def _named_views(self, observation: PicfObservation) -> list[tuple[str, np.ndarray]]:
        views = [("static", np.asarray(observation.rgb_static))]
        if self.config.include_gripper_image and observation.rgb_gripper is not None:
            views.append(("gripper", np.asarray(observation.rgb_gripper)))
        return views

    def _views(self, observation: PicfObservation) -> list[np.ndarray]:
        return [image for _, image in self._named_views(observation)]

    def _view_transform(self, image: np.ndarray, *, target_h: int = 224, target_w: int = 224) -> PaliGemmaViewTransform:
        arr = np.asarray(image)
        if arr.ndim != 3:
            raise ValueError(f"Expected HWC image for view transform, got shape={tuple(arr.shape)}")
        cur_h, cur_w = int(arr.shape[0]), int(arr.shape[1])
        if cur_h <= 0 or cur_w <= 0:
            raise ValueError(f"Expected positive image size, got original_hw={(cur_h, cur_w)}")
        ratio = max(cur_w / float(target_w), cur_h / float(target_h))
        resized_h = int(cur_h / ratio)
        resized_w = int(cur_w / ratio)
        pad_top, rem_h = divmod(target_h - resized_h, 2)
        pad_bottom = pad_top + rem_h
        pad_left, rem_w = divmod(target_w - resized_w, 2)
        pad_right = pad_left + rem_w
        return PaliGemmaViewTransform(
            original_hw=(cur_h, cur_w),
            target_hw=(target_h, target_w),
            resized_hw=(resized_h, resized_w),
            pad_top=int(pad_top),
            pad_bottom=int(pad_bottom),
            pad_left=int(pad_left),
            pad_right=int(pad_right),
            scale_y=float(resized_h / max(cur_h, 1)),
            scale_x=float(resized_w / max(cur_w, 1)),
        )

    def _prepare_image(self, image: np.ndarray) -> torch.Tensor:
        arr = np.asarray(image)
        tensor = torch.as_tensor(arr)
        if tensor.ndim != 3 or tensor.shape[-1] != 3:
            raise ValueError(f"Expected HWC RGB image for semantic encoding, got shape={tuple(tensor.shape)}")
        tensor = tensor.to(dtype=torch.float32)
        if float(tensor.max().item()) > 1.5:
            tensor = tensor / 255.0
        elif float(tensor.min().item()) < -0.1:
            tensor = (tensor + 1.0) * 0.5
        tensor = torch.clamp(tensor, 0.0, 1.0)
        tensor = tensor * 2.0 - 1.0
        resized = image_tools.resize_with_pad_torch(tensor[None, :], 224, 224)
        if resized.ndim == 3:
            resized = resized[None, :]
        return resized.permute(0, 3, 1, 2).contiguous().to(device=self.device, dtype=self.dtype)

    def _state_for_prompt(self, observation: PicfObservation) -> np.ndarray:
        raw = np.asarray(
            observation.proprio if observation.proprio is not None else observation.robot_obs,
            dtype=np.float32,
        ).reshape(-1)
        if self.prompt_state_normalizer is not None:
            return self.prompt_state_normalizer.normalize_np(raw)
        return np.clip(raw, -1.0, 1.0)

    def _prepare_prompt(self, prompt: str, observation: PicfObservation) -> tuple[torch.Tensor, torch.Tensor]:
        debug_prompt = os.environ.get("OPENPI_DEBUG_PALIGEMMA_PROMPT", "").strip() not in {"", "0", "false", "False"}
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
        start = time.perf_counter()
        if debug_prompt:
            logging.info("paligemma_prompt rank=%s begin prompt=%r", rank, str(prompt))
        state = self._state_for_prompt(observation) if self.config.inject_state_into_prompt else None
        tokens_np, mask_np = self.tokenizer.tokenize(str(prompt), state=state)
        if debug_prompt:
            logging.info(
                "paligemma_prompt rank=%s tokenize_sec=%.3f token_len=%s",
                rank,
                time.perf_counter() - start,
                len(tokens_np),
            )
        # Avoid direct numpy->GPU `as_tensor(...)` in the live DDP path. We first
        # materialize compact CPU tensors, then perform an explicit device copy.
        cpu_start = time.perf_counter()
        tokens_cpu = torch.from_numpy(np.asarray(tokens_np, dtype=np.int64).copy())[None, :].contiguous()
        mask_cpu = torch.from_numpy(np.asarray(mask_np, dtype=np.bool_).copy())[None, :].contiguous()
        if debug_prompt:
            logging.info("paligemma_prompt rank=%s cpu_tensor_sec=%.3f", rank, time.perf_counter() - cpu_start)
        gpu_start = time.perf_counter()
        tokens = tokens_cpu.to(device=self.device, dtype=torch.long, non_blocking=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        if debug_prompt:
            logging.info("paligemma_prompt rank=%s tokens_to_device_sec=%.3f", rank, time.perf_counter() - gpu_start)
        mask_start = time.perf_counter()
        mask = mask_cpu.to(device=self.device, dtype=torch.bool, non_blocking=False)
        if self.device.type == "cuda":
            torch.cuda.synchronize(device=self.device)
        if debug_prompt:
            logging.info("paligemma_prompt rank=%s mask_to_device_sec=%.3f total_sec=%.3f", rank, time.perf_counter() - mask_start, time.perf_counter() - start)
        return tokens, mask

    def _apply_checkpoint(self, func, *args):
        runtime = getattr(self, "paligemma_with_expert", None)
        paligemma = getattr(runtime, "paligemma", None) if runtime is not None else None
        gemma_expert = getattr(runtime, "gemma_expert", None) if runtime is not None else None
        native_gc_active = any(
            bool(getattr(module, "gradient_checkpointing", False))
            for module in (
                getattr(paligemma, "language_model", None),
                getattr(paligemma, "vision_tower", None),
                getattr(gemma_expert, "model", None),
            )
            if module is not None
        )
        if bool(
            self.trainable
            and self.training
            and self.gradient_checkpointing_enabled
            and not native_gc_active
            and _checkpoint_inputs_require_grad(*args)
        ):
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False)
        return func(*args)

    def _embed_prefix(
        self,
        observation: PicfObservation,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        torch.Tensor,
        tuple[tuple[int, int], ...],
        tuple[tuple[int, int], ...],
        tuple[str, ...],
        tuple[PaliGemmaViewTransform, ...],
    ]:
        lang_tokens, lang_masks = self._prepare_prompt(observation.prompt, observation)
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []
        image_token_count = 0
        image_token_ranges: list[tuple[int, int]] = []
        image_grid_shapes: list[tuple[int, int]] = []
        image_view_names: list[str] = []
        image_view_transforms: list[PaliGemmaViewTransform] = []
        cursor = 0
        for view_name, image in self._named_views(observation):
            image_view_names.append(view_name)
            image_view_transforms.append(self._view_transform(image))
            image_tensor = self._prepare_image(image)

            def _image_embed(x: torch.Tensor) -> torch.Tensor:
                return self.paligemma_with_expert.embed_image(x)

            img_emb = self._apply_checkpoint(_image_embed, image_tensor)
            batch_size, num_img_tokens = img_emb.shape[:2]
            num_img_tokens = int(num_img_tokens)
            start, end = cursor, cursor + num_img_tokens
            image_token_ranges.append((start, end))
            grid_side = int(round(math.sqrt(num_img_tokens)))
            if grid_side * grid_side != num_img_tokens:
                raise RuntimeError(
                    "PaliGemma spatial token contract requires a square image-token grid. "
                    f"view={view_name!r} num_img_tokens={num_img_tokens}"
                )
            image_grid_shapes.append((grid_side, grid_side))
            cursor = end
            image_token_count += num_img_tokens
            embs.append(img_emb)
            pad_masks.append(torch.ones((batch_size, num_img_tokens), device=self.device, dtype=torch.bool))
            att_masks += [0] * num_img_tokens

        def _lang_embed(tokens: torch.Tensor) -> torch.Tensor:
            _assert_index_tensor_in_range(
                tokens,
                size=module_num_embeddings(self.paligemma_with_expert.paligemma.language_model.embed_tokens),
                name="paligemma_language_token_ids",
            )
            lang_emb = self.paligemma_with_expert.embed_language_tokens(tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(_lang_embed, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * int(lang_emb.shape[1])

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks = torch.as_tensor(att_masks, device=self.device, dtype=torch.int32)[None, :]
        prefix_att_masks = prefix_att_masks.expand(prefix_pad_masks.shape[0], -1)
        model_dtype = self._model_runtime_dtype()
        if model_dtype in (torch.float16, torch.bfloat16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
        return (
            prefix_embs,
            prefix_pad_masks,
            prefix_att_masks,
            image_token_count,
            lang_masks,
            tuple(image_token_ranges),
            tuple(image_grid_shapes),
            tuple(image_view_names),
            tuple(image_view_transforms),
        )

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        # Use a bf16/fp16-safe additive mask.  The original PI0 value
        # (-2.38e38) is finite in fp32/bf16, but it can drive SDPA/Gemma expert
        # kernels into NaN on long prefix+suffix forwards after the mask is cast
        # to the query dtype.  -1e4 is still effectively zero probability after
        # softmax, while remaining numerically stable across fp32/bf16/fp16.
        return torch.where(att_2d_masks_4d, 0.0, -1.0e4)

    def encode_observation(self, observation: PicfObservation) -> PaliGemmaSemanticFeatures:
        timing_enabled = _runtime_timing_enabled()
        timing: dict[str, float] = {}
        total_start = _timing_start(timing_enabled)
        use_grad = bool(self.trainable and self.training)
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            stage_start = _timing_start(timing_enabled)
            (
                prefix_embs,
                prefix_pad_masks,
                prefix_att_masks,
                image_token_count,
                lang_masks,
                image_token_ranges,
                image_grid_shapes,
                image_view_names,
                image_view_transforms,
            ) = self._embed_prefix(observation)
            _timing_record(timing, "encode_embed_prefix", stage_start, timing_enabled)
            stage_start = _timing_start(timing_enabled)
            att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = _masked_position_ids(prefix_pad_masks)
            attn_mask_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)
            _timing_record(timing, "encode_masks", stage_start, timing_enabled)

            def _forward_prefix(
                embeddings: torch.Tensor,
                attention_mask: torch.Tensor,
                positions: torch.Tensor,
            ) -> torch.Tensor:
                outputs, _ = self.paligemma_with_expert.forward(
                    inputs_embeds=[embeddings, None],
                    attention_mask=attention_mask,
                    position_ids=positions,
                    past_key_values=None,
                    use_cache=False,
                )
                return outputs[0]

            stage_start = _timing_start(timing_enabled)
            prefix_output = self._apply_checkpoint(_forward_prefix, prefix_embs, attn_mask_4d, position_ids)
            _timing_record(timing, "encode_prefix_forward", stage_start, timing_enabled)
            stage_start = _timing_start(timing_enabled)
            image_hidden = prefix_output[:, :image_token_count, :] if image_token_count > 0 else None
            text_hidden = prefix_output[:, image_token_count:, :]
            image_tokens = None if image_hidden is None else image_hidden[0]
            text_tokens = text_hidden[0]
            if not use_grad:
                image_tokens = None if image_tokens is None else image_tokens.detach().clone()
                text_tokens = text_tokens.detach().clone()
            result = PaliGemmaSemanticFeatures(
                tokens=_take_valid_prefix_tokens(prefix_output[0], prefix_pad_masks[0]),
                summary=_summary_from_outputs(
                    hidden_states=text_hidden,
                    image_hidden_states=image_hidden,
                    prompt_mask=lang_masks.to(dtype=text_hidden.dtype),
                ),
                prefix_embeddings=prefix_embs[0],
                prefix_pad_masks=prefix_pad_masks[0],
                prefix_att_masks=prefix_att_masks[0],
                image_tokens=image_tokens,
                text_tokens=text_tokens,
                image_token_ranges=image_token_ranges,
                image_grid_shapes=image_grid_shapes,
                image_view_names=image_view_names,
                image_view_transforms=image_view_transforms,
            )
            _timing_record(timing, "encode_pack_features", stage_start, timing_enabled)
            _timing_record(timing, "encode_total", total_start, timing_enabled)
            self._last_runtime_timing = timing
            return result

    def supports_pi0_action_generation(self) -> bool:
        return True

    def _combine_prefix(
        self,
        features: PaliGemmaSemanticFeatures,
        *,
        extra_prefix_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.prefix_embeddings is None or features.prefix_pad_masks is None:
            raise RuntimeError("PI0 action generation requires prefix embeddings and prefix pad masks.")
        prefix_embs = features.prefix_embeddings
        prefix_pad_masks = features.prefix_pad_masks
        prefix_att_masks = (
            features.prefix_att_masks
            if features.prefix_att_masks is not None
            else torch.zeros_like(prefix_pad_masks, dtype=torch.int32)
        )
        if prefix_embs.ndim == 2:
            prefix_embs = prefix_embs[None, :]
        if prefix_pad_masks.ndim == 1:
            prefix_pad_masks = prefix_pad_masks[None, :]
        if prefix_att_masks.ndim == 1:
            prefix_att_masks = prefix_att_masks[None, :]
        if extra_prefix_tokens is not None and extra_prefix_tokens.numel() > 0:
            if extra_prefix_tokens.ndim == 2:
                extra_prefix_tokens = extra_prefix_tokens[None, :]
            extra_prefix_tokens = extra_prefix_tokens.to(device=self.device, dtype=prefix_embs.dtype)
            prefix_embs = torch.cat([prefix_embs, extra_prefix_tokens], dim=1)
            ones = torch.ones(
                (prefix_pad_masks.shape[0], extra_prefix_tokens.shape[1]),
                device=self.device,
                dtype=torch.bool,
            )
            zeros = torch.zeros(
                (prefix_att_masks.shape[0], extra_prefix_tokens.shape[1]),
                device=self.device,
                dtype=torch.int32,
            )
            prefix_pad_masks = torch.cat([prefix_pad_masks, ones], dim=1)
            prefix_att_masks = torch.cat([prefix_att_masks, zeros], dim=1)
        return prefix_embs, prefix_pad_masks, prefix_att_masks

    def _embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.action_in_proj.out_features,
            min_period=4e-3,
            max_period=4.0,
            device=timestep.device,
        ).to(dtype=torch.float32, device=timestep.device)

        def _action_proj(x: torch.Tensor) -> torch.Tensor:
            return self.action_in_proj(x)

        action_emb = self._apply_checkpoint(_action_proj, noisy_actions)

        def _time_mlp(x: torch.Tensor) -> torch.Tensor:
            y = self.time_mlp_in(x)
            y = torch.nn.functional.silu(y)
            y = self.time_mlp_out(y)
            return torch.nn.functional.silu(y)

        adarms_cond = self._apply_checkpoint(_time_mlp, time_emb)
        bsize, action_steps = action_emb.shape[:2]
        pad_masks = torch.ones((bsize, action_steps), dtype=torch.bool, device=timestep.device)
        att_masks = torch.tensor(
            [1] + ([0] * max(action_steps - 1, 0)),
            dtype=torch.int32,
            device=timestep.device,
        )[None, :].expand(bsize, action_steps)
        return action_emb, pad_masks, att_masks, adarms_cond

    def _apply_action_context_adapter(
        self,
        suffix_embs: torch.Tensor,
        context_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Let action tokens directly read PICF belief/context tokens.

        Passive extra-prefix tokens preserve the original PI API but can be
        ignored by the action expert. This adapter is the explicit action-side
        route: action suffix tokens query bounded PICF context through gated
        cross-attention while the prefix length and suffix position ids remain
        unchanged.
        """

        if context_tokens is None or not isinstance(context_tokens, torch.Tensor) or context_tokens.numel() == 0:
            return suffix_embs, {}
        if suffix_embs.ndim != 3:
            raise ValueError(f"Expected suffix embeddings [B,H,D], got {tuple(suffix_embs.shape)}")
        if context_tokens.ndim == 2:
            context_tokens = context_tokens[None, :, :]
        if context_tokens.ndim != 3:
            raise ValueError(f"Expected action context [T,D] or [B,T,D], got {tuple(context_tokens.shape)}")
        if context_tokens.shape[0] == 1 and suffix_embs.shape[0] != 1:
            context_tokens = context_tokens.expand(suffix_embs.shape[0], -1, -1)
        if context_tokens.shape[0] != suffix_embs.shape[0]:
            raise ValueError(
                "Action context adapter batch mismatch: "
                f"context={tuple(context_tokens.shape)} suffix={tuple(suffix_embs.shape)}"
            )

        dtype = suffix_embs.dtype
        device = suffix_embs.device
        context = context_tokens.to(device=device, dtype=dtype)
        if context.shape[-1] != suffix_embs.shape[-1]:
            in_proj = getattr(self, "action_context_in_proj", None)
            if not isinstance(in_proj, nn.Module):
                raise ValueError(
                    "Action context adapter dimension mismatch: "
                    f"context={tuple(context.shape)} suffix={tuple(suffix_embs.shape)}"
                )
            in_features = getattr(in_proj, "in_features", None)
            out_features = getattr(in_proj, "out_features", None)
            wrapped_module = getattr(in_proj, "module", None)
            if not isinstance(wrapped_module, nn.Module):
                wrapped_module = getattr(in_proj, "_fsdp_wrapped_module", None)
            if isinstance(wrapped_module, nn.Module):
                in_features = getattr(wrapped_module, "in_features", in_features)
                out_features = getattr(wrapped_module, "out_features", out_features)
            if in_features is not None and int(context.shape[-1]) != int(in_features):
                raise ValueError(
                    "Action context adapter input projection mismatch: "
                    f"context={tuple(context.shape)} in_features={int(in_features)}"
                )
            if out_features is not None and int(suffix_embs.shape[-1]) != int(out_features):
                raise ValueError(
                    "Action context adapter output projection mismatch: "
                    f"suffix={tuple(suffix_embs.shape)} out_features={int(out_features)}"
                )
            context = in_proj(context)
            if context.shape[-1] != suffix_embs.shape[-1]:
                raise ValueError(
                    "Action context adapter projection returned wrong width: "
                    f"context={tuple(context.shape)} suffix={tuple(suffix_embs.shape)}"
                )
        suffix_norm = torch.nn.functional.layer_norm(suffix_embs, (suffix_embs.shape[-1],))
        context_norm = torch.nn.functional.layer_norm(context, (context.shape[-1],))
        q = self.action_context_q_proj(suffix_norm)
        k = self.action_context_k_proj(context_norm)
        v = self.action_context_v_proj(context)
        logits = torch.matmul(q.to(dtype=torch.float32), k.to(dtype=torch.float32).transpose(-1, -2))
        logits = logits * (float(suffix_embs.shape[-1]) ** -0.5)
        attn = torch.softmax(logits, dim=-1).to(device=device, dtype=dtype)
        residual = self.action_context_out_proj(torch.matmul(attn, v))

        eps = 1.0e-6
        suffix_rms = torch.sqrt(torch.mean(suffix_embs.detach().to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
        residual_rms = torch.sqrt(torch.mean(residual.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
        if bool(getattr(self, "action_context_adapter_rms_cap", True)):
            cap = torch.clamp(suffix_rms / torch.clamp(residual_rms, min=eps), max=1.0)
            residual = residual * cap.to(device=device, dtype=dtype)
        gate = torch.sigmoid(self.action_context_gate_logit.to(device=device, dtype=torch.float32)).to(dtype=dtype)
        adapted = suffix_embs + gate * residual

        entropy = -(attn.to(dtype=torch.float32) * torch.log(torch.clamp(attn.to(dtype=torch.float32), min=1.0e-12))).sum(
            dim=-1
        )
        zero = adapted.reshape(-1)[0].detach() * 0.0
        metrics = {
            "picf_action_context_adapter_token_count": zero + float(context.shape[1]),
            "picf_action_context_adapter_gate": zero + gate.detach().to(device=device, dtype=dtype),
            "picf_action_context_adapter_attention_entropy_mean": entropy.detach().mean().to(device=device, dtype=dtype),
            "picf_action_context_adapter_residual_rms_mean": torch.sqrt(
                torch.mean(residual.detach().to(dtype=torch.float32).square(), dim=-1)
            )
            .mean()
            .to(device=device, dtype=dtype),
        }
        return adapted, metrics

    def _project_action_context_to_action_width(
        self,
        context_tokens: torch.Tensor,
        *,
        batch: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Normalize PICF action context to `[B, T, action_width]`.

        PICF context is produced in the PaliGemma prefix width, while the PI0.5
        action expert uses the action-expert width.  Both the suffix adapter and
        the readout auxiliary must use the same projection contract; otherwise a
        readout success would not prove that the deployed action bridge can see
        the same representation.
        """

        context = context_tokens
        if context.ndim == 2:
            context = context[None, :, :]
        if context.ndim != 3:
            raise ValueError(f"Expected action context [T,D] or [B,T,D], got {tuple(context.shape)}")
        if context.shape[0] == 1 and batch != 1:
            context = context.expand(batch, -1, -1)
        if context.shape[0] != batch:
            raise ValueError(
                "Action context batch mismatch: "
                f"context={tuple(context.shape)} batch={batch}"
            )
        context = context.to(device=device, dtype=dtype)
        if context.shape[-1] != width:
            in_proj = getattr(self, "action_context_in_proj", None)
            if not isinstance(in_proj, nn.Module):
                raise ValueError(
                    "Action context width mismatch without input projection: "
                    f"context={tuple(context.shape)} width={width}"
                )
            context = in_proj(context)
        if context.shape[-1] != width:
            raise ValueError(
                "Action context projection returned wrong width: "
                f"context={tuple(context.shape)} width={width}"
            )
        return context

    def _action_context_readout_state(
        self,
        context_tokens: torch.Tensor,
        *,
        batch: int,
        horizon: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read bounded PICF context into horizon-indexed action states.

        The continuous readout, deployed flow residual, and action-token
        auxiliary share this state.  Keeping one readout contract avoids a
        false positive where a side objective learns one PICF context
        vocabulary while the deployed action path consumes another.
        """

        width = int(self.action_context_readout_query.shape[-1])
        context = self._project_action_context_to_action_width(
            context_tokens,
            batch=batch,
            width=width,
            device=device,
            dtype=dtype,
        )
        query = self.action_context_readout_query[:horizon].to(device=device, dtype=dtype)
        query = query[None, :, :].expand(batch, -1, -1)
        query_norm = torch.nn.functional.layer_norm(query, (query.shape[-1],))
        context_norm = torch.nn.functional.layer_norm(context, (context.shape[-1],))
        q = self.action_context_readout_q_proj(query_norm)
        k = self.action_context_readout_k_proj(context_norm)
        v = self.action_context_readout_v_proj(context)
        logits = torch.matmul(q.to(dtype=torch.float32), k.to(dtype=torch.float32).transpose(-1, -2))
        logits = logits * (float(width) ** -0.5)
        attn = torch.softmax(logits, dim=-1).to(device=device, dtype=dtype)
        readout = torch.matmul(attn, v)
        entropy = -(attn.to(dtype=torch.float32) * torch.log(torch.clamp(attn.to(dtype=torch.float32), min=1.0e-12))).sum(
            dim=-1
        )
        return readout, context, entropy

    def _predict_action_chunk_from_context(
        self,
        context_tokens: torch.Tensor,
        *,
        batch: int,
        horizon: int,
        action_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read a dense PICF context into an action-chunk target estimate."""

        readout, context, entropy = self._action_context_readout_state(
            context_tokens,
            batch=batch,
            horizon=horizon,
            device=device,
            dtype=dtype,
        )
        pred = self.action_context_readout_out_proj(readout).to(dtype=torch.float32)[..., :action_dim]
        return pred, context, entropy

    def _compute_action_context_readout_aux(
        self,
        context_tokens: torch.Tensor | None,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Predict the action chunk from PICF context alone.

        This is the G22 fast gate for the G20 causal failure.  It intentionally
        does not read noisy action suffix embeddings, so a decreasing loss means
        the PICF belief context itself carries motor-readable information.
        """

        zero = target.reshape(-1)[0] * 0.0
        weight = float(getattr(self, "action_context_readout_aux_weight", 0.0))
        if weight <= 0.0:
            return zero, {}
        if context_tokens is None or not isinstance(context_tokens, torch.Tensor) or context_tokens.numel() == 0:
            return zero, {
                "picf_action_context_readout_enabled": zero + 0.0,
                "picf_action_context_readout_loss": zero,
                "picf_action_context_readout_mse": zero,
                "picf_action_context_readout_weighted_total": zero,
                "picf_action_context_readout_weight": zero + weight,
                "picf_action_context_readout_token_count": zero,
                "picf_action_context_readout_attention_entropy_mean": zero,
            }
        if target.ndim != 3:
            raise ValueError(f"Expected action readout target [B,H,A], got {tuple(target.shape)}")

        batch, horizon, action_dim = target.shape
        dtype = target.dtype
        device = target.device
        pred, context, entropy = self._predict_action_chunk_from_context(
            context_tokens,
            batch=batch,
            horizon=horizon,
            action_dim=action_dim,
            device=device,
            dtype=dtype,
        )
        target_float = target.to(dtype=torch.float32)
        mse = torch.nn.functional.mse_loss(pred, target_float)
        mode = str(getattr(self, "action_context_readout_aux_loss", "smooth_l1"))
        huber_delta = float(getattr(self, "action_context_readout_aux_huber_delta", 1.0))
        loss = _action_flow_objective_loss(target_float, pred, mode=mode, huber_delta=huber_delta)
        weighted = loss * weight
        metrics = {
            "picf_action_context_readout_enabled": zero + 1.0,
            "picf_action_context_readout_loss": loss.detach().to(device=device, dtype=dtype),
            "picf_action_context_readout_mse": mse.detach().to(device=device, dtype=dtype),
            "picf_action_context_readout_weighted_total": weighted.detach().to(device=device, dtype=dtype),
            "picf_action_context_readout_weight": zero + weight,
            "picf_action_context_readout_token_count": zero + float(context.shape[1]),
            "picf_action_context_readout_attention_entropy_mean": entropy.detach().mean().to(device=device, dtype=dtype),
        }
        return weighted, metrics

    def _compute_action_context_token_aux(
        self,
        context_tokens: torch.Tensor | None,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """FAST-style action-token objective for PICF context.

        Native OpenPI FAST supervision is a next-token CE over tokenized
        actions.  PICF's current PyTorch wrapper drops LM heads and trains the
        PI0.5 continuous flow path, so this auxiliary implements the equivalent
        representation pressure locally: bounded PICF context must classify a
        discretized action chunk.  It uses the same readout state as the
        deployed flow residual to keep the objective tied to `dA/dC_picf`.
        """

        zero = target.reshape(-1)[0] * 0.0
        weight = float(getattr(self, "action_context_token_aux_weight", 0.0))
        bins = max(int(getattr(self, "action_context_token_aux_bins", 256)), 2)
        clip = max(float(getattr(self, "action_context_token_aux_clip", 1.0)), 1.0e-6)
        if weight <= 0.0:
            return zero, {}
        if context_tokens is None or not isinstance(context_tokens, torch.Tensor) or context_tokens.numel() == 0:
            # Keep the enabled token head graph-connected even when a rank sees
            # an empty PICF context.  Without this DDP can hang because another
            # rank may exercise the token head on the same optimizer step.
            graph_zero = zero
            for module in self._action_context_token_aux_modules():
                for parameter in module.parameters():
                    graph_zero = graph_zero + parameter.reshape(-1)[0].to(device=zero.device, dtype=zero.dtype) * 0.0
            return graph_zero, {
                "picf_action_context_token_aux_enabled": zero + 0.0,
                "picf_action_context_token_aux_loss": zero,
                "picf_action_context_token_aux_accuracy": zero,
                "picf_action_context_token_aux_weighted_total": graph_zero.detach().to(device=zero.device, dtype=zero.dtype),
                "picf_action_context_token_aux_weight": zero + weight,
                "picf_action_context_token_aux_bins": zero + float(bins),
                "picf_action_context_token_aux_clip": zero + clip,
                "picf_action_context_token_aux_token_count": zero,
                "picf_action_context_token_aux_attention_entropy_mean": zero,
            }
        if target.ndim != 3:
            raise ValueError(f"Expected action token target [B,H,A], got {tuple(target.shape)}")

        batch, horizon, action_dim = target.shape
        dtype = target.dtype
        device = target.device
        readout, context, entropy = self._action_context_readout_state(
            context_tokens,
            batch=batch,
            horizon=horizon,
            device=device,
            dtype=dtype,
        )
        logits = self.action_context_token_readout_out_proj(readout).to(dtype=torch.float32)
        logits = logits.reshape(batch, horizon, self.model_action_dim, bins)[..., :action_dim, :]

        target_float = target.to(dtype=torch.float32).clamp(min=-clip, max=clip)
        labels = torch.round(((target_float + clip) / (2.0 * clip)) * float(bins - 1)).to(dtype=torch.long)
        labels = labels.clamp(min=0, max=bins - 1)

        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, bins),
            labels.reshape(-1),
            reduction="mean",
        )
        pred = torch.argmax(logits.detach(), dim=-1)
        accuracy = (pred == labels).to(dtype=torch.float32).mean()
        weighted = loss * weight
        metrics = {
            "picf_action_context_token_aux_enabled": zero + 1.0,
            "picf_action_context_token_aux_loss": loss.detach().to(device=device, dtype=dtype),
            "picf_action_context_token_aux_accuracy": accuracy.detach().to(device=device, dtype=dtype),
            "picf_action_context_token_aux_weighted_total": weighted.detach().to(device=device, dtype=dtype),
            "picf_action_context_token_aux_weight": zero + weight,
            "picf_action_context_token_aux_bins": zero + float(bins),
            "picf_action_context_token_aux_clip": zero + clip,
            "picf_action_context_token_aux_token_count": zero + float(context.shape[1]),
            "picf_action_context_token_aux_attention_entropy_mean": entropy.detach().mean().to(
                device=device,
                dtype=dtype,
            ),
        }
        return weighted, metrics

    def _apply_action_context_flow_residual(
        self,
        v_t: torch.Tensor,
        x_t: torch.Tensor,
        time_expanded: torch.Tensor,
        context_tokens: torch.Tensor | None,
        *,
        target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Inject PICF context into the deployed PI0.5 flow velocity.

        PI0.5 flow training uses `x_t = t * noise + (1-t) * y` and target
        velocity `u_t = noise - y`.  A context readout that predicts `y_c`
        therefore implies a compatible velocity

            u_c = (x_t - y_c) / max(t, eps).

        The deployed velocity is a bounded residual blend between native
        `v_t` and `u_c`.  This makes the context readout causal for both
        training and sampling instead of remaining a side auxiliary.
        """

        zero = v_t.reshape(-1)[0].detach() * 0.0
        if not bool(getattr(self, "action_context_flow_residual_enabled", False)):
            return v_t, {}
        if context_tokens is None or not isinstance(context_tokens, torch.Tensor) or context_tokens.numel() == 0:
            return v_t, {
                "picf_action_context_flow_residual_enabled": zero + 1.0,
                "picf_action_context_flow_residual_gate": zero,
                "picf_action_context_flow_residual_token_count": zero,
                "picf_action_context_flow_residual_rms_mean": zero,
                "picf_action_context_flow_context_velocity_rms_mean": zero,
                "picf_action_context_flow_context_target_mse": zero,
                "picf_action_context_flow_residual_time_floor": zero
                + float(getattr(self, "action_context_flow_residual_time_floor", 0.05)),
            }

        batch, horizon, action_dim = v_t.shape
        pred_target, context, _entropy = self._predict_action_chunk_from_context(
            context_tokens,
            batch=batch,
            horizon=horizon,
            action_dim=action_dim,
            device=v_t.device,
            dtype=v_t.dtype,
        )
        pred_target = pred_target.to(device=v_t.device, dtype=v_t.dtype)
        time_floor = float(getattr(self, "action_context_flow_residual_time_floor", 0.05))
        time_safe = torch.clamp(time_expanded.to(device=v_t.device, dtype=v_t.dtype), min=time_floor)
        context_velocity = (x_t.to(device=v_t.device, dtype=v_t.dtype) - pred_target) / time_safe
        residual = context_velocity - v_t
        if bool(getattr(self, "action_context_flow_residual_rms_cap", True)):
            eps = 1.0e-6
            residual_rms = torch.sqrt(torch.mean(residual.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            base_rms = torch.sqrt(torch.mean(v_t.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
            scale = torch.clamp(base_rms / torch.clamp(residual_rms, min=eps), max=1.0)
            residual = residual * scale.to(device=residual.device, dtype=residual.dtype)
        gate = torch.sigmoid(self.action_context_flow_residual_gate_logit.to(device=v_t.device, dtype=v_t.dtype))
        adapted = v_t + gate.view(1, 1, 1) * residual

        residual_rms_mean = torch.sqrt(torch.mean(residual.to(dtype=torch.float32).square(), dim=-1)).mean()
        context_velocity_rms_mean = torch.sqrt(torch.mean(context_velocity.to(dtype=torch.float32).square(), dim=-1)).mean()
        target_mse = zero
        if target is not None:
            target_mse = torch.nn.functional.mse_loss(
                pred_target.to(dtype=torch.float32),
                target.to(device=v_t.device, dtype=torch.float32)[..., :action_dim],
            ).detach().to(device=v_t.device, dtype=v_t.dtype)
        metrics = {
            "picf_action_context_flow_residual_enabled": zero + 1.0,
            "picf_action_context_flow_residual_gate": gate.detach().reshape(-1)[0].to(device=v_t.device, dtype=v_t.dtype),
            "picf_action_context_flow_residual_token_count": zero + float(context.shape[1]),
            "picf_action_context_flow_residual_rms_mean": residual_rms_mean.detach().to(device=v_t.device, dtype=v_t.dtype),
            "picf_action_context_flow_context_velocity_rms_mean": context_velocity_rms_mean.detach().to(
                device=v_t.device,
                dtype=v_t.dtype,
            ),
            "picf_action_context_flow_context_target_mse": target_mse,
            "picf_action_context_flow_residual_time_floor": zero + time_floor,
        }
        return adapted, metrics

    def _apply_action_expert_router(
        self,
        suffix_embs: torch.Tensor,
        features: PaliGemmaSemanticFeatures,
        context_tokens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Apply a semantic-conditioned action-expert residual adapter.

        This is the action-only routing path requested by the VLA repair plan:
        it does not MoE the VLM backbone and does not turn PICF context into a
        new supervised target.  A detached semantic/PICF condition selects a
        mixture of low-rank action-suffix experts, and the residual is bounded
        before entering the restored Gemma action expert.
        """

        if not bool(getattr(self, "action_expert_router_enabled", False)):
            return suffix_embs, {}
        if suffix_embs.ndim != 3:
            raise ValueError(f"Expected suffix embeddings [B,H,D], got {tuple(suffix_embs.shape)}")
        dtype = suffix_embs.dtype
        device = suffix_embs.device
        batch, _, width = suffix_embs.shape

        summary = features.summary
        if summary.ndim == 1:
            summary = summary[None, :]
        if summary.ndim != 2:
            raise ValueError(f"Expected semantic summary [D] or [B,D], got {tuple(summary.shape)}")
        if summary.shape[0] == 1 and batch != 1:
            summary = summary.expand(batch, -1)
        if summary.shape[0] != batch:
            raise ValueError(
                "Action expert router summary batch mismatch: "
                f"summary={tuple(summary.shape)} suffix={tuple(suffix_embs.shape)}"
            )
        summary = summary.detach().to(device=device, dtype=dtype)
        single_proj = self.action_expert_router_summary_proj
        pair_proj = getattr(self, "action_expert_router_summary_pair_proj", None)
        single_width = int(single_proj.in_features)
        summary_width = int(summary.shape[-1])
        if summary_width == single_width:
            cond = single_proj(summary)
        elif isinstance(pair_proj, nn.Linear) and summary_width == int(pair_proj.in_features):
            cond = pair_proj(summary)
        elif summary_width % single_width == 0:
            # Future multi-summary variants can concatenate several same-width
            # summaries.  Average them before the shared projection instead of
            # silently truncating object/context evidence.
            summary = summary.reshape(summary.shape[0], summary_width // single_width, single_width).mean(dim=1)
            cond = single_proj(summary)
        else:
            raise ValueError(
                "Action expert router summary width mismatch: "
                f"summary={summary_width} single={single_width} "
                f"pair={getattr(pair_proj, 'in_features', None)}"
            )

        if context_tokens is not None and isinstance(context_tokens, torch.Tensor) and context_tokens.numel() > 0:
            context = context_tokens.detach()
            if context.ndim == 2:
                context = context[None, :, :]
            if context.ndim != 3:
                raise ValueError(f"Expected action router context [T,D] or [B,T,D], got {tuple(context.shape)}")
            if context.shape[0] == 1 and batch != 1:
                context = context.expand(batch, -1, -1)
            if context.shape[0] != batch:
                raise ValueError(
                    "Action expert router context batch mismatch: "
                    f"context={tuple(context.shape)} suffix={tuple(suffix_embs.shape)}"
                )
            context = context.to(device=device, dtype=dtype)
            if context.shape[-1] != width:
                context = self.action_context_in_proj(context)
            if context.shape[-1] != width:
                raise ValueError(
                    "Action expert router context projection returned wrong width: "
                    f"context={tuple(context.shape)} suffix={tuple(suffix_embs.shape)}"
                )
            cond = cond + context.mean(dim=1)

        cond = self.action_expert_router_norm(cond)
        logits = self.action_expert_router_logits(cond).to(dtype=torch.float32)
        logits = logits / float(getattr(self, "action_expert_router_temperature", 1.0))
        weights = torch.softmax(logits, dim=-1).to(device=device, dtype=dtype)

        suffix_norm = torch.nn.functional.layer_norm(suffix_embs, (width,))
        residual = torch.zeros_like(suffix_embs)
        for expert_idx, (down, up) in enumerate(
            zip(self.action_expert_router_down, self.action_expert_router_up, strict=True)
        ):
            expert = up(torch.nn.functional.silu(down(suffix_norm)))
            residual = residual + weights[:, expert_idx].view(batch, 1, 1) * expert

        eps = 1.0e-6
        suffix_rms = torch.sqrt(torch.mean(suffix_embs.detach().to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
        residual_rms = torch.sqrt(torch.mean(residual.to(dtype=torch.float32).square(), dim=-1, keepdim=True) + eps)
        if bool(getattr(self, "action_expert_router_rms_cap", True)):
            cap = torch.clamp(suffix_rms / torch.clamp(residual_rms, min=eps), max=1.0)
            residual = residual * cap.to(device=device, dtype=dtype)
        gate = torch.sigmoid(self.action_expert_router_gate_logit.to(device=device, dtype=torch.float32)).to(dtype=dtype)
        adapted = suffix_embs + gate * residual

        entropy = -(weights.to(dtype=torch.float32) * torch.log(torch.clamp(weights.to(dtype=torch.float32), min=1.0e-12))).sum(
            dim=-1
        )
        top_weight = weights.detach().to(dtype=torch.float32).amax(dim=-1)
        zero = adapted.reshape(-1)[0].detach() * 0.0
        metrics = {
            "picf_action_expert_router_enabled": zero + 1.0,
            "picf_action_expert_router_gate": zero + gate.detach().to(device=device, dtype=dtype),
            "picf_action_expert_router_entropy_mean": entropy.detach().mean().to(device=device, dtype=dtype),
            "picf_action_expert_router_top_weight_mean": top_weight.detach().mean().to(device=device, dtype=dtype),
            "picf_action_expert_router_residual_rms_mean": torch.sqrt(
                torch.mean(residual.detach().to(dtype=torch.float32).square(), dim=-1)
            )
            .mean()
            .to(device=device, dtype=dtype),
        }
        return adapted, metrics

    def _prepare_action_chunk_target(
        self,
        action_chunk_target: torch.Tensor | np.ndarray,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        target = torch.as_tensor(action_chunk_target, device=device, dtype=dtype)
        if target.ndim == 1:
            target = target[None, :]
        if target.ndim != 2:
            raise RuntimeError(f"Expected action chunk target to have shape [H, A], got {tuple(target.shape)}")
        if target.shape[0] > self.action_horizon:
            target = target[: self.action_horizon]
        elif target.shape[0] < self.action_horizon:
            pad_rows = torch.zeros(
                (self.action_horizon - target.shape[0], target.shape[1]),
                device=device,
                dtype=dtype,
            )
            target = torch.cat([target, pad_rows], dim=0)
        return _pad_last_dim(target, dim=self.model_action_dim)

    def compute_action_flow_loss(
        self,
        features: PaliGemmaSemanticFeatures,
        *,
        extra_prefix_tokens: torch.Tensor | None,
        extra_action_context_tokens: torch.Tensor | None = None,
        action_chunk_target: torch.Tensor | np.ndarray,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dtype = torch.float32
        target = self._prepare_action_chunk_target(action_chunk_target, device=self.device, dtype=dtype)[None, :]
        if noise is None:
            noise = torch.randn_like(target)
        if time is None:
            beta = torch.distributions.Beta(
                torch.tensor(float(getattr(self, "action_flow_time_alpha", 1.5)), device=self.device),
                torch.tensor(float(getattr(self, "action_flow_time_beta", 1.0)), device=self.device),
            )
            time = (beta.sample((1,)) * 0.999 + 0.001).to(device=self.device, dtype=dtype)
        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1.0 - time_expanded) * target
        u_t = noise - target

        prefix_embs, prefix_pad_masks, prefix_att_masks = self._combine_prefix(
            features,
            extra_prefix_tokens=extra_prefix_tokens,
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(x_t, time)
        suffix_embs, adapter_metrics = self._apply_action_context_adapter(suffix_embs, extra_action_context_tokens)
        suffix_embs, router_metrics = self._apply_action_expert_router(
            suffix_embs,
            features,
            extra_action_context_tokens,
        )
        adapter_metrics.update(router_metrics)
        model_dtype = self._model_runtime_dtype()
        if model_dtype in (torch.bfloat16, torch.float16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
            suffix_embs = suffix_embs.to(dtype=model_dtype)
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = _masked_position_ids(pad_masks)
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)

        def _forward(prefix: torch.Tensor, suffix: torch.Tensor, attn: torch.Tensor, pos: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=attn,
                position_ids=pos,
                past_key_values=None,
                inputs_embeds=[prefix, suffix],
                use_cache=False,
                adarms_cond=[None, cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(_forward, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond)
        suffix_out = suffix_out[:, -self.action_horizon :].to(dtype=torch.float32)

        def _project(out: torch.Tensor) -> torch.Tensor:
            return self.action_out_proj(out)

        v_t = self._apply_checkpoint(_project, suffix_out)
        base_v_t = v_t
        v_t, flow_context_metrics = self._apply_action_context_flow_residual(
            v_t,
            x_t,
            time_expanded,
            extra_action_context_tokens,
            target=target,
        )
        if flow_context_metrics:
            adapter_metrics.update(flow_context_metrics)
        base_mse_total = torch.nn.functional.mse_loss(u_t, base_v_t)
        mse_total = torch.nn.functional.mse_loss(u_t, v_t)
        mse_pos = torch.nn.functional.mse_loss(u_t[..., :3], v_t[..., :3])
        mse_rot = torch.nn.functional.mse_loss(u_t[..., 3:6], v_t[..., 3:6])
        mse_grip = torch.nn.functional.mse_loss(u_t[..., 6:7], v_t[..., 6:7])
        mode = str(getattr(self, "action_flow_loss", "mse"))
        huber_delta = float(getattr(self, "action_flow_huber_delta", 1.0))
        training_total = _action_flow_objective_loss(u_t, v_t, mode=mode, huber_delta=huber_delta)
        training_pos = _action_flow_objective_loss(u_t[..., :3], v_t[..., :3], mode=mode, huber_delta=huber_delta)
        training_rot = _action_flow_objective_loss(u_t[..., 3:6], v_t[..., 3:6], mode=mode, huber_delta=huber_delta)
        training_grip = _action_flow_objective_loss(u_t[..., 6:7], v_t[..., 6:7], mode=mode, huber_delta=huber_delta)
        readout_weighted, readout_metrics = self._compute_action_context_readout_aux(
            extra_action_context_tokens,
            target,
        )
        if readout_metrics:
            training_total = training_total + readout_weighted
            adapter_metrics.update(readout_metrics)
        token_weighted, token_metrics = self._compute_action_context_token_aux(
            extra_action_context_tokens,
            target,
        )
        if token_metrics:
            training_total = training_total + token_weighted
            adapter_metrics.update(token_metrics)
        if flow_context_metrics:
            adapter_metrics.update(
                {
                    "picf_action_context_flow_base_mse": base_mse_total.detach().to(
                        device=mse_total.device,
                        dtype=mse_total.dtype,
                    ),
                    "picf_action_context_flow_adapted_mse": mse_total.detach().to(
                        device=mse_total.device,
                        dtype=mse_total.dtype,
                    ),
                    "picf_action_context_flow_gain_mse_delta": (base_mse_total - mse_total).detach().to(
                        device=mse_total.device,
                        dtype=mse_total.dtype,
                    ),
                }
            )
        predicted_chunk = _recover_flow_target(x_t, v_t, time_expanded).detach()
        predicted = predicted_chunk[:, 0, :7]
        zero = mse_total.detach() * 0.0
        return {
            # Canonical PI0.5 parity reports.  These keep historical
            # `loss_action_default_equiv` comparisons valid even when G15 uses
            # a robust training objective.
            "total": mse_total,
            "action_pos": mse_pos,
            "action_rot": mse_rot,
            "action_gripper": mse_grip,
            # Actual action objective used for gradient when the trainer opts
            # into the training_* keys.
            "training_total": training_total,
            "training_action_pos": training_pos,
            "training_action_rot": training_rot,
            "training_action_gripper": training_grip,
            "action_flow_objective_mode_id": zero + {
                "mse": 0.0,
                "l2": 0.0,
                "l1": 1.0,
                "mae": 1.0,
                "huber": 2.0,
                "smooth_l1": 3.0,
                "smoothl1": 3.0,
            }.get(str(mode).strip().lower().replace("-", "_"), -1.0),
            "action_flow_time_mean": zero + time.detach().to(device=mse_total.device, dtype=mse_total.dtype).mean(),
            "predicted_action": predicted[0],
            "predicted_chunk": predicted_chunk[0],
            **adapter_metrics,
        }

    @torch.no_grad()
    def sample_action_chunk(
        self,
        features: PaliGemmaSemanticFeatures,
        *,
        extra_prefix_tokens: torch.Tensor | None,
        extra_action_context_tokens: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        timing_enabled = _runtime_timing_enabled()
        timing: dict[str, float] = {}
        total_start = _timing_start(timing_enabled)
        stage_start = _timing_start(timing_enabled)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._combine_prefix(
            features,
            extra_prefix_tokens=extra_prefix_tokens,
        )
        model_dtype = self._model_runtime_dtype()
        if model_dtype in (torch.bfloat16, torch.float16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
        _timing_record(timing, "sample_combine_prefix", stage_start, timing_enabled)
        stage_start = _timing_start(timing_enabled)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=prefix_embs.dtype)
        _timing_record(timing, "sample_prefix_masks", stage_start, timing_enabled)
        stage_start = _timing_start(timing_enabled)
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=_masked_position_ids(prefix_pad_masks),
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        _timing_record(timing, "sample_prefix_kv_forward", stage_start, timing_enabled)
        if noise is None:
            stage_start = _timing_start(timing_enabled)
            noise = torch.randn((1, self.action_horizon, self.model_action_dim), device=self.device, dtype=torch.float32)
            _timing_record(timing, "sample_noise", stage_start, timing_enabled)
        x_t = noise
        steps = int(num_steps or self.denoise_steps)
        dt = torch.tensor(-1.0 / max(steps, 1), dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        denoise_start = _timing_start(timing_enabled)
        while time >= (-dt / 2):
            step_start = _timing_start(timing_enabled)
            timestep = time.expand(1)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(x_t, timestep)
            suffix_embs, _adapter_metrics = self._apply_action_context_adapter(
                suffix_embs,
                extra_action_context_tokens,
            )
            suffix_embs, _router_metrics = self._apply_action_expert_router(
                suffix_embs,
                features,
                extra_action_context_tokens,
            )
            suffix_len = suffix_pad_masks.shape[1]
            prefix_len = prefix_pad_masks.shape[1]
            prefix_pad_2d = prefix_pad_masks[:, None, :].expand(1, suffix_len, prefix_len)
            suffix_att_2d = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            full_att_2d = torch.cat([prefix_pad_2d, suffix_att_2d], dim=2)
            position_ids = prefix_pad_masks.to(torch.int64).sum(dim=-1, keepdim=True) + torch.cumsum(
                suffix_pad_masks.to(torch.int64),
                dim=1,
            ) - 1
            full_att_2d_4d = self._prepare_attention_masks_4d(full_att_2d).to(dtype=suffix_embs.dtype)
            if model_dtype in (torch.bfloat16, torch.float16):
                suffix_embs = suffix_embs.to(dtype=model_dtype)
            outputs_embeds, _ = self.paligemma_with_expert.forward(
                attention_mask=full_att_2d_4d,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=[None, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            suffix_out = outputs_embeds[1][:, -self.action_horizon :].to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
            v_t, _flow_context_metrics = self._apply_action_context_flow_residual(
                v_t,
                x_t,
                timestep[:, None, None],
                extra_action_context_tokens,
                target=None,
            )
            x_t = x_t + dt * v_t
            time += dt
            _timing_record(timing, "sample_denoise_last_step", step_start, False)
        _timing_record(timing, "sample_denoise_loop", denoise_start, timing_enabled)
        _timing_record(timing, "sample_total", total_start, timing_enabled)
        self._last_runtime_timing = timing
        return x_t[0]

    def forward(self, op: str, /, *args: Any, **kwargs: Any):
        if op == "encode_observation":
            return self.encode_observation(*args, **kwargs)
        if op == "compute_action_flow_loss":
            return self.compute_action_flow_loss(*args, **kwargs)
        if op == "sample_action_chunk":
            return self.sample_action_chunk(*args, **kwargs)
        raise ValueError(f"Unsupported semantic forward op: {op!r}")


class PaliGemmaSemanticEncoder(nn.Module):
    def __init__(self, config: PaliGemmaSemanticConfig | None = None):
        super().__init__()
        self.config = _stage_local_pi0_config(config or PaliGemmaSemanticConfig())
        source = _resolve_source(self.config)
        if source == "pi0_pytorch":
            self.encoder = _Pi0PaliGemmaSemanticEncoder(self.config)
        elif source == "hf":
            self.encoder = _HFPaliGemmaSemanticEncoder(self.config)
        else:
            raise ValueError(f"Unsupported semantic source: {source!r}")
        self.source = source
        self.device = self.encoder.device
        self.dtype = self.encoder.dtype
        self.trainable = self.encoder.trainable
        self.gradient_checkpointing_enabled = bool(getattr(self.encoder, "gradient_checkpointing_enabled", False))
        self.gradient_checkpointing_non_reentrant = bool(
            getattr(self.encoder, "gradient_checkpointing_non_reentrant", False)
        )

    def encode_observation(self, observation: PicfObservation) -> PaliGemmaSemanticFeatures:
        return self.encoder.encode_observation(observation)

    def supports_pi0_action_generation(self) -> bool:
        return bool(getattr(self.encoder, "supports_pi0_action_generation", lambda: False)())

    def compute_action_flow_loss(self, *args, **kwargs):
        fn = getattr(self.encoder, "compute_action_flow_loss", None)
        if fn is None:
            raise RuntimeError("Semantic encoder does not implement PI0 action flow loss.")
        return fn(*args, **kwargs)

    def sample_action_chunk(self, *args, **kwargs):
        fn = getattr(self.encoder, "sample_action_chunk", None)
        if fn is None:
            raise RuntimeError("Semantic encoder does not implement PI0 action chunk sampling.")
        return fn(*args, **kwargs)

    @property
    def last_runtime_timing(self) -> dict[str, float]:
        value = getattr(self.encoder, "last_runtime_timing", None)
        if isinstance(value, dict):
            return dict(value)
        return {}

    def fsdp_runtime_leaf_module_specs(self) -> list[tuple[nn.Module, str, str]]:
        fn = getattr(self.encoder, "fsdp_runtime_leaf_module_specs", None)
        if fn is None:
            return []
        return fn()

    def forward(self, op: str, /, *args: Any, **kwargs: Any):
        return self.encoder(op, *args, **kwargs)

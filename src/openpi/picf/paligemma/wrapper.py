from __future__ import annotations

import contextlib
import dataclasses
import hashlib
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
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.gemma as _gemma
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.pi0_pytorch import create_sinusoidal_pos_embedding
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import _ensure_transformers_replace_is_ready
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

    def _views(self, observation: PicfObservation) -> list[np.ndarray]:
        views = [np.asarray(observation.rgb_static)]
        if self.config.include_gripper_image and observation.rgb_gripper is not None:
            views.append(np.asarray(observation.rgb_gripper))
        return views

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
        self.paligemma_with_expert = self._build_paligemma_with_expert(
            paligemma_variant=paligemma_variant,
            action_expert_variant=config.action_expert_variant,
            precision=precision,
            pi05=bool(config.pi05),
        )
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        self.action_in_proj = nn.Linear(self.model_action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, self.model_action_dim)
        if not bool(config.pi05):
            raise RuntimeError("PICF PI0 action restoration only supports pi05=True.")
        self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
        self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        self._load_full_pi0_weights(checkpoint)
        self._drop_unused_generation_heads()
        self.to(device=self.device)
        self.tokenizer = PaligemmaTokenizer(max_len=max_token_len)
        if self.trainable:
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

        for parameter in self.model.parameters():
            parameter.requires_grad_(bool(self.trainable))

    @property
    def model(self) -> PaliGemmaWithExpertModel:
        return self.paligemma_with_expert

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
        model = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if pi05 else [False, False],
            precision=precision,
            tokenwise_chunk_size=tokenwise_chunk_size,
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
        return specs

    def _model_runtime_dtype(self) -> torch.dtype:
        return module_parameter_dtype(self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj)

    def _views(self, observation: PicfObservation) -> list[np.ndarray]:
        views = [np.asarray(observation.rgb_static)]
        if self.config.include_gripper_image and observation.rgb_gripper is not None:
            views.append(np.asarray(observation.rgb_gripper))
        return views

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
        state = np.zeros((self.model_action_dim,), dtype=np.float32)
        take = min(int(raw.shape[0]), self.model_action_dim)
        if take > 0:
            state[:take] = raw[:take]
        return np.clip(state, -1.0, 1.0)

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        lang_tokens, lang_masks = self._prepare_prompt(observation.prompt, observation)
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []
        image_token_count = 0
        for image in self._views(observation):
            image_tensor = self._prepare_image(image)

            def _image_embed(x: torch.Tensor) -> torch.Tensor:
                return self.paligemma_with_expert.embed_image(x)

            img_emb = self._apply_checkpoint(_image_embed, image_tensor)
            batch_size, num_img_tokens = img_emb.shape[:2]
            image_token_count += int(num_img_tokens)
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
        return prefix_embs, prefix_pad_masks, prefix_att_masks, image_token_count, lang_masks

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def encode_observation(self, observation: PicfObservation) -> PaliGemmaSemanticFeatures:
        use_grad = bool(self.trainable and self.training)
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            prefix_embs, prefix_pad_masks, prefix_att_masks, image_token_count, lang_masks = self._embed_prefix(observation)
            att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = _masked_position_ids(prefix_pad_masks)
            attn_mask_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)

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

            prefix_output = self._apply_checkpoint(_forward_prefix, prefix_embs, attn_mask_4d, position_ids)
            image_hidden = prefix_output[:, :image_token_count, :] if image_token_count > 0 else None
            text_hidden = prefix_output[:, image_token_count:, :]
            return PaliGemmaSemanticFeatures(
                tokens=_take_valid_prefix_tokens(prefix_output[0], prefix_pad_masks[0]),
                summary=_summary_from_outputs(
                    hidden_states=text_hidden,
                    image_hidden_states=image_hidden,
                    prompt_mask=lang_masks.to(dtype=text_hidden.dtype),
                ),
                prefix_embeddings=prefix_embs[0],
                prefix_pad_masks=prefix_pad_masks[0],
                prefix_att_masks=prefix_att_masks[0],
            )

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
                torch.tensor(1.5, device=self.device),
                torch.tensor(1.0, device=self.device),
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
        total = torch.nn.functional.mse_loss(u_t, v_t)
        pos = torch.nn.functional.mse_loss(u_t[..., :3], v_t[..., :3])
        rot = torch.nn.functional.mse_loss(u_t[..., 3:6], v_t[..., 3:6])
        grip = torch.nn.functional.mse_loss(u_t[..., 6:7], v_t[..., 6:7])
        predicted_chunk = _recover_flow_target(x_t, v_t, time_expanded).detach()
        predicted = predicted_chunk[:, 0, :7]
        return {
            "total": total,
            "action_pos": pos,
            "action_rot": rot,
            "action_gripper": grip,
            "predicted_action": predicted[0],
            "predicted_chunk": predicted_chunk[0],
        }

    @torch.no_grad()
    def sample_action_chunk(
        self,
        features: PaliGemmaSemanticFeatures,
        *,
        extra_prefix_tokens: torch.Tensor | None,
        noise: torch.Tensor | None = None,
        num_steps: int | None = None,
    ) -> torch.Tensor:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self._combine_prefix(
            features,
            extra_prefix_tokens=extra_prefix_tokens,
        )
        model_dtype = self._model_runtime_dtype()
        if model_dtype in (torch.bfloat16, torch.float16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks).to(dtype=prefix_embs.dtype)
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=_masked_position_ids(prefix_pad_masks),
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        if noise is None:
            noise = torch.randn((1, self.action_horizon, self.model_action_dim), device=self.device, dtype=torch.float32)
        x_t = noise
        steps = int(num_steps or self.denoise_steps)
        dt = torch.tensor(-1.0 / max(steps, 1), dtype=torch.float32, device=self.device)
        time = torch.tensor(1.0, dtype=torch.float32, device=self.device)
        while time >= (-dt / 2):
            timestep = time.expand(1)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self._embed_suffix(x_t, timestep)
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
            x_t = x_t + dt * v_t
            time += dt
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

    def fsdp_runtime_leaf_module_specs(self) -> list[tuple[nn.Module, str, str]]:
        fn = getattr(self.encoder, "fsdp_runtime_leaf_module_specs", None)
        if fn is None:
            return []
        return fn()

    def forward(self, op: str, /, *args: Any, **kwargs: Any):
        return self.encoder(op, *args, **kwargs)

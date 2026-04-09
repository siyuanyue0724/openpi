from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from safetensors import safe_open
from transformers import AutoProcessor
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING

import openpi.models.gemma as _gemma
from openpi.models.tokenizer import PaligemmaTokenizer
from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
from openpi.models_pytorch.pi0_pytorch import _ensure_transformers_replace_is_ready
from openpi.picf.contracts import PicfObservation
from openpi.picf.paligemma.config import PaliGemmaSemanticConfig
from openpi.shared import image_tools


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

    def encode_observation(self, observation: PicfObservation) -> torch.Tensor:
        views = self._views(observation)
        prompt = str(observation.prompt)
        summaries: list[torch.Tensor] = []
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
                summaries.append(
                    _summary_from_outputs(
                        hidden_states=hidden_states,
                        image_hidden_states=image_hidden_states,
                        prompt_mask=prompt_mask,
                    )
                )
        if not summaries:
            raise RuntimeError("PaliGemma semantic encoder did not receive any image views.")
        return torch.stack(summaries, dim=0).mean(dim=0)


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
        self.model = self._build_paligemma_model(paligemma_variant=paligemma_variant, precision=precision)
        self._load_paligemma_weights(checkpoint)
        self.model.to(device=self.device)
        self.tokenizer = PaligemmaTokenizer(max_len=max_token_len)
        if self.trainable:
            if hasattr(self.model, "gradient_checkpointing_disable"):
                self.model.gradient_checkpointing_disable()
            if config.gradient_checkpointing and hasattr(self.model, "gradient_checkpointing_enable"):
                enabled, non_reentrant = _enable_gradient_checkpointing_non_reentrant(self.model)
                self.gradient_checkpointing_enabled = enabled
                self.gradient_checkpointing_non_reentrant = non_reentrant
            if hasattr(self.model.config, "use_cache"):
                self.model.config.use_cache = False
            self.model.train()
        else:
            self.model.eval()

        for parameter in self.model.parameters():
            parameter.requires_grad_(bool(self.trainable))

    def _build_paligemma_model(self, *, paligemma_variant: str, precision: str) -> PaliGemmaForConditionalGeneration:
        config = _gemma.get_config(paligemma_variant)
        hf_config = CONFIG_MAPPING["paligemma"]()
        hf_config._vocab_size = 257152  # noqa: SLF001
        hf_config.image_token_index = 257152
        hf_config.text_config.hidden_size = config.width
        hf_config.text_config.intermediate_size = config.mlp_dim
        hf_config.text_config.num_attention_heads = config.num_heads
        hf_config.text_config.head_dim = config.head_dim
        hf_config.text_config.num_hidden_layers = config.depth
        hf_config.text_config.num_key_value_heads = config.num_kv_heads
        hf_config.text_config.hidden_activation = "gelu_pytorch_tanh"
        hf_config.text_config.torch_dtype = "float32"
        hf_config.text_config.vocab_size = 257152
        hf_config.vision_config.intermediate_size = 4304
        hf_config.vision_config.projection_dim = int(config.width)
        hf_config.vision_config.projector_hidden_act = "gelu_fast"
        hf_config.vision_config.torch_dtype = "float32"
        model = PaliGemmaForConditionalGeneration(config=hf_config)
        self._cast_selected_params(model, precision=precision)
        return model

    def _cast_selected_params(self, model: PaliGemmaForConditionalGeneration, *, precision: str) -> None:
        if precision == "bfloat16":
            model.to(dtype=torch.bfloat16)
        elif precision == "float16":
            model.to(dtype=torch.float16)
        else:
            model.to(dtype=torch.float32)
            return
        keep_fp32 = (
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        )
        for name, parameter in model.named_parameters():
            if any(token in name for token in keep_fp32):
                parameter.data = parameter.data.to(dtype=torch.float32)

    def _load_paligemma_weights(self, checkpoint: Path) -> None:
        prefix = "paligemma_with_expert.paligemma."
        state_dict: dict[str, torch.Tensor] = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key.startswith(prefix):
                    state_dict[key[len(prefix) :]] = handle.get_tensor(key)
        if not state_dict:
            raise RuntimeError(
                "Local PI0 checkpoint does not contain paligemma_with_expert.paligemma.* weights."
            )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if unexpected:
            raise RuntimeError(
                "Unexpected keys when loading local PI0 PaliGemma semantic checkpoint:\n"
                + "\n".join(map(str, unexpected[:200]))
            )
        repaired_missing = _repair_missing_tied_embeddings(self.model, missing_keys=list(missing))
        bad_missing = [key for key in repaired_missing if key not in {"lm_head.weight"}]
        if bad_missing:
            raise RuntimeError(
                "Unexpected missing keys when loading local PI0 PaliGemma semantic checkpoint:\n"
                + "\n".join(map(str, bad_missing[:200]))
            )

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

    def _prepare_prompt(self, prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        tokens_np, mask_np = self.tokenizer.tokenize(str(prompt), state=None)
        tokens = torch.as_tensor(tokens_np[None, :], device=self.device, dtype=torch.long)
        mask = torch.as_tensor(mask_np[None, :], device=self.device, dtype=torch.bool)
        return tokens, mask

    def _apply_checkpoint(self, func, *args):
        if bool(
            self.trainable
            and self.training
            and self.gradient_checkpointing_enabled
            and _checkpoint_inputs_require_grad(*args)
        ):
            return torch.utils.checkpoint.checkpoint(func, *args, use_reentrant=False, preserve_rng_state=False)
        return func(*args)

    def _embed_prefix(
        self,
        observation: PicfObservation,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, torch.Tensor]:
        lang_tokens, lang_masks = self._prepare_prompt(observation.prompt)
        embs: list[torch.Tensor] = []
        pad_masks: list[torch.Tensor] = []
        att_masks: list[int] = []
        image_token_count = 0
        for image in self._views(observation):
            image_tensor = self._prepare_image(image)

            def _image_embed(x: torch.Tensor) -> torch.Tensor:
                return self.model.model.get_image_features(x)

            img_emb = self._apply_checkpoint(_image_embed, image_tensor)
            batch_size, num_img_tokens = img_emb.shape[:2]
            image_token_count += int(num_img_tokens)
            embs.append(img_emb)
            pad_masks.append(torch.ones((batch_size, num_img_tokens), device=self.device, dtype=torch.bool))
            att_masks += [0] * num_img_tokens

        def _lang_embed(tokens: torch.Tensor) -> torch.Tensor:
            lang_emb = self.model.language_model.embed_tokens(tokens)
            return lang_emb * math.sqrt(lang_emb.shape[-1])

        lang_emb = self._apply_checkpoint(_lang_embed, lang_tokens)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        att_masks += [0] * int(lang_emb.shape[1])

        prefix_embs = torch.cat(embs, dim=1)
        prefix_pad_masks = torch.cat(pad_masks, dim=1)
        prefix_att_masks = torch.as_tensor(att_masks, device=self.device, dtype=torch.int32)[None, :]
        prefix_att_masks = prefix_att_masks.expand(prefix_pad_masks.shape[0], -1)
        model_dtype = self.model.language_model.layers[0].self_attn.q_proj.weight.dtype
        if model_dtype in (torch.float16, torch.bfloat16):
            prefix_embs = prefix_embs.to(dtype=model_dtype)
        return prefix_embs, prefix_pad_masks, prefix_att_masks, image_token_count, lang_masks

    def _prepare_attention_masks_4d(self, att_2d_masks: torch.Tensor) -> torch.Tensor:
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def encode_observation(self, observation: PicfObservation) -> torch.Tensor:
        use_grad = bool(self.trainable and self.training)
        context = contextlib.nullcontext() if use_grad else torch.inference_mode()
        with context:
            prefix_embs, prefix_pad_masks, prefix_att_masks, image_token_count, lang_masks = self._embed_prefix(observation)
            att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            position_ids = torch.cumsum(prefix_pad_masks.to(torch.int64), dim=1) - 1
            attn_mask_4d = self._prepare_attention_masks_4d(att_2d_masks).to(dtype=prefix_embs.dtype)

            def _forward_prefix(
                embeddings: torch.Tensor,
                attention_mask: torch.Tensor,
                positions: torch.Tensor,
            ) -> torch.Tensor:
                outputs = self.model.language_model.forward(
                    inputs_embeds=embeddings,
                    attention_mask=attention_mask,
                    position_ids=positions,
                    past_key_values=None,
                    use_cache=False,
                )
                return outputs.last_hidden_state

            prefix_output = self._apply_checkpoint(_forward_prefix, prefix_embs, attn_mask_4d, position_ids)
            image_hidden = prefix_output[:, :image_token_count, :] if image_token_count > 0 else None
            text_hidden = prefix_output[:, image_token_count:, :]
            return _summary_from_outputs(
                hidden_states=text_hidden,
                image_hidden_states=image_hidden,
                prompt_mask=lang_masks.to(dtype=text_hidden.dtype),
            )


class PaliGemmaSemanticEncoder(nn.Module):
    def __init__(self, config: PaliGemmaSemanticConfig | None = None):
        super().__init__()
        self.config = config or PaliGemmaSemanticConfig()
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

    def encode_observation(self, observation: PicfObservation) -> torch.Tensor:
        return self.encoder.encode_observation(observation)

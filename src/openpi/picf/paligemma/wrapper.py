from __future__ import annotations

import contextlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
from transformers import AutoProcessor
from transformers import PaliGemmaForConditionalGeneration

from openpi.picf.contracts import PicfObservation
from openpi.picf.paligemma.config import PaliGemmaSemanticConfig


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


class PaliGemmaSemanticEncoder(nn.Module):
    def __init__(self, config: PaliGemmaSemanticConfig | None = None):
        super().__init__()
        self.config = config or PaliGemmaSemanticConfig()
        self.device = _resolve_device(self.config)
        self.dtype = _resolve_dtype(self.config, self.device)
        self.trainable = bool(self.config.trainable)
        model_id = self.config.checkpoint_path or self.config.model_name
        local_only = Path(model_id).exists()
        self.processor = AutoProcessor.from_pretrained(
            model_id,
            revision=self.config.revision,
            local_files_only=local_only,
        )
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_id,
            revision=self.config.revision,
            torch_dtype=self.dtype,
            local_files_only=local_only,
        )
        self.model.to(device=self.device, dtype=self.dtype)
        if self.trainable:
            if hasattr(self.model, "gradient_checkpointing_enable") and self.config.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
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

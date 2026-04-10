# ruff: noqa: F401, FBT001, FBT002, I001, RET504, UP006, UP007, UP035

from __future__ import annotations

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.clip.modeling_clip import CLIPVisionConfig
from transformers.models.clip.modeling_clip import CLIPVisionTransformer


class TactileVideoMAE(nn.Module):
    def __init__(
        self,
        config,
        *,
        num_frames: int,
        stride: int,
        mask_ratio: float = 0.0,
    ):
        super().__init__()
        config.vision_config.num_frames = num_frames
        config.vision_config.tube_size = 1
        config.vision_config.mask_ratio = mask_ratio
        config.vision_config.stride = stride
        self.stride = stride
        self.num_frames = num_frames

        self.touch_model = CLIPVisionTransformer(config.vision_config)
        self.touch_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)
        self.num_image_feature_patches = int(
            (config.vision_config.image_size // config.vision_config.patch_size) ** 2 * (num_frames // stride)
        )
        self.touch_model.embeddings.num_patches = self.num_image_feature_patches
        self.touch_model.embeddings.patch_embedding = nn.Conv3d(
            in_channels=config.vision_config.num_channels,
            out_channels=self.touch_model.embeddings.embed_dim,
            kernel_size=(stride, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            stride=(stride, self.touch_model.embeddings.patch_size, self.touch_model.embeddings.patch_size),
            bias=False,
        )
        self.touch_model.embeddings.position_embedding = nn.Embedding(
            self.num_image_feature_patches + 1,
            self.touch_model.embeddings.embed_dim,
        )
        self.sensor_token = nn.Parameter(torch.zeros(20, 5, config.vision_config.hidden_size))
        self.new_position_ids = nn.Parameter(
            torch.arange(self.num_image_feature_patches + 1, dtype=torch.int64).unsqueeze(0),
            requires_grad=False,
        )
        self.touch_model.forward = self.touch_forward
        self.touch_model.embeddings.forward = self.emb_forward

    def touch_forward(
        self,
        pixel_values: torch.FloatTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
        sensor_type=None,
        use_mask: bool = True,
        probe: bool = False,
    ) -> tuple | BaseModelOutputWithPooling:
        del use_mask, probe
        output_attentions = output_attentions if output_attentions is not None else self.touch_model.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.touch_model.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.touch_model.config.use_return_dict

        hidden_states = self.touch_model.embeddings(pixel_values, sensor_type=sensor_type, use_mask=False)
        hidden_states = self.touch_model.pre_layrnorm(hidden_states)
        encoder_outputs = self.touch_model.encoder(
            inputs_embeds=hidden_states,
            attention_mask=None,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden_state = encoder_outputs[0]
        pooled_output = self.touch_model.post_layernorm(last_hidden_state[:, 0, :])
        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]
        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

    def emb_forward(self, pixel_values: torch.FloatTensor | None = None, noise=None, sensor_type=None, use_mask=True) -> torch.Tensor:
        del noise, use_mask
        batch_size = int(pixel_values.shape[0])
        target_dtype = self.touch_model.embeddings.patch_embedding.weight.dtype
        patch_embeds = self.touch_model.embeddings.patch_embedding(pixel_values.to(dtype=target_dtype))
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)
        if sensor_type is None:
            raise RuntimeError("AnyTouch emb_forward requires a sensor_type tensor.")
        min_sensor = int(sensor_type.min().item()) if sensor_type.numel() > 0 else 0
        max_sensor = int(sensor_type.max().item()) if sensor_type.numel() > 0 else -1
        if min_sensor < 0 or max_sensor >= int(self.sensor_token.shape[0]):
            raise RuntimeError(
                "AnyTouch sensor_type out of range: "
                f"min={min_sensor} max={max_sensor} num_sensor_tokens={self.sensor_token.shape[0]}"
            )
        pos_size = int(self.touch_model.embeddings.position_embedding.num_embeddings)
        pos_max = int(self.new_position_ids.max().item()) if self.new_position_ids.numel() > 0 else -1
        if pos_max >= pos_size:
            raise RuntimeError(
                "AnyTouch position ids out of range: "
                f"max={pos_max} num_position_embeddings={pos_size}"
            )
        pos_emb = self.touch_model.embeddings.position_embedding(self.new_position_ids)
        img_embeddings = patch_embeds + pos_emb[:, 1:, :]
        class_embeds = self.touch_model.embeddings.class_embedding + pos_emb[:, 0, :]
        class_embeds = class_embeds.expand(batch_size, 1, -1)
        sensor_emb = self.sensor_token[sensor_type]
        return torch.cat([class_embeds, sensor_emb, img_embeddings], dim=1)

    def forward(self, x=None, sensor_type=None, probe: bool = False, get_cls: bool = False):
        if x is not None and len(x.shape) == 4:
            x = x.unsqueeze(2).repeat(1, 1, self.num_frames, 1, 1)
        elif x is not None and x.shape[1] != 3:
            x = x.permute(0, 2, 1, 3, 4)
        latent = self.forward_encoder(x=x, sensor_type=sensor_type, use_mask=False, get_cls=get_cls, probe=probe)
        return latent

    def forward_encoder(self, x=None, sensor_type=None, use_mask=False, get_cls=False, probe=False):
        del use_mask
        if len(x.shape) == 4:
            x = x.unsqueeze(1).repeat(1, self.num_frames, 1, 1, 1)
        x = self.touch_model(x, sensor_type=sensor_type, use_mask=False, probe=probe)
        if get_cls:
            return self.touch_projection(x.pooler_output)
        if probe:
            return x.last_hidden_state
        out = self.touch_model.post_layernorm(x.last_hidden_state)
        return self.touch_projection(out)

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Sequence

import torch
import yaml
from torch import nn

from picf_next.wla_upstream import WLASourceReceipt, load_wla_world_symbols


WLA_WORLD_LAYER_COUNT: Final = 28
WLA_WORLD_IN_CHANNELS: Final = 64
WLA_WORLD_TARGET_SIZE: Final = 512
WLA_WORLD_DOWNSAMPLE_FACTOR: Final = 32
WLA_WORLD_LOSS_WEIGHT: Final = 0.1


@dataclass(frozen=True, slots=True)
class LingBotWLAWorldCondition:
    layerwise_embeddings: tuple[torch.Tensor, ...]
    attention_mask: torch.Tensor


@dataclass(frozen=True, slots=True)
class LingBotWLAWorldOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    target_velocity: torch.Tensor
    timesteps: torch.Tensor
    condition: LingBotWLAWorldCondition


def build_wla_target_transform(
    source_root: Path | str,
    *,
    target_size: int = WLA_WORLD_TARGET_SIZE,
) -> Callable[[Any], torch.Tensor]:
    """Build WLA's published resize/pad/tensor/normalize target transform."""

    if target_size != WLA_WORLD_TARGET_SIZE:
        raise ValueError("pinned WLA LIBERO source uses 512-pixel world targets")
    symbols = load_wla_world_symbols(source_root)
    from torchvision.transforms import v2

    return v2.Compose(
        [
            lambda image: symbols.resize_with_pad(image, target_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize([0.5], [0.5]),
        ]
    )


class LingBotWLAWorldExpert(nn.Module):
    """WLA's complete SANA world branch conditioned by the shared LingBot host.

    The donor's SANA transformer, one-layer bidirectional Qwen2 connector,
    frozen AutoencoderDC, flow scheduler, per-world-layer conditioning, and
    flow-matching objective are retained. The two explicit adaptations are the
    LingBot host width (2560 instead of RynnBrain's 2048) and extraction of the
    current CALVIN visual span from LingBot's released prefix surface.
    """

    def __init__(
        self,
        *,
        world_expert: nn.Module,
        connector: nn.Module,
        vae: nn.Module,
        noise_scheduler: Any,
        source_world_forward: Callable[..., torch.Tensor],
        density_sampler: Callable[..., torch.Tensor],
        loss_weighting: Callable[..., torch.Tensor],
        source: WLASourceReceipt,
        host_width: int,
    ) -> None:
        super().__init__()
        blocks = getattr(world_expert, "transformer_blocks", None)
        if not isinstance(blocks, nn.ModuleList) or len(blocks) != WLA_WORLD_LAYER_COUNT:
            raise ValueError("WLA world expert differs from its exact 28-layer SANA topology")
        config = getattr(world_expert, "config", None)
        if (
            int(getattr(config, "in_channels", -1)) != WLA_WORLD_IN_CHANNELS
            or int(getattr(config, "out_channels", -1)) != WLA_WORLD_IN_CHANNELS
            or int(getattr(config, "sample_size", -1)) != 16
        ):
            raise ValueError("WLA SANA latent surface differs from the published source")
        if not isinstance(connector, nn.Sequential) or len(connector) != 5:
            raise ValueError("WLA connector must retain its exact five-stage source topology")
        first_linear = connector[1]
        second_linear = connector[3]
        caption_width = int(getattr(config, "caption_channels", -1))
        if (
            not isinstance(first_linear, nn.Linear)
            or first_linear.in_features != host_width
            or first_linear.out_features != caption_width
            or not isinstance(second_linear, nn.Linear)
            or second_linear.in_features != caption_width
            or second_linear.out_features != caption_width
        ):
            raise ValueError("WLA connector dimensions differ from the shared-host contract")
        if not callable(source_world_forward) or not callable(density_sampler) or not callable(loss_weighting):
            raise TypeError("WLA source objective functions must remain callable")

        self.world_expert = world_expert
        self.connector = connector
        self.vae = vae
        self.noise_scheduler = noise_scheduler
        self.source = source
        self.host_width = host_width
        self._source_world_forward = source_world_forward
        self._density_sampler = density_sampler
        self._loss_weighting = loss_weighting

    @classmethod
    def from_pinned_source(
        cls,
        source_root: Path | str,
        pretrained_root: Path | str,
        *,
        host_width: int = 2560,
        world_device: torch.device | str | None = None,
        vae_device: torch.device | str | None = None,
    ) -> "LingBotWLAWorldExpert":
        symbols = load_wla_world_symbols(source_root)
        source_config = yaml.safe_load(
            (symbols.source.root / "configs/libero_all_image_action.yaml").read_text()
        )
        expected = {
            "connector_num_hidden_layers": 1,
            "in_channels": WLA_WORLD_IN_CHANNELS,
            "target_image_size": WLA_WORLD_TARGET_SIZE,
            "vae_downsample_f": WLA_WORLD_DOWNSAMPLE_FACTOR,
            "action_condition_type": "no_action_condition",
        }
        observed = {name: source_config.get(name) for name in expected}
        if observed != expected:
            raise ValueError(f"pinned WLA LIBERO world config changed: {observed}")
        if int(source_config["diffusion_model_cfg"]["num_layers"]) != WLA_WORLD_LAYER_COUNT:
            raise ValueError("pinned WLA world-layer count changed")
        if host_width <= 0 or host_width % 64:
            raise ValueError("WLA's Qwen2 connector requires a host width divisible by 64")

        pretrained = Path(pretrained_root).expanduser().resolve(strict=True)
        world_expert = symbols.sana_transformer.from_pretrained(
            pretrained,
            subfolder="transformer",
            torch_dtype=torch.bfloat16,
            use_safetensors=True,
        ).to(device=world_device)
        connector_out = (
            getattr(world_expert.config, "caption_channels", None)
            or getattr(world_expert.config, "encoder_hid_dim", None)
            or getattr(world_expert.config, "cross_attention_dim", None)
        )
        norm = symbols.rms_norm(
            connector_out,
            eps=1e-5,
            elementwise_affine=True,
        )
        with torch.no_grad():
            norm.weight.fill_(math.sqrt(5.5))
        encoder = symbols.qwen2_encoder(
            symbols.qwen2_config(
                hidden_size=host_width,
                intermediate_size=host_width * 4,
                num_hidden_layers=source_config["connector_num_hidden_layers"],
                num_attention_heads=host_width // 64,
                num_key_value_heads=host_width // 64,
                initializer_range=0.014,
                use_cache=False,
                rope=True,
                qk_norm=True,
            )
        )
        connector = nn.Sequential(
            encoder,
            nn.Linear(host_width, connector_out),
            nn.GELU(approximate="tanh"),
            nn.Linear(connector_out, connector_out),
            norm,
        ).to(device=world_device)

        vae = symbols.autoencoder.from_pretrained(
            pretrained,
            subfolder="vae",
            use_safetensors=True,
        ).to(device=vae_device if vae_device is not None else world_device)
        vae.requires_grad_(False)
        vae.eval()
        noise_scheduler = symbols.flow_scheduler.from_pretrained(
            pretrained,
            subfolder="scheduler",
            use_safetensors=True,
        )
        encoder.gradient_checkpointing_enable({"use_reentrant": False})
        world_expert.enable_gradient_checkpointing()
        return cls(
            world_expert=world_expert,
            connector=connector,
            vae=vae,
            noise_scheduler=noise_scheduler,
            source_world_forward=symbols.mllm_in_context.forward,
            density_sampler=symbols.density_sampler,
            loss_weighting=symbols.loss_weighting,
            source=symbols.source,
            host_width=host_width,
        )

    def encode_condition(
        self,
        *,
        current_visual_embeddings: torch.Tensor,
        current_visual_valid: torch.Tensor,
        layerwise_query_states: Sequence[torch.Tensor],
    ) -> LingBotWLAWorldCondition:
        if len(layerwise_query_states) != WLA_WORLD_LAYER_COUNT:
            raise ValueError("WLA world expert requires one query surface per SANA layer")
        if current_visual_embeddings.ndim != 3 or current_visual_embeddings.shape[-1] != self.host_width:
            raise ValueError("current LingBot visual embeddings have the wrong shape")
        batch, visual_count, _ = current_visual_embeddings.shape
        if (
            current_visual_valid.shape != (batch, visual_count)
            or current_visual_valid.dtype != torch.bool
        ):
            raise ValueError("current LingBot visual mask has the wrong shape or dtype")
        if not current_visual_valid.all():
            raise ValueError("WLA source does not pad inside the selected current-image span")

        connector_parameter = next(self.connector.parameters())
        connector_device = connector_parameter.device
        visual = current_visual_embeddings.to(device=connector_device)
        conditions: list[torch.Tensor] = []
        for query in layerwise_query_states:
            if query.shape != (batch, 64, self.host_width):
                raise ValueError("WLA layerwise metaquery surface has the wrong shape")
            conditions.append(self.connector(torch.cat((visual, query.to(connector_device)), dim=1)))
        attention_mask = torch.ones(
            batch,
            visual_count + 64,
            dtype=torch.bool,
            device=connector_device,
        )
        return LingBotWLAWorldCondition(
            layerwise_embeddings=tuple(conditions),
            attention_mask=attention_mask,
        )

    def _get_sigmas(
        self,
        timesteps: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
        dimensions: int,
    ) -> torch.Tensor:
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device)
        selected = timesteps.to(device)
        step_indices = [(schedule_timesteps == value).nonzero().item() for value in selected]
        sigma = sigmas[step_indices].flatten()
        while sigma.ndim < dimensions:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def forward(
        self,
        *,
        target_images: torch.Tensor,
        current_visual_embeddings: torch.Tensor,
        current_visual_valid: torch.Tensor,
        layerwise_query_states: Sequence[torch.Tensor],
    ) -> LingBotWLAWorldOutput:
        if (
            target_images.ndim != 4
            or target_images.shape[1:] != (3, WLA_WORLD_TARGET_SIZE, WLA_WORLD_TARGET_SIZE)
            or not target_images.is_floating_point()
            or not torch.isfinite(target_images).all()
        ):
            raise ValueError("WLA target images must be finite float [batch,3,512,512]")
        if float(target_images.min()) < -1.0001 or float(target_images.max()) > 1.0001:
            raise ValueError("WLA target-image normalization must remain in [-1,1]")

        condition = self.encode_condition(
            current_visual_embeddings=current_visual_embeddings,
            current_visual_valid=current_visual_valid,
            layerwise_query_states=layerwise_query_states,
        )
        vae_parameter = next(self.vae.parameters())
        vae_images = target_images.to(device=vae_parameter.device, dtype=vae_parameter.dtype)
        latents = self.vae.encode(vae_images).latent
        latent_depth = torch.zeros_like(latents)
        latents = torch.cat((latents, latent_depth), dim=1)
        if "shift_factor" in self.vae.config and self.vae.config.shift_factor is not None:
            latents = latents - self.vae.config.shift_factor
        latents = latents * self.vae.config.scaling_factor

        world_parameter = next(self.world_expert.parameters())
        world_device = world_parameter.device
        latents = latents.to(device=world_device)
        batch = latents.shape[0]
        if batch != current_visual_embeddings.shape[0]:
            raise ValueError("WLA world targets and LingBot host batch sizes differ")
        noise = torch.randn_like(latents, device=world_device)
        weighting_scheme = "uniform"
        u = self._density_sampler(
            weighting_scheme=weighting_scheme,
            batch_size=batch,
            logit_mean=0.0,
            logit_std=1.0,
            mode_scale=1.29,
        )
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=world_device)
        sigmas = self._get_sigmas(
            timesteps,
            device=world_device,
            dtype=latents.dtype,
            dimensions=latents.ndim,
        )
        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        layerwise_condition = [value.to(device=world_device) for value in condition.layerwise_embeddings]
        condition_mask = condition.attention_mask.to(device=world_device)
        prediction = self._source_world_forward(
            self,
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=layerwise_condition,
            encoder_attention_mask=condition_mask,
        )
        target_velocity = noise - latents
        weighting = self._loss_weighting(
            weighting_scheme=weighting_scheme,
            sigmas=sigmas,
        )
        difference = weighting.float() * (prediction.float() - target_velocity.float()).square()
        difference = difference[:, : WLA_WORLD_IN_CHANNELS // 2]
        loss = difference.mean()
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise RuntimeError("pinned WLA world source returned a non-finite scalar")
        return LingBotWLAWorldOutput(
            loss=loss,
            prediction=prediction,
            target_velocity=target_velocity,
            timesteps=timesteps,
            condition=condition,
        )

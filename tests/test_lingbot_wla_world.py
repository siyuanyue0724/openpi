from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn.functional as functional
from torch import nn

from picf_next.lingbot_wla_world import (
    WLA_WORLD_LAYER_COUNT,
    LingBotWLAWorldExpert,
)
from picf_next.wla_upstream import WLASourceReceipt


class _Config(dict):
    def __getattr__(self, name: str):
        return self[name]


class _FakeWorldExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))
        self.transformer_blocks = nn.ModuleList(
            [nn.Identity() for _ in range(WLA_WORLD_LAYER_COUNT)]
        )
        self.config = SimpleNamespace(
            in_channels=64,
            out_channels=64,
            sample_size=16,
            caption_channels=32,
        )


class _FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()), requires_grad=False)
        self.config = _Config(shift_factor=None, scaling_factor=1.0)

    def encode(self, images: torch.Tensor) -> SimpleNamespace:
        latent = functional.adaptive_avg_pool2d(images, (16, 16)).mean(1, keepdim=True)
        return SimpleNamespace(latent=latent.repeat(1, 32, 1, 1))


class _FakeScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.arange(9, -1, -1)
        self.sigmas = torch.linspace(1.0, 0.0, 11)
        self.config = SimpleNamespace(num_train_timesteps=10)


def _source_world_forward(
    owner: LingBotWLAWorldExpert,
    *,
    hidden_states: torch.Tensor,
    timestep: torch.Tensor,
    encoder_hidden_states: list[torch.Tensor],
    encoder_attention_mask: torch.Tensor,
) -> torch.Tensor:
    del timestep
    assert len(encoder_hidden_states) == WLA_WORLD_LAYER_COUNT
    assert encoder_attention_mask.dtype == torch.bool
    condition = torch.stack([value.mean() for value in encoder_hidden_states]).mean()
    return hidden_states * owner.world_expert.scale + condition


def _density_sampler(**kwargs) -> torch.Tensor:
    return torch.full((kwargs["batch_size"],), 0.25)


def _loss_weighting(*, weighting_scheme: str, sigmas: torch.Tensor) -> torch.Tensor:
    assert weighting_scheme == "uniform"
    return torch.ones_like(sigmas)


def test_world_objective_preserves_all_28_query_surfaces_and_gradients() -> None:
    host_width = 64
    connector = nn.Sequential(
        nn.Identity(),
        nn.Linear(host_width, 32),
        nn.GELU(approximate="tanh"),
        nn.Linear(32, 32),
        nn.Identity(),
    )
    world = LingBotWLAWorldExpert(
        world_expert=_FakeWorldExpert(),
        connector=connector,
        vae=_FakeVAE(),
        noise_scheduler=_FakeScheduler(),
        source_world_forward=_source_world_forward,
        density_sampler=_density_sampler,
        loss_weighting=_loss_weighting,
        source=WLASourceReceipt(root=__file__, commit="test", files=()),
        host_width=host_width,
    )
    visual = torch.randn(2, 7, host_width, requires_grad=True)
    queries = [
        torch.randn(2, 64, host_width, requires_grad=True)
        for _ in range(WLA_WORLD_LAYER_COUNT)
    ]
    output = world(
        target_images=torch.rand(2, 3, 512, 512) * 2.0 - 1.0,
        current_visual_embeddings=visual,
        current_visual_valid=torch.ones(2, 7, dtype=torch.bool),
        layerwise_query_states=queries,
    )
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert output.prediction.shape == (2, 64, 16, 16)
    assert len(output.condition.layerwise_embeddings) == WLA_WORLD_LAYER_COUNT
    assert output.condition.attention_mask.shape == (2, 71)

    output.loss.backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert all(query.grad is not None and torch.isfinite(query.grad).all() for query in queries)
    assert world.world_expert.scale.grad is not None
    assert connector[1].weight.grad is not None

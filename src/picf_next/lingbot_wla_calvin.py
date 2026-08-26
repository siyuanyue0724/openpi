from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
import torch
from PIL import Image
from torch import nn

from picf_next.data.calvin_physical_supervision_schema import source_array_sha256
from picf_next.lingbot_wla_shared import (
    LingBotWLAActionOutput,
    LingBotWLASharedInterface,
    run_lingbot_wla_calvin_forward,
)
from picf_next.lingbot_wla_world import (
    WLA_WORLD_LOSS_WEIGHT,
    LingBotWLAWorldExpert,
    LingBotWLAWorldOutput,
)


@dataclass(frozen=True, slots=True)
class WLACalvinTargetBatch:
    images: torch.Tensor
    source_global_indices: tuple[int, ...]
    source_rgb_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.images, torch.Tensor)
            or self.images.ndim != 4
            or self.images.shape[1:] != (3, 512, 512)
            or not self.images.is_floating_point()
            or not torch.isfinite(self.images).all()
        ):
            raise ValueError("WLA target images must be finite floating [batch,3,512,512]")
        batch = self.images.shape[0]
        if batch <= 0 or not (
            len(self.source_global_indices) == len(self.source_rgb_sha256) == batch
        ):
            raise ValueError("WLA target provenance must align with its image batch")
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in self.source_global_indices
        ):
            raise ValueError("WLA target source indices must be non-negative integers")
        if any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in self.source_rgb_sha256
        ):
            raise ValueError("WLA target RGB provenance must contain lowercase SHA-256 values")

    def to(self, *, device: torch.device | str, dtype: torch.dtype) -> "WLACalvinTargetBatch":
        return WLACalvinTargetBatch(
            images=self.images.to(device=device, dtype=dtype),
            source_global_indices=self.source_global_indices,
            source_rgb_sha256=self.source_rgb_sha256,
        )


@dataclass(frozen=True, slots=True)
class LingBotWLAFullCalvinOutput:
    loss: torch.Tensor
    action: LingBotWLAActionOutput
    world: LingBotWLAWorldOutput
    native_root_outputs: tuple[torch.Tensor, ...]


def build_wla_calvin_target_batch(
    dataset: Any,
    sample_keys: Sequence[str],
    *,
    action_horizon: int,
    target_transform: Callable[[Image.Image], torch.Tensor],
) -> WLACalvinTargetBatch:
    """Load WLA's t+horizon static-camera targets from verified CALVIN bytes.

    This is a loss-target-only path. The returned tensors are never inserted
    into LingBot inputs, PICF evidence, or posterior state.
    """

    if not sample_keys or any(not isinstance(key, str) or not key for key in sample_keys):
        raise ValueError("WLA CALVIN target keys must be a nonempty string sequence")
    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError("WLA CALVIN action horizon must be positive")
    index = getattr(dataset, "index", None)
    if index is None:
        raise TypeError("WLA CALVIN targets require a manifest-verified CALVIN dataset")
    source_global_indices = tuple(
        int(dataset.source_global_index_by_key(sample_key)) for sample_key in sample_keys
    )
    return build_wla_calvin_target_batch_from_source_indices(
        index,
        source_global_indices,
        action_horizon=action_horizon,
        target_transform=target_transform,
    )


def build_wla_calvin_target_batch_from_source_indices(
    index: Any,
    source_global_indices: Sequence[int],
    *,
    action_horizon: int,
    target_transform: Callable[[Image.Image], torch.Tensor],
) -> WLACalvinTargetBatch:
    """Resolve WLA targets on CALVIN's unique raw-episode time axis."""

    if not source_global_indices or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in source_global_indices
    ):
        raise ValueError("WLA CALVIN source indices must be nonempty non-negative integers")
    if isinstance(action_horizon, bool) or not isinstance(action_horizon, int) or action_horizon <= 0:
        raise ValueError("WLA CALVIN action horizon must be positive")
    if not callable(getattr(index, "source_episode", None)) or not callable(
        getattr(index, "validated_source_frame_arrays", None)
    ):
        raise TypeError("WLA CALVIN targets require a manifest-verified source index")

    images: list[torch.Tensor] = []
    indices: list[int] = []
    digests: list[str] = []
    for source_global_index in source_global_indices:
        episode = index.source_episode(source_global_index)
        global_index = source_global_index + action_horizon
        if global_index > episode.end:
            raise ValueError("WLA CALVIN target horizon crosses a raw episode reset")
        source = index.validated_source_frame_arrays(global_index, fields=("rgb_static",))
        rgb = np.asarray(source["rgb_static"], dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("CALVIN static RGB target has the wrong shape")
        transformed = target_transform(Image.fromarray(rgb))
        if not isinstance(transformed, torch.Tensor) or transformed.shape != (3, 512, 512):
            raise TypeError("pinned WLA target transform returned the wrong tensor surface")
        images.append(transformed)
        indices.append(global_index)
        digests.append(source_array_sha256("rgb_static", rgb))
    return WLACalvinTargetBatch(
        images=torch.stack(images),
        source_global_indices=tuple(indices),
        source_rgb_sha256=tuple(digests),
    )


def run_lingbot_wla_full_calvin_objective(
    policy: nn.Module,
    action_interface: LingBotWLASharedInterface,
    world_expert: LingBotWLAWorldExpert,
    *,
    model_inputs: dict[str, Any],
    picf_native_context: Any,
    target_images: torch.Tensor,
) -> LingBotWLAFullCalvinOutput:
    """Run one shared host, then WLA's complete action and world objectives."""

    calvin = run_lingbot_wla_calvin_forward(
        policy,
        action_interface,
        model_inputs=model_inputs,
        picf_native_context=picf_native_context,
    )
    world = world_expert(
        target_images=target_images,
        current_visual_embeddings=calvin.current_visual_embeddings,
        current_visual_valid=calvin.current_visual_valid,
        layerwise_query_states=calvin.action.host.layerwise_query_states,
    )
    loss = calvin.action.loss + WLA_WORLD_LOSS_WEIGHT * world.loss
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise RuntimeError("WLA CALVIN combined objective is not a finite scalar")
    return LingBotWLAFullCalvinOutput(
        loss=loss,
        action=calvin.action,
        world=world,
        native_root_outputs=calvin.native_root_outputs,
    )

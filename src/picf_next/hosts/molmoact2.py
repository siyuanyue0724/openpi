"""Deep typed-context adapter for the pinned official MolmoAct2 action expert.

The block ordering is substantially adapted from
`allenai/molmoact2@c2282820f9b188b60e66ea1636b3efd81c45cbb4`,
`experiments/olmo/hf_model/modeling_molmoact2.py::ActionExpertBlock.forward`
and `ActionExpert.forward_with_context`, under Apache-2.0.

PICF adds two separately normalized cross-attention domains after the official
native VLM cross-read and before the official MLP. It never appends to or edits
the VLM sequence. Object keys accept identity address only; dynamic state and
innovation are values only. Token ownership adds the same persistent-address
coordinate to dense K/V projections while preserving every native token.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from olmo.hf_model.modeling_molmoact2 import (
    ActionExpert,
    ActionExpertContext,
    ActionExpertCrossAttention,
    ActionExpertRMSNorm,
    ActionExpertStepModulation,
    MolmoAct2ForConditionalGeneration,
)
from torch import nn

from picf_next.hosts.context import PICFActionEvidence
from picf_next.hosts.molmoact2_layout import (
    MOLMO_VISION_PATCH_MODALITY,
    MolmoAct2ImagePatchSpan,
    MolmoAct2VisionPatchLayout,
)
from picf_next.models.evidence import NativeTokenBank

_ADAPTER_CONFIG_NAME = "picf_adapter_config.json"
_ADAPTER_WEIGHTS_NAME = "picf_adapter.safetensors"
_ADAPTER_FORMAT = "picf-next.molmoact2-adapter.v5"
PICFDenseEvidence = NativeTokenBank


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _strict_adapter_dimension(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError("adapter dimensions must be positive JSON integers")
    return value


def _strict_adapter_modality(value: object) -> str:
    if not isinstance(value, str) or not value or "." in value:
        raise ValueError("adapter modality names must be nonempty strings without dots")
    return value


@dataclass(frozen=True, slots=True)
class PICFActionContext:
    """Projected immutable per-layer KV banks and independent masks."""

    dense_kv_contexts: Sequence[tuple[torch.Tensor, torch.Tensor]] | None
    dense_mask: torch.Tensor | None
    object_kv_contexts: Sequence[tuple[torch.Tensor, torch.Tensor]] | None
    object_mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class _PICFActionLayerContext:
    """One selected and flow-batch-expanded PICF layer context."""

    dense_kv_context: tuple[torch.Tensor, torch.Tensor] | None
    dense_mask: torch.Tensor | None
    object_kv_context: tuple[torch.Tensor, torch.Tensor] | None
    object_mask: torch.Tensor | None


@dataclass(frozen=True, slots=True)
class MolmoAct2EncoderBundle:
    """One-pass native VLM context plus every pre-pooling vision patch.

    The host's learned pooled image tokens remain in `encoder_kv_states`.
    `vision_patch_bank` is a second, lossless discovery/action view; it does not
    replace, prune or modify the native VLM sequence.
    """

    encoder_kv_states: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    encoder_attention_mask: torch.Tensor
    vision_patch_bank: NativeTokenBank | None


@dataclass(frozen=True, slots=True)
class MolmoAct2LeRobotObservation:
    """One differentiable visual pass prepared for the official LeRobot policy.

    ``model_inputs`` contains ``inputs_embeds`` instead of raw pixels, so the
    official joint VLM/action loop cannot run the ViT a second time. The dense
    bank and the pooled image-token contribution share the same pre-pooling
    features and autograd graph. ``action_condition_input_ids`` preserves the
    deploy-visible token identities used only by the official action mask and
    depth-gate logic; it is not sent back through the embedding or vision path.
    """

    model_inputs: Mapping[str, torch.Tensor]
    action_condition_input_ids: torch.LongTensor
    vision_patch_bank: NativeTokenBank | None
    vision_patch_layout: MolmoAct2VisionPatchLayout | None


def _molmoact2_vision_patch_layout(
    policy: nn.Module,
    *,
    model_inputs: Mapping[str, torch.Tensor],
    images: torch.Tensor,
    batched_token_pooling: torch.Tensor,
    dense_valid: torch.Tensor,
) -> MolmoAct2VisionPatchLayout:
    """Reconcile official flat processor metadata with batched dense patches."""

    raw_crops = model_inputs.get("image_num_crops")
    raw_grids = model_inputs.get("image_grids")
    raw_pooling = model_inputs.get("image_token_pooling")
    if raw_crops is None or raw_grids is None or raw_pooling is None:
        raise ValueError("MolmoAct2 image inputs require complete processor layout metadata")
    if (
        images.ndim != 4
        or dense_valid.shape != (images.shape[0], images.shape[1] * images.shape[2])
        or batched_token_pooling.ndim != 3
        or raw_crops.ndim != 1
        or raw_grids.ndim != 2
        or raw_grids.shape[1] != 4
        or raw_pooling.ndim != 2
    ):
        raise ValueError("MolmoAct2 processor layout metadata has unexpected rank")
    batch_size, maximum_crops, patches_per_crop = images.shape[:3]
    total_images = int(raw_crops.numel())
    if total_images <= 0 or total_images % batch_size:
        raise ValueError("MolmoAct2 requires one fixed image count per batch example")
    images_per_example = total_images // batch_size
    if raw_grids.shape[0] != total_images:
        raise ValueError("MolmoAct2 image grid count differs from image crop metadata")

    configured_keys = tuple(str(value) for value in getattr(policy.config, "image_keys", ()))
    semantic_keys = bool(configured_keys)
    if configured_keys and len(configured_keys) != images_per_example:
        raise ValueError("MolmoAct2 configured image keys differ from processor image count")
    image_keys = configured_keys or tuple(
        f"__processor_image_{index}" for index in range(images_per_example)
    )

    crop_values = tuple(int(value) for value in raw_crops.detach().cpu().tolist())
    grid_values = tuple(
        tuple(int(value) for value in row) for row in raw_grids.detach().cpu().tolist()
    )
    raw_pooling_cpu = raw_pooling.detach().cpu()
    if any(value <= 0 for value in crop_values):
        raise ValueError("MolmoAct2 image crop counts must be positive")

    rows: list[tuple[MolmoAct2ImagePatchSpan, ...]] = []
    pooling_cursor = 0
    for batch_index in range(batch_size):
        spans: list[MolmoAct2ImagePatchSpan] = []
        dense_cursor = 0
        expected_batched_rows: list[torch.Tensor] = []
        for local_image_index, image_key in enumerate(image_keys):
            flat_image_index = batch_index * images_per_example + local_image_index
            crop_count = crop_values[flat_image_index]
            grid = grid_values[flat_image_index]
            pooled_count = grid[0] * grid[1] + grid[2] * grid[3]
            if pooled_count <= 0 or pooling_cursor + pooled_count > raw_pooling_cpu.shape[0]:
                raise ValueError("MolmoAct2 image pooling rows do not cover the image grid")
            local_pooling = raw_pooling_cpu[pooling_cursor : pooling_cursor + pooled_count]
            pooling_cursor += pooled_count
            capacity = crop_count * patches_per_crop
            if ((local_pooling < -1) | (local_pooling >= capacity)).any():
                raise ValueError("MolmoAct2 per-image pooling index is outside its crop bank")
            expected_batched_rows.append(
                torch.where(local_pooling >= 0, local_pooling + dense_cursor, local_pooling)
            )
            spans.append(
                MolmoAct2ImagePatchSpan(
                    image_key=image_key,
                    start=dense_cursor,
                    stop=dense_cursor + capacity,
                    image_num_crops=crop_count,
                    patches_per_crop=patches_per_crop,
                    image_grid=grid,
                    image_token_pooling=tuple(
                        tuple(int(value) for value in row) for row in local_pooling.tolist()
                    ),
                )
            )
            dense_cursor += capacity
        if dense_cursor > maximum_crops * patches_per_crop:
            raise ValueError("MolmoAct2 image crop metadata exceeds the batched image tensor")
        expected_valid = (
            torch.arange(
                maximum_crops * patches_per_crop,
                device=dense_valid.device,
            )
            < dense_cursor
        )
        if not torch.equal(dense_valid[batch_index], expected_valid):
            raise ValueError("MolmoAct2 dense validity disagrees with camera patch spans")
        expected_pooling = torch.cat(expected_batched_rows, dim=0).to(
            device=batched_token_pooling.device,
            dtype=batched_token_pooling.dtype,
        )
        actual_pooling = batched_token_pooling[batch_index]
        if expected_pooling.shape[0] > actual_pooling.shape[0] or not torch.equal(
            actual_pooling[: expected_pooling.shape[0]], expected_pooling
        ):
            raise ValueError("MolmoAct2 batched pooling differs from processor camera ordering")
        if (actual_pooling[expected_pooling.shape[0] :] != -1).any():
            raise ValueError("MolmoAct2 padded pooling rows must be exactly -1")
        rows.append(tuple(spans))
    if pooling_cursor != raw_pooling_cpu.shape[0]:
        raise ValueError("MolmoAct2 image pooling contains trailing rows")
    return MolmoAct2VisionPatchLayout(
        rows=tuple(rows),
        tokens_per_row=maximum_crops * patches_per_crop,
        semantic_image_keys=semantic_keys,
    )


@dataclass(frozen=True, slots=True)
class MolmoAct2HostCheckpointIdentity:
    """Externally verified host identity bound into a standalone adapter.

    The manifest digest is produced by the checkpoint preflight that hashes
    every released weight shard. This small contract does not pretend that an
    in-memory module can recover the provenance of weights already loaded into
    it; it makes that provenance an explicit deployment input instead.
    """

    checkpoint_id: str
    revision: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.checkpoint_id, self.revision)
        ):
            raise ValueError("host checkpoint ID and revision must be nonempty strings")
        if (
            not isinstance(self.manifest_sha256, str)
            or len(self.manifest_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.manifest_sha256)
        ):
            raise ValueError("host checkpoint manifest must be one lowercase SHA-256 digest")

    @property
    def payload(self) -> dict[str, str]:
        return {
            "checkpoint_id": self.checkpoint_id,
            "manifest_sha256": self.manifest_sha256,
            "revision": self.revision,
        }


_LE_ROBOT_VISUAL_INPUT_KEYS = {
    "input_ids",
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
    "attention_mask",
    "position_ids",
    "token_type_ids",
}
_LE_ROBOT_RAW_VISUAL_KEYS = {
    "pixel_values",
    "image_token_pooling",
    "image_grids",
    "image_num_crops",
    "pixel_values_videos",
    "video_token_pooling",
    "video_grids",
}


def _dense_patch_partition(
    pooled_patches_idx: torch.Tensor,
    *,
    num_crops: int,
    patches_per_crop: int,
) -> torch.Tensor:
    """Recover valid dense patches only from a complete pooling partition.

    Released resize-mode MolmoAct2 pooling assigns every source patch exactly
    once. Batched crops occupy a contiguous prefix; padded crops have no pooled
    index. Rejecting every other layout prevents padded image features or an
    unsupported overlapping/multicrop contract from entering PICF silently.
    """

    if pooled_patches_idx.ndim != 3:
        raise ValueError("MolmoAct2 pooling indices must be batch-by-token-by-support")
    if pooled_patches_idx.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("MolmoAct2 pooling indices must be integer tensors")
    if num_crops <= 0 or patches_per_crop <= 0:
        raise ValueError("MolmoAct2 dense patch geometry must be positive")

    capacity = num_crops * patches_per_crop
    if (pooled_patches_idx < -1).any() or (pooled_patches_idx >= capacity).any():
        raise ValueError("MolmoAct2 pooling index is outside the dense patch bank")

    valid = pooled_patches_idx >= 0
    valid_count = valid.sum(dim=(1, 2))
    if (valid_count % patches_per_crop != 0).any():
        raise ValueError("MolmoAct2 pooling does not cover whole contiguous crops")

    counts = torch.zeros(
        pooled_patches_idx.shape[0],
        capacity,
        dtype=torch.long,
        device=pooled_patches_idx.device,
    )
    counts.scatter_add_(
        1,
        pooled_patches_idx.clamp_min(0).reshape(pooled_patches_idx.shape[0], -1).long(),
        valid.reshape(pooled_patches_idx.shape[0], -1).long(),
    )
    dense_valid = torch.arange(capacity, device=pooled_patches_idx.device).unsqueeze(0) < (
        valid_count.unsqueeze(1)
    )
    if (counts != dense_valid.long()).any():
        raise ValueError("MolmoAct2 pooling must partition every valid dense patch exactly once")
    return dense_valid


def _encode_and_pool_vision_once(
    vision_backbone: nn.Module,
    images: torch.Tensor,
    pooled_patches_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return official pooled features and all pre-pooling features in one pass.

    This is a substantial adaptation of
    `MolmoAct2VisionBackbone.forward` from the pinned Apache-2.0 upstream
    revision named in the module header. The operation order and pooling math
    are intentionally identical; the only additional result is the masked
    pre-pooling patch bank. No hook or second ViT forward is used.
    """

    if images.ndim != 4:
        raise ValueError("MolmoAct2 images must be batch-by-crop-by-patch-by-pixels")
    if pooled_patches_idx.device != images.device:
        raise ValueError("MolmoAct2 images and pooling indices must share a device")
    if images.device != vision_backbone.device:
        raise ValueError("MolmoAct2 images and pooling indices must already be on the ViT device")
    if images.dtype != torch.uint8 and not torch.is_floating_point(images):
        raise ValueError("MolmoAct2 images must be uint8 or floating tensors")
    batch_size, num_crops, patches_per_crop = images.shape[:3]
    if pooled_patches_idx.shape[0] != batch_size:
        raise ValueError("MolmoAct2 images and pooling indices must share a batch size")
    dense_valid = _dense_patch_partition(
        pooled_patches_idx,
        num_crops=num_crops,
        patches_per_crop=patches_per_crop,
    )

    if images.dtype == torch.uint8:
        images = images.to(dtype=torch.float32) / 255.0
        images = images * 2.0 - 1.0
    elif torch.is_floating_point(images):
        images = torch.round(((images.to(dtype=torch.float32) + 1.0) * 0.5) * 255.0)
        images = torch.clamp(images, 0.0, 255.0) / 255.0
        images = images * 2.0 - 1.0
    images = images.to(dtype=vision_backbone.dtype)

    image_features = vision_backbone.encode_image(images)
    image_features = vision_backbone.image_feature_dropout(image_features)
    feature_dim = image_features.shape[-1]
    valid_support = pooled_patches_idx >= 0
    valid_pooled_token = torch.any(valid_support, dim=-1)

    batch_idx = torch.arange(
        pooled_patches_idx.shape[0],
        dtype=torch.long,
        device=pooled_patches_idx.device,
    )
    batch_idx = torch.tile(
        batch_idx.view(batch_size, 1, 1),
        [1, pooled_patches_idx.shape[1], pooled_patches_idx.shape[2]],
    )
    to_pool = image_features.reshape(batch_size, -1, feature_dim)[
        batch_idx,
        torch.clip(pooled_patches_idx, 0),
    ]
    to_pool = to_pool * valid_support.to(vision_backbone.dtype)[:, :, :, None]
    to_pool = to_pool.reshape(-1, pooled_patches_idx.shape[-1], feature_dim)
    if vision_backbone.adapter_config.pooling_attention_mask:
        attention_mask = valid_support.reshape(-1, 1, 1, valid_support.shape[-1])
        denominator = valid_support.view(-1, to_pool.shape[-2]).float().sum(-1)
        denominator = torch.where(denominator == 0, 1, denominator)
        query = to_pool.sum(-2, keepdim=True) / denominator[:, None, None].to(to_pool.dtype)
    else:
        attention_mask = None
        query = to_pool.mean(-2, keepdim=True)
    pooled_features = vision_backbone.image_pooling_2d(
        query,
        to_pool,
        attn_mask=attention_mask,
    )
    pooled_features = pooled_features.reshape(
        batch_size,
        -1,
        pooled_features.shape[-1],
    )
    pooled_features = vision_backbone.image_projector(pooled_features)
    pooled_features = pooled_features.view(-1, pooled_features.shape[-1])[
        valid_pooled_token.flatten()
    ]

    dense_features = image_features.reshape(batch_size, -1, feature_dim)
    dense_features = dense_features * dense_valid.unsqueeze(-1)
    return pooled_features, dense_features, dense_valid


def prepare_molmoact2_lerobot_observation(
    policy: nn.Module,
    observation_inputs: Mapping[str, torch.Tensor],
) -> MolmoAct2LeRobotObservation:
    """Build the native VLM image embeddings and dense PICF bank in one ViT pass.

    This function consumes deploy-visible observation fields only. Action
    targets and structural labels have no argument path. It substantially
    follows the pinned LeRobot ``_model_inputs`` and MolmoAct2
    ``build_input_embeddings`` order, while replacing the vision forward with
    :func:`_encode_and_pool_vision_once` so pre-pooling patches remain visible.
    """

    unexpected = sorted(set(observation_inputs) - _LE_ROBOT_VISUAL_INPUT_KEYS)
    if unexpected:
        raise ValueError(f"unsupported MolmoAct2 observation fields: {unexpected}")
    if "input_ids" not in observation_inputs:
        raise ValueError("MolmoAct2 observation inputs require input_ids")
    if any(
        observation_inputs.get(name) is not None
        for name in ("pixel_values_videos", "video_token_pooling", "video_grids")
    ):
        raise NotImplementedError(
            "same-forward MolmoAct2 video dense-token extraction is not validated"
        )

    normalize_inputs = getattr(policy, "_model_inputs", None)
    get_backbone = getattr(policy, "_backbone", None)
    if not callable(normalize_inputs) or not callable(get_backbone):
        raise TypeError("policy does not expose the pinned MolmoAct2 LeRobot input contract")
    model_inputs = dict(normalize_inputs(dict(observation_inputs)))
    input_ids = model_inputs.get("input_ids")
    if input_ids is None or input_ids.ndim != 2:
        raise ValueError("MolmoAct2 input_ids must be a batch-by-token tensor")

    backbone = get_backbone()
    merge_visual_inputs = getattr(backbone, "merge_visual_inputs", None)
    if not callable(merge_visual_inputs):
        raise RuntimeError("MolmoAct2 backbone does not expose merge_visual_inputs")
    images, token_pooling = merge_visual_inputs(
        input_ids=input_ids,
        pixel_values=model_inputs.get("pixel_values"),
        image_token_pooling=model_inputs.get("image_token_pooling"),
        image_grids=model_inputs.get("image_grids"),
        image_num_crops=model_inputs.get("image_num_crops"),
        pixel_values_videos=None,
        video_token_pooling=None,
        video_grids=None,
    )

    model_dtype_name = getattr(policy.config, "model_dtype", "float32")
    model_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(model_dtype_name)
    if model_dtype is None:
        raise ValueError(f"unsupported MolmoAct2 model dtype: {model_dtype_name!r}")
    device = next(backbone.parameters()).device
    autocast_context = (
        torch.autocast(device_type=device.type, dtype=model_dtype)
        if device.type in {"cuda", "cpu"} and model_dtype in {torch.bfloat16, torch.float16}
        else nullcontext()
    )
    with autocast_context:
        safe_input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        input_embeddings = backbone.transformer.wte(safe_input_ids)
        vision_patch_bank = None
        vision_patch_layout = None
        if images is not None:
            if token_pooling is None:
                raise ValueError("MolmoAct2 image inputs require pooling indices")
            pooled_features, dense_features, dense_valid = _encode_and_pool_vision_once(
                backbone.vision_backbone,
                images,
                token_pooling,
            )
            is_image_patch = safe_input_ids.reshape(-1) == backbone.config.image_patch_id
            if int(is_image_patch.sum().item()) != len(pooled_features):
                raise ValueError("MolmoAct2 pooled image features do not align with input tokens")
            flat_embeddings = input_embeddings.reshape(-1, input_embeddings.shape[-1]).clone()
            flat_embeddings[is_image_patch] = flat_embeddings[is_image_patch] + pooled_features.to(
                flat_embeddings.device
            )
            input_embeddings = flat_embeddings.reshape_as(input_embeddings)
            vision_patch_bank = NativeTokenBank(
                modality=MOLMO_VISION_PATCH_MODALITY,
                tokens=dense_features,
                valid=dense_valid,
            )
            vision_patch_layout = _molmoact2_vision_patch_layout(
                policy,
                model_inputs=model_inputs,
                images=images,
                batched_token_pooling=token_pooling,
                dense_valid=dense_valid,
            )
        input_embeddings = backbone.transformer.emb_drop(input_embeddings)

    prepared = {
        name: value
        for name, value in model_inputs.items()
        if name != "input_ids" and name not in _LE_ROBOT_RAW_VISUAL_KEYS
    }
    prepared["inputs_embeds"] = input_embeddings
    if prepared.get("attention_mask") is None:
        prepared["attention_mask"] = input_ids != -1
    return MolmoAct2LeRobotObservation(
        model_inputs=prepared,
        action_condition_input_ids=input_ids,
        vision_patch_bank=vision_patch_bank,
        vision_patch_layout=vision_patch_layout,
    )


def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Exact official MolmoAct2 action-expert modulation equation."""

    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def _initialize_linear(linear: nn.Linear, *, scale: float = 1.0) -> None:
    """Match the official action-expert linear initialization."""

    nn.init.xavier_uniform_(linear.weight)
    if scale != 1.0:
        with torch.no_grad():
            linear.weight.mul_(scale)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class _TypedResidualCrossAttention(nn.Module):
    """One independent attention domain with an exactly zero residual gate."""

    def __init__(self, expert: ActionExpert) -> None:
        super().__init__()
        config = expert.config
        factory_parameter = next(expert.parameters())
        factory_kwargs = {
            "device": factory_parameter.device,
            "dtype": torch.float32,
        }
        self.norm = ActionExpertRMSNorm(config.hidden_size, eps=1e-6)
        self.attention = ActionExpertCrossAttention(
            config.hidden_size,
            config.num_heads,
            attn_dropout=config.attn_dropout,
            proj_dropout=config.dropout,
            qk_norm=config.qk_norm,
            qk_norm_eps=config.qk_norm_eps,
        ).to(**factory_kwargs)
        self.gate = nn.Parameter(torch.zeros((), **factory_kwargs))

        residual_scale = (2 * max(config.num_layers, 1)) ** -0.5
        _initialize_linear(self.attention.q_proj)
        _initialize_linear(self.attention.out_proj, scale=residual_scale)

    def forward(
        self,
        x: torch.Tensor,
        shift: torch.Tensor,
        scale: torch.Tensor,
        *,
        kv: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        branch = self.attention(
            _modulate(self.norm(x), shift, scale),
            kv_k=kv[0],
            kv_v=kv[1],
            attn_mask=attention_mask,
        )
        if attention_mask is not None:
            if attention_mask.ndim != 4 or attention_mask.shape[0] != x.shape[0]:
                raise ValueError("typed PICF attention mask must be batch-by-head-by-query-by-key")
            # A finite minimum mask is required by the pinned host kernel, but
            # softmax over an entirely masked row is still a normalized row.
            # Explicitly remove that branch so a learned output-projection bias
            # cannot fabricate evidence for a missing modality/object bank.
            evidence_available = (attention_mask > torch.finfo(attention_mask.dtype).min).any(
                dim=-1
            )
            branch = branch * evidence_available.to(dtype=branch.dtype)
        return self.gate * branch


class MolmoAct2PICFActionExpert(nn.Module):
    """Wrap a loaded official expert with deep dense/object PICF reads.

    Wrap only after loading the vanilla checkpoint. The official expert remains
    owned by the host model and is held here through an unregistered reference,
    so adapter checkpoints cannot duplicate or rename vanilla weights. PICF
    parameters live in explicitly named adapter modules. `vanilla` outputs are
    exact when no PICF context is supplied or all residual gates are zero.
    """

    def __init__(
        self,
        vanilla: ActionExpert,
        *,
        dense_token_dims: Mapping[str, int],
        object_address_dim: int,
        object_value_dim: int,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        if not dense_token_dims:
            raise ValueError("at least one dense modality must be configured")
        normalized_dense_dims = dict(dense_token_dims)
        for modality, width in normalized_dense_dims.items():
            if not modality or "." in modality:
                raise ValueError("dense modality names must be nonempty and cannot contain dots")
            if not isinstance(width, int) or isinstance(width, bool) or width <= 0:
                raise ValueError(f"dense token width for {modality} must be positive")
        for name, value in {
            "object_address_dim": object_address_dim,
            "object_value_dim": object_value_dim,
        }.items():
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be positive")
        if not isinstance(validate_tensor_values, bool):
            raise ValueError("validate_tensor_values must be boolean")

        # Bypass nn.Module.__setattr__: the host remains the sole parameter
        # owner, while this strong reference keeps direct adapter use alive.
        object.__setattr__(self, "_vanilla_unregistered", vanilla)
        self.validate_tensor_values = validate_tensor_values
        hidden_size = vanilla.config.hidden_size
        parameter = next(vanilla.parameters())
        # These projections and residual branches are initialized from scratch.
        # Keep their master parameters in float32 even when the released host is
        # stored in bfloat16; the production autocast context still executes
        # their matrix multiplications in the host activation dtype.
        factory_kwargs = {"device": parameter.device, "dtype": torch.float32}
        self.dense_k_proj = nn.ModuleDict(
            {
                modality: nn.Linear(width, hidden_size, bias=False, **factory_kwargs)
                for modality, width in normalized_dense_dims.items()
            }
        )
        self.dense_v_proj = nn.ModuleDict(
            {
                modality: nn.Linear(width, hidden_size, bias=False, **factory_kwargs)
                for modality, width in normalized_dense_dims.items()
            }
        )
        self.object_k_proj = nn.Linear(
            object_address_dim,
            hidden_size,
            bias=False,
            **factory_kwargs,
        )
        self.object_v_proj = nn.Linear(
            object_value_dim,
            hidden_size,
            bias=False,
            **factory_kwargs,
        )
        self.dense_owner_v_proj = nn.Linear(
            object_address_dim,
            hidden_size,
            bias=False,
            **factory_kwargs,
        )
        self.dense_context_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.object_context_norm = ActionExpertRMSNorm(hidden_size, eps=1e-6)
        self.dense_branches = nn.ModuleList(
            [_TypedResidualCrossAttention(vanilla) for _ in vanilla.blocks]
        )
        self.object_branches = nn.ModuleList(
            [_TypedResidualCrossAttention(vanilla) for _ in vanilla.blocks]
        )
        for projection in (
            *self.dense_k_proj.values(),
            *self.dense_v_proj.values(),
            self.object_k_proj,
            self.object_v_proj,
            self.dense_owner_v_proj,
        ):
            _initialize_linear(projection)

    @property
    def vanilla(self) -> ActionExpert:
        return self._vanilla_unregistered

    @property
    def dense_gates(self) -> torch.Tensor:
        return torch.stack([branch.gate for branch in self.dense_branches])

    @property
    def object_gates(self) -> torch.Tensor:
        return torch.stack([branch.gate for branch in self.object_branches])

    def set_posterior_action_context_trainable(self, enabled: bool) -> None:
        """Match object-route trainability to a declared causal ablation."""

        if not isinstance(enabled, bool):
            raise TypeError("posterior action-context trainability must be boolean")
        modules = (
            self.object_k_proj,
            self.object_v_proj,
            self.dense_owner_v_proj,
            self.object_context_norm,
            self.object_branches,
        )
        for module in modules:
            module.requires_grad_(enabled)

    def _validate_host_colocation(self) -> None:
        host_parameter = next(self.vanilla.parameters())
        adapter_parameter = self.object_k_proj.weight
        if host_parameter.device != adapter_parameter.device:
            raise ValueError(
                "vanilla host and PICF adapter must share a device; "
                "move the owning host before constructing or moving the adapter"
            )
        unexpected = {
            (name, parameter.dtype)
            for name, parameter in self.named_parameters()
            if parameter.is_floating_point() and parameter.dtype != torch.float32
        }
        if unexpected:
            raise ValueError(
                f"scratch PICF adapter parameters must use float32 storage: {sorted(unexpected)}"
            )

    def _validate_bank(
        self,
        name: str,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        valid: torch.Tensor | None,
        key_width: int,
        value_width: int,
        expected_device: torch.device,
        expected_dtype: torch.dtype,
    ) -> None:
        if keys is None or values is None or valid is None:
            if not (keys is None and values is None and valid is None):
                raise ValueError(f"{name} keys, values and validity must be all present or absent")
            return
        if keys.ndim != 3 or values.ndim != 3:
            raise ValueError(f"{name} keys and values must be rank three")
        if keys.shape[:2] != values.shape[:2]:
            raise ValueError(f"{name} keys and values must align by batch and token")
        if keys.shape[2] != key_width or values.shape[2] != value_width:
            raise ValueError(f"{name} feature width does not match the adapter contract")
        if valid.dtype != torch.bool or valid.shape != keys.shape[:2]:
            raise ValueError(f"{name} validity must be a bool batch-by-token tensor")
        if keys.device != expected_device or values.device != expected_device:
            raise ValueError(f"{name} tensors must already be on the action-expert device")
        if valid.device != expected_device:
            raise ValueError(f"{name} validity must be on the action-expert device")
        if keys.dtype != expected_dtype or values.dtype != expected_dtype:
            raise ValueError(
                f"{name} tensors must already use the action-expert dtype "
                f"{expected_dtype}; got keys={keys.dtype}, values={values.dtype}"
            )
        if self.validate_tensor_values:
            if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
                raise ValueError(f"{name} contains NaN or infinity")
            if (keys[~valid] != 0.0).any() or (values[~valid] != 0.0).any():
                raise ValueError(f"{name} masked padding keys and values must be exactly zero")

    @staticmethod
    def _attention_mask(
        valid: torch.Tensor | None,
        dtype: torch.dtype,
        log_prior: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        if valid is None:
            if log_prior is not None:
                raise ValueError("object log prior cannot exist without validity")
            return None
        if log_prior is None:
            log_prior = torch.zeros(valid.shape, dtype=dtype, device=valid.device)
        elif (
            log_prior.shape != valid.shape
            or log_prior.device != valid.device
            or log_prior.dtype != dtype
        ):
            raise ValueError("attention log prior must align with validity and adapter dtype")
        minimum = torch.finfo(dtype).min
        additive = torch.where(
            valid,
            log_prior,
            torch.full_like(log_prior, minimum),
        )
        return additive[:, None, None, :]

    def _make_layer_contexts(
        self,
        key_base: torch.Tensor,
        value_base: torch.Tensor,
        branches: nn.ModuleList,
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]]:
        key_heads = self.vanilla._reshape_hidden_to_heads(key_base)
        value_heads = self.vanilla._reshape_hidden_to_heads(value_base)
        contexts = []
        for branch in branches:
            normalized_key = key_heads
            if branch.attention.k_norm is not None:
                normalized_key = branch.attention.k_norm(key_heads.transpose(1, 2)).transpose(1, 2)
            contexts.append((normalized_key, value_heads))
        return contexts

    def _project_bank(
        self,
        keys: torch.Tensor | None,
        values: torch.Tensor | None,
        key_projection: nn.Linear,
        value_projection: nn.Linear,
        context_norm: nn.Module,
        branches: nn.ModuleList,
    ) -> Sequence[tuple[torch.Tensor, torch.Tensor]] | None:
        if keys is None or values is None or keys.shape[1] == 0:
            return None
        key_base = context_norm(key_projection(keys))
        value_base = context_norm(value_projection(values))
        return self._make_layer_contexts(key_base, value_base, branches)

    def _project_dense_banks(
        self,
        banks: tuple[NativeTokenBank, ...],
        owner_addresses: tuple[torch.Tensor | None, ...],
        expected_device: torch.device,
        expected_dtype: torch.dtype,
    ) -> tuple[Sequence[tuple[torch.Tensor, torch.Tensor]] | None, torch.Tensor | None]:
        if not banks:
            return None, None
        seen: set[str] = set()
        key_parts = []
        value_parts = []
        valid_parts = []
        batch_size = None
        if len(owner_addresses) != len(banks):
            raise ValueError("dense owner-address metadata must align with dense banks")
        for bank, owner_address in zip(banks, owner_addresses, strict=True):
            if bank.modality in seen:
                raise ValueError(f"dense modality {bank.modality} appears more than once")
            seen.add(bank.modality)
            if bank.modality not in self.dense_k_proj:
                raise ValueError(f"dense modality {bank.modality} is not configured")
            width = self.dense_k_proj[bank.modality].in_features
            self._validate_bank(
                f"dense/{bank.modality}",
                bank.tokens,
                bank.tokens,
                bank.valid,
                width,
                width,
                expected_device,
                expected_dtype,
            )
            if batch_size is None:
                batch_size = bank.tokens.shape[0]
            elif bank.tokens.shape[0] != batch_size:
                raise ValueError("all dense modalities must share a batch size")
            projected_key = self.dense_k_proj[bank.modality](bank.tokens)
            projected_value = self.dense_v_proj[bank.modality](bank.tokens)
            if owner_address is not None:
                projected_key = projected_key + self.object_k_proj(owner_address)
                projected_value = projected_value + self.dense_owner_v_proj(owner_address)
            key_parts.append(projected_key)
            value_parts.append(projected_value)
            valid_parts.append(bank.valid)
        if not key_parts:
            return None, None
        concatenated_keys = torch.cat(key_parts, dim=1)
        concatenated_values = torch.cat(value_parts, dim=1)
        if concatenated_keys.shape[1] == 0:
            return None, None
        key_base = self.dense_context_norm(concatenated_keys)
        value_base = self.dense_context_norm(concatenated_values)
        valid = torch.cat(valid_parts, dim=1)
        return (
            self._make_layer_contexts(key_base, value_base, self.dense_branches),
            self._attention_mask(valid, expected_dtype),
        )

    def prepare_picf_context(self, evidence: PICFActionEvidence) -> PICFActionContext:
        self._validate_host_colocation()
        evidence.batch_size()
        host_parameter = next(self.vanilla.parameters())
        expected_device = host_parameter.device
        expected_dtype = host_parameter.dtype
        owner_addresses = evidence.ownership_weighted_addresses(
            validate_tensor_values=self.validate_tensor_values
        )
        dense_contexts, dense_mask = self._project_dense_banks(
            evidence.dense_banks,
            owner_addresses,
            expected_device,
            expected_dtype,
        )
        self._validate_bank(
            "object",
            evidence.object_address,
            evidence.object_value,
            evidence.object_valid,
            self.object_k_proj.in_features,
            self.object_v_proj.in_features,
            expected_device,
            expected_dtype,
        )
        return PICFActionContext(
            dense_kv_contexts=dense_contexts,
            dense_mask=dense_mask,
            object_kv_contexts=self._project_bank(
                evidence.object_address,
                evidence.object_value,
                self.object_k_proj,
                self.object_v_proj,
                self.object_context_norm,
                self.object_branches,
            ),
            object_mask=self._attention_mask(
                evidence.object_valid,
                expected_dtype,
                evidence.object_log_prior,
            ),
        )

    @staticmethod
    def _expand_context_tensor(
        tensor: torch.Tensor | None,
        *,
        target_batch_size: int,
    ) -> torch.Tensor | None:
        """Repeat one observation context over official flow-time samples."""

        if tensor is None or tensor.shape[0] == target_batch_size:
            return tensor
        source_batch_size = tensor.shape[0]
        if source_batch_size <= 0 or target_batch_size % source_batch_size:
            raise ValueError("PICF context batch must divide the action flow batch exactly")
        return tensor.repeat_interleave(target_batch_size // source_batch_size, dim=0)

    def _select_picf_layer_context(
        self,
        context: PICFActionContext | None,
        *,
        layer_index: int,
        target_batch_size: int,
    ) -> _PICFActionLayerContext | None:
        if context is None:
            return None

        def select_layer(
            name: str,
            layers: Sequence[tuple[torch.Tensor, torch.Tensor]] | None,
            mask: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor] | None:
            if layers is None:
                if mask is not None:
                    raise ValueError(f"{name} PICF mask has no corresponding context")
                return None
            if len(layers) != len(self.vanilla.blocks):
                raise ValueError(f"{name} PICF context layer count differs from vanilla")
            key, value = layers[layer_index]
            return (
                self._expand_context_tensor(key, target_batch_size=target_batch_size),
                self._expand_context_tensor(value, target_batch_size=target_batch_size),
            )

        return _PICFActionLayerContext(
            dense_kv_context=select_layer(
                "dense",
                context.dense_kv_contexts,
                context.dense_mask,
            ),
            dense_mask=self._expand_context_tensor(
                context.dense_mask,
                target_batch_size=target_batch_size,
            ),
            object_kv_context=select_layer(
                "object",
                context.object_kv_contexts,
                context.object_mask,
            ),
            object_mask=self._expand_context_tensor(
                context.object_mask,
                target_batch_size=target_batch_size,
            ),
        )

    def apply_training_layer(
        self,
        action_hidden_states: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        layer_index: int,
        cross_kv: tuple[torch.Tensor, torch.Tensor],
        self_attn_mask: torch.Tensor | None,
        attn_mask: torch.Tensor | None,
        is_causal: bool,
        modulation: tuple[torch.Tensor, ...] | None,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        context: PICFActionContext | None,
    ) -> torch.Tensor:
        """Run one official action block with an explicit typed PICF residual.

        This protocol is called by the pinned LeRobot joint VLM/action training
        loop. The policy may flatten several flow times into the batch axis, so
        one observation context is repeated in the same example-major order as
        the official implementation. No context is cached on the module.
        """

        self._validate_host_colocation()
        if not isinstance(layer_index, int) or isinstance(layer_index, bool):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < len(self.vanilla.blocks):
            raise IndexError("layer_index is outside the MolmoAct2 action expert")
        if not isinstance(is_causal, bool):
            raise TypeError("is_causal must be a boolean")
        host_is_causal = self.vanilla.config.causal_attn
        if not isinstance(host_is_causal, bool):
            raise TypeError("MolmoAct2 causal_attn config must be a boolean")
        if is_causal != host_is_causal:
            raise ValueError("host and PICF adapter disagree on causal action attention")
        layer_context = self._select_picf_layer_context(
            context,
            layer_index=layer_index,
            target_batch_size=action_hidden_states.shape[0],
        )
        return self._forward_block(
            layer_index,
            action_hidden_states,
            conditioning,
            native_kv=cross_kv,
            native_attention_mask=attn_mask,
            self_attention_mask=self_attn_mask,
            rope_cache=rope_cache,
            modulation=modulation,
            picf_context=layer_context,
        )

    def _forward_block(
        self,
        layer: int,
        x: torch.Tensor,
        conditioning: torch.Tensor,
        *,
        native_kv: tuple[torch.Tensor, torch.Tensor],
        native_attention_mask: torch.Tensor | None,
        self_attention_mask: torch.Tensor | None,
        rope_cache: tuple[torch.Tensor, torch.Tensor] | None,
        modulation: tuple[torch.Tensor, ...] | None,
        picf_context: _PICFActionLayerContext | None,
    ) -> torch.Tensor:
        if not isinstance(layer, int) or isinstance(layer, bool):
            raise TypeError("layer must be an integer")
        if not 0 <= layer < len(self.vanilla.blocks):
            raise IndexError("layer is outside the MolmoAct2 action expert")
        block = self.vanilla.blocks[layer]
        if picf_context is not None:
            for name, contexts, mask in (
                ("dense", picf_context.dense_kv_context, picf_context.dense_mask),
                ("object", picf_context.object_kv_context, picf_context.object_mask),
            ):
                if contexts is None:
                    if mask is not None:
                        raise ValueError(f"{name} PICF mask has no corresponding context")
                    continue
                key, value = contexts
                if key.shape[0] != x.shape[0] or value.shape[0] != x.shape[0]:
                    raise ValueError(f"{name} PICF context batch differs from action batch")
                if mask is not None and mask.shape[0] != x.shape[0]:
                    raise ValueError(f"{name} PICF mask batch differs from action batch")
        if modulation is None:
            modulation = block.modulation(conditioning).chunk(9, dim=1)
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mca,
            scale_mca,
            gate_mca,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = modulation
        x = x + gate_msa.unsqueeze(1) * block.self_attn(
            _modulate(block.self_norm(x), shift_msa, scale_msa),
            attn_mask=self_attention_mask,
            is_causal=self.vanilla.config.causal_attn,
            rope_cache=rope_cache,
        )
        x = x + gate_mca.unsqueeze(1) * block.cross_attn(
            _modulate(block.cross_norm(x), shift_mca, scale_mca),
            kv_k=native_kv[0],
            kv_v=native_kv[1],
            attn_mask=native_attention_mask,
        )
        if picf_context is not None and picf_context.dense_kv_context is not None:
            x = x + self.dense_branches[layer](
                x,
                shift_mca,
                scale_mca,
                kv=picf_context.dense_kv_context,
                attention_mask=picf_context.dense_mask,
            )
        if picf_context is not None and picf_context.object_kv_context is not None:
            x = x + self.object_branches[layer](
                x,
                shift_mca,
                scale_mca,
                kv=picf_context.object_kv_context,
                attention_mask=picf_context.object_mask,
            )
        x = x + gate_mlp.unsqueeze(1) * block.mlp(_modulate(block.ff_norm(x), shift_mlp, scale_mlp))
        return x

    def forward_with_context(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        context: ActionExpertContext,
        picf_context: PICFActionContext | None = None,
        modulation: ActionExpertStepModulation | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = actions.shape
        if sequence_length > self.vanilla.config.max_action_horizon:
            raise ValueError("action sequence exceeds the vanilla maximum horizon")
        if len(context.kv_contexts) != len(self.vanilla.blocks):
            raise ValueError("native context layer count differs from the vanilla expert")
        if picf_context is not None:
            for name, contexts in {
                "dense": picf_context.dense_kv_contexts,
                "object": picf_context.object_kv_contexts,
            }.items():
                if contexts is not None and len(contexts) != len(self.vanilla.blocks):
                    raise ValueError(f"{name} PICF context layer count differs from vanilla")

        if modulation is None:
            conditioning = self.vanilla._time_conditioning(timesteps)
            block_modulations: Sequence[tuple[torch.Tensor, ...] | None] = [None] * len(
                self.vanilla.blocks
            )
            final_modulation = None
        else:
            conditioning = modulation.conditioning
            block_modulations = modulation.block_modulations
            final_modulation = modulation.final_modulation

        x = self.vanilla.action_embed(actions)
        if context.valid_action is not None:
            x = x * context.valid_action
        for layer, (native_kv, block_modulation) in enumerate(
            zip(context.kv_contexts, block_modulations, strict=True)
        ):
            layer_picf_context = self._select_picf_layer_context(
                picf_context,
                layer_index=layer,
                target_batch_size=x.shape[0],
            )
            x = self._forward_block(
                layer,
                x,
                conditioning,
                native_kv=native_kv,
                native_attention_mask=context.cross_mask,
                self_attention_mask=context.self_mask,
                rope_cache=context.rope_cache,
                modulation=block_modulation,
                picf_context=layer_picf_context,
            )
            if context.valid_action is not None:
                x = x * context.valid_action
        output = self.vanilla.final_layer(
            x,
            conditioning,
            modulation=final_modulation,
        )
        if context.valid_action is not None:
            output = output * context.valid_action
        if output.shape[:2] != (batch_size, sequence_length):
            raise RuntimeError("wrapped action expert changed batch or horizon shape")
        return output

    def forward(
        self,
        actions: torch.Tensor,
        timesteps: torch.Tensor,
        *,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        encoder_attention_mask: torch.Tensor | None = None,
        action_attention_mask: torch.Tensor | None = None,
        evidence: PICFActionEvidence | None = None,
    ) -> torch.Tensor:
        self._validate_host_colocation()
        batch_size, sequence_length, _ = actions.shape
        native_context = self.vanilla.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            action_attention_mask=action_attention_mask,
            state_embeddings=None,
            batch_size=batch_size,
            seq_len=sequence_length,
            device=actions.device,
            dtype=actions.dtype,
        )
        picf_context = None if evidence is None else self.prepare_picf_context(evidence)
        return self.forward_with_context(
            actions,
            timesteps,
            context=native_context,
            picf_context=picf_context,
        )


class MolmoAct2PICFForConditionalGeneration(nn.Module):
    """Loaded official MolmoAct2 host plus a separately serialized PICF adapter.

    `generate_actions_from_inputs` substantially adapts
    `MolmoAct2Model.generate_actions_from_inputs` and `_run_action_flow_loop`
    from the pinned Apache-2.0 upstream commit named in this file's module
    header. The adaptation adds an explicit immutable PICF context to every
    flow step and intentionally bypasses the upstream CUDA graph manager until
    a graph-safe typed-context contract is proven.
    """

    def __init__(
        self,
        host: MolmoAct2ForConditionalGeneration,
        *,
        dense_token_dims: Mapping[str, int],
        object_address_dim: int,
        object_value_dim: int,
        host_checkpoint_identity: MolmoAct2HostCheckpointIdentity | None = None,
        validate_tensor_values: bool = True,
    ) -> None:
        super().__init__()
        if host.model.action_expert is None:
            raise ValueError("the loaded MolmoAct2 host has no continuous action expert")
        if MOLMO_VISION_PATCH_MODALITY in dense_token_dims:
            vision_backbone = host.model.vision_backbone
            expected_vision_width = vision_backbone.vit_config.hidden_size * len(
                vision_backbone.vit_layers
            )
            configured_vision_width = dense_token_dims[MOLMO_VISION_PATCH_MODALITY]
            if configured_vision_width != expected_vision_width:
                raise ValueError(
                    "molmo_vision_patch width must equal the concatenated pre-pooling "
                    f"ViT width ({expected_vision_width}), got {configured_vision_width}"
                )
        self.host = host
        if host_checkpoint_identity is not None and not isinstance(
            host_checkpoint_identity, MolmoAct2HostCheckpointIdentity
        ):
            raise TypeError("host_checkpoint_identity must use the typed identity contract")
        self.host_checkpoint_identity = host_checkpoint_identity
        self.action_adapter = MolmoAct2PICFActionExpert(
            host.model.action_expert,
            dense_token_dims=dense_token_dims,
            object_address_dim=object_address_dim,
            object_value_dim=object_value_dim,
            validate_tensor_values=validate_tensor_values,
        )

    @property
    def config(self):
        return self.host.config

    @property
    def adapter_dimensions(self) -> dict[str, Any]:
        return {
            "dense_token_dims": {
                name: projection.in_features
                for name, projection in self.action_adapter.dense_k_proj.items()
            },
            "object_address_dim": self.action_adapter.object_k_proj.in_features,
            "object_value_dim": self.action_adapter.object_v_proj.in_features,
        }

    def forward(self, *args, **kwargs):
        return self.host(*args, **kwargs)

    def _validate_encoder_kv_states(
        self,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]],
        *,
        input_ids: torch.Tensor,
        encoder_attention_mask: torch.Tensor | None,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        states = tuple(encoder_kv_states)
        expected_layers = len(self.action_adapter.vanilla.blocks)
        if len(states) != expected_layers:
            raise ValueError(
                f"encoder_kv_states must contain {expected_layers} layers, got {len(states)}"
            )
        if input_ids.ndim != 2 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be one int64 batch-by-token tensor")
        batch_size, sequence_length = input_ids.shape
        if batch_size <= 0 or sequence_length <= 0:
            raise ValueError("input_ids batch and sequence dimensions must be positive")

        expected_width = self.action_adapter.vanilla.llm_kv_dim
        reference_device: torch.device | None = None
        reference_dtype: torch.dtype | None = None
        for layer_index, layer in enumerate(states):
            if not isinstance(layer, tuple | list) or len(layer) != 2:
                raise ValueError(f"encoder KV layer {layer_index} must be one key/value pair")
            key, value = layer
            if not isinstance(key, torch.Tensor) or not isinstance(value, torch.Tensor):
                raise TypeError(f"encoder KV layer {layer_index} must contain tensors")
            expected_shape = (batch_size, sequence_length, expected_width)
            if key.shape != expected_shape or value.shape != expected_shape:
                raise ValueError(
                    f"encoder KV layer {layer_index} must contain key/value shape {expected_shape}"
                )
            if not key.is_floating_point() or not value.is_floating_point():
                raise ValueError(f"encoder KV layer {layer_index} must be floating point")
            if key.device != value.device or key.dtype != value.dtype:
                raise ValueError(
                    f"encoder KV layer {layer_index} key/value must share dtype and device"
                )
            if not torch.isfinite(key).all() or not torch.isfinite(value).all():
                raise ValueError(f"encoder KV layer {layer_index} contains NaN or infinity")
            if reference_device is None:
                reference_device = key.device
                reference_dtype = key.dtype
            elif key.device != reference_device or key.dtype != reference_dtype:
                raise ValueError("all encoder KV layers must share one dtype and device")

        if input_ids.device != reference_device:
            raise ValueError("input_ids and encoder KV states must share a device")
        if encoder_attention_mask is not None:
            if encoder_attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("encoder attention mask must align with encoder KV tokens")
            if encoder_attention_mask.device != reference_device:
                raise ValueError("encoder attention mask and encoder KV states must share a device")
            if encoder_attention_mask.dtype == torch.bool:
                pass
            elif encoder_attention_mask.dtype in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }:
                if ((encoder_attention_mask != 0) & (encoder_attention_mask != 1)).any():
                    raise ValueError("integer encoder attention mask values must be zero or one")
            else:
                raise ValueError("encoder attention mask must be boolean or integer")
        return tuple((key, value) for key, value in states)

    def encode_inputs_for_picf(
        self,
        *,
        input_ids: torch.LongTensor,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
    ) -> MolmoAct2EncoderBundle:
        """Encode native context and expose exact image patches without re-encoding.

        Image resize mode and text-only inputs are supported. Video and
        multicrop semantics remain fail-closed until their source-patch
        partition is separately proved against the released processor.
        """

        if pixel_values_videos is not None:
            raise NotImplementedError(
                "exact MolmoAct2 video dense-token extraction is not yet validated"
            )
        if video_token_pooling is not None or video_grids is not None:
            raise ValueError("video metadata cannot be supplied without validated video extraction")

        model = self.host.model
        images, token_pooling = model.merge_visual_inputs(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_token_pooling=image_token_pooling,
            image_grids=image_grids,
            image_num_crops=image_num_crops,
        )
        safe_input_ids = input_ids * (input_ids != -1).to(input_ids.dtype)
        input_embeddings = model.transformer.wte(safe_input_ids)
        vision_patch_bank = None
        if images is not None:
            if token_pooling is None:
                raise ValueError("MolmoAct2 image inputs require pooling indices")
            pooled_features, dense_features, dense_valid = _encode_and_pool_vision_once(
                model.vision_backbone,
                images,
                token_pooling,
            )
            is_image_patch = safe_input_ids.view(-1) == model.config.image_patch_id
            if int(is_image_patch.sum().item()) != len(pooled_features):
                raise ValueError("MolmoAct2 pooled image features do not align with input tokens")
            input_embeddings.view(-1, input_embeddings.shape[-1])[is_image_patch] += (
                pooled_features.to(input_embeddings.device)
            )
            vision_patch_bank = NativeTokenBank(
                modality=MOLMO_VISION_PATCH_MODALITY,
                tokens=dense_features,
                valid=dense_valid,
            )
        input_embeddings = model.transformer.emb_drop(input_embeddings)
        outputs = model(
            inputs_embeds=input_embeddings,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            use_cache=True,
        )
        encoder_kv_states = tuple(model._extract_kv_states(outputs.past_key_values))
        encoder_attention_mask = model._get_encoder_attention_mask(input_ids, attention_mask)
        if encoder_attention_mask is None:
            raise RuntimeError("MolmoAct2 encoder attention mask was unexpectedly absent")
        return MolmoAct2EncoderBundle(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            vision_patch_bank=vision_patch_bank,
        )

    @torch.no_grad()
    def generate_actions_from_inputs(
        self,
        *,
        input_ids: torch.LongTensor,
        evidence: PICFActionEvidence | None,
        pixel_values: torch.Tensor | None = None,
        image_token_pooling: torch.Tensor | None = None,
        image_grids: torch.Tensor | None = None,
        image_num_crops: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_token_pooling: torch.Tensor | None = None,
        video_grids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        states: torch.Tensor | None = None,
        action_dim_is_pad: torch.Tensor | None = None,
        action_horizon: int | None = None,
        num_steps: int | None = None,
        generator: torch.Generator | None = None,
        encoder_kv_states: Sequence[tuple[torch.Tensor, torch.Tensor]] | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model = self.host.model
        vanilla = self.action_adapter.vanilla
        if encoder_kv_states is None:
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_token_pooling=image_token_pooling,
                image_grids=image_grids,
                image_num_crops=image_num_crops,
                pixel_values_videos=pixel_values_videos,
                video_token_pooling=video_token_pooling,
                video_grids=video_grids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                use_cache=True,
            )
            encoder_kv_states = model._extract_kv_states(outputs.past_key_values)
            encoder_attention_mask = model._get_encoder_attention_mask(input_ids, attention_mask)
        else:
            ignored_visual_inputs = {
                "pixel_values": pixel_values,
                "image_token_pooling": image_token_pooling,
                "image_grids": image_grids,
                "image_num_crops": image_num_crops,
                "pixel_values_videos": pixel_values_videos,
                "video_token_pooling": video_token_pooling,
                "video_grids": video_grids,
            }
            supplied = sorted(
                name for name, value in ignored_visual_inputs.items() if value is not None
            )
            if supplied:
                raise ValueError(
                    "raw visual inputs cannot accompany external encoder_kv_states; "
                    f"they would be ignored: {supplied}"
                )
            if encoder_attention_mask is None:
                encoder_attention_mask = model._get_encoder_attention_mask(
                    input_ids, attention_mask
                )

        encoder_kv_states = self._validate_encoder_kv_states(
            encoder_kv_states,
            input_ids=input_ids,
            encoder_attention_mask=encoder_attention_mask,
        )
        source_batch_size = input_ids.shape[0]
        if evidence is not None:
            evidence_batch_size = evidence.batch_size()
            if evidence_batch_size is not None and evidence_batch_size != source_batch_size:
                raise ValueError("PICF evidence and encoder KV states must share a batch size")

        depth_gate, depth_mask = model._depth_gate_from_condition(
            input_ids=input_ids,
            encoder_attention_mask=encoder_attention_mask,
            layer_kv_states=encoder_kv_states,
        )
        encoder_kv_states = model._apply_depth_gate_to_layer_kv_states(
            encoder_kv_states,
            depth_mask,
            depth_gate,
        )
        raw_steps = model.config.flow_matching_num_steps if num_steps is None else num_steps
        if not isinstance(raw_steps, int) or isinstance(raw_steps, bool):
            raise ValueError(f"num_steps must be an integer, got {raw_steps!r}")
        steps = raw_steps
        if steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {steps}")
        source_tensor = encoder_kv_states[0][0]
        batch_size = source_tensor.shape[0]
        device = source_tensor.device
        if action_horizon is not None and (
            not isinstance(action_horizon, int) or isinstance(action_horizon, bool)
        ):
            raise ValueError("action_horizon must be an integer when supplied")
        resolved_horizon = model._resolve_action_horizon(action_horizon)
        if action_dim_is_pad is not None and action_dim_is_pad.dtype != torch.bool:
            raise ValueError("action_dim_is_pad must be a boolean tensor")
        trajectory = torch.randn(
            (batch_size, resolved_horizon, model.config.max_action_dim),
            device=device,
            dtype=vanilla.action_embed.weight.dtype,
            generator=generator,
        )
        trajectory = model._mask_action_dim_tensor(
            trajectory,
            action_dim_is_pad=action_dim_is_pad,
            enabled=model.config.mask_action_dim_padding,
        )
        action_context = vanilla.prepare_context(
            encoder_kv_states=encoder_kv_states,
            encoder_attention_mask=encoder_attention_mask,
            state_embeddings=states,
            batch_size=batch_size,
            seq_len=trajectory.shape[1],
            device=device,
            dtype=trajectory.dtype,
        )
        picf_context = (
            None if evidence is None else self.action_adapter.prepare_picf_context(evidence)
        )
        flow_timesteps = [
            torch.full((batch_size,), idx / steps, device=device, dtype=torch.float32)
            for idx in range(steps)
        ]
        modulations = vanilla.get_or_prepare_modulation_cache(
            flow_timesteps,
            cache_key=(steps, batch_size, device, trajectory.dtype),
        )
        dt = 1.0 / steps
        for modulation in modulations:
            velocity = self.action_adapter.forward_with_context(
                trajectory,
                modulation.conditioning,
                context=action_context,
                picf_context=picf_context,
                modulation=modulation,
            )
            velocity = model._mask_action_dim_tensor(
                velocity,
                action_dim_is_pad=action_dim_is_pad,
                enabled=model.config.mask_action_dim_padding,
            )
            trajectory = trajectory + dt * velocity
            trajectory = model._mask_action_dim_tensor(
                trajectory,
                action_dim_is_pad=action_dim_is_pad,
                enabled=model.config.mask_action_dim_padding,
            )
        return trajectory

    def save_adapter_pretrained(self, output_dir: str | Path) -> None:
        from safetensors.torch import save_file

        if self.host_checkpoint_identity is None:
            raise RuntimeError(
                "standalone adapter publication requires an externally verified host "
                "checkpoint identity"
            )
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.action_adapter.state_dict().items()
        }
        if not state:
            raise RuntimeError("adapter state is unexpectedly empty")
        if any(name.startswith("vanilla.") for name in state):
            raise RuntimeError("adapter state unexpectedly contains vanilla host parameters")
        if any(
            tensor.is_floating_point() and not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise ValueError("adapter state contains NaN or infinity")

        weights_path = output / _ADAPTER_WEIGHTS_NAME
        config_path = output / _ADAPTER_CONFIG_NAME
        temporary_paths: list[Path] = []
        try:
            with tempfile.NamedTemporaryFile(
                dir=output,
                prefix=f".{_ADAPTER_WEIGHTS_NAME}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_weights = Path(handle.name)
            temporary_paths.append(temporary_weights)
            save_file(state, temporary_weights)
            with temporary_weights.open("rb") as handle:
                os.fsync(handle.fileno())
            weights_sha256 = _sha256_file(temporary_weights)

            host_source = getattr(self.host.config, "_name_or_path", None)
            payload: dict[str, Any] = {
                "format": _ADAPTER_FORMAT,
                "host_architecture": type(self.host).__name__,
                "host_source": str(host_source) if host_source else None,
                "host_checkpoint": self.host_checkpoint_identity.payload,
                "weights_sha256": weights_sha256,
                **self.adapter_dimensions,
            }
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=output,
                prefix=f".{_ADAPTER_CONFIG_NAME}.",
                suffix=".tmp",
                encoding="utf-8",
                delete=False,
            ) as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
                temporary_config = Path(handle.name)
            temporary_paths.append(temporary_config)

            os.replace(temporary_weights, weights_path)
            temporary_paths.remove(temporary_weights)
            os.replace(temporary_config, config_path)
            temporary_paths.remove(temporary_config)
            _fsync_directory(output)
        finally:
            for temporary_path in temporary_paths:
                temporary_path.unlink(missing_ok=True)

    def load_adapter_pretrained(self, adapter_dir: str | Path) -> None:
        from safetensors.torch import load_file

        if self.host_checkpoint_identity is None:
            raise RuntimeError(
                "standalone adapter loading requires an externally verified host "
                "checkpoint identity"
            )
        root = Path(adapter_dir)
        payload = json.loads((root / _ADAPTER_CONFIG_NAME).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("MolmoAct2 PICF adapter config must be a JSON object")
        if payload.get("format") != _ADAPTER_FORMAT:
            raise ValueError("unsupported MolmoAct2 PICF adapter format")
        expected_architecture = type(self.host).__name__
        if payload.get("host_architecture") != expected_architecture:
            raise ValueError(
                "adapter host architecture differs: "
                f"checkpoint={payload.get('host_architecture')!r}, "
                f"model={expected_architecture!r}"
            )
        raw_host_checkpoint = payload.get("host_checkpoint")
        if not isinstance(raw_host_checkpoint, dict):
            raise ValueError("adapter host_checkpoint must be a JSON object")
        checkpoint_identity = MolmoAct2HostCheckpointIdentity(
            checkpoint_id=raw_host_checkpoint.get("checkpoint_id"),
            revision=raw_host_checkpoint.get("revision"),
            manifest_sha256=raw_host_checkpoint.get("manifest_sha256"),
        )
        if checkpoint_identity != self.host_checkpoint_identity:
            raise ValueError(
                "adapter host checkpoint differs: "
                f"checkpoint={checkpoint_identity.payload}, "
                f"model={self.host_checkpoint_identity.payload}"
            )
        expected = self.adapter_dimensions
        raw_dense_dims = payload.get("dense_token_dims")
        if not isinstance(raw_dense_dims, dict):
            raise ValueError("adapter dense_token_dims must be a JSON object")
        actual = {
            "dense_token_dims": {
                _strict_adapter_modality(name): _strict_adapter_dimension(width)
                for name, width in raw_dense_dims.items()
            },
            "object_address_dim": _strict_adapter_dimension(payload.get("object_address_dim")),
            "object_value_dim": _strict_adapter_dimension(payload.get("object_value_dim")),
        }
        if actual != expected:
            raise ValueError(f"adapter dimensions differ: checkpoint={actual}, model={expected}")
        expected_hash = payload.get("weights_sha256")
        if (
            not isinstance(expected_hash, str)
            or len(expected_hash) != 64
            or any(character not in "0123456789abcdef" for character in expected_hash)
        ):
            raise ValueError("adapter weights_sha256 must be one lowercase SHA-256 digest")
        weights_path = root / _ADAPTER_WEIGHTS_NAME
        actual_hash = _sha256_file(weights_path)
        if actual_hash != expected_hash:
            raise ValueError(
                f"adapter weight hash differs: checkpoint={expected_hash}, actual={actual_hash}"
            )
        state = load_file(weights_path, device="cpu")
        if any(
            tensor.is_floating_point() and not torch.isfinite(tensor).all()
            for tensor in state.values()
        ):
            raise ValueError("adapter checkpoint contains NaN or infinity")
        self.action_adapter.load_state_dict(state, strict=True)

    @classmethod
    def from_pretrained(
        cls,
        host_name_or_path: str | Path,
        *,
        dense_token_dims: Mapping[str, int],
        object_address_dim: int,
        object_value_dim: int,
        adapter_dir: str | Path | None = None,
        host_checkpoint_identity: MolmoAct2HostCheckpointIdentity | None = None,
        **host_kwargs,
    ) -> MolmoAct2PICFForConditionalGeneration:
        host = MolmoAct2ForConditionalGeneration.from_pretrained(
            host_name_or_path,
            **host_kwargs,
        )
        model = cls(
            host,
            dense_token_dims=dense_token_dims,
            object_address_dim=object_address_dim,
            object_value_dim=object_value_dim,
            host_checkpoint_identity=host_checkpoint_identity,
        )
        if adapter_dir is not None:
            model.load_adapter_pretrained(adapter_dir)
        return model


def install_molmoact2_lerobot_picf_adapter(
    policy: nn.Module,
    adapter: MolmoAct2PICFActionExpert,
) -> None:
    """Register PICF on the pinned patched LeRobot policy before DDP/FSDP.

    The exact host action expert remains the sole owner of vanilla parameters;
    the policy registers only the PICF adapter parameters. Identity validation
    prevents accidentally coupling an adapter to a different loaded checkpoint.
    """

    setter = getattr(policy, "set_action_layer_adapter", None)
    get_expert = getattr(policy, "_action_expert", None)
    if not callable(setter) or not callable(get_expert):
        raise TypeError(
            "policy is not the pinned patched MolmoAct2 LeRobot host; "
            "apply references/patches/molmoact2_lerobot_action_layer_adapter.patch"
        )
    if adapter.vanilla is not get_expert():
        raise ValueError("PICF adapter wraps a different MolmoAct2 action expert")
    installed = getattr(policy, "action_layer_adapter", None)
    if installed is not None and installed is not adapter:
        raise RuntimeError("MolmoAct2 policy already has a different action-layer adapter")
    setter(adapter)
    if getattr(policy, "action_layer_adapter", None) is not adapter:
        raise RuntimeError("MolmoAct2 policy did not register the PICF adapter")

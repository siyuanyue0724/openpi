from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass
class Object3DSlotOutput:
    """Point-to-slot assignment output for PICF Object3D evidence."""

    slots: torch.Tensor
    point_slot_weights: torch.Tensor
    object_point_priors: torch.Tensor
    centers: torch.Tensor
    covariance_diag: torch.Tensor
    objectness: torch.Tensor
    background_weight: torch.Tensor
    encoder_attention: torch.Tensor
    reconstruction: torch.Tensor | None = None


class Object3DSlotAttention(nn.Module):
    """SlotAttention block adapted from SlotLifter's 3D slot encoder.

    SlotLifter uses SlotAttention to form object slots and a JointDecoder to
    map sampled 3D points to empty + object slots.  This module keeps that
    exact structural split for CALVIN RGB-D point tokens, but does not include
    SlotLifter's NeRF renderer because CALVIN is a robot manipulation stream,
    not a many-view static-scene reconstruction dataset.
    """

    def __init__(
        self,
        input_dim: int,
        slot_dim: int = 128,
        num_slots: int = 8,
        num_iterations: int = 3,
        num_heads: int = 1,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if slot_dim % num_heads != 0:
            raise ValueError(f"slot_dim={slot_dim} must be divisible by num_heads={num_heads}")
        self.input_dim = int(input_dim)
        self.slot_dim = int(slot_dim)
        self.num_slots = int(num_slots)
        self.num_iterations = int(num_iterations)
        self.num_heads = int(num_heads)
        self.epsilon = float(epsilon)

        self.feature_proj = nn.Linear(input_dim, slot_dim)
        self.norm_features = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)
        self.project_q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_k = nn.Linear(slot_dim, slot_dim, bias=False)
        self.project_v = nn.Linear(slot_dim, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 4),
            nn.ReLU(),
            nn.Linear(slot_dim * 4, slot_dim),
        )
        self.slots_init = nn.Parameter(torch.empty(1, num_slots, slot_dim))
        nn.init.xavier_uniform_(self.slots_init)

    def forward(self, point_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if point_features.ndim != 3:
            raise ValueError(f"point_features must be [B,N,D], got {tuple(point_features.shape)}")
        batch, _, _ = point_features.shape
        features = self.norm_features(self.feature_proj(point_features))
        k = self.project_k(features)
        v = self.project_v(features)
        slots = self.slots_init.expand(batch, -1, -1)
        attn = features.new_zeros((batch, self.num_slots, features.shape[1]))

        for _ in range(max(self.num_iterations, 1)):
            slots_prev = slots
            slots_norm = self.norm_slots(slots)
            q = self.project_q(slots_norm)
            q = q.reshape(batch, self.num_slots, self.num_heads, self.slot_dim // self.num_heads).transpose(1, 2)
            kh = k.reshape(batch, -1, self.num_heads, self.slot_dim // self.num_heads).transpose(1, 2)
            logits = torch.matmul(q, kh.transpose(-1, -2)) * ((self.slot_dim // self.num_heads) ** -0.5)
            logits = logits.mean(dim=1)
            attn = F.softmax(logits, dim=1)
            attn_norm = attn / (attn.sum(dim=-1, keepdim=True) + self.epsilon)
            updates = torch.einsum("bkn,bnd->bkd", attn_norm, v)
            slots = self.gru(updates.reshape(-1, self.slot_dim), slots_prev.reshape(-1, self.slot_dim))
            slots = slots.reshape(batch, self.num_slots, self.slot_dim)
            slots = slots + self.mlp(self.norm_mlp(slots))

        return slots, attn, features


class Object3DSlotLifter(nn.Module):
    """SlotLifter-style point-slot mapper for CALVIN RGB-D point evidence."""

    def __init__(
        self,
        input_dim: int,
        slot_dim: int = 128,
        num_slots: int = 8,
        num_iterations: int = 3,
        num_heads: int = 1,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        self.slot_encoder = Object3DSlotAttention(
            input_dim=input_dim,
            slot_dim=slot_dim,
            num_slots=num_slots,
            num_iterations=num_iterations,
            num_heads=num_heads,
            epsilon=epsilon,
        )
        self.empty_slot = nn.Parameter(torch.empty(1, 1, slot_dim))
        nn.init.xavier_uniform_(self.empty_slot)
        self.point_query = nn.Linear(slot_dim, slot_dim)
        self.slot_key = nn.Linear(slot_dim, slot_dim)
        self.slot_to_feature = nn.Linear(slot_dim, input_dim)
        self.epsilon = float(epsilon)

    @property
    def num_slots(self) -> int:
        return self.slot_encoder.num_slots

    def forward(
        self,
        point_features: torch.Tensor,
        xyz: torch.Tensor,
        valid_mask: torch.Tensor | None = None,
    ) -> Object3DSlotOutput:
        if xyz.ndim != 3 or xyz.shape[-1] != 3:
            raise ValueError(f"xyz must be [B,N,3], got {tuple(xyz.shape)}")
        if point_features.shape[:2] != xyz.shape[:2]:
            raise ValueError(
                "point_features and xyz must share [B,N], "
                f"got {tuple(point_features.shape)} and {tuple(xyz.shape)}"
            )
        slots, attn, encoded_points = self.slot_encoder(point_features)
        batch, num_points, _ = encoded_points.shape
        if valid_mask is None:
            valid_mask = torch.ones((batch, num_points), dtype=torch.bool, device=encoded_points.device)
        else:
            valid_mask = valid_mask.to(device=encoded_points.device, dtype=torch.bool)
            if valid_mask.shape != (batch, num_points):
                raise ValueError(f"valid_mask must be {(batch, num_points)}, got {tuple(valid_mask.shape)}")

        all_slots = torch.cat([self.empty_slot.expand(batch, -1, -1), slots], dim=1)
        q = self.point_query(encoded_points)
        k = self.slot_key(all_slots)
        logits = torch.matmul(q, k.transpose(-1, -2)) * (encoded_points.shape[-1] ** -0.5)
        logits = logits.masked_fill(~valid_mask[..., None], -1e4)
        weights = F.softmax(logits, dim=-1)
        object_weights = weights[..., 1:]
        background_weight = weights[..., 0]

        valid_float = valid_mask.to(dtype=xyz.dtype)
        object_mass = (object_weights * valid_float[..., None]).sum(dim=1)
        denom = object_mass.clamp_min(self.epsilon)
        centers = torch.einsum("bnk,bnd->bkd", object_weights * valid_float[..., None], xyz) / denom[..., None]
        centered = xyz[:, None, :, :] - centers[:, :, None, :]
        covariance_diag = (
            (object_weights.transpose(1, 2)[..., None] * valid_float[:, None, :, None] * centered.square()).sum(dim=2)
            / denom[..., None]
        )
        point_count = valid_float.sum(dim=1).clamp_min(self.epsilon)
        support_fraction = object_mass / point_count[:, None]
        entropy = -(object_weights.clamp_min(self.epsilon) * object_weights.clamp_min(self.epsilon).log()).sum(dim=-1)
        max_entropy = torch.log(torch.tensor(float(max(self.num_slots, 1)), device=xyz.device, dtype=xyz.dtype))
        concentration = 1.0 - torch.clamp(entropy / max_entropy.clamp_min(self.epsilon), min=0.0, max=1.0)
        objectness = support_fraction * (
            torch.einsum("bnk,bn->bk", object_weights, concentration * valid_float) / denom
        )
        object_point_priors = object_weights.transpose(1, 2)
        reconstruction = torch.matmul(weights, self.slot_to_feature(all_slots))
        return Object3DSlotOutput(
            slots=slots,
            point_slot_weights=weights,
            object_point_priors=object_point_priors,
            centers=centers,
            covariance_diag=covariance_diag,
            objectness=objectness,
            background_weight=background_weight,
            encoder_attention=attn,
            reconstruction=reconstruction,
        )


def make_object3d_point_features(
    xyz_norm: torch.Tensor,
    rgb: torch.Tensor,
    view_ids: torch.Tensor,
    *,
    num_views: int = 2,
) -> torch.Tensor:
    if xyz_norm.ndim != 3 or xyz_norm.shape[-1] != 3:
        raise ValueError(f"xyz_norm must be [B,N,3], got {tuple(xyz_norm.shape)}")
    if rgb.shape != xyz_norm.shape:
        raise ValueError(f"rgb must have same shape as xyz_norm, got {tuple(rgb.shape)}")
    if view_ids.shape != xyz_norm.shape[:2]:
        raise ValueError(f"view_ids must be {tuple(xyz_norm.shape[:2])}, got {tuple(view_ids.shape)}")
    one_hot = F.one_hot(view_ids.clamp(min=0, max=num_views - 1).long(), num_classes=num_views).to(dtype=xyz_norm.dtype)
    radius = torch.linalg.norm(xyz_norm, dim=-1, keepdim=True)
    return torch.cat([xyz_norm, rgb, one_hot, radius], dim=-1)
